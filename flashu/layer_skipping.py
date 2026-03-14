"""Dynamic layer skipping (Paper Sec. 3.3).

Provides:
  - calculate_layers_to_skip(): Cosine-similarity based layer importance scoring
  - patch_layer_skipping(): Monkey-patches Qwen2ForCausalLM.forward to support
    skipping specified layers and passing use_full_network to Qwen2DecoderLayer
"""

import logging
import types
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


def calculate_layers_to_skip(
    model,
    all_hidden_states: Tuple[torch.Tensor],
    ratio: float,
) -> List[int]:
    """Select layers to skip based on cosine similarity between input/output states.

    Layers with high input-output similarity (low importance) are skipped first.

    Args:
        model: Showo2Qwen2_5 model instance (used to get layer count).
        all_hidden_states: Tuple of hidden states from all layers (including
            input embedding layer at index 0).
        ratio: r_LS, fraction of layers to skip.

    Returns:
        Sorted list of layer indices to skip.
    """
    if (all_hidden_states is None
            or not isinstance(all_hidden_states, tuple)
            or len(all_hidden_states) <= 1):
        return []

    num_layers = len(model.showo.model.layers)

    if len(all_hidden_states) < num_layers + 1:
        return []

    importances = []
    for i in range(num_layers):
        input_state = F.normalize(all_hidden_states[i], p=2, dim=-1)
        output_state = F.normalize(all_hidden_states[i + 1], p=2, dim=-1)
        similarity = (input_state * output_state).sum(dim=-1).mean().item()
        importances.append(1 - similarity)

    num_to_prune = int(num_layers * ratio)
    if num_to_prune == 0:
        return []

    sorted_indices = sorted(range(num_layers), key=lambda k: importances[k])
    layers_to_skip = sorted(sorted_indices[:num_to_prune])

    return layers_to_skip


def patch_layer_skipping(model):
    """Monkey-patch Qwen2ForCausalLM.forward to support layer skipping and hybrid FFN.

    This replaces the entire forward pass of model.showo (Qwen2ForCausalLM) so that:
      - layers_to_skip: list of layer indices to skip entirely
      - use_full_network: passed through to Qwen2DecoderLayer.forward for Hybrid FFN

    The original forward is saved as model.showo._original_forward for reversibility.

    Args:
        model: Showo2Qwen2_5 model instance.
    """
    qwen2_causal_lm = model.showo

    if hasattr(qwen2_causal_lm, "_original_forward"):
        return  # Already patched

    qwen2_causal_lm._original_forward = qwen2_causal_lm.forward

    def patched_qwen2_for_causal_lm_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        layers_to_skip: Optional[List[int]] = None,
        use_full_network: bool = False,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = (
            output_attentions if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict
        )

        core_model = self.model

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = core_model.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length()
                if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = core_model.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_cache = past_key_values if use_cache else None

        for i, decoder_layer in enumerate(core_model.layers):
            if layers_to_skip and i in layers_to_skip:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                continue

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=next_cache,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_full_network=use_full_network,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = core_model.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits, labels, self.vocab_size, **loss_kwargs
            )

        if not return_dict:
            output = (logits, next_cache, all_hidden_states, all_self_attns)
            final_output = tuple(v for v in output if v is not None)
            return (loss,) + final_output if loss is not None else final_output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    qwen2_causal_lm.forward = types.MethodType(
        patched_qwen2_for_causal_lm_forward, qwen2_causal_lm
    )
    logger.info(
        "[FlashU] Layer skipping patch applied to Qwen2ForCausalLM.forward"
    )


def unpatch_layer_skipping(model):
    """Reverse the layer-skipping monkey patch.

    Args:
        model: Showo2Qwen2_5 model instance.
    """
    qwen2_causal_lm = model.showo
    if hasattr(qwen2_causal_lm, "_original_forward"):
        qwen2_causal_lm.forward = qwen2_causal_lm._original_forward
        del qwen2_causal_lm._original_forward
        logger.info(
            "[FlashU] Layer skipping patch removed from Qwen2ForCausalLM.forward"
        )
