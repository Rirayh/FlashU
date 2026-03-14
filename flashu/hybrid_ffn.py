"""Hybrid FFN Network switching (Paper Sec. 3.5, Eq. 4).

Patches Qwen2MLP.forward to support switching between pruned and full
(original) FFN weights based on the use_full_network flag.

Also patches Qwen2DecoderLayer.forward to accept and pass through the
use_full_network argument.
"""

import logging
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Keep references to originals for unpatch
_original_mlp_forward = None
_original_decoder_layer_forwards = {}


def patch_hybrid_ffn(model):
    """Monkey-patch Qwen2MLP.forward and Qwen2DecoderLayer.forward for hybrid FFN.

    After FFN pruning, each MLP may have pruned weight copies
    (gate_proj_pruned, up_proj_pruned, down_proj_pruned). This patch makes
    the MLP.forward method select between pruned and full weights based on
    the use_full_network flag.

    Args:
        model: Showo2Qwen2_5 model instance.
    """
    global _original_mlp_forward

    llm = model.showo.model

    # Set the hybrid-pruned flag on the model
    if not hasattr(llm, 'is_hybrid_pruned'):
        llm.is_hybrid_pruned = False

    # Patch each MLP instance
    for layer in llm.layers:
        mlp = layer.mlp
        if not hasattr(mlp, '_original_forward'):
            mlp._original_forward = mlp.forward

        def hybrid_mlp_forward(self, hidden_state, use_full_network: bool = False):
            if use_full_network and hasattr(self, 'gate_proj_pruned'):
                gate_proj_to_use = self.gate_proj
                up_proj_to_use = self.up_proj
                down_proj_to_use = self.down_proj
            elif hasattr(self, 'gate_proj_pruned'):
                gate_proj_to_use = self.gate_proj_pruned
                up_proj_to_use = self.up_proj_pruned
                down_proj_to_use = self.down_proj_pruned
            else:
                gate_proj_to_use = self.gate_proj
                up_proj_to_use = self.up_proj
                down_proj_to_use = self.down_proj

            return down_proj_to_use(
                self.act_fn(gate_proj_to_use(hidden_state)) * up_proj_to_use(hidden_state)
            )

        mlp.forward = types.MethodType(hybrid_mlp_forward, mlp)

    # Patch each DecoderLayer to pass use_full_network to MLP
    for layer_idx, layer in enumerate(llm.layers):
        if not hasattr(layer, '_original_forward'):
            layer._original_forward = layer.forward

        _original_decoder_layer_forwards[layer_idx] = layer._original_forward

        def patched_decoder_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_full_network: bool = False,
            **kwargs,
        ) -> Tuple[torch.FloatTensor, ...]:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, use_full_network=use_full_network)
            hidden_states = residual + hidden_states

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (self_attn_weights,)
            if use_cache:
                outputs += (present_key_value,)
            return outputs

        layer.forward = types.MethodType(patched_decoder_forward, layer)

    logger.info(
        "[FlashU] Hybrid FFN patch applied to Qwen2MLP.forward "
        "and Qwen2DecoderLayer.forward"
    )


def unpatch_hybrid_ffn(model):
    """Reverse the hybrid FFN monkey patches.

    Args:
        model: Showo2Qwen2_5 model instance.
    """
    llm = model.showo.model

    for layer in llm.layers:
        mlp = layer.mlp
        if hasattr(mlp, '_original_forward'):
            mlp.forward = mlp._original_forward
            del mlp._original_forward

    for layer_idx, layer in enumerate(llm.layers):
        if hasattr(layer, '_original_forward'):
            layer.forward = layer._original_forward
            del layer._original_forward

    _original_decoder_layer_forwards.clear()

    logger.info(
        "[FlashU] Hybrid FFN patch removed from Qwen2MLP and Qwen2DecoderLayer"
    )


def should_use_full_network(
    current_step: int,
    total_steps: int,
    tau: int,
) -> bool:
    """Determine if the full (unpruned) network should be used at this step.

    Implements Eq. (4): use full network for the last tau steps.

    Args:
        current_step: Current denoising step (1-indexed).
        total_steps: Total number of denoising steps.
        tau: Number of final steps using the full network.

    Returns:
        True if full network should be used.
    """
    if tau <= 0:
        return False
    return current_step > (total_steps - tau)
