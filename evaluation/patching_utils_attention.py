import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
import types
import logging
import math

from transformers.cache_utils import Cache

from models.qwen2 import Qwen2Attention, Qwen2SdpaAttention, repeat_kv, apply_rotary_pos_emb
from torch.nn.attention.flex_attention import flex_attention, BlockMask

logger = logging.getLogger(__name__)


def _patched_qwen2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Patched Qwen2Attention.forward with pruning-safe reshape logic."""
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    # Use actual (possibly pruned) head count instead of self.hidden_size
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _patched_qwen2_sdpa_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Patched Qwen2SdpaAttention.forward with pruning-safe view logic."""
    if output_attentions:
        return _patched_qwen2_attention_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask

    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    is_causal = True if causal_mask is None and q_len > 1 else False

    query_states = query_states.to(value_states.dtype)
    key_states = key_states.to(value_states.dtype)

    if type(attention_mask) == BlockMask:
        attn_output = flex_attention(query_states, key_states, value_states, block_mask=attention_mask)
    else:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    # Use actual (possibly pruned) head count instead of self.hidden_size
    attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def apply_attention_reshape_patch(model):
    """Patch all Qwen2 attention layers to handle pruned head dimensions correctly."""
    try:
        if not hasattr(model, 'showo') or not hasattr(model.showo, 'model') or not hasattr(model.showo.model, 'layers'):
             logger.warning("[AttnPatch] Cannot find model.showo.model.layers. Skipping attention patch.")
             return

        patched_count = 0
        for layer in model.showo.model.layers:
            attn_module = layer.self_attn

            if hasattr(attn_module, '_is_patched_for_reshape') and attn_module._is_patched_for_reshape:
                continue

            if isinstance(attn_module, Qwen2SdpaAttention):
                attn_module.forward = types.MethodType(_patched_qwen2_sdpa_attention_forward, attn_module)
                attn_module._is_patched_for_reshape = True
                patched_count += 1
            elif isinstance(attn_module, Qwen2Attention):
                attn_module.forward = types.MethodType(_patched_qwen2_attention_forward, attn_module)
                attn_module._is_patched_for_reshape = True
                patched_count += 1

        if patched_count > 0:
            logger.info(f"[INFO] Successfully applied attention reshape patch to {patched_count} layers.")
        else:
             logger.warning("[AttnPatch] No Qwen2Attention or Qwen2SdpaAttention layers found to patch.")

    except Exception as e:
        logger.error(f"[ERROR] Failed to apply attention reshape patch: {e}", exc_info=True)
