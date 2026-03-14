"""Structural pruning for FFN neurons and attention heads (Paper Sec. 3.2).

Standalone functions that operate on a baseline Showo2Qwen2_5 model,
creating pruned weight copies for the Hybrid FFN Network (Eq. 4) and
physically replacing attention projections for head pruning.
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FRIENDLY_MULTIPLE = 64  # GPU tensor-core alignment


def apply_structural_pruning(
    model,
    head_prune_ratio: float = 0.0,
    ffn_prune_ratio: float = 0.0,
    scores_by_layer: Dict[str, torch.Tensor] = None,
):
    """Apply one-shot structural pruning based on externally computed importance scores.

    For FFN pruning (ratio r_p): creates pruned copies of gate/up/down projections
    stored alongside originals for Hybrid FFN Network switching (Eq. 4).
    For attention head pruning: physically replaces Q/K/V/O projections with smaller
    ones, respecting GQA (Grouped Query Attention) structure.

    Args:
        model: Showo2Qwen2_5 model instance.
        head_prune_ratio: Fraction of attention head groups to prune per layer.
        ffn_prune_ratio: Fraction r_p of FFN intermediate neurons to prune per layer.
        scores_by_layer: Dict mapping layer weight names to per-element importance
            scores (e.g., from WandaPruner).
    """
    if (head_prune_ratio == 0.0 and ffn_prune_ratio == 0.0) or scores_by_layer is None:
        logger.info("[Structural Pruning] Ratios or scores are zero/None. Skipping.")
        return

    logger.info(
        f"[Structural Pruning] Applying: head_ratio={head_prune_ratio}, "
        f"ffn_ratio={ffn_prune_ratio}"
    )

    llm_model = model.showo
    total_heads_pruned = 0
    total_ffn_neurons_pruned = 0

    for layer_idx, layer in enumerate(llm_model.model.layers):

        # --- FFN Neuron Pruning (r_p) ---
        if ffn_prune_ratio > 0.0:
            n_pruned = _prune_ffn_layer(layer, layer_idx, ffn_prune_ratio, scores_by_layer)
            if n_pruned > 0:
                if not hasattr(llm_model.model, 'is_hybrid_pruned'):
                    llm_model.model.is_hybrid_pruned = False
                llm_model.model.is_hybrid_pruned = True
            total_ffn_neurons_pruned += n_pruned

        # --- Attention Head Pruning (GQA-Aware) ---
        if head_prune_ratio > 0.0:
            n_pruned = _prune_heads_layer(layer, layer_idx, head_prune_ratio, scores_by_layer)
            total_heads_pruned += n_pruned

    logger.info(
        f"[Structural Pruning] Complete. "
        f"Q heads pruned: {total_heads_pruned}, FFN neurons pruned: {total_ffn_neurons_pruned}"
    )


def _prune_ffn_layer(
    layer, layer_idx: int, ffn_prune_ratio: float,
    scores_by_layer: Dict[str, torch.Tensor],
) -> int:
    """Prune FFN intermediate dimension for a single layer. Returns neurons pruned."""
    mlp = layer.mlp
    original_intermediate_size = mlp.intermediate_size

    ffn_score_name = f"model.layers.{layer_idx}.mlp.down_proj"

    if original_intermediate_size == 0:
        return 0

    if ffn_score_name not in scores_by_layer:
        logger.warning(
            f"[Pruning] Layer {layer_idx} FFN: No scores for {ffn_score_name}. Skipping."
        )
        return 0

    neuron_importance = scores_by_layer[ffn_score_name]

    if len(neuron_importance) != original_intermediate_size:
        logger.warning(
            f"[Pruning] Layer {layer_idx} FFN: Importance size mismatch "
            f"(scores={len(neuron_importance)}, layer={original_intermediate_size}). Skipping."
        )
        return 0

    num_neurons_to_prune_ideal = int(original_intermediate_size * ffn_prune_ratio)
    num_neurons_to_keep_ideal = original_intermediate_size - num_neurons_to_prune_ideal

    num_neurons_to_keep = (num_neurons_to_keep_ideal // FRIENDLY_MULTIPLE) * FRIENDLY_MULTIPLE

    if num_neurons_to_keep <= 0 and original_intermediate_size > 0:
        num_neurons_to_keep = FRIENDLY_MULTIPLE
        if num_neurons_to_keep > original_intermediate_size:
            num_neurons_to_keep = max(num_neurons_to_keep_ideal, 1)

    num_neurons_to_prune = original_intermediate_size - num_neurons_to_keep

    if num_neurons_to_prune > 0 and num_neurons_to_keep > 0:
        keep_indices = torch.topk(neuron_importance, num_neurons_to_keep, largest=True).indices
        keep_indices = torch.sort(keep_indices).values

        new_intermediate_size = num_neurons_to_keep
        hidden_size = mlp.gate_proj.in_features
        device = mlp.gate_proj.weight.device
        dtype = mlp.gate_proj.weight.dtype

        new_gate_proj = nn.Linear(hidden_size, new_intermediate_size, bias=False, device=device, dtype=dtype)
        new_up_proj = nn.Linear(hidden_size, new_intermediate_size, bias=False, device=device, dtype=dtype)
        new_down_proj = nn.Linear(new_intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

        new_gate_proj.weight.data.copy_(mlp.gate_proj.weight.data[keep_indices, :])
        new_up_proj.weight.data.copy_(mlp.up_proj.weight.data[keep_indices, :])
        new_down_proj.weight.data.copy_(mlp.down_proj.weight.data[:, keep_indices])

        # Eq. (4): Store pruned weights alongside originals for Hybrid FFN Network
        layer.mlp.gate_proj_pruned = new_gate_proj
        layer.mlp.up_proj_pruned = new_up_proj
        layer.mlp.down_proj_pruned = new_down_proj
        layer.mlp.pruned_intermediate_size = new_intermediate_size

        return num_neurons_to_prune

    return 0


def _prune_heads_layer(
    layer, layer_idx: int, head_prune_ratio: float,
    scores_by_layer: Dict[str, torch.Tensor],
) -> int:
    """Prune attention heads (GQA-aware) for a single layer. Returns heads pruned."""
    attn = layer.self_attn
    original_num_heads = attn.num_heads
    original_num_key_value_heads = attn.num_key_value_heads

    head_score_name = f"model.layers.{layer_idx}.self_attn.o_proj"

    if original_num_heads == 0:
        return 0

    if head_score_name not in scores_by_layer:
        logger.warning(
            f"[Pruning] Layer {layer_idx} Head: No scores for {head_score_name}. Skipping."
        )
        return 0

    head_importance_flat = scores_by_layer[head_score_name]

    if original_num_key_value_heads <= 0 or original_num_heads % original_num_key_value_heads != 0:
        logger.warning(
            f"[Pruning] Layer {layer_idx}: Invalid GQA config "
            f"(Hq={original_num_heads}, Hkv={original_num_key_value_heads}). Skipping."
        )
        return 0

    head_dim = attn.head_dim
    group_size = original_num_heads // original_num_key_value_heads
    device = head_importance_flat.device

    if len(head_importance_flat) != (original_num_heads * head_dim):
        logger.warning(
            f"[Pruning] Layer {layer_idx} Head: Importance size mismatch "
            f"(scores={len(head_importance_flat)}, expected={original_num_heads * head_dim}). Skipping."
        )
        return 0

    try:
        group_importance = head_importance_flat.view(
            original_num_key_value_heads, group_size * head_dim
        ).sum(dim=1)
    except RuntimeError as e:
        logger.warning(
            f"[Pruning] Layer {layer_idx}: Failed to reshape head importance: {e}. Skipping."
        )
        return 0

    num_kv_heads_to_prune = int(original_num_key_value_heads * head_prune_ratio)
    num_kv_heads_to_keep = original_num_key_value_heads - num_kv_heads_to_prune

    if num_kv_heads_to_keep == 0 and original_num_key_value_heads > 0:
        num_kv_heads_to_keep = 1

    if num_kv_heads_to_keep == original_num_key_value_heads:
        return 0

    keep_kv_indices = torch.topk(group_importance, num_kv_heads_to_keep, largest=True).indices
    keep_kv_indices = torch.sort(keep_kv_indices).values

    group_offsets = torch.arange(group_size, device=device).unsqueeze(0)
    keep_q_indices = (keep_kv_indices.unsqueeze(1) * group_size + group_offsets).view(-1)
    keep_q_indices = torch.sort(keep_q_indices).values

    new_num_key_value_heads = num_kv_heads_to_keep
    new_num_heads = new_num_key_value_heads * group_size
    num_heads_pruned = original_num_heads - new_num_heads

    hidden_size = attn.hidden_size
    dtype = attn.q_proj.weight.dtype

    new_q_size = new_num_heads * head_dim
    new_kv_size = new_num_key_value_heads * head_dim

    new_q_proj = nn.Linear(hidden_size, new_q_size, bias=False, device=device, dtype=dtype)
    new_k_proj = nn.Linear(hidden_size, new_kv_size, bias=False, device=device, dtype=dtype)
    new_v_proj = nn.Linear(hidden_size, new_kv_size, bias=False, device=device, dtype=dtype)
    new_o_proj = nn.Linear(new_q_size, hidden_size, bias=False, device=device, dtype=dtype)

    q_weights = attn.q_proj.weight.data.view(original_num_heads, head_dim, hidden_size)
    new_q_proj.weight.data.copy_(q_weights[keep_q_indices, :, :].reshape(new_q_size, hidden_size))

    k_weights = attn.k_proj.weight.data.view(original_num_key_value_heads, head_dim, hidden_size)
    new_k_proj.weight.data.copy_(k_weights[keep_kv_indices, :, :].reshape(new_kv_size, hidden_size))

    v_weights = attn.v_proj.weight.data.view(original_num_key_value_heads, head_dim, hidden_size)
    new_v_proj.weight.data.copy_(v_weights[keep_kv_indices, :, :].reshape(new_kv_size, hidden_size))

    o_weights = attn.o_proj.weight.data.view(hidden_size, original_num_heads, head_dim)
    new_o_proj.weight.data.copy_(o_weights[:, keep_q_indices, :].reshape(hidden_size, new_q_size))

    layer.self_attn.q_proj = new_q_proj
    layer.self_attn.k_proj = new_k_proj
    layer.self_attn.v_proj = new_v_proj
    layer.self_attn.o_proj = new_o_proj

    layer.self_attn.num_heads = new_num_heads
    layer.self_attn.num_key_value_heads = new_num_key_value_heads

    return num_heads_pruned
