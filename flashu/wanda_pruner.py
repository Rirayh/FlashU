"""Wanda-based structural importance scorer (Paper Sec. 3.2, citing [48]).

Computes per-layer importance scores for structured pruning of attention heads
and FFN neurons using weight-activation product norms.
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class WandaPruner:
    """Compute Wanda structural importance scores for attention heads and FFN neurons.

    Workflow:
        1. Register forward hooks on target linear layers.
        2. Run calibration forward passes to accumulate activation norms.
        3. Return per-layer structural importance scores for external pruning.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.act_scales: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self.target_layers: Dict[str, nn.Linear] = {}
        self.device = next(model.parameters()).device

        target_layer_names = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(
                name.endswith(ln) for ln in target_layer_names
            ):
                self.target_layers[name] = module
                hook_fn = lambda m, i, o, layer_name=name: self._capture_activation_norm(
                    layer_name, m, i, o
                )
                self.hooks.append(module.register_forward_hook(hook_fn))

        if not self.target_layers:
            logger.warning(
                "WandaPruner: no target linear layers found in the model."
            )

    def _capture_activation_norm(self, layer_name, module, inp, outp):
        """Accumulate squared L2 norms of input activations."""
        X = inp[0].detach()
        if X.dim() == 3:
            sum_sq = X.pow(2).sum(dim=[0, 1])
        else:
            sum_sq = X.pow(2).sum(dim=0)

        if layer_name not in self.act_scales:
            self.act_scales[layer_name] = sum_sq.to(self.device)
        else:
            self.act_scales[layer_name].add_(sum_sq.to(self.device))

    def add_batch(self, batch: dict):
        """Run one calibration forward pass to trigger hooks."""
        try:
            self.model(**batch)
        except Exception as e:
            logger.error(
                f"WandaPruner: forward pass error in add_batch: {e}",
                exc_info=True,
            )

    def _finalize_activations(self):
        """Convert accumulated squared sums to L2 norms."""
        logger.info(
            f"WandaPruner: finalizing activation scales for "
            f"{len(self.act_scales)} layers."
        )
        for name, sum_sq in self.act_scales.items():
            self.act_scales[name] = torch.sqrt(sum_sq).to(self.device)

    def get_structural_importance_scores(self) -> Dict[str, torch.Tensor]:
        """Return per-layer Wanda importance scores for structured pruning.

        For output-neuron layers (gate_proj, up_proj, q/k/v_proj):
            score_i = sum_j(|W_ij| * ||X_j||_2)   (row importance)
        For input-neuron layers (down_proj, o_proj):
            score_j = ||X_j||_2 * sum_i(|W_ij|)    (column importance)
        """
        self._finalize_activations()
        scores_by_layer: Dict[str, torch.Tensor] = {}

        for name, module in self.target_layers.items():
            if not isinstance(module, nn.Linear):
                continue
            if name not in self.act_scales:
                logger.warning(
                    f"WandaPruner: no activation scales for {name}, skipping."
                )
                continue

            W = module.weight.data.float()
            act = self.act_scales[name].float().to(W.device)

            if W.shape[1] != act.shape[0]:
                logger.error(
                    f"WandaPruner: dimension mismatch for {name} "
                    f"(W cols={W.shape[1]}, act={act.shape[0]}). Skipping."
                )
                continue

            importance = None
            if name.endswith(("gate_proj", "up_proj", "q_proj", "k_proj", "v_proj")):
                metric = W.abs() * act.unsqueeze(0)
                importance = metric.sum(dim=1)
            elif name.endswith(("down_proj", "o_proj")):
                w_col_abs_sum = W.abs().sum(dim=0)
                importance = w_col_abs_sum * act

            if importance is not None:
                scores_by_layer[name] = importance

        logger.info("WandaPruner: structural importance scores computed.")
        return scores_by_layer

    def remove_hooks(self):
        """Remove all registered forward hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        logger.info("WandaPruner: all hooks removed.")
