"""FlashU acceleration configuration.

Default values match Paper Table 7 for the Show-o2 1.5B model.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class FlashUConfig:
    """Hyperparameters for the FlashU acceleration framework.

    Attributes:
        r_p: FFN pruning ratio (Sec. 3.2). Fraction of intermediate neurons
            to prune per layer. 0.0 disables FFN pruning.
        r_LS: Layer-skipping ratio (Sec. 3.3). Fraction of transformer layers
            to skip based on cosine-similarity importance.
        T_LS: Layer-skipping recalculation interval (Sec. 3.3). Number of
            denoising steps between re-evaluating which layers to skip.
            -1 disables dynamic layer skipping entirely.
        tau: Hybrid FFN Network switching threshold (Sec. 3.5, Eq. 4). Number
            of final denoising steps that use the full (unpruned) network.
            0 disables hybrid switching.
        T_cache: Diffusion head caching interval (Sec. 3.4). Reuse cached
            diffusion head output every T_cache steps. -1 disables caching.
        adaptive_guidance_schedule: Adaptive guidance scale s(t) schedule
            (Sec. 3.6). List of (num_steps, guidance_scale) tuples defining
            piecewise-constant guidance. The total number of denoising steps
            is the sum of the first elements. None disables adaptive guidance.
        head_prune_ratio: Attention head pruning ratio. Fraction of GQA
            key-value groups to prune per layer. 0.0 disables head pruning.
        num_inference_steps: Total number of denoising steps. If None,
            derived from the adaptive guidance schedule.
        calibration_samples: Number of calibration forward passes for Wanda
            importance scoring.
    """

    # Structural pruning (Sec. 3.2)
    r_p: float = 0.20
    head_prune_ratio: float = 0.0

    # Dynamic layer skipping (Sec. 3.3)
    r_LS: float = 0.20
    T_LS: int = 10

    # Diffusion head caching (Sec. 3.4)
    T_cache: int = 5

    # Hybrid FFN Network (Sec. 3.5, Eq. 4)
    # Paper: tau = 0.3, meaning the first tau*T steps use full FFN.
    # In code: tau = number of FINAL steps that use full FFN.
    # With T=35, tau = 0.3 * 35 = 10.5, rounded to 11.
    tau: int = 11

    # Adaptive guidance scale s(t) (Sec. 3.6, Eq. 6)
    # Paper: s(t) = s_low * 1(t > t_switch) + s_high * 1(t <= t_switch)
    # t_switch = 0.3T = 0.3*35 = 10.5, rounded to 11 steps from end.
    # In forward-step indexing (step 1..35):
    #   steps 1-24 (early, high noise): s_low = 0.75 * s0
    #   steps 25-35 (late, low noise):  s_high = s0
    # s0 is the baseline guidance_scale (=10 for DPG-Bench).
    # s_low = 0.75 * 10 = 7.5, s_high = 10.
    adaptive_guidance_schedule: Optional[List[Tuple[int, float]]] = field(
        default_factory=lambda: [
            (24, 7.5),
            (11, 10.0),
        ]
    )

    # Inference steps: Paper Table 7: T = 0.7 * T0 = 0.7 * 50 = 35
    num_inference_steps: Optional[int] = 35

    # Wanda calibration
    calibration_samples: int = 1

    @property
    def total_steps(self) -> int:
        """Total denoising steps, derived from schedule or explicit setting."""
        if self.num_inference_steps is not None:
            return self.num_inference_steps
        if self.adaptive_guidance_schedule:
            return sum(s for s, _ in self.adaptive_guidance_schedule)
        return 50

    @property
    def adaptive_guidance_config(self) -> Optional[dict]:
        """Build the guidance config dict expected by the patched model."""
        if self.adaptive_guidance_schedule is None:
            return None
        return {
            "schedule": self.adaptive_guidance_schedule,
            "original_gs": None,  # Filled at patch time from the model call
        }
