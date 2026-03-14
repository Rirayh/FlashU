"""Adaptive guidance scale s(t) (Paper Sec. 3.6).

Computes a time-varying classifier-free guidance scale based on a
piecewise-constant schedule defined in FlashUConfig.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_adaptive_guidance_scale(
    current_step: int,
    adaptive_guidance_config: Optional[dict],
    default_guidance_scale: float,
) -> float:
    """Compute the effective guidance scale s(t) for the current denoising step.

    The schedule is a list of (num_steps_in_phase, guidance_scale) tuples.
    Steps are 1-indexed. If the current step falls beyond all phases, the
    original guidance scale is used as fallback.

    Args:
        current_step: Current denoising step (1-indexed).
        adaptive_guidance_config: Dict with keys "schedule" (list of
            (steps, gs) tuples) and "original_gs" (fallback scale).
        default_guidance_scale: Guidance scale to use if no config is provided.

    Returns:
        Effective guidance scale for this step.
    """
    if adaptive_guidance_config is None:
        return default_guidance_scale

    schedule = adaptive_guidance_config.get("schedule", [])
    if not schedule:
        return default_guidance_scale

    step_count = 0
    for steps_in_phase, gs_for_phase in schedule:
        step_count += steps_in_phase
        if current_step <= step_count:
            return gs_for_phase

    return adaptive_guidance_config.get("original_gs", default_guidance_scale)
