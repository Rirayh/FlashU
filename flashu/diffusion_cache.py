"""Diffusion head caching (Paper Sec. 3.4).

Provides helper functions to determine whether to use a cached diffusion head
output on a given step, and to store/retrieve the cache.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def should_use_diffusion_cache(
    current_step: int,
    T_cache: int,
    use_full_network: bool,
    cache: Optional[torch.Tensor],
) -> bool:
    """Determine whether to reuse the cached diffusion head output.

    Cache is used when:
      - Caching is enabled (T_cache > 0)
      - Not in full-network mode (Hybrid FFN switching)
      - Current step is not a recompute step (step % T_cache != 0)
      - A cached value exists

    Args:
        current_step: Current denoising step (1-indexed).
        T_cache: Cache reuse interval. -1 or 0 disables caching.
        use_full_network: True if full (unpruned) network is active.
        cache: Previously cached tensor, or None.

    Returns:
        True if the cache should be reused.
    """
    return (
        T_cache > 0
        and not use_full_network
        and current_step % T_cache != 0
        and cache is not None
    )


def should_store_diffusion_cache(
    current_step: int,
    T_cache: int,
) -> bool:
    """Determine whether to store the diffusion head output in cache.

    Args:
        current_step: Current denoising step (1-indexed).
        T_cache: Cache reuse interval. -1 or 0 disables caching.

    Returns:
        True if the output should be cached on this step.
    """
    return T_cache > 0 and current_step % T_cache == 0
