"""FlashU patch entry point.

Applies all FlashU acceleration hooks to a baseline Showo2Qwen2_5 model
via monkey-patching, without modifying the original model files.

Usage:
    from flashu import apply_flashu_patch, FlashUConfig

    config = FlashUConfig(r_p=0.20, r_LS=0.20, T_LS=10, tau=10, T_cache=5)
    apply_flashu_patch(model, config)
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from .config import FlashUConfig
from .layer_skipping import (
    calculate_layers_to_skip,
    patch_layer_skipping,
    unpatch_layer_skipping,
)
from .hybrid_ffn import (
    patch_hybrid_ffn,
    unpatch_hybrid_ffn,
    should_use_full_network,
)
from .adaptive_guidance import compute_adaptive_guidance_scale
from .diffusion_cache import should_use_diffusion_cache, should_store_diffusion_cache

logger = logging.getLogger(__name__)

# Expected hidden_size for Showo2Qwen2_5 1.5B
_EXPECTED_HIDDEN_SIZE = 1536


def apply_flashu_patch(model, config: Optional[FlashUConfig] = None):
    """Apply the FlashU acceleration framework to a baseline Showo2Qwen2_5 model.

    This is the main entry point. It:
      1. Validates the model is Showo2Qwen2_5 1.5B (hidden_size == 1536)
      2. Applies the Hybrid FFN monkey patch (Qwen2MLP.forward, Qwen2DecoderLayer.forward)
      3. Applies the Layer Skipping monkey patch (Qwen2ForCausalLM.forward)
      4. Injects acceleration state attributes onto the model
      5. Replaces model.forward and model.t2i_generate with accelerated versions

    All patches are reversible via remove_flashu_patch().

    Args:
        model: An instance of Showo2Qwen2_5 (baseline).
        config: FlashUConfig instance. If None, uses Paper Table 7 defaults.

    Returns:
        The patched model (same object, modified in-place).

    Raises:
        ValueError: If the model's hidden_size does not match the expected 1.5B config.
    """
    if config is None:
        config = FlashUConfig()

    # --- 1. Validate model ---
    hidden_size = getattr(model.config, 'hidden_size', None)
    if hidden_size != _EXPECTED_HIDDEN_SIZE:
        raise ValueError(
            f"FlashU is designed for Showo2Qwen2_5 1.5B (hidden_size={_EXPECTED_HIDDEN_SIZE}), "
            f"but got hidden_size={hidden_size}. "
            f"Use at your own risk or adjust _EXPECTED_HIDDEN_SIZE."
        )

    logger.info(
        f"[FlashU] Applying patch: r_p={config.r_p}, r_LS={config.r_LS}, "
        f"T_LS={config.T_LS}, tau={config.tau}, T_cache={config.T_cache}, "
        f"steps={config.total_steps}"
    )

    # --- 2. Apply Hybrid FFN patch (must come before layer skipping) ---
    patch_hybrid_ffn(model)

    # --- 3. Apply Layer Skipping patch ---
    patch_layer_skipping(model)

    # --- 4. Inject acceleration state ---
    model._flashu_config = config
    model._flashu_current_step = 0
    model._flashu_diffusion_cache = None
    model._flashu_layers_to_skip_cache = None
    model._flashu_total_inference_steps = config.total_steps
    model._flashu_adaptive_guidance_config = config.adaptive_guidance_config

    # Save originals for reversibility
    if not hasattr(model, '_original_forward'):
        model._original_forward = model.forward
    if not hasattr(model, '_original_t2i_generate'):
        model._original_t2i_generate = model.t2i_generate

    # --- 5. Replace forward ---
    import types
    model.forward = types.MethodType(_accelerated_forward, model)
    model.t2i_generate = types.MethodType(_accelerated_t2i_generate, model)
    model.reset_acceleration_state = types.MethodType(_reset_acceleration_state, model)
    model.calculate_and_select_layers_to_skip = types.MethodType(
        _calculate_and_select_layers_to_skip, model
    )

    logger.info("[FlashU] All patches applied successfully.")
    return model


def remove_flashu_patch(model):
    """Remove all FlashU patches and restore the baseline model.

    Args:
        model: A previously patched Showo2Qwen2_5 model.
    """
    # Restore original methods
    if hasattr(model, '_original_forward'):
        model.forward = model._original_forward
        del model._original_forward
    if hasattr(model, '_original_t2i_generate'):
        model.t2i_generate = model._original_t2i_generate
        del model._original_t2i_generate

    # Remove injected methods
    for attr in ['reset_acceleration_state', 'calculate_and_select_layers_to_skip']:
        if hasattr(model, attr) and hasattr(getattr(model, attr), '__func__'):
            try:
                delattr(model, attr)
            except AttributeError:
                pass

    # Reverse sub-patches
    unpatch_layer_skipping(model)
    unpatch_hybrid_ffn(model)

    # Clean up state
    for attr in [
        '_flashu_config', '_flashu_current_step', '_flashu_diffusion_cache',
        '_flashu_layers_to_skip_cache', '_flashu_total_inference_steps',
        '_flashu_adaptive_guidance_config',
    ]:
        if hasattr(model, attr):
            delattr(model, attr)

    logger.info("[FlashU] All patches removed. Model restored to baseline.")


# ---------------------------------------------------------------------------
# Injected methods
# ---------------------------------------------------------------------------

def _reset_acceleration_state(
    self,
    diffusion_head_cache_interval: int = -1,
    layer_skip_recalc_interval: int = -1,
    num_inference_steps: int = 50,
    adaptive_guidance_config: Optional[dict] = None,
    hybrid_network_final_steps: int = 0,
):
    """Reset all acceleration caches before each new generation task.

    Args:
        diffusion_head_cache_interval: T_cache. -1 disables caching.
        layer_skip_recalc_interval: T_LS. -1 disables dynamic layer skipping.
        num_inference_steps: Total denoising steps.
        adaptive_guidance_config: Dict for s(t) schedule.
        hybrid_network_final_steps: tau, final steps using full network.
    """
    self._flashu_current_step = 0
    self._flashu_diffusion_cache = None
    self._flashu_layers_to_skip_cache = None
    self._flashu_diffusion_head_cache_interval = diffusion_head_cache_interval
    self._flashu_layer_skip_recalc_interval = layer_skip_recalc_interval
    self._flashu_total_inference_steps = num_inference_steps
    self._flashu_adaptive_guidance_config = adaptive_guidance_config
    self._flashu_hybrid_network_final_steps = hybrid_network_final_steps


def _calculate_and_select_layers_to_skip(
    self,
    all_hidden_states: Tuple[torch.Tensor],
    ratio: float,
) -> List[int]:
    """Delegate to the layer_skipping module."""
    return calculate_layers_to_skip(self, all_hidden_states, ratio)


def _accelerated_forward(
    self,
    text_tokens=None,
    image_latents=None,
    t=None,
    attention_mask=None,
    modality_positions=None,
    layers_to_skip: Optional[list] = None,
    image_labels=None, text_labels=None, image_masks=None,
    max_seq_len=None, device='cuda:0',
    profiler=None,
    use_full_network: bool = False,
    **kwargs,
):
    """Accelerated forward pass with diffusion head caching.

    Mirrors the baseline forward but adds:
      - layers_to_skip / use_full_network passed to the LLM
      - Diffusion head output caching (T_cache)
    """
    from einops import rearrange
    from models.omni_attention import omni_attn_mask

    if image_latents is None:
        return self.showo(
            input_ids=text_tokens,
            attention_mask=attention_mask,
            layers_to_skip=layers_to_skip,
            use_full_network=use_full_network,
        )

    # --- 1. Feature Extraction ---
    if profiler:
        profiler.start('1. Feature Extraction')
    input_embeds_full = self.showo.model.embed_tokens(text_tokens)
    dtype = input_embeds_full.dtype

    if len(image_latents.shape) != 4:
        b, c, T, h, w = image_latents.shape
    else:
        b, c, h, w = image_latents.shape
        T = 0

    if T == 0:
        image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
        image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
    else:
        image_latents_re = rearrange(image_latents, 'b c t h w -> (b t) c h w')
        image_embeds_und = self.image_embedder_und(image_latents_re.to(dtype))
        image_embeds_und = image_embeds_und.reshape(b, T, -1, self.config.clip_latent_dim)
        image_embeds_und = rearrange(image_embeds_und, 'b t l d -> (b t) l d')
        image_embeds_gen = self.image_embedder_gen(image_latents_re.to(dtype))
        image_embeds_gen = image_embeds_gen.reshape(b, T, -1, self.config.hidden_size)
        image_embeds_gen = rearrange(image_embeds_gen, 'b t l d -> b (t l) d')

    from models.misc import interpolate_pos_encoding

    p = self.config.patch_size
    h_, w_ = h // p, w // p

    if self.position_embedding.weight.shape[-1] == self.image_position_ids.shape[-1]:
        image_embeds_und = image_embeds_und + self.position_embedding(self.image_position_ids)
    else:
        image_embeds_und = image_embeds_und + interpolate_pos_encoding(
            self.config.clip_latent_dim, self.position_embedding, h_, w_, 1
        )
    image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']
    if T != 0:
        image_embeds_und = image_embeds_und.reshape(b, T, image_embeds_und.shape[1], -1)
        image_embeds_und = rearrange(image_embeds_und, 'b t l d -> b (t l) d')

    image_embeds = self.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))

    time_embeds = self.time_embed(t, dtype)
    time_embeds_proj = (
        self.time_embed_proj(time_embeds) if hasattr(self, 'time_embed_proj')
        else time_embeds
    )
    if profiler:
        profiler.stop('1. Feature Extraction')

    # --- 2. Compose Input Embeddings ---
    text_len = modality_positions[0][0][0]

    for i, modality_batch in enumerate(modality_positions):
        for j, (offset, length) in enumerate(modality_batch):
            if self.config.add_time_embeds:
                input_embeds_full[i, offset] = time_embeds_proj[i * modality_positions.size(1) + j]
                input_embeds_full[i, offset + 1:offset + 1 + length - 1] = \
                    image_embeds[i * modality_positions.size(1) + j, :max(length - 1, 0)]
            else:
                input_embeds_full[i, offset:offset + length] = \
                    image_embeds[i * modality_positions.size(1) + j, :length]

    # --- 3. LLM Forward Pass ---
    if profiler:
        profiler.start('2. LLM Forward Pass')
    position_ids_full = torch.arange(
        input_embeds_full.shape[1], device=device
    ).unsqueeze(0).repeat(input_embeds_full.shape[0], 1)

    outputs = self.showo(
        inputs_embeds=input_embeds_full,
        attention_mask=attention_mask,
        position_ids=position_ids_full,
        use_cache=False,
        output_hidden_states=True,
        layers_to_skip=layers_to_skip,
        use_full_network=use_full_network,
    )
    if profiler:
        profiler.stop('2. LLM Forward Pass')

    last_hidden_states = outputs.hidden_states[-1]
    logits = outputs.logits
    all_hidden_states_tuple = outputs.hidden_states

    # --- 4. Diffusion Head ---
    if profiler:
        profiler.start('4. Diffusion Head')
    diff_head_input = (
        self.diff_proj(last_hidden_states)
        if hasattr(self, 'diff_proj') else last_hidden_states
    )

    T_cache = getattr(self, '_flashu_diffusion_head_cache_interval', -1)
    current_step = getattr(self, '_flashu_current_step', 0)
    cache = getattr(self, '_flashu_diffusion_cache', None)

    use_cache = should_use_diffusion_cache(current_step, T_cache, use_full_network, cache)

    if use_cache:
        processed_hidden_states = cache
    else:
        processed_hidden_states = diff_head_input
        full_attention_mask = create_block_mask(
            omni_attn_mask(modality_positions),
            B=last_hidden_states.size(0), H=None,
            Q_LEN=last_hidden_states.size(1), KV_LEN=last_hidden_states.size(1),
            device=device,
        )

        position_ids = torch.arange(
            processed_hidden_states.shape[1], device=processed_hidden_states.device
        ).unsqueeze(0)

        for diff_layer in self.diffusion_head_a:
            processed_hidden_states = diff_layer(
                hidden_states=processed_hidden_states,
                adaln_input=time_embeds,
                attention_mask=full_attention_mask,
                position_ids=position_ids,
                modality_positions=modality_positions,
            )[0]

        if should_store_diffusion_cache(current_step, T_cache):
            self._flashu_diffusion_cache = processed_hidden_states.detach()

    v_pred = self.diffusion_head_b(processed_hidden_states, time_embeds, modality_positions)
    if profiler:
        profiler.stop('4. Diffusion Head')

    # --- 5. Output ---
    if text_labels is not None or image_labels is not None:
        return logits, v_pred, last_hidden_states
    else:
        v_pred_ = []
        num_imgs = 0
        for i, modality_batch in enumerate(modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if length > 0:
                    v_pred_.append(v_pred[i, offset:offset + length])
                    num_imgs += 1
        v_pred_ = torch.stack(v_pred_)

        if self.config.add_time_embeds:
            v_pred_ = v_pred_[:, 1:, :]

        v_pred_ = self.unpatchify(v_pred_, h_, w_, T=T)

        if T == 0:
            v_pred_ = rearrange(v_pred_, 'i j k -> i k j')
            v_pred_ = v_pred_.reshape(num_imgs, c, h, w)
        else:
            v_pred_ = rearrange(v_pred_, 'b t l c -> b c t l')
            v_pred_ = v_pred_.reshape(num_imgs, c, T, h, w)

        return logits, v_pred_, all_hidden_states_tuple


@torch.no_grad()
def _accelerated_t2i_generate(
    self,
    image_latents=None, t=None, text_tokens=None, attention_mask=None,
    modality_positions=None, max_seq_len=None, guidance_scale=0.0,
    layers_to_skip: Optional[List[int]] = None,
    layer_skipping_ratio: Optional[float] = None,
    profiler=None,
    **kwargs,
):
    """Text-to-image generation with all FlashU optimizations.

    Orchestrates:
      - Adaptive Guidance Scale s(t)
      - Hybrid FFN Network switching (tau)
      - Dynamic Layer Skipping (r_LS, T_LS)
      - Diffusion Head Caching (T_cache)
    """
    self._flashu_current_step = getattr(self, '_flashu_current_step', 0) + 1
    current_step = self._flashu_current_step
    is_recalc_step = False
    final_layers_to_skip = layers_to_skip

    total_steps = getattr(self, '_flashu_total_inference_steps', 50)
    tau = getattr(self, '_flashu_hybrid_network_final_steps', 0)
    T_LS = getattr(self, '_flashu_layer_skip_recalc_interval', -1)

    # --- Adaptive Guidance Scale s(t) ---
    adaptive_cfg = getattr(self, '_flashu_adaptive_guidance_config', None)
    effective_guidance_scale = compute_adaptive_guidance_scale(
        current_step, adaptive_cfg, guidance_scale
    )

    # --- Hybrid FFN Network switching (Eq. 4, threshold tau) ---
    use_full = should_use_full_network(current_step, total_steps, tau)

    if use_full:
        final_layers_to_skip = None
        if self._flashu_diffusion_cache is not None:
            self._flashu_diffusion_cache = None

    # --- Dynamic Layer Skipping (r_LS, T_LS) ---
    if layer_skipping_ratio is not None and T_LS > 0:
        if not use_full and (current_step - 1) % T_LS == 0:
            is_recalc_step = True
            final_layers_to_skip = None
        else:
            if not use_full:
                final_layers_to_skip = getattr(
                    self, '_flashu_layers_to_skip_cache', final_layers_to_skip
                )

    # --- Forward Pass ---
    _, v, all_hidden_states_tuple = self(
        text_tokens=text_tokens, image_latents=image_latents, t=t,
        attention_mask=attention_mask, modality_positions=modality_positions,
        max_seq_len=max_seq_len, layers_to_skip=final_layers_to_skip,
        profiler=profiler, use_full_network=use_full,
        **kwargs,
    )

    # Update layer importance cache on recalculation steps
    if is_recalc_step:
        self._flashu_layers_to_skip_cache = self.calculate_and_select_layers_to_skip(
            all_hidden_states_tuple, layer_skipping_ratio
        )

    # --- Classifier-Free Guidance with effective s(t) ---
    if effective_guidance_scale > 0.0:
        if profiler:
            profiler.start('Guidance Calculation')
        v_cond, v_uncond = torch.chunk(v, 2)
        v = v_uncond + effective_guidance_scale * (v_cond - v_uncond)
        v_doubled = torch.cat([v, v], dim=0)
        if profiler:
            profiler.stop('Guidance Calculation')
        return v_doubled, all_hidden_states_tuple
    else:
        return v, all_hidden_states_tuple
