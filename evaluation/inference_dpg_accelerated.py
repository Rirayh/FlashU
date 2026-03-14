# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DPG-Bench inference script for the FlashU acceleration framework.

Runs baseline (unaccelerated) and FlashU-accelerated generation on the
DPG-Bench prompt set, then prints a comparison summary.  Default hyper-
parameters match Paper Table 7 for the 1.5B model configuration.

Uses the new FlashU plugin API (flashu package) instead of inline
WandaPruner / SimpleProfiler classes.
"""

import os
import sys
import pathlib
import time
import json
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image
import argparse
import logging

project_root = str(pathlib.Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import torch.nn as nn
from tqdm import tqdm
from accelerate.logging import get_logger
from accelerate import PartialState
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from models import Showo2Qwen2_5
from models import omni_attn_mask
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import (
    get_config, denorm, get_hyper_params,
    path_to_llm_name, load_state_dict, save_images_as_grid,
)
from transport import Sampler, create_transport

from flashu import apply_flashu_patch, FlashUConfig
from flashu.wanda_pruner import WandaPruner
from flashu.structural_pruning import apply_structural_pruning

logger = get_logger(__name__, log_level="INFO")

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)


# ---------------------------------------------------------------------------
# Simple wall-clock profiler
# ---------------------------------------------------------------------------
class SimpleProfiler:
    """Measures wall-clock time of named code sections (with CUDA sync)."""

    def __init__(self):
        self.timings: dict[str, float] = {}
        self.start_times: dict[str, float] = {}

    def start(self, name: str):
        torch.cuda.synchronize()
        self.start_times[name] = time.time()

    def stop(self, name: str):
        torch.cuda.synchronize()
        elapsed = time.time() - self.start_times.pop(name, time.time())
        self.timings[name] = self.timings.get(name, 0.0) + elapsed

    def reset(self):
        self.timings.clear()
        self.start_times.clear()

    def report(self, num_items: int, total_time: float):
        header = (
            "\n" + "=" * 20 + " PROFILING REPORT " + "=" * 20 + "\n"
            f"Total prompts profiled: {num_items}\n"
            f"Total wall time: {total_time:.4f}s\n\n"
        )
        if total_time == 0:
            logger.info(header + "No profiled sections to report.\n")
            return

        lines = [
            f"{'Section':<45} | {'Time (s)':<12} | {'Pct':>7}",
            "-" * 70,
        ]
        for name, sec in sorted(
            self.timings.items(), key=lambda x: x[1], reverse=True
        ):
            pct = (sec / total_time) * 100
            lines.append(f"{name:<45} | {sec:<12.4f} | {pct:6.2f}%")

        unaccounted = total_time - sum(self.timings.values())
        if unaccounted > 0:
            pct = (unaccounted / total_time) * 100
            lines.append(f"{'Unaccounted':<45} | {unaccounted:<12.4f} | {pct:6.2f}%")

        lines.append("=" * 70)
        logger.info(header + "\n".join(lines))


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------
def run_test_strategy(
    model_instance,
    prompts_data: list,
    output_dir: str,
    run_name: str,
    config_params: tuple,
    tokenizer,
    vae,
    sampler_instance,
    config,
    profiler: SimpleProfiler,
    # FlashU acceleration knobs (Paper Sec. 3)
    diffusion_head_cache_interval: int = -1,
    layer_skip_recalc_interval: int = -1,
    layer_skipping_ratio: float = 0.0,
    adaptive_guidance_config: Optional[dict] = None,
    hybrid_network_final_steps: int = 0,
    num_inference_steps: int = 50,
) -> float:
    """Run the full evaluation loop over *prompts_data* and return avg time.

    The 10th prompt (0-indexed: step 9) is profiled in detail; the first
    prompt is treated as a warm-up and excluded from timing statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"--- Starting: {run_name} ---")
    logger.info(f"Output directory: {output_dir}")

    (
        num_t2i, _, _, max_seq, max_text, latent_dim, patch_sz, lat_w, lat_h,
        pad, bos, eos, boi, eoi, _, _, img_pad, _, guidance_scale,
        batch_size, device, weight_type,
    ) = config_params

    validation_prompts = [item["prompt"] for item in prompts_data]
    filenames = [item["prompt_file_name"] for item in prompts_data]
    run_times: list[float] = []

    for step in tqdm(range(len(filenames)), desc=f"Running {run_name}", file=sys.stdout):
        is_warmup = step == 0
        is_profiling = step == 9  # profile the 10th image

        fn = filenames[step]
        prompts = [validation_prompts[step]] * batch_size

        # Reset per-generation acceleration state (Paper Sec. 3).
        model_instance.reset_acceleration_state(
            diffusion_head_cache_interval=diffusion_head_cache_interval,
            layer_skip_recalc_interval=layer_skip_recalc_interval,
            num_inference_steps=num_inference_steps,
            adaptive_guidance_config=adaptive_guidance_config,
            hybrid_network_final_steps=hybrid_network_final_steps,
        )

        current_profiler = profiler if is_profiling else None
        if is_profiling:
            profiler.reset()

        torch.cuda.synchronize()
        t_start = time.time()

        with torch.no_grad():
            (
                batch_text_tokens, batch_text_tokens_null,
                batch_modality_positions, batch_modality_positions_null,
            ) = prepare_gen_input(
                prompts, tokenizer, num_t2i, bos, eos, boi, eoi,
                pad, img_pad, max_text, device,
            )

            z = torch.randn(
                (len(prompts), latent_dim, lat_h * patch_sz, lat_w * patch_sz),
                device=device, dtype=weight_type,
            )

            if guidance_scale > 0:
                z = torch.cat([z, z], dim=0)
                text_tokens = torch.cat(
                    [batch_text_tokens, batch_text_tokens_null], dim=0
                )
                modality_positions = torch.cat(
                    [batch_modality_positions, batch_modality_positions_null], dim=0
                )
            else:
                text_tokens = batch_text_tokens
                modality_positions = batch_modality_positions

            omni_mask_fn = omni_attn_mask(modality_positions)
            block_mask = create_block_mask(
                omni_mask_fn, B=z.size(0), H=None,
                Q_LEN=max_seq, KV_LEN=max_seq, device=device,
            )

            model_kwargs = dict(
                text_tokens=text_tokens,
                attention_mask=block_mask,
                modality_positions=modality_positions,
                output_hidden_states=True,
                max_seq_len=max_seq,
                guidance_scale=guidance_scale,
                layer_skipping_ratio=layer_skipping_ratio,
                profiler=current_profiler,
            )

            model_fn = lambda x, t, **kw: model_instance.t2i_generate(
                image_latents=x, t=t, **kw
            )[0]

            sample_fn = sampler_instance.sample_ode(
                sampling_method=config.transport.sampling_method,
                num_steps=num_inference_steps,
                atol=config.transport.atol,
                rtol=config.transport.rtol,
                time_shifting_factor=config.transport.time_shifting_factor,
            )

            if current_profiler:
                current_profiler.start("Sampler (Total ODE Steps)")
            samples_history = sample_fn(z, model_fn, **model_kwargs)
            if current_profiler:
                current_profiler.stop("Sampler (Total ODE Steps)")

            final_sample = samples_history[-1]
            if guidance_scale > 0:
                final_sample = torch.chunk(final_sample, 2)[0]

        # VAE decode
        if current_profiler:
            current_profiler.start("VAE Decode")
        if config.model.vae_model.type == "wan21":
            decoded = vae.batch_decode(final_sample.unsqueeze(2)).squeeze(2)
        else:
            raise NotImplementedError(
                f"Unsupported VAE type: {config.model.vae_model.type}"
            )
        if current_profiler:
            current_profiler.stop("VAE Decode")

        # Save images
        image_array = denorm(decoded)
        pil_images = [Image.fromarray(img) for img in image_array]
        grid_map = {1: (1, 1), 4: (2, 2), 16: (4, 4)}
        grid_size = grid_map.get(batch_size, (1, 1))
        save_images_as_grid(pil_images, fn, output_dir, grid_size=grid_size)

        torch.cuda.synchronize()
        iter_time = time.time() - t_start

        if not is_warmup:
            run_times.append(iter_time)

        if is_profiling:
            profiler.report(num_items=1, total_time=iter_time)

    if run_times:
        avg_time = sum(run_times) / len(run_times)
        logger.info(
            f"[{run_name}] Average time over {len(run_times)} prompts "
            f"(excl. warm-up): {avg_time:.4f}s"
        )
        return avg_time
    return 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # -- Argument parsing ----------------------------------------------------
    parser = argparse.ArgumentParser(
        description="DPG-Bench inference with FlashU acceleration."
    )
    parser.add_argument("--device_id", type=int, required=True)
    parser.add_argument("--num_devices", type=int, required=True)
    parser.add_argument(
        "--log_dir", type=str, default="logs_flashu",
        help="Directory for log files.",
    )
    parser.add_argument(
        "--ffn_pruning_ratio", type=float, default=FlashUConfig.r_p,
        help="FFN pruning ratio r_p (Paper Sec. 3.2).",
    )
    parser.add_argument(
        "--layer_skipping_ratio", type=float, default=FlashUConfig.r_LS,
        help="Layer-skipping ratio r_LS (Paper Sec. 3.3).",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=None,
        help="Override total diffusion steps (default: sum of guidance schedule).",
    )
    parser.add_argument(
        "--benchmark", type=str, default="dpg",
        choices=["dpg"],
        help="Benchmark to run (currently only 'dpg' is implemented).",
    )
    args, _ = parser.parse_known_args()

    # -- Logging setup -------------------------------------------------------
    state = PartialState()
    os.makedirs(args.log_dir, exist_ok=True)
    log_filename = os.path.join(
        args.log_dir, f"flashu_run_{time.strftime('%Y%m%d-%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info(f"Log file: {log_filename}")

    # -- Config & device -----------------------------------------------------
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_type = torch.bfloat16
    outdir = config.get("outdir", "flashu_dpg_results")

    # -- Model, tokenizer, VAE -----------------------------------------------
    from models import WanVAE

    vae_model = WanVAE(
        vae_pth=config.model.vae_model.pretrained_model_path,
        dtype=weight_type, device=device,
    )
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )

    # Load baseline model (not accelerated variant)
    if config.model.showo.get("load_from_showo", False):
        model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path, use_safetensors=False,
        ).to(device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(device)
        state_dict = load_state_dict(config.model_path)
        model.load_state_dict(state_dict)

    model.to(weight_type).eval()

    # -- Hyper-parameters tuple -----------------------------------------------
    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1

    if config.get("guidance_scale", None) is not None:
        config.transport.guidance_scale = config.guidance_scale
    if config.get("num_inference_steps", None) is not None:
        config.transport.num_inference_steps = config.num_inference_steps

    hyper_params = list(get_hyper_params(config, text_tokenizer, showo_token_ids))
    hyper_params.extend([int(config.get("batch_size", 4)), device, weight_type])
    hyper_params = tuple(hyper_params)

    (
        num_t2i, _, _, max_seq, max_text, latent_dim, patch_sz, lat_w, lat_h,
        pad, bos, eos, boi, eoi, _, _, img_pad, _, guidance_scale,
        batch_size, device, weight_type,
    ) = hyper_params

    # -- Prompt data ----------------------------------------------------------
    validation_prompts_file = config.get(
        "validation_prompts_file",
        config.dataset.params.validation_prompts_file,
    )
    with open(validation_prompts_file) as fp:
        meta_data_full = json.load(fp)

    num_prompts = int(config.get("num_prompts", len(meta_data_full)))
    meta_data_full = meta_data_full[:num_prompts]

    if args.num_devices > 1:
        meta_data_subset = list(
            np.array_split(meta_data_full, args.num_devices)[args.device_id]
        )
    else:
        meta_data_subset = meta_data_full

    logger.info(
        f"Device {args.device_id}/{args.num_devices}: "
        f"running on {len(meta_data_subset)}/{num_prompts} prompts."
    )

    # -- Transport / sampler --------------------------------------------------
    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
    )
    sampler = Sampler(transport)

    # -- FlashU configuration (Paper Table 7, 1.5B) ---------------------------
    flashu_config = FlashUConfig(
        r_p=args.ffn_pruning_ratio,
        r_LS=args.layer_skipping_ratio,
        num_inference_steps=args.num_inference_steps,
    )

    ffn_pruning_ratio = flashu_config.r_p
    layer_skipping_ratio = flashu_config.r_LS
    layer_skip_recalc_interval = flashu_config.T_LS
    diffusion_head_cache_interval = flashu_config.T_cache
    hybrid_network_final_steps = flashu_config.tau
    adaptive_guidance_schedule = flashu_config.adaptive_guidance_schedule
    total_steps = flashu_config.total_steps

    adaptive_guidance_config = flashu_config.adaptive_guidance_config
    if adaptive_guidance_config is not None:
        adaptive_guidance_config["original_gs"] = guidance_scale

    # -- Shared evaluation arguments ------------------------------------------
    profiler = SimpleProfiler()
    common_eval_args = dict(
        prompts_data=meta_data_subset,
        config_params=hyper_params,
        tokenizer=text_tokenizer,
        vae=vae_model,
        sampler_instance=sampler,
        config=config,
        profiler=profiler,
    )

    results: dict[str, float] = {}

    # ======================================================================
    # 1. Baseline (no acceleration)
    # ======================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Baseline (no acceleration)")
    logger.info("=" * 60)

    # Apply FlashU patch for baseline run (needed for reset_acceleration_state)
    apply_flashu_patch(model, FlashUConfig(
        r_p=0.0, r_LS=0.0, T_LS=-1, T_cache=-1, tau=0,
        adaptive_guidance_schedule=None,
    ))

    baseline_outdir = os.path.join(outdir, "Baseline")
    avg_time = run_test_strategy(
        model_instance=model,
        output_dir=baseline_outdir,
        run_name="Baseline",
        layer_skipping_ratio=0.0,
        diffusion_head_cache_interval=-1,
        layer_skip_recalc_interval=-1,
        adaptive_guidance_config=None,
        hybrid_network_final_steps=0,
        num_inference_steps=config.num_inference_steps,
        **common_eval_args,
    )
    if avg_time > 0:
        results["Baseline"] = avg_time

    del model
    torch.cuda.empty_cache()

    # ======================================================================
    # 2. FlashU accelerated (single config from Paper Table 7)
    # ======================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: FlashU Accelerated")
    logger.info(
        f"  r_p={ffn_pruning_ratio}, r_LS={layer_skipping_ratio}, "
        f"T_LS={layer_skip_recalc_interval}, T_cache={diffusion_head_cache_interval}, "
        f"tau={hybrid_network_final_steps}/{total_steps}, steps={total_steps}"
    )
    logger.info("=" * 60)

    # Reload a clean baseline model for pruning.
    if config.model.showo.get("load_from_showo", False):
        pruned_model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path, use_safetensors=False,
        ).to(device)
    else:
        pruned_model = Showo2Qwen2_5(**config.model.showo).to(device)
        sd = load_state_dict(config.model_path)
        pruned_model.load_state_dict(sd)

    pruned_model.to(weight_type).eval()

    # -- Wanda calibration (Paper Sec. 3.2) ----------------------------------
    logger.info("Preparing calibration batch for Wanda pruning...")
    cal_prompts = [meta_data_subset[0]["prompt"]] * batch_size
    cal_text, _, cal_pos, _ = prepare_gen_input(
        cal_prompts, text_tokenizer, num_t2i, bos, eos, boi, eoi,
        pad, img_pad, max_text, device,
    )
    cal_mask_fn = omni_attn_mask(cal_pos)
    cal_block_mask = create_block_mask(
        cal_mask_fn, B=batch_size, H=None,
        Q_LEN=max_seq, KV_LEN=max_seq, device=device,
    )
    calibration_batch = {
        "input_ids": cal_text,
        "attention_mask": cal_block_mask,
        "position_ids": None,
        "past_key_values": None,
        "inputs_embeds": None,
        "use_cache": False,
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict": True,
    }

    pruner = WandaPruner(pruned_model.showo)
    with torch.no_grad():
        pruner.add_batch(calibration_batch)
    pruner.remove_hooks()

    scores = pruner.get_structural_importance_scores()

    # Apply structural pruning via flashu package
    apply_structural_pruning(
        pruned_model,
        head_prune_ratio=0.0,
        ffn_prune_ratio=ffn_pruning_ratio,
        scores_by_layer=scores,
    )
    logger.info("Wanda structural pruning applied.")
    del pruner, scores
    torch.cuda.empty_cache()

    # Apply FlashU patch to the pruned model
    apply_flashu_patch(pruned_model, flashu_config)

    # -- Run FlashU evaluation -----------------------------------------------
    flashu_outdir = os.path.join(
        outdir,
        f"FlashU_rp{int(ffn_pruning_ratio*100)}_rLS{int(layer_skipping_ratio*100)}"
        f"_T{total_steps}_HN{hybrid_network_final_steps}",
    )
    avg_time = run_test_strategy(
        model_instance=pruned_model,
        output_dir=flashu_outdir,
        run_name="FlashU",
        layer_skipping_ratio=layer_skipping_ratio,
        diffusion_head_cache_interval=diffusion_head_cache_interval,
        layer_skip_recalc_interval=layer_skip_recalc_interval,
        adaptive_guidance_config=adaptive_guidance_config,
        hybrid_network_final_steps=hybrid_network_final_steps,
        num_inference_steps=total_steps,
        **common_eval_args,
    )
    if avg_time > 0:
        results["FlashU"] = avg_time

    del pruned_model
    torch.cuda.empty_cache()

    # ======================================================================
    # 3. Comparison summary
    # ======================================================================
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    header = f"{'Strategy':<30} | {'Avg Time (s)':>13} | {'Speedup':>8}"
    sep = "-" * 58
    logger.info(header)
    logger.info(sep)

    baseline_time = results.get("Baseline", 0.0)
    for name in ["Baseline", "FlashU"]:
        if name not in results:
            continue
        t = results[name]
        speedup = (baseline_time / t) if (t > 0 and baseline_time > 0) else 0.0
        logger.info(f"{name:<30} | {t:>13.4f} | {speedup:>7.2f}x")

    logger.info(sep)
    logger.info(f"All outputs saved to: {outdir}")
    logger.info("=" * 60)
