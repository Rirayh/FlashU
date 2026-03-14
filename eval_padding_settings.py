"""
DPG-Bench evaluation with FlashU acceleration.

Usage:
  CUDA_VISIBLE_DEVICES=X python eval_padding_settings.py \
    --accel_mode baseline|flashu \
    config=configs/showo2_1.5b_demo_432x432.yaml \
    guidance_scale=10 num_inference_steps=50 \
    validation_prompts_file=prompts/dpg_bench_meta_data.json \
    outdir=dpg_eval
"""

import sys
import os
import pathlib
import time
import json
import argparse
import logging

project_root = str(pathlib.Path(__file__).resolve().parent)
sys.path.insert(0, project_root)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from models import Showo2Qwen2_5, WanVAE
from models import omni_attn_mask
from models.misc import get_text_tokenizer
from utils import (
    get_config, get_hyper_params, path_to_llm_name,
    load_state_dict, denorm, save_images_as_grid,
)
from transport import Sampler, create_transport
from flashu import apply_flashu_patch, FlashUConfig
from flashu.wanda_pruner import WandaPruner
from flashu.structural_pruning import apply_structural_pruning

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)


def prepare_gen_input(prompts, text_tokenizer, num_image_tokens,
                      bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                      max_text_len, device):
    batch_text_tokens = []
    batch_modality_positions = []
    batch_text_tokens_null = []
    batch_modality_positions_null = []
    for prompt in prompts:
        text_tokens = text_tokenizer(prompt, add_special_tokens=False)['input_ids'][:max_text_len]
        modality_positions = torch.tensor([len(text_tokens) + 1 + 1, num_image_tokens]).unsqueeze(0)
        text_tokens = ([bos_id] + text_tokens + [boi_id] +
                       [img_pad_id] * num_image_tokens +
                       [eoi_id] + [eos_id] +
                       [pad_id] * (max_text_len - len(text_tokens)))
        batch_text_tokens.append(torch.tensor(text_tokens))
        batch_modality_positions.append(modality_positions)

        text_tokens_null = []
        modality_positions_null = torch.tensor([len(text_tokens_null) + 1 + 1, num_image_tokens]).unsqueeze(0)
        text_tokens_null = ([bos_id] + text_tokens_null + [boi_id] +
                            [img_pad_id] * num_image_tokens +
                            [eoi_id] + [eos_id] +
                            [pad_id] * (max_text_len - len(text_tokens_null)))
        batch_text_tokens_null.append(torch.tensor(text_tokens_null))
        batch_modality_positions_null.append(modality_positions_null)

    batch_text_tokens = torch.stack(batch_text_tokens, dim=0).to(device)
    batch_modality_positions = torch.stack(batch_modality_positions, dim=0).to(device)
    batch_text_tokens_null = torch.stack(batch_text_tokens_null, dim=0).to(device)
    batch_modality_positions_null = torch.stack(batch_modality_positions_null, dim=0).to(device)
    seq_len = batch_text_tokens.shape[1]
    return batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null, seq_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accel_mode", type=str, required=True, choices=["baseline", "flashu"])
    args, _ = parser.parse_known_args()

    accel_mode = args.accel_mode
    setting_tag = accel_mode

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_type = torch.bfloat16

    config = get_config()

    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )

    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
    if config.get("guidance_scale", None) is not None:
        config.transport.guidance_scale = config.guidance_scale
    if config.get("num_inference_steps", None) is not None:
        config.transport.num_inference_steps = config.num_inference_steps

    hyper_list = list(get_hyper_params(config, text_tokenizer, showo_token_ids))
    (num_t2i, _, _, max_seq, max_text, latent_dim, patch_sz, lat_w, lat_h,
     pad, bos, eos, boi, eoi, _, _, img_pad, _, guidance_scale) = tuple(hyper_list)

    validation_prompts_file = config.get(
        "validation_prompts_file",
        config.dataset.params.validation_prompts_file,
    )
    with open(validation_prompts_file) as fp:
        meta_data_full = json.load(fp)

    outdir_base = config.get("outdir", "dpg_eval")
    output_dir = os.path.join(outdir_base, setting_tag)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Setting: {setting_tag}")
    logger.info(f"Total prompts: {len(meta_data_full)}")
    logger.info(f"Output: {output_dir}")

    vae_model = WanVAE(
        vae_pth=config.model.vae_model.pretrained_model_path,
        dtype=weight_type, device=device,
    )

    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
    )
    sampler = Sampler(transport)

    flashu_config = FlashUConfig()
    adaptive_guidance_config = flashu_config.adaptive_guidance_config
    if adaptive_guidance_config is not None:
        adaptive_guidance_config["original_gs"] = guidance_scale

    if config.model.showo.get("load_from_showo", False):
        model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path, use_safetensors=False,
        ).to(device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(device)
        sd = load_state_dict(config.model_path)
        model.load_state_dict(sd)
    model.to(weight_type).eval()

    if accel_mode == "flashu":
        cal_prompts = [meta_data_full[0]["prompt"]]
        cal_text, _, cal_pos, _, cal_seq_len = prepare_gen_input(
            cal_prompts, text_tokenizer, num_t2i, bos, eos, boi, eoi, pad, img_pad, max_text, device,
        )
        cal_mask_fn = omni_attn_mask(cal_pos)
        cal_block_mask = create_block_mask(
            cal_mask_fn, B=1, H=None,
            Q_LEN=cal_seq_len, KV_LEN=cal_seq_len, device=device,
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
        pruner = WandaPruner(model.showo)
        with torch.no_grad():
            pruner.add_batch(calibration_batch)
        pruner.remove_hooks()
        scores = pruner.get_structural_importance_scores()
        apply_structural_pruning(
            model, head_prune_ratio=0.0,
            ffn_prune_ratio=flashu_config.r_p,
            scores_by_layer=scores,
        )
        del pruner, scores
        torch.cuda.empty_cache()

        apply_flashu_patch(model, flashu_config)
        num_steps = flashu_config.total_steps
        diff_cache_interval = flashu_config.T_cache
        ls_recalc_interval = flashu_config.T_LS
        ls_ratio = flashu_config.r_LS
        hybrid_final = flashu_config.tau
    else:
        apply_flashu_patch(model, FlashUConfig(
            r_p=0.0, r_LS=0.0, T_LS=-1, T_cache=-1, tau=0,
            adaptive_guidance_schedule=None,
        ))
        num_steps = config.num_inference_steps
        diff_cache_interval = -1
        ls_recalc_interval = -1
        ls_ratio = 0.0
        hybrid_final = 0
        adaptive_guidance_config = None

    validation_prompts = [item["prompt"] for item in meta_data_full]
    filenames = [item["prompt_file_name"] for item in meta_data_full]
    run_times = []

    for step in tqdm(range(len(filenames)), desc=setting_tag, file=sys.stdout):
        fn = filenames[step]
        save_name = fn if "." in fn else fn + ".png"
        out_path = os.path.join(output_dir, save_name)
        if os.path.exists(out_path):
            continue

        prompts = [validation_prompts[step]]

        model.reset_acceleration_state(
            diffusion_head_cache_interval=diff_cache_interval,
            layer_skip_recalc_interval=ls_recalc_interval,
            num_inference_steps=num_steps,
            adaptive_guidance_config=adaptive_guidance_config,
            hybrid_network_final_steps=hybrid_final,
        )

        torch.cuda.synchronize()
        t_start = time.time()

        with torch.no_grad():
            bt, bt_null, bp, bp_null, seq_len = prepare_gen_input(
                prompts, text_tokenizer, num_t2i, bos, eos, boi, eoi,
                pad, img_pad, max_text, device,
            )

            z = torch.randn(
                (1, latent_dim, lat_h * patch_sz, lat_w * patch_sz),
                device=device, dtype=weight_type,
            )

            if guidance_scale > 0:
                z = torch.cat([z, z], dim=0)
                text_tokens = torch.cat([bt, bt_null], dim=0)
                modality_positions = torch.cat([bp, bp_null], dim=0)
            else:
                text_tokens = bt
                modality_positions = bp

            omni_mask_fn = omni_attn_mask(modality_positions)
            block_mask = create_block_mask(
                omni_mask_fn, B=z.size(0), H=None,
                Q_LEN=seq_len, KV_LEN=seq_len, device=device,
            )

            model_kwargs = dict(
                text_tokens=text_tokens,
                attention_mask=block_mask,
                modality_positions=modality_positions,
                output_hidden_states=True,
                max_seq_len=seq_len,
                guidance_scale=guidance_scale,
                layer_skipping_ratio=ls_ratio,
            )

            model_fn = lambda x, t, **kw: model.t2i_generate(
                image_latents=x, t=t, **kw
            )[0]

            sample_fn = sampler.sample_ode(
                sampling_method=config.transport.sampling_method,
                num_steps=num_steps,
                atol=config.transport.atol,
                rtol=config.transport.rtol,
                time_shifting_factor=config.transport.time_shifting_factor,
            )

            samples_history = sample_fn(z, model_fn, **model_kwargs)
            final_sample = samples_history[-1]
            if guidance_scale > 0:
                final_sample = torch.chunk(final_sample, 2)[0]

        if config.model.vae_model.type == "wan21":
            decoded = vae_model.batch_decode(final_sample.unsqueeze(2)).squeeze(2)
        else:
            raise NotImplementedError(f"Unsupported VAE: {config.model.vae_model.type}")

        image_array = denorm(decoded)
        pil_images = [Image.fromarray(img) for img in image_array]

        save_name = fn if "." in fn else fn + ".png"
        pil_images[0].save(os.path.join(output_dir, save_name))

        torch.cuda.synchronize()
        elapsed = time.time() - t_start

        if step > 0:
            run_times.append(elapsed)

    if run_times:
        avg = sum(run_times) / len(run_times)
        logger.info(f"[{setting_tag}] Avg time: {avg:.4f}s over {len(run_times)} prompts")
    logger.info(f"[{setting_tag}] Done. Images saved to {output_dir}")


if __name__ == "__main__":
    main()
