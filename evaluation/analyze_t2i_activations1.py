# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# This script is an adaptation of analyze_mmu_activations_v3.py for T2I analysis.
# Key differences:
# - The "time" dimension is diffusion steps, not generated tokens.
# - Analysis focuses on the full hidden state of the LLM backbone during denoising.
# - A custom sampler wrapper is used to capture activations during the ODE solve.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import gc
from pathlib import Path
import warnings
import itertools

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from omegaconf import OmegaConf

from models import Showo2Qwen2_5, omni_attn_mask_naive
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import get_hyper_params, path_to_llm_name, load_state_dict, denorm
from transport import Sampler, create_transport

# Suppress Matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# --- Global storage for hooks ---
CURRENT_FORWARD_PASS_ACTIVATIONS = []

def get_capture_hook():
    """Hook factory to capture input and output of a module."""
    def hook(module, input_tensor, output_tensor):
        global CURRENT_FORWARD_PASS_ACTIVATIONS
        CURRENT_FORWARD_PASS_ACTIVATIONS.append({
            "input": input_tensor[0].detach(),
            "output": output_tensor[0].detach()
        })
    return hook


# --- Reusable Analysis & Plotting Functions (from MMU script) ---
# Note: The term "Token" in titles/labels will be replaced with "Step" for clarity.

def calculate_intra_layer_similarity(activations, num_layers, time_label="Step"):
    """
    Calculates intra-layer similarity (input vs. output of the same layer) for generative models.
    Averages over all steps.
    """
    sorted_steps = sorted(activations.keys())
    activations_list = [activations[step] for step in sorted_steps]
    num_steps = len(activations_list)

    print(f"    [Metrics] Calculating Intra-Layer Similarity for {num_layers} layers.")

    intra_sims_per_layer = [[] for _ in range(num_layers)]
    for t in range(num_steps):
        for l in range(num_layers):
            input_l = activations_list[t][l]['input'][0] # Select the 0-th element of the batch
            output_l = activations_list[t][l]['output'][0] # Select the 0-th element of the batch
            if input_l.shape == output_l.shape:
                sim = F.cosine_similarity(input_l.flatten().to(torch.float32), output_l.flatten().to(torch.float32), dim=0).cpu().item()
                intra_sims_per_layer[l].append(sim)
            else:
                intra_sims_per_layer[l].append(0) # Handle potential shape mismatches gracefully

    # Average similarity for each layer across all steps
    avg_intra_sim = np.array([np.mean(sims) if sims else 0 for sims in intra_sims_per_layer])
    return avg_intra_sim


def calculate_metrics_generative(activations, num_layers, time_label="Step"):
    """
    Calculates metrics for generative models (T2I, Mixed-Modality).
    'Full Sequence' refers to the full LLM hidden state.
    'Last Token' is not applicable, so we use a placeholder or mean representation.
    """
    # activations is a dict {step_idx: layer_activations}, convert to list of lists
    sorted_steps = sorted(activations.keys())
    activations_list = [activations[step] for step in sorted_steps]
    num_steps = len(activations_list)

    print(f"\n    [Metrics] Calculating for {num_steps} {time_label}s across {num_layers} layers.")

    # Inter-layer similarity (averaged over steps)
    inter_sims = []
    for t in range(num_steps):
        sim_matrix = np.zeros((num_layers, num_layers))
        for i, j in itertools.product(range(num_layers), repeat=2):
            # Select the 0-th element of the batch (the real prompt, not the null one)
            output_i = activations_list[t][i]['output'][0]
            output_j = activations_list[t][j]['output'][0]
            if output_i.shape == output_j.shape:
                sim_matrix[i, j] = F.cosine_similarity(output_i.flatten().to(torch.float32), output_j.flatten().to(torch.float32), dim=0).cpu().item()
            else:
                sim_matrix[i, j] = 0 # Handle potential shape mismatches gracefully
        inter_sims.append(sim_matrix)

    avg_inter_sim = np.mean(inter_sims, axis=0)

    # Temporal similarity (all-pairs, averaged over layers) - Keep for completeness, though not plotted in this specific request
    temporal_matrix = np.zeros((num_steps, num_steps))
    if num_steps > 1:
        sims_per_pair = [[] for _ in range(num_steps * num_steps)]
        for l in range(num_layers):
            for t1, t2 in itertools.product(range(num_steps), repeat=2):
                # Select the 0-th element of the batch
                output_t1 = activations_list[t1][l]['output'][0]
                output_t2 = activations_list[t2][l]['output'][0]
                sim = F.cosine_similarity(output_t1.flatten().to(torch.float32), output_t2.flatten().to(torch.float32), dim=0).cpu().item()
                sims_per_pair[t1 * num_steps + t2].append(sim)

        avg_sims = [np.mean(sims) if sims else 0 for sims in sims_per_pair]
        temporal_matrix = np.array(avg_sims).reshape(num_steps, num_steps)

    return avg_inter_sim, temporal_matrix # Now also returning temporal_matrix, even if not explicitly plotted in new scheme


def plot_overall_average_metrics(avg_intra_sim, avg_inter_sim, title, output_path, num_layers):
    """Generates and saves a 1x2 plot for overall average generative metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8)) # Adjusted figsize for 1x2
    fig.suptitle(title, fontsize=20, y=1.02) # Adjusted title font size

    # --- Plot Intra-Layer Similarity (Input vs. Output) ---
    axes[0].bar(range(num_layers), avg_intra_sim, color='skyblue')
    axes[0].set_title('Intra-Layer Similarity (Input vs. Output)', fontsize=14)
    axes[0].set_xlabel('Transformer Layer Index', fontsize=12)
    axes[0].set_ylabel('Cosine Similarity', fontsize=12)
    axes[0].set_xticks(range(0, num_layers, 1))
    axes[0].set_ylim(0.8, 1.0) # Match the provided image's y-axis range
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.6)


    # --- Plot Inter-Layer Similarity (Depth-wise Redundancy) ---
    sns.heatmap(avg_inter_sim, ax=axes[1], cmap='viridis', annot=False, vmin=0, vmax=1)
    axes[1].set_title('Inter-Layer Similarity (Depth-wise Redundancy)', fontsize=14)
    axes[1].set_xlabel('Transformer Layer Index', fontsize=12)
    axes[1].set_ylabel('Transformer Layer Index', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close(fig)
    print(f"    -> Overall average similarity plot saved to {output_path}")

# SVD analysis functions are not modified as per request to keep it minimal
def analyze_and_plot_svd_generative(activations, sample_output_dir, time_label="Step"):
    """Performs SVD analysis for generative models."""
    print(f"    [SVD] Analyzing representations...")
    # NOTE: The layers are 28, but SVD analysis from 2-26 is fine.
    layers_to_analyze = range(2, 26)

    sorted_steps = sorted(activations.keys())
    activations_list = [activations[step] for step in sorted_steps]
    num_steps = len(activations_list)

    svd_by_layer_dir = sample_output_dir / "svd_by_layer_over_time"
    svd_by_time_dir = sample_output_dir / "svd_by_time_over_layers"
    svd_by_layer_dir.mkdir(exist_ok=True)
    svd_by_time_dir.mkdir(exist_ok=True)

    # By Layer
    for l in tqdm(layers_to_analyze, desc="    SVD by Layer"):
        # --- FIX: Select only the 0-th element from the batch ---
        reps = [activations_list[t][l]['output'][0].cpu().to(torch.float32).mean(dim=0).squeeze() for t in range(num_steps)]
        if not reps: continue
        reps_tensor = torch.stack(reps) # Shape will be [num_steps, D]
        U, S, _ = torch.linalg.svd(reps_tensor, full_matrices=False)
        projected = U[:, :2] * S[:2] # Shape will be [num_steps, 2]

        plt.figure(figsize=(10, 8))
        # Now 'projected' (size num_steps) matches 'sorted_steps' (size num_steps)
        scatter = plt.scatter(projected[:, 0], projected[:, 1], c=sorted_steps, cmap='viridis', alpha=0.8)
        plt.title(f'SVD of Layer {l} Representations Over {time_label}s', fontsize=16)
        plt.xlabel('PC 1'); plt.ylabel('PC 2'); plt.grid(True)
        plt.colorbar(scatter, label=f'{time_label} Index')
        plt.savefig(svd_by_layer_dir / f"layer_{l}_svd.png"); plt.close()

    # By Time
    for t_idx, t in enumerate(tqdm(sorted_steps, desc="    SVD by Time")):
        # --- FIX: Select only the 0-th element from the batch ---
        reps = [activations_list[t_idx][l]['output'][0].cpu().to(torch.float32).mean(dim=0).squeeze() for l in layers_to_analyze]
        if not reps: continue
        reps_tensor = torch.stack(reps) # Shape will be [num_layers_analyzed, D]
        U, S, _ = torch.linalg.svd(reps_tensor, full_matrices=False)
        projected = U[:, :2] * S[:2] # Shape will be [num_layers_analyzed, 2]

        plt.figure(figsize=(10, 8))
        # Now 'projected' (size num_layers_analyzed) matches 'list(layers_to_analyze)'
        scatter = plt.scatter(projected[:, 0], projected[:, 1], c=list(layers_to_analyze), cmap='viridis', alpha=0.8)
        plt.title(f'SVD of Representations at {time_label} {t} Across Layers', fontsize=16)
        plt.xlabel('PC 1'); plt.ylabel('PC 2'); plt.grid(True)
        plt.colorbar(scatter, label='Layer Index')
        plt.savefig(svd_by_time_dir / f"step_{t}_svd.png"); plt.close()


@torch.no_grad()
def sample_and_capture(sampler, z, model_fn, model_kwargs, config):
    """
    A wrapper for the sampler.sample_ode to capture activations at each step.
    """
    global CURRENT_FORWARD_PASS_ACTIVATIONS
    all_activations = {}

    # --- FIX: Hook the self_attn modules, not the full layers ---
    target_layers = model_kwargs['model'].showo.model.layers
    hooks = [layer.register_forward_hook(get_capture_hook()) for layer in target_layers]

    def model_wrapper_fn(x, t):
        nonlocal all_activations
        global CURRENT_FORWARD_PASS_ACTIVATIONS # <-- This is the scope fix

        CURRENT_FORWARD_PASS_ACTIVATIONS = []

        # --- FIX: Handle batched time tensor from ODE solver ---
        time_scalar = t[0].item() if t.numel() > 0 else t.item()
        step_idx = int((1.0 - time_scalar) * (config.transport.num_inference_steps - 1))
        # --- End of FIX ---

        v = model_fn(image_latents=x, t=t, **model_kwargs)

        if CURRENT_FORWARD_PASS_ACTIVATIONS:
            all_activations[step_idx] = CURRENT_FORWARD_PASS_ACTIVATIONS

        return v

    sample_fn = sampler.sample_ode(
        sampling_method=config.transport.sampling_method,
        num_steps=config.transport.num_inference_steps,
        atol=config.transport.atol, rtol=config.transport.rtol,
    )
    samples = sample_fn(z, model_wrapper_fn)

    for hook in hooks: hook.remove()

    final_sample = samples[-1]
    return final_sample, all_activations

def main():
    # --- Configuration & Setup ---
    cli_args = OmegaConf.from_cli()
    if 'config' not in cli_args:
        raise ValueError("Please provide a configuration file, e.g., config=configs/showo2_1.5b_demo_432x432.yaml")
    config = OmegaConf.load(cli_args.config)
    config = OmegaConf.merge(config, cli_args)

    # --- [REMOVED] FIX 4 was here ---
    # The previous fix (forcing config.model.showo.add_time_embeds = False)
    # was incorrect and had no effect on the loaded model's internal state.

    output_dir = Path("analysis_results_t2i")
    output_dir.mkdir(exist_ok=True)

    # --- Model Loading ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    print(f"--- Using device: {device} with dtype: {weight_type} ---")

    if config.model.vae_model.type == 'wan21':
        from models import WanVAE
        vae_model = WanVAE(vae_pth=config.model.vae_model.pretrained_model_path, dtype=weight_type, device=device)
    else: raise NotImplementedError

    text_tokenizer, showo_token_ids = get_text_tokenizer(config.model.showo.llm_model_path, add_showo_tokens=True, return_showo_token_ids=True, llm_name=path_to_llm_name[config.model.showo.llm_model_path])
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    # --- NEW FIX (replaces FIX 4): Emulate the working script's setup ---
    # The working script increments the token count *before*
    # get_hyper_params if time embeds are used. This ensures
    # prepare_gen_input creates data with the correct sequence length
    # (e.g., 730 tokens = 729 image + 1 time). The model's
    # internal logic (which we can't change) *expects* this
    # extra token, which it strips, leaving the correct 729 for unpatchify.
    if config.model.showo.add_time_embeds:
        print("   [Debug] add_time_embeds=True. Incrementing num_t2i_image_tokens by 1 to match model's expectation.")
        # Note: We only increment T2I tokens, as this script is T2I-only
        config.dataset.preprocessing.num_t2i_image_tokens += 1
    # --- End of NEW FIX ---

    # --- FIX 1: Correctly unpack img_pad_id (and other needed params) ---
    # (Based on the working script's get_hyper_params return order)
    num_t2i_tokens, _, _, _, max_text_len, image_latent_dim, patch_size, latent_width, latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, _, _, img_pad_id, _, _ = \
        get_hyper_params(config, text_tokenizer, showo_token_ids)

    model = Showo2Qwen2_5.from_pretrained(config.model.showo.pretrained_model_path, use_safetensors=False).to(device, dtype=weight_type)
    model.eval()

    # --- FIX 2: Correctly parse JSON prompt file ---
    with open("prompts/dpg_bench_meta_data.json", "r") as f:
        try:
            meta_data = json.load(f)
            prompts = [item['prompt'] for item in meta_data]
        except (json.JSONDecodeError, TypeError, KeyError):
            # Fallback in case it's a simple list of strings per line
            print("Warning: Could not parse JSON, falling back to readlines.")
            f.seek(0)
            prompts = [line.strip() for line in f if line.strip()]

    transport = create_transport(path_type=config.transport.path_type, prediction=config.transport.prediction)
    sampler = Sampler(transport)
# --- Global lists to store metrics for averaging ---
    all_intra_sims = []
    all_inter_sims = []
    MAX_SAMPLES_FOR_AVERAGE = 200 #0 # Limit to 20 samples for averaging

    # --- Main Analysis Loop ---
    for i, prompt in enumerate(tqdm(prompts, desc="Analyzing T2I Prompts")):
        if len(all_intra_sims) >= MAX_SAMPLES_FOR_AVERAGE:
            print(f"    Reached {MAX_SAMPLES_FOR_AVERAGE} samples, stopping for averaging.")
            break # Stop after MAX_SAMPLES_FOR_AVERAGE samples
        sample_name = f"prompt_{i}"
        sample_output_dir = output_dir / sample_name
        sample_output_dir.mkdir(exist_ok=True)
        print(f"\n--- Processing prompt {i+1}/{len(prompts)}: '{prompt[:50]}...' ---")

        # 1. Prepare Inputs
        # --- FIX 1 (cont.): Pass the correct img_pad_id variable ---
        batch_text_tokens, batch_text_tokens_null, batch_modality_positions, _ = \
            prepare_gen_input([prompt], text_tokenizer, num_t2i_tokens, 
                              bos_id, eos_id, boi_id, eoi_id, 
                              pad_id, img_pad_id, # <-- Was -100, now correct variable
                              max_text_len, device)

        # --- FIX 3: Use latent_width/height and patch_size for z shape ---
        # The model expects noise in the VAE latent space dimensions,
        # not the final pixel resolution dimensions.
        z = torch.randn((1, image_latent_dim, latent_height * patch_size, latent_width * patch_size)).to(weight_type).to(device)
        
        text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
        modality_positions = batch_modality_positions.repeat(2, 1, 1)
        
        block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                          text_tokens.size(1),
                                          modality_positions,
                                          device).to(weight_type)
        
        model_kwargs = dict(
            model=model, # Pass model for the wrapper to use
            text_tokens=text_tokens,
            attention_mask=block_mask,
            modality_positions=modality_positions,
            guidance_scale=config.guidance_scale,
        )

        # 2. Sample & Capture
        final_sample, activations = sample_and_capture(sampler, torch.cat([z, z]), model.t2i_generate, model_kwargs, config)

        if not activations:
            print("    - Warning: No activations captured, skipping sample.")
            continue

        # 3. Calculate Intra-Layer & Inter-Layer Similarity
        num_layers = len(model.showo.model.layers)
        intra_sim = calculate_intra_layer_similarity(activations, num_layers)
        inter_sim, _ = calculate_metrics_generative(activations, num_layers, time_label="Diffusion Step") # Only need inter_sim here

        all_intra_sims.append(intra_sim)
        all_inter_sims.append(inter_sim)

        # # 5. Decode and save final image for reference (Optional, uncomment if needed per sample)
        # final_sample = torch.chunk(final_sample, 2)[0].unsqueeze(2)
        # image = vae_model.batch_decode(final_sample).squeeze(2)
        # image = denorm(image)[0]
        # Image.fromarray(image).save(sample_output_dir / "generated_image.png")

        del activations, intra_sim, inter_sim
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

#     # --- After processing all samples (or MAX_SAMPLES_FOR_AVERAGE) ---
#     if all_intra_sims and all_inter_sims:
#         print(f"\n--- Averaging metrics over {len(all_intra_sims)} samples ---")
#         avg_intra_sim_overall = np.mean(all_intra_sims, axis=0)
#         avg_inter_sim_overall = np.mean(all_inter_sims, axis=0)

#         # Plot the overall averaged metrics
#         overall_plot_path = output_dir / "overall_average_analysis.png"
#         plot_overall_average_metrics(avg_intra_sim_overall, avg_inter_sim_overall,
#                                       f'Overall Average Analysis ({len(all_intra_sims)} Samples)',
#                                       overall_plot_path, num_layers)
#     else:
#         print("\n--- No samples processed or no activations captured for averaging. ---")

#     print("\n--- T2I Analysis finished successfully! ---")

# if __name__ == '__main__':
#     main()

# --- After processing all samples (or MAX_SAMPLES_FOR_AVERAGE) ---
    if all_intra_sims and all_inter_sims:
        print(f"\n--- Averaging metrics over {len(all_intra_sims)} samples ---")
        avg_intra_sim_overall = np.mean(all_intra_sims, axis=0)
        avg_inter_sim_overall = np.mean(all_inter_sims, axis=0)

        # Save average metrics
        metrics_save_path = output_dir / "overall_average_metrics.npz"
        
        # Save both arrays with named keys for later loading
        np.savez(metrics_save_path, 
                 intra=avg_intra_sim_overall, 
                 inter=avg_inter_sim_overall)
        
        print(f"--- Saved average metrics to {metrics_save_path} ---")

        # Plot the overall averaged metrics
        overall_plot_path = output_dir / "overall_average_analysis.png"
        plot_overall_average_metrics(avg_intra_sim_overall, avg_inter_sim_overall,
                                     f'Overall Average Analysis ({len(all_intra_sims)} Samples)',
                                     overall_plot_path, num_layers)
    else:
        print("\n--- No samples processed or no activations captured for averaging. ---")

    print("\n--- T2I Analysis finished successfully! ---")

if __name__ == '__main__':
    main()