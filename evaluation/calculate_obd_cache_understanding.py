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

import os
import sys
import pathlib
import time
import json
import numpy as np
import argparse
import logging
from PIL import Image
from torchvision import transforms
project_root = str(pathlib.Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm.auto import tqdm
from typing import Dict, List
from accelerate.logging import get_logger
from accelerate import PartialState

from models import Showo2Qwen2_5, WanVAE
from models import omni_attn_mask_naive
from models.misc import get_text_tokenizer, interpolate_pos_encoding
from utils import get_config, path_to_llm_name, load_state_dict, get_hyper_params

import types

logger = get_logger(__name__, log_level="INFO")


def _obd_mlp_forward(self, hidden_state):
    gate_out = self.act_fn(self.gate_proj(hidden_state))
    up_out = self.up_proj(hidden_state)
    intermediate = gate_out * up_out
    prune_k = getattr(self, '_temp_prune_ffn_k', None)
    if prune_k is not None:
        intermediate[:, :, prune_k] = 0.0
    return self.down_proj(intermediate)


def _obd_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    position_embeddings=None,
):
    from models.qwen2 import apply_rotary_pos_emb, repeat_kv
    from torch.nn.attention.flex_attention import flex_attention, BlockMask

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

    prune_k = getattr(self, '_temp_prune_kv_group_k', None)
    if prune_k is not None:
        key_states[:, prune_k, :, :] = 0.0
        value_states[:, prune_k, :, :] = 0.0

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    query_states = query_states.contiguous().to(value_states.dtype)
    key_states = key_states.contiguous().to(value_states.dtype)
    value_states = value_states.contiguous()

    if type(attention_mask) == BlockMask:
        attn_output = flex_attention(query_states, key_states, value_states, block_mask=attention_mask)
    else:
        causal_mask = attention_mask
        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def apply_obd_measurement_patch(model):
    patched_mlp = 0
    patched_attn = 0
    for layer in model.showo.model.layers:
        layer.mlp.forward = types.MethodType(_obd_mlp_forward, layer.mlp)
        patched_mlp += 1
        layer.self_attn.forward = types.MethodType(_obd_attn_forward, layer.self_attn)
        patched_attn += 1
    logger.info(f"Applied OBD measurement patch to {patched_mlp} MLP and {patched_attn} Attention layers.")


class ReconstructionErrorPruner:
    """
    Helper class for reconstruction-error-based pruning (OBD/SparseGPT).
    Supports distributed (multi-GPU) computation via accelerate PartialState.
    """
    def __init__(self, model: Showo2Qwen2_5, calibration_batch: Dict[str, torch.Tensor]):
        self.model = model
        self.llm = model.showo  # Qwen2ForCausalLM instance
        self.calibration_batch = calibration_batch
        self.device = next(model.parameters()).device
        self.state: PartialState = None  # Injected in main()
        logger.info(f"ReconstructionErrorPruner initialized on device {self.device}.")

    @torch.no_grad()
    def _get_ffn_error_scores(
        self, layer, layer_input_args, Y_baseline, valid_token_mask
    ) -> torch.Tensor:
        """Compute squared L2 reconstruction error for FFN neurons, distributed across all GPUs."""
        intermediate_size = layer.mlp.gate_proj.out_features
        scores = torch.zeros(intermediate_size, device=self.device, dtype=torch.float32)
        
        state = self.state
        indices_to_process = range(intermediate_size)[state.process_index::state.num_processes]
        
        iterator = tqdm(indices_to_process, desc=f"  Calculating FFN Errors (size={intermediate_size})",
                        leave=False, miniters=1000, maxinterval=float("inf"), dynamic_ncols=True, disable=not state.is_main_process)
        
        for k in iterator:
            setattr(layer.mlp, '_temp_prune_ffn_k', k)
            Y_pruned_k = layer(*layer_input_args)[0]
            diff = Y_baseline.float() - Y_pruned_k.float()
            # Keep one error per sequence position so its padding can be masked.
            per_position_error = diff.square().sum(dim=-1)
            # Exclude padded positions from the neuron importance score.
            per_prompt_error = per_position_error.masked_fill(~valid_token_mask, 0.0).sum(dim=1)
            error = per_prompt_error.mean()
            scores[k] = error.item()
            delattr(layer.mlp, '_temp_prune_ffn_k')
            
        # Synchronize: sum scores across all processes (only when num_processes > 1)
        if self.state.num_processes > 1:
            dist.all_reduce(scores, op=dist.ReduceOp.SUM)

        return scores

    @torch.no_grad()
    def _get_attn_error_scores(
        self, layer, layer_input_args, Y_baseline, valid_token_mask
    ) -> torch.Tensor:
        """Compute squared L2 reconstruction error for attention KV groups, distributed across all GPUs."""
        num_kv_heads = layer.self_attn.num_key_value_heads
        scores = torch.zeros(num_kv_heads, device=self.device, dtype=torch.float32)

        if num_kv_heads == 0:
            logger.warning(f"Layer {layer.self_attn} has 0 KV heads. Skipping.")
            return scores

        state = self.state
        indices_to_process = range(num_kv_heads)[state.process_index::state.num_processes]

        iterator = tqdm(indices_to_process, desc=f"  Calculating Attn Errors (groups={num_kv_heads})",
                        leave=False, miniters=1, dynamic_ncols=True, disable=not state.is_main_process)
        
        for k in iterator:
            setattr(layer.self_attn, '_temp_prune_kv_group_k', k)
            Y_pruned_k = layer(*layer_input_args)[0]
            diff = Y_baseline.float() - Y_pruned_k.float()
            # Keep one error per sequence position so its padding can be masked.
            per_position_error = diff.square().sum(dim=-1)
            # Exclude padded positions from the attention importance score.
            per_prompt_error = per_position_error.masked_fill(~valid_token_mask, 0.0).sum(dim=1)
            error = per_prompt_error.mean()
            scores[k] = error.item()
            delattr(layer.self_attn, '_temp_prune_kv_group_k')
            
        # Synchronize: sum scores across all processes (only when num_processes > 1)
        if self.state.num_processes > 1:
            dist.all_reduce(scores, op=dist.ReduceOp.SUM)
            
        return scores

    @torch.no_grad()
    def calculate_obd_scores(self) -> Dict[str, List[float]]:
        """
        Run full OBD computation across all layers and return a dict of raw error scores.
        Computation is synchronized across all GPUs; only the main process saves results.
        """
        obd_scores_map = {}
        
        input_embeds = self.calibration_batch['input_embeds'].to(self.device)
        attention_mask = self.calibration_batch['attention_mask'].to(self.device)
        valid_token_mask = self.calibration_batch['valid_token_mask'].to(self.device)

        cache_position = torch.arange(input_embeds.shape[1], device=self.device)
        position_ids = cache_position.unsqueeze(0)
        hidden_states = input_embeds
        position_embeddings = self.llm.model.rotary_emb(hidden_states, position_ids)
        
        logger.info(f"Starting layer-by-layer reconstruction error calculation on {self.state.num_processes} GPUs...")

        layer_iterator = self.llm.model.layers
        if self.state.is_main_process:
            layer_iterator = tqdm(layer_iterator, desc="Processing Layers", leave=True, dynamic_ncols=True)
            
        for layer_idx, layer in enumerate(layer_iterator):
            logger.info(f"[OBD] Starting layer {layer_idx}/27")
            layer_name_str = f"model.layers.{layer_idx}"
            
            layer_input_args = (
                hidden_states, attention_mask, position_ids, None, False, False,
                cache_position, position_embeddings
            )
            
            Y_baseline = layer(*layer_input_args)[0]

            ffn_scores = self._get_ffn_error_scores(
                layer, layer_input_args, Y_baseline, valid_token_mask
            )
            if self.state.is_main_process:
                obd_scores_map[f"{layer_name_str}.mlp"] = ffn_scores.cpu().tolist()

            attn_scores = self._get_attn_error_scores(
                layer, layer_input_args, Y_baseline, valid_token_mask
            )
            if self.state.is_main_process:
                obd_scores_map[f"{layer_name_str}.self_attn"] = attn_scores.cpu().tolist()
            
            hidden_states = Y_baseline.detach()
            logger.info(f"[OBD] Finished layer {layer_idx}/27")

        logger.info("Reconstruction error calculation complete.")
        return obd_scores_map


def main():
    config = get_config()
    state = PartialState()
    
    # Get sample index to process (passed via CLI, e.g. start_sample_index=5)
    start_sample_index = int(config.get("start_sample_index", 0))
    num_prompts = int(config.get("num_prompts", 1))
    batch_size = int(config.get("batch_size", num_prompts))
    understanding_data_file = config.get(
        "understanding_data_file",
        "prompts/understanding_calibration.json",
    )

    # --- 1. Logging setup ---
    log_dir = config.get("log_dir", "logs_obd_cache")
    os.makedirs(log_dir, exist_ok=True)
    
    # Log filename includes sample index
    log_filename = os.path.join(log_dir, f"calculate_obd_understanding_sample{start_sample_index:02d}_{time.strftime('%Y%m%d-%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    
    logger.info(f"Logging configured. Output will be saved to {log_filename}")
    logger.info(f"Process {state.process_index} of {state.num_processes} initialized on device {state.device}.")
    logger.info(f"Config-loaded args (first 5): {list(config.items())[:5]}")
    logger.info(
        f"Processing {num_prompts} understanding record(s) starting at index "
        f"{start_sample_index} from {understanding_data_file}."
    )

    # --- 2. Check cache file ---
    # Cache filename is dynamic based on sample index
    cache_filename = os.path.join(log_dir, f"obd_scores_understanding_sample{start_sample_index:02d}.json")
    
    if state.is_main_process and os.path.exists(cache_filename):
        logger.warning(f"Cache file {cache_filename} already exists. Skipping calculation for this sample.")
        sys.exit(0)

    # --- 3. Load config and model ---
    device = state.device
    weight_type = torch.bfloat16
    
    logger.info(f"Loading model on device: {device}")

    vae_model = WanVAE(
        vae_pth=config.model.vae_model.pretrained_model_path,
        dtype=weight_type,
        device=device,
    )
    logger.info("WanVAE loaded successfully.")

    text_tokenizer, showo_token_ids = get_text_tokenizer(config.model.showo.llm_model_path, add_showo_tokens=True, return_showo_token_ids=True, llm_name=path_to_llm_name[config.model.showo.llm_model_path])
    
    if config.model.showo.get("load_from_showo", False):
        model = Showo2Qwen2_5.from_pretrained(config.model.showo.pretrained_model_path, use_safetensors=False).to(device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(device)
        state_dict = load_state_dict(config.model_path)
        model.load_state_dict(state_dict)
    
    model.to(weight_type)
    model.eval()
    logger.info("Model loaded successfully.")

    # --- 4. Apply required patches ---
    apply_obd_measurement_patch(model)
    logger.info("Applied OBD measurement patch (MLP + Attention).")

    # --- 5. Load understanding calibration records ---
    with open(understanding_data_file, encoding="utf-8") as fp:
        meta_data_full = json.load(fp)

    if not isinstance(meta_data_full, list) or not meta_data_full:
        raise ValueError(
            f"Understanding data file {understanding_data_file} must contain "
            f"a non-empty JSON list"
        )

    if num_prompts <= 0:
        raise ValueError("num_prompts must be greater than 0")
    if start_sample_index < 0:
        raise ValueError("start_sample_index must be greater than or equal to 0")

    end_sample_index = start_sample_index + num_prompts
    if end_sample_index > len(meta_data_full):
        raise ValueError(
            f"Requested records [{start_sample_index}:{end_sample_index}], "
            f"but the dataset contains only {len(meta_data_full)} records"
        )

    selected_records = meta_data_full[start_sample_index:end_sample_index]
    for record_index, record in enumerate(
        selected_records,
        start=start_sample_index,
    ):
        if not isinstance(record, dict):
            raise ValueError(
                f"Understanding record {record_index} must be a JSON object"
            )

        image_path = record.get("image_path")
        prompt = record.get("prompt")
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError(
                f"Understanding record {record_index} has no valid image_path"
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Understanding record {record_index} has no valid prompt"
            )
        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"Image for understanding record {record_index} was not found: "
                f"{image_path}"
            )
        
    # Load hyper-parameters using the understanding inference configuration
    if config.model.showo.add_time_embeds:
        if 'hyper_params' not in locals():
            config.dataset.preprocessing.num_mmu_image_tokens += 1
            hyper_params = list(get_hyper_params(config, text_tokenizer, showo_token_ids))
            hyper_params.extend([batch_size, device, weight_type])
            hyper_params = tuple(hyper_params)
    
    if 'hyper_params' not in locals():
        hyper_params = list(get_hyper_params(config, text_tokenizer, showo_token_ids))
        hyper_params.extend([batch_size, device, weight_type])
        hyper_params = tuple(hyper_params)

    (_, num_mmu, _, max_seq, max_text, latent_dim, patch_sz, lat_w, lat_h,
     pad, bos, eos, boi, eoi, _, _, img_pad, _, guide_scale, batch_size, device, weight_type) = hyper_params
    
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    # --- Define the official Show-o2 preprocessing for understanding images. ---
    understanding_image_transform = transforms.Compose([
        transforms.Resize(
            config.dataset.preprocessing.resolution,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop((
            config.dataset.preprocessing.resolution,
            config.dataset.preprocessing.resolution,
        )),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    # --- Prepare the chat template and question-token budget. ---
    # Tokenize the three fixed chat-template segments once.
    system_prompt_ids = text_tokenizer(
        "system\nYou are a helpful assistant.<|im_end|>",
        add_special_tokens=False,
    )["input_ids"]
    user_role_ids = text_tokenizer(
        "\n<|im_start|>user\n",
        add_special_tokens=False,
    )["input_ids"]
    assistant_role_ids = text_tokenizer(
        "\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )["input_ids"]
    understanding_prefix_ids = [bos] + system_prompt_ids + user_role_ids

    # Calculate how many question tokens fit after all fixed positions.
    max_question_tokens = (
        max_seq
        - len(understanding_prefix_ids)
        - num_mmu
        - 2
        - len(assistant_role_ids)
    )
    if max_question_tokens <= 0:
        raise ValueError(
            "The configured sequence length cannot fit the Show-o2 "
            "understanding input structure"
        )

    # --- Prepare and score understanding calibration batches. ---
    logger.info(
        f"Starting reconstruction error calculation for "
        f"{num_prompts} understanding record(s)..."
    )
    logger.warning("This process will be VERY SLOW and memory-intensive.")

    start_time = time.time()
    score_sums = {}
    num_scored_records = 0

    for batch_start in range(0, num_prompts, batch_size):
        records = selected_records[batch_start:batch_start + batch_size]
        current_batch_size = len(records)
        first_record_index = start_sample_index + batch_start
        last_record_index = first_record_index + current_batch_size - 1

        try:
            logger.info(
                f"Preparing calibration batch for understanding record indices "
                f"{first_record_index}-{last_record_index}."
            )

            # --- Preprocess the images and encode them as WanVAE latents. ---
            # Load every image in the batch and apply the shared transform.
            batch_images = []
            for record in records:
                with Image.open(record["image_path"]) as image:
                    image = image.convert("RGB")
                    batch_images.append(understanding_image_transform(image))

            # Stack the pixels, add a one-frame axis, and compress with the VAE.
            batch_images = torch.stack(batch_images, dim=0).to(device)
            image_latents = vae_model.sample(
                batch_images.unsqueeze(2)
            ).squeeze(2).to(weight_type)
            logger.info(
                f"Encoded understanding image batch to latents with shape "
                f"{tuple(image_latents.shape)}."
            )

            # --- Tokenize each question using the Show-o2 understanding format. ---
            # Repeat the understanding prefix for every record in the batch.
            batch_text_tokens_a = torch.tensor(
                [understanding_prefix_ids] * current_batch_size,
                dtype=torch.long,
                device=device,
            )
            # Build the BOI/EOI markers and question suffix for every record.
            batch_text_tokens_b = []
            for record_offset, record in enumerate(records):
                question_ids = text_tokenizer(
                    record["prompt"],
                    add_special_tokens=False,
                )["input_ids"]
                if len(question_ids) > max_question_tokens:
                    logger.warning(
                        f"Truncating understanding prompt at record "
                        f"{first_record_index + record_offset} from "
                        f"{len(question_ids)} to {max_question_tokens} tokens."
                    )
                    question_ids = question_ids[:max_question_tokens]

                batch_text_tokens_b.append(torch.tensor(
                    [boi, eoi] + question_ids + assistant_role_ids,
                    dtype=torch.long,
                    device=device,
                ))

            # --- Build fused multimodal embeddings from the text and images. ---
            with torch.no_grad():
                # Convert both text-token sections into model embeddings.
                text_embeds_a = model.showo.model.embed_tokens(
                    batch_text_tokens_a
                )
                text_embeds_b_per_record = [
                    model.showo.model.embed_tokens(
                        text_tokens_b.unsqueeze(0)
                    )
                    for text_tokens_b in batch_text_tokens_b
                ]
                embedding_dtype = text_embeds_a.dtype

                # Encode the VAE latents through both Show-o2 visual branches.
                image_embeds_und = model.image_embedder_und(
                    image_latents.to(embedding_dtype)
                )
                image_embeds_gen = model.image_embedder_gen(
                    image_latents.to(embedding_dtype)
                )

                # Add learned positions to the understanding image branch.
                _, _, latent_height, latent_width = image_latents.shape
                patch_size = model.config.patch_size
                embedding_height = latent_height // patch_size
                embedding_width = latent_width // patch_size
                if (
                    model.position_embedding.weight.shape[0]
                    == model.image_position_ids.shape[-1]
                ):
                    image_embeds_und = image_embeds_und + (
                        model.position_embedding(model.image_position_ids)
                    )
                else:
                    image_embeds_und = image_embeds_und + (
                        interpolate_pos_encoding(
                            model.config.clip_latent_dim,
                            model.position_embedding,
                            embedding_height,
                            embedding_width,
                            1,
                        )
                    )

                # Run the understanding transformer, then fuse both image branches.
                image_embeds_und = model.und_trans(
                    image_embeds_und
                )["last_hidden_state"]
                image_embeds = model.fusion_proj(torch.cat(
                    [image_embeds_und, image_embeds_gen],
                    dim=-1,
                ))

                # Check the image-embedding count and create the optional time embedding.
                add_time_embeds = bool(config.model.showo.add_time_embeds)
                expected_image_tokens = num_mmu - int(add_time_embeds)
                if image_embeds.shape[1] != expected_image_tokens:
                    raise ValueError(
                        f"Expected {expected_image_tokens} understanding image "
                        f"embeddings, but produced {image_embeds.shape[1]}"
                    )

                if add_time_embeds:
                    time_values = torch.ones(
                        current_batch_size,
                        device=device,
                    )
                    time_embeds = model.time_embed(
                        time_values,
                        embedding_dtype,
                    )
                    if hasattr(model, "time_embed_proj"):
                        time_embeds = model.time_embed_proj(time_embeds)
                    time_embeds = time_embeds.unsqueeze(1)

                # Place the optional time embedding and image embeddings between BOI and EOI.
                input_embeds_per_record = []
                for record_offset, text_embeds_b in enumerate(
                    text_embeds_b_per_record
                ):
                    embed_parts = [
                        text_embeds_a[record_offset:record_offset + 1],
                        text_embeds_b[:, :1],
                    ]
                    if add_time_embeds:
                        embed_parts.append(
                            time_embeds[record_offset:record_offset + 1]
                        )
                    embed_parts.extend([
                        image_embeds[record_offset:record_offset + 1],
                        text_embeds_b[:, 1:],
                    ])
                    input_embeds_per_record.append(
                        torch.cat(embed_parts, dim=1).squeeze(0)
                    )

                # --- Pad the multimodal sequences and build understanding masks. ---
                # Find the longest sequence and prefill the batch with padding.
                sequence_lengths = torch.tensor(
                    [embeds.shape[0] for embeds in input_embeds_per_record],
                    dtype=torch.long,
                    device=device,
                )
                padded_sequence_length = int(sequence_lengths.max().item())
                pad_embedding = model.showo.model.embed_tokens(
                    torch.tensor([pad], dtype=torch.long, device=device)
                ).squeeze(0)
                batch_input_embeds = pad_embedding.view(1, 1, -1).expand(
                    current_batch_size,
                    padded_sequence_length,
                    -1,
                ).clone()
                # Copy each real sequence over the padding and mark its positions.
                valid_token_mask = torch.zeros(
                    current_batch_size,
                    padded_sequence_length,
                    dtype=torch.bool,
                    device=device,
                )
                for record_offset, record_embeds in enumerate(
                    input_embeds_per_record
                ):
                    record_length = record_embeds.shape[0]
                    batch_input_embeds[record_offset, :record_length] = record_embeds
                    valid_token_mask[record_offset, :record_length] = True

                # Store [start position, MMU length] for the attention mask.
                modality_offset = len(understanding_prefix_ids) + (2 if add_time_embeds else 1)
                batch_modality_positions = torch.tensor(
                    [modality_offset, num_mmu],
                    dtype=torch.long,
                    device=device,
                ).view(1, 1, 2).expand(
                    current_batch_size, -1, -1
                ).clone()
                # Build the attention mask and prevent attention to padding.
                attention_mask = omni_attn_mask_naive(
                    B=current_batch_size,
                    LEN=padded_sequence_length,
                    modalities=batch_modality_positions,
                    device=device,
                    inverted=True,
                ).to(embedding_dtype)
                attention_mask = attention_mask.masked_fill(
                    ~valid_token_mask[:, None, None, :],
                    torch.finfo(embedding_dtype).min,
                )

                # --- Package the prepared understanding inputs for OBD scoring. ---
                calibration_batch = {
                    "input_embeds": batch_input_embeds,
                    "attention_mask": attention_mask,
                    "valid_token_mask": valid_token_mask,
                }
        except Exception as e:
            logger.error(f"Failed to create calibration batch: {e}.", exc_info=True)
            sys.exit(1)

        pruner = ReconstructionErrorPruner(model, calibration_batch)
        pruner.state = state
        batch_scores_map = pruner.calculate_obd_scores()

        if state.is_main_process:
            for score_name, batch_scores in batch_scores_map.items():
                batch_scores = np.asarray(batch_scores, dtype=np.float64)
                if score_name not in score_sums:
                    score_sums[score_name] = np.zeros_like(batch_scores)
                score_sums[score_name] += batch_scores * current_batch_size
            num_scored_records += current_batch_size

    if state.is_main_process:
        obd_scores_map = {
            score_name: (score_sum / num_scored_records).tolist()
            for score_name, score_sum in score_sums.items()
        }
    else:
        obd_scores_map = {}
    
    end_time = time.time()
    if state.is_main_process:
        logger.info(
            f"Calculation for {num_scored_records} understanding record(s) "
            f"finished in "
            f"{(end_time - start_time) / 60:.2f} minutes."
        )

    # --- 7. Save to cache ---
    if state.is_main_process:
        try:
            with open(cache_filename, 'w') as f:
                json.dump(obd_scores_map, f, indent=2)
            logger.info(f"Successfully saved full OBD scores map to {cache_filename}")
        except Exception as e:
            logger.error(f"Failed to save scores map to {cache_filename}: {e}", exc_info=True)
            dist.destroy_process_group()
            sys.exit(1)

    logger.info(f"Process {state.process_index} finished.")


if __name__ == '__main__':
    main()
