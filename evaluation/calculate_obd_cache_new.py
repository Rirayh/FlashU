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
from torch.nn.attention.flex_attention import create_block_mask

from models import Showo2Qwen2_5
from models import omni_attn_mask
from models.misc import get_text_tokenizer, prepare_gen_input
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
    def _get_ffn_error_scores(self, layer, layer_input_args, Y_baseline) -> torch.Tensor:
        """Compute L2 reconstruction error for FFN neurons, distributed across all GPUs."""
        intermediate_size = layer.mlp.gate_proj.out_features
        scores = torch.zeros(intermediate_size, device=self.device, dtype=torch.float32)
        
        state = self.state
        indices_to_process = range(intermediate_size)[state.process_index::state.num_processes]
        
        iterator = tqdm(indices_to_process, desc=f"  Calculating FFN Errors (size={intermediate_size})",
                        leave=False, mininterval=2, dynamic_ncols=True, disable=not state.is_main_process)
        
        for k in iterator:
            setattr(layer.mlp, '_temp_prune_ffn_k', k)
            Y_pruned_k = layer(*layer_input_args)[0]
            error = torch.norm(Y_baseline - Y_pruned_k, p=2)
            scores[k] = error.item()
            delattr(layer.mlp, '_temp_prune_ffn_k')
            
        # Synchronize: sum scores across all processes (only when num_processes > 1)
        if self.state.num_processes > 1:
            dist.all_reduce(scores, op=dist.ReduceOp.SUM)

        return scores

    @torch.no_grad()
    def _get_attn_error_scores(self, layer, layer_input_args, Y_baseline) -> torch.Tensor:
        """Compute L2 reconstruction error for attention KV groups, distributed across all GPUs."""
        num_kv_heads = layer.self_attn.num_key_value_heads
        scores = torch.zeros(num_kv_heads, device=self.device, dtype=torch.float32)

        if num_kv_heads == 0:
            logger.warning(f"Layer {layer.self_attn} has 0 KV heads. Skipping.")
            return scores

        state = self.state
        indices_to_process = range(num_kv_heads)[state.process_index::state.num_processes]

        iterator = tqdm(indices_to_process, desc=f"  Calculating Attn Errors (groups={num_kv_heads})",
                        leave=False, mininterval=2, dynamic_ncols=True, disable=not state.is_main_process)
        
        for k in iterator:
            setattr(layer.self_attn, '_temp_prune_kv_group_k', k)
            Y_pruned_k = layer(*layer_input_args)[0]
            error = torch.norm(Y_baseline - Y_pruned_k, p=2)
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
        
        input_ids = self.calibration_batch['input_ids'].to(self.device)
        attention_mask = self.calibration_batch['attention_mask'].to(self.device)
        
        cache_position = torch.arange(input_ids.shape[1], device=self.device)
        position_ids = cache_position.unsqueeze(0)
        hidden_states = self.llm.model.embed_tokens(input_ids)
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

            ffn_scores = self._get_ffn_error_scores(layer, layer_input_args, Y_baseline)
            if self.state.is_main_process:
                obd_scores_map[f"{layer_name_str}.mlp"] = ffn_scores.cpu().tolist()

            attn_scores = self._get_attn_error_scores(layer, layer_input_args, Y_baseline)
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

    # --- 1. Logging setup ---
    log_dir = config.get("log_dir", "logs_obd_cache")
    os.makedirs(log_dir, exist_ok=True)
    
    # Log filename includes sample index
    log_filename = os.path.join(log_dir, f"calculate_obd_scores_sample{start_sample_index:02d}_{time.strftime('%Y%m%d-%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    
    logger.info(f"Logging configured. Output will be saved to {log_filename}")
    logger.info(f"Process {state.process_index} of {state.num_processes} initialized on device {state.device}.")
    logger.info(f"Config-loaded args (first 5): {list(config.items())[:5]}")
    logger.info(f"Processing DPG sample index: {start_sample_index}")

    # --- 2. Check cache file ---
    # Cache filename is dynamic based on sample index
    cache_filename = os.path.join(log_dir, f"obd_scores_dpgsample{start_sample_index:02d}.json")
    
    if state.is_main_process and os.path.exists(cache_filename):
        logger.warning(f"Cache file {cache_filename} already exists. Skipping calculation for this sample.")
        sys.exit(0)

    # --- 3. Load config and model ---
    device = state.device
    weight_type = torch.bfloat16
    
    logger.info(f"Loading model on device: {device}")

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

    # --- 5. Prepare calibration batch ---
    validation_prompts_file = config.get("validation_prompts_file", config.dataset.params.validation_prompts_file)
    with open(validation_prompts_file) as fp:
        meta_data_full = json.load(fp)

    if not meta_data_full:
        logger.error(f"Failed to load prompts from {validation_prompts_file}. File is empty or invalid.")
        sys.exit(1)
        
    # Load hyper-parameters using the same logic as inference scripts
    if config.model.showo.add_time_embeds:
        if 'hyper_params' not in locals():
            config.dataset.preprocessing.num_t2i_image_tokens += 1
            hyper_params = list(get_hyper_params(config, text_tokenizer, showo_token_ids))
            hyper_params.extend([int(config.get("batch_size", 4)), device, weight_type])
            hyper_params = tuple(hyper_params)
    
    if 'hyper_params' not in locals():
        hyper_params = list(get_hyper_params(config, text_tokenizer, showo_token_ids))
        hyper_params.extend([int(config.get("batch_size", 4)), device, weight_type])
        hyper_params = tuple(hyper_params)

    (num_t2i, _, _, max_seq, max_text, latent_dim, patch_sz, lat_w, lat_h, 
     pad, bos, eos, boi, eoi, _, _, img_pad, _, guide_scale, batch_size, device, weight_type) = hyper_params
    
    batch_size = int(config.get("batch_size", 4))

    sample_batch_for_pruning = None
    try:
        logger.info(f"Preparing sample batch for OBD calibration using sample {start_sample_index}...")
        
        # Use the prompt at start_sample_index for calibration
        if start_sample_index >= len(meta_data_full):
            logger.error(f"Error: start_sample_index ({start_sample_index}) is out of bounds for dataset (size={len(meta_data_full)}).")
            if state.is_main_process:
                # Try to gracefully stop other processes
                dist.barrier()
            sys.exit(1)
            
        logger.info(f"Using prompt from meta_data_full[{start_sample_index}] for calibration.")
        prompts = [meta_data_full[start_sample_index]['prompt']] * batch_size
        
        batch_text_tokens, _, batch_modality_positions, _ = \
            prepare_gen_input(prompts, text_tokenizer, num_t2i, bos, eos, boi, eoi, pad, img_pad, max_text, device)

        omni_mask_fn = omni_attn_mask(batch_modality_positions)
        block_mask = create_block_mask(omni_mask_fn, B=batch_size, H=None, Q_LEN=max_seq, KV_LEN=max_seq, device=device)

        sample_batch_for_pruning = {
            "input_ids": batch_text_tokens,
            "attention_mask": block_mask
        }
        logger.info("Sample batch prepared successfully.")
    except Exception as e:
        logger.error(f"Failed to create sample batch for pruning: {e}.", exc_info=True)
        sys.exit(1)

    # --- 6. Run Pruner ---
    logger.info(f"Starting reconstruction error calculation for sample {start_sample_index}...")
    logger.warning("This process will be VERY SLOW and memory-intensive.")
    
    start_time = time.time()
    
    pruner = ReconstructionErrorPruner(model, sample_batch_for_pruning)
    pruner.state = state
    obd_scores_map = pruner.calculate_obd_scores()
    
    end_time = time.time()
    if state.is_main_process:
        logger.info(f"Calculation for sample {start_sample_index} finished in {(end_time - start_time) / 60:.2f} minutes.")

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