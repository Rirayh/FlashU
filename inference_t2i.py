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

"""Simple text-to-image inference demo for Show-o2."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from PIL import Image
import torch
from tqdm import tqdm
from accelerate.logging import get_logger
from models import Showo2Qwen2_5, omni_attn_mask_naive
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import (
    get_config, denorm, get_hyper_params,
    path_to_llm_name, load_state_dict,
)
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)

from transport import Sampler, create_transport

logger = get_logger(__name__, log_level="INFO")

if __name__ == "__main__":
    config = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_type = torch.bfloat16

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
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    if config.model.showo.load_from_showo:
        model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path, use_safetensors=False,
        ).to(device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(device)
        state_dict = load_state_dict(config.model_path)
        model.load_state_dict(state_dict)

    model.to(weight_type).eval()

    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1

    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()

    (
        num_t2i_image_tokens, _, _, max_seq_len, max_text_len,
        image_latent_dim, patch_size, latent_width, latent_height,
        pad_id, bos_id, eos_id, boi_id, eoi_id, _, _, img_pad_id, _,
        guidance_scale,
    ) = get_hyper_params(config, text_tokenizer, showo_token_ids)

    batch_size = config.batch_size
    guidance_scale = config.guidance_scale
    config.transport.num_inference_steps = config.num_inference_steps

    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
    )
    sampler = Sampler(transport)

    outdir = config.get("outdir", "output_t2i")
    os.makedirs(outdir, exist_ok=True)

    for step in tqdm(range(0, len(validation_prompts), batch_size)):
        prompts = validation_prompts[step:step + batch_size]

        (
            batch_text_tokens, batch_text_tokens_null,
            batch_modality_positions, batch_modality_positions_null,
        ) = prepare_gen_input(
            prompts, text_tokenizer, num_t2i_image_tokens,
            bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
            max_text_len, device,
        )

        z = torch.randn(
            (len(prompts), image_latent_dim,
             latent_height * patch_size, latent_width * patch_size),
            dtype=weight_type, device=device,
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

        block_mask = omni_attn_mask_naive(
            text_tokens.size(0), max_seq_len,
            modality_positions, device,
        ).to(weight_type)

        model_kwargs = dict(
            text_tokens=text_tokens,
            attention_mask=block_mask,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=max_seq_len,
            guidance_scale=guidance_scale,
        )

        sample_fn = sampler.sample_ode(
            sampling_method=config.transport.sampling_method,
            num_steps=config.transport.num_inference_steps,
            atol=config.transport.atol,
            rtol=config.transport.rtol,
        )
        samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]
        samples = torch.chunk(samples, 2)[0]

        samples = samples.unsqueeze(2)
        images = vae_model.batch_decode(samples).squeeze(2)

        images = denorm(images)
        pil_images = [Image.fromarray(image) for image in images]
        for i, img in enumerate(pil_images):
            img.save(os.path.join(outdir, f"{step + i}.png"))
