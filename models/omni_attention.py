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
import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def full(b, h, q_idx, kv_idx):
    return q_idx >= 0


def modality(offset, length):
    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx >= offset) & (kv_idx < (offset + length))

    return mask_fn


# code is borrowed from https://github.com/lucidrains/transfusion-pytorch
def omni_attn_mask(modalities):
    modalities = modalities.long()

    def mask_mod(b, h, q_idx, kv_idx):
        mask = causal(b, h, q_idx, kv_idx)

        modality_batch = modalities[b]

        for offset, length in modality_batch:
            mask = mask | modality(offset, length)(b, h, q_idx, kv_idx)

        return mask

    return mask_mod


def omni_attn_mask_naive(B, LEN, modalities, device, inverted=True):
    attention_mask = torch.tril(torch.ones((B, 1, LEN, LEN), dtype=torch.long)).to(device)
    for b in range(B):
        modality_batch = modalities[b]
        for offset, length in modality_batch:
            attention_mask[b, :, offset:offset + length, offset:offset + length] = 1

    if inverted:
        inverted_attention_mask = 1 - attention_mask
        inverted_attention_mask = inverted_attention_mask.masked_fill(
            inverted_attention_mask.to(torch.bool), torch.iinfo(torch.long).min
        )
        return inverted_attention_mask
    else:
        return attention_mask


def full_attn_mask_naive(B, LEN, device, inverted=True):
    attention_mask = torch.ones((B, 1, LEN, LEN), dtype=torch.long).to(device)
    if inverted:
        inverted_attention_mask = 1 - attention_mask
        inverted_attention_mask = inverted_attention_mask.masked_fill(
            inverted_attention_mask.to(torch.bool), torch.iinfo(torch.long).min
        )
        return inverted_attention_mask
    else:
        return attention_mask

def causal_attn_mask_naive(B, LEN, device, inverted=True):
    attention_mask = torch.tril(torch.ones((B, 1, LEN, LEN), dtype=torch.long)).to(device)
    if inverted:
        inverted_attention_mask = 1 - attention_mask
        inverted_attention_mask = inverted_attention_mask.masked_fill(
            inverted_attention_mask.to(torch.bool), torch.iinfo(torch.long).min
        )
        return inverted_attention_mask
    else:
        return attention_mask
