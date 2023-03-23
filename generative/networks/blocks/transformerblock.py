# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.blocks.mlp import MLPBlock

from generative.networks.blocks.selfattention import SABlock


class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Args:
        hidden_size: dimension of hidden layer.
        mlp_dim: dimension of feedforward layer.
        num_heads: number of attention heads.
        dropout_rate: faction of the input units to drop.
        qkv_bias: apply bias term for the qkv linear layer
        causal: whether to use causal attention.
        sequence_length: if causal is True, it is necessary to specify the sequence length.
        with_cross_attention: Whether to use cross attention for conditioning.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        causal: bool = False,
        sequence_length: int | None = None,
        with_cross_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        self.with_cross_attention = with_cross_attention
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias,
            causal=causal,
            sequence_length=sequence_length,
            use_flash_attention=use_flash_attention,
        )

        self.norm2 = None
        self.cross_attn = None
        if self.with_cross_attention:
            self.norm2 = nn.LayerNorm(hidden_size)
            self.cross_attn = SABlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                qkv_bias=qkv_bias,
                with_cross_attention=with_cross_attention,
                causal=False,
                use_flash_attention=use_flash_attention,
            )

        self.norm3 = nn.LayerNorm(hidden_size)
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        if self.with_cross_attention:
            x = x + self.cross_attn(self.norm2(x), context=context)
        x = x + self.mlp(self.norm3(x))
        return x
