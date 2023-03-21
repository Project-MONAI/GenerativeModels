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

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Args:
        hidden_size: dimension of hidden layer.
        num_heads: number of attention heads.
        dropout_rate: dropout ratio. Defaults to no dropout.
        qkv_bias: bias term for the qkv linear layer.
        causal: whether to use causal attention.
        sequence_length: if causal is True, it is necessary to specify the sequence length.
        with_cross_attention: Whether to use cross attention for conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        causal: bool = False,
        sequence_length: int | None = None,
        with_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.causal = causal
        self.sequence_length = sequence_length
        self.with_cross_attention = with_cross_attention

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        if causal and sequence_length is None:
            raise ValueError("sequence_length is necessary for causal attention.")

        # key, query, value projections
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        # regularization
        self.drop_weights = nn.Dropout(dropout_rate)
        self.drop_output = nn.Dropout(dropout_rate)

        # output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        if causal and sequence_length is not None:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.mask = torch.tril(torch.ones(sequence_length, sequence_length)).view(
                1, 1, sequence_length, sequence_length
            )
        else:
            self.mask = None

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        b, t, c = x.size()  # batch size, sequence length, embedding dimensionality (hidden_size)

        if self.sequence_length is not None and t != self.sequence_length:
            raise ValueError("sequence length should be equal to the one specified in the SABlock constructor.")

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query = self.to_q(x)

        kv = context if context is not None else x
        _, kv_t, _ = kv.size()
        key = self.to_k(context)
        value = self.to_v(context)

        query = query.view(b, t, self.num_heads, c // self.num_heads).transpose(1, 2)  # (b, nh, t, hs)
        key = key.view(b, kv_t, self.num_heads, c // self.num_heads).transpose(1, 2)  # (b, nh, kv_t, hs)
        value = value.view(b, kv_t, self.num_heads, c // self.num_heads).transpose(1, 2)  # (b, nh, kv_t, hs)

        # manual implementation of attention
        attention_scores = (query @ key.transpose(-2, -1)) * self.scale

        if self.causal:
            attention_scores = attention_scores.masked_fill(self.mask[:, :, :t, :kv_t] == 0, float("-inf"))

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.drop_weights(attention_probs)
        y = attention_probs @ value  # (b, nh, t, kv_t) x (b, nh, kv_t, hs) -> (b, nh, t, hs)
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # re-assemble all head outputs side by side

        y = self.out_proj(y)
        y = self.drop_output(y)
        return y
