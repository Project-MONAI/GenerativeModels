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

import importlib.util

import torch
import torch.nn as nn

if importlib.util.find_spec("x_transformers") is not None:
    from x_transformers import Decoder, TransformerWrapper

    has_x_transformers = True
else:
    has_x_transformers = False


__all__ = ["DecoderOnlyTransformer"]


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only (Autoregressive) Transformer model.

    Args:
        num_tokens: Number of tokens in the vocabulary.
        max_seq_len: Maximum sequence length.
        attn_layers_dim: Dimensionality of the attention layers.
        attn_layers_depth: Number of attention layers.
        attn_layers_heads: Number of attention heads.
        with_cross_attention: Whether to use cross attention for conditioning.
    """

    def __init__(
        self,
        num_tokens: int,
        max_seq_len: int,
        attn_layers_dim: int,
        attn_layers_depth: int,
        attn_layers_heads: int,
        with_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.attn_layers_dim = attn_layers_dim
        self.attn_layers_depth = attn_layers_depth
        self.attn_layers_heads = attn_layers_heads

        if has_x_transformers:
            self.model = TransformerWrapper(
                num_tokens=self.num_tokens,
                max_seq_len=self.max_seq_len,
                attn_layers=Decoder(
                    dim=self.attn_layers_dim,
                    depth=self.attn_layers_depth,
                    heads=self.attn_layers_heads,
                    cross_attend=with_cross_attention,
                ),
            )
        else:
            raise ImportError("x-transformers is not installed.")

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(x, context=context)
