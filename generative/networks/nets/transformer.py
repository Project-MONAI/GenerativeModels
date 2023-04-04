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

from generative.networks.blocks.transformerblock import TransformerBlock

__all__ = ["DecoderOnlyTransformer"]


class AbsolutePositionalEmbedding(nn.Module):
    """Absolute positional embedding.

    Args:
        max_seq_len: Maximum sequence length.
        embedding_dim: Dimensionality of the embedding.
    """

    def __init__(self, max_seq_len: int, embedding_dim: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).repeat(batch_size, 1)
        return self.embedding(positions)


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only (Autoregressive) Transformer model.

    Args:
        num_tokens: Number of tokens in the vocabulary.
        max_seq_len: Maximum sequence length.
        attn_layers_dim: Dimensionality of the attention layers.
        attn_layers_depth: Number of attention layers.
        attn_layers_heads: Number of attention heads.
        with_cross_attention: Whether to use cross attention for conditioning.
        embedding_dropout_rate: Dropout rate for the embedding.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        num_tokens: int,
        max_seq_len: int,
        attn_layers_dim: int,
        attn_layers_depth: int,
        attn_layers_heads: int,
        with_cross_attention: bool = False,
        embedding_dropout_rate: float = 0.0,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.attn_layers_dim = attn_layers_dim
        self.attn_layers_depth = attn_layers_depth
        self.attn_layers_heads = attn_layers_heads
        self.with_cross_attention = with_cross_attention

        self.token_embeddings = nn.Embedding(num_tokens, attn_layers_dim)
        self.position_embeddings = AbsolutePositionalEmbedding(max_seq_len=max_seq_len, embedding_dim=attn_layers_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout_rate)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=attn_layers_dim,
                    mlp_dim=attn_layers_dim * 4,
                    num_heads=attn_layers_heads,
                    dropout_rate=0.0,
                    qkv_bias=False,
                    causal=True,
                    sequence_length=max_seq_len,
                    with_cross_attention=with_cross_attention,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(attn_layers_depth)
            ]
        )

        self.to_logits = nn.Linear(attn_layers_dim, num_tokens)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        tok_emb = self.token_embeddings(x)
        pos_emb = self.position_embeddings(x)
        x = self.embedding_dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x, context=context)

        return self.to_logits(x)
