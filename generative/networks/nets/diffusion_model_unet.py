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

# =========================================================================
# Adapted from https://github.com/huggingface/diffusers
# which has the following license:
# https://github.com/huggingface/diffusers/blob/main/LICENSE

# Copyright 2022 UC Berkeley Team and The HuggingFace Team. All rights reserved.
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
# =========================================================================

import math
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Pool
from torch import einsum, nn

__all__ = ["DiffusionModelUNet"]


class GEGLU(nn.Module):
    """
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Args:
        dim_in: number of channels in the input.
        dim_out: number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """
    A feed-forward layer.

    Args:
        num_channels: number of channels in the input.
        dim_out: number of channels in the output. If not given, defaults to `dim`.
        mult: multiplier to use for the hidden dimension.
        glu: whether to use GLU activation.
        dropout: dropout probability to use.
    """

    def __init__(
        self, num_channels: int, dim_out: Optional[int] = None, mult: int = 4, glu: bool = False, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim = int(num_channels * mult)
        dim_out = dim_out if dim_out is not None else num_channels
        project_in = (
            nn.Sequential(nn.Linear(num_channels, inner_dim), nn.GELU()) if not glu else GEGLU(num_channels, inner_dim)
        )

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    """
    A cross attention layer.

    Args:
        query_dim: number of channels in the query.
        cross_attention_dim: number of channels in the context.
        num_attention_heads: number of heads to use for multi-head attention.
        attention_head_dim: number of channels in each head.
        dropout: dropout probability to use.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = attention_head_dim * num_attention_heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = attention_head_dim**-0.5
        self.heads = num_attention_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        head_size = self.heads
        x = x.reshape(batch_size, seq_len, head_size, dim // head_size)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        head_size = self.heads
        x = x.reshape(batch_size // head_size, head_size, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return x

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        query = self.to_q(x)
        context = context if context is not None else x
        key = self.to_k(context)
        value = self.to_v(context)

        # TODO: Maybe make use of xformers to improve attention speed
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        attention_scores = einsum("b i d, b j d -> b i j", query, key) * self.scale

        attention_probs = attention_scores.softmax(dim=-1)

        hidden_states = einsum("b i j, b j d -> b i d", attention_probs, value)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return self.to_out(hidden_states)


class BasicTransformerBlock(nn.Module):
    """
    A basic Transformer block.

    Args:
        num_channels: number of channels in the input and output.
        num_attention_heads: number of heads to use for multi-head attention.
        attention_head_dim: number of channels in each head.
        dropout: dropout probability to use.
        cross_attention_dim: size of the context vector for cross attention.
        gated_ff: whether to use a gated feed-forward network.
    """

    def __init__(
        self,
        num_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        gated_ff: bool = True,
    ) -> None:
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=num_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
        )  # is a self-attention
        self.ff = FeedForward(num_channels, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=num_channels,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
        )  # is a self-attention if context is None
        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)
        self.norm3 = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data. First, project the input (aka embedding) and reshape to b, t, d. Then apply
    standard transformer action. Finally, reshape to image.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of channels in the input and output.
        num_attention_heads: number of heads to use for multi-head attention.
        attention_head_dim: number of channels in each head.
        num_layers: number of layers of Transformer blocks to use.
        dropout: dropout probability to use.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        cross_attention_dim: number of context dimensions to use.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        cross_attention_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        inner_dim = num_attention_heads * attention_head_dim
        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)

        self.proj_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=inner_dim,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    num_channels=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj_out = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=inner_dim,
                out_channels=in_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # note: if no context is given, cross-attention defaults to self-attention
        if self.spatial_dims == 2:
            b, c, h, w = x.shape
        if self.spatial_dims == 3:
            b, c, h, w, d = x.shape

        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)

        if self.spatial_dims == 2:
            x = rearrange(x, "b c h w -> b (h w) c")
        if self.spatial_dims == 3:
            x = rearrange(x, "b c h w d -> b (h w d) c")

        for block in self.transformer_blocks:
            x = block(x, context=context)

        if self.spatial_dims == 2:
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if self.spatial_dims == 3:
            x = rearrange(x, "b (h w d) c -> b c h w d", h=h, w=w, d=d)

        x = self.proj_out(x)
        return x + x_in


class QKVAttentionLegacy(nn.Module):
    """
    A qkv attention mechanism.

    Args:
        num_attention_heads: number of attention heads.
    """

    def __init__(self, num_attention_heads: int) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        bs, width, length = qkv.shape
        assert width % (3 * self.num_attention_heads) == 0
        ch = width // (3 * self.num_attention_heads)
        q, k, v = qkv.reshape(bs * self.num_attention_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """
    An attention block.

    Args:
        num_channels: number of channels in the input and output.
        num_attention_heads:  number of attention heads.
        attention_head_dim: number of channels in each head.
        norm_num_groups: number of groups to use for group norm.
        norm_eps: epsilon value to use for group norm.
    """

    def __init__(
        self,
        num_channels: int,
        num_attention_heads: int = 1,
        attention_head_dim: int = -1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channels = num_channels
        if attention_head_dim == -1:
            self.num_heads = num_attention_heads
        else:
            assert (
                num_channels % attention_head_dim == 0
            ), f"q,k,v channels {num_channels} is not divisible by attention_head_dim {attention_head_dim}"
            self.num_heads = num_channels // attention_head_dim
        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=norm_eps, affine=True)
        self.qkv = nn.Conv1d(num_channels, num_channels * 3, 1)
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(num_channels, num_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        embedding_dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    freqs = torch.exp(exponent / half_dim)

    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

    return embedding


class Downsample(nn.Module):
    """
    Downsampling layer.

    Args:
        spatial_dims: number of spatial dimensions.
        num_channels: number of input channels.
        use_conv: if True uses Convolution instead of Pool average to perform downsampling.
        out_channels: number of output channels.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        use_conv: bool,
        out_channels: Optional[int] = None,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if use_conv:
            self.op = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=2,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            assert self.num_channels == self.out_channels
            self.op = Pool[Pool.AVG, spatial_dims](kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.num_channels
        return self.op(x)


class Upsample(nn.Module):
    """
    Upsampling layer.

    Args:
        spatial_dims: number of spatial dimensions.
        num_channels: number of input channels
        use_conv: if True uses Convolution instead of Pool average to perform downsampling.
        out_channels: number of output channels.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points for each
            dimension.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        use_conv: bool,
        out_channels: Optional[int] = None,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.num_channels
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    """
    Residual block with timestep conditioning.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels
        temb_channels: number of timestep embedding  channels
        out_channels: number of output channels.
        up: if True, performs upsampling.
        down: if True, performs downsampling.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        out_channels: Optional[int] = None,
        up: bool = False,
        down: bool = False,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = in_channels
        self.emb_channels = temb_channels
        self.out_channels = out_channels or in_channels
        self.up = up
        self.down = down

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.nonlinearity = nn.SiLU()
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample(spatial_dims, in_channels, use_conv=False)
        elif down:
            self.downsample = Downsample(spatial_dims, in_channels, use_conv=False)

        self.time_emb_proj = nn.Linear(
            temb_channels,
            self.out_channels,
        )

        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.out_channels, eps=norm_eps, affine=True)
        self.conv2 = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)

        if self.upsample is not None:
            if h.shape[0] >= 64:
                x = x.contiguous()
                h = h.contiguous()
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)

        if self.spatial_dims == 2:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None]
        else:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None, None]
        h = h + temb

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)

        return self.skip_connection(x) + h


def get_attention_parameters(
    num_channels: int, attention_head_dim: int, num_attention_heads: int, legacy: bool, with_conditioning: bool
) -> Tuple[int, int]:
    """
    Get the number of attention heads and their dimensions depending on the model parameters.

    Args:
        num_channels: number of channels.
        attention_head_dim: number of channels in each head.
        num_attention_heads: number of attention heads.
        legacy: if True, use legacy way to compute dim_head for attention blocks.
        with_conditioning: if true together with legacy, use ch // num_heads as head dimension.

    """
    if attention_head_dim == -1:
        dim_head = num_channels // num_attention_heads
    else:
        num_attention_heads = num_channels // attention_head_dim
        dim_head = attention_head_dim
    if legacy:
        dim_head = num_channels // num_attention_heads if with_conditioning else attention_head_dim
    return dim_head, num_attention_heads


class DownBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsampler = Downsample(
                spatial_dims=spatial_dims,
                num_channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                padding=downsample_padding,
            )
        else:
            self.downsampler = None

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Any]:
        output_states = []

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class AttnDownBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        num_attention_heads: int = 1,
        attention_head_dim: int = 1,
    ) -> None:
        super().__init__()
        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                AttentionBlock(
                    num_channels=out_channels,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_downsample:
            self.downsampler = Downsample(
                spatial_dims=spatial_dims,
                num_channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                padding=downsample_padding,
            )
        else:
            self.downsampler = None

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Any]:
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnDownBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        num_attention_heads: int = 1,
        attention_head_dim: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

            attentions.append(
                SpatialTransformer(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    num_layers=transformer_num_layers,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    cross_attention_dim=cross_attention_dim,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_downsample:
            self.downsampler = Downsample(
                spatial_dims=spatial_dims,
                num_channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                padding=downsample_padding,
            )
        else:
            self.downsampler = None

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Any]:
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=context)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class AttnMidBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_attention_heads: int = 1,
        attention_head_dim: int = 1,
    ) -> None:
        super().__init__()
        self.attention = None

        self.resnet_1 = ResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )
        self.attention = AttentionBlock(
            num_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

        self.resnet_2 = ResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.resnet_2(hidden_states, temb)

        return hidden_states


class CrossAttnMidBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_attention_heads: int = 1,
        attention_head_dim: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.attention = None

        self.resnet_1 = ResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )
        self.attention = SpatialTransformer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=transformer_num_layers,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
        )
        self.resnet_2 = ResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attention(hidden_states, context=context)
        hidden_states = self.resnet_2(hidden_states, temb)

        return hidden_states


class UpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsampler = Upsample(
                spatial_dims=spatial_dims, num_channels=out_channels, use_conv=True, out_channels=out_channels
            )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for i, resnet in enumerate(self.resnets):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)

        return hidden_states


class AttnUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        num_attention_heads: int = 1,
        attention_head_dim: int = 1,
    ) -> None:
        super().__init__()
        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                AttentionBlock(
                    num_channels=out_channels,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_upsample:
            self.upsampler = Upsample(
                spatial_dims=spatial_dims, num_channels=out_channels, use_conv=True, out_channels=out_channels
            )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)

        return hidden_states


class CrossAttnUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        num_attention_heads: int = 1,
        attention_head_dim: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                SpatialTransformer(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    num_layers=transformer_num_layers,
                    cross_attention_dim=cross_attention_dim,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_upsample:
            self.upsampler = Upsample(
                spatial_dims=spatial_dims, num_channels=out_channels, use_conv=True, out_channels=out_channels
            )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=context)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)

        return hidden_states


def get_down_block(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    num_res_blocks: int,
    norm_num_groups: int,
    norm_eps: float,
    add_downsample: bool,
    with_attn: bool,
    with_cross_attn: bool,
    num_attention_heads: int,
    attention_head_dim: int,
    transformer_num_layers: int,
    cross_attention_dim: Optional[int],
) -> nn.Module:
    if with_attn:
        return AttnDownBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_downsample=add_downsample,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
        )
    elif with_cross_attn:
        return CrossAttnDownBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_downsample=add_downsample,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
        )
    else:
        return DownBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_downsample=add_downsample,
        )


def get_mid_block(
    spatial_dims: int,
    in_channels: int,
    temb_channels: int,
    norm_num_groups: int,
    norm_eps: float,
    with_conditioning: bool,
    num_attention_heads: int,
    attention_head_dim: int,
    transformer_num_layers: int,
    cross_attention_dim: Optional[int],
) -> nn.Module:
    if with_conditioning:
        return CrossAttnMidBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
        )
    else:
        return AttnMidBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
        )


def get_up_block(
    spatial_dims: int,
    in_channels: int,
    prev_output_channel: int,
    out_channels: int,
    temb_channels: int,
    num_res_blocks: int,
    norm_num_groups: int,
    norm_eps: float,
    add_upsample: bool,
    with_attn: bool,
    with_cross_attn: bool,
    num_attention_heads: int,
    attention_head_dim: int,
    transformer_num_layers: int,
    cross_attention_dim: Optional[int],
) -> nn.Module:
    if with_attn:
        return AttnUpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_upsample=add_upsample,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
        )
    elif with_cross_attn:
        return CrossAttnUpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_upsample=add_upsample,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
        )
    else:
        return UpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_upsample=add_upsample,
        )


class DiffusionModelUNet(nn.Module):
    """
    Unet network with timestep embedding and attention mechanisms for conditioning based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        block_out_channels: tuple of block output channels.
        attention_levels: list of levels to add attention.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        num_heads: number of attention heads.
        num_head_channels: number of channels in each head.
        legacy: if True, use legacy way to compute dim_head for attention blocks.
        with_conditioning: if True add spatial transformers to perform conditioning.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        block_out_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_heads: int = -1,
        num_head_channels: int = -1,
        legacy: bool = True,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if with_conditioning is True and cross_attention_dim is None:
            raise ValueError(
                (
                    "DiffusionModelUNet expects dimension of the cross-attention conditioning (cross_attention_dim) "
                    "when using with_conditioning."
                )
            )
        if cross_attention_dim is not None and with_conditioning is False:
            raise ValueError(
                "DiffusionModelUNet expects use_spatial_transformer=True when specifying the " "cross_attention_dim."
            )

        if num_heads == -1 and num_head_channels == -1:
            raise ValueError("DiffusionModelUNet expects that either num_heads or num_head_channels has to be set.")

        # The number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in block_out_channels):
            raise ValueError("DiffusionModelUNet expects all block_out_channels being multiple of norm_num_groups")

        self.in_channels = in_channels
        self.block_out_channels = block_out_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        # input
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # time
        time_embed_dim = block_out_channels[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(block_out_channels[0], time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # down
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            dim_head, num_heads = get_attention_parameters(
                input_channel, num_head_channels, num_heads, legacy, with_conditioning
            )

            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=num_res_blocks,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_attention_heads=num_heads,
                attention_head_dim=dim_head,
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
            )

            self.down_blocks.append(down_block)

        # mid
        dim_head, num_heads = get_attention_parameters(
            block_out_channels[-1], num_head_channels, num_heads, legacy, with_conditioning
        )
        self.middle_block = get_mid_block(
            spatial_dims=spatial_dims,
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_conditioning=with_conditioning,
            num_attention_heads=num_heads,
            attention_head_dim=dim_head,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
        )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_levels = list(reversed(attention_levels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            dim_head, num_heads = get_attention_parameters(
                output_channel, num_head_channels, num_heads, legacy, with_conditioning
            )

            up_block = get_up_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=num_res_blocks + 1,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_upsample=not is_final_block,
                with_attn=(reversed_attention_levels[i] and not with_conditioning),
                with_cross_attn=(reversed_attention_levels[i] and with_conditioning),
                num_attention_heads=num_heads,
                attention_head_dim=dim_head,
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_out_channels[0], eps=norm_eps, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=block_out_channels[0],
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor. (N, C, SpatialDims)
            timesteps: timestep tensor (N,)
            context: context tensor (N, 1, ContextDim)
        """
        # 1. time
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])
        emb = self.time_embed(t_emb)

        # 2. initial convolution
        h = self.conv_in(x)

        # 3. down
        down_block_res_samples = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            down_block_res_samples.extend(res_samples)

        # 4. mid
        h = self.middle_block(hidden_states=h, temb=emb, context=context)

        # 5. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            h = upsample_block(hidden_states=h, res_hidden_states_tuple=res_samples, temb=emb, context=context)

        # 6. output block
        h = self.out(h)

        return h
