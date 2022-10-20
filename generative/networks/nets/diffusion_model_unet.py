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

import math
from abc import abstractmethod
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Pool
from torch import einsum, nn

from generative.utils.misc import default, exists

__all__ = ["DiffusionModelUNet"]


class GEGLU(nn.Module):
    # TODO: Add docstring
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    # TODO: Add docstring
    def __init__(
        self, dim: int, dim_out: Optional[int] = None, mult: int = 4, glu: bool = False, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    # TODO: Add docstring
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    # TODO: Add docstring
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
    ) -> None:
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention if context is None
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    # TODO: Add docstring
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
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
                    dim=inner_dim, n_heads=n_heads, d_head=d_head, dropout=dropout, context_dim=context_dim
                )
                for _ in range(depth)
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


def timestep_embedding(timesteps: int, dim: int, max_period: int = 10000):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    # TODO: Add docstring
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


def zero_module(module: nn.Module) -> nn.Module:
    # TODO: Add docstring
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class TimestepBlock(nn.Module):
    # TODO: Add docstring
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    # TODO: Add docstring
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x: torch.Tensor, emb: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class QKVAttentionLegacy(nn.Module):
    # TODO: Add docstring
    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    # TODO: Add docstring
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=channels, eps=norm_eps, affine=True)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class Downsample(nn.Module):
    # TODO: Add docstring
    def __init__(
        self,
        spatial_dims: int,
        channels: int,
        use_conv: bool,
        out_channels: Optional[int] = None,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.op = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.channels,
                out_channels=self.out_channels,
                strides=2,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            assert self.channels == self.out_channels
            self.op = Pool[Pool.AVG, spatial_dims](kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):
    # TODO: Add docstring
    def __init__(
        self,
        spatial_dims: int,
        channels: int,
        use_conv: bool,
        out_channels: Optional[int] = None,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock(TimestepBlock):
    # TODO: Add docstring
    def __init__(
        self,
        spatial_dims: int,
        channels: int,
        emb_channels: int,
        dropout: float = 0.0,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        up: bool = False,
        down: bool = False,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=channels, eps=norm_eps, affine=True),
            nn.SiLU(),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            ),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(spatial_dims, channels, False)
            self.x_upd = Upsample(spatial_dims, channels, False)
        elif down:
            self.h_upd = Downsample(spatial_dims, channels, False)
            self.x_upd = Downsample(spatial_dims, channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.out_channels, eps=norm_eps, affine=True),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )

        else:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


def get_attention_parameters(
    ch: int, num_head_channels: int, num_heads: int, legacy: bool, use_spatial_transformer: bool
) -> Tuple[int, int]:
    # TODO: Add docstring
    """
    Get the number of attention heads and their dimensions depending on the model parameters.

    Args:
        ch:
        num_head_channels:
        num_heads:
        legacy:
        use_spatial_transformer:

    """
    if num_head_channels == -1:
        dim_head = ch // num_heads
    else:
        num_heads = ch // num_head_channels
        dim_head = num_head_channels
    if legacy:
        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
    return dim_head, num_heads


class DiffusionModelUNet(nn.Module):
    # TODO: Add docstring
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Sequence[int],
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        num_heads: int = -1,
        num_head_channels: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        use_spatial_transformer: bool = False,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        legacy: bool = True,
    ) -> None:
        super().__init__()

        if use_spatial_transformer is True and context_dim is None:
            raise ValueError(
                (
                    "DiffusionModelUNet expects dimension of the cross-attention conditioning (context_dim) when using "
                    "use_spatial_transformer."
                )
            )
        if context_dim is not None and use_spatial_transformer is False:
            raise ValueError("DiffusionModelUNet expects use_spatial_transformer=True when specifying the context_dim.")

        if num_heads == -1 and num_head_channels == -1:
            raise ValueError("DiffusionModelUNet expects that either num_heads or num_head_channels has to be set.")

        # The number of channels should be multiple of num_groups
        if (model_channels % norm_num_groups) != 0:
            raise ValueError("DiffusionModelUNet expects model_channels being multiple of norm_num_groups")

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=in_channels,
                        out_channels=model_channels,
                        strides=1,
                        kernel_size=3,
                        padding=1,
                        conv_only=True,
                    )
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        spatial_dims=spatial_dims,
                        channels=ch,
                        emb_channels=time_embed_dim,
                        out_channels=mult * model_channels,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head, num_heads = get_attention_parameters(
                        ch, num_head_channels, num_heads, legacy, use_spatial_transformer
                    )

                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            spatial_dims=spatial_dims,
                            in_channels=ch,
                            n_heads=num_heads,
                            d_head=dim_head,
                            depth=transformer_depth,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            context_dim=context_dim,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            spatial_dims=spatial_dims,
                            channels=ch,
                            emb_channels=time_embed_dim,
                            out_channels=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                        )
                        if resblock_updown
                        else Downsample(spatial_dims=spatial_dims, channels=ch, use_conv=True, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        dim_head, num_heads = get_attention_parameters(
            ch, num_head_channels, num_heads, legacy, use_spatial_transformer
        )

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                spatial_dims=spatial_dims,
                channels=ch,
                emb_channels=time_embed_dim,
                use_scale_shift_norm=use_scale_shift_norm,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=dim_head,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
            )
            if not use_spatial_transformer
            else SpatialTransformer(
                spatial_dims=spatial_dims,
                in_channels=ch,
                n_heads=num_heads,
                d_head=dim_head,
                depth=transformer_depth,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                context_dim=context_dim,
            ),
            ResBlock(
                spatial_dims=spatial_dims,
                channels=ch,
                emb_channels=time_embed_dim,
                use_scale_shift_norm=use_scale_shift_norm,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        spatial_dims=spatial_dims,
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        out_channels=model_channels * mult,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    dim_head, num_heads = get_attention_parameters(
                        ch, num_head_channels, num_heads, legacy, use_spatial_transformer
                    )

                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            spatial_dims=spatial_dims,
                            in_channels=ch,
                            n_heads=num_heads,
                            d_head=dim_head,
                            depth=transformer_depth,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            context_dim=context_dim,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            spatial_dims=spatial_dims,
                            channels=ch,
                            emb_channels=time_embed_dim,
                            out_channels=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                        )
                        if resblock_updown
                        else Upsample(spatial_dims=spatial_dims, channels=ch, use_conv=True, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=ch, eps=norm_eps, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=model_channels,
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
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        return self.out(h)
