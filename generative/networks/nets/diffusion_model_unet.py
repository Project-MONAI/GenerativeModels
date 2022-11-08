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
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Pool
from torch import einsum, nn

from generative.utils.misc import default

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
        dim: number of channels in the input.
        dim_out: number of channels in the output. If not given, defaults to `dim`.
        mult: multiplier to use for the hidden dimension.
        glu: whether to use GLU activation.
        dropout: dropout probability to use.
    """

    def __init__(
        self, dim: int, dim_out: Optional[int] = None, mult: int = 4, glu: bool = False, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        # TODO: Try to remove default usage
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    """
    A cross attention layer.

    Args:
        query_dim: number of channels in the query.
        context_dim: number of channels in the context.
        heads: number of heads to use for multi-head attention.
        dim_head: number of channels in each head.
        dropout: dropout probability to use.
    """

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
        # TODO: Try to remove default usage
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.heads

        q = self.to_q(x)
        # TODO: Try to remove default usage
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    """
    A basic Transformer block.

    Args:
        dim: number of channels in the input and output.
        n_heads: number of heads to use for multi-head attention.
        d_head: number of channels in each head.
        dropout: dropout probability to use.
        context_dim: size of the context vector for cross attention.
        gated_ff: whether to use a gated feed-forward network.
    """

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
        n_heads: number of heads to use for multi-head attention.
        d_head: number of channels in each head.
        depth: number of layers of Transformer blocks to use.
        dropout: dropout probability to use.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        context_dim: number of context dimensions to use.
    """

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


class QKVAttentionLegacy(nn.Module):
    """
    A qkv attention mechanism.

    Args:
        n_heads: number of attention heads.
    """

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
    """
    An attention block.

    Args:
        channels: number of channels in the input and output.
        num_heads:  number of attention heads.
        num_head_channels: number of channels in each head.
        norm_num_groups: number of groups to use for group norm.
        norm_eps: epsilon value to use for group norm.
    """

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


def get_timestep_embedding(timesteps: int, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
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
        channels: number of input channels.
        use_conv: if True uses Convolution instead of Pool average to perform downsampling.
        out_channels: number of output channels.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension.
    """

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
    """
    Upsampling layer.

    Args:
        spatial_dims: number of spatial dimensions.
        channels: number of input channels
        use_conv: if True uses Convolution instead of Pool average to perform downsampling.
        out_channels: number of output channels.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension.
    """

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


class ResnetBlock(nn.Module):
    """
    Residual block with timestep conditioning.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels
        temb_channels: number of timestep embedding  channels
        out_channels: number of output channels.
        use_conv: if True uses Convolution instead of Identity in skip connection.
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
        use_conv: bool = False,
        up: bool = False,
        down: bool = False,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channels = in_channels
        self.emb_channels = temb_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True),
            nn.SiLU(),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            ),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(spatial_dims, in_channels, False)
            self.x_upd = Upsample(spatial_dims, in_channels, False)
        elif down:
            self.h_upd = Downsample(spatial_dims, in_channels, False)
            self.x_upd = Downsample(spatial_dims, in_channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                temb_channels,
                self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.out_channels, eps=norm_eps, affine=True),
            nn.SiLU(),
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

        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )

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

        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


def get_attention_parameters(
    ch: int, num_head_channels: int, num_heads: int, legacy: bool, use_spatial_transformer: bool
) -> Tuple[int, int]:
    """
    Get the number of attention heads and their dimensions depending on the model parameters.

    Args:
        ch:
        num_head_channels: number of channels in each head.
        num_heads: number of attention heads.
        legacy: if True, use legacy way to compute dim_head for attention blocks.
        use_spatial_transformer: if true together with legacy, use ch // num_heads as head dimension.

    """
    if num_head_channels == -1:
        dim_head = ch // num_heads
    else:
        num_heads = ch // num_head_channels
        dim_head = num_head_channels
    if legacy:
        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
    return dim_head, num_heads


class DownBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        add_downsample=True,
        downsample_padding=1,
    ):
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
                channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                padding=downsample_padding,
            )
        else:
            self.downsampler = None

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


class MidBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_heads=1,
        num_head_channels=1,
    ):
        super().__init__()

        self.resnet_1 = ResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

        self.attention = AttentionBlock(
            channels=in_channels,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
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

    def forward(self, hidden_states, temb=None):
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attention(hidden_states)
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
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        add_upsample=True,
    ):
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
                spatial_dims=spatial_dims, channels=out_channels, use_conv=True, out_channels=out_channels
            )
        else:
            self.upsampler = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)

        return hidden_states


class DiffusionModelUNet(nn.Module):
    """
    Unet network with timestep embedding and attention mechanisms for conditioning based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        model_channels:
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        attention_resolutions: list of levels to add attention.
        num_heads: number of attention heads.
        num_head_channels: number of channels in each head.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        use_spatial_transformer: if True add spatial transformers to perform conditioning.
        transformer_depth: number of layers of Transformer blocks to use.
        context_dim: number of context dimensions to use.
        legacy: if True, use legacy way to compute dim_head for attention blocks.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Sequence[int] = (False, False, True, True),
        block_out_channels: Sequence[int] = (32, 64, 64, 64),
        num_heads: int = -1,
        num_head_channels: int = -1,
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
        # if (model_channels % norm_num_groups) != 0:
        #     raise ValueError("DiffusionModelUNet expects model_channels being multiple of norm_num_groups")

        self.in_channels = in_channels
        self.block_out_channels = block_out_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
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

            down_block = DownBlock(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=num_res_blocks,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
            )
            self.down_blocks.append(down_block)

        # mid
        dim_head, num_heads = get_attention_parameters(
            block_out_channels[-1], num_head_channels, num_heads, legacy, use_spatial_transformer
        )
        self.middle_block = MidBlock(
            spatial_dims=spatial_dims,
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_heads=num_heads,
            num_head_channels=dim_head,
        )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = UpBlock(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=num_res_blocks + 1,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_upsample=not is_final_block,
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
        down_block_res_samples = (h,)
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb)
            down_block_res_samples += res_samples

        # 4. mid
        h = self.middle_block(h, emb)

        # 5. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            h = upsample_block(h, res_samples, emb)

        # 6. output block
        h = self.out(h)

        return h
