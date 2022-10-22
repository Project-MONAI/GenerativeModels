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

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution

__all__ = ["AutoencoderKL"]


class Upsample(nn.Module):
    """
    Convolution-based upsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels to the layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
    ) -> None:
        super().__init__()
        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
           x: BxCx[SPATIAL DIMS] tensor
        """
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Convolution-based downsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
    ) -> None:
        super().__init__()
        if spatial_dims == 2:
            self.pad = (0, 1, 0, 1)
        elif spatial_dims == 3:
            self.pad = (0, 1, 0, 1, 0, 1)

        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=2,
            kernel_size=3,
            padding=0,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: BxCx[SPATIAL DIMS] tensor
        """
        x = nn.functional.pad(x, self.pad, mode="constant", value=0.0)
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block consisting of a cascade of 2 convolutions + activation + normalisation block, and a
    residual connection between input and output.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: input channels to the layer.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon for the normalisation.
        out_channels: number of output channels.
    """

    def __init__(
        self, spatial_dims: int, in_channels: int, norm_num_groups: int, norm_eps: float, out_channels
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=norm_eps, affine=True)
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: BxCx[SPATIAL DIMS]
        """
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    """
    Attention block.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon for the normalisation.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm_num_groups: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.q = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.k = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.v = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.proj_out = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: BxCx[SPATIAL DIMS] tensor
        """
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        if self.spatial_dims == 2:
            b, c, h, w = q.shape
            n_spatial_elements = h * w
        if self.spatial_dims == 3:
            b, c, h, w, d = q.shape
            n_spatial_elements = h * w * d

        q = q.reshape(b, c, n_spatial_elements)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, n_spatial_elements)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # Attend to values
        v = v.reshape(b, c, n_spatial_elements)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)

        if self.spatial_dims == 2:
            h_ = h_.reshape(b, c, h, w)
        if self.spatial_dims == 3:
            h_ = h_.reshape(b, c, h, w, d)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    """
    Convolutional cascade that downsamples the image into a spatial latent space.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
        n_channels: number of filters in the first downsampling.
        out_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        ch_mult: list of multipliers of n_channels in the initial layer and in  each downsampling layer. Example: if
            you want three downsamplings, you have to input a 4-element list. If you input [1, 1, 2, 2],
            the first downsampling will leave n_channels to n_channels, the next will multiply n_channels by 2,
            and the next will multiply n_channels*2 by 2 again, resulting in 8, 8, 16 and 32 channels.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        resolution: spatial dimensions of the input image.
        norm_num_groups: number of groups for the GroupNorm layers, n_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        with_attention: whether to include Attention Blocks or not.
        attn_resolutions: containing the max spatial sizes of latent space representation that trigger the inclusion
            of an attention block. i.e. if 8 is in the list, Attention will be applied when the max activation spatial
            size is 8.
        with_nonlocal_attn: if True use non-local attention block.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        n_channels: int,
        out_channels: int,
        ch_mult: Sequence[int],
        num_res_blocks: int,
        resolution: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        with_attention: bool,
        attn_resolutions: Optional[Sequence[int]],
        with_nonlocal_attn: bool = True,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.with_attention = with_attention
        self.attn_resolutions = attn_resolutions

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        # Initial convolution
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=n_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Residual and downsampling blocks
        for i in range(self.num_resolutions):
            block_in_ch = n_channels * in_ch_mult[i]
            block_out_ch = n_channels * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(
                    ResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=block_out_ch,
                    )
                )
                block_in_ch = block_out_ch
                if self.with_attention and max(curr_res) in attn_resolutions:
                    blocks.append(AttnBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps))

            if i != self.num_resolutions - 1:
                blocks.append(Downsample(spatial_dims, block_in_ch))
                curr_res = tuple(ti // 2 for ti in curr_res)

        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(ResBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps, block_in_ch))
            if self.with_attention:
                blocks.append(AttnBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps))
            blocks.append(ResBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps, block_in_ch))

        # Normalise and convert to latent size
        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_in_ch, eps=norm_eps, affine=True))
        blocks.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=block_in_ch,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: BxCx[SPATIAL DIMS] tensor
        """
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    """
    Convolutional cascade upsampling from a spatial latent space into an image space.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        n_channels: number of filters in the last upsampling.
        in_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        out_channels: number of output channels.
        ch_mult: list of multipliers of n_channels that make for all the upsampling layers before the last. In the
            last layer, there will be a transition from n_channels to out_channels. In the layers before that,
            channels will be the product of the previous number of channels by ch_mult.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        resolution: spatial dimensions of the input image
        norm_num_groups: number of groups for the GroupNorm layers, n_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        with_attention: whether to include Attention Blocks or not.
        attn_resolutions: containing the max spatial sizes of latent space representation that trigger the inclusion
            of an attention block. i.e. if 8 is in the list, Attention will be applied when the max activation spatial
            size is 8.
        with_nonlocal_attn: if True use non-local attention block.
    """

    def __init__(
        self,
        spatial_dims: int,
        n_channels: int,
        in_channels: int,
        out_channels: int,
        ch_mult: Sequence[int],
        num_res_blocks: int,
        resolution: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        with_attention: bool,
        attn_resolutions: Optional[Sequence[int]],
        with_nonlocal_attn: bool = True,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.n_channels = n_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.with_attention = with_attention
        self.attn_resolutions = attn_resolutions

        block_in_ch = n_channels * self.ch_mult[-1]
        curr_res = tuple(ti // 2 ** (self.num_resolutions - 1) for ti in resolution)

        blocks = []
        # Initial conv
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=block_in_ch,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(ResBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps, block_in_ch))
            if self.with_attention:
                blocks.append(AttnBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps))
            blocks.append(ResBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps, block_in_ch))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = n_channels * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps, block_out_ch))
                block_in_ch = block_out_ch

                if self.with_attention and max(curr_res) in self.attn_resolutions:
                    blocks.append(AttnBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps))

            if i != 0:
                blocks.append(Upsample(spatial_dims, block_in_ch))
                curr_res = tuple(ti * 2 for ti in curr_res)

        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_in_ch, eps=norm_eps, affine=True))
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=block_in_ch,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class AutoencoderKL(nn.Module):
    """
    Autoencoder model with KL-regularized latent space based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models"
    https://arxiv.org/abs/2112.10752

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
        out_channels: number of output channels.
        n_channels: number of filters in the first downsampling / last upsampling.
        latent_channels: latent embedding dimension.
        ch_mult: multiplier of the number of channels in each downsampling layer (+ initial one). i.e.: If you want 3
            downsamplings, it should be a 4-element list. num_res_blocks: number of residual blocks (see ResBlock) per
            level.
        resolution: spatial dimensions of the input image.
        norm_num_groups: number of groups for the GroupNorm layers, n_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        with_attention: whether to include Attention Blocks or not.
        attn_resolutions: containing the max spatial sizes of latent space representation that trigger the inclusion
            of an attention block. i.e. if 8 is in the list, Attention will be applied when the max activation spatial
            size is 8.
        with_encoder_nonlocal_attn: if True use non-local attention block in the encoder.
        with_decoder_nonlocal_attn: if True use non-local attention block in the decoder.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        n_channels: int,
        latent_channels: int,
        ch_mult: Sequence[int],
        num_res_blocks: int,
        resolution: Sequence[int],
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_attention: bool = True,
        attn_resolutions: Optional[Sequence[int]] = None,
        with_encoder_nonlocal_attn: bool = True,
        with_decoder_nonlocal_attn: bool = True,
    ) -> None:
        super().__init__()
        if attn_resolutions is None:
            attn_resolutions = []

        # The number of channels should be multiple of num_groups
        if (n_channels % norm_num_groups) != 0:
            raise ValueError("AutoencoderKL expects number of channels being multiple of number of groups")

        self.encoder = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            n_channels=n_channels,
            out_channels=latent_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_attention=with_attention,
            attn_resolutions=attn_resolutions,
            with_nonlocal_attn=with_encoder_nonlocal_attn,
        )
        self.decoder = Decoder(
            spatial_dims=spatial_dims,
            n_channels=n_channels,
            in_channels=latent_channels,
            out_channels=out_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_attention=with_attention,
            attn_resolutions=attn_resolutions,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
        )
        self.quant_conv_mu = Convolution(spatial_dims, latent_channels, latent_channels, 1, conv_only=True)
        self.quant_conv_log_sigma = Convolution(spatial_dims, latent_channels, latent_channels, 1, conv_only=True)
        self.post_quant_conv = Convolution(spatial_dims, latent_channels, latent_channels, 1, conv_only=True)
        self.latent_channels = latent_channels

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forwards an image through the spatial encoder, obtaining the latent mean and sigma representations.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        """
        h = self.encoder(x)

        z_mu = self.quant_conv_mu(h)
        z_log_var = self.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        return z_mu, z_sigma

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        """
        From the mean and sigma representations resulting of encoding an image through the latent space,
        obtains a noise sample resulting from sampling gaussian noise, multiplying by the variance (sigma) and
        adding the mean.

        Args:
            z_mu: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] mean vector obtained by the encoder when you encode an image
            z_sigma: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] variance vector obtained by the encoder when you encode an image

        Returns:
            sample of shape Bx[Z_CHANNELS]x[LATENT SPACE SIZE]
        """
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes and decodes an input image.

        Args:
            x: BxCx[SPATIAL DIMENSIONS] tensor.

        Returns:
            reconstructed image, of the same shape as input
        """
        z_mu, _ = self.encode(x)
        reconstruction = self.decode(z_mu)
        return reconstruction

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Based on a latent space sample, forwards it through the Decoder.

        Args:
            z: Bx[Z_CHANNELS]x[LATENT SPACE SHAPE]

        Returns:
            decoded image tensor
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: BxCx[SPATIAL DIMS] input image tensor

        Returns:
            reconstructed image, and the mu and sigma vectors of the encoder.
        """
        z_mu, z_sigma = self.encode(x)
        z = self.sampling(z_mu, z_sigma)
        reconstruction = self.decode(z)
        return reconstruction, z_mu, z_sigma
