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
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.utils import ensure_tuple_rep

from generative.networks.blocks.spade_norm import SPADE
from generative.networks.nets.autoencoderkl import AttentionBlock, Encoder, Upsample

# To install xformers, use pip install xformers==0.0.16rc401
if importlib.util.find_spec("xformers") is not None:
    import xformers

    has_xformers = True
else:
    xformers = None
    has_xformers = False

# TODO: Use MONAI's optional_import
# from monai.utils import optional_import
# xformers, has_xformers = optional_import("xformers.ops", name="xformers")

__all__ = ["SPADEAutoencoderKL"]


class SPADEResBlock(nn.Module):
    """
    Residual block consisting of a cascade of 2 convolutions + activation + normalisation block, and a
    residual connection between input and output.
    Enables SPADE normalisation for semantic conditioning (Park et. al (2019): https://github.com/NVlabs/SPADE)

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: input channels to the layer.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon for the normalisation.
        out_channels: number of output channels.
        label_nc: number of semantic channels for SPADE normalisation
        spade_intermediate_channels: number of intermediate channels for SPADE block layer
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm_num_groups: int,
        norm_eps: float,
        out_channels: int,
        label_nc: int,
        spade_intermediate_channels: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = SPADE(
            label_nc=label_nc,
            norm_nc=in_channels,
            norm="GROUP",
            norm_params={"num_groups": norm_num_groups, "affine": False},
            hidden_channels=spade_intermediate_channels,
            kernel_size=3,
            spatial_dims=spatial_dims,
        )
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
        self.norm2 = SPADE(
            label_nc=label_nc,
            norm_nc=out_channels,
            norm="GROUP",
            norm_params={"num_groups": norm_num_groups, "affine": False},
            hidden_channels=spade_intermediate_channels,
            kernel_size=3,
            spatial_dims=spatial_dims,
        )
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

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h, seg)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h, seg)
        h = F.silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class SPADEDecoder(nn.Module):
    """
    Convolutional cascade upsampling from a spatial latent space into an image space.
    Enables SPADE normalisation for semantic conditioning (Park et. al (2019): https://github.com/NVlabs/SPADE)
    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        num_channels: sequence of block output channels.
        in_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, num_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from num_channels contain an attention block.
        label_nc: number of semantic channels for SPADE normalisation.
        with_nonlocal_attn: if True use non-local attention block.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
        spade_intermediate_channels: number of intermediate channels for SPADE block layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: Sequence[int],
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        label_nc: int,
        with_nonlocal_attn: bool = True,
        use_flash_attention: bool = False,
        spade_intermediate_channels: int = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels
        self.label_nc = label_nc

        reversed_block_out_channels = list(reversed(num_channels))

        blocks = []
        # Initial convolution
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=reversed_block_out_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(
                SPADEResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                    label_nc=label_nc,
                    spade_intermediate_channels=spade_intermediate_channels,
                )
            )
            blocks.append(
                AttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                SPADEResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                    label_nc=label_nc,
                    spade_intermediate_channels=spade_intermediate_channels,
                )
            )

        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        block_out_ch = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            block_in_ch = block_out_ch
            block_out_ch = reversed_block_out_channels[i]
            is_final_block = i == len(num_channels) - 1

            for _ in range(reversed_num_res_blocks[i]):
                blocks.append(
                    SPADEResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=block_out_ch,
                        label_nc=label_nc,
                        spade_intermediate_channels=spade_intermediate_channels,
                    )
                )
                block_in_ch = block_out_ch

                if reversed_attention_levels[i]:
                    blocks.append(
                        AttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=block_in_ch,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                blocks.append(Upsample(spatial_dims=spatial_dims, in_channels=block_in_ch, use_convtranspose=False))

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

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if isinstance(block, SPADEResBlock):
                x = block(x, seg)
            else:
                x = block(x)
        return x


class SPADEAutoencoderKL(nn.Module):
    """
    Autoencoder model with KL-regularized latent space based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162
    Enables SPADE normalisation for semantic conditioning (Park et. al (2019): https://github.com/NVlabs/SPADE)
    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        label_nc: number of semantic channels for SPADE normalisation.
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        num_channels: sequence of block output channels.
        attention_levels: sequence of levels to add attention.
        latent_channels: latent embedding dimension.
        norm_num_groups: number of groups for the GroupNorm layers, num_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        with_encoder_nonlocal_attn: if True use non-local attention block in the encoder.
        with_decoder_nonlocal_attn: if True use non-local attention block in the decoder.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
        spade_intermediate_channels: number of intermediate channels for SPADE block layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        label_nc: int,
        in_channels: int = 1,
        out_channels: int = 1,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        latent_channels: int = 3,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_encoder_nonlocal_attn: bool = True,
        with_decoder_nonlocal_attn: bool = True,
        use_flash_attention: bool = False,
        spade_intermediate_channels: int = 128,
    ) -> None:
        super().__init__()

        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError("SPADEAutoencoderKL expects all num_channels being multiple of norm_num_groups")

        if len(num_channels) != len(attention_levels):
            raise ValueError("SPADEAutoencoderKL expects num_channels being same size of attention_levels")

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        if use_flash_attention is True and not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. Flash attention is only available for GPU."
            )

        self.encoder = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channels=num_channels,
            out_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_encoder_nonlocal_attn,
            use_flash_attention=use_flash_attention,
        )
        self.decoder = SPADEDecoder(
            spatial_dims=spatial_dims,
            num_channels=num_channels,
            in_channels=latent_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            label_nc=label_nc,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
            use_flash_attention=use_flash_attention,
            spade_intermediate_channels=spade_intermediate_channels,
        )
        self.quant_conv_mu = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.quant_conv_log_sigma = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.post_quant_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.latent_channels = latent_channels

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def reconstruct(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """
        Encodes and decodes an input image.

        Args:
            x: BxCx[SPATIAL DIMENSIONS] tensor.
            seg: Bx[LABEL_NC]x[SPATIAL DIMENSIONS] tensor of segmentations for SPADE norm.
        Returns:
            reconstructed image, of the same shape as input
        """
        z_mu, _ = self.encode(x)
        reconstruction = self.decode(z_mu, seg)
        return reconstruction

    def decode(self, z: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """
        Based on a latent space sample, forwards it through the Decoder.

        Args:
            z: Bx[Z_CHANNELS]x[LATENT SPACE SHAPE]
            seg: Bx[LABEL_NC]x[SPATIAL DIMENSIONS] tensor of segmentations for SPADE norm.
        Returns:
            decoded image tensor
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z, seg)
        return dec

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mu, z_sigma = self.encode(x)
        z = self.sampling(z_mu, z_sigma)
        reconstruction = self.decode(z, seg)
        return reconstruction, z_mu, z_sigma

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma = self.encode(x)
        z = self.sampling(z_mu, z_sigma)
        return z

    def decode_stage_2_outputs(self, z: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        image = self.decode(z, seg)
        return image
