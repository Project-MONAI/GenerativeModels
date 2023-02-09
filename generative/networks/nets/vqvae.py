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

from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.layers import Act

from generative.networks.layers.vector_quantizer import EMAQuantizer, VectorQuantizer

__all__ = ["VQVAE"]


class VQVAEResidualUnit(nn.Module):
    """
    Implementation of the ResidualLayer used in the VQVAE network as originally used in Morphology-preserving
    Autoregressive 3D Generative Modelling of the Brain by Tudosiu et al. (https://arxiv.org/pdf/2209.03177.pdf) and
    the original implementation that can be found at
    https://github.com/AmigoLab/SynthAnatomy/blob/main/src/networks/vqvae/baseline.py#L150.

    Args:
        spatial_dims: number of spatial spatial_dims of the input data.
        num_channels: number of input channels.
        num_res_channels: number of channels in the residual layers.
        adn_ordering : a string representing the ordering of activation, normalization, and dropout. Defaults to "NDA".
        act: activation type and arguments. Defaults to RELU.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: dimension along which to apply dropout. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        num_res_channels: int,
        adn_ordering: str = "NDA",
        act: tuple | str | None = "RELU",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.spatial_dims = spatial_dims
        self.num_channels = num_channels
        self.num_res_channels = num_res_channels
        self.adn_ordering = adn_ordering
        self.act = act
        self.dropout = dropout
        self.dropout_dim = dropout_dim
        self.bias = bias

        self.conv1 = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=self.num_channels,
            out_channels=self.num_res_channels,
            adn_ordering=self.adn_ordering,
            act=self.act,
            norm=None,
            dropout=self.dropout,
            dropout_dim=self.dropout_dim,
            bias=self.bias,
        )

        self.conv2 = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=self.num_res_channels,
            out_channels=self.num_channels,
            adn_ordering=self.adn_ordering,
            act=None,
            norm=None,
            dropout=None,
            dropout_dim=self.dropout_dim,
            bias=self.bias,
        )

    def forward(self, x):
        return torch.nn.functional.relu(x + self.conv2(self.conv1(x)), True)


class VQVAE(nn.Module):
    """
    Single bottleneck implementation of Vector-Quantised Variational Autoencoder (VQ-VAE) as originally used in
    Morphology-preserving Autoregressive 3D Generative Modelling of the Brain by Tudosiu et al.
    (https://arxiv.org/pdf/2209.03177.pdf) and the original implementation that can be found at
    https://github.com/AmigoLab/SynthAnatomy/blob/main/src/networks/vqvae/baseline.py#L163/

    Args:
        spatial_dims: number of spatial spatial_dims.
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_levels: number of levels that the network has.
        downsample_parameters: A Tuple of Tuples for defining the downsampling convolutions. Each Tuple should hold the
            following information stride (int), kernel_size (int), dilation (int) and padding (int).
        upsample_parameters: A Tuple of Tuples for defining the upsampling convolutions. Each Tuple should hold the
            following information stride (int), kernel_size (int), dilation (int), padding (int), output_padding (int).
            If use_subpixel_conv is True, only the stride will be used for the last conv as the scale_factor.
        num_res_layers: number of sequential residual layers at each level.
        num_channels: number of channels at each level.
        num_res_channels: number of channels in the residual layers at each level.
        num_embeddings: VectorQuantization number of atomic elements in the codebook.
        embedding_dim: VectorQuantization number of channels of the input and atomic elements.
        commitment_cost: VectorQuantization commitment_cost.
        decay: VectorQuantization decay.
        epsilon: VectorQuantization epsilon.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout, e.g. "NDA".
        act: activation type and arguments.
        dropout: dropout ratio.
        ddp_sync: whether to synchronize the codebook across processes.
    """

    # < Python 3.9 TorchScript requirement for ModuleList
    __constants__ = ["encoder", "quantizer", "decoder"]

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_levels: int = 3,
        downsample_parameters: tuple[tuple[int, int, int, int], ...] = ((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
        upsample_parameters: tuple[tuple[int, int, int, int, int], ...] = (
            (2, 4, 1, 1, 0),
            (2, 4, 1, 1, 0),
            (2, 4, 1, 1, 0),
        ),
        num_res_layers: int = 3,
        num_channels: Sequence[int] = (96, 96, 192),
        num_res_channels: Sequence[int] = (96, 96, 192),
        num_embeddings: int = 32,
        embedding_dim: int = 64,
        embedding_init: str = "normal",
        commitment_cost: float = 0.25,
        decay: float = 0.5,
        epsilon: float = 1e-5,
        adn_ordering: str = "NDA",
        dropout: tuple | str | float | None = 0.1,
        act: tuple | str | None = "RELU",
        output_act: tuple | str | None = None,
        ddp_sync: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims

        assert (
            num_levels == len(downsample_parameters)
            and num_levels == len(upsample_parameters)
            and num_levels == len(num_channels)
            and num_levels == len(num_res_channels)
        ), (
            f"downsample_parameters, upsample_parameters, num_channels and num_res_channels must have the same number of"
            f" elements as num_levels. But got {len(downsample_parameters)}, {len(upsample_parameters)}, "
            f"{len(num_channels)} and {len(num_res_channels)} instead of {num_levels}."
        )

        self.num_levels = num_levels
        self.downsample_parameters = downsample_parameters
        self.upsample_parameters = upsample_parameters
        self.num_res_layers = num_res_layers
        self.num_channels = num_channels
        self.num_res_channels = num_res_channels

        self.dropout = dropout
        self.act = act
        self.adn_ordering = adn_ordering

        self.output_act = output_act

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_init = embedding_init
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.ddp_sync = ddp_sync

        self.encoder = self.construct_encoder()
        self.quantizer = self.construct_quantizer()
        self.decoder = self.construct_decoder()

    def construct_encoder(self) -> torch.nn.Sequential:
        encoder = []

        for idx in range(self.num_levels):
            encoder.append(
                Convolution(
                    spatial_dims=self.spatial_dims,
                    in_channels=self.in_channels if idx == 0 else self.num_channels[idx - 1],
                    out_channels=self.num_channels[idx],
                    strides=self.downsample_parameters[idx][0],
                    kernel_size=self.downsample_parameters[idx][1],
                    adn_ordering=self.adn_ordering,
                    act=self.act,
                    norm=None,
                    dropout=None if idx == 0 else self.dropout,
                    dropout_dim=1,
                    dilation=self.downsample_parameters[idx][2],
                    groups=1,
                    bias=True,
                    conv_only=False,
                    is_transposed=False,
                    padding=self.downsample_parameters[idx][3],
                    output_padding=None,
                )
            )

            for _ in range(self.num_res_layers):
                encoder.append(
                    VQVAEResidualUnit(
                        spatial_dims=self.spatial_dims,
                        num_channels=self.num_channels[idx],
                        num_res_channels=self.num_res_channels[idx],
                        adn_ordering=self.adn_ordering,
                        act=self.act,
                        dropout=self.dropout,
                        dropout_dim=1,
                        bias=True,
                    )
                )

        encoder.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=self.num_channels[len(self.num_channels) - 1],
                out_channels=self.embedding_dim,
                strides=1,
                kernel_size=3,
                adn_ordering=self.adn_ordering,
                act=None,
                norm=None,
                dropout=None,
                dropout_dim=1,
                dilation=1,
                bias=True,
                conv_only=True,
                is_transposed=False,
                padding=1,
                output_padding=None,
            )
        )

        return torch.nn.Sequential(*encoder)

    # TODO: Include lucidrains' vector quantizer as an option
    def construct_quantizer(self) -> torch.nn.Module:
        return VectorQuantizer(
            quantizer=EMAQuantizer(
                spatial_dims=self.spatial_dims,
                num_embeddings=self.num_embeddings,
                embedding_dim=self.embedding_dim,
                commitment_cost=self.commitment_cost,
                decay=self.decay,
                epsilon=self.epsilon,
                embedding_init=self.embedding_init,
                ddp_sync=self.ddp_sync,
            )
        )

    def construct_decoder(self) -> torch.nn.Sequential:
        decoder_num_channels = list(reversed(self.num_channels))
        decoder_num_res_channels = list(reversed(self.num_res_channels))

        decoder = [
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=self.embedding_dim,
                out_channels=decoder_num_channels[0],
                strides=1,
                kernel_size=3,
                adn_ordering=self.adn_ordering,
                act=None,
                dropout=None,
                norm=None,
                dropout_dim=1,
                dilation=1,
                bias=True,
                is_transposed=False,
                padding=1,
                output_padding=None,
            )
        ]

        for idx in range(self.num_levels):
            for _ in range(self.num_res_layers):
                decoder.append(
                    VQVAEResidualUnit(
                        spatial_dims=self.spatial_dims,
                        num_channels=decoder_num_channels[idx],
                        num_res_channels=decoder_num_res_channels[idx],
                        adn_ordering=self.adn_ordering,
                        act=self.act,
                        dropout=self.dropout,
                        dropout_dim=1,
                        bias=True,
                    )
                )

            decoder.append(
                Convolution(
                    spatial_dims=self.spatial_dims,
                    in_channels=decoder_num_channels[idx],
                    out_channels=self.out_channels if idx == self.num_levels - 1 else decoder_num_channels[idx + 1],
                    strides=self.upsample_parameters[idx][0],
                    kernel_size=self.upsample_parameters[idx][1],
                    adn_ordering=self.adn_ordering,
                    act=self.act,
                    dropout=self.dropout if idx != self.num_levels - 1 else None,
                    norm=None,
                    dropout_dim=1,
                    dilation=self.upsample_parameters[idx][2],
                    groups=1,
                    bias=True,
                    conv_only=idx == self.num_levels - 1,
                    is_transposed=True,
                    padding=self.upsample_parameters[idx][3],
                    output_padding=self.upsample_parameters[idx][4],
                )
            )

        if self.output_act:
            decoder.append(Act[self.output_act]())

        return torch.nn.Sequential(*decoder)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def quantize(self, encodings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_loss, x = self.quantizer(encodings)
        return x, x_loss

    def decode(self, quantizations: torch.Tensor) -> torch.Tensor:
        return self.decoder(quantizations)

    def index_quantize(self, images: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantize(self.encode(images=images))

    def decode_samples(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.decode(self.quantizer.embed(embedding_indices))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quantizations, quantization_losses = self.quantize(self.encode(images))
        reconstruction = self.decode(quantizations)

        return reconstruction, quantization_losses

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        e, _ = self.quantize(z)
        return e

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        e, _ = self.quantize(z)
        image = self.decode(e)
        return image
