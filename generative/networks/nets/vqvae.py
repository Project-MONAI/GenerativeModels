from typing import Optional, List, Union, Tuple

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution
from monai.networks.layers import Act

from generative.networks.layers.vector_quantizer import EMAQuantizer, VectorQuantizer

__all__ = ["ResidualUnit", "VQVAE"]


class ResidualUnit(nn.Module):
    """
    Implementation of the ResidualLayer used in the VQVAE network.

    Arg:
        spatial_dims: number of spatial spatial_dims of the input data.
        no_channels: number of input channels.
        no_res_channels: number of channels in the residual layers.
        adn_ordering : a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to RELU.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: dimension along which to apply dropout. Defaults to 1.
        groups: controls the connections between inputs and outputs. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
    """

    def __init__(
        self,
        spatial_dims: int,
        no_channels: int,
        no_res_channels: int,
        adn_ordering: str = "NDA",
        act: Optional[Union[Tuple, str]] = "RELU",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.spatial_dims = spatial_dims
        self.no_channels = no_channels
        self.no_res_channels = no_res_channels
        self.adn_ordering = adn_ordering
        self.act = act
        self.dropout = dropout
        self.dropout_dim = dropout_dim
        self.groups = groups
        self.bias = bias

        self.conv1 = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=self.no_channels,
            out_channels=self.no_res_channels,
            adn_ordering=self.adn_ordering,
            act=self.act,
            norm=None,
            dropout=self.dropout,
            dropout_dim=self.dropout_dim,
            bias=self.bias,
        )

        self.conv2 = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=self.no_res_channels,
            out_channels=self.no_channels,
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
    Single bottleneck implementation of Vector-Quantised Variational Autoencoder (VQ-VAE).

    Args:
        spatial_dims (int): number of spatial spatial_dims.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        no_levels (int): number of levels that the network has. Defaults to 3.
        downsample_parameters (Tuple[Tuple[int,int,int,int],...]): A Tuple of Tuples for defining the downsampling
            convolutions. Each Tuple should hold the following information stride (int), kernel_size (int),
            dilation(int) and padding (int). Defaults to ((2,4,1,1),(2,4,1,1),(2,4,1,1)).
        upsample_parameters (Tuple[Tuple[int,int,int,int,int],...]): A Tuple of Tuples for defining the upsampling
            convolutions. Each Tuple should hold the following information stride (int), kernel_size (int),
            dilation(int), padding (int), output_padding (int). If use_subpixel_conv is True, only the stride will
            be used for the last conv as the scale_factor. Defaults to ((2,4,1,1,0),(2,4,1,1,0),(2,4,1,1,0)).
        no_res_layers (int): number of sequential residual layers at each level. Defaults to 3.
        no_channels (int): number of channels at the deepest level, besides that is no_channels//2 .
            Defaults to 192.
        num_embeddings (int): VectorQuantization number of atomic elements in the codebook. Defaults to 32.
        embedding_dim (int): VectorQuantization number of channels of the input and atomic elements.
            Defaults to 64.
        commitment_cost (float): VectorQuantization commitment_cost. Defaults to 0.25 as per [1].
        decay (float): VectorQuantization decay. Defaults to 0.5.
        epsilon (float): VectorQuantization epsilon. Defaults to 1e-5 as per [1].
        adn_ordering (str): a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act Optional[Union[Tuple, str]]: activation type and arguments. Defaults to Relu.
        dropout (Optional[Union[Tuple, str, float]]): dropout ratio. Defaults to 0.1.
        ddp_sync (bool): whether to synchronize the codebook across processes. Defaults to True.

    References:
        [1] Oord, A., Vinyals, O., and kavukcuoglu, k. 2017.
        Neural Discrete Representation Learning.
        In Advances in Neural Information Processing Systems (pp. 6306–6315).
        Curran Associates, Inc..
    """

    # < Python 3.9 TorchScript requirement for ModuleList
    __constants__ = ["encoder", "quantizer", "decoder"]

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        no_levels: int = 3,
        downsample_parameters: Tuple[Tuple[int, int, int, int], ...] = ((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
        upsample_parameters: Tuple[Tuple[int, int, int, int, int], ...] = (
            (2, 4, 1, 1, 0),
            (2, 4, 1, 1, 0),
            (2, 4, 1, 1, 0),
        ),
        no_res_layers: int = 3,
        no_channels: int = 192,
        num_embeddings: int = 32,
        embedding_dim: int = 64,
        embedding_init: str = "normal",
        commitment_cost: float = 0.25,
        decay: float = 0.5,
        epsilon: float = 1e-5,
        adn_ordering: str = "NDA",
        dropout: Optional[Union[Tuple, str, float]] = 0.1,
        act: Optional[Union[Tuple, str]] = "RELU",
        output_act: Optional[Union[Tuple, str]] = None,
        ddp_sync:bool = True,
    ):
        super(VQVAE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims

        assert no_levels == len(downsample_parameters) and no_levels == len(upsample_parameters), (
            f"downsample_parameters, upsample_parameters must have the same number of elements as no_levels. "
            f"But got {len(downsample_parameters)} and {len(upsample_parameters)}, instead of {no_levels}."
        )

        self.no_levels = no_levels
        self.downsample_parameters = downsample_parameters
        self.upsample_parameters = upsample_parameters
        self.no_res_layers = no_res_layers
        self.no_channels = no_channels

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

    def get_ema_decay(self) -> float:
        return self.quantizer.get_ema_decay()

    def set_ema_decay(self, decay: float) -> float:
        self.quantizer.set_ema_decay(decay)

        return self.get_ema_decay()

    def get_commitment_cost(self) -> float:
        return self.quantizer.get_commitment_cost()

    def set_commitment_cost(self, commitment_factor: float) -> float:
        self.quantizer.set_commitment_cost(commitment_factor)

        return self.get_commitment_cost()

    def get_perplexity(self) -> float:
        return self.quantizer.get_perplexity()

    def construct_encoder(self) -> torch.nn.Sequential:
        encoder = []

        for idx in range(self.no_levels):
            encoder.append(
                Convolution(
                    spatial_dims=self.spatial_dims,
                    in_channels=self.in_channels if idx == 0 else self.no_channels // 2,
                    out_channels=self.no_channels // (2 if idx != self.no_levels - 1 else 1),
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

            encoder.append(
                nn.Sequential(
                    *[
                        ResidualUnit(
                            spatial_dims=self.spatial_dims,
                            no_channels=self.no_channels // (2 if idx != self.no_levels - 1 else 1),
                            no_res_channels=self.no_channels // (2 if idx != self.no_levels - 1 else 1),
                            adn_ordering=self.adn_ordering,
                            act=self.act,
                            dropout=self.dropout,
                            dropout_dim=1,
                            groups=1,
                            bias=True,
                        )
                        for _ in range(self.no_res_layers)
                    ]
                )
            )

        encoder.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=self.no_channels,
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
        decoder = [
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=self.embedding_dim,
                out_channels=self.no_channels,
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

        for idx in range(self.no_levels):
            decoder.append(
                nn.Sequential(
                    *[
                        ResidualUnit(
                            spatial_dims=self.spatial_dims,
                            no_channels=self.no_channels // (1 if idx == 0 else 2),
                            no_res_channels=self.no_channels // (1 if idx == 0 else 2),
                            adn_ordering=self.adn_ordering,
                            act=self.act,
                            dropout=self.dropout,
                            dropout_dim=1,
                            groups=1,
                            bias=True,
                        )
                        for _ in range(self.no_res_layers)
                    ]
                )
            )

            decoder.append(
                Convolution(
                    spatial_dims=self.spatial_dims,
                    in_channels=self.no_channels // (1 if idx == 0 else 2),
                    out_channels=self.out_channels if idx == self.no_levels - 1 else self.no_channels // 2,
                    strides=self.upsample_parameters[idx][0],
                    kernel_size=self.upsample_parameters[idx][1],
                    adn_ordering=self.adn_ordering,
                    act=self.act,
                    dropout=self.dropout if idx != self.no_levels - 1 else None,
                    norm=None,
                    dropout_dim=1,
                    dilation=self.upsample_parameters[idx][2],
                    groups=1,
                    bias=True,
                    conv_only=idx == self.no_levels - 1,
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

    def quantize(self, encodings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_loss, x = self.quantizer(encodings)
        return x, x_loss

    def decode(self, quantizations: torch.Tensor) -> torch.Tensor:
        return self.decoder(quantizations)

    def index_quantize(self, images: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantize(self.encode(images=images))

    def decode_samples(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.decode(self.quantizer.embed(embedding_indices))

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantizations, quantization_losses = self.quantize(self.encode(images))
        reconstruction = self.decode(quantizations)

        return reconstruction, quantization_losses
