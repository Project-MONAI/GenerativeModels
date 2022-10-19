from typing import Optional, List, Union, Sequence, Tuple, Dict

import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.layers import Act

from generative.networks.blocks.residual_unit import ResidualUnit
from generative.networks.layers.vector_quantizer import VectorQuantizerEMA


class VQVAE(nn.Module):
    """
    Single bottleneck implementation of Vector-Quantised Variational Autoencoder (VQ-VAE).

    Args:
        dimensions (int): number of spatial dimensions.
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
        use_subpixel_conv (bool): Whether or not to use SubPixelConvolution for the last deconvolution.
            Defaults to True.
        use_slim_residual (bool): Whether or not to have the kernel of the last convolution in each residual unit
            be equal to 1. Default to True
        codebook_type (str): VectorQuantization module type between "ema", "gradient", "relaxation".
            Defaults to "ema".
        num_embeddings (Tuple[int]): VectorQuantization number of atomic elements in the codebook. Defaults to 32.
        embedding_dim (Tuple[int]): VectorQuantization number of channels of the input and atomic elements.
            Defaults to 64.
        commitment_cost (Tuple[float]): VectorQuantization commitment_cost. Defaults to 0.25 as per [1].
        decay (Tuple[float]): VectorQuantization decay. Defaults to 0.5.
        epsilon (Tuple[float]): VectorQuantization epsilon. Defaults to 1e-5 as per [1].
        adn_ordering (str): a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act Optional[Union[Tuple, str]]: activation type and arguments. Defaults to Relu.
        dropout (Optional[Union[Tuple, str, float]]): dropout ratio. Defaults to 0.1.

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
        dimensions: int,
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
        num_embeddings: Tuple[int] = (32,),
        embedding_dim: Tuple[int] = (64,),
        embedding_init: Tuple[str] = ("normal",),
        commitment_cost: Tuple[float] = (0.25,),
        decay: Tuple[float] = (0.5,),
        epsilon: Tuple[float] = (1e-5,),
        adn_ordering: str = "NDA",
        dropout: Optional[Union[Tuple, str, float]] = 0.1,
        act: Optional[Union[Tuple, str]] = "RELU",
        output_act: Optional[Union[Tuple, str]] = None,
    ):
        super(VQVAE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimensions = dimensions

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

        self.encoder = self.construct_encoder()
        self.quantizer = self.construct_quantizer()
        self.decoder = self.construct_decoder()

    def get_ema_decay(self) -> Sequence[float]:
        return [self.quantizer[0].get_ema_decay()]

    def set_ema_decay(self, decay: Union[Sequence[float], float], index: int = None) -> Sequence[float]:
        self.quantizer[0].set_ema_decay(decay[0] if isinstance(decay, list) else decay)

        return self.get_ema_decay()

    def get_commitment_cost(self) -> Sequence[float]:
        return [self.quantizer[0].get_commitment_cost()]

    def set_commitment_cost(self, commitment_factor: Union[Sequence[float], float]) -> Sequence[float]:
        self.quantizer[0].set_commitment_cost(
            commitment_factor[0] if isinstance(commitment_factor, list) else commitment_factor
        )

        return self.get_commitment_cost()

    def get_perplexity(self) -> Sequence[float]:
        return [self.quantizer[0].get_perplexity()]

    def construct_encoder(self) -> nn.ModuleList:
        encoder = []

        for idx in range(self.no_levels):
            encoder.append(
                Convolution(
                    spatial_dims=self.dimensions,
                    in_channels=1 if idx == 0 else self.no_channels // 2,
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
                            spatial_dims=self.dimensions,
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
                spatial_dims=self.dimensions,
                in_channels=self.no_channels,
                out_channels=self.embedding_dim[0],
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

        return nn.ModuleList(encoder)

    def construct_quantizer(self) -> nn.ModuleList:
        # TODO: Include lucidrains' vector quantizer as an option
        quantizer = VectorQuantizerEMA(
            dimensions=self.dimensions,
            num_embeddings=self.num_embeddings[0],
            embedding_dim=self.embedding_dim[0],
            commitment_cost=self.commitment_cost[0],
            decay=self.decay[0],
            epsilon=self.epsilon[0],
            embedding_init=self.embedding_init[0],
        )

        return nn.ModuleList([quantizer])

    def construct_decoder(self) -> nn.ModuleList:
        decoder = [
            Convolution(
                spatial_dims=self.dimensions,
                in_channels=self.embedding_dim[0],
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
                            spatial_dims=self.dimensions,
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
                    spatial_dims=self.dimensions,
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

        return nn.ModuleList(decoder)

    def get_last_layer(self) -> nn.parameter.Parameter:
        if self.use_subpixel_conv:
            return list(self.decoder.modules())[-2 if self.output_act else -1].conv_block.weight
        else:
            return list(self.decoder.modules())[-2 if self.output_act else -1].weight

    def encode(self, images: torch.Tensor) -> List[torch.Tensor]:
        x = images

        for step in self.encoder:
            x = step(x)

        return [x]

    def quantize(self, encodings: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x_loss, x = self.quantizer[0](encodings[0])
        return [x], [x_loss]

    def decode(self, quantizations: List[torch.Tensor]) -> torch.Tensor:
        x = quantizations[0]

        for step in self.decoder:
            x = step(x)

        return x

    def index_quantize(self, images: torch.Tensor) -> List[torch.Tensor]:
        encodings = self.encode(images=images)
        _, _, encoding_indices = self.quantizer[0].quantize(encodings[0])
        return [encoding_indices]

    def decode_samples(self, embedding_indices: List[torch.Tensor]) -> torch.Tensor:
        samples_codes = self.quantizer[0].embed(embedding_indices[0])
        samples_images = self.decode([samples_codes])

        return samples_images

    def forward(self, images: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        encodings = self.encode(images)
        quantizations, quantization_losses = self.quantize(encodings)
        reconstruction = self.decode(quantizations)

        return {"reconstruction": [reconstruction], "quantization_losses": quantization_losses}
