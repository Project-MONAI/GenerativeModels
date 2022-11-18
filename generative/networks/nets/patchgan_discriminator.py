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

from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from monai.networks.blocks import Convolution


class MultiScalePatchDiscriminator(nn.Sequential):
    """
    Multi-scale Patch-GAN discriminator based on Pix2PixHD:
    High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
    Ting-Chun Wang1, Ming-Yu Liu1, Jun-Yan Zhu2, Andrew Tao1, Jan Kautz1, Bryan Catanzaro (1)
    (1) NVIDIA Corporation, 2UC Berkeley
    In CVPR 2018.
    Multi-Scale discriminator made up of several Patch-GAN discriminators, that process the images
    up to different spatial scales.
    Args:
        num_D: number of discriminators
        num_layers_D: number of Convolution layers (Conv + activation + normalisation + [dropout]) in each
        of the discriminators. In each layer, the number of channels are doubled and the spatial size is
        divided by 2.
        spatial_dims: number of spatial dimensions (1D, 2D etc.)
        num_channels: number of filters in the first convolutional layer (double of the value is taken from then on)
        in_channels: number of input channels
        out_channels: number of output channels in each discriminator
        kernel_size: kernel size of the convolution layers
        activation: activation layer type
        norm: normalisation type
        bias: introduction of layer bias
        dropout: proportion of dropout applied, defaults to 0.
        minimum_size_im: minimum spatial size of the input image. Introduced to make sure the architecture
        requested isn't going to downsample the input image beyond value of 1.
    """

    def __init__(
        self,
        num_D,
        num_layers_D,
        spatial_dims: int,
        num_channels: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        minimum_size_im: int = 256,
    ) -> None:
        super().__init__()
        self.num_D = num_D
        self.num_layers_D = num_layers_D
        self.num_channels = num_channels
        self.padding = tuple([int((kernel_size - 1) / 2)] * spatial_dims)
        for i in range(self.num_D):
            subnetD = PatchDiscriminator(
                self.num_layers_D,
                spatial_dims=spatial_dims,
                num_channels=self.num_channels,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                norm=norm,
                bias=bias,
                padding=self.padding,
                index_d=i,
                dropout=dropout,
                minimum_size_im=minimum_size_im,
            )
            self.add_module("discriminator_%d" % i, subnetD)

    def forward(self, i) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """

        Args:
            i: Input tensor
        Returns:
            list of outputs and another list of lists with the intermediate features
            of each discriminator.
        """

        out: List[torch.Tensor] = []
        intermediate_features: List[List[torch.Tensor]] = []
        for D in self.children():
            out_D: List[torch.Tensor] = D(i)
            out.append(out_D[-1])
            intermediate_features.append(out_D[:-1])

        return out, intermediate_features


class PatchDiscriminator(nn.Sequential):
    """
    Patch-GAN discriminator based on Pix2PixHD:
    High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
    Ting-Chun Wang1, Ming-Yu Liu1, Jun-Yan Zhu2, Andrew Tao1, Jan Kautz1, Bryan Catanzaro (1)
    (1) NVIDIA Corporation, 2UC Berkeley
    In CVPR 2018.
    Args:
        num_layers_D: number of Convolution layers (Conv + activation + normalisation + [dropout]) in each
        of the discriminators. In each layer, the number of channels are doubled and the spatial size is
        divided by 2.
        spatial_dims: number of spatial dimensions (1D, 2D etc.)
        num_channels: number of filters in the first convolutional layer (double of the value is taken from then on)
        in_channels: number of input channels
        out_channels: number of output channels in each discriminator
        kernel_size: kernel size of the convolution layers
        activation: activation layer type
        norm: normalisation type
        bias: introduction of layer bias
        padding: padding to be applied to the convolutional layers
        index_d: index of the discriminator in the multi-scale chain, defaults to 0
        dropout: proportion of dropout applied, defaults to 0.
        minimum_size_im: minimum spatial size of the input image. Introduced to make sure the architecture
        requested isn't going to downsample the input image beyond value of 1.
    """

    def __init__(
        self,
        num_layers_D: int,
        spatial_dims: int,
        num_channels: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        padding: Sequence[int],
        index_d: int = 0,
        dropout: Union[float, tuple] = 0.0,
        minimum_size_im: int = 256,
    ) -> None:
        super().__init__()
        self.num_layers_D = num_layers_D * (index_d + 1)
        self.num_channels = num_channels
        output_size = float(minimum_size_im) / (2 ** (num_layers_D * (index_d + 1)))
        if output_size < 1:
            raise AssertionError(
                "Your image size is too small to take in up to %d discriminators with num_layers = %d."
                "Please reduce num_layers, reduce num_D or enter bigger images." % (index_d, self.num_layers_D)
            )

        input_channels = in_channels
        output_channels = num_channels * 2
        for l_ in range(self.num_layers_D):
            layer = Convolution(
                spatial_dims=spatial_dims,
                kernel_size=kernel_size,
                in_channels=input_channels,
                out_channels=output_channels,
                act=activation,
                bias=bias,
                norm=norm,
                dropout=dropout,
                padding=padding,
                strides=2,
            )
            self.add_module("%d" % l_, layer)
            input_channels = output_channels
            output_channels = output_channels * 2

        self.add_module(
            "final_conv",
            Convolution(
                spatial_dims=spatial_dims,
                kernel_size=1,
                in_channels=input_channels,
                out_channels=out_channels,
                act=activation,
                bias=bias,
                norm=norm,
                dropout=dropout,
                strides=1,
            ),
        )

    def forward(self, x) -> List[torch.Tensor]:
        """
        Args:
            x: input tensor
            feature-matching loss (regulariser loss) on the discriminators as well (see Pix2Pix paper).
        Returns:
            list of intermediate features, with the last element being the output.
        """
        out = [x]
        for submodel in self.children():
            intermediate_output = submodel(out[-1])
            out.append(intermediate_output)

        return out[1:]
