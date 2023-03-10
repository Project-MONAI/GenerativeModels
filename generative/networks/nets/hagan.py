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

from collections.abc import Sequence

import torch
from monai.networks.blocks import Convolution
from monai.networks.layers import Act, Norm
from torch import nn
from torch.nn import functional as F


class SNConv(nn.Module):
    """
    Spectral Normalization Convolution Layer

    Args:
        spatial_dims: number of spatial dimensions of the input data.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: size of the convolving kernel.
        strides: stride of the convolution.
        padding: implicit paddings on both sides of the input.
        bias: whether to add a learnable bias to the output.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int = 3,
        strides: Sequence[int] | int = 1,
        padding: Sequence[int] | int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            bias=bias,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.utils.parametrizations.spectral_norm(self.conv.conv)(x)


class SNLinear(nn.Module):
    """
    Spectral Normalization Linear Layer

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        bias: whether to add a learnable bias to the output.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.utils.parametrizations.spectral_norm(self.linear)(x)


class CodeDiscriminator(nn.Module):
    """
    Code Discriminator to force the distribution of the code to be indistinguishable from that of random noise

    Args:
        code_size: size of the code.
        num_units: number of units in the hidden layers.
    """
    def __init__(self, code_size: int, num_units: int = 256) -> None:
        super().__init__()

        self.layer_1 = SNLinear(code_size, num_units)
        self.layer_2 = SNLinear(num_units, num_units)
        self.layer_3 = SNLinear(num_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.layer_1(x), negative_slope=0.2)
        x = F.leaky_relu(self.layer_2(x), negative_slope=0.2)
        x = self.layer_3(x)

        return x


class SubEncoder(nn.Module):
    """
    Sub encoder

    Args:
        spatial_dims: number of spatial dimensions of the input data.
        num_channels: number of input channels.
        latent_dim: size of the latent code.
    """
    def __init__(self, spatial_dims: int, num_channels: int = 256, latent_dim: int = 1024) -> None:
        super().__init__()

        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels // 4,
            out_channels=num_channels // 8,
            kernel_size=4,
            strides=2,
            padding=1,
            adn_ordering="NA",
            act=Act.RELU,
            norm=(Norm.GROUP, {"num_groups": 8, "num_channels": num_channels // 8}),
        )
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels // 8,
            out_channels=num_channels // 4,
            kernel_size=4,
            strides=2,
            padding=1,
            adn_ordering="NA",
            act=Act.RELU,
            norm=(Norm.GROUP, {"num_groups": 8, "num_channels": num_channels // 4}),
        )
        self.conv3 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels // 4,
            out_channels=num_channels // 2,
            kernel_size=4,
            strides=2,
            padding=1,
            adn_ordering="NA",
            act=Act.RELU,
            norm=(Norm.GROUP, {"num_groups": 8, "num_channels": num_channels // 2}),
        )
        self.conv4 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels // 2,
            out_channels=num_channels,
            kernel_size=4,
            strides=2,
            padding=1,
            adn_ordering="NA",
            act=Act.RELU,
            norm=(Norm.GROUP, {"num_groups": 8, "num_channels": num_channels}),
        )
        self.conv5 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels,
            out_channels=latent_dim,
            kernel_size=4,
            strides=1,
            padding=0,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class Encoder(nn.Module):
    """
    Encoder

    Args:
        spatial_dims: number of spatial dimensions of the input data.
        num_channels: number of input channels.
    """
    def __init__(self, spatial_dims: int, num_channels: int = 64) -> None:
        super().__init__()

        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=num_channels // 2,
            kernel_size=4,
            strides=2,
            padding=1,
            adn_ordering="NA",
            act=Act.RELU,
            norm=(Norm.GROUP, {"num_groups": 8, "num_channels": num_channels // 2}),
        )
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels // 2,
            out_channels=num_channels // 2,
            kernel_size=3,
            strides=1,
            padding=1,
            adn_ordering="NA",
            act=Act.RELU,
            norm=(Norm.GROUP, {"num_groups": 8, "num_channels": num_channels // 2}),
        )
        self.conv3 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels // 2,
            out_channels=num_channels,
            kernel_size=4,
            strides=2,
            padding=1,
            adn_ordering="NA",
            act=Act.RELU,
            norm=(Norm.GROUP, {"num_groups": 8, "num_channels": num_channels // 2}),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class SubDiscriminator(nn.Module):
    """
    Sub discriminator for the low resolution images.

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        num_channels: base number of channels for the convolutional layers.
        num_class: number of classes for the conditional HA-GAN.
    """
    def __init__(self, spatial_dims: int, num_channels: int =256, num_class:int =0) -> None:
        super().__init__()
        self.channel = num_channels
        self.num_class = num_class

        self.conv1 = SNConv(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=num_channels // 8,
            kernel_size=4,
            strides=2,
            padding=1,
        )  # in:[64,64,64], out:[32,32,32]
        self.conv2 = SNConv(
            spatial_dims=spatial_dims,
            in_channels=num_channels // 8,
            out_channels=num_channels // 4,
            kernel_size=4,
            strides=2,
            padding=1,
        )  # out:[16,16,16]
        self.conv3 = SNConv(
            spatial_dims=spatial_dims,
            in_channels=num_channels // 4,
            out_channels=num_channels // 2,
            kernel_size=4,
            strides=2,
            padding=1,
        )  # out:[8,8,8]
        self.conv4 = SNConv(
            spatial_dims=spatial_dims,
            in_channels=num_channels // 2,
            out_channels=num_channels,
            kernel_size=4,
            strides=2,
            padding=1,
        )  # out:[4,4,4]
        self.conv5 = SNConv(
            spatial_dims=spatial_dims,
            in_channels=num_channels,
            out_channels=1 + num_class,
            kernel_size=4,
            strides=1,
            padding=0,
        )  # out:[1,1,1,1]

    def forward(self, x: torch.Tensor) -> torch.Tensor | Sequence[torch.Tensor, torch.Tensor]:
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        if self.num_class == 0:
            x = self.conv5(x).view((-1, 1))
            return x
        else:
            x = self.conv5(x).view((-1, 1 + self.num_class))
            return x[:, :1], x[:, 1:]


class Discriminator(nn.Module):
    """
    Discriminator network of the Hierarchical Amortized GAN (HA-GAN) model for 3D medical image generation.

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        num_channels: base number of channels for the convolutional layers.
        num_class: number of classes for the conditional HA-GAN.
    """
    def __init__(self, spatial_dims: int, num_class: int = 0, num_channels: int = 512) -> None:
        super().__init__()
        self.channel = num_channels
        self.num_class = num_class

        # D^H
        self.conv1 = SNConv(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=num_channels // 32,
            kernel_size=4,
            strides=2,
            padding=1,
        )  # in:[32,256,256], out:[16,128,128]
        self.conv2 = SNConv(
            spatial_dims=spatial_dims,
            in_channels=num_channels // 32,
            out_channels=num_channels // 16,
            kernel_size=4,
            strides=2,
            padding=1,
        )  # out:[8,64,64,64]
        self.conv3 = SNConv(
            spatial_dims=spatial_dims,
            in_channels=num_channels // 16,
            out_channels=num_channels // 8,
            kernel_size=4,
            strides=2,
            padding=1,
        )  # out:[4,32,32,32]
        self.conv4 = SNConv(
            spatial_dims=spatial_dims,
            in_channels=num_channels // 8,
            out_channels=num_channels // 4,
            kernel_size=(2, 4, 4),
            strides=(2, 2, 2),
            padding=(0, 1, 1),
        )  # out:[2,16,16,16]
        self.conv5 = SNConv(
            spatial_dims=spatial_dims,
            in_channels=num_channels // 4,
            out_channels=num_channels // 2,
            kernel_size=(2, 4, 4),
            strides=(2, 2, 2),
            padding=(0, 1, 1),
        )  # out:[1,8,8,8]
        self.conv6 = SNConv(
            spatial_dims=spatial_dims,
            in_channels=num_channels // 2,
            out_channels=num_channels,
            kernel_size=(1, 4, 4),
            strides=(1, 2, 2),
            padding=(0, 1, 1),
        )  # out:[1,4,4,4]
        self.conv7 = SNConv(
            num_channels, num_channels // 4, kernel_size=(1, 4, 4), strides=1, padding=0
        )  # out:[1,1,1,1]
        self.fc1 = SNLinear(num_channels // 4 + 1, num_channels // 8)
        self.fc2 = SNLinear(num_channels // 8, 1)
        if num_class > 0:
            self.fc2_class = SNLinear(num_channels // 8, num_class)

        # D^L
        self.sub_discriminator = SubDiscriminator(num_class)

    def forward(
        self, h: torch.Tensor, h_small: torch.Tensor, crop_idx
    ) -> torch.Tensor | Sequence[torch.Tensor, torch.Tensor]:
        h = F.leaky_relu(self.conv1(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv5(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv6(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv7(h), negative_slope=0.2).squeeze()
        h = torch.cat([h, (crop_idx / 224.0 * torch.ones((h.size(0), 1))).cuda()], 1)  # 256*7/8
        h = F.leaky_relu(self.fc1(h), negative_slope=0.2)
        h_logit = self.fc2(h)
        if self.num_class > 0:
            h_class_logit = self.fc2_class(h)

            h_small_logit, h_small_class_logit = self.sub_discriminator(h_small)
            return (h_logit + h_small_logit) / 2.0, (h_class_logit + h_small_class_logit) / 2.0
        else:
            h_small_logit = self.sub_discriminator(h_small)
            return (h_logit + h_small_logit) / 2.0


class SubGenerator(nn.Module):
    """
    Sub Generator for generating the low resolution images.

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        num_channels: base number of channels for the convolutional layers.
        norm_num_groups: number of groups for the group normalization.
    """
    def __init__(self, spatial_dims: int, num_channels: int = 16, norm_num_groups: int = 8) -> None:
        super().__init__()

        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels * 4,
            out_channels=num_channels * 2,
            kernel_size=3,
            strides=1,
            padding=1,
            adn_ordering="NA",
            act=Act.RELU,
            norm=(Norm.GROUP, {"num_groups": norm_num_groups, "num_channels": num_channels * 2}),
        )
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels * 2,
            out_channels=num_channels,
            kernel_size=3,
            strides=1,
            padding=1,
            adn_ordering="NA",
            act=Act.RELU,
            norm=(Norm.GROUP, {"num_groups": norm_num_groups, "num_channels": num_channels}),
        )
        self.conv3 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels,
            out_channels=1,
            kernel_size=3,
            strides=1,
            padding=1,
            adn_ordering="A",
            act=Act.TANH,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Generator(nn.Module):
    """
    Generator network of the Hierarchical Amortized GAN (HA-GAN) model for 3D medical image generation.

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        num_channels: base number of channels for the convolutional layers.
        num_latent_channels: number of channels for the latent space.
        num_class: number of classes for the conditional HA-GAN.
        mode: mode of the network, can be "train" or "eval".
        norm_num_groups: number of groups for the group normalization.
    """
    def __init__(
        self,
            spatial_dims: int,
            num_channels: int = 32,
            num_latent_channels: int = 1024,
            num_class: int = 0,
            mode: str = "train",
            norm_num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.relu = nn.ReLU()
        self.num_class = num_class

        # Common block (G^A) and high-resolution block (G^H) of the Generator
        self.fc1 = nn.Linear(num_latent_channels + num_class, 4 * 4 * 4 * num_channels * 16)

        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels * 16,
            out_channels=num_channels * 16,
            kernel_size=3,
            strides=1,
            padding=1,
            adn_ordering="NA",
            act=Act.RELU,
            norm=(Norm.GROUP, {"num_groups": norm_num_groups, "num_channels": num_channels * 16}),
        )
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels * 16,
            out_channels=num_channels * 16,
            kernel_size=3,
            strides=1,
            padding=1,
            adn_ordering="NA",
            act=Act.RELU,
            norm=(Norm.GROUP, {"num_groups": norm_num_groups, "num_channels": num_channels * 16}),
        )
        self.conv3 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels * 16,
            out_channels=num_channels * 8,
            kernel_size=3,
            strides=1,
            padding=1,
            adn_ordering="NA",
            act=Act.RELU,
            norm=(Norm.GROUP, {"num_groups": norm_num_groups, "num_channels": num_channels * 8}),
        )
        self.conv4 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels * 8,
            out_channels=num_channels * 4,
            kernel_size=3,
            strides=1,
            padding=1,
            adn_ordering="NA",
            act=Act.RELU,
            norm=(Norm.GROUP, {"num_groups": norm_num_groups, "num_channels": num_channels * 4}),
        )
        self.conv5 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels * 4,
            out_channels=num_channels * 2,
            kernel_size=3,
            strides=1,
            padding=1,
            adn_ordering="NA",
            act=Act.RELU,
            norm=(Norm.GROUP, {"num_groups": norm_num_groups, "num_channels": num_channels * 2}),
        )
        self.conv6 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels * 2,
            out_channels=num_channels,
            kernel_size=3,
            strides=1,
            padding=1,
            adn_ordering="NA",
            act=Act.RELU,
            norm=(Norm.GROUP, {"num_groups": norm_num_groups, "num_channels": num_channels}),
        )

        self.conv7 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels,
            out_channels=1,
            kernel_size=3,
            strides=1,
            padding=1,
            adn_ordering="A",
            act=Act.TANH,
        )

        # Low-resolution block (G^L) of the generator
        self.sub_generator = SubGenerator(num_channels=num_channels // 2)

    def forward(self, h: torch.Tensor, crop_idx=None, class_label=None) -> torch.Tensor | Sequence[torch.Tensor, torch.Tensor]:
        # Generate from random noise
        if crop_idx != None or self.mode == "eval":
            if self.num_class > 0:
                h = torch.cat((h, class_label), dim=1)

            h = self.fc1(h)

            h = h.view(-1, 512, 4, 4, 4)
            h = self.conv1(h)

            h = F.interpolate(h, scale_factor=2)
            h = self.conv2(h)

            h = F.interpolate(h, scale_factor=2)
            h = self.conv3(h)

            h = F.interpolate(h, scale_factor=2)
            h = self.conv4(h)

            h = F.interpolate(h, scale_factor=2)
            h_latent = self.conv5(h)

            if self.mode == "train":
                h_small = self.sub_generator(h_latent)
                h = h_latent[:, :, crop_idx // 4 : crop_idx // 4 + 8, :, :]  # Crop, out: (8, 64, 64)
            else:
                h = h_latent

        # Generate from latent feature
        h = F.interpolate(h, scale_factor=2)
        h = self.conv6(h)

        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv7(h)

        if crop_idx != None and self.mode == "train":
            return h, h_small

        return h
