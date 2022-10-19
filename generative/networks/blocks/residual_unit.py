from typing import Optional, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.convolutions import Convolution


class ResidualUnit(nn.Module):
    """
    Implementation of the ResidualLayer used in the VQVAE network.

    Args:
        spatial_dims: number of spatial dimensions of the input data.
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
    def __int__(
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
            adn_ordering=self.adni_ordering,
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
            adn_ordering=self.adni_ordering,
            act=None,
            norm=None,
            dropout=None,
            dropout_dim=self.dropout_dim,
            bias=self.bias,
        )

    def forward(self, x):
        return F.relu(x + self.conv2(self.conv1(x)), True)
