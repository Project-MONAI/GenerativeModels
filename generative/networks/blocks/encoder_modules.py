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

from functools import partial

import torch
import torch.nn as nn
from monai.networks.blocks import Convolution

__all__ = ["SpatialRescaler"]


class SpatialRescaler(nn.Module):
    """
        SpatialRescaler based on https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/encoders/modules.py

        Args:
            n_stages: number of interpolation stages.
            method: algorithm used for sampling.
            multiplier: multiplier for spatial size. If scale_factor is a tuple,
                its length has to match the number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            bias: whether to have a bias term.
    """

    def __init__(self,
                 spatial_dims: int = 2,
                 n_stages: int = 1,
                 method: str = 'bilinear',
                 multiplier: float = 0.5,
                 in_channels: int = 3,
                 out_channels: int = None,
                 bias: bool = False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                conv_only=True,
                bias=bias
            )

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)
