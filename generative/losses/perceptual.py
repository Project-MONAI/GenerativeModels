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

from typing import Dict

import torch
import torch.nn as nn

# import torch.nn.functional as F
from lpips import LPIPS

# from torch.nn.modules.loss import _Loss


# TODO: Add MedicalNet for true 3D computation (https://github.com/Tencent/MedicalNet)
# TODO: Add RadImageNet for 2D computaion with networks pretrained using radiological images
#  (https://github.com/BMEII-AI/RadImageNet)
class PerceptualLoss(nn.Module):
    """
    Perceptual loss based on the lpips library. The 3D implementation is based on a 2.5D approach where we batchify
    every spatial dimension one after another so we obtain better spatial consistency. There is also a pixel
    component as well.

    Based on: Zhang, et al. "The unreasonable effectiveness of deep features as a perceptual metric."
    https://arxiv.org/abs/1801.03924

    Args:
        spatial_dims: number of spatial dimensions.
        is_fake_3d: whether we use 2.5D approach for a 3D perceptual loss
        fake_3d_n_slices: how many, as a ratio, slices we drop in the 2.5D approach

    References:
        [1] Zhang, R., Isola, P., Efros, A.A., Shechtman, E. and Wang, O., 2018.
        The unreasonable effectiveness of deep features as a perceptual metric.
        In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 586-595).
    """

    def __init__(
        self,
        spatial_dims: int,
        is_fake_3d: bool = True,
        fake_3d_n_slices: float = 0.0,
        lpips_kwargs: Dict = None,
        lpips_normalize: bool = True,
    ):
        super().__init__()

        if not (spatial_dims in [2, 3]):
            raise NotImplementedError("Perceptual loss is implemented only in 2D and 3D.")

        if spatial_dims == 3 and is_fake_3d is False:
            raise NotImplementedError("True 3D perceptual loss is not implemented yet.")

        self.spatial_dims = spatial_dims
        self.lpips_kwargs = (
            {
                "pretrained": True,
                "net": "alex",
                "version": "0.1",
                "lpips": True,
                "spatial": False,
                "pnet_rand": False,
                "pnet_tune": False,
                "use_dropout": True,
                "model_path": None,
                "eval_mode": True,
                "verbose": False,
            }
            if lpips_kwargs is None
            else lpips_kwargs
        )

        self.lpips_normalize = lpips_normalize
        self.perceptual_function = LPIPS(**self.lpips_kwargs) if self.dimensions == 2 or is_fake_3d else None

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.mean(self.perceptual_function.forward(input, target, normalize=self.lpips_normalize))

        return loss
