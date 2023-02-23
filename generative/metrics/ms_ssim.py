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

import torch
import torch.nn.functional as F
from monai.metrics import SSIMMetric
from monai.metrics.regression import RegressionMetric
from monai.utils import MetricReduction


class MSSSIM(RegressionMetric):
    """
    Computes Multi-Scale Structural Similarity Index Measure.

    [1] Wang, Z., Simoncelli, E.P. and Bovik, A.C., 2003, November.
            Multiscale structural similarity for image quality assessment.
            In The Thrity-Seventh Asilomar Conference on Signals, Systems
            & Computers, 2003 (Vol. 2, pp. 1398-1402). Ieee.

    Args:
        data_range: dynamic range of the data
        win_size: gaussian weighting window size
        k1: stability constant used in the luminance denominator
        k2: stability constant used in the contrast denominator
        spatial_dims: if 2, input shape is expected to be (B,C,W,H);
                      if 3, it is expected to be (B,C,W,H,D)
        weights: parameters for image similarity and contrast sensitivity
                 at different resolution scores.
        reduction: {``"none"``, ``"mean"``, ``"sum"``}
            Specifies the reduction to apply to the output.
            Defaults to ``"mean"``.
            - ``"none"``: no reduction will be applied.
            - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
            - ``"sum"``: the output will be summed.
    """

    def __init__(
        self,
        data_range: torch.Tensor | float,
        win_size: int = 7,
        k1: float = 0.01,
        k2: float = 0.03,
        spatial_dims: int = 2,
        weights: list | None = None,
        reduction: MetricReduction | str = MetricReduction.MEAN,
    ) -> None:
        super().__init__()

        if not (win_size % 2 == 1):
            raise ValueError("Window size should be odd.")

        self.data_range = data_range
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.spatial_dims = spatial_dims
        self.weights = weights
        self.reduction = reduction

        self.SSIM = SSIMMetric(self.data_range, self.win_size, self.k1, self.k2, self.spatial_dims)

    def _compute_metric(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: first sample (e.g., the reference image). Its shape is
                (B,C,W,H) for 2D data and (B,C,W,H,D) for 3D.
                A fastMRI sample should use the 2D format with C being
                the number of slices.
            y: second sample (e.g., the reconstructed image). It has similar
               shape as x

        """

        if not x.shape == y.shape:
            raise ValueError(f"Input images should have the same dimensions, but got {x.shape} and {y.shape}.")

        for d in range(len(x.shape) - 1, 1, -1):
            x = x.squeeze(dim=d)
            y = y.squeeze(dim=d)

        if len(x.shape) == 4:
            avg_pool = F.avg_pool2d
        elif len(x.shape) == 5:
            avg_pool = F.avg_pool3d
        else:
            raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {x.shape}")

        if self.weights is None:
            # as per Ref 1 - Sec 3.2.
            self.weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.weights = torch.tensor(self.weights)

        divisible_by = 2 ** (len(self.weights) - 1)
        bigger_than = (self.win_size + 2) * 2 ** (len(self.weights) - 1)
        for idx, shape_size in enumerate(x.shape[2:]):
            if shape_size % divisible_by != 0:
                raise ValueError(
                    f"Image size needs to be divisible by {divisible_by} but "
                    f"dimension {idx + 2} has size {shape_size}"
                )

            if shape_size < bigger_than:
                raise ValueError(
                    f"Image size should be larger than {bigger_than} due to "
                    f"the {len(self.weights) - 1} downsamplings in MS-SSIM."
                )

        levels = self.weights.shape[0]
        mcs_list: list[torch.Tensor] = []
        for i in range(levels):
            ssim, cs = self.SSIM._compute_metric_and_contrast(x, y)

            if i < levels - 1:
                mcs_list.append(torch.relu(cs))
                padding = [s % 2 for s in x.shape[2:]]
                x = avg_pool(x, kernel_size=2, padding=padding)
                y = avg_pool(y, kernel_size=2, padding=padding)

        ssim = torch.relu(ssim)  # (batch, 1)
        # (level, batch, 1)
        mcs_and_ssim = torch.stack(mcs_list + [ssim], dim=0)
        ms_ssim = torch.prod(mcs_and_ssim ** self.weights.view(-1, 1, 1), dim=0)

        if self.reduction == MetricReduction.MEAN.value:
            ms_ssim = ms_ssim.mean()
        elif self.reduction == MetricReduction.SUM.value:
            ms_ssim = ms_ssim.sum()
        elif self.reduction == MetricReduction.NONE.value:
            pass

        return ms_ssim
