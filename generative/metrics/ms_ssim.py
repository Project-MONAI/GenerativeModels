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
import torch.nn.functional as F
from monai.metrics.regression import RegressionMetric
from monai.utils import MetricReduction, StrEnum, ensure_tuple_rep

from generative.metrics.ssim import compute_ssim_and_cs


class KernelType(StrEnum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"


class MultiScaleSSIMMetric(RegressionMetric):
    """
    Computes the Multi-Scale Structural Similarity Index Measure (MS-SSIM).

    [1] Wang, Z., Simoncelli, E.P. and Bovik, A.C., 2003, November.
            Multiscale structural similarity for image quality assessment.
            In The Thirty-Seventh Asilomar Conference on Signals, Systems
            & Computers, 2003 (Vol. 2, pp. 1398-1402). Ieee.

    Args:
        spatial_dims: number of spatial dimensions of the input images.
        data_range: value range of input images. (usually 1.0 or 255)
        kernel_type: type of kernel, can be "gaussian" or "uniform".
        kernel_size: size of kernel
        kernel_sigma: standard deviation for Gaussian kernel.
        k1: stability constant used in the luminance denominator
        k2: stability constant used in the contrast denominator
        weights: parameters for image similarity and contrast sensitivity at different resolution scores.
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans)
    """

    def __init__(
        self,
        spatial_dims: int,
        data_range: float = 1.0,
        kernel_type: KernelType | str = KernelType.GAUSSIAN,
        kernel_size: int | Sequence[int, ...] = 11,
        kernel_sigma: float | Sequence[float, ...] = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        weights: Sequence[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)

        self.spatial_dims = spatial_dims
        self.data_range = data_range
        self.kernel_type = kernel_type

        if not isinstance(kernel_size, Sequence):
            kernel_size = ensure_tuple_rep(kernel_size, spatial_dims)
        self.kernel_size = kernel_size

        if not isinstance(kernel_sigma, Sequence):
            kernel_sigma = ensure_tuple_rep(kernel_sigma, spatial_dims)
        self.kernel_sigma = kernel_sigma

        self.k1 = k1
        self.k2 = k2
        self.weights = weights

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted image.
                It must be a 2D or 3D batch-first tensor [B,C,H,W] or [B,C,H,W,D].
            y: Reference image.
                It must be a 2D or 3D batch-first tensor [B,C,H,W] or [B,C,H,W,D].

        Raises:
            ValueError: when `y_pred` is not a 2D or 3D image.
        """
        dims = y_pred.ndimension()
        if self.spatial_dims == 2 and dims != 4:
            raise ValueError(
                f"y_pred should have 4 dimensions (batch, channel, height, width) when using {self.spatial_dims} "
                f"spatial dimensions, got {dims}."
            )

        if self.spatial_dims == 3 and dims != 5:
            raise ValueError(
                f"y_pred should have 4 dimensions (batch, channel, height, width, depth) when using {self.spatial_dims}"
                f" spatial dimensions, got {dims}."
            )

        # check if image have enough size for the number of downsamplings and the size of the kernel
        weights_div = max(1, (len(self.weights) - 1)) ** 2
        y_pred_spatial_dims = y_pred.shape[2:]
        for i in range(len(y_pred_spatial_dims)):
            if y_pred_spatial_dims[i] // weights_div <= self.kernel_size[i] - 1:
                raise ValueError(
                    f"For a given number of `weights` parameters {len(self.weights)} and kernel size "
                    f"{self.kernel_size[i]}, the image height must be larger than "
                    f"{(self.kernel_size[i] - 1) * weights_div}."
                )

        weights = torch.tensor(self.weights, device=y_pred.device, dtype=torch.float)

        avg_pool = getattr(F, f"avg_pool{self.spatial_dims}d")

        multiscale_list: list[torch.Tensor] = []
        for _ in range(len(weights)):
            ssim, cs = compute_ssim_and_cs(
                y_pred=y_pred,
                y=y,
                spatial_dims=self.spatial_dims,
                data_range=self.data_range,
                kernel_type=self.kernel_type,
                kernel_size=self.kernel_size,
                kernel_sigma=self.kernel_sigma,
                k1=self.k1,
                k2=self.k2,
            )

            cs_per_batch = cs.view(cs.shape[0], -1).mean(1)

            multiscale_list.append(torch.relu(cs_per_batch))
            y_pred = avg_pool(y_pred, kernel_size=2)
            y = avg_pool(y, kernel_size=2)

        ssim = ssim.view(ssim.shape[0], -1).mean(1)
        multiscale_list[-1] = torch.relu(ssim)
        multiscale_list = torch.stack(multiscale_list)

        ms_ssim_value_full_image = torch.prod(multiscale_list ** weights.view(-1, 1), dim=0)

        ms_ssim_per_batch: torch.Tensor = ms_ssim_value_full_image.view(ms_ssim_value_full_image.shape[0], -1).mean(
            1, keepdim=True
        )

        return ms_ssim_per_batch
