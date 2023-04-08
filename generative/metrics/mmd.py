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

from collections.abc import Callable

import torch
from monai.metrics.metric import Metric


class MMDMetric(Metric):
    """
    Unbiased Maximum Mean Discrepancy (MMD) is a kernel-based method for measuring the similarity between two
    distributions. It is a non-negative metric where a smaller value indicates a closer match between the two
    distributions.

    Gretton, A., et al,, 2012.  A kernel two-sample test. The Journal of Machine Learning Research, 13(1), pp.723-773.

    Args:
        y_transform: Callable to transform the y tensor before computing the metric. It is usually a Gaussian or Laplace
            filter, but it can be any function that takes a tensor as input and returns a tensor as output such as a
            feature extractor or an Identity function.
        y_pred_transform: Callable to transform the y_pred tensor before computing the metric.
    """

    def __init__(self, y_transform: Callable | None = None, y_pred_transform: Callable | None = None) -> None:
        super().__init__()

        self.y_transform = y_transform
        self.y_pred_transform = y_pred_transform

    def __call__(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: first sample (e.g., the reference image). Its shape is (B,C,W,H) for 2D data and (B,C,W,H,D) for 3D.
            y_pred: second sample (e.g., the reconstructed image). It has similar shape as y.
        """

        # Beta and Gamma are not calculated since torch.mean is used at return
        beta = 1.0
        gamma = 2.0

        if self.y_transform is not None:
            y = self.y_transform(y)

        if self.y_pred_transform is not None:
            y_pred = self.y_pred_transform(y_pred)

        if y_pred.shape != y.shape:
            raise ValueError(
                "y_pred and y shapes dont match after being processed "
                f"by their transforms, received y_pred: {y_pred.shape} and y: {y.shape}"
            )

        for d in range(len(y.shape) - 1, 1, -1):
            y = y.squeeze(dim=d)
            y_pred = y_pred.squeeze(dim=d)

        y = y.view(y.shape[0], -1)
        y_pred = y_pred.view(y_pred.shape[0], -1)

        y_y = torch.mm(y, y.t())
        y_pred_y_pred = torch.mm(y_pred, y_pred.t())
        y_pred_y = torch.mm(y_pred, y.t())

        y_y = y_y / y.shape[1]
        y_pred_y_pred = y_pred_y_pred / y.shape[1]
        y_pred_y = y_pred_y / y.shape[1]

        # Ref. 1 Eq. 3 (found under Lemma 6)
        return beta * (torch.mean(y_y) + torch.mean(y_pred_y_pred)) - gamma * torch.mean(y_pred_y)
