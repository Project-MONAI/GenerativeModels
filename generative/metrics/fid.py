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
#
# =========================================================================
# Adapted from https://github.com/photosynthesis-team/piq
# which has the following license:
# https://github.com/photosynthesis-team/piq/blob/master/LICENSE
#
# Copyright 2023 photosynthesis-team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import annotations

import torch
from monai.metrics.metric import Metric


class FID(Metric):
    """
    Frechet Inception Distance (FID). The FID calculates the distance between two distributions of feature vectors.
    Based on: Heusel M. et al. "Gans trained by a two time-scale update rule converge to a local nash equilibrium."
    https://arxiv.org/abs/1706.08500#. The inputs for this metric should be two groups of feature vectors (with format
    (number images, number of features)) extracted from the a pretrained network.

    Originally, it was proposed to use the activations of the pool_3 layer of an Inception v3 pretrained with Imagenet.
    However, others networks pretrained on medical datasets can be used as well (for example, RadImageNwt for 2D and
    MedicalNet for 3D images). If the chosen model output is not a scalar, usually it is used a global spatial
    average pooling.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        return get_fid_score(y_pred, y)


def get_fid_score(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y = y.float()
    y_pred = y_pred.float()

    if y.ndimension() > 2:
        raise ValueError("Inputs should have (number images, number of features) shape.")

    mu_y_pred = torch.mean(y_pred, dim=0)
    sigma_y_pred = _cov(y_pred, rowvar=False)
    mu_y = torch.mean(y, dim=0)
    sigma_y = _cov(y, rowvar=False)

    return compute_frechet_distance(mu_y_pred, sigma_y_pred, mu_y, sigma_y)


def _cov(m: torch.Tensor, rowvar: bool = True) -> torch.Tensor:
    """
    Estimate a covariance matrix of the variables.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations. Each row of `m` represents a variable,
            and each column a single observation of all those variables.
        rowvar: If rowvar is True (default), then each row represents a variable, with observations in the columns.
            Otherwise, the relationship is transposed: each column represents a variable, while the rows contain
            observations.
    """
    if m.dim() < 2:
        m = m.view(1, -1)

    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def _sqrtm_newton_schulz(matrix: torch.Tensor, num_iters: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Square root of matrix using Newton-Schulz Iterative method. Based on:
    https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py. Bechmark shown in:
    https://github.com/photosynthesis-team/piq/issues/190#issuecomment-742039303

    Args:
        matrix: matrix or batch of matrices
        num_iters: Number of iteration of the method

    """
    dim = matrix.size(0)
    norm_of_matrix = matrix.norm(p="fro")
    y_matrix = matrix.div(norm_of_matrix)
    i_matrix = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)
    z_matrix = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)

    s_matrix = torch.empty_like(matrix)
    error = torch.empty(1, device=matrix.device, dtype=matrix.dtype)

    for _ in range(num_iters):
        t = 0.5 * (3.0 * i_matrix - z_matrix.mm(y_matrix))
        y_matrix = y_matrix.mm(t)
        z_matrix = t.mm(z_matrix)

        s_matrix = y_matrix * torch.sqrt(norm_of_matrix)

        norm_of_matrix = torch.norm(matrix)
        error = matrix - torch.mm(s_matrix, s_matrix)
        error = torch.norm(error) / norm_of_matrix

        if torch.isclose(error, torch.tensor([0.0], device=error.device, dtype=error.dtype), atol=1e-5):
            break

    return s_matrix, error


def compute_frechet_distance(
    mu_x: torch.Tensor, sigma_x: torch.Tensor, mu_y: torch.Tensor, sigma_y: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """The Frechet distance between multivariate normal distributions."""
    diff = mu_x - mu_y
    covmean, _ = _sqrtm_newton_schulz(sigma_x.mm(sigma_y))

    # If calculation produces singular product, epsilon is added to diagonal of cov estimates
    if not torch.isfinite(covmean).all():
        offset = torch.eye(sigma_x.size(0), device=mu_x.device, dtype=mu_x.dtype) * epsilon
        covmean, _ = _sqrtm_newton_schulz((sigma_x + offset).mm(sigma_y + offset))

    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma_x) + torch.trace(sigma_y) - 2 * tr_covmean
