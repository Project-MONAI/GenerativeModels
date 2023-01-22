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

# =========================================================================
# Adapted from https://github.com/photosynthesis-team/piq
# which has the following license:
# https://github.com/photosynthesis-team/piq/blob/master/LICENSE

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
from torchvision.models import Inception_V3_Weights, inception_v3


class FID(Metric):
    """
    Frechet Inception Distance (FID). FID can compare two data distributions with different number of samples.
    But dimensionalities should match, otherwise it won't be possible to correctly compute statistics. Based on:
    Heusel M. et al. "Gans trained by a two time-scale update rule converge to a local nash equilibrium."
    https://arxiv.org/abs/1706.08500#

    """

    def __init__(
        self,
        feature_extractor_type: str | None = "imagenet",
    ) -> None:
        super().__init__()
        self.feature_extractor_type = feature_extractor_type
        self.feature_extractor = None

        if feature_extractor_type:
            if feature_extractor_type == "imagenet":
                # TODO: Add feature extractor
                weights = Inception_V3_Weights.IMAGENET1K_V1
                self.feature_extractor = inception_v3(weights=weights).eval()
            elif feature_extractor_type == "radimagenet":
                # TODO: Add feature extractor
                self.feature_extractor = inception_v3().eval()
            elif feature_extractor_type == "medicalnet":
                # TODO: Add feature extractor
                self.feature_extractor = inception_v3().eval()

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        if self.feature_extractor_type in ["radimagenet", "imagenet"] and (
            y_pred.ndimension() != 4 or y.ndimension() != 4
        ):
            raise ValueError("FID requires RGB images.")

        if self.feature_extractor_type == "medicalnet" and (y_pred.ndimension() != 5 or y.ndimension() != 5):
            raise ValueError("FID requires RGB images.")

        if y_pred.ndimension() < 2 or y.ndimension() < 2:
            raise ValueError("y_pred should have at least two dimensions.")

        if self.feature_extractor:
            y_pred_features = self.feature_extractor.features(y_pred)
            y_features = self.feature_extractor.features(y)
        else:
            y_pred_features = y_pred
            y_features = y

        mu_y_pred, sigma_y_pred = compute_statistics(y_pred_features)
        mu_y, sigma_y = compute_statistics(y_features)

        return compute_frechet_distance(mu_y_pred, sigma_y_pred, mu_y, sigma_y)


def _sqrtm_newton_schulz(matrix: torch.Tensor, num_iters: int = 100) -> torch.Tensor:
    """
    Square root of matrix using Newton-Schulz Iterative method. Based on:
    https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py

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
        T = 0.5 * (3.0 * i_matrix - z_matrix.mm(y_matrix))
        y_matrix = y_matrix.mm(T)
        z_matrix = T.mm(z_matrix)

        s_matrix = y_matrix * torch.sqrt(norm_of_matrix)

        norm_of_matrix = torch.norm(matrix)
        error = matrix - torch.mm(s_matrix, s_matrix)
        error = torch.norm(error) / norm_of_matrix

        if torch.isclose(error, torch.tensor([0.0], device=error.device, dtype=error.dtype), atol=1e-5):
            break

    return s_matrix, error


def compute_statistics(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the statistics used by FID

    Args:
        samples:  Low-dimension representation of image set.
            Shape (N_samples, dims) and dtype: np.float32 in range 0 - 1

    Returns:
        mu: mean over all activations from the encoder.
        sigma: covariance matrix over all activations from the encoder.
    """
    mu = torch.mean(samples, dim=0)

    # Estimate a covariance matrix
    if samples.dim() < 2:
        samples = samples.view(1, -1)

    if samples.size(0) != 1:
        samples = samples.t()

    fact = 1.0 / (samples.size(1) - 1)
    samples = samples - torch.mean(samples, dim=1, keepdim=True)
    samplest = samples.t()
    sigma = fact * samples.matmul(samplest).squeeze()

    return mu, sigma


def compute_frechet_distance(mu_x, sigma_x, mu_y, sigma_y, eps=1e-6):
    """
    The Frechet distance between two multivariate Gaussians
    """

    diff = mu_x - mu_y
    covmean, _ = _sqrtm_newton_schulz(sigma_x.mm(sigma_y))

    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        offset = torch.eye(sigma_x.size(0), device=mu_x.device, dtype=mu_x.dtype) * eps
        covmean, _ = _sqrtm_newton_schulz((sigma_x + offset).mm(sigma_y + offset))

    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma_x) + torch.trace(sigma_y) - 2 * tr_covmean
