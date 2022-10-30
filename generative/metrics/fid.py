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

from typing import Tuple, Union

import torch
from monai.metrics import CumulativeIterationMetric
from monai.utils import MetricReduction
from torchvision.models import Inception_V3_Weights, inception_v3

RADIMAGENET_URL = "https://drive.google.com/uc?id=1p0q9AhG3rufIaaUE1jc2okpS8sdwN6PU"
RADIMAGENET_WEIGHTS = "RadImageNet-InceptionV3_notop.h5"


# TODO: get a better name for parameters
# TODO: Transform radimagenet's Keras weight to Torch weights following https://github.com/BMEII-AI/RadImageNet/issues/3
# TODO: Create Mednet3D
class FID(CumulativeIterationMetric):
    """
    Frechet Inception Distance (FID). FID can compare two data distributions with different number of samples.
    But dimensionalities should match, otherwise it won't be possible to correctly compute statistics. Based on:
    Heusel M. et al. "Gans trained by a two time-scale update rule converge to a local nash equilibrium."
    https://arxiv.org/abs/1706.08500#

    Args:
        reduction:
        extract_features:
        feature_extractor:
    """

    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        extract_features: bool = True,
        feature_extractor: str = "imagenet",
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.feature_extractor = feature_extractor

        # TODO: Download pretrained network.
        self.network = None
        if extract_features:
            if feature_extractor == "imagenet:":
                self.network = inception_v3(Inception_V3_Weights.IMAGENET1K_V1)
            elif feature_extractor == "radimagenet":
                self.network = inception_v3()
            elif feature_extractor == "medicalnet":
                pass

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        Args:
            y_pred:
            y:
        """
        pass

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):
        """
        Args:
            reduction:

        Returns:
        """
        pass


def _sqrtm_newton_schulz(matrix: torch.Tensor, num_iters: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Square root of matrix using Newton-Schulz Iterative method. Based on:
    https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py

    Args:
        matrix: matrix or batch of matrices
        num_iters: Number of iteration of the method

    Returns:
        Square root of matrix
        Error
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


def _compute_statistics(samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


def compute_fid_from_features(x_features: torch.Tensor, y_features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Fits multivariate Gaussians, then computes FID.

    Args:
        x_features: Samples from data distribution. Shape :math:`(N_x, D)`
        y_features: Samples from data distribution. Shape :math:`(N_y, D)`
        eps:

    Returns:
        The Frechet Distance.
    """

    mu_x, sigma_x = _compute_statistics(x_features)
    mu_y, sigma_y = _compute_statistics(y_features)

    # The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    # and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).
    diff = mu_x - mu_y
    covmean, _ = _sqrtm_newton_schulz(sigma_x.mm(sigma_y))

    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        offset = torch.eye(sigma_x.size(0), device=mu_x.device, dtype=mu_x.dtype) * eps
        covmean, _ = _sqrtm_newton_schulz((sigma_x + offset).mm(sigma_y + offset))

    tr_covmean = torch.trace(covmean)
    score = diff.dot(diff) + torch.trace(sigma_x) + torch.trace(sigma_y) - 2 * tr_covmean

    return score
