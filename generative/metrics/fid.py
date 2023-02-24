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

import numpy as np
from scipy import linalg

import torch
from monai.metrics.metric import Metric


class FID(Metric):
    """
    Frechet Inception Distance (FID). The FID calculates the distance between two distributions of feature vectors.
    Based on: Heusel M. et al. "Gans trained by a two time-scale update rule converge to a local nash equilibrium."
    https://arxiv.org/abs/1706.08500#. The inputs for this metric should be two groups of feature vectors (with format
    (number images, number of features)) extracted from a pretrained network.

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

    if y.ndimension() > 2:
        raise ValueError("Inputs should have (number images, number of features) shape.")

    # transform the ys into np arrays
    y = y.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    mu_y_pred = np.mean(y_pred, axis=0)
    sigma_y_pred = np.cov(y_pred, rowvar=False)
    mu_y = np.mean(y, axis=0)
    sigma_y = np.cov(y, rowvar=False)

    return compute_frechet_distance(mu_y_pred, sigma_y_pred, mu_y, sigma_y)


def compute_frechet_distance(
    mu_x: np.ndarray, sigma_x: np.ndarray, mu_y: np.ndarray, sigma_y: np.ndarray, epsilon: float = 1e-6
) -> np.float:
    """The Frechet distance between multivariate normal distributions.
    This implementation is based on https://github.com/mseitzer/pytorch-fid/blob/0a754fb8e66021700478fd365b79c2eaa316e31b/src/pytorch_fid/fid_score.py
     """

    mu_x = np.atleast_1d(mu_x)
    mu_y = np.atleast_1d(mu_y)

    sigma_x = np.atleast_2d(sigma_x)
    sigma_y = np.atleast_2d(sigma_y)

    assert mu_x.shape == mu_y.shape, \
        'Synthetic and real mean vectors have different lengths'
    assert sigma_x.shape == sigma_y.shape, \
        'Synthetic and real covariances have different dimensions'

    diff = mu_x - mu_y

    # Product might be almost singular
    import pdb
    pdb.set_trace()
    covmean, _ = linalg.sqrtm(sigma_x.dot(sigma_y), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % epsilon
        print(msg)
        offset = np.eye(sigma_x.shape[0]) * epsilon
        covmean = linalg.sqrtm((sigma_x + offset).dot(sigma_y + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma_x)
            + np.trace(sigma_y) - 2 * tr_covmean)
