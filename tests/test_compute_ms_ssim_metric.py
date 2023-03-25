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

import unittest

import numpy as np
import torch
from parameterized import parameterized

from generative.metrics import MSSSIM

TEST_CASES = [
    [{"data_range": 1.0}, {"x": torch.ones([3, 3, 144, 144]) / 2, "y": torch.ones([3, 3, 144, 144]) / 2}, 1.0],
    [
        {"data_range": torch.tensor(1.0)},
        {"x": torch.ones([3, 3, 144, 144]) / 2, "y": torch.ones([3, 3, 144, 144]) / 2},
        1.0,
    ],
    [
        {"data_range": torch.tensor(1.0), "spatial_dims": 3},
        {"x": torch.ones([3, 3, 144, 144, 144]) / 2, "y": torch.ones([3, 3, 144, 144, 144]) / 2},
        1.0,
    ],
]


class TestMSSSIMMetric(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_results(self, input_param, input_data, expected_val):
        results = MSSSIM(**input_param)._compute_metric(**input_data)
        np.testing.assert_allclose(results.detach().cpu().numpy(), expected_val, rtol=1e-4)

    def test_win_size_not_odd(self):
        with self.assertRaises(ValueError):
            MSSSIM(data_range=1.0, win_size=8)

    def test_if_inputs_different_shapes(self):
        with self.assertRaises(ValueError):
            MSSSIM(data_range=1.0)(torch.ones([3, 3, 144, 144]), torch.ones([3, 3, 145, 145]))

    def test_wrong_shape(self):
        with self.assertRaises(ValueError):
            MSSSIM(data_range=1.0)(torch.ones([3, 144, 144]), torch.ones([3, 144, 144]))

    def test_input_too_small(self):
        with self.assertRaises(ValueError):
            MSSSIM(data_range=1.0)(torch.ones([3, 3, 8, 8]), torch.ones([3, 3, 8, 8]))

    def test_input_non_divisible(self):
        with self.assertRaises(ValueError):
            MSSSIM(data_range=1.0)(torch.ones([3, 3, 149, 149]), torch.ones([3, 3, 149, 149]))


if __name__ == "__main__":
    unittest.main()
