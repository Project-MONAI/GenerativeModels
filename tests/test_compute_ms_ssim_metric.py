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


import unittest

import numpy as np
import torch
from parameterized import parameterized

from generative.metrics import MS_SSIM

TEST_CASES = [
    [  # shape: (1, 2, 2, 3)
        {"data_range": torch.tensor(1.0)},
        {
            "x": torch.Tensor([[[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
            "y": torch.Tensor([[[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
        },
        1.0,
    ],
    [  # shape: (1, 1, 2, 2, 3)
        {
            "data_range": torch.tensor(1.0),
            "spatial_dims": 3,
        },
        {
            "x": torch.Tensor([[[[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]]),
            "y": torch.Tensor([[[[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]]),
        },
        1.0,
    ],
]


class TestMSSSIMMetric(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_results(self, input_param, input_data, expected_val):
        results = MS_SSIM(**input_param).forward(**input_data)
        np.testing.assert_allclose(results.detach().cpu().numpy(), expected_val, rtol=1e-4)

    def test_2D_shape(self):
        results = MS_SSIM(spatial_dims=2, reduction="none").forward(**TEST_CASES[0][1])
        self.assertEqual(results.shape, (1, 2, 2, 3))

    def test_3D_shape(self):
        results = MS_SSIM(spatial_dims=3, reduction="none").forward(**TEST_CASES[1][1])
        self.assertEqual(results.shape, (1, 2, 2, 2, 3))


if __name__ == "__main__":
    unittest.main()
