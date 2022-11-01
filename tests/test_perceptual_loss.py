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

import torch
from parameterized import parameterized

from generative.losses.perceptual import PerceptualLoss

TEST_CASES = [
    [
        {"spatial_dims": 2, "network_type": "squeeze"},
        (2, 1, 64, 64),
        (2, 1, 64, 64),
    ],
    [
        {"spatial_dims": 3, "network_type": "squeeze", "is_fake_3d": True, "n_slices_per_axis": 20},
        (2, 1, 64, 64, 64),
        (2, 1, 64, 64, 64),
    ],
]


class TestTverskyLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_shape, target_shape):
        loss = PerceptualLoss(**input_param)
        result = loss(torch.randn(input_shape), torch.randn(target_shape))
        self.assertEqual(result.shape, torch.Size([]))

    @parameterized.expand(TEST_CASES)
    def test_identical_input(self, input_param, input_shape, target_shape):
        loss = PerceptualLoss(**input_param)
        tensor = torch.randn(input_shape)
        result = loss(tensor, tensor)
        self.assertEqual(result, torch.Tensor([0.0]))

    def test_true_3d(self):
        with self.assertRaises(NotImplementedError):
            PerceptualLoss(spatial_dims=3, is_fake_3d=False)

    def test_1d(self):
        with self.assertRaises(NotImplementedError):
            PerceptualLoss(spatial_dims=1)


if __name__ == "__main__":
    unittest.main()
