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

import torch
from parameterized import parameterized

from generative.networks.blocks import SpatialRescaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CASES = [
    [
        {
            "spatial_dims": 2,
            "n_stages": 1,
            "method": "bilinear",
            "multiplier": 0.5,
            "in_channels": None,
            "out_channels": None,
        },
        (1, 1, 16, 16),
        (1, 1, 8, 8),
    ],
    [
        {
            "spatial_dims": 2,
            "n_stages": 1,
            "method": "bilinear",
            "multiplier": 0.5,
            "in_channels": 3,
            "out_channels": 2,
        },
        (1, 3, 16, 16),
        (1, 2, 8, 8),
    ],
    [
        {
            "spatial_dims": 3,
            "n_stages": 1,
            "method": "trilinear",
            "multiplier": 0.5,
            "in_channels": None,
            "out_channels": None,
        },
        (1, 1, 16, 16, 16),
        (1, 1, 8, 8, 8),
    ],
    [
        {
            "spatial_dims": 3,
            "n_stages": 1,
            "method": "trilinear",
            "multiplier": 0.5,
            "in_channels": 3,
            "out_channels": 2,
        },
        (1, 3, 16, 16, 16),
        (1, 2, 8, 8, 8),
    ],
    [
        {
            "spatial_dims": 3,
            "n_stages": 1,
            "method": "trilinear",
            "multiplier": (0.25, 0.5, 0.75),
            "in_channels": 3,
            "out_channels": 2,
        },
        (1, 3, 20, 20, 20),
        (1, 2, 5, 10, 15),
    ],
    [
        {"spatial_dims": 2, "n_stages": 1, "size": (8, 8), "method": "bilinear", "in_channels": 3, "out_channels": 2},
        (1, 3, 16, 16),
        (1, 2, 8, 8),
    ],
    [
        {
            "spatial_dims": 3,
            "n_stages": 1,
            "size": (8, 8, 8),
            "method": "trilinear",
            "in_channels": None,
            "out_channels": None,
        },
        (1, 1, 16, 16, 16),
        (1, 1, 8, 8, 8),
    ],
]


class TestSpatialRescaler(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        module = SpatialRescaler(**input_param).to(device)

        result = module(torch.randn(input_shape).to(device))
        self.assertEqual(result.shape, expected_shape)

    def test_method_not_in_available_options(self):
        with self.assertRaises(AssertionError):
            SpatialRescaler(method="none")

    def test_n_stages_is_negative(self):
        with self.assertRaises(AssertionError):
            SpatialRescaler(n_stages=-1)

    def test_use_size_but_n_stages_is_not_one(self):
        with self.assertRaises(ValueError):
            SpatialRescaler(n_stages=2, size=[8, 8, 8])

    def test_both_size_and_multiplier_defined(self):
        with self.assertRaises(ValueError):
            SpatialRescaler(size=[1, 2, 3], multiplier=0.5)


if __name__ == "__main__":
    unittest.main()
