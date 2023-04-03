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
from monai.networks import eval_mode
from parameterized import parameterized

from generative.networks.blocks.selfattention import SABlock

TEST_CASE_SABLOCK = [
    [
        {"hidden_size": 16, "num_heads": 8, "dropout_rate": 0.2, "causal": False, "sequence_length": None},
        (2, 4, 16),
        (2, 4, 16),
    ],
    [
        {"hidden_size": 16, "num_heads": 8, "dropout_rate": 0.2, "causal": True, "sequence_length": 4},
        (2, 4, 16),
        (2, 4, 16),
    ],
]


class TestResBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASE_SABLOCK)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = SABlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            SABlock(hidden_size=12, num_heads=4, dropout_rate=6.0)

        with self.assertRaises(ValueError):
            SABlock(hidden_size=12, num_heads=4, dropout_rate=-6.0)

        with self.assertRaises(ValueError):
            SABlock(hidden_size=20, num_heads=8, dropout_rate=0.4)

        with self.assertRaises(ValueError):
            SABlock(hidden_size=12, num_heads=4, dropout_rate=0.4, causal=True, sequence_length=None)


if __name__ == "__main__":
    unittest.main()
