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
from parameterized import parameterized
from generative.networks.nets import DiffusionModelUNet

CASE_WRONG = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "dropout_cattn": 3.0
        }
    ],
]

CASE_OK = [
            [
                {
                    "spatial_dims": 2,
                    "in_channels": 1,
                    "out_channels": 1,
                    "num_res_blocks": 1,
                    "num_channels": (8, 8, 8),
                    "attention_levels": (False, False, True),
                    "num_head_channels": 4,
                    "norm_num_groups": 8,
                    "with_conditioning": True,
                    "transformer_num_layers": 1,
                    "cross_attention_dim": 3,
                    "dropout_cattn": 0.25
                }
            ],
            [
                {
                    "spatial_dims": 2,
                    "in_channels": 1,
                    "out_channels": 1,
                    "num_res_blocks": 1,
                    "num_channels": (8, 8, 8),
                    "attention_levels": (False, False, True),
                    "num_head_channels": 4,
                    "norm_num_groups": 8,
                    "with_conditioning": True,
                    "transformer_num_layers": 1,
                    "cross_attention_dim": 3
                }
            ],
]

class TestDiffusionModelUNetDropout(unittest.TestCase):
    @parameterized.expand(CASE_WRONG)
    def test_wrong(self, input_param):
        with self.assertRaises(ValueError):
            net = DiffusionModelUNet(**input_param)

    @parameterized.expand(CASE_OK)
    def test_right(self, input_param):
        net = DiffusionModelUNet(**input_param)

if __name__ == "__main__":
    unittest.main()
