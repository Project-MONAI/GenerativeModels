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
from monai.networks import eval_mode
from parameterized import parameterized

# from tests.utils import TorchImageTestCase2D, TorchImageTestCase3D
from tests.utils import TorchImageTestCase2D

from generative.networks.nets import DiffusionModelUNet

UNCOND_CASES = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "model_channels": 32,
            "out_channels": 1,
            "num_res_blocks": 1,
            "attention_resolutions": [16, 8],
            "channel_mult": [1, 1, 1, 1],
            "num_heads": 1,
        },
        (1, 1, 64, 64),
        (1,),
        (1, 1, 64, 64),
    ],
    # [
    #     {
    #         "spatial_dims": 2,
    #         "in_channels": 6,
    #         "model_channels": 32,
    #         "out_channels": 3,
    #         "num_res_blocks": 1,
    #         "attention_resolutions": [16, 8],
    #         "channel_mult": [1, 1, 1, 1],
    #         "num_heads": 1,
    #     },
    #     (1, 6, 64, 64),
    #     (1,),
    #     (1, 3, 64, 64),
    # ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "model_channels": 32,
            "out_channels": 1,
            "num_res_blocks": 1,
            "attention_resolutions": [16, 8],
            "channel_mult": [1, 1, 1, 1],
            "num_heads": -1,
            "num_head_channels": 1,
        },
        (1, 1, 64, 64),
        (1,),
        (1, 1, 64, 64),
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "model_channels": 32,
            "out_channels": 1,
            "num_res_blocks": 1,
            "attention_resolutions": [16, 8],
            "channel_mult": [1, 1, 1, 1],
            "num_heads": 4,
            "num_head_channels": 2,
            "legacy": False,
        },
        (1, 1, 64, 64),
        (1,),
        (1, 1, 64, 64),
    ],
    # [
    #     {
    #         "spatial_dims": 3,
    #         "in_channels": 3,
    #         "model_channels": 32,
    #         "out_channels": 3,
    #         "num_res_blocks": 1,
    #         "attention_resolutions": [16, 8],
    #         "channel_mult": [1, 1, 1, 1],
    #         "num_heads": 1,
    #     },
    #     (1, 3, 32, 32, 32),
    #     (1,),
    #     (1, 3, 32, 32, 32),
    # ],
    # [
    #     {
    #         "spatial_dims": 3,
    #         "in_channels": 6,
    #         "model_channels": 32,
    #         "out_channels": 3,
    #         "num_res_blocks": 1,
    #         "attention_resolutions": [16, 8],
    #         "channel_mult": [1, 1, 1, 1],
    #         "num_heads": 1,
    #     },
    #     (1, 6, 32, 32, 32),
    #     (1,),
    #     (1, 3, 32, 32, 32),
    # ],
    # [
    #     {
    #         "spatial_dims": 3,
    #         "in_channels": 3,
    #         "model_channels": 32,
    #         "out_channels": 3,
    #         "num_res_blocks": 1,
    #         "attention_resolutions": [16, 8],
    #         "channel_mult": [1, 1, 1, 1],
    #         "num_heads": -1,
    #         "num_head_channels": 1,
    #     },
    #     (1, 3, 32, 32, 32),
    #     (1,),
    #     (1, 3, 32, 32, 32),
    # ],
    # [
    #     {
    #         "spatial_dims": 3,
    #         "in_channels": 3,
    #         "model_channels": 32,
    #         "out_channels": 3,
    #         "num_res_blocks": 1,
    #         "attention_resolutions": [16, 8],
    #         "channel_mult": [1, 1, 1, 1],
    #         "num_heads": 1,
    #         "num_head_channels": 1,
    #         "legacy": False,
    #     },
    #     (1, 3, 32, 32, 32),
    #     (1,),
    #     (1, 3, 32, 32, 32),
    # ]
]


class TestDiffusionModelUNet2D(TorchImageTestCase2D):
    @parameterized.expand(UNCOND_CASES)
    def test_shape_unconditioned_models(self, input_param, input_shape, timestep_shape, expected_shape):
        net = DiffusionModelUNet(**input_param)
        with eval_mode(net):
            result = net.forward(self.imt, torch.randint(0, 1000, timestep_shape).long())
            expected_shape = (1, self.input_channels, self.im_shape[0], self.im_shape[1])
            self.assertEqual(result.shape, expected_shape)

    def test_attention_heads_not_declared(self):
        with self.assertRaises(ValueError):
            DiffusionModelUNet(
                spatial_dims=2,
                in_channels=3,
                model_channels=32,
                out_channels=3,
                num_res_blocks=1,
                attention_resolutions=[16, 8],
                channel_mult=[1, 1, 1, 1],
                num_heads=-1,
                num_head_channels=-1,
            )

    def test_model_channels_not_multiple_of_norm_num_group(self):
        with self.assertRaises(ValueError):
            DiffusionModelUNet(
                spatial_dims=2,
                in_channels=3,
                model_channels=40,
                out_channels=3,
                num_res_blocks=1,
                attention_resolutions=[16, 8],
                channel_mult=[1, 1, 1, 1],
                norm_num_groups=32,
            )


if __name__ == "__main__":
    unittest.main()
