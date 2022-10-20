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

from generative.networks.nets import DiffusionModelUNet

UNCOND_CASES_2D = [
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
            "num_heads": -1,
            "num_head_channels": 1,
        },
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
    ],
]

UNCOND_CASES_3D = [
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "model_channels": 16,
            "out_channels": 1,
            "num_res_blocks": 1,
            "attention_resolutions": [16, 8],
            "channel_mult": [1, 1, 1, 1],
            "num_heads": 1,
            "norm_num_groups": 16,
        },
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "model_channels": 16,
            "out_channels": 1,
            "num_res_blocks": 1,
            "attention_resolutions": [16, 8],
            "channel_mult": [1, 1, 1, 1],
            "num_heads": -1,
            "num_head_channels": 1,
            "norm_num_groups": 16,
        },
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "model_channels": 16,
            "out_channels": 1,
            "num_res_blocks": 1,
            "attention_resolutions": [16, 8],
            "channel_mult": [1, 1, 1, 1],
            "num_heads": 1,
            "num_head_channels": 1,
            "legacy": False,
            "norm_num_groups": 16,
        },
    ],
]


class TestDiffusionModelUNet2D(unittest.TestCase):
    @parameterized.expand(UNCOND_CASES_2D)
    def test_shape_unconditioned_models(self, input_param):
        net = DiffusionModelUNet(**input_param)
        with eval_mode(net):
            result = net.forward(torch.rand((1, 1, 32, 64)), torch.randint(0, 1000, (1,)).long())
            self.assertEqual(result.shape, (1, 1, 32, 64))

    def test_shape_with_different_in_channel_out_channel(self):
        in_channels = 6
        out_channels = 3
        net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=in_channels,
            model_channels=32,
            out_channels=out_channels,
            num_res_blocks=1,
            attention_resolutions=[16, 8],
            channel_mult=[1, 1, 1, 1],
            num_heads=1,
        )
        with eval_mode(net):
            result = net.forward(torch.rand((1, in_channels, 64, 64)), torch.randint(0, 1000, (1,)).long())
            self.assertEqual(result.shape, (1, out_channels, 64, 64))

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

    def test_shape_conditioned_models(self):
        net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            model_channels=32,
            out_channels=1,
            num_res_blocks=1,
            attention_resolutions=[16, 8],
            channel_mult=[1, 1, 1, 1],
            num_heads=1,
            use_spatial_transformer=True,
            transformer_depth=1,
            context_dim=3,
        )
        with eval_mode(net):
            result = net.forward(
                x=torch.rand((1, 1, 32, 64)),
                timesteps=torch.randint(0, 1000, (1,)).long(),
                context=torch.rand((1, 1, 3)),
            )
            self.assertEqual(result.shape, (1, 1, 32, 64))


class TestDiffusionModelUNet3D(unittest.TestCase):
    @parameterized.expand(UNCOND_CASES_3D)
    def test_shape_unconditioned_models(self, input_param):
        net = DiffusionModelUNet(**input_param)
        with eval_mode(net):
            result = net.forward(torch.rand((1, 1, 16, 32, 48)), torch.randint(0, 1000, (1,)).long())
            self.assertEqual(result.shape, (1, 1, 16, 32, 48))

    def test_shape_with_different_in_channel_out_channel(self):
        in_channels = 6
        out_channels = 3
        net = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=in_channels,
            model_channels=16,
            out_channels=out_channels,
            num_res_blocks=1,
            attention_resolutions=[16, 8],
            channel_mult=[1, 1, 1, 1],
            num_heads=1,
            norm_num_groups=16,
        )
        with eval_mode(net):
            result = net.forward(torch.rand((1, in_channels, 16, 32, 48)), torch.randint(0, 1000, (1,)).long())
            self.assertEqual(result.shape, (1, out_channels, 16, 32, 48))

    def test_shape_conditioned_models(self):
        net = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=1,
            model_channels=16,
            out_channels=1,
            num_res_blocks=1,
            attention_resolutions=[16, 8],
            channel_mult=[1, 1, 1, 1],
            num_heads=1,
            norm_num_groups=16,
            use_spatial_transformer=True,
            transformer_depth=1,
            context_dim=3,
        )
        with eval_mode(net):
            result = net.forward(
                x=torch.rand((1, 1, 16, 32, 48)),
                timesteps=torch.randint(0, 1000, (1,)).long(),
                context=torch.rand((1, 1, 3)),
            )
            self.assertEqual(result.shape, (1, 1, 16, 32, 48))


if __name__ == "__main__":
    unittest.main()
