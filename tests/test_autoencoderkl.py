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
from tests.utils import test_script_save

from generative.networks.nets import AutoencoderKL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TEST_CASE_0 = [
    {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "num_channels": 32,
        "latent_channels": 8,
        "ch_mult": [1, 1, 1],
        "num_res_blocks": 1,
    },
    (2, 1, 64, 64),
    (2, 1, 64, 64),
    (2, 8, 16, 16),
]

TEST_CASE_1 = [
    {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "num_channels": 32,
        "latent_channels": 32,
        "ch_mult": [1, 1, 1, 1],
        "num_res_blocks": 1,
    },
    (2, 1, 64, 64),
    (2, 1, 64, 64),
    (2, 32, 8, 8),
]

TEST_CASE_2 = [
    {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "num_channels": 32,
        "latent_channels": 32,
        "ch_mult": [1, 1, 1, 1],
        "num_res_blocks": 1,
        "attention_levels": (False, False, False, True),
        "with_encoder_nonlocal_attn": False,
    },
    (2, 1, 64, 64),
    (2, 1, 64, 64),
    (2, 32, 8, 8),
]

TEST_CASE_3 = [
    {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "num_channels": 32,
        "latent_channels": 32,
        "ch_mult": [1, 1, 1, 1],
        "num_res_blocks": 1,
        "attention_levels": (True, True, True, True),
        "with_encoder_nonlocal_attn": False,
        "with_decoder_nonlocal_attn": False,
    },
    (2, 1, 64, 64),
    (2, 1, 64, 64),
    (2, 32, 8, 8),
]

CASES = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]


class TestAutoEncoderKL(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape, expected_latent_shape):
        net = AutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result[0].shape, expected_shape)
            self.assertEqual(result[1].shape, expected_latent_shape)
            self.assertEqual(result[2].shape, expected_latent_shape)

    def test_script(self):
        input_param, input_shape, _, _ = CASES[0]
        net = AutoencoderKL(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)

    def test_model_channels_not_multiple_of_norm_num_group(self):
        with self.assertRaises(ValueError):
            AutoencoderKL(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_channels=24,
                latent_channels=8,
                ch_mult=[1, 1, 1],
                num_res_blocks=1,
                norm_num_groups=16,
            )

    def test_model_ch_mult_not_same_size_of_attention_levels(self):
        with self.assertRaises(ValueError):
            AutoencoderKL(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_channels=24,
                latent_channels=8,
                ch_mult=[1, 1, 1],
                num_res_blocks=1,
                norm_num_groups=16,
                attention_levels=(True,),
            )

    @parameterized.expand(CASES)
    def test_shape_reconstruction(self, input_param, input_shape, expected_shape, _):
        net = AutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.reconstruct(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
