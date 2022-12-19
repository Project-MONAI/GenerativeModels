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

from generative.networks.nets.vqvae import VQVAE

TEST_CASES = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_levels": 2,
            "downsample_parameters": [(2, 4, 1, 1)] * 2,
            "upsample_parameters": [(2, 4, 1, 1, 0)] * 2,
            "num_res_layers": 1,
            "num_channels": [8, 8],
            "num_res_channels": [8, 8],
            "num_embeddings": 16,
            "embedding_dim": 8,
            "embedding_init": "normal",
            "commitment_cost": 0.25,
            "decay": 0.5,
            "epsilon": 1e-5,
            "adn_ordering": "NDA",
            "dropout": 0.1,
            "act": "RELU",
            "output_act": None,
        },
        (1, 1, 16, 16),
        (1, 1, 16, 16),
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_levels": 2,
            "downsample_parameters": [(2, 4, 1, 1)] * 2,
            "upsample_parameters": [(2, 4, 1, 1, 0)] * 2,
            "num_res_layers": 1,
            "num_channels": [8, 8],
            "num_res_channels": [8, 8],
            "num_embeddings": 16,
            "embedding_dim": 8,
            "embedding_init": "normal",
            "commitment_cost": 0.25,
            "decay": 0.5,
            "epsilon": 1e-5,
            "adn_ordering": "NDA",
            "dropout": 0.1,
            "act": "RELU",
            "output_act": None,
        },
        (1, 1, 16, 16, 16),
        (1, 1, 16, 16, 16),
    ],
]

# 1-channel 2D, should fail because of number of levels, number of downsamplings, number of upsamplings mismatch.
TEST_CASE_FAIL = {
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 1,
    "num_levels": 3,
    "downsample_parameters": [(2, 4, 1, 1)] * 2,
    "upsample_parameters": [(2, 4, 1, 1, 0)] * 4,
    "num_res_layers": 1,
    "num_channels": [8, 8],
    "num_res_channels": [8, 8],
    "num_embeddings": 16,
    "embedding_dim": 8,
    "embedding_init": "normal",
    "commitment_cost": 0.25,
    "decay": 0.5,
    "epsilon": 1e-5,
    "adn_ordering": "NDA",
    "dropout": 0.1,
    "act": "RELU",
    "output_act": None,
}

TEST_LATENT_SHAPE = {
    "spatial_dims": 2,
    "in_channels": 1,
    "out_channels": 1,
    "num_levels": 2,
    "downsample_parameters": [(2, 4, 1, 1)] * 2,
    "upsample_parameters": [(2, 4, 1, 1, 0)] * 2,
    "num_res_layers": 1,
    "num_channels": [8, 8],
    "num_res_channels": [8, 8],
    "num_embeddings": 16,
    "embedding_dim": 8,
    "embedding_init": "normal",
    "commitment_cost": 0.25,
    "decay": 0.5,
    "epsilon": 1e-5,
    "adn_ordering": "NDA",
    "dropout": 0.1,
    "act": "RELU",
    "output_act": None,
}


class TestVQVAE(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        net = VQVAE(**input_param).to(device)

        with eval_mode(net):
            result, _ = net(torch.randn(input_shape).to(device))

        self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        net = VQVAE(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_levels=2,
            downsample_parameters=tuple([(2, 4, 1, 1)] * 2),
            upsample_parameters=tuple([(2, 4, 1, 1, 0)] * 2),
            num_res_layers=1,
            num_channels=[8, 8],
            num_res_channels=[8, 8],
            num_embeddings=16,
            embedding_dim=8,
            embedding_init="normal",
            commitment_cost=0.25,
            decay=0.5,
            epsilon=1e-5,
            adn_ordering="NDA",
            dropout=0.1,
            act="RELU",
            output_act=None,
            ddp_sync=False,
        )
        test_data = torch.randn(1, 1, 16, 16)
        test_script_save(net, test_data)

    def test_level_upsample_downsample_difference(self):
        with self.assertRaises(AssertionError):
            VQVAE(**TEST_CASE_FAIL)

    def test_encode_shape(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        net = VQVAE(**TEST_LATENT_SHAPE).to(device)

        with eval_mode(net):
            latent = net.encode(torch.randn(1, 1, 32, 32).to(device))

        self.assertEqual(latent.shape, (1, 8, 8, 8))

    def test_index_quantize_shape(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        net = VQVAE(**TEST_LATENT_SHAPE).to(device)

        with eval_mode(net):
            latent = net.index_quantize(torch.randn(1, 1, 32, 32).to(device))

        self.assertEqual(latent.shape, (1, 8, 8))

    def test_decode_shape(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        net = VQVAE(**TEST_LATENT_SHAPE).to(device)

        with eval_mode(net):
            latent = net.decode(torch.randn(1, 8, 8, 8).to(device))

        self.assertEqual(latent.shape, (1, 1, 32, 32))

    def test_decode_samples_shape(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        net = VQVAE(**TEST_LATENT_SHAPE).to(device)

        with eval_mode(net):
            latent = net.decode_samples(torch.randint(low=0, high=16, size=(1, 8, 8)).to(device))

        self.assertEqual(latent.shape, (1, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
