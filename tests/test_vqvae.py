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

from monai.networks import eval_mode
from generative.networks.nets.vqvae import VQVAE
from tests.utils import test_script_save

CASES_2D = CASES_3D = []
# Number of downsamplings
for no_levels in [2, 4]:
    # Embedding dimension
    for embedding_dim in [16, 64]:
        # Batch size
        for batch_size in [1, 3]:
            # Number of input channels
            for in_channels in [1, 3]:
                # Input shape
                for d1 in [64, 256]:
                    CASES_2D.append(
                        [
                            {
                                "spatial_dims": 2,
                                "in_channels": in_channels,
                                "out_channels": in_channels,
                                "no_levels": no_levels,
                                "downsample_parameters": [(2, 4, 1, 1)] * no_levels,
                                "upsample_parameters": [(2, 4, 1, 1, 0)] * no_levels,
                                "no_res_layers": 1,
                                "no_channels": 8,
                                "num_embeddings": 2048,
                                "embedding_dim": embedding_dim,
                                "embedding_init": "normal",
                                "commitment_cost": 0.25,
                                "decay": 0.5,
                                "epsilon": 1e-5,
                                "adn_ordering": "NDA",
                                "dropout": 0.1,
                                "act": "RELU",
                                "output_act": None,
                            },
                            (batch_size, in_channels, d1, d1),
                            (batch_size, in_channels, d1, d1),
                        ]
                    )

                    CASES_3D.append(
                        [
                            {
                                "spatial_dims": 3,
                                "in_channels": in_channels,
                                "out_channels": in_channels,
                                "no_levels": no_levels,
                                "downsample_parameters": [(2, 4, 1, 1)] * no_levels,
                                "upsample_parameters": [(2, 4, 1, 1, 0)] * no_levels,
                                "no_res_layers": 1,
                                "no_channels": 8,
                                "num_embeddings": 2048,
                                "embedding_dim": embedding_dim,
                                "embedding_init": "normal",
                                "commitment_cost": 0.25,
                                "decay": 0.5,
                                "epsilon": 1e-5,
                                "adn_ordering": "NDA",
                                "dropout": 0.1,
                                "act": "RELU",
                                "output_act": None,
                            },
                            (batch_size, in_channels, d1, d1, d1),
                            (batch_size, in_channels, d1, d1, d1),
                        ]
                    )


# 1-channel 2D, should fail because of number of levels, number of downsamplings, number of upsamplings mismatch.
TEST_CASE_FAIL = {
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 1,
    "no_levels": 3,
    "downsample_parameters": [(2, 4, 1, 1)] * 2,
    "upsample_parameters": [(2, 4, 1, 1, 0)] * 4,
    "no_res_layers": 1,
    "no_channels": 8,
    "num_embeddings": 2048,
    "embedding_dim": 32,
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
    "no_levels": 4,
    "downsample_parameters": [(2, 4, 1, 1)] * 4,
    "upsample_parameters": [(2, 4, 1, 1, 0)] * 4,
    "no_res_layers": 1,
    "no_channels": 8,
    "num_embeddings": 2048,
    "embedding_dim": 32,
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
    @parameterized.expand(CASES_2D + CASES_3D)
    def test_shape(self, input_param, input_shape, expected_shape):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(input_param)
        net = VQVAE(**input_param).to(device)
        with eval_mode(net):
            result, _ = net(torch.randn(input_shape).to(device))
        self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        net = VQVAE(
            **{
                "spatial_dims": 2,
                "in_channels": 1,
                "out_channels": 1,
                "no_levels": 4,
                "downsample_parameters": [(2, 4, 1, 1)] * 4,
                "upsample_parameters": [(2, 4, 1, 1, 0)] * 4,
                "no_res_layers": 1,
                "no_channels": 256,
                "num_embeddings": 2048,
                "embedding_dim": 32,
                "embedding_init": "normal",
                "commitment_cost": 0.25,
                "decay": 0.5,
                "epsilon": 1e-5,
                "adn_ordering": "NDA",
                "dropout": 0.1,
                "act": "RELU",
                "output_act": None,
                "ddp_sync": False,
            }
        )
        test_data = torch.randn(2, 1, 256, 256)
        test_script_save(net, test_data)

    def test_level_upsample_downsample_difference(self):
        with self.assertRaises(AssertionError):
            VQVAE(**TEST_CASE_FAIL)

    def test_latent_shape(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = VQVAE(**TEST_LATENT_SHAPE).to(device)
        test_data = torch.randn(2, 1, 256, 256).to(device)
        with eval_mode(net):
            latent = net.encode(test_data)

        self.assertEqual(latent.shape, (2, 32, 16, 16))


if __name__ == "__main__":
    unittest.main()
