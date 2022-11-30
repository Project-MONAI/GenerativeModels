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

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import VQVAE, AutoencoderKL, DiffusionModelUNet
from generative.schedulers import DDPMScheduler

TEST_CASES = [
    [
        "AutoencoderKL",
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_channels": 8,
            "latent_channels": 3,
            "ch_mult": [1, 1, 1],
            "attention_levels": [False, False, False],
            "num_res_blocks": 1,
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "norm_num_groups": 8,
        },
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "out_channels": 3,
            "model_channels": 8,
            "norm_num_groups": 8,
            "attention_resolutions": [8],
            "num_res_blocks": 1,
            "channel_mult": [1, 1, 1],
            "num_heads": 1,
        },
        (1, 1, 32, 32),
        (1, 3, 8, 8),
    ],
    [
        "VQVAE",
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_channels": 8,
            "latent_channels": 3,
            "ch_mult": [1, 1, 1],
            "attention_levels": [False, False, False],
            "num_res_blocks": 1,
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "norm_num_groups": 8,
        },
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "out_channels": 3,
            "model_channels": 8,
            "norm_num_groups": 8,
            "attention_resolutions": [8],
            "num_res_blocks": 1,
            "channel_mult": [1, 1, 1],
            "num_heads": 1,
        },
        (1, 1, 32, 32),
        (1, 3, 8, 8),
    ],
]


class TestDiffusionSamplingInferer(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_prediction_shape(self, model_type, stage_1_params, stage_2_params, input_shape, latent_shape):
        if model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**stage_1_params)
        if model_type == "VQVAE":
            stage_1 = VQVAE(**stage_1_params)
        stage_2 = DiffusionModelUNet(**stage_2_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_1.train()
        input = torch.randn(input_shape).to(device)
        noise = torch.randn(latent_shape).to(device)
        scheduler = DDPMScheduler(
            num_train_timesteps=10,
        )
        inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)
        prediction = inferer(inputs=input, stage_1_model=stage_1, diffusion_model=stage_2, noise=noise)
        self.assertEqual(prediction.shape, latent_shape)

    @parameterized.expand(TEST_CASES)
    def test_sample_shape(self, stage_1_params, stage_2_params, input_shape, latent_shape):
        stage_1 = AutoencoderKL(**stage_1_params)
        stage_2 = DiffusionModelUNet(**stage_2_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_1.train()
        noise = torch.randn(latent_shape).to(device)
        scheduler = DDPMScheduler(
            num_train_timesteps=10,
        )
        inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)
        sample = inferer.sample(input_noise=noise, stage_1_model=stage_1, diffusion_model=stage_2, scheduler=scheduler)
        self.assertEqual(sample.shape, input_shape)


if __name__ == "__main__":
    unittest.main()
