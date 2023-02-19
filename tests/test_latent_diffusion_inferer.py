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

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import VQVAE, AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

TEST_CASES = [
    [
        "AutoencoderKL",
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_channels": (4, 4),
            "latent_channels": 3,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "norm_num_groups": 4,
        },
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "out_channels": 3,
            "num_channels": [4, 4],
            "norm_num_groups": 4,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "num_head_channels": 4,
        },
        (1, 1, 8, 8),
        (1, 3, 4, 4),
    ],
    [
        "VQVAE",
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_channels": [4, 4],
            "num_res_layers": 1,
            "num_res_channels": [4, 4],
            "downsample_parameters": ((2, 4, 1, 1), (2, 4, 1, 1)),
            "upsample_parameters": ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            "num_embeddings": 16,
            "embedding_dim": 3,
        },
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "out_channels": 3,
            "num_channels": [8, 8],
            "norm_num_groups": 8,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "num_head_channels": 8,
        },
        (1, 1, 16, 16),
        (1, 3, 4, 4),
    ],
    [
        "VQVAE",
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_channels": [4, 4],
            "num_res_layers": 1,
            "num_res_channels": [4, 4],
            "downsample_parameters": ((2, 4, 1, 1), (2, 4, 1, 1)),
            "upsample_parameters": ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            "num_embeddings": 16,
            "embedding_dim": 3,
        },
        {
            "spatial_dims": 3,
            "in_channels": 3,
            "out_channels": 3,
            "num_channels": [8, 8],
            "norm_num_groups": 8,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "num_head_channels": 8,
        },
        (1, 1, 16, 16, 16),
        (1, 3, 4, 4, 4),
    ],
]


class TestDiffusionSamplingInferer(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_prediction_shape(self, model_type, autoencoder_params, stage_2_params, input_shape, latent_shape):
        if model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        if model_type == "VQVAE":
            stage_1 = VQVAE(**autoencoder_params)
        stage_2 = DiffusionModelUNet(**stage_2_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        input = torch.randn(input_shape).to(device)
        noise = torch.randn(latent_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        timesteps = torch.randint(0, scheduler.num_train_timesteps, (input_shape[0],), device=input.device).long()
        prediction = inferer(
            inputs=input, autoencoder_model=stage_1, diffusion_model=stage_2, noise=noise, timesteps=timesteps
        )
        self.assertEqual(prediction.shape, latent_shape)

    @parameterized.expand(TEST_CASES)
    def test_sample_shape(self, model_type, autoencoder_params, stage_2_params, input_shape, latent_shape):
        if model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        if model_type == "VQVAE":
            stage_1 = VQVAE(**autoencoder_params)
        stage_2 = DiffusionModelUNet(**stage_2_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        noise = torch.randn(latent_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        sample = inferer.sample(
            input_noise=noise, autoencoder_model=stage_1, diffusion_model=stage_2, scheduler=scheduler
        )
        self.assertEqual(sample.shape, input_shape)

    @parameterized.expand(TEST_CASES)
    def test_sample_intermediates(self, model_type, autoencoder_params, stage_2_params, input_shape, latent_shape):
        if model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        if model_type == "VQVAE":
            stage_1 = VQVAE(**autoencoder_params)
        stage_2 = DiffusionModelUNet(**stage_2_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        noise = torch.randn(latent_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        sample, intermediates = inferer.sample(
            input_noise=noise,
            autoencoder_model=stage_1,
            diffusion_model=stage_2,
            scheduler=scheduler,
            save_intermediates=True,
            intermediate_steps=1,
        )
        self.assertEqual(len(intermediates), 10)
        self.assertEqual(intermediates[0].shape, input_shape)

    @parameterized.expand(TEST_CASES)
    def test_get_likelihoods(self, model_type, autoencoder_params, stage_2_params, input_shape, latent_shape):
        if model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        if model_type == "VQVAE":
            stage_1 = VQVAE(**autoencoder_params)
        stage_2 = DiffusionModelUNet(**stage_2_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        input = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        sample, intermediates = inferer.get_likelihood(
            inputs=input,
            autoencoder_model=stage_1,
            diffusion_model=stage_2,
            scheduler=scheduler,
            save_intermediates=True,
        )
        self.assertEqual(len(intermediates), 10)
        self.assertEqual(intermediates[0].shape, latent_shape)

    @parameterized.expand(TEST_CASES)
    def test_resample_likelihoods(self, model_type, autoencoder_params, stage_2_params, input_shape, latent_shape):
        if model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        if model_type == "VQVAE":
            stage_1 = VQVAE(**autoencoder_params)
        stage_2 = DiffusionModelUNet(**stage_2_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        input = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        sample, intermediates = inferer.get_likelihood(
            inputs=input,
            autoencoder_model=stage_1,
            diffusion_model=stage_2,
            scheduler=scheduler,
            save_intermediates=True,
            resample_latent_likelihoods=True,
        )
        self.assertEqual(len(intermediates), 10)
        self.assertEqual(intermediates[0].shape[2:], input_shape[2:])


if __name__ == "__main__":
    unittest.main()
