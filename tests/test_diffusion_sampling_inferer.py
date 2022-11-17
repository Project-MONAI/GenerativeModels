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

from generative.inferers import DiffusionSamplingInferer
from generative.networks.nets import DiffusionModelUNet
from generative.schedulers import DDIMScheduler, DDPMScheduler

TEST_CASES = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "model_channels": 8,
            "norm_num_groups": 8,
            "attention_resolutions": [1],
            "num_res_blocks": 1,
            "channel_mult": [1],
            "num_heads": 1,
        },
        (2, 1, 8, 8),
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "model_channels": 8,
            "norm_num_groups": 8,
            "attention_resolutions": [1],
            "num_res_blocks": 1,
            "channel_mult": [1],
            "num_heads": 1,
        },
        (2, 1, 8, 8, 8),
    ],
]


class TestDiffusionSamplingInferer(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_2d_and_3d(self, model_params, input_shape):

        model = DiffusionModelUNet(**model_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        noise = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(
            num_train_timesteps=10,
        )
        sampler = DiffusionSamplingInferer()
        scheduler.set_timesteps(num_inference_steps=10)
        sample = sampler(input_noise=noise, diffusion_model=model, scheduler=scheduler)
        self.assertEqual(sample.shape, input_shape)

    @parameterized.expand(TEST_CASES)
    def test_intermediates(self, model_params, input_shape):
        model = DiffusionModelUNet(**model_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        noise = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(
            num_train_timesteps=10,
        )
        sampler = DiffusionSamplingInferer()
        scheduler.set_timesteps(num_inference_steps=10)
        sample, intermediates = sampler(
            input_noise=noise, diffusion_model=model, scheduler=scheduler, save_intermediates=True, intermediate_steps=1
        )
        self.assertEqual(len(intermediates), 10)

    @parameterized.expand(TEST_CASES)
    def test_ddim_sampler(self, model_params, input_shape):
        model = DiffusionModelUNet(**model_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        noise = torch.randn(input_shape).to(device)
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
        )
        sampler = DiffusionSamplingInferer()
        scheduler.set_timesteps(num_inference_steps=10)
        sample, intermediates = sampler(
            input_noise=noise, diffusion_model=model, scheduler=scheduler, save_intermediates=True, intermediate_steps=1
        )
        self.assertEqual(len(intermediates), 10)


if __name__ == "__main__":
    unittest.main()
