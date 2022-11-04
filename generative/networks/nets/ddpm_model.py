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

import torch
from torch import nn
from tqdm import tqdm

__all__ = ["DDPM"]


# TODO: Discuss other methods that should be included


class DDPM(nn.Module):
    """
    DDPM Model class. This class takes care of storing all components for the DDPM and handles methods for sampling.

    Args:
        unet_network: Conditional U-Net architecture to denoise the encoded image latents.
        scheduler: Variance scheduler.
    """

    def __init__(
        self,
        unet_network: nn.Module,
        scheduler: nn.Module,
    ) -> None:

        super().__init__()
        self.unet_network = unet_network
        self.scheduler = scheduler

    # TODO: Implement sampling methods
    def sample_unconditioned(self):
        raise NotImplementedError

    def train(self):
        self.unet_network.train()

    def eval(self):
        self.unet_network.eval()

    def add_noise(self, *args, **kwargs):
        return self.scheduler.add_noise(*args, **kwargs)

    def sample(self, sample_shape, num_timesteps, device, save_intermediates=False, intermediate_steps=100):
        image = torch.randn(sample_shape).to(device)
        self.scheduler.set_timesteps(num_timesteps)
        progress_bar = tqdm(self.scheduler.timesteps)
        intermediary = []
        for t in progress_bar:
            # 1. predict noise model_output
            with torch.no_grad():
                model_output = self.unet_network(image, torch.asarray((t,)).to(device))
            # 2. compute previous image: x_t -> x_t-1
            image, _ = self.scheduler.step(model_output, t, image)
            if save_intermediates and t % intermediate_steps == 0:
                intermediary.append(image)
        if save_intermediates:
            return image, intermediary
        else:
            return image

    def forward(self, *args, **kwargs):
        return self.unet_network(*args, **kwargs)
