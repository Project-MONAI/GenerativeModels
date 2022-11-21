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

from typing import Callable, Optional

import torch
from monai.inferers import Inferer
from tqdm import tqdm


class DiffusionInferer(Inferer):
    """
    DiffusionSamplingInferer takes a trained diffusion model and a scheduler and produces a sample.
    """

    def __init__(self) -> None:
        Inferer.__init__(self)

    def __call__(
        self,
        inputs: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ):
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: Input image to which noise is added
            diffusion_model: model
            scheduler: diffusion scheduler.
            input_noise: random noise, of the same shape as the input.
            condition: Conditioning for network input.
        """
        num_timesteps = scheduler.num_train_timesteps
        timesteps = torch.randint(0, num_timesteps, (inputs.shape[0],), device=inputs.device).long()
        noisy_image = scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)
        noise_pred = diffusion_model(x=noisy_image, timesteps=timesteps, context=condition)

        return noise_pred

    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor],
        save_intermediates: bool = False,
        intermediate_steps: int = 100,
        conditioning: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler.
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.

        """
        image = input_noise.clone()
        progress_bar = tqdm(scheduler.timesteps)
        intermediates = []
        for t in progress_bar:
            # 1. predict noise model_output
            with torch.no_grad():
                model_output = diffusion_model(
                    image, timesteps=torch.Tensor((t,)).to(input_noise.device), context=conditioning
                )
            # 2. compute previous image: x_t -> x_t-1
            image, _ = scheduler.step(model_output, t, image)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)
        if save_intermediates:
            return image, intermediates
        else:
            return image
