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
import torch.nn as nn
from monai.inferers import Inferer
from monai.utils import optional_import

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")


class DiffusionInferer(Inferer):

    """
    DiffusionInferer takes a trained diffusion model and a scheduler and can be used to perform a signal forward pass
    for a training iteration, and sample from the model.
    """

    def __init__(self, scheduler) -> None:
        Inferer.__init__(self)
        self.scheduler = scheduler

    def __call__(
        self,
        inputs: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: Input image to which noise is added
            diffusion_model: model
            scheduler: diffusion scheduler.
            input_noise: random noise, of the same shape as the input.
            condition: Conditioning for network input.
        """
        num_timesteps = self.scheduler.num_train_timesteps
        timesteps = torch.randint(0, num_timesteps, (inputs.shape[0],), device=inputs.device).long()
        noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)
        prediction = diffusion_model(x=noisy_image, timesteps=timesteps, context=condition)

        return prediction

    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Optional[Callable[..., torch.Tensor]] = None,
        save_intermediates: Optional[bool] = False,
        intermediate_steps: Optional[int] = 100,
        conditioning: Optional[torch.Tensor] = None,
        verbose: Optional[bool] = True,
    ) -> torch.Tensor:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
        """
        if not scheduler:
            scheduler = self.scheduler
        image = input_noise
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
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


class LatentDiffusionInferer(Inferer):
    """
    LatentDiffusionInferer takes a stage 1 model (VQVAE or AutoencoderKL), diffusion model, and a scheduler, and can
    be used to perform a signal forward pass for a training iteration, and sample from the model.

    Args:
        scheduler: a scheduler to be used in combination with `unet` to denoise the encoded image latents.
        scale_factor: scale factor to multiply the values of the latent representation before processing it by the
            second stage.
    """

    def __init__(self, scheduler: nn.Module, scale_factor: float = 1.0) -> None:
        Inferer.__init__(self)
        self.scheduler = scheduler
        self.scale_factor = scale_factor

    def __call__(
        self,
        inputs: torch.Tensor,
        stage_1_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: input image to which noise is added.
            stage_1_model: first stage model.
            diffusion_model: diffusion model.
            noise: random noise, of the same shape as the input.
            condition: conditioning for network input.
        """
        with torch.no_grad():
            latent = stage_1_model.encode_stage_2_inputs(inputs) * self.scale_factor

        num_timesteps = self.scheduler.num_train_timesteps
        timesteps = torch.randint(0, num_timesteps, (inputs.shape[0],), device=inputs.device).long()
        noisy_latent = self.scheduler.add_noise(original_samples=latent, noise=noise, timesteps=timesteps)
        prediction = diffusion_model(x=noisy_latent, timesteps=timesteps, context=condition)

        return prediction

    def sample(
        self,
        input_noise: torch.Tensor,
        stage_1_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Optional[Callable[..., torch.Tensor]] = None,
        conditioning: Optional[torch.Tensor] = None,
        verbose: Optional[bool] = True,
    ) -> torch.Tensor:
        """
        Args:
            input_noise: random noise, of the same shape as the desired latent space.
            stage_1_model: first stage model.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            conditioning: Conditioning for network input.
        """
        if not scheduler:
            scheduler = self.scheduler
        latent = input_noise
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        for t in progress_bar:
            with torch.no_grad():
                model_output = diffusion_model(
                    latent, timesteps=torch.Tensor((t,)).to(input_noise.device), context=conditioning
                )

            latent, _ = scheduler.step(model_output, t, latent)

        with torch.no_grad():
            image = stage_1_model.decode_stage_2_outputs(latent) * self.scale_factor

        return image
