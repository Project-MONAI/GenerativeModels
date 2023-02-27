from __future__ import annotations

import torch
import torch.nn as nn
from monai.utils import optional_import
from torch.cuda.amp import autocast

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")


class Sampler:
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def sampling_fn(
        self,
        input_noise: torch.Tensor,
        autoencoder_model: nn.Module,
        diffusion_model: nn.Module,
        scheduler: nn.Module,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        if has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)

        image = input_noise
        cond_concat = conditioning.squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cond_concat = cond_concat.expand(list(cond_concat.shape[0:2]) + list(input_noise.shape[2:]))
        for t in progress_bar:
            with torch.no_grad():
                model_output = diffusion_model(
                    torch.cat((image, cond_concat), dim=1),
                    timesteps=torch.Tensor((t,)).to(input_noise.device).long(),
                    context=conditioning,
                )
                image, _ = scheduler.step(model_output, t, image)

        with torch.no_grad():
            with autocast():
                sample = autoencoder_model.decode_stage_2_outputs(image)

        return sample
