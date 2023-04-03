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
        noise: torch.Tensor,
        autoencoder_model: nn.Module,
        diffusion_model: nn.Module,
        scheduler: nn.Module,
        prompt_embeds: torch.Tensor,
        guidance_scale: float = 7.0,
        scale_factor: float = 0.3,
    ) -> torch.Tensor:
        if has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)

        for t in progress_bar:
            noise_input = torch.cat([noise] * 2)
            model_output = diffusion_model(
                noise_input, timesteps=torch.Tensor((t,)).to(noise.device).long(), context=prompt_embeds
            )
            noise_pred_uncond, noise_pred_text = model_output.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise, _ = scheduler.step(noise_pred, t, noise)

        with autocast():
            sample = autoencoder_model.decode_stage_2_outputs(noise / scale_factor)

        return sample
