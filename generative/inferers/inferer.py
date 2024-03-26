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

import math
from collections.abc import Callable, Sequence
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import decollate_batch
from monai.inferers import Inferer
from monai.transforms import CenterSpatialCrop, SpatialPad
from monai.utils import optional_import

from generative.networks.nets import VQVAE, SPADEAutoencoderKL, SPADEDiffusionModelUNet

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")


class DiffusionInferer(Inferer):
    """
    DiffusionInferer takes a trained diffusion model and a scheduler and can be used to perform a signal forward pass
    for a training iteration, and sample from the model.

    Args:
        scheduler: diffusion scheduler.
    """

    def __init__(self, scheduler: nn.Module) -> None:
        Inferer.__init__(self)
        self.scheduler = scheduler

    def __call__(
        self,
        inputs: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
        mode: str = "crossattn",
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: Input image to which noise is added.
            diffusion_model: diffusion model.
            noise: random noise, of the same shape as the input.
            timesteps: random timesteps.
            condition: Conditioning for network input.
            mode: Conditioning mode for the network.
            seg: if model is instance of SPADEDiffusionModelUnet, segmentation must be
            provided on the forward (for SPADE-like AE or SPADE-like DM)
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)
        if mode == "concat":
            noisy_image = torch.cat([noisy_image, condition], dim=1)
            condition = None
        diffusion_model = (
            partial(diffusion_model, seg=seg)
            if isinstance(diffusion_model, SPADEDiffusionModelUNet)
            else diffusion_model
        )
        prediction = diffusion_model(x=noisy_image, timesteps=timesteps, context=condition)

        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

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
            diffusion_model = (
                partial(diffusion_model, seg=seg)
                if isinstance(diffusion_model, SPADEDiffusionModelUNet)
                else diffusion_model
            )
            if mode == "concat":
                model_input = torch.cat([image, conditioning], dim=1)
                model_output = diffusion_model(
                    model_input, timesteps=torch.Tensor((t,)).to(input_noise.device), context=None
                )
            else:
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

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
        verbose: bool = True,
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Computes the log-likelihoods for an input.

        Args:
            inputs: input images, NxCxHxW[xD]
            diffusion_model: model to compute likelihood from
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
        """

        if not scheduler:
            scheduler = self.scheduler
        if scheduler._get_name() != "DDPMScheduler":
            raise NotImplementedError(
                f"Likelihood computation is only compatible with DDPMScheduler,"
                f" you are using {scheduler._get_name()}"
            )
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        noise = torch.randn_like(inputs).to(inputs.device)
        total_kl = torch.zeros(inputs.shape[0]).to(inputs.device)
        for t in progress_bar:
            timesteps = torch.full(inputs.shape[:1], t, device=inputs.device).long()
            noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)
            diffusion_model = (
                partial(diffusion_model, seg=seg)
                if isinstance(diffusion_model, SPADEDiffusionModelUNet)
                else diffusion_model
            )
            if mode == "concat":
                noisy_image = torch.cat([noisy_image, conditioning], dim=1)
                model_output = diffusion_model(noisy_image, timesteps=timesteps, context=None)
            else:
                model_output = diffusion_model(x=noisy_image, timesteps=timesteps, context=conditioning)

            # get the model's predicted mean,  and variance if it is predicted
            if model_output.shape[1] == inputs.shape[1] * 2 and scheduler.variance_type in ["learned", "learned_range"]:
                model_output, predicted_variance = torch.split(model_output, inputs.shape[1], dim=1)
            else:
                predicted_variance = None

            # 1. compute alphas, betas
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = scheduler.alphas_cumprod[t - 1] if t > 0 else scheduler.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            if scheduler.prediction_type == "epsilon":
                pred_original_sample = (noisy_image - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            elif scheduler.prediction_type == "sample":
                pred_original_sample = model_output
            elif scheduler.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * noisy_image - (beta_prod_t**0.5) * model_output
            # 3. Clip "predicted x_0"
            if scheduler.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * scheduler.betas[t]) / beta_prod_t
            current_sample_coeff = scheduler.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t

            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            predicted_mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * noisy_image

            # get the posterior mean and variance
            posterior_mean = scheduler._get_mean(timestep=t, x_0=inputs, x_t=noisy_image)
            posterior_variance = scheduler._get_variance(timestep=t, predicted_variance=predicted_variance)

            log_posterior_variance = torch.log(posterior_variance)
            log_predicted_variance = torch.log(predicted_variance) if predicted_variance else log_posterior_variance

            if t == 0:
                # compute -log p(x_0|x_1)
                kl = -self._get_decoder_log_likelihood(
                    inputs=inputs,
                    means=predicted_mean,
                    log_scales=0.5 * log_predicted_variance,
                    original_input_range=original_input_range,
                    scaled_input_range=scaled_input_range,
                )
            else:
                # compute kl between two normals
                kl = 0.5 * (
                    -1.0
                    + log_predicted_variance
                    - log_posterior_variance
                    + torch.exp(log_posterior_variance - log_predicted_variance)
                    + ((posterior_mean - predicted_mean) ** 2) * torch.exp(-log_predicted_variance)
                )
            total_kl += kl.view(kl.shape[0], -1).mean(axis=1)
            if save_intermediates:
                intermediates.append(kl.cpu())

        if save_intermediates:
            return total_kl, intermediates
        else:
            return total_kl

    def _approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal. Code adapted from https://github.com/openai/improved-diffusion.
        """

        return 0.5 * (
            1.0 + torch.tanh(torch.sqrt(torch.Tensor([2.0 / math.pi]).to(x.device)) * (x + 0.044715 * torch.pow(x, 3)))
        )

    def _get_decoder_log_likelihood(
        self,
        inputs: torch.Tensor,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
    ) -> torch.Tensor:
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image. Code adapted from https://github.com/openai/improved-diffusion.

        Args:
            input: the target images. It is assumed that this was uint8 values,
                      rescaled to the range [-1, 1].
            means: the Gaussian mean Tensor.
            log_scales: the Gaussian log stddev Tensor.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
        """
        assert inputs.shape == means.shape
        bin_width = (scaled_input_range[1] - scaled_input_range[0]) / (
            original_input_range[1] - original_input_range[0]
        )
        centered_x = inputs - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + bin_width / 2)
        cdf_plus = self._approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - bin_width / 2)
        cdf_min = self._approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            inputs < -0.999,
            log_cdf_plus,
            torch.where(inputs > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
        )
        assert log_probs.shape == inputs.shape
        return log_probs


class LatentDiffusionInferer(DiffusionInferer):
    """
    LatentDiffusionInferer takes a stage 1 model (VQVAE or AutoencoderKL), diffusion model, and a scheduler, and can
    be used to perform a signal forward pass for a training iteration, and sample from the model.

    Args:
        scheduler: a scheduler to be used in combination with `unet` to denoise the encoded image latents.
        scale_factor: scale factor to multiply the values of the latent representation before processing it by the
            second stage.
        ldm_latent_shape: desired spatial latent space shape. Used if there is a difference in the autoencoder model's latent shape.
        autoencoder_latent_shape:  autoencoder_latent_shape: autoencoder spatial latent space shape. Used if there is a
             difference between the autoencoder's latent shape and the DM shape.
    """

    def __init__(
        self,
        scheduler: nn.Module,
        scale_factor: float = 1.0,
        ldm_latent_shape: list | None = None,
        autoencoder_latent_shape: list | None = None,
    ) -> None:
        super().__init__(scheduler=scheduler)
        self.scale_factor = scale_factor
        if (ldm_latent_shape is None) ^ (autoencoder_latent_shape is None):
            raise ValueError("If ldm_latent_shape is None, autoencoder_latent_shape must be None" "and vice versa.")
        self.ldm_latent_shape = ldm_latent_shape
        self.autoencoder_latent_shape = autoencoder_latent_shape
        if self.ldm_latent_shape is not None:
            self.ldm_resizer = SpatialPad(spatial_size=self.ldm_latent_shape)
            self.autoencoder_resizer = CenterSpatialCrop(roi_size=self.autoencoder_latent_shape)

    def __call__(
        self,
        inputs: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
        mode: str = "crossattn",
        seg: torch.Tensor | None = None,
        quantized: bool = True,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: input image to which the latent representation will be extracted and noise is added.
            autoencoder_model: first stage model.
            diffusion_model: diffusion model.
            noise: random noise, of the same shape as the latent representation.
            timesteps: random timesteps.
            condition: conditioning for network input.
            mode: Conditioning mode for the network.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
            quantized: if autoencoder_model is a VQVAE, quantized controls whether the latents to the LDM
            are quantized or not.
        """
        with torch.no_grad():
            autoencode = autoencoder_model.encode_stage_2_inputs
            if isinstance(autoencoder_model, VQVAE):
                autoencode = partial(autoencoder_model.encode_stage_2_inputs, quantized=quantized)
            latent = autoencode(inputs) * self.scale_factor

        if self.ldm_latent_shape is not None:
            latent = torch.stack([self.ldm_resizer(i) for i in decollate_batch(latent)], 0)

        call = super().__call__
        if isinstance(diffusion_model, SPADEDiffusionModelUNet):
            call = partial(super().__call__, seg=seg)

        prediction = call(
            inputs=latent,
            diffusion_model=diffusion_model,
            noise=noise,
            timesteps=timesteps,
            condition=condition,
            mode=mode,
        )
        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired latent representation.
            autoencoder_model: first stage model.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, or autoencoder_model
             is instance of SPADEAutoencoderKL, segmentation must be provided.
        """

        if (
            isinstance(autoencoder_model, SPADEAutoencoderKL)
            and isinstance(diffusion_model, SPADEDiffusionModelUNet)
            and autoencoder_model.decoder.label_nc != diffusion_model.label_nc
        ):
            raise ValueError(
                "If both autoencoder_model and diffusion_model implement SPADE, the number of semantic"
                "labels for each must be compatible. "
            )

        sample = super().sample
        if isinstance(diffusion_model, SPADEDiffusionModelUNet):
            sample = partial(super().sample, seg=seg)

        outputs = sample(
            input_noise=input_noise,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            intermediate_steps=intermediate_steps,
            conditioning=conditioning,
            mode=mode,
            verbose=verbose,
        )

        if save_intermediates:
            latent, latent_intermediates = outputs
        else:
            latent = outputs

        if self.autoencoder_latent_shape is not None:
            latent = torch.stack([self.autoencoder_resizer(i) for i in decollate_batch(latent)], 0)
            if save_intermediates:
                latent_intermediates = [
                    torch.stack([self.autoencoder_resizer(i) for i in decollate_batch(l)], 0)
                    for l in latent_intermediates
                ]

        decode = autoencoder_model.decode_stage_2_outputs
        if isinstance(autoencoder_model, SPADEAutoencoderKL):
            decode = partial(autoencoder_model.decode_stage_2_outputs, seg=seg)
        image = decode(latent / self.scale_factor)

        if save_intermediates:
            intermediates = []
            for latent_intermediate in latent_intermediates:
                decode = autoencoder_model.decode_stage_2_outputs
                if isinstance(autoencoder_model, SPADEAutoencoderKL):
                    decode = partial(autoencoder_model.decode_stage_2_outputs, seg=seg)
                intermediates.append(decode(latent_intermediate / self.scale_factor))
            return image, intermediates

        else:
            return image

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
        verbose: bool = True,
        resample_latent_likelihoods: bool = False,
        resample_interpolation_mode: str = "nearest",
        seg: torch.Tensor | None = None,
        quantized: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Computes the log-likelihoods of the latent representations of the input.

        Args:
            inputs: input images, NxCxHxW[xD]
            autoencoder_model: first stage model.
            diffusion_model: model to compute likelihood from
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
            resample_latent_likelihoods: if true, resamples the intermediate likelihood maps to have the same spatial
                dimension as the input images.
            resample_interpolation_mode: if use resample_latent_likelihoods, select interpolation 'nearest', 'bilinear',
                or 'trilinear;
            seg: if diffusion model is instance of SPADEDiffusionModel, or autoencoder_model
             is instance of SPADEAutoencoderKL, segmentation must be provided.
            quantized: if autoencoder_model is a VQVAE, quantized controls whether the latents to the LDM
            are quantized or not.
        """
        if resample_latent_likelihoods and resample_interpolation_mode not in ("nearest", "bilinear", "trilinear"):
            raise ValueError(
                f"resample_interpolation mode should be either nearest, bilinear, or trilinear, got {resample_interpolation_mode}"
            )

        autoencode = autoencoder_model.encode_stage_2_inputs
        if isinstance(autoencoder_model, VQVAE):
            autoencode = partial(autoencoder_model.encode_stage_2_inputs, quantized=quantized)
        latents = autoencode(inputs) * self.scale_factor

        if self.ldm_latent_shape is not None:
            latents = torch.stack([self.ldm_resizer(i) for i in decollate_batch(latents)], 0)

        get_likelihood = super().get_likelihood
        if isinstance(diffusion_model, SPADEDiffusionModelUNet):
            get_likelihood = partial(super().get_likelihood, seg=seg)

        outputs = get_likelihood(
            inputs=latents,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            conditioning=conditioning,
            mode=mode,
            verbose=verbose,
        )

        if save_intermediates and resample_latent_likelihoods:
            intermediates = outputs[1]
            resizer = nn.Upsample(size=inputs.shape[2:], mode=resample_interpolation_mode)
            intermediates = [resizer(x) for x in intermediates]
            outputs = (outputs[0], intermediates)
        return outputs


class ControlNetDiffusionInferer(DiffusionInferer):
    """
    ControlNetDiffusionInferer takes a trained diffusion model and a scheduler and can be used to perform a signal
    forward pass for a training iteration, and sample from the model, supporting ControlNet-based conditioning.

    Args:
        scheduler: diffusion scheduler.
    """

    def __init__(self, scheduler: nn.Module) -> None:
        Inferer.__init__(self)
        self.scheduler = scheduler

    def __call__(
        self,
        inputs: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        controlnet: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        cn_cond: torch.Tensor,
        condition: torch.Tensor | None = None,
        mode: str = "crossattn",
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: Input image to which noise is added.
            diffusion_model: diffusion model.
            controlnet: controlnet sub-network.
            noise: random noise, of the same shape as the input.
            timesteps: random timesteps.
            cn_cond: conditioning image for the ControlNet.
            condition: Conditioning for network input.
            mode: Conditioning mode for the network.
            seg: if model is instance of SPADEDiffusionModelUnet, segmentation must be
            provided on the forward (for SPADE-like AE or SPADE-like DM)
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)

        if mode == "concat":
            noisy_image = torch.cat([noisy_image, condition], dim=1)
            condition = None

        down_block_res_samples, mid_block_res_sample = controlnet(
            x=noisy_image, timesteps=timesteps, controlnet_cond=cn_cond, context=condition
        )

        diffuse = diffusion_model
        if isinstance(diffusion_model, SPADEDiffusionModelUNet):
            diffuse = partial(diffusion_model, seg=seg)

        prediction = diffuse(
            x=noisy_image,
            timesteps=timesteps,
            context=condition,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )

        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        controlnet: Callable[..., torch.Tensor],
        cn_cond: torch.Tensor,
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            controlnet: controlnet sub-network.
            cn_cond: conditioning image for the ControlNet.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        if not scheduler:
            scheduler = self.scheduler
        image = input_noise
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        for t in progress_bar:
            if mode == "concat":
                model_input = torch.cat([image, conditioning], dim=1)
                context_ = None
            else:
                model_input = image
                context_ = conditioning

            # 1. ControlNet forward
            down_block_res_samples, mid_block_res_sample = controlnet(
                x=model_input,
                timesteps=torch.Tensor((t,)).to(input_noise.device),
                controlnet_cond=cn_cond,
                context=context_,
            )
            # 2. predict noise model_output
            diffuse = diffusion_model
            if isinstance(diffusion_model, SPADEDiffusionModelUNet):
                diffuse = partial(diffusion_model, seg=seg)

            model_output = diffuse(
                model_input,
                timesteps=torch.Tensor((t,)).to(input_noise.device),
                context=context_,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

            # 3. compute previous image: x_t -> x_t-1
            image, _ = scheduler.step(model_output, t, image)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)
        if save_intermediates:
            return image, intermediates
        else:
            return image

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        controlnet: Callable[..., torch.Tensor],
        cn_cond: torch.Tensor,
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
        verbose: bool = True,
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Computes the log-likelihoods for an input.

        Args:
            inputs: input images, NxCxHxW[xD]
            diffusion_model: model to compute likelihood from
            controlnet: controlnet sub-network.
            cn_cond: conditioning image for the ControlNet.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
        """

        if not scheduler:
            scheduler = self.scheduler
        if scheduler._get_name() != "DDPMScheduler":
            raise NotImplementedError(
                f"Likelihood computation is only compatible with DDPMScheduler,"
                f" you are using {scheduler._get_name()}"
            )
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        noise = torch.randn_like(inputs).to(inputs.device)
        total_kl = torch.zeros(inputs.shape[0]).to(inputs.device)
        for t in progress_bar:
            timesteps = torch.full(inputs.shape[:1], t, device=inputs.device).long()
            noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)

            if mode == "concat":
                noisy_image = torch.cat([noisy_image, conditioning], dim=1)
                conditioning = None

            down_block_res_samples, mid_block_res_sample = controlnet(
                x=noisy_image,
                timesteps=torch.Tensor((t,)).to(inputs.device),
                controlnet_cond=cn_cond,
                context=conditioning,
            )

            diffuse = diffusion_model
            if isinstance(diffusion_model, SPADEDiffusionModelUNet):
                diffuse = partial(diffusion_model, seg=seg)

            model_output = diffuse(
                noisy_image,
                timesteps=timesteps,
                context=conditioning,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

            # get the model's predicted mean,  and variance if it is predicted
            if model_output.shape[1] == inputs.shape[1] * 2 and scheduler.variance_type in ["learned", "learned_range"]:
                model_output, predicted_variance = torch.split(model_output, inputs.shape[1], dim=1)
            else:
                predicted_variance = None

            # 1. compute alphas, betas
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = scheduler.alphas_cumprod[t - 1] if t > 0 else scheduler.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            if scheduler.prediction_type == "epsilon":
                pred_original_sample = (noisy_image - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            elif scheduler.prediction_type == "sample":
                pred_original_sample = model_output
            elif scheduler.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * noisy_image - (beta_prod_t**0.5) * model_output
            # 3. Clip "predicted x_0"
            if scheduler.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * scheduler.betas[t]) / beta_prod_t
            current_sample_coeff = scheduler.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t

            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            predicted_mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * noisy_image

            # get the posterior mean and variance
            posterior_mean = scheduler._get_mean(timestep=t, x_0=inputs, x_t=noisy_image)
            posterior_variance = scheduler._get_variance(timestep=t, predicted_variance=predicted_variance)

            log_posterior_variance = torch.log(posterior_variance)
            log_predicted_variance = torch.log(predicted_variance) if predicted_variance else log_posterior_variance

            if t == 0:
                # compute -log p(x_0|x_1)
                kl = -super()._get_decoder_log_likelihood(
                    inputs=inputs,
                    means=predicted_mean,
                    log_scales=0.5 * log_predicted_variance,
                    original_input_range=original_input_range,
                    scaled_input_range=scaled_input_range,
                )
            else:
                # compute kl between two normals
                kl = 0.5 * (
                    -1.0
                    + log_predicted_variance
                    - log_posterior_variance
                    + torch.exp(log_posterior_variance - log_predicted_variance)
                    + ((posterior_mean - predicted_mean) ** 2) * torch.exp(-log_predicted_variance)
                )
            total_kl += kl.view(kl.shape[0], -1).mean(axis=1)
            if save_intermediates:
                intermediates.append(kl.cpu())

        if save_intermediates:
            return total_kl, intermediates
        else:
            return total_kl


class ControlNetLatentDiffusionInferer(ControlNetDiffusionInferer):
    """
    ControlNetLatentDiffusionInferer takes a stage 1 model (VQVAE or AutoencoderKL), diffusion model, controlnet,
    and a scheduler, and can be used to perform a signal forward pass for a training iteration, and sample from
    the model.

    Args:
        scheduler: a scheduler to be used in combination with `unet` to denoise the encoded image latents.
        scale_factor: scale factor to multiply the values of the latent representation before processing it by the
            second stage.
        ldm_latent_shape: desired spatial latent space shape. Used if there is a difference in the autoencoder model's latent shape.
        autoencoder_latent_shape:  autoencoder_latent_shape: autoencoder spatial latent space shape. Used if there is a
             difference between the autoencoder's latent shape and the DM shape.
    """

    def __init__(
        self,
        scheduler: nn.Module,
        scale_factor: float = 1.0,
        ldm_latent_shape: list | None = None,
        autoencoder_latent_shape: list | None = None,
    ) -> None:
        super().__init__(scheduler=scheduler)
        self.scale_factor = scale_factor
        if (ldm_latent_shape is None) ^ (autoencoder_latent_shape is None):
            raise ValueError("If ldm_latent_shape is None, autoencoder_latent_shape must be None" "and vice versa.")
        self.ldm_latent_shape = ldm_latent_shape
        self.autoencoder_latent_shape = autoencoder_latent_shape
        if self.ldm_latent_shape is not None:
            self.ldm_resizer = SpatialPad(spatial_size=self.ldm_latent_shape)
            self.autoencoder_resizer = CenterSpatialCrop(roi_size=[-1] + self.autoencoder_latent_shape)

    def __call__(
        self,
        inputs: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        controlnet: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        cn_cond: torch.Tensor,
        condition: torch.Tensor | None = None,
        mode: str = "crossattn",
        seg: torch.Tensor | None = None,
        quantized: bool = True,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: input image to which the latent representation will be extracted and noise is added.
            autoencoder_model: first stage model.
            diffusion_model: diffusion model.
            controlnet: instance of ControlNet model
            noise: random noise, of the same shape as the latent representation.
            timesteps: random timesteps.
            cn_cond: conditioning tensor for the ControlNet network
            condition: conditioning for network input.
            mode: Conditioning mode for the network.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
            quantized: if autoencoder_model is a VQVAE, quantized controls whether the latents to the LDM
            are quantized or not.
        """
        with torch.no_grad():
            autoencode = autoencoder_model.encode_stage_2_inputs
            if isinstance(autoencoder_model, VQVAE):
                autoencode = partial(autoencoder_model.encode_stage_2_inputs, quantized=quantized)
            latent = autoencode(inputs) * self.scale_factor

        if self.ldm_latent_shape is not None:
            latent = torch.stack([self.ldm_resizer(i) for i in decollate_batch(latent)], 0)

        if cn_cond.shape[2:] != latent.shape[2:]:
            cn_cond = F.interpolate(cn_cond, latent.shape[2:])

        call = super().__call__
        if isinstance(diffusion_model, SPADEDiffusionModelUNet):
            call = partial(super().__call__, seg=seg)

        prediction = call(
            inputs=latent,
            diffusion_model=diffusion_model,
            controlnet=controlnet,
            noise=noise,
            timesteps=timesteps,
            cn_cond=cn_cond,
            condition=condition,
            mode=mode,
        )

        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        controlnet: Callable[..., torch.Tensor],
        cn_cond: torch.Tensor,
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired latent representation.
            autoencoder_model: first stage model.
            diffusion_model: model to sample from.
            controlnet: instance of ControlNet model.
            cn_cond: conditioning tensor for the ControlNet network.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, or autoencoder_model
             is instance of SPADEAutoencoderKL, segmentation must be provided.
        """

        if (
            isinstance(autoencoder_model, SPADEAutoencoderKL)
            and isinstance(diffusion_model, SPADEDiffusionModelUNet)
            and autoencoder_model.decoder.label_nc != diffusion_model.label_nc
        ):
            raise ValueError(
                "If both autoencoder_model and diffusion_model implement SPADE, the number of semantic"
                "labels for each must be compatible. "
            )

        if cn_cond.shape[2:] != input_noise.shape[2:]:
            cn_cond = F.interpolate(cn_cond, input_noise.shape[2:])

        sample = super().sample
        if isinstance(diffusion_model, SPADEDiffusionModelUNet):
            sample = partial(super().sample, seg=seg)

        outputs = sample(
            input_noise=input_noise,
            diffusion_model=diffusion_model,
            controlnet=controlnet,
            cn_cond=cn_cond,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            intermediate_steps=intermediate_steps,
            conditioning=conditioning,
            mode=mode,
            verbose=verbose,
        )

        if save_intermediates:
            latent, latent_intermediates = outputs
        else:
            latent = outputs

        if self.autoencoder_latent_shape is not None:
            latent = torch.stack([self.autoencoder_resizer(i) for i in decollate_batch(latent)], 0)
            if save_intermediates:
                latent_intermediates = [
                    torch.stack([self.autoencoder_resizer(i) for i in decollate_batch(l)], 0)
                    for l in latent_intermediates
                ]

        decode = autoencoder_model.decode_stage_2_outputs
        if isinstance(autoencoder_model, SPADEAutoencoderKL):
            decode = partial(autoencoder_model.decode_stage_2_outputs, seg=seg)

        image = decode(latent / self.scale_factor)

        if save_intermediates:
            intermediates = []
            for latent_intermediate in latent_intermediates:
                decode = autoencoder_model.decode_stage_2_outputs
                if isinstance(autoencoder_model, SPADEAutoencoderKL):
                    decode = partial(autoencoder_model.decode_stage_2_outputs, seg=seg)
                intermediates.append(decode(latent_intermediate / self.scale_factor))
            return image, intermediates

        else:
            return image

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        controlnet: Callable[..., torch.Tensor],
        cn_cond: torch.Tensor,
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
        verbose: bool = True,
        resample_latent_likelihoods: bool = False,
        resample_interpolation_mode: str = "nearest",
        seg: torch.Tensor | None = None,
        quantized: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Computes the log-likelihoods of the latent representations of the input.

        Args:
            inputs: input images, NxCxHxW[xD]
            autoencoder_model: first stage model.
            diffusion_model: model to compute likelihood from
            controlnet: instance of ControlNet model.
            cn_cond: conditioning tensor for the ControlNet network.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
            resample_latent_likelihoods: if true, resamples the intermediate likelihood maps to have the same spatial
                dimension as the input images.
            resample_interpolation_mode: if use resample_latent_likelihoods, select interpolation 'nearest', 'bilinear',
                or 'trilinear;
            seg: if diffusion model is instance of SPADEDiffusionModel, or autoencoder_model
             is instance of SPADEAutoencoderKL, segmentation must be provided.
            quantized: if autoencoder_model is a VQVAE, quantized controls whether the latents to the LDM
            are quantized or not.
        """
        if resample_latent_likelihoods and resample_interpolation_mode not in ("nearest", "bilinear", "trilinear"):
            raise ValueError(
                f"resample_interpolation mode should be either nearest, bilinear, or trilinear, got {resample_interpolation_mode}"
            )

        with torch.no_grad():
            autoencode = autoencoder_model.encode_stage_2_inputs
            if isinstance(autoencoder_model, VQVAE):
                autoencode = partial(autoencoder_model.encode_stage_2_inputs, quantized=quantized)
            latents = autoencode(inputs) * self.scale_factor

        if cn_cond.shape[2:] != latents.shape[2:]:
            cn_cond = F.interpolate(cn_cond, latents.shape[2:])

        if self.ldm_latent_shape is not None:
            latents = torch.stack([self.ldm_resizer(i) for i in decollate_batch(latents)], 0)

        get_likelihood = super().get_likelihood
        if isinstance(diffusion_model, SPADEDiffusionModelUNet):
            get_likelihood = partial(super().get_likelihood, seg=seg)

        outputs = get_likelihood(
            inputs=latents,
            diffusion_model=diffusion_model,
            controlnet=controlnet,
            cn_cond=cn_cond,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            conditioning=conditioning,
            mode=mode,
            verbose=verbose,
        )

        if save_intermediates and resample_latent_likelihoods:
            intermediates = outputs[1]
            resizer = nn.Upsample(size=inputs.shape[2:], mode=resample_interpolation_mode)
            intermediates = [resizer(x) for x in intermediates]
            outputs = (outputs[0], intermediates)
        return outputs


class VQVAETransformerInferer(Inferer):
    """
    Class to perform inference with a VQVAE + Transformer model.
    """

    def __init__(self) -> None:
        Inferer.__init__(self)

    def __call__(
        self,
        inputs: torch.Tensor,
        vqvae_model: Callable[..., torch.Tensor],
        transformer_model: Callable[..., torch.Tensor],
        ordering: Callable[..., torch.Tensor],
        condition: torch.Tensor | None = None,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: input image to which the latent representation will be extracted.
            vqvae_model: first stage model.
            transformer_model: autoregressive transformer model.
            ordering: ordering of the quantised latent representation.
            return_latent: also return latent sequence and spatial dim of the latent.
            condition: conditioning for network input.
        """
        with torch.no_grad():
            latent = vqvae_model.index_quantize(inputs)

        latent_spatial_dim = tuple(latent.shape[1:])
        latent = latent.reshape(latent.shape[0], -1)
        latent = latent[:, ordering.get_sequence_ordering()]

        # get the targets for the loss
        target = latent.clone()
        # Use the value from vqvae_model's num_embeddings as the starting token, the "Begin Of Sentence" (BOS) token.
        # Note the transformer_model must have vqvae_model.num_embeddings + 1 defined as num_tokens.
        latent = F.pad(latent, (1, 0), "constant", vqvae_model.num_embeddings)
        # crop the last token as we do not need the probability of the token that follows it
        latent = latent[:, :-1]
        latent = latent.long()

        # train on a part of the sequence if it is longer than max_seq_length
        seq_len = latent.shape[1]
        max_seq_len = transformer_model.max_seq_len
        if max_seq_len < seq_len:
            start = torch.randint(low=0, high=seq_len + 1 - max_seq_len, size=(1,)).item()
        else:
            start = 0
        prediction = transformer_model(x=latent[:, start : start + max_seq_len], context=condition)
        if return_latent:
            return prediction, target[:, start : start + max_seq_len], latent_spatial_dim
        else:
            return prediction

    @torch.no_grad()
    def sample(
        self,
        latent_spatial_dim: Sequence[int, int, int] | Sequence[int, int],
        starting_tokens: torch.Tensor,
        vqvae_model: Callable[..., torch.Tensor],
        transformer_model: Callable[..., torch.Tensor],
        ordering: Callable[..., torch.Tensor],
        conditioning: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Sampling function for the VQVAE + Transformer model.

        Args:
            latent_spatial_dim: shape of the sampled image.
            starting_tokens: starting tokens for the sampling. It must be vqvae_model.num_embeddings value.
            vqvae_model: first stage model.
            transformer_model: model to sample from.
            conditioning: Conditioning for network input.
            temperature: temperature for sampling.
            top_k: top k sampling.
            verbose: if true, prints the progression bar of the sampling process.
        """
        seq_len = math.prod(latent_spatial_dim)

        if verbose and has_tqdm:
            progress_bar = tqdm(range(seq_len))
        else:
            progress_bar = iter(range(seq_len))

        latent_seq = starting_tokens.long()
        for _ in progress_bar:
            # if the sequence context is growing too long we must crop it at block_size
            if latent_seq.size(1) <= transformer_model.max_seq_len:
                idx_cond = latent_seq
            else:
                idx_cond = latent_seq[:, -transformer_model.max_seq_len :]

            # forward the model to get the logits for the index in the sequence
            logits = transformer_model(x=idx_cond, context=conditioning)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # remove the chance to be sampled the BOS token
            probs[:, vqvae_model.num_embeddings] = 0
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            latent_seq = torch.cat((latent_seq, idx_next), dim=1)

        latent_seq = latent_seq[:, 1:]
        latent_seq = latent_seq[:, ordering.get_revert_sequence_ordering()]
        latent = latent_seq.reshape((starting_tokens.shape[0],) + latent_spatial_dim)

        return vqvae_model.decode_samples(latent)

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        vqvae_model: Callable[..., torch.Tensor],
        transformer_model: Callable[..., torch.Tensor],
        ordering: Callable[..., torch.Tensor],
        condition: torch.Tensor | None = None,
        resample_latent_likelihoods: bool = False,
        resample_interpolation_mode: str = "nearest",
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Computes the log-likelihoods of the latent representations of the input.

        Args:
            inputs: input images, NxCxHxW[xD]
            vqvae_model: first stage model.
            transformer_model: autoregressive transformer model.
            ordering: ordering of the quantised latent representation.
            condition: conditioning for network input.
            resample_latent_likelihoods: if true, resamples the intermediate likelihood maps to have the same spatial
                dimension as the input images.
            resample_interpolation_mode: if use resample_latent_likelihoods, select interpolation 'nearest', 'bilinear',
                or 'trilinear;
            verbose: if true, prints the progression bar of the sampling process.

        """
        if resample_latent_likelihoods and resample_interpolation_mode not in ("nearest", "bilinear", "trilinear"):
            raise ValueError(
                f"resample_interpolation mode should be either nearest, bilinear, or trilinear, got {resample_interpolation_mode}"
            )

        with torch.no_grad():
            latent = vqvae_model.index_quantize(inputs)

        latent_spatial_dim = tuple(latent.shape[1:])
        latent = latent.reshape(latent.shape[0], -1)
        latent = latent[:, ordering.get_sequence_ordering()]
        seq_len = math.prod(latent_spatial_dim)

        # Use the value from vqvae_model's num_embeddings as the starting token, the "Begin Of Sentence" (BOS) token.
        # Note the transformer_model must have vqvae_model.num_embeddings + 1 defined as num_tokens.
        latent = F.pad(latent, (1, 0), "constant", vqvae_model.num_embeddings)
        latent = latent.long()

        # get the first batch, up to max_seq_length, efficiently
        logits = transformer_model(x=latent[:, : transformer_model.max_seq_len], context=condition)
        probs = F.softmax(logits, dim=-1)
        # target token for each set of logits is the next token along
        target = latent[:, 1:]
        probs = torch.gather(probs, 2, target[:, : transformer_model.max_seq_len].unsqueeze(2)).squeeze(2)

        # if we have not covered the full sequence we continue with inefficient looping
        if probs.shape[1] < target.shape[1]:
            if verbose and has_tqdm:
                progress_bar = tqdm(range(transformer_model.max_seq_len, seq_len))
            else:
                progress_bar = iter(range(transformer_model.max_seq_len, seq_len))

            for i in progress_bar:
                idx_cond = latent[:, i + 1 - transformer_model.max_seq_len : i + 1]
                # forward the model to get the logits for the index in the sequence
                logits = transformer_model(x=idx_cond, context=condition)
                # pluck the logits at the final step
                logits = logits[:, -1, :]
                # apply softmax to convert logits to (normalized) probabilities
                p = F.softmax(logits, dim=-1)
                # select correct values and append
                p = torch.gather(p, 1, target[:, i].unsqueeze(1))

                probs = torch.cat((probs, p), dim=1)

        # convert to log-likelihood
        probs = torch.log(probs)

        # reshape
        probs = probs[:, ordering.get_revert_sequence_ordering()]
        probs_reshaped = probs.reshape((inputs.shape[0],) + latent_spatial_dim)
        if resample_latent_likelihoods:
            resizer = nn.Upsample(size=inputs.shape[2:], mode=resample_interpolation_mode)
            probs_reshaped = resizer(probs_reshaped[:, None, ...])

        return probs_reshaped
