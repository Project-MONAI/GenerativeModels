# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
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

# %% [markdown]
# # Super-resolution using Stable Diffusion v2 Upscalers
#
# Tutorial to illustrate the super-resolution task on medical images using Latent Diffusion Models (LDMs) [1]. For that, we will use an autoencoder to obtain a latent representation of the high-resolution images. Then, we train a diffusion model to infer this latent representation when conditioned on a low-resolution image.
#
# To improve the performance of our models, we will use a method called "noise conditioning augmentation" (introduced in [2] and used in Stable Diffusion v2.0 and Imagen Video [3]). During the training, we add noise to the low-resolution images using a random signal-to-noise ratio, and we condition the diffusion models on the amount of noise added. At sampling time, we use a fixed signal-to-noise ratio, representing a small amount of augmentation that aids in removing artefacts in the samples.
#
#
# [1] - Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
#
# [2] - Ho et al. "Cascaded diffusion models for high fidelity image generation" https://arxiv.org/abs/2106.15282
#
# [3] - Ho et al. "High Definition Video Generation with Diffusion Models" https://arxiv.org/abs/2210.02303

# %% [markdown]
# ## Set up environment using Colab
#

# %%
# !python -c "import monai" || pip install -q "monai-weekly[tqdm]"
# !python -c "import matplotlib" || pip install -q matplotlib
# %matplotlib inline

# %% [markdown]
# ## Set up imports

# %%
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.utils import first, set_determinism
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

print_config()

# %%
# for reproducibility purposes set a seed
set_determinism(42)

# %% [markdown]
# ## Setup a data directory and download dataset
# Specify a MONAI_DATA_DIRECTORY variable, where the data will be downloaded. If not specified a temporary directory will be used.

# %%
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


# %% [markdown]
# ## Setup utils functions

# %%
def get_train_transforms():
    image_size = 64
    train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0,
                                        b_max=1.0, clip=True),
        transforms.RandAffined(
            keys=["image"],
            rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
            translate_range=[(-1, 1), (-1, 1)],
            scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
            spatial_size=[image_size, image_size],
            padding_mode="zeros",
            prob=0.5,
        ),
        transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
        transforms.Resized(keys=["low_res_image"], spatial_size=(16, 16)),
    ]
    )
    return train_transforms

def get_val_transforms():
    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, 
                                            b_max=1.0, clip=True),
            transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
            transforms.Resized(keys=["low_res_image"], spatial_size=(16, 16)),
        ]
    )
    return val_transforms



# %% [markdown]
# ## Define the LightningModule for AutoEncoder (transforms, network, loaders, etc)
# The LightningModule contains a refactoring of your training code. The following module is a reformatiing of the code in 2d_stable_diffusion_v2_super_resolution.
#

# %%
class AutoEnconder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.data_dir = root_dir
        self.autoencoderkl = AutoencoderKL(spatial_dims=2,
                                           in_channels=1,
                                           out_channels=1,
                                           num_channels=(256, 512, 512),
                                           latent_channels=3,
                                           num_res_blocks=2,
                                           norm_num_groups=32,
                                           attention_levels=(False, False, True))
        self.discriminator = PatchDiscriminator(spatial_dims=2, in_channels=1,
                                                num_layers_d=3, num_channels=64)
        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")
        self.perceptual_weight = 0.002
        self.autoencoder_warm_up_n_epochs = 10
        self.automatic_optimization = False
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.adv_weight = 0.005
        self.kl_weight = 1e-6
        
    def forward(self, z):
        return self.autoencoderkl(z)

    def prepare_data(self):
        train_transforms = get_train_transforms()
        val_transforms = get_val_transforms()
        
        train_data = MedNISTDataset(root_dir=self.data_dir, section="training", download=True, seed=0)
        train_datalist = [{"image": item["image"]} for item in train_data.data if item["class_name"] == "HeadCT"]
        val_data = MedNISTDataset(root_dir=self.data_dir, section="validation", download=True, seed=0)
        val_datalist = [{"image": item["image"]} for item in val_data.data if item["class_name"] == "HeadCT"]
        
        self.train_ds = CacheDataset(data=train_datalist, transform=train_transforms)
        self.val_ds = CacheDataset(data=val_datalist, transform=val_transforms)
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=16, shuffle=True,
                          num_workers=4, persistent_workers=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=16, shuffle=False,
                          num_workers=4)
                          
    def _compute_loss_generator(self, images, reconstruction, z_mu, z_sigma):
        recons_loss = F.l1_loss(reconstruction.float(), images.float())
        p_loss = self.perceptual_loss(reconstruction.float(), images.float())
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss_g = recons_loss + (self.kl_weight * kl_loss) + (self.perceptual_weight * p_loss)
        return loss_g,recons_loss
    
    def _compute_loss_discriminator(self, reconstruction):
        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = discriminator(images.contiguous().detach())[-1]
        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
        loss_d = self.adv_weight * discriminator_loss
        return loss_d, discriminator_loss
        
    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()
        images = batch["image"]
        reconstruction, z_mu, z_sigma = self.forward(images)
        loss_g, recons_loss = self._compute_loss_generator(images, reconstruction, z_mu, z_sigma)
        self.log("recons_loss", recons_loss, batch_size=16, prog_bar=True)

        if self.current_epoch > self.autoencoder_warm_up_n_epochs:
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += adv_weight * generator_loss
                          

        self.log("loss_g", loss_g, batch_size=16, prog_bar=True)
        self.manual_backward(loss_g)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        if self.current_epoch > self.autoencoder_warm_up_n_epochs:
            loss_d, discriminator_loss = self._compute_loss_discriminator(reconstruction)
            self.log("train_loss_d", loss_d, batch_size=16, prog_bar=True)
            self.manual_backward(loss_d)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)

                          
        if self.current_epoch > self.autoencoder_warm_up_n_epochs:
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()
            self.log("gen_loss", gen_loss, batch_size=16, prog_bar=True)
            self.log("disc_loss", disc_loss, batch_size=16, prog_bar=True)
                          
                                    
    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        reconstruction, z_mu, z_sigma = self.autoencoderkl(images)
        recons_loss = F.l1_loss(images.float(), reconstruction.float())
        self.log("val_loss_d", recons_loss, prog_bar=True)
                              

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.autoencoderkl.parameters(), lr=5e-5)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        return [optimizer_g, optimizer_d], []
                          


# %% [markdown]
# ## Train Autoencoder

# %%
n_epochs = 1
val_interval = 1

                          
# initialise the LightningModule
ae_net = AutoEnconder()

# set up checkpoints

checkpoint_callback = ModelCheckpoint(dirpath=root_dir, filename="best_metric_model")

                         
# initialise Lightning's trainer.
trainer = pl.Trainer(devices=1,
                     max_epochs=n_epochs,
                     check_val_every_n_epoch=val_interval,
                     callbacks=checkpoint_callback,
                     default_root_dir=root_dir)

# train
trainer.fit(ae_net)

# %% [markdown]
# ## Rescaling factor
#
# As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) became crucial in image-to-image translation models (such as the ones used for super-resolution). For this reason, we will compute the component-wise standard deviation to be used as scaling factor.

# %%
train_loader = ae_net.train_dataloader()
check_data = first(train_loader)
z = ae_net.autoencoderkl.train(mode=False).encode_stage_2_inputs(check_data["image"].to(ae_net.device))
print(f"Scaling factor set to {1/torch.std(z)}")
scale_factor = 1 / torch.std(z)


# %% [markdown]
# ## Define the LightningModule for DiffusionModelUnet (transforms, network, loaders, etc)
# The LightningModule contains a refactoring of your training code. The following module is a reformatiing of the code in 2d_stable_diffusion_v2_super_resolution.

# %%
class DiffusionUNET(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.data_dir = root_dir
        self.unet = DiffusionModelUNet(
                                spatial_dims=2,
                                in_channels=4,
                                out_channels=3,
                                num_res_blocks=2,
                                num_channels=(256, 256, 512, 1024),
                                attention_levels=(False, False, True, True),
                                num_head_channels=(0, 0, 64, 64),
                            )
        self.max_noise_level = 350
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, 
                          beta_schedule="linear",
                          beta_start=0.0015,
                          beta_end=0.0195)
        self.z = ae_net.autoencoderkl.train(mode=False)


    def forward(self, x, timesteps, low_res_timesteps):
        return self.unet(x=x, 
                         timesteps=timesteps,
                         class_labels=low_res_timesteps)
    
    
    def prepare_data(self):
        train_transforms = get_train_transforms()
        val_transforms = get_val_transforms()
        
        train_data = MedNISTDataset(root_dir=self.data_dir, section="training",
                                    download=True, seed=0)
        train_datalist = [{"image": item["image"]} for item in train_data.data if item["class_name"] == "HeadCT"]
        val_data = MedNISTDataset(root_dir=self.data_dir, section="validation",
                                  download=True, seed=0)
        val_datalist = [{"image": item["image"]} for item in val_data.data if item["class_name"] == "HeadCT"]
        
        self.train_ds = CacheDataset(data=train_datalist, transform=train_transforms)
        self.val_ds = CacheDataset(data=val_datalist, transform=val_transforms)
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=16, shuffle=True,
                          num_workers=4, persistent_workers=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=16, shuffle=True,
                          num_workers=4)
    
    def _calculate_loss(self, batch, batch_idx):
        images = batch["image"]
        low_res_image = batch["low_res_image"]       
        latent = self.z.encode_stage_2_inputs(images) * scale_factor
        latent =  latent.detach() # avoid adding this to graph.
        optimizer = self.optimizers()
        
        # Noise augmentation
        noise = torch.randn_like(latent)
        low_res_noise = torch.randn_like(low_res_image)
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (latent.shape[0],),
                                  device=latent.device).long()
        low_res_timesteps = torch.randint(
            0, self.max_noise_level, (low_res_image.shape[0],), device=latent.device
        ).long()

        noisy_latent = self.scheduler.add_noise(original_samples=latent, 
                                           noise=noise, timesteps=timesteps)
        noisy_low_res_image = self.scheduler.add_noise(
            original_samples=low_res_image, noise=low_res_noise, 
            timesteps=low_res_timesteps
        )

        latent_model_input = torch.cat([noisy_latent, noisy_low_res_image], dim=1)

        noise_pred = self.forward(latent_model_input, timesteps, low_res_timesteps)
        loss = F.mse_loss(noise_pred.float(), noise.float())
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, batch_idx)
        self.log("train_loss", loss, batch_size=16, prog_bar=True)
        return loss

        
    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, batch_idx)
        self.log("val_loss", loss, batch_size=16, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=5e-5)
        return optimizer


# %% [markdown]
# ## Train Diffusion Model
#
# In order to train the diffusion model to perform super-resolution, we will need to concatenate the latent representation of the high-resolution with the low-resolution image. For this, we create a Diffusion model with `in_channels=4`. Since only the outputted latent representation is interesting, we set `out_channels=3`.
#
# As mentioned, we will use the conditioned augmentation (introduced in [2] section 3 and used on Stable Diffusion Upscalers and Imagen Video [3] Section 2.5) as it has been shown critical for cascaded diffusion models, as well for super-resolution tasks. For this, we apply Gaussian noise augmentation to the low-resolution images. We will use a scheduler low_res_scheduler to add this noise, with the t step defining the signal-to-noise ratio and use the t value to condition the diffusion model (inputted using class_labels argument).

# %%
n_epochs = 3
val_interval = 3

                          
# initialise the LightningModule
d_net = DiffusionUNET()

# set up checkpoints

checkpoint_callback = ModelCheckpoint(dirpath=root_dir, filename="best_metric_model_dunet")

                         
# initialise Lightning's trainer.
trainer = pl.Trainer(devices=1,
                     max_epochs=n_epochs,
                     check_val_every_n_epoch=val_interval,
                     callbacks=checkpoint_callback,
                     default_root_dir=root_dir)

# train
trainer.fit(d_net)

# %% [markdown]
# ### Plotting sampling example

# %%
# Sampling image during training
num_samples = 3
val_loader = d_net.val_dataloader()
check_data = first(val_loader)
images = check_data["image"]
sampling_image = check_data["low_res_image"][:num_samples]

# %%
latents = torch.randn((num_samples, 3, 16, 16)).to(images.device)
low_res_noise = torch.randn((num_samples, 1, 16, 16)).to(images.device)
noise_level = 10
noise_level = torch.Tensor((noise_level,)).long().to(images.device)
scheduler = d_net.scheduler
noisy_low_res_image = scheduler.add_noise(original_samples=sampling_image, 
                                          noise=low_res_noise,
                                          timesteps=torch.Tensor((noise_level,)).long())

scheduler.set_timesteps(num_inference_steps=1000)
for t in tqdm(scheduler.timesteps, ncols=110):
    with torch.no_grad():
        with autocast(enabled=True):
            latent_model_input = torch.cat([latents, noisy_low_res_image], dim=1)
            noise_pred = d_net.forward(x=latent_model_input,
                                       timesteps=torch.Tensor((t,)),
                                       low_res_timesteps=noise_level)

        # 2. compute previous image: x_t -> x_t-1
        latents, _ = scheduler.step(noise_pred, t, latents)

with torch.no_grad():
    decoded = ae_net.autoencoderkl.decode_stage_2_outputs(latents / scale_factor)

# %%
low_res_bicubic = nn.functional.interpolate(sampling_image, (64, 64), mode="bicubic")
fig, axs = plt.subplots(num_samples, 3, figsize=(8, 8))
axs[0, 0].set_title("Original image")
axs[0, 1].set_title("Low-resolution Image")
axs[0, 2].set_title("Outputted image")
for i in range(0, num_samples):
    axs[i, 0].imshow(images[i, 0].cpu(), vmin=0, vmax=1, cmap="gray")
    axs[i, 0].axis("off")
    axs[i, 1].imshow(low_res_bicubic[i, 0].cpu(), vmin=0, vmax=1, cmap="gray")
    axs[i, 1].axis("off")
    axs[i, 2].imshow(decoded[i, 0].cpu(), vmin=0, vmax=1, cmap="gray")
    axs[i, 2].axis("off")
plt.tight_layout()

# %% [markdown]
# ### Clean-up data directory

# %%
if directory is None:
    shutil.rmtree(root_dir)
