""" Tutorial for training an unconditioned Latent Diffusion Model on MEDNIST
Based on
https://github.com/Project-MONAI/tutorials/blob/main/2d_registration/registration_mednist.ipynb
https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb
"""
# %%
# # Denoising Diffusion Probabilistic Models with MedNIST Dataset
#
# This tutorial illustrates how to use MONAI for training a denoising diffusion probabilistic model (DDPM)[1] to create
# synthetic 2D images.
#
# [1] - Ho et al. "Denoising Diffusion Probabilistic Models" https://arxiv.org/abs/2006.11239
#
# TODO: Add Open in Colab
#
# ## Setup environment
# %%
# !python -c "import monai" || pip install -q "monai-weekly[pillow, tqdm, einops]"
# !python -c "import matplotlib" || pip install -q matplotlib
# %matplotlib inline
# %%
# ## Setup imports
# %%
# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
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
from tqdm import tqdm

from generative.networks.nets import DDPM, DiffusionModelUNet
from generative.schedulers import DDPMScheduler

print_config()
# %%
# ## Setup data directory
#
# You can specify a directory with the MONAI_DATA_DIRECTORY environment variable.
# This allows you to save results and reuse downloads.
# If not specified a temporary directory will be used.
# %%
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

# %%
# ## Set deterministic training for reproducibility
# %%
set_determinism(42)
# %%
# ## Setup MedNIST Dataset and training and validation dataloaders
# In this tutorial, we will train our models on the MedNIST dataset available on MONAI
# (https://docs.monai.io/en/stable/apps.html#monai.apps.MedNISTDataset). In order to train faster, we will select just
# one of the available classes ("Hand"), resulting in a training set with 7999 2D images.
# %%
train_data = MedNISTDataset(root_dir=root_dir, section="training", download=True, seed=0)
train_datalist = [{"image": item["image"]} for item in train_data.data if item["class_name"] == "Hand"]
# %%
train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.RandAffined(
            keys=["image"],
            rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
            translate_range=[(-1, 1), (-1, 1)],
            scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
            spatial_size=[64, 64],
            padding_mode="zeros",
            prob=0.5,
        ),
    ]
)
train_ds = CacheDataset(data=train_datalist, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
# %%
val_data = MedNISTDataset(root_dir=root_dir, section="validation", download=True, seed=0)
val_datalist = [{"image": item["image"]} for item in train_data.data if item["class_name"] == "Hand"]
val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    ]
)
val_ds = CacheDataset(data=val_datalist, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)
# %%
# ### Visualisation of the training images
# %%
check_data = first(train_loader)
print(f"batch shape: {check_data['image'].shape}")
image_visualisation = torch.concat(
    [check_data["image"][0, 0], check_data["image"][1, 0], check_data["image"][2, 0], check_data["image"][3, 0]], dim=1
)
plt.figure("training images", (12, 6))
plt.imshow(image_visualisation, vmin=0, vmax=1, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
# %%
# ### Define network, scheduler and optimizer
# At this step, we instantiate the MONAI components to create a DDPM, the UNET and the noise scheduler. We are using
# the original ddpm scheduler containing 1000 timesteps in its Markov chain, and a 2D unet with attention mechanisms
# in the 2nd and 4th levels, each with 1 attention head.
# %%
device = torch.device("cuda")

unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    model_channels=64,
    attention_resolutions=[2, 4],
    num_res_blocks=1,
    channel_mult=[1, 2, 2],
    num_heads=1,
)
unet.to(device)

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
)

optimizer = torch.optim.Adam(unet.parameters(), 2.5e-5)

# Combine model and scheduler in a DDPM class
ddpm = DDPM(unet_network=unet, scheduler=scheduler)
# %%
# ### Model training
# %%
n_epochs = 10
val_interval = 5
epoch_loss_list = []
val_epoch_loss_list = []
for epoch in range(n_epochs):
    ddpm.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)

        # Randomly select the timesteps to be used for the minibacth
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        # Add noise to the minibatch images with intensity defined by the scheduler and timesteps
        noise = torch.randn_like(images).to(device)
        noisy_image = ddpm.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

        # In this example, we are parametrising our DDPM to learn the added noise (epsilon).
        # For this reason, we are using our network to predict the added noise and then using L1 loss to predict
        # its performance.
        noise_pred = ddpm(x=noisy_image, timesteps=timesteps)
        loss = F.l1_loss(noise_pred.float(), noise.float())

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        progress_bar.set_postfix(
            {
                "loss": epoch_loss / (step + 1),
            }
        )
    epoch_loss_list.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        ddpm.eval()
        val_epoch_loss = 0
        progress_bar = tqdm(enumerate(val_loader), total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch} - Validation set")
        for step, batch in progress_bar:
            images = batch["image"].to(device)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
            noise = torch.randn_like(images).to(device)
            with torch.no_grad():
                noisy_image = ddpm.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
                noise_pred = ddpm(x=noisy_image, timesteps=timesteps)
                val_loss = F.l1_loss(noise_pred.float(), noise.float())

            val_epoch_loss += val_loss.item()
            progress_bar.set_postfix(
                {
                    "val_loss": val_epoch_loss / (step + 1),
                }
            )
        val_epoch_loss_list.append(val_epoch_loss / (step + 1))

        # Sampling image during training
        print(f"Epoch {epoch} - Sampling...")
        sample = ddpm.sample(sample_shape=(1, 1, 64, 64), num_timesteps=1000, device=device)

        plt.imshow(sample[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
        plt.tight_layout()
        plt.axis("off")
        plt.show()

# %%
# ### Learning curves
# %%
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_loss_list)
plt.plot(np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)), val_epoch_loss_list)
plt.tight_layout()
plt.show()

# %%
# ### Plotting sampling process along DDPM's Markov chain
# %%
ddpm.eval()
sample, intermediary = ddpm.sample(
    sample_shape=(1, 1, 64, 64), num_timesteps=1000, device=device, save_intermediates=True, intermediate_steps=100
)


chain = torch.concat(intermediary, dim=-1)
plt.imshow(chain[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
plt.tight_layout()
plt.axis("off")
plt.show()
