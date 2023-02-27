# %% [markdown]
# # Conditional Diffusion Probabilistic Model (CDPM) for 3D Images generation
#
# This tutorial illustrates how to use MONAI for training a 2D CDPM[1] for 3D images generation.

# [1] - [Peng et al. "Generating Realistic 3D Brain MRIs Using a Conditional Diffusion Probabilistic Model"](https://arxiv.org/abs/2212.08034)


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
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.transforms import AddChanneld, CenterSpatialCropd, Compose, Lambdad, LoadImaged, Resized, ScaleIntensityd
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from generative.inferers import DiffusionInferer

# TODO: Add right import reference after deployed
from generative.networks.nets.cdpm import UNet_2Plus1_Model
from generative.networks.schedulers import DDPMScheduler

print_config()

# %% [markdown]
# ## Setup data directory
#
# You can specify a directory with the MONAI_DATA_DIRECTORY environment variable.
#
# This allows you to save results and reuse downloads.
#
# If not specified a temporary directory will be used.


# %%  export MONAI_DATA_DIRECTORY="/home/Nobias/data/MONAI_DATA_DIRECTORY/"
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

# %% [markdown]
# ## Set deterministic training for reproducibility

# %%
set_determinism(0)

# %% [markdown]
# ## Setup Decathlon Dataset and training and validation dataloaders

# %%
train_transform = Compose(
    [
        LoadImaged(keys=["image"]),
        Lambdad(keys="image", func=lambda x: x[:, :, :, 1]),
        AddChanneld(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        CenterSpatialCropd(keys=["image"], roi_size=[176, 224, 155]),
        Resized(keys=["image"], spatial_size=(32, 48, 32)),
    ]
)

val_transform = Compose(
    [
        LoadImaged(keys=["image"]),
        Lambdad(keys="image", func=lambda x: x[:, :, :, 1]),
        AddChanneld(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        CenterSpatialCropd(keys=["image"], roi_size=[176, 224, 155]),
        Resized(keys=["image"], spatial_size=(32, 48, 32)),
    ]
)

# %% Task01_BrainTumour
train_ds = DecathlonDataset(
    root_dir=root_dir, task="Task01_BrainTumour", transform=train_transform, section="training", download=False
)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=1)

val_ds = DecathlonDataset(
    root_dir=root_dir, task="Task01_BrainTumour", transform=val_transform, section="validation", download=False
)

val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=1)


# %% [markdown]
# ### Visualization of the training images

# # %%
# plt.subplots(1, 4, figsize=(10, 6))
# for i in range(4):
#     plt.subplot(1, 4, i + 1)
#     plt.imshow(train_ds[i * 20]["image"][0, :, :, 15].detach().cpu(), vmin=0, vmax=1, cmap="gray")
#     plt.axis("off")
# plt.tight_layout()
# plt.show()

# %% [markdown]
# ### Define network, scheduler, optimizer, and inferer

# %%
device = torch.device("cuda")

model = UNet_2Plus1_Model(
    in_channels=1,
    out_channels=1,
    model_channels=128,
    attention_resolutions=[16, 8],
    num_res_blocks=2,
    channel_mult=[1, 1, 2, 4],
    num_heads=1,
)
model.to(device)

scheduler = DDPMScheduler(num_train_timesteps=1000)

inferer = DiffusionInferer(scheduler)

optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)


# %% [markdown]
# ### Model training

# %%
n_epochs = 200
val_interval = 50
epoch_loss_list = []
val_epoch_loss_list = []

scaler = GradScaler()
total_start = time.time()
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    # At most sample 16 slices for each batch
    max_frames=16
    for step, batch in progress_bar:
        images = batch["image"].to(device)#[1, 1, 32, 48, 32]

        ### Create images and context
        # randomly sample next batch to fill the unused tensor
        batch_1 = next(iter(train_loader))["image"].to(device)
        context = model.get_image_context(images.permute(0,4,1,2,3), batch_1.permute(0,4,1,2,3), max_frames)
        # Create timesteps
        timesteps = torch.randint(
            0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
        ).long()
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            # Generate random noise
            images = context[-1]
            noise = torch.randn_like(images).to(device)

            # Get model prediction
            # images = context[-1]
            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps, condition=context)

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_loss_list.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_epoch_loss = 0
        for step, batch in enumerate(val_loader):
            images = batch["image"].to(device)

            batch_1 = next(iter(val_loader))["image"].to(device)
            context = model.get_image_context(images.permute(0,4,1,2,3), batch_1.permute(0,4,1,2,3), max_frames)
            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            images = context[-1]
            noise = torch.randn_like(images).to(device)
            with torch.no_grad():
                with autocast(enabled=True):
                    noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps, condition=context)
                    val_loss = F.mse_loss(noise_pred.float(), noise.float())

            val_epoch_loss += val_loss.item()
            progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
        val_epoch_loss_list.append(val_epoch_loss / (step + 1))

        # Sampling image during training
        done_frames = [0]
        mri_length = 32
        max_frames = 16
        step_size = 8
        Sample = torch.zeros((1, 32, 1, 32, 48))
        while len(done_frames)<mri_length:
            obs_frame_indices, latent_frame_indices, obs_mask, latent_mask = model.next_indices(done_frames, mri_length, max_frames, step_size)
            frame_indices = torch.cat([torch.tensor(obs_frame_indices), torch.tensor(latent_frame_indices)]).long()

            sampled = Sample[:,frame_indices]
            print(f'Conditioning on {sorted(obs_frame_indices)} slices, predicting {sorted(latent_frame_indices)}.')

            sampled = sampled.to(device)
            context = (frame_indices.view(1,-1).to(device), obs_mask.to(device), latent_mask.to(device), sampled)

            scheduler.set_timesteps(num_inference_steps=1000)
            done_frames = done_frames + latent_frame_indices
            with autocast(enabled=True):
                image = inferer.sample(input_noise=sampled, diffusion_model=model, scheduler=scheduler, conditioning=context)
                Sample[0,latent_frame_indices] = image[len(obs_frame_indices):]

        plt.figure(figsize=(2, 2))
        plt.imshow(image[0,15, 0, :, :].cpu(), vmin=0, vmax=1, cmap="gray")
        plt.tight_layout()
        plt.axis("off")
        plt.show()

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")
# %% [markdown]
# ### Learning curves

# %%
plt.style.use("seaborn-v0_8")
plt.title("Learning Curves", fontsize=20)
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_loss_list, color="C0", linewidth=2.0, label="Train")
plt.plot(
    np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
    val_epoch_loss_list,
    color="C1",
    linewidth=2.0,
    label="Validation",
)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.show()


# %% [markdown]
# ### Plotting synthetic sample

# %%
model.eval()
noise = torch.randn((1, 1, 32, 48, 32))
noise = noise.to(device)
scheduler.set_timesteps(num_inference_steps=1000)
image = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=scheduler)


# %%
plt.style.use("default")
plotting_image_0 = np.concatenate([image[0, 0, :, :, 15].cpu(), np.flipud(image[0, 0, :, 24, :].cpu().T)], axis=1)
plotting_image_1 = np.concatenate([np.flipud(image[0, 0, 15, :, :].cpu().T), np.zeros((32, 32))], axis=1)
plt.imshow(np.concatenate([plotting_image_0, plotting_image_1], axis=0), vmin=0, vmax=1, cmap="gray")
plt.tight_layout()
plt.axis("off")
plt.show()
