# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Anomaly Detection with Transformers
#
# This tutorial illustrates how to use MONAI to perform image-wise anomaly detection with transformers based on the method proposed in [1].
#
# We will work with the MedNIST dataset available on MONAI
# (https://docs.monai.io/en/stable/apps.html#monai.apps.MedNISTDataset). Similar to "Experiment 2 – image-wise anomaly detection on 2D synthetic data", we will train our models on HeadCT images and check the likelihood of similar images (in-distribution) and images from other classes
#
# [1] - [Pinaya et al. "Unsupervised brain imaging 3D anomaly detection and segmentation with transformers"](https://doi.org/10.1016/j.media.2022.102475)
#
#
# ### Setup imports

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
from torch.nn import L1Loss, CrossEntropyLoss
import torch.nn.functional as F
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from tqdm import tqdm
from ignite.utils import convert_tensor

from generative.networks.nets import VQVAE, DecoderOnlyTransformer
from generative.utils.ordering import Ordering
from generative.utils.enums import OrderingType

print_config()

# %%
# for reproducibility purposes set a seed
set_determinism(42)

# %% [markdown]
# ### Setup a data directory and download dataset
#
# Specify a `MONAI_DATA_DIRECTORY` variable, where the data will be downloaded. If not
# specified a temporary directory will be used.

# %%
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

# %% [markdown]
# ### Download training data

# %%
train_data = MedNISTDataset(root_dir=root_dir, section="training", download=True, seed=0)
train_datalist = [{"image": item["image"]} for item in train_data.data if item["class_name"] == "HeadCT"]
image_size = 64
train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.RandAffined(
            keys=["image"],
            rotate_range=[(-np.pi / 18, np.pi / 18), (-np.pi / 18, np.pi / 18)],
            translate_range=[(-1, 1), (-1, 1)],
            scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
            spatial_size=[image_size, image_size],
            padding_mode="zeros",
            prob=0.5,
        ),
    ]
)
train_ds = Dataset(data=train_datalist, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)

# %% [markdown]
# ### Visualse some examples from the dataset

# %%
# Plot 3 examples from the training set
check_data = first(train_loader)
fig, ax = plt.subplots(nrows=1, ncols=3)
for image_n in range(3):
    ax[image_n].imshow(check_data["image"][image_n, 0, :, :], cmap="gray")
    ax[image_n].axis("off")

# %% [markdown]
# ### Download Validation Data

# %%
val_data = MedNISTDataset(root_dir=root_dir, section="validation", download=True, seed=0)
val_datalist = [{"image": item["image"]} for item in train_data.data if item["class_name"] == "HeadCT"]
val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    ]
)
val_ds = Dataset(data=val_datalist, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)

# %% [markdown]
# ## Vector Quantized Variational Autoencoder (VQ-VAE) Training
#
# The first step is to train a VQVAE network - once this is done we can use the trained vqvae model to encode the 2d images to generate the inputs required for the transformer

# %% [markdown]
# ### Define network, optimizer and losses

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
vqvae_model = VQVAE(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_res_layers=2,
    downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
    upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
    num_channels=(256, 256),
    num_res_channels=(256, 256),
    num_embeddings=16,
    embedding_dim=64,
)
vqvae_model.to(device)

# %%
optimizer = torch.optim.Adam(params=vqvae_model.parameters(), lr=5e-4)
l1_loss = L1Loss()

# %% [markdown]
# ### VQVAE Model training
# We will run our model for 100 epochs

# %%
n_epochs = 10
val_interval = 5
epoch_recon_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []
n_example_images = 4

total_start = time.time()
for epoch in range(n_epochs):
    vqvae_model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)

        # model outputs reconstruction and the quantization error
        reconstruction, quantization_loss = vqvae_model(images=images)

        recons_loss = l1_loss(reconstruction.float(), images.float())

        loss = recons_loss + quantization_loss

        loss.backward()
        optimizer.step()

        epoch_loss += recons_loss.item()

        progress_bar.set_postfix(
            {"recons_loss": epoch_loss / (step + 1), "quantization_loss": quantization_loss.item() / (step + 1)}
        )
    epoch_recon_loss_list.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        vqvae_model.eval()
        val_loss = 0
        with torch.no_grad():
            k = 0
            for val_step, batch in enumerate(val_loader, start=1):
                k += 1
                if k == 3:
                    break
                images = batch["image"].to(device)

                reconstruction, quantization_loss = vqvae_model(images=images)

                # get the first sample from the first validation batch for
                # visualizing how the training evolves
                if val_step == 1:
                    intermediary_images.append(reconstruction[:n_example_images, 0])

                recons_loss = l1_loss(reconstruction.float(), images.float())

                val_loss += recons_loss.item()

        val_loss /= val_step
        val_recon_epoch_loss_list.append(val_loss)

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")

# %% [markdown]
# ###  Plotting  evolution of reconstruction performance

# %%
# Plot every evaluation as a new line and example as columns
val_samples = np.linspace(val_interval, n_epochs, int(n_epochs / val_interval))
fig, ax = plt.subplots(nrows=len(val_samples), ncols=1, sharey=True)
fig.set_size_inches(18, 30)
for image_n in range(len(val_samples)):
    reconstructions = torch.reshape(intermediary_images[image_n], (64 * n_example_images, 64)).T
    ax[image_n].imshow(reconstructions.cpu(), cmap="gray")
    ax[image_n].set_xticks([])
    ax[image_n].set_yticks([])
    ax[image_n].set_ylabel(f"Epoch {val_samples[image_n]:.0f}")


# %% [markdown]
# ### Plot reconstructions of final trained vqvae model

# %%
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(images[0, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
ax[0].axis("off")
ax[0].title.set_text("Inputted Image")
ax[1].imshow(reconstruction[0, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
ax[1].axis("off")
ax[1].title.set_text("Reconstruction")
plt.show()

# %% [markdown]
# ## Autoregressive Transformer Training
#
# Now that a vqvae model has been trained, we can use this model to encode the data into its discrete latent representations. These inputs can then be flattened into a 1D sequence for the transformer to learn in an autoregressive manor.
#
# For this tutorial we will use the first appraoch and use the vqvae network to encode the data during the training cycle

# %% [markdown]
# ### Datasets
# We can use the same dataloader with augmentations as used for training the VQVAE model. However given the memory intensive nature of Transformer models we will need to reduce the batch size

# %%
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=True, num_workers=4)

# %% [markdown]
# ### Latent sequence ordering
# We need to define an ordering of which we convert our 2D latent space into a 1D sequence. For this we will use a simple raster scan.

# %%
spatial_shape = next(iter(train_loader))["image"].shape[2:]

# %%
# Get spatial dimensions of data
# We divide the spatial shape by 4 as the vqvae downsamples the image by a factor of 4 along each dimension
spatial_shape = next(iter(train_loader))["image"].shape[2:]
spatial_shape = (int(spatial_shape[0] / 4), int(spatial_shape[1] / 4))

ordering = Ordering(ordering_type=OrderingType.RASTER_SCAN.value, spatial_dims=2, dimensions=(1,) + spatial_shape)

sequence_ordering = ordering.get_sequence_ordering()
revert_sequence_ordering = ordering.get_revert_sequence_ordering()


# %% [markdown]
# ## Define Network, optimizer and losses

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer_model = DecoderOnlyTransformer(
    num_tokens=256,  # must be equal to num_embeddings input of VQVAE
    max_seq_len=spatial_shape[0] * spatial_shape[1],
    attn_layers_dim=64,
    attn_layers_depth=12,
    attn_layers_heads=8,
)
transformer_model.to(device)

# %%
optimizer = torch.optim.Adam(params=transformer_model.parameters(), lr=1e-3)
ce_loss = CrossEntropyLoss()

# %% [markdown]
# ### Transformer Model Training
# We will train the model for 100 epochs

# %%
n_epochs = 100
val_interval = 10
epoch_ce_loss_list = []
val_ce_epoch_loss_list = []
intermediary_images = []
vqvae_model.eval()

total_start = time.time()
for epoch in range(n_epochs):
    transformer_model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:

        images = batch["image"].to(device)
        # Encode images using vqvae and transformer to 1D sequence
        quantizations = vqvae_model.index_quantize(images)
        quantizations = quantizations.reshape(quantizations.shape[0], -1)
        quantizations = quantizations[:, sequence_ordering]

        # Pad input to give start of sequence token
        quantizations = F.pad(quantizations, (1, 0), "constant", 255)  # pad with 0 i.e. vocab size of vqvae
        quantizations = quantizations.long()

        quantizations_input = convert_tensor(quantizations[:, :-1], device, non_blocking=True)
        quantizations_target = convert_tensor(quantizations[:, 1:], device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # model outputs
        logits = transformer_model(x=quantizations_input).transpose(1, 2)

        loss = ce_loss(logits, quantizations_target)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"ce_loss": epoch_loss / (step + 1)})
    epoch_ce_loss_list.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        transformer_model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):

                images = batch["image"].to(device)
                # Encode images using vqvae and transformer to 1D sequence
                quantizations = vqvae_model.index_quantize(images)
                quantizations = quantizations.reshape(quantizations.shape[0], -1)
                quantizations = quantizations[:, sequence_ordering]

                # Pad input to give start of sequence token
                quantizations = F.pad(quantizations, (1, 0), "constant", 255)  # pad with 255 i.e. vocab size of vqvae
                quantizations = quantizations.long()

                quantizations_input = convert_tensor(quantizations[:, :-1], device, non_blocking=True)
                quantizations_target = convert_tensor(quantizations[:, 1:], device, non_blocking=True)

                # model outputs
                logits = transformer_model(x=quantizations_input).transpose(1, 2)

                loss = ce_loss(logits, quantizations_target)

                val_loss += loss.item()

        val_loss /= val_step
        val_ce_epoch_loss_list.append(val_loss)

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")

# %% [markdown]
# ### Plot evoluation of Generated Samples

# %%
# Plot every evaluation as a new line and example as columns
val_samples = np.linspace(val_interval, n_epochs, int(n_epochs / val_interval))
print(len(val_samples))
fig, ax = plt.subplots(nrows=len(val_samples), ncols=1, sharey=True)
fig.set_size_inches(12, 30)
for image_n in range(len(val_samples)):
    reconstructions = intermediary_images[image_n][0]
    ax[image_n].imshow(reconstructions.cpu(), cmap="gray")
    ax[image_n].set_xticks([])
    ax[image_n].set_yticks([])
    ax[image_n].set_ylabel(f"Epoch {val_samples[image_n]:.0f}")


# %% [markdown]
# ### Generating samples from the trained model

# Add anomaly detection using inferer