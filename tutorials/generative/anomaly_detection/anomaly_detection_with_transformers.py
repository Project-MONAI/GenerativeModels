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
# # Anomaly Detection with Transformers
#
# This tutorial illustrates how to use MONAI to perform image-wise and localised anomaly detection with transformers based on the method proposed in Pinaya et al.[1].
#
# Here, we will work with the [MedNIST dataset](https://docs.monai.io/en/stable/apps.html#monai.apps.MedNISTDataset) available on MONAI, and similar to "Experiment 2 – image-wise anomaly detection on 2D synthetic data" from [1], we will train a general-purpose VQ-VAE (using all MEDNIST classes), and then a generative models (i.e., Transformer) on `HeadCT` images.
#
# We will compute the log-likelihood of images from the same class (in-distribution class) and images from other classes (out-of-distribution). We will also provide an example of performing localised anomaly detection with these trained models.
#
# [1] - [Pinaya et al. "Unsupervised brain imaging 3D anomaly detection and segmentation with transformers"](https://doi.org/10.1016/j.media.2022.102475)

# %% [markdown]
# ### Setup environment

# %%
# !python -c "import seaborn" || pip install -q seaborn
# !python -c "import monai" || pip install -q "monai-weekly[tqdm]"
# !python -c "import matplotlib" || pip install -q matplotlib
# %matplotlib inline

# %% [markdown]
# ### Setup imports

# %%
import os
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from torch.nn import CrossEntropyLoss, L1Loss
from tqdm import tqdm
from generative.inferers import VQVAETransformerInferer
from generative.networks.nets import VQVAE, DecoderOnlyTransformer
from generative.utils.enums import OrderingType
from generative.utils.ordering import Ordering

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
train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.RandAffined(
            keys=["image"],
            rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
            translate_range=[(-1, 1), (-1, 1)],
            scale_range=[(-0.01, 0.01), (-0.01, 0.01)],
            spatial_size=[64, 64],
            padding_mode="zeros",
            prob=0.5,
        ),
    ]
)
train_data = MedNISTDataset(root_dir=root_dir, section="training", download=True, seed=0, transform=train_transforms)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)

# %% [markdown]
# ### Visualise some examples from the dataset

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
val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    ]
)
val_data = MedNISTDataset(root_dir=root_dir, section="validation", download=True, seed=0, transform=val_transforms)
val_loader = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=4, persistent_workers=True)

# %% [markdown]
# ## Vector Quantized Variational Autoencoder
#
# The first step is to train a Vector Quantized Variation Autoencoder (VQ-VAE). This network is responsible for creating a compressed version of the inputted data. Once its training is done, we can use the encoder to obtain smaller and discrete representations of the 2D images to generate the inputs required for our autoregressive transformer.
#
# For its training, we will use the L1 loss, and we will update its codebook using a method based on Exponential Moving Average (EMA).

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
# ### VQ-VAE Model training
# We will train our VQ-VAE for 20 epochs.

# %%
n_epochs = 20
val_interval = 5
epoch_losses = []
val_epoch_losses = []

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
    epoch_losses.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        vqvae_model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch["image"].to(device)
                reconstruction, quantization_loss = vqvae_model(images=images)
                recons_loss = l1_loss(reconstruction.float(), images.float())
                val_loss += recons_loss.item()

        val_loss /= val_step
        val_epoch_losses.append(val_loss)

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")

# %% [markdown]
# ### Plot reconstructions of final trained vqvae model

# %%
images = first(val_loader)["image"].to(device)
reconstruction, quantization_loss = vqvae_model(images=images)
nrows = 4
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(3, 4))
for i in range(nrows):
    ax.flat[i * 2].imshow(images[i + 20, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
    ax.flat[i * 2].axis("off")
    ax.flat[i * 2 + 1].imshow(reconstruction[i + 20, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
    ax.flat[i * 2 + 1].axis("off")
ax.flat[0].title.set_text("Image")
ax.flat[1].title.set_text("Reconstruction")
plt.show()

# %% [markdown]
# # Autoregressive Transformer
#
# Now that our VQ-VAE model has been trained, we can use this model to encode the data into its discrete latent representations. Then, to be able to input it into the autoregressive Transformer, it is necessary to transform this 2D latent representation into a 1D sequence.
#
# In order to train it in an autoregressive manner, we will use the CrossEntropy Loss as the Transformer will try to predict the next token value for each position of the sequence.
#
# Here we will use the MONAI's `VQVAETransformerInferer` class to help with the forward pass and to get the predicted likelihood from the VQ-VAE + Transformer models.

# %% [markdown]
# ### Datasets
# To train the transformer, we only use the `HeadCT` class.

# %%
in_distribution_class = "HeadCT"

train_data = MedNISTDataset(root_dir=root_dir, section="training", seed=0)
train_datalist = [{"image": item["image"]} for item in train_data.data if item["class_name"] == in_distribution_class]
train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.RandAffined(
            keys=["image"],
            rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
            translate_range=[(-1, 1), (-1, 1)],
            scale_range=[(-0.01, 0.01), (-0.01, 0.01)],
            spatial_size=[64, 64],
            padding_mode="zeros",
            prob=0.5,
        ),
    ]
)
train_ds = Dataset(data=train_datalist, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)

val_data = MedNISTDataset(root_dir=root_dir, section="validation", seed=0)
val_datalist = [{"image": item["image"]} for item in val_data.data if item["class_name"] == in_distribution_class]
val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    ]
)
val_ds = Dataset(data=val_datalist, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, persistent_workers=True)

# %% [markdown]
# ### 2D latent representation -> 1D sequence
# We need to define an ordering of which we convert our 2D latent space into a 1D sequence. For this we will use a simple raster scan.

# %%
# Get spatial dimensions of data
test_data = next(iter(train_loader))["image"].to(device)
spatial_shape = vqvae_model.encode_stage_2_inputs(test_data).shape[2:]

ordering = Ordering(ordering_type=OrderingType.RASTER_SCAN.value, spatial_dims=2, dimensions=(1,) + spatial_shape)


# %% [markdown]
# ### Define network, inferer, optimizer and loss function

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer_model = DecoderOnlyTransformer(
    num_tokens=16 + 1,
    max_seq_len=spatial_shape[0] * spatial_shape[1],
    attn_layers_dim=128,
    attn_layers_depth=16,
    attn_layers_heads=16,
)
transformer_model.to(device)

inferer = VQVAETransformerInferer()

# %%
optimizer = torch.optim.Adam(params=transformer_model.parameters(), lr=1e-4)
ce_loss = CrossEntropyLoss()

# %% [markdown]
# ### Transformer Training
# We will train the Transformer for 20 epochs.

# %%
n_epochs = 20
val_interval = 5
epoch_losses = []
val_epoch_losses = []
vqvae_model.eval()

total_start = time.time()
for epoch in range(n_epochs):
    transformer_model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits, target, _ = inferer(images, vqvae_model, transformer_model, ordering, return_latent=True)
        logits = logits.transpose(1, 2)

        loss = ce_loss(logits, target)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"ce_loss": epoch_loss / (step + 1)})
    epoch_losses.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        transformer_model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch["image"].to(device)

                logits, quantizations_target, _ = inferer(
                    images, vqvae_model, transformer_model, ordering, return_latent=True
                )
                logits = logits.transpose(1, 2)

                loss = ce_loss(logits[:, :, :-1], quantizations_target[:, 1:])

                val_loss += loss.item()
        # get sample
        sample = inferer.sample(
            vqvae_model=vqvae_model,
            transformer_model=transformer_model,
            ordering=ordering,
            latent_spatial_dim=(spatial_shape[0], spatial_shape[1]),
            starting_tokens=vqvae_model.num_embeddings * torch.ones((1, 1), device=device),
        )
        plt.imshow(sample[0, 0, ...].cpu().detach())
        plt.title(f"Sample epoch {epoch}")
        plt.show()
        val_loss /= val_step
        val_epoch_losses.append(val_loss)
        val_loss /= val_step
        val_epoch_losses.append(val_loss)

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")

# %% [markdown]
# ## Image-wise anomaly detection
#
# To verify the performance of the VQ-VAE + Transformer performing unsupervised anomaly detection, we will use the images from the test set of the MedNIST dataset. We will consider images from the `HeadCT` class as in-distribution images.

# %%
vqvae_model.eval()
transformer_model.eval()

test_data = MedNISTDataset(root_dir=root_dir, section="test", download=True, seed=0)

in_distribution_datalist = [
    {"image": item["image"]} for item in test_data.data if item["class_name"] == in_distribution_class
]
in_distribution_ds = Dataset(data=in_distribution_datalist, transform=val_transforms)
in_distribution_loader = DataLoader(
    in_distribution_ds, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True
)

in_likelihoods = []

progress_bar = tqdm(enumerate(in_distribution_loader), total=len(in_distribution_loader), ncols=110)
progress_bar.set_description(f"In-distribution data")
for step, batch in progress_bar:
    images = batch["image"].to(device)

    log_likelihood = inferer.get_likelihood(
        inputs=images, vqvae_model=vqvae_model, transformer_model=transformer_model, ordering=ordering
    )
    in_likelihoods.append(log_likelihood.sum(dim=(1, 2)).cpu().numpy())

in_likelihoods = np.concatenate(in_likelihoods)

# %% [markdown]
# We will use all other classes for the out-of-distribution examples.

# %%
all_classes = {item["class_name"] for item in test_data.data}
all_classes.remove(in_distribution_class)

all_likelihoods = {}
for c in all_classes:
    ood_datalist = [{"image": item["image"]} for item in test_data.data if item["class_name"] == c]
    ood_ds = Dataset(data=ood_datalist, transform=val_transforms)
    ood_loader = DataLoader(ood_ds, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)

    ood_likelihoods = []

    progress_bar = tqdm(enumerate(ood_loader), total=len(ood_loader), ncols=110)
    progress_bar.set_description(f"out-of-distribution data {c}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)

        log_likelihood = inferer.get_likelihood(
            inputs=images, vqvae_model=vqvae_model, transformer_model=transformer_model, ordering=ordering
        )
        ood_likelihoods.append(log_likelihood.sum(dim=(1, 2)).cpu().numpy())

    ood_likelihoods = np.concatenate(ood_likelihoods)
    all_likelihoods[c] = ood_likelihoods

# %% [markdown]
# ## Log-likelihood plot
#
# Here, we plot the log-likelihood of the images. In this case, the lower the log-likelihood, the more unlikely the image belongs to the training set.

# %%
sns.set_style("whitegrid", {"axes.grid": False})
sns.kdeplot(in_likelihoods, bw_adjust=1, label="In-distribution", fill=True, cut=True)
for c, l in all_likelihoods.items():
    sns.kdeplot(l, bw_adjust=1, label=f"OOD {c}", cut=True, fill=True)
plt.legend(loc="upper right")
plt.xlabel("Log-likelihood")
# plt.xlim([-200,10])
# plt.ylim([0,10])

# %% [markdown]
# # Localised anomaly detection
# First we create a synthetic corruption of an in-distribution image

# %%
input_image = first(in_distribution_loader)
image_clean = input_image["image"][0, ...]
plt.subplot(1, 2, 1)
plt.imshow(image_clean[0, ...], cmap="gray")
plt.axis("off")
plt.title("Clean image")
image_corrupted = image_clean.clone()
image_corrupted[0, 25:40, 40:50] = 1
plt.subplot(1, 2, 2)
plt.imshow(image_corrupted[0, ...], cmap="gray")
plt.axis("off")
plt.title("Corrupted image")
plt.show()

# %% [markdown]
# Get the log-likelihood and convert into a mask of the 5% lowest-likelihood tokens

# %%
log_likelihood = inferer.get_likelihood(
    inputs=image_corrupted[None, ...].to(device),
    vqvae_model=vqvae_model,
    transformer_model=transformer_model,
    ordering=ordering,
)
likelihood = torch.exp(log_likelihood)
plt.subplot(1, 2, 1)
plt.imshow(likelihood.cpu()[0, ...])
plt.axis("off")
plt.title("Log-likelihood")
plt.subplot(1, 2, 2)
mask = log_likelihood.cpu()[0, ...] < torch.quantile(log_likelihood, 0.03).item()
plt.imshow(mask)
plt.axis("off")
plt.title("Healing mask")
plt.show()

# %% [markdown]
# Use this mask and the trained transformer to 'heal' the sequence

# %%
# flatten the mask
mask_flattened = mask.reshape(-1)
mask_flattened = mask_flattened[ordering.get_sequence_ordering()]

latent = vqvae_model.index_quantize(image_corrupted[None, ...].to(device))
latent = latent.reshape(latent.shape[0], -1)
latent = latent[:, ordering.get_sequence_ordering()]
latent = F.pad(latent, (1, 0), "constant", vqvae_model.num_embeddings)
latent = latent.long()
latent_healed = latent.clone()

# heal the sequence
# loop over tokens
for i in range(1, latent.shape[1]):
    if mask_flattened[i - 1]:
        # if token is low probability, replace with tranformer's most likely token
        logits = transformer_model(latent_healed[:, :i])
        probs = F.softmax(logits, dim=-1)
        # don't sample beginning of sequence token
        probs[:, :, vqvae_model.num_embeddings] = 0
        index = torch.argmax(probs[0, -1, :])
        latent_healed[:, i] = index


# reconstruct
latent_healed = latent_healed[:, 1:]
latent_healed = latent_healed[:, ordering.get_revert_sequence_ordering()]
latent_healed = latent_healed.reshape((16, 16))

image_healed = vqvae_model.decode_samples(latent_healed[None, ...]).cpu().detach()
plt.imshow(image_healed[0, 0, ...], cmap="gray")
plt.axis("off")
plt.title("Healed image")
plt.show()

# %% [markdown]
# ## Create anomaly maps

# %%
# Get a naive anomaly map using the difference
difference_map = torch.abs(image_healed[0, 0, ...] - image_corrupted[0, ...])

# Further mask with the healing mask
resizer = torch.nn.Upsample(size=(64, 64), mode="nearest")
mask_upsampled = resizer(mask[None, None, ...].float()).int()

fig, ax = plt.subplots(1, 4, figsize=(14, 8))
plt.subplot(1, 4, 1)
plt.imshow(image_clean[0, ...], cmap="gray")
plt.axis("off")
plt.title("Clean image")
image_corrupted = image_clean.clone()
image_corrupted[0, 25:40, 40:50] = 1
plt.subplot(1, 4, 2)
plt.imshow(image_corrupted[0, ...], cmap="gray")
plt.axis("off")
plt.title("Corrupted image")
plt.subplot(1, 4, 3)
plt.imshow(image_corrupted[0, ...] - image_clean[0, ...], cmap="gray")
plt.axis("off")
plt.title("Ground-Truth anomaly mask")
plt.subplot(1, 4, 4)
plt.imshow(mask_upsampled[0, 0, ...] * difference_map, cmap="gray")
plt.axis("off")
plt.title("Predicted anomaly mask")
plt.show()
