# # 3D AutoencoderKL

# %cd /mnt_homes/home4T7/jdafflon/GenerativeModels

# ## Setup imports

# +
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from tqdm import tqdm

from generative.networks.nets import AutoencoderKL

print_config()
# -

# for reproducibility purposes set a seed
set_determinism(42)

# ## Setup a data directory and download dataset
#
# Specify a `MONAI_DATA_DIRECTORY` variable, where the data will be downloaded. If not specified a temporary directory will be used.

directory = os.environ.get("MONAI_DATA_DIRECTORY")
# root_dir = tempfile.mkdtemp() if directory is None else directory
root_dir = "/tmp/tmpyxyg6wxs"
print(root_dir)

# ## Download the training set

# Note: The DecatholonDataset has 7GB. So make sure that you have enought space when running the next line

# +
channel = 0  # 0 = Flair
assert channel in [0, 1, 2, 3], "Choose a valid channel"

train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.Spacingd(
            keys=["image"],
            pixdim=(2.0, 2.0, 2.0),
            mode=("bilinear"),
        ),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=(128, 128, 64)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
    ]
)
train_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    section="training",
    cache_rate=1.0,
    num_workers=4,
    download=False,
    seed=0,
    transform=train_transforms,
)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
print(f'Image shape {train_ds[0]["image"].shape}')
# -

# ## Visualise examples from the training set

# +
check_data = first(train_loader)
idx = 0

img = check_data["image"][idx, channel]
fig, axs = plt.subplots(nrows=1, ncols=3)
for ax in axs:
    ax.axis("off")
ax = axs[0]
ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
ax = axs[1]
ax.imshow(img[:, img.shape[1] // 2, ...].rot90(), cmap="gray")
ax = axs[2]
ax.imshow(img[img.shape[0] // 2, ...].rot90(), cmap="gray")
# -

# ## Download the validation set

val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.Spacingd(
            keys=["image"],
            pixdim=(2.0, 2.0, 2),
            mode=("bilinear"),
        ),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=(120, 120, 64)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
    ]
)
val_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    section="validation",
    cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=4,
    download=True,
    seed=0,
    transform=val_transforms,
)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4)
print(f'Image shape {val_ds[0]["image"].shape}')

# ## Define the network

# +
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

model = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=64,
    latent_channels=8,
    ch_mult=(1, 2, 3),
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False, False, True),
)
model.to(device)
# -

# ## Model training

# +
kl_weight = 1e-6
optimizer = torch.optim.Adam(model.parameters(), 2.5e-5)
n_epochs = 10
val_interval = 2
epoch_loss_list = []
val_epoch_loss_list = []
intermediary_images = []
n_example_images = 4

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        # select only one channel from the Decathlon dataset
        one_channel = batch["image"][:, None, channel, ...]
        images = one_channel.to(device)
        optimizer.zero_grad(set_to_none=True)
        reconstruction, z_mu, z_sigma = model(images)

        mse_loss = F.mse_loss(reconstruction.float(), images.float())

        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # TODO: Add adversarial component
        # TODO: Add perceptual loss

        loss = mse_loss + kl_weight * kl_loss
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
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                # select only one channel from the Decathlon dataset
                one_channel = batch["image"][:, None, channel, ...]
                images = one_channel.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction, z_mu, z_sigma = model(images)
                # get the first sammple from the first validation batch for visualisation
                # purposes
                if val_step == 1:
                    intermediary_images.append(reconstruction[:n_example_images, 0])

                mse_loss = F.mse_loss(reconstruction.float(), images.float())

                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                # TODO: Add adversarial component
                # TODO: Add perceptual loss

                loss = mse_loss + kl_weight * kl_loss
                val_loss += loss.item()

        val_loss /= val_step
        val_epoch_loss_list.append(val_loss)

        print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
progress_bar.close()
# -
# ## Evaluate the trainig

plt.figure()
val_samples = np.linspace(val_interval, n_epochs, int(n_epochs / val_interval))
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_loss_list, label="Train")
plt.plot(val_samples, val_epoch_loss_list, label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

# ### Visualise some reconstruction images

# +
img = check_data["image"][idx, channel]
fig, axs = plt.subplots(nrows=len(intermediary_images), ncols=3, constrained_layout=True, figsize=(8, 6))

# Remove ticks
for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

for image_n in range(len(intermediary_images)):
    axs[image_n, 0].imshow(intermediary_images[image_n][0, ..., img.shape[2] // 2].cpu(), cmap="gray")
    axs[image_n, 1].imshow(intermediary_images[image_n][0, :, img.shape[1] // 2, ...].cpu().rot90(), cmap="gray")
    axs[image_n, 2].imshow(intermediary_images[image_n][0, img.shape[0] // 2, ...].cpu().rot90(), cmap="gray")
    axs[image_n, 0].set_ylabel(f"Epoch {val_samples[image_n]:.0f}")