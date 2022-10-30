""" Tutorial for training an unconditioned Latent Diffusion Model on MEDNIST
Based on
https://github.com/Project-MONAI/tutorials/blob/main/2d_registration/registration_mednist.ipynb
https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb
"""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from tqdm import tqdm

from generative.networks.nets import DiffusionModelUNet
from generative.schedulers import DDPMScheduler

print_config()
set_determinism(42)

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


train_data = MedNISTDataset(root_dir=root_dir, section="training", download=True, seed=0)
train_datalist = [{"image": item["image"]} for item in train_data.data if item["class_name"] == "Hand"]
# TODO: Add affine
# TODO: Add correct flip
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
            prob=1,
        ),
    ]
)
train_ds = Dataset(data=train_datalist, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)


val_data = MedNISTDataset(root_dir=root_dir, section="validation", download=True, seed=0)
val_datalist = [{"image": item["image"]} for item in train_data.data if item["class_name"] == "Hand"]
val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    ]
)
val_ds = Dataset(data=val_datalist, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=True, num_workers=4)

device = torch.device("cuda")

model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    model_channels=64,
    attention_resolutions=[4, 2],
    num_res_blocks=1,
    channel_mult=[1, 1, 2],
    num_heads=1,
)
model.to(device)

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
)

optimizer = torch.optim.Adam(model.parameters(), 2.5e-5)

n_epochs = 50
val_interval = 10
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)

        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
        noise = torch.randn_like(images).to(device)
        noisy_image = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
        noise_pred = model(x=noisy_image, timesteps=timesteps)

        loss = F.l1_loss(noise_pred.float(), noise.float())

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        progress_bar.set_postfix(
            {
                "loss": epoch_loss / (step + 1),
            }
        )


model.eval()
image = torch.randn(
    (1, 1, 64, 64),
)
image = image.to(device)
scheduler.set_timesteps(1000)

intermediary = []
for t in tqdm(scheduler.timesteps):
    # 1. predict noise model_output
    with torch.no_grad():
        model_output = model(image, torch.asarray((t,)).to(device))

    # 2. compute previous image: x_t -> x_t-1
    image, _ = scheduler.step(model_output, t, image)
    if t % 200 == 0:
        intermediary.append(image)

intermediary.append(image)

plt.imshow(image[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
plt.axis("off")
plt.show()

chain = torch.concat(intermediary, dim=-1)
plt.imshow(chain[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
plt.axis("off")
plt.show()

# TODO: Get fidelity and variability metrics
