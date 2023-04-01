# +
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
# -

# # Finetuning Stable Diffusion to Generate 2D Medical Images
#
# In this tutorial, we will convert the Stable Diffusion weights to be loaded using MONAI Generative Model classes. Next, we will use a similar approach presented in [1,2] and finetune (and train from scratch) the second stage of the latent diffusion model.
#
# [1] - Chambon et al. "RoentGen: Vision-Language Foundation Model for Chest X-ray Generation." https://arxiv.org/abs/2211.12737
#
# [2] - Chambon et al. "Adapting Pretrained Vision-Language Foundational Models to Medical Imaging Domains." https://arxiv.org/abs/2210.04133

# ## Setup imports

# +
import os
import tempfile
import time
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import sys
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast


print_config()
# -

# ### Setup data directory

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory

# ### Set deterministic training for reproducibility

set_determinism(42)

# ## Setup BRATS Dataset  - Transforms for extracting 2D slices from 3D volumes
#
# We now download the BraTS dataset and extract the 2D slices from the 3D volumes. The `slice_label` is used to indicate whether the slice contains an anomaly or not.
#
# Here we use transforms to augment the training dataset, as usual:
#
# 1. `LoadImaged` loads the brain images from files.
# 2. `EnsureChannelFirstd` ensures the original data to construct "channel first" shape.
# 3.  The first `Lambdad` transform chooses the first channel of the image, which is the T1-weighted image.
# 4. `Spacingd` resamples the image to the specified voxel spacing, we use 3,3,2 mm to match the original paper.
# 5. `ScaleIntensityRangePercentilesd` Apply range scaling to a numpy array based on the intensity distribution of the input. Transform is very common with MRI images.
# 6. `RandSpatialCropd` randomly crop out a 2D patch from the 3D image.
# 6. The last `Lambdad` transform obtains `slice_label` by summing up the label to have a single scalar value (healthy `=1` or not `=2` ).

# +
channel = 0  # 0 = Flair
assert channel in [0, 1, 2, 3], "Choose a valid channel"

train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.Lambdad(keys=["image"], func=lambda x: x[channel, :, :, :]),
        transforms.AddChanneld(keys=["image"]),
        transforms.EnsureTyped(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"], pixdim=(3.0, 3.0, 2.0), mode=("bilinear", "nearest")),
        transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=(64, 64, 44)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        transforms.RandSpatialCropd(keys=["image", "label"], roi_size=(64, 64, 1), random_size=False),
        transforms.Lambdad(keys=["image", "label"], func=lambda x: x.squeeze(-1)),
        transforms.CopyItemsd(keys=["label"], times=1, names=["slice_label"]),
        transforms.Lambdad(keys=["slice_label"], func=lambda x: 2.0 if x.sum() > 0 else 1.0),
    ]
)
# -

# ### Load Training and Validation Datasets

# +
train_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    section="training",
    cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=4,
    download=False,  # Set download to True if the dataset hasnt been downloaded yet
    seed=0,
    transform=train_transforms,
)
print(f"Length of training data: {len(train_ds)}")
print(f'Train image shape {train_ds[0]["image"].shape}')

val_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    section="validation",
    cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=4,
    download=False,  # Set download to True if the dataset hasnt been downloaded yet
    seed=0,
    transform=train_transforms,
)
print(f"Length of training data: {len(val_ds)}")
print(f'Validation Image shape {val_ds[0]["image"].shape}')
# -

# ## Converting Stable Diffusion weights



# ## Finetuning Diffusion Model
#
# At this step, we instantiate the MONAI components to create a DDIM, the UNET with conditioning, the noise scheduler, and the inferer used for training and sampling. We are using
# the deterministic DDIM scheduler containing 1000 timesteps, and a 2D UNET with attention mechanisms.
#
# The `attention` mechanism is essential for ensuring good conditioning and images manipulation here.
#
# An `embedding layer`, which is also optimised during training, is used in the original work because it was empirically shown to improve conditioning compared to a single scalar information.

# +
condition_dropout = 0.15
n_iterations = 2e4
batch_size = 64
val_interval = 100
iter_loss_list = []
val_iter_loss_list = []
iterations = []
iteration = 0
iter_loss = 0

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True
)

scaler = GradScaler()
total_start = time.time()

while iteration < n_iterations:
    for batch in train_loader:
        iteration += 1
        model.train()
        images, classes = batch["image"].to(device), batch["slice_label"].to(device)
        # 15% of the time, class conditioning dropout
        classes = classes * (torch.rand_like(classes) > condition_dropout)
        # cross attention expects shape [batch size, sequence length, channels]
        class_embedding = embed(classes.long().to(device)).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        # pick a random time step t
        timesteps = torch.randint(0, 1000, (len(images),)).to(device)

        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(images).to(device)
            # Get model prediction
            noise_pred = inferer(
                inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps, condition=class_embedding
            )
            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        iter_loss += loss.item()
        sys.stdout.write(f"Iteration {iteration}/{n_iterations} - train Loss {loss.item():.4f}" + "\r")
        sys.stdout.flush()

        if (iteration) % val_interval == 0:
            model.eval()
            val_iter_loss = 0
            for val_step, val_batch in enumerate(val_loader):
                images, classes = val_batch["image"].to(device), val_batch["slice_label"].to(device)
                # cross attention expects shape [batch size, sequence length, channels]
                class_embedding = embed(classes.long().to(device)).unsqueeze(1)
                timesteps = torch.randint(0, 1000, (len(images),)).to(device)
                with torch.no_grad():
                    with autocast(enabled=True):
                        noise = torch.randn_like(images).to(device)
                        noise_pred = inferer(
                            inputs=images,
                            diffusion_model=model,
                            noise=noise,
                            timesteps=timesteps,
                            condition=class_embedding,
                        )
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())
                val_iter_loss += val_loss.item()
            iter_loss_list.append(iter_loss / val_interval)
            val_iter_loss_list.append(val_iter_loss / (val_step + 1))
            iterations.append(iteration)
            iter_loss = 0
            print(
                f"Train Loss {loss.item():.4f}, Interval Loss {iter_loss_list[-1]:.4f}, Interval Loss Val {val_iter_loss_list[-1]:.4f}"
            )


total_time = time.time() - total_start

print(f"train diffusion completed, total time: {total_time}.")

plt.style.use("seaborn-bright")
plt.title("Learning Curves Diffusion Model", fontsize=20)
plt.plot(iterations, iter_loss_list, color="C0", linewidth=2.0, label="Train")
plt.plot(
    iterations, val_iter_loss_list, color="C1", linewidth=2.0, label="Validation"
)  # np.linspace(1, n_iterations, len(val_iter_loss_list))
plt.yticks(fontsize=12), plt.xticks(fontsize=12)
plt.xlabel("Iterations", fontsize=16), plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.show()
