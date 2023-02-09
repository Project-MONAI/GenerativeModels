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

# # Diff-SCM
# 
# This tutorial illustrates how to load the 2D BRATS dataset.
# 
# 
# ## Setup environment

# %%


get_ipython().system('python -c "import monai" || pip install -q "monai-weekly[pillow, tqdm, einops]"')
get_ipython().system('python -c "import matplotlib" || pip install -q matplotlib')
get_ipython().run_line_magic('matplotlib', 'inline')
print('done')


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
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from generative.inferers import DiffusionInferer

# TODO: Add right import reference after deployed
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

print_config()


# ## Setup data directory

# %%


directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)
root_dir= '/tmp/tmp6o69ziv1'


# ## Set deterministic training for reproducibility

# %%


set_determinism(42)


# ## Setup MedNIST Dataset and training and validation dataloaders
# In this tutorial, we will train our models on the MedNIST dataset available on MONAI
# (https://docs.monai.io/en/stable/apps.html#monai.apps.MedNISTDataset).
# Here, we will use the "Hand" and "HeadCT", where our conditioning variable `class` will specify the modality.

# %%


batch_size = 2
channel = 0  # 0 = Flair
assert channel in [0, 1, 2, 3], "Choose a valid channel"

train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image","label"]),
        transforms.EnsureChannelFirstd(keys=["image","label"]),
        transforms.Lambdad(keys=["image"], func=lambda x: x[channel, :, :, :]),
        transforms.AddChanneld(keys=["image"]),
        transforms.EnsureTyped(keys=["image","label"]),
        transforms.Orientationd(keys=["image","label"], axcodes="RAS"),
        transforms.Spacingd(
            keys=["image","label"],
            pixdim=(3.0, 3.0, 2.0),
            mode=("bilinear", "nearest"),
        ),
        transforms.CenterSpatialCropd(keys=["image","label"], roi_size=(64, 64, 64)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        transforms.CopyItemsd(keys=["label"], times=1, names=["slice_label"]),
        transforms.Lambdad(keys=["slice_label"], func=lambda x: (x.reshape(x.shape[0], -1, x.shape[-1]).sum(1) > 0 ).float().squeeze()),
    ]
)
train_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    section="training",  # validation
    cache_rate=0.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=4,
    download=True,  # Set download to True if the dataset hasnt been downloaded yet
    seed=0,
    transform=train_transforms,
)
nb_3D_images_to_mix = 2
train_loader_3D = DataLoader(train_ds, batch_size=nb_3D_images_to_mix, shuffle=True, num_workers=4)
print(f'Image shape {train_ds[0]["image"].shape}')


# %%


from typing import Dict
def get_batched_2d_axial_slices(data : Dict):
    images_3D = data['image']
    batched_2d_slices = torch.cat(images_3D.split(1, dim = -1), 0).squeeze(-1) # images_3D.view(images_3D.shape[0]*images_3D.shape[-1],*images_3D.shape[1:-1])
    slice_label = data['slice_label']
    #slice_label = (mask_label.reshape(mask_label.shape[0], -1, mask_label.shape[-1]).sum(1) > 0 ).float()
    slice_label = torch.cat(slice_label.split(1, dim = -1),0).squeeze()
    return batched_2d_slices, slice_label


# ### Visualisation of the training images

# %%


check_data = first(train_loader_3D)
print('check_data', check_data["image"].shape, check_data["slice_label"].shape)


# %%


batched_2d_slices, slice_label = get_batched_2d_axial_slices(check_data)
idx = list(torch.randperm(batched_2d_slices.shape[0]))
print('idx', idx, len(idx))
slices = [0,30,45,63]
print(f"Batch shape: {batched_2d_slices.shape}")
print(f"Slices class: {slice_label[idx][slices].view(-1)}")
image_visualisation = torch.cat(batched_2d_slices[idx][slices].squeeze().split(1), dim=2).squeeze()
plt.figure("training images", (12, 6))
plt.imshow(image_visualisation, vmin=0, vmax=1, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()


# %%


slice_label.shape


# ## Check Distribution of Healthy / Unhealthy

# %%

subset_2D = zip(batched_2d_slices.split(batch_size),slice_label.split(batch_size))#
a,b = next(subset_2D)  #what is a, what is b?
a.shape, b.shape


# %%


plt.hist(slice_label.view(-1).numpy(),bins = 5);
plt.title("Distribution of slices with and without tumour \n 0 = no tumour, 1 = tumour");


# %%
