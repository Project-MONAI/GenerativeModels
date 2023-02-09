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
# # Anomaly Detection with classifier guidance
#
# This tutorial illustrates how to use MONAI for training a 2D gradient-guided anomaly detection using DDIMs [1].
#
#
# [1] - Wolleb et al. "Diffusion Models for Medical Anomaly Detection" https://arxiv.org/abs/2203.04306
#
#
# TODO: Add Open in Colab
#
# ## Setup environment

# %%
# !python /home/juliawolleb/PycharmProjects/MONAI/GenerativeModels/setup.py install
# !python -c "import monai" || pip install -q "monai-weekly[pillow, tqdm, einops]"
# !python -c "import matplotlib" || pip install -q matplotlib
# !python -c "import seaborn" || pip install -q seaborn
# %matplotlib inline

# %% [markdown]
# ## Setup imports

# %% jupyter={"outputs_hidden": false}
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
import os
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import MedNISTDataset, DecathlonDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from generative.inferers import DiffusionInferer

# TODO: Add right import reference after deployed
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet, DiffusionModelEncoder

from generative.networks.schedulers.ddpm import DDPMScheduler
from generative.networks.schedulers.ddim import DDIMScheduler
print_config()

train=False


# %% [markdown]
# ## Setup data directory

# %% jupyter={"outputs_hidden": false}
directory = os.environ.get("MONAI_DATA_DIRECTORY")
#root_dir = tempfile.mkdtemp() if directory is None else directory
root_dir='/home/juliawolleb/PycharmProjects/MONAI/val_brats'
root_dir_val='/home/juliawolleb/PycharmProjects/MONAI/val_brats'

print(root_dir, root_dir_val)

# %% [markdown]
# ## Set deterministic training for reproducibility

# %% jupyter={"outputs_hidden": false}
set_determinism(42)

# %% [markdown]
# ## Setup BRATS Dataset for 2D slices  and training and validation dataloaders
# As baseline, we use the load_2d_brats.ipynb written by Pedro in issue 150

# %% jupyter={"outputs_hidden": false}


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
print('download training set')
train_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    section="training",  # validation
    cache_rate=0.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=4,
    download=False,  # Set download to True if the dataset hasnt been downloaded yet
    seed=0,
    transform=train_transforms,
)
nb_3D_images_to_mix =20
train_loader_3D = DataLoader(train_ds, batch_size=nb_3D_images_to_mix, shuffle=True, num_workers=4)

print(f'Image shape {train_ds[0]["image"].shape}')



print('download val set')

# %%

val_ds = DecathlonDataset(
    root_dir=root_dir_val,
    task="Task01_BrainTumour",
    section="validation",  # validation
    cache_rate=0.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=4,
    download=False,  # Set download to True if the dataset hasnt been downloaded yet
    seed=0,
    transform=train_transforms,
)
val_loader_3D = DataLoader(val_ds, batch_size=2, shuffle=True, num_workers=4)
print(f'Image shape {val_ds[0]["image"].shape}')



# %% [markdown]
# Here we use transforms to augment the training dataset, as usual:
#
# 1. `LoadImaged` loads the hands images from files.
# 1. `EnsureChannelFirstd` ensures the original data to construct "channel first" shape.
# 1. `ScaleIntensityRanged` extracts intensity range [0, 255] and scales to [0, 1].
# 1. `RandAffined` efficiently performs rotate, scale, shear, translate, etc. together based on PyTorch affine transform.
#
#

# %% [markdown]
# ### Visualisation of the training images

# %% jupyter={"outputs_hidden": false}


from typing import Dict
def get_batched_2d_axial_slices(data : Dict):
    images_3D = data['image']
    batched_2d_slices = torch.cat(images_3D.split(1, dim = -1), 0).squeeze(-1) # images_3D.view(images_3D.shape[0]*images_3D.shape[-1],*images_3D.shape[1:-1])
    slice_label = data['slice_label']
    #slice_label = (mask_label.reshape(mask_label.shape[0], -1, mask_label.shape[-1]).sum(1) > 0 ).float()
    slice_label = torch.cat(slice_label.split(1, dim = -1),0).squeeze()
    return batched_2d_slices, slice_label
print('check data')

if train==True:
    check_data = first(train_loader_3D)
    batched_2d_slices, slice_label = get_batched_2d_axial_slices(check_data)
    idx = list(torch.randperm(batched_2d_slices.shape[0]))
    print('idx', len(idx))
    print(f"Batch shape: {batched_2d_slices.shape}")
    print(f"Slices class: {slice_label[idx][slices].view(-1)}")
    subset_2D = zip(batched_2d_slices.split(batch_size), slice_label.split(batch_size))  #

check_data_val = first(val_loader_3D)
batched_2d_slices_val, slice_label_val = get_batched_2d_axial_slices(check_data_val)



idx_val=list(torch.randperm(batched_2d_slices_val.shape[0]))
slices = [0,30,45,63]

image_visualisation = torch.cat(batched_2d_slices_val[idx_val][slices].squeeze().split(1), dim=2).squeeze()
plt.figure("training images", (12, 6))
plt.imshow(image_visualisation, vmin=0, vmax=1, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()

# %%

subset_2D_val = zip(batched_2d_slices_val.split(1),slice_label_val.split(1))#



# %% [markdown]
# ### Define network, scheduler, optimizer, and inferer
# At this step, we instantiate the MONAI components to create a DDPM, the UNET, the noise scheduler, and the inferer used for training and sampling. We are using
# the original DDPM scheduler containing 1000 timesteps in its Markov chain, and a 2D UNET with attention mechanisms
# in the 3rd level, each with 1 attention head (`num_head_channels=64`).
#
# In order to pass conditioning variables with dimension of 1 (just specifying the modality of the image), we use:
#
# `
# with_conditioning=True,
# cross_attention_dim=1,
# `

# %% jupyter={"outputs_hidden": false}
device = torch.device("cuda")

model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(64, 64, 64),
    attention_levels=(False, False, True),
    num_res_blocks=1,
    num_head_channels=64,
    with_conditioning=False,
  #  cross_attention_dim=1,
)
model.to(device)

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
)

optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)

inferer = DiffusionInferer(scheduler)
# %% [markdown] tags=[]
# ### Model training of the Diffusion Model
# Here, we are training our diffusion model for 75 epochs (training time: ~50 minutes).

# %% jupyter={"outputs_hidden": false}
n_epochs =100
val_interval = 1
epoch_loss_list = []
val_epoch_loss_list = []

if train==False:
    model.load_state_dict(torch.load("./model.pt", map_location={'cuda:0': 'cpu'}))
else:
    scaler = GradScaler()
    total_start = time.time()
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        subset_2D = zip(batched_2d_slices.split(batch_size), slice_label.split(batch_size))
        subset_2D_val = zip(batched_2d_slices_val.split(1), slice_label.split(1))  #

        progress_bar = tqdm(enumerate(subset_2D), total=len(idx), ncols=10)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, (a,b) in progress_bar:
            print('step', step, a.shape, b.shape, b)
            images = a.to(device)
            classes = b.to(device)
            optimizer.zero_grad(set_to_none=True)
            timesteps = torch.randint(0, 1000, (len(images),)).to(device)

            with autocast(enabled=True):
                # Generate random noise
                noise = torch.randn_like(images).to(device)

                # Get model prediction
                noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)  #remove the class conditioning

                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            progress_bar.set_postfix(
                {
                    "loss": epoch_loss / (step + 1),
                }
            )
        epoch_loss_list.append(epoch_loss / (step + 1))

        if (epoch) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            progress_bar_val = tqdm(enumerate(subset_2D_val), total=len(idx_val), ncols=70)
            progress_bar.set_description(f"Epoch {epoch}")
            for    step, (a, b) in progress_bar_val:
                images = a.to(device)
                classes = b.to(device)
                timesteps = torch.randint(0, 1000, (len(images),)).to(device)#torch.from_numpy(np.arange(0, 1000)[::-1].copy())

                with torch.no_grad():
                    with autocast(enabled=True):
                        noise = torch.randn_like(images).to(device)
                        noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())

                val_epoch_loss += val_loss.item()
                progress_bar.set_postfix(
                    {
                        "val_loss": val_epoch_loss / (step + 1),
                    }
                )
            val_epoch_loss_list.append(val_epoch_loss / (step + 1))

    total_time = time.time() - total_start
    print(f"train diffusion completed, total time: {total_time}.")

    plt.style.use("seaborn-bright")
    plt.title("Learning Curves Diffusion Model", fontsize=20)
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
    #plt.show()
    #torch.save(model.state_dict(), "./model.pt")


# %%
### Model training of the Classification Model
#Here, we are training our binary classification model for 5 epochs.

# %%
## First, we define the classification model


# %%
classifier = DiffusionModelEncoder(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(64, 64, 64),
   # attention_levels=(False, False, True),
    num_res_blocks=1,
    num_head_channels=64,
    with_conditioning=False,
  #  cross_attention_dim=1,
)
classifier.to(device)
batch_size=6


# %%
n_epochs = 100
val_interval = 1
epoch_loss_list = []
val_epoch_loss_list = []
optimizer = torch.optim.Adam(params=classifier.parameters(), lr=2.5e-5)

classifier.to(device)


if train==False:
    classifier.load_state_dict(torch.load("./classifier.pt", map_location={'cuda:0': 'cpu'}))
else:
    scaler = GradScaler()
    total_start = time.time()
    for epoch in range(n_epochs):
        classifier.train()
        epoch_loss = 0
        subset_2D = zip(batched_2d_slices.split(batch_size), slice_label.split(batch_size))
        subset_2D_val = zip(batched_2d_slices_val.split(1), slice_label.split(1))  #
        progress_bar = tqdm(enumerate(subset_2D), total=len(idx), ncols=20)
        progress_bar.set_description(f"Epoch {epoch}")


        for step, (a,b) in progress_bar:
            images = a.to(device)
            classes = b.to(device)
            optimizer.zero_grad(set_to_none=True)
            timesteps = torch.randint(0, 1000, (len(images),)).to(device)

            with autocast(enabled=True):
                # Generate random noise
                noise = 0*torch.randn_like(images).to(device)

                # Get model prediction
               # pred=classifier(images)

                pred = inferer(inputs=images, diffusion_model=classifier, noise=noise, timesteps=timesteps)  #remove the class conditioning
                print('pred', pred)
              #  noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)  #remove the class conditioning
                loss = F.binary_cross_entropy_with_logits(pred[:,0].float(), classes.float())
                print('loss', loss)
            #scaler.scale(loss).backward()
           # scaler.step(optimizer)
            loss.backward()
            optimizer.step()
            #scaler.update()

            epoch_loss += loss.item()

            progress_bar.set_postfix(
                {
                    "loss": epoch_loss / (step + 1),
                }
            )
        epoch_loss_list.append(epoch_loss / (step + 1))

        if (epoch + 1) % val_interval == 0:
            classifier.eval()
            val_epoch_loss = 0
            progress_bar = tqdm(enumerate(subset_2D_val), total=len(idx), ncols=70)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, (a,b) in progress_bar:
                images = a.to(device)
                classes = b.to(device)

                timesteps = torch.randint(0, 1000, (len(images),)).to(device)#torch.from_numpy(np.arange(0, 1000)[::-1].copy())

                with torch.no_grad():
                    with autocast(enabled=True):
                        noise = 0*torch.randn_like(images).to(device)
                        pred = inferer(inputs=images, diffusion_model=classifier, noise=noise, timesteps=timesteps)
                        val_loss = F.binary_cross_entropy_with_logits(pred[:,0].float(), classes.float())

                val_epoch_loss += val_loss.item()
                progress_bar.set_postfix(
                    {
                        "val_loss": val_epoch_loss / (step + 1),
                    }
                )
            val_epoch_loss_list.append(val_epoch_loss / (step + 1))

    total_time = time.time() - total_start
    print(f"train completed, total time: {total_time}.")
 #   torch.save(classifier.state_dict(), "./classifier.pt")
# %% [markdown]
# ### Learning curves

# %% jupyter={"outputs_hidden": false}
    plt.style.use("seaborn-bright")
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
    #plt.show()

# %% [markdown]
# ### Sampling process with classifier-free guidance
# In order to sample using classifier-free guidance, for each step of the process we need to have 2 elements, one generated conditioned in the desired class (here we want to condition on Hands `=1`) and one using the unconditional class (`=-1`).
# Instead using directly the predicted class in every step, we use the unconditional plus the direction vector pointing to the condition that we want (`noise_pred_text - noise_pred_uncond`). The effect of the condition is defined by the `guidance_scale` defining the influence of our direction vector.

# %% jupyter={"outputs_hidden": false}
model.eval()
guidance_scale = 0
conditioning = torch.cat([-1 * torch.ones(1, 1, 1).float(), torch.ones(1, 1, 1).float()], dim=0).to(device)

# %% [markdown]
# ### Pick an input slice to be transformed

inputimg = batched_2d_slices_val[50][0,...]
plt.figure("input")
plt.imshow(inputimg, vmin=0, vmax=1, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()


noise = inputimg[None,None,...]#torch.randn((1, 1, 64, 64))
noise = noise.to(device)
scheduler.set_timesteps(num_inference_steps=1000)
L=20
progress_bar = tqdm(range(L))   #go back and forth L timesteps



for t in progress_bar:  #go through the noising process
    print('t noising', t)

    with autocast(enabled=True):
        with torch.no_grad():

            noise_input = noise
            print('inputshape', noise_input.shape)
            model_output = model(noise_input, timesteps=torch.Tensor((t,)).to(noise.device))
          #  noise_pred_uncond, noise_pred_text = model_output.chunk(2)    #this is supposed to be epsilon
            noise_pred = model_output  #this is supposed to be epsilon

    noise, _ = scheduler.reversed_step(noise_pred, t, noise)

plt.style.use("default")
plt.imshow(noise[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
plt.tight_layout()
plt.axis("off")
plt.show()


def cond_fn(x, t, y=None):  #compute the gradient
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        a = th.autograd.grad(selected.sum(), x_in)[0]
        return a, a * args.classifier_scale
#desired class
y=torch.tensor(0)
scale=100

for i in progress_bar:  #go through the denoising process
    t=L-i
    print('t denoising', t)
    with autocast(enabled=True):
        with torch.enable_grad():
            noise_input = noise
            print('inputshape', noise_input.shape)
            model_output = model(noise_input, timesteps=torch.Tensor((t,)).to(noise.device))

            x_in = noise_input.detach().requires_grad_(True)

            logits = classifier(x_in, timesteps=torch.Tensor((t,)).to(noise.device))
            print('logits', logits)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            a = torch.autograd.grad(selected.sum(), x_in)[0]
            #  noise_pred_uncond, noise_pred_text = model_output.chunk(2)    #this is supposed to be epsilon
            noise_pred = model_output  # this is supposed to be epsilon
            updated_noise=noise_pred - scale*a

    noise, _ = scheduler.step(updated_noise, t, noise)

plt.style.use("default")
plt.imshow(noise[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
plt.tight_layout()
plt.axis("off")
plt.show()


diff=inputimg.cpu()-noise[0, 0].cpu()
plt.style.use("default")
plt.imshow(diff, cmap="jet")
plt.tight_layout()
plt.axis("off")
plt.show()
# %% [markdown]
# ### Cleanup data directory
#
# Remove directory if a temporary was used.

# %%

