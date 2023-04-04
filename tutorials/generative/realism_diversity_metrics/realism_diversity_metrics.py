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

# # Evaluate Realism and Diversity of the generated images

# This notebook illustrates how to use the generative model package to compute:
# - the realism of generated image using the s Frechet Inception Distance (FID) [1] and Maximum Mean Discrepancy (MMD) [2]
# - the image diversity using the MS-SSIM [3] and SSIM [4]
#
# Note: We are using the RadImageNet [5] to compute the feature space necessary to compute the FID.
#
# [1] - Heusel et al., "Gans trained by a two time-scale update rule converge to a local nash equilibrium", https://arxiv.org/pdf/1706.08500.pdf
#
# [2] - Gretton et al., "A Kernel Two-Sample Test", https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
#
# [3] - Wang et al., "Multiscale structural similarity for image quality assessment", https://ieeexplore.ieee.org/document/1292216
#
# [4] - Wang et al., "Image quality assessment: from error visibility to structural similarity", https://ieeexplore.ieee.org/document/1284395
#
# [5] = Mei et al., "RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning, https://pubs.rsna.org/doi/10.1148/ryai.210315

# ## Setup environment

# +
import torch
import os
import torch
from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt
from monai.apps import MedNISTDataset
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.networks.layers import Act


from monai.config import print_config
from monai.utils import set_determinism

from generative.metrics import FIDMetric, MMD, MultiScaleSSIMMetric, SSIMMetric
from generative.networks.nets import DiffusionModelUNet, PatchDiscriminator, AutoencoderKL
from generative.networks.schedulers import DDIMScheduler
from generative.inferers import DiffusionInferer

print_config()


# -

# The transformations defined below are necessary in order to transform the input images in the same way that the images were
# processed for the RadNet train.

# +
def subtract_mean(x: torch.Tensor) -> torch.Tensor:
    mean = [0.406, 0.456, 0.485]
    x[:, 0, :, :] -= mean[0]
    x[:, 1, :, :] -= mean[1]
    x[:, 2, :, :] -= mean[2]
    return x

def normalize_tensor(x: torch.Tensor, eps: float=1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)

def spatial_average(x: torch.Tensor, keepdim: bool=True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)

def get_features(image):

    # If input has just 1 channel, repeat channel to have 3 channels
    if image.shape[1]:
        image = image.repeat(1, 3, 1, 1)

    # Change order from 'RGB' to 'BGR'
    image = image[:, [2, 1, 0], ...]

    # Subtract mean used during training
    image = subtract_mean(image)

    # Get model outputs
    with torch.no_grad():
        feature_image = radnet.forward(image)
        # flattens the image spatially
        feature_image = spatial_average(feature_image, keepdim=False)

    return feature_image


# -

# ## Setup data directory
#
# You can specify a directory with the MONAI_DATA_DIRECTORY environment variable.
# This allows you to save results and reuse downloads.
#
# If not specified a temporary directory will be used.

directory = os.environ.get("MONAI_DATA_DIRECTORY")
#root_dir = tempfile.mkdtemp() if directory is None else directory
root_dir = "/tmp/tmpzmzorzlg"
print(root_dir)

# ## Set deterministic training for reproducibility

set_determinism(0)

# ## Define the models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

autoencoderkl = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    latent_channels=3,
    num_channels=[64, 128, 128],
    num_res_blocks=1,
    norm_num_groups=32,
    attention_levels=(False, False, True),
)
autoencoderkl = autoencoderkl.to(device)

# +
unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_res_blocks=(1, 1, 1),
    num_channels=(64, 128, 128),
    attention_levels=(False, True, True),
    num_head_channels=128
)

scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear", beta_start=0.0015, beta_end=0.0195)

inferer = DiffusionInferer(scheduler)

discriminator = PatchDiscriminator(
    spatial_dims=2,
    num_layers_d=3,
    num_channels=32,
    in_channels=1,
    out_channels=1,
    kernel_size=4,
    activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
    norm="BATCH",
    bias=False,
    padding=1,
)
discriminator.to(device)
unet = unet.to(device)
# -

# ## Load pre-trained model

use_pre_trained = True

if use_pre_trained:
    unet = torch.hub.load("marksgraham/pretrained_generative_models:v0.2", model="ddpm_2d", verbose=True)
    unet = unet.to(device)
else:
    cwd = Path.cwd()
    model_path = cwd / Path("tutorials/generative/2d_ldm/best_aeutoencoderkl.pth")
    autoencoderkl.load_state_dict(torch.load(str(model_path)))
    cwd = Path.cwd()
    model_path = cwd / Path("tutorials/generative/2d_ldm/best_unet.pth")
    unet.load_state_dict(torch.load(str(model_path)))

# ## Get the real images and syntethic data

val_data = MedNISTDataset(root_dir=root_dir, section="validation", download=True, seed=0)
val_datalist = [{"image": item["image"]} for item in val_data.data if item["class_name"] == "Hand"]
val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    ]
)
val_ds = Dataset(data=val_datalist, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=True, num_workers=4)

# +
# Get the real data
real_images = []

pbar = tqdm(enumerate(val_loader), total=len(val_loader))
for step, x in pbar:
    real_img = x["image"].to(device)
    real_images.append(real_img)
    pbar.update()

real_images = torch.cat(real_images, axis=0)
# -

# Use the model to generate synthetic images. This step will take about 9 mins.

synth_images = []
unet.eval()
for step, x in enumerate(val_loader):
    n_synthetic_images = len(x['image'])
    noise = torch.randn((n_synthetic_images, 1, 64, 64))
    noise = noise.to(device)
    scheduler.set_timesteps(num_inference_steps=25)

    with torch.no_grad():
        syn_image, intermediates = inferer.sample(input_noise=noise, diffusion_model=unet,
                                                  scheduler=scheduler,save_intermediates=True,
                                                  intermediate_steps=100)
        synth_images.append(syn_image)
synth_images = torch.cat(synth_images, axis=0)

# Plot 3 examples from the synthetic data
fig, ax = plt.subplots(nrows=1, ncols=3)
for image_n in range(3):
    ax[image_n].imshow(syn_image[image_n, 0, :, :].cpu(), cmap="gray")
    ax[image_n].axis("off")

# ## Compute FID

# The FID measures the distance between the feature vectors from the real images and those obtained from generated images. In order to compute the FID the images need to be passed into a pre-trained network to get the desired feature vectors. Although the FID is commonly computed using the Inception network, here, we used a pre-trained version of the RadImageNet to calculate the feature space.

radnet = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True)
radnet.to(device)
radnet.eval()

# +
# Get the features for the real data
real_eval_feats = get_features(real_images)

# Get the features for the synthetic data
synth_eval_feats = get_features(synth_images)
# -

synth_eval_feats.shape, real_eval_feats.shape

fid = FIDMetric()
fid_res = fid(synth_eval_feats.to(device), real_eval_feats.to(device))
print(f"FID Score: {fid_res}")

# # Compute MMD

y = torch.ones([3, 3, 144, 144, 144])
y_pred =  torch.ones([3, 3, 144, 144, 144])
mmd = MMD()
res = mmd(y, y_pred)

y = torch.ones([3, 144, 144, 144])
y_pred =  torch.ones([3, 144, 144, 144])
mmd = MMD()
res = mmd._compute_metric(y, y_pred)
print(res)

y = torch.ones([3, 3, 144, 144, 144])
y_pred =  torch.ones([3, 3, 144, 144, 144])
mmd = MMD()
res = mmd(y, y_pred)

# +
mmd_scores = []
autoencoderkl.eval()

mmd = MMD()

for step, x in list(enumerate(val_loader)):
    image = x["image"].to(device)

    with torch.no_grad():
        image_recon = autoencoderkl.reconstruct(image)

    mmd_scores.append(mmd._compute_metric(image, image_recon))

mmd_scores = torch.stack(mmd_scores)
print(f"MS-SSIM score: {mmd_scores.mean().item():.4f} +- {mmd_scores.std().item():.4f}")

# -

# # Compute MultiScaleSSIMMetric and SSIMMetric
#
# Both MS-SSIM and SSIM can be used as metric to evaluate the diversity
#
# Compute the MS-SSIM and SSIM Meteric between the real images and those reconstructed by the AutoencoderKL.

# +
ms_ssim_recon_scores = []
ssim_recon_scores = []
autoencoderkl.eval()

ms_ssim = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)
ssim = SSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)

for step, x in list(enumerate(val_loader)):
    image = x["image"].to(device)

    with torch.no_grad():
        image_recon = autoencoderkl.reconstruct(image)

    ms_ssim_recon_scores.append(ms_ssim(image, image_recon))
    ssim_recon_scores.append(ssim(image, image_recon))

ms_ssim_recon_scores = torch.cat(ms_ssim_recon_scores, dim=0)
ssim_recon_scores = torch.cat(ssim_recon_scores, dim=0)

print(f"MS-SSIM Metric: {ms_ssim_recon_scores.mean():.7f} +- {ms_ssim_recon_scores.std():.7f}")
print(f"SSIM Metric: {ssim_recon_scores.mean():.7f} +- {ssim_recon_scores.std():.7f}")

# -

# Compute the SSIM and MS-SSIM between synthetic and real images

# +
ms_ssim_scores = []
ssim_scores = []

ms_ssim_scores.append(ms_ssim(real_images, synth_images))
ssim_scores.append(ssim(real_images, synth_images))

ms_ssim_scores = torch.cat(ms_ssim_scores, dim=0)
ssim_scores = torch.cat(ssim_scores, dim=0)

print(f"MS-SSIM Metric: {ms_ssim_scores.mean():.7f} +- {ms_ssim_scores.std():.7f}")
print(f"SSIM Metric: {ssim_scores.mean():.7f} +- {ssim_scores.std():.7f}")
# -


