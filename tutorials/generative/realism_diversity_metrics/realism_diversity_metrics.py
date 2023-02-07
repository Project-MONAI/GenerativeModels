# +
# TODO: Add Open in Colab
# -

# ## Setup environment

# %cd /home/jdafflon/GenerativeModels

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

from generative.metrics import FID, MMD, MSSSIM
from generative.networks.nets import DiffusionModelUNet, PatchDiscriminator, AutoencoderKL
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer

print_config()


# +
def subtract_mean(x: torch.Tensor) -> torch.Tensor:
    mean = [0.406, 0.456, 0.485]
    x[:, 0, :, :] -= mean[0]
    x[:, 1, :, :] -= mean[1]
    x[:, 2, :, :] -= mean[2]
    return x

def normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)

def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
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

    # normalise through channels
    #features_image = normalize_tensor(feature_image)

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
    num_channels=64,
    latent_channels=3,
    ch_mult=(1, 2, 2),
    num_res_blocks=1,
    norm_num_groups=32,
    attention_levels=(False, False, True),
)
autoencoderkl = autoencoderkl.to(device)

# +
unet = DiffusionModelUNet(
    spatial_dims=2, in_channels=3, out_channels=3, num_res_blocks=1, num_channels=(128, 256, 256), num_head_channels=256
)

scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", beta_start=0.0015, beta_end=0.0195)

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

cwd = Path.cwd()
model_path = cwd / Path("tutorials/generative/2d_ldm/best_aeutoencoderkl.pth")
autoencoderkl.load_state_dict(torch.load(str(model_path)))

cwd = Path.cwd()
model_path = cwd / Path("tutorials/generative/2d_ldm/best_unet.pth")
unet.load_state_dict(torch.load(str(model_path)))

# ## Get the validation split for the real images

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

# ## Get features for real data

radnet = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True)
radnet.to(device)
radnet.eval()

# +
real_eval_feats = []

pbar = tqdm(enumerate(val_loader), total=len(val_loader))
for step, x in pbar:
    real_img = x["image"].to(device)
    features_real = get_features(real_img)
    real_eval_feats.append(features_real.cpu())
    pbar.update()

real_eval_feats = torch.cat(real_eval_feats, axis=0)
# -

# ## Generate synthetic images

# +
synth_eval_feats = []
unet.eval()

#pbar = tqdm(enumerate(val_loader), total=len(val_loader))
for step, x in enumerate(val_loader):
    print(step)
    n_synthetic_images = len(x['image'])
    syn_image = torch.randn((n_synthetic_images, 1, 64, 64))
    syn_image = syn_image.to(device)
    scheduler.set_timesteps(num_inference_steps=1000)

    with torch.no_grad():

        z_mu, z_sigma = autoencoderkl.encode(syn_image)
        z = autoencoderkl.sampling(z_mu, z_sigma)

        noise = torch.randn_like(z).to(device)
        syn_image, intermediates = inferer.sample(
            input_noise=z, diffusion_model=unet, scheduler=scheduler, save_intermediates=True, intermediate_steps=100
        )
        syn_image = autoencoderkl.decode(syn_image)

        features_syn_image = get_features(syn_image)
        synth_eval_feats.append(features_syn_image)
# -

# Plot 3 examples from the synthetic data
fig, ax = plt.subplots(nrows=1, ncols=3)
for image_n in range(3):
    ax[image_n].imshow(syn_image[image_n, 0, :, :].cpu(), cmap="gray")
    ax[image_n].axis("off")

synch_eval_feats = torch.cat(synth_eval_feats, axis=0)
print(synch_eval_feats.shape, real_eval_feats.shape)

fid = FID()
results = fid(real_eval_feats.to(device), synch_eval_feats)
results.item()

# Even when passing the same image, it returns NaNs
fid = FID()
results = fid(synch_eval_feats, synch_eval_feats)
results.item()

# Even when passing the same image, it returns NaNs
fid = FID()
results = fid(real_eval_feats.to(device), real_eval_feats.to(device))
results.item()

# # Compute MMD

# +
# Generate a few samples (the 45 for the last batch)

n_synthetic_images = len(real_img)
syn_image = torch.randn((n_synthetic_images, 1, 64, 64))
syn_image = syn_image.to(device)
scheduler.set_timesteps(num_inference_steps=1000)

with torch.no_grad():

    z_mu, z_sigma = autoencoderkl.encode(syn_image)
    z = autoencoderkl.sampling(z_mu, z_sigma)

    noise = torch.randn_like(z).to(device)
    syn_image, intermediates = inferer.sample(
        input_noise=z, diffusion_model=unet, scheduler=scheduler, save_intermediates=True, intermediate_steps=100
    )
    syn_image = autoencoderkl.decode(syn_image)
# -

mmd = MMD()
mmd(real_img, syn_image)

real_img.shape

syn_image.cpu().shape

# # Compute SSIM

data_range = 1.0
mssim = MSSSIM(data_range=data_range)
mssim(real_img, syn_image)

real_img.max()

image1 = torch.ones([3, 3, 144, 144]) / 2
image2 = torch.ones([3, 3, 144, 144]) / 2

data_range = 1.0
mssim = MSSSIM(data_range=data_range)
mssim(image1, image2)

# +
from generative.metrics import  MSSSIM
import torch

data_range = torch.ones(1, 3)
image1 = torch.ones([3, 3, 144, 144]) / 2
image2 = torch.ones([3, 3, 144, 144]) / 2

mssim = MSSSIM(data_range=data_range)
mssim(image1, image2)
