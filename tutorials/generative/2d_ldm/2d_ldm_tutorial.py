# # 2D Latent Diffusion Model

# +
# TODO: Add buttom with "Open with Colab"
# -

# ## Set up environment using Colab
#

# !python -c "import monai" || pip install -q "monai-weekly[tqdm]"
# !python -c "import matplotlib" || pip install -q matplotlib
# %matplotlib inline

# ## Set up imports

# +
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
from monai.utils import first, set_determinism
from tqdm import tqdm

from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.schedulers import DDPMScheduler

print_config()
# -

# for reproducibility purposes set a seed
set_determinism(42)

# ## Setup a data directory and download dataset
# Specify a MONAI_DATA_DIRECTORY variable, where the data will be downloaded. If not specified a temporary directory will be used.

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

# ## Download the training set

train_data = MedNISTDataset(root_dir=root_dir, section="training", download=True, seed=0)
train_datalist = [{"image": item["image"]} for item in train_data.data if item["class_name"] == "Hand"]
image_size = 64
train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        # TODO: Change transformations
        transforms.RandAffined(
            keys=["image"],
            rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
            translate_range=[(-1, 1), (-1, 1)],
            scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
            spatial_size=[image_size, image_size],
            padding_mode="zeros",
            prob=0.5,
        ),
    ]
)
train_ds = Dataset(data=train_datalist, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)

# ## Visualise examples from the training set

# Plot 3 examples from the training set
check_data = first(train_loader)
fig, ax = plt.subplots(nrows=1, ncols=3)
for image_n in range(3):
    ax[image_n].imshow(check_data["image"][image_n, 0, :, :], cmap="gray")
    ax[image_n].axis("off")
# TODO: remove path
plt.savefig("/project/tutorials/generative/2d_ldm/hand_examples.png")

# ## Download the validation set

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

# ## Define the network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# +
autoencoderkl = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=64,
    latent_channels=1,
    ch_mult=(1, 2, 2),
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False, False, True),
)

unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_res_blocks=1,
    attention_resolutions=[2, 4],
    channel_mult=[1, 2, 2],
    model_channels=64,
    # TODO: play with this number
    num_heads=1,
)

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="linear",
    beta_start=0.0015,
    beta_end=0.0195,
)


# ## Train AutoencoderKL
autoencoderkl = autoencoderkl.to(device)
kl_weight = 1e-6
kl_optimizer = torch.optim.Adam(autoencoderkl.parameters(), 1e-5)
n_epochs = 50
val_interval = 10
kl_epoch_loss_list = []
kl_val_epoch_loss_list = []
intermediary_images = []
n_example_images = 4

for epoch in range(n_epochs):
    autoencoderkl.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        kl_optimizer.zero_grad(set_to_none=True)
        reconstruction, z_mu, z_sigma = autoencoderkl(images)

        rec_loss = F.mse_loss(reconstruction.float(), images.float())

        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # TODO: Add adversarial component
        # TODO: Add perceptual loss

        loss = rec_loss + kl_weight * kl_loss
        loss.backward()
        kl_optimizer.step()
        epoch_loss += loss.item()

        progress_bar.set_postfix(
            {
                "loss": epoch_loss / (step + 1),
            }
        )
    kl_epoch_loss_list.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        autoencoderkl.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch["image"].to(device)
                kl_optimizer.zero_grad(set_to_none=True)
                reconstruction, z_mu, z_sigma = autoencoderkl(images)
                # TODO: Remove this
                # Get the first sammple from the first validation batch for visualisation
                # purposes
                if val_step == 1:
                    intermediary_images.append(reconstruction[:n_example_images, 0])

                rec_loss = F.mse_loss(reconstruction.float(), images.float())

                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                # TODO: Add adversarial component
                # TODO: Add perceptual loss

                loss = rec_loss + kl_weight * kl_loss
                val_loss += loss.item()

        val_loss /= val_step
        kl_val_epoch_loss_list.append(val_loss)

        print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
progress_bar.close()

# ### Visualise the results from the autoencoderKL
val_samples = np.linspace(val_interval, n_epochs, int(n_epochs / val_interval))
fig, ax = plt.subplots(nrows=len(val_samples), ncols=1, sharey=True)
for image_n in range(len(val_samples)):
    reconstructions = torch.reshape(intermediary_images[image_n], (image_size * n_example_images, image_size)).T
    ax[image_n].imshow(reconstructions.cpu(), cmap="gray")
    ax[image_n].set_xticks([])
    ax[image_n].set_yticks([])
    ax[image_n].set_ylabel(f"Epoch {val_samples[image_n]:.0f}")
plt.savefig("/project/tutorials/generative/2d_ldm/autoencoderkl.png")

# ## Train Diffusion Model
optimizer = torch.optim.Adam(unet.parameters(), lr=2.5e-5)
# TODO: Add lr_scheduler with warm-up
# TODO: Add EMA model

unet = unet.to(device)
n_epochs = 50
val_interval = 5
epoch_loss_list = []
val_epoch_loss_list = []
for epoch in range(n_epochs):
    unet.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)

        # TODO: check how to deal with next commands with multi-GPU and for FL
        with torch.no_grad():
            z_mu, z_sigma = autoencoderkl.encode(images)
            z = autoencoderkl.sampling(z_mu, z_sigma)

        timesteps = torch.randint(0, scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()

        noise = torch.randn_like(z).to(device)
        noisy_latent = scheduler.add_noise(original_samples=z, noise=noise, timesteps=timesteps)
        noise_pred = unet(noisy_latent, timesteps)

        loss = F.mse_loss(noise_pred.float(), noise.float())

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
        unet.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch["image"].to(device)
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    z_mu, z_sigma = autoencoderkl.encode(images)
                    z = autoencoderkl.sampling(z_mu, z_sigma)

                timesteps = torch.randint(0, scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()

                noise = torch.randn_like(z).to(device)
                noisy_latent = scheduler.add_noise(original_samples=z, noise=noise, timesteps=timesteps)
                noise_pred = unet(noisy_latent, timesteps)

                loss = F.mse_loss(noise_pred.float(), noise.float())

                val_loss += loss.item()
        val_loss /= val_step
        val_epoch_loss_list.append(val_loss)
        print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
progress_bar.close()


# ## Plotting sampling example
unet.eval()
image = torch.randn(
    (1, 1, 64, 64),
)
image = image.to(device)
scheduler.set_timesteps(num_inference_steps=1000)

intermediary = []
for t in tqdm(scheduler.timesteps, ncols=70):
    # 1. predict noise model_output
    with torch.no_grad():
        z_mu, z_sigma = autoencoderkl.encode(image)
        z = autoencoderkl.sampling(z_mu, z_sigma)
        model_output = unet(z, torch.Tensor((t,)).to(device))

    # 2. compute previous image: x_t -> x_t-1
    r_image, _ = scheduler.step(model_output, t, z)
    if t % 100 == 0:
        intermediary.append(r_image)

# Invert the autoencoderKL model
decoded_images = []
for image in intermediary:
    decoded = autoencoderkl.decode(image)
    decoded_images.append(decoded)
plt.figure()
chain = torch.cat(decoded_images, dim=-1)
plt.style.use("default")
plt.imshow(chain[0, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
plt.tight_layout()
plt.axis("off")
plt.savefig("/project/tutorials/generative/2d_ldm/intermediary.png")


# ## Plot learning curves
plt.figure()
plt.title("Learning Curves", fontsize=20)
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_loss_list, linewidth=2.0, label="Train")
plt.plot(
    np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
    val_epoch_loss_list,
    linewidth=2.0,
    label="Validation",
)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.savefig("/project/tutorials/generative/2d_ldm/learning_curve.png")
# plt.show()
