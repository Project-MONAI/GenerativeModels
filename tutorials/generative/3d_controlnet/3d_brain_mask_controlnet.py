import numpy as np
import os
import nibabel as nib
import monai
from generative.networks.nets.controlnet import ControlNet
from monai.bundle.config_parser import ConfigParser
import torch
from tqdm import tqdm
from generative.networks.schedulers.ddpm import DDPMScheduler
from copy import deepcopy

## Arguments
path_to_image_files = ""
path_to_label_files = ""
val_perc = 0.2
batch_size = 8
num_workers = 2
input_shape = [160, 224, 160]
path_to_bundle = ""
lr = 0.00002
validation_epochs = 5
saving_epochs = 1
scale_factor = 1.0
num_epochs = 100
checkpoints_dir = ""
autoencoder = "/home/vf19/PycharmProjects/controlNet_fiddling/model-zoo/models/brain_image_synthesis_latent_diffusion_model/models/autoencoder.pth"
ldm = "/home/vf19/PycharmProjects/controlNet_fiddling/model-zoo/models/brain_image_synthesis_latent_diffusion_model/models/diffusion_model.pth"

## Auxiliary functions
def get_loader(path_to_image_files, path_to_label_files, input_shape = [160, 224, 160], val_perc = 0.20, batch_size = 8,
               num_workers = 2):

    all_images = os.listdir(path_to_image_files)
    val_files = [{'image': os.path.join(path_to_image_files, f),
                  'label': os.path.join(path_to_label_files, f),
                  'gender': float(f.strip(".nii.gz").split("_")[-4]),
                  'age': float(f.strip(".nii.gz").split("_")[-3]),
                  'ventricular_vol': float(f.strip(".nii.gz").split("_")[-2]),
                  'brain_vol': float(f.strip(".nii.gz").split("_")[-1]),} for f in all_images[:int(val_perc * len(all_images))]]
    train_files = [{'image': os.path.join(path_to_image_files, f),
                    'label': os.path.join(path_to_label_files, f),
                    'gender': float(f.strip(".nii.gz").split("_")[-4]),
                    'age': float(f.strip(".nii.gz").split("_")[-3]),
                    'ventricular_vol': float(f.strip(".nii.gz").split("_")[-2]),
                    'brain_vol': float(f.strip(".nii.gz").split("_")[-1]),
                    } for f in all_images[int(val_perc * len(all_images)):]]

    train_transforms = monai.transforms.Compose(
        [monai.transforms.LoadImaged(keys = ['image', 'label']),
         monai.transforms.EnsureChannelFirstd(keys = ['image', 'label']),
         monai.transforms.CenterSpatialCropd(keys = ['image', 'label'], roi_size=input_shape),
         monai.transforms.SpatialPadd(keys=['image', 'label'], spatial_size=input_shape,),
         monai.transforms.EnsureTyped(keys=["image", "label"]),
         monai.transforms.Orientationd(keys=["image", 'label'], axcodes="RAS"),
         monai.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
         monai.transforms.ToTensord(keys=['image', 'label', 'gender', 'age', 'ventricular_vol', 'brain_vol']),
                                    ]
    )
    #monai.transforms.MaskIntensityd(keys=['image'], mask_key='label'),
    train_dataset = monai.data.Dataset(data=train_files, transform = train_transforms)
    val_dataset = monai.data.Dataset(data=val_files, transform=train_transforms)
    train_loader = monai.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers)
    val_loader = monai.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers)
    return train_loader, val_loader

def load_model_from_bundle(bundle_path):
    cp = ConfigParser()
    bundle_root = bundle_path
    cp.read_meta(f"{bundle_root}/configs/metadata.json")
    cp.read_config(f"{bundle_root}/configs/inference.json")
    cp["bundle_root"] = bundle_root
    autoencoder = cp.get_parsed_content("autoencoder_def")
    autoencoder.load_state_dict(torch.load(os.path.join(bundle_path, 'models', 'autoencoder.pth')))
    diffusion = cp.get_parsed_content("diffusion_def")
    diffusion.load_state_dict(torch.load(os.path.join(bundle_path, 'models', 'diffusion_model.pth')))
    return autoencoder, diffusion

def create_mask(data, input_shape):
    input_mask = torch.ones([data['image'].shape[0]] + [4] + input_shape[2:])
    for b in range(input_mask.shape[0]):
        input_mask[b, 0, :] *= data['gender'][b]
        input_mask[b, 1, :] *= data['age'][b]
        input_mask[b, 2, :] *= data['ventricular_vol'][b]
        input_mask[b, 3, :] *= data['brain_vol'][b]
    return input_mask

def translate_parameters(controlnet, diffusion):
    controlnet.load_state_dict(diffusion.state_dict(), strict = False)
    return controlnet

# Data loading
#TODO: get data from the Drive link and extract it, create temporary file for data, and models, access the model zoo.

device = "cuda" if torch.cuda.is_available() else "cpu"
# Create checkpoint
if not os.path.isdir(checkpoints_dir):
    os.makedirs(checkpoints_dir)
samples_folder = os.path.join(checkpoints_dir, 'samples')
if not os.path.isdir(samples_folder):
    os.makedirs(samples_folder)
# Load diffusion model
autoencoder, diffusion = load_model_from_bundle(path_to_bundle)
for p in autoencoder.parameters():
    p.requires_grad = False # Freeze weights
for p in diffusion.parameters():
    p.requires_grad = False # Freeze weights
autoencoder = autoencoder.to(device)
diffusion = diffusion.to(device)
# Create control net
controlnet = ControlNet(spatial_dims=3, in_channels=7,
                        num_res_blocks=diffusion.num_res_blocks,
                        num_channels=diffusion.block_out_channels, with_conditioning=False,
                        attention_levels=diffusion.attention_levels,
                        )
# Copy weights from the DM to the controlnet
controlnet = translate_parameters(controlnet, diffusion)
controlnet = controlnet.to(device)
train_loader, val_loader = get_loader(path_to_image_files, path_to_label_files, input_shape,
                                      val_perc, batch_size, num_workers)
scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0015, beta_end= 0.0195,
                          clip_sample=False,)
optimizer = torch.optim.Adam(params = controlnet.parameters(), lr = lr, betas = (0.9, 0.99))
# See if checkpoints saved
if os.path.isfile(os.path.join(checkpoints_dir, 'checkpoint.pth')):
    print("Using checkpoint!")
    checkpoint = torch.load(os.path.join(checkpoints_dir, 'checkpoint.pth'))
    controlnet.load_state_dict(checkpoint['controlnet'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_val_loss = checkpoint['best_val_loss']
    i_e = checkpoint['initial_epoch']
else:
    best_val_loss = 1000
    i_e = 0

# Loop
for e in range(i_e, num_epochs):
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {e}")
    epoch_loss = 0.0
    for step, data in progress_bar:
        input_conditioning = data['label'].to(device)
        input_image = data['image'].to(device)
        with torch.no_grad():
            latent = autoencoder.encode_stage_2_inputs(input_image) * scale_factor
        # We concatenate gender, age, ventricular volume and brain volume as additional channels
        input_mask = create_mask(data, list(latent.shape)).to(device)
        latent = torch.cat([latent, input_mask], dim = 1, )
        timesteps = torch.randint(0, scheduler.num_train_timesteps,(latent.shape[0],), device=device).long()
        noise = torch.randn(list(latent.shape)).to(device)
        noisy_latents = scheduler.add_noise(original_samples=latent, noise=noise, timesteps=timesteps)
        optimizer.zero_grad()
        down_block_res_samples, mid_block_res_sample = controlnet(
            noisy_latents,
            timesteps,
            controlnet_cond = input_conditioning,
            context = None,
        )
        cond = torch.cat([data['gender'].unsqueeze(0),
                          data['age'].unsqueeze(0),
                          data['ventricular_vol'].unsqueeze(0),
                          data['brain_vol'].unsqueeze(0)], dim = 1).unsqueeze(1)
        prediction = diffusion(x=noisy_latents,
                               timesteps=timesteps,
                               context=cond.to(device),
                               down_block_additional_residuals = down_block_res_samples,
                               mid_block_additional_residual = mid_block_res_sample)
        loss = torch.nn.functional.l1_loss(prediction, noise[:, :3, ...])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

    if step % validation_epochs == 0:
        sampling_step = np.random.randint(len(val_loader))
        val_loss = 0.0
        for ind, data in enumerate(val_loader):
            with torch.no_grad():
                diffusion = diffusion.eval()
                controlnet = controlnet.eval()
                input_conditioning = data['label'].to(device)
                input_image = data['image'].to(device)
                # We concatenate gender, age, ventricular volume and brain volume as additional channels
                latent = autoencoder.encode_stage_2_inputs(input_image) * scale_factor
                input_mask = create_mask(data, list(latent.shape)).to(device)
                latent = torch.cat([latent, input_mask], dim=1, )
                timesteps = torch.randint(0, scheduler.num_train_timesteps, (latent.shape[0],),
                                          device=device).long()
                noise = torch.randn(list(latent.shape)).to(device)
                noisy_latents = scheduler.add_noise(original_samples=latent, noise=noise, timesteps=timesteps)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    controlnet_cond=input_conditioning,
                    context=None,
                )
                cond = torch.cat([data['gender'].unsqueeze(0),
                                  data['age'].unsqueeze(0),
                                  data['ventricular_vol'].unsqueeze(0),
                                  data['brain_vol'].unsqueeze(0)], dim=1).unsqueeze(1)
                prediction = diffusion(x=noisy_latents,
                                       timesteps=timesteps,
                                       context=cond.to(device),
                                       down_block_additional_residuals=down_block_res_samples,
                                       mid_block_additional_residual=mid_block_res_sample)
                val_loss += torch.nn.functional.l1_loss(prediction, noise[:, :3, ...]).item()

                if ind == sampling_step:
                    noise = torch.randn(list(noisy_latents.shape))
                    noise_pred = deepcopy(noise).to(device)
                    progress_bar_sampling = tqdm(scheduler.timesteps, total=len(scheduler.timesteps), ncols=110)
                    progress_bar_sampling.set_description("sampling...")
                    for t in progress_bar_sampling:
                        down_block_res_samples, mid_block_res_sample = controlnet(noise_pred,
                                                                              timesteps,
                                                                              controlnet_cond=input_conditioning)
                        noise_pred = diffusion(
                            noise_pred,
                            timesteps=torch.Tensor((t,)).to(device),
                            context=cond.to(device),
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample)
                        noise_pred = torch.cat([noise_pred, input_mask], 1)
                    decoded = autoencoder.decode_stage_2_outputs(noise_pred[:, :3, ...]).detach().cpu()
                    # Log image
                    # We make a 2 x 2 grid with: GT / Predicted / Mask
                    input_conditioning = input_conditioning.detach().cpu().squeeze(1) # Remove channel
                    input_image = input_image.detach().cpu().squeeze(1) # Remove channel
                    for b in range(input_conditioning.shape[0]):
                        to_save = torch.cat([input_image[b, ...], decoded[b, 0,...]], 1)
                        to_save = torch.cat([to_save, torch.cat([input_conditioning[b, ...],
                                                                torch.zeros_like(input_conditioning[b, ...])],
                                                                1),
                                             ],0)
                        to_save = to_save.numpy()
                        nifti_image = nib.Nifti1Image(to_save, affine = np.eye(4))
                        nib.save(nifti_image, os.path.join(samples_folder, "val_epoch_%d_%d.nii.gz" %(e, b)))
        val_loss = val_loss / len(val_loader)
        print("Validation loss epoch %d: %.3f" %(e, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(controlnet.state_dict(), os.path.join(checkpoints_dir, 'best_model.pth'))
        controlnet.train()
        diffusion.train()
    if step % saving_epochs == 0:
        checkpoint = {}
        checkpoint['controlnet'] = controlnet.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['best_val_loss'] = best_val_loss
        checkpoint['initial_epoch'] = e
print("Training finished...")
torch.save(controlnet.state_dict(), os.path.join(checkpoints_dir, 'final_model.pth'))
