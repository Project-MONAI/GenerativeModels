# MONAI Generative Models Tutorials
This directory hosts the MONAI Generative Models tutorials.

## Requirements
To run the tutorials, you will need to install the Generative Models package.
Besides that, most of the examples and tutorials require
[matplotlib](https://matplotlib.org/) and [Jupyter Notebook](https://jupyter.org/).

These can be installed with the following:

```bash
python -m pip install -U pip
python -m pip install -U matplotlib
python -m pip install -U notebook
```

Some of the examples may require optional dependencies. In case of any optional import errors,
please install the relevant packages according to MONAI's [installation guide](https://docs.monai.io/en/latest/installation.html).
Or install all optional requirements with the following:

```bash
pip install -r requirements-dev.txt
```

## List of notebooks and examples

### Table of Contents
1. [Diffusion Models](#1-diffusion-models)
2. [Latent Diffusion Models](#2-latent-diffusion-models)
3. [VQ-VAE + Transformers](#3-vq-vae--transformers)


### 1. Diffusion Models

#### Image synthesis with Diffusion Models

* [Training a 3D Denoising Diffusion Probabilistic Model](./generative/3d_ddpm/3d_ddpm_tutorial.ipynb): This tutorial shows how to easily
train a DDPM on 3D medical data. In this example, we use a downsampled version of the BraTS dataset. We will show how to
make use of the UNet model and the Noise Scheduler necessary to train a diffusion model. Besides that, we show how to
use the DiffusionInferer class to simplify the training and sampling processes. Finally, after training the model, we
show how to use a Noise Scheduler with fewer timesteps to sample synthetic images.

* [Training a 2D Denoising Diffusion Probabilistic Model](./generative/2d_ddpm/2d_ddpm_tutorial.ipynb): This tutorial shows how to easily
train a DDPM on medical data. In this example, we use the MedNIST dataset, which is very suitable for beginners as a tutorial.

* [Comparing different noise schedulers](./generative/2d_ddpm/2d_ddpm_compare_schedulers.ipynb): In this tutorial, we compare the
performance of different noise schedulers. We will show how to sample a diffusion model using the DDPM, DDIM, and PNDM
schedulers and how different numbers of timesteps affect the quality of the samples.

* [Training a 2D Denoising Diffusion Probabilistic Model with different parameterisation](./generative/2d_ddpm/2d_ddpm_tutorial_v_prediction.ipynb):
In MONAI Generative Models, we support different parameterizations for the diffusion model (epsilon, sample, and
v-prediction). In this tutorial, we show how to train a DDPM using the v-prediction parameterization, which improves the
stability and convergence of the model.

* [Training a 2D DDPM using Pytorch Ignite](./generative/2d_ddpm/2d_ddpm_compare_schedulers.ipynb): Here, we show how to train a DDPM
on medical data using Pytorch Ignite. We will show how to use the DiffusionPrepareBatch to prepare the model inputs and MONAI's SupervisedTrainer and SupervisedEvaluator to train DDPMs.

* [Using a 2D DDPM to inpaint images](./generative/2d_ddpm/2d_ddpm_inpainting.ipynb): In this tutorial, we show how to use a DDPM to
inpaint of 2D images from the MedNIST dataset using the RePaint method.

* [Generating conditional samples with a 2D DDPM using classifier-free guidance](./generative/classifier_free_guidance/2d_ddpm_classifier_free_guidance_tutorial.ipynb):
This tutorial shows how easily we can train a Diffusion Model and generate conditional samples using classifier-free guidance in
the MONAI's framework.

* [Training Diffusion models with Distributed Data Parallel](./generative/distributed_training/ddpm_training_ddp.py): This example shows how to execute distributed training and evaluation based on PyTorch native DistributedDataParallel
module with torch.distributed.launch.

#### Anomaly Detection with Diffusion Models

* [Weakly Supervised Anomaly Detection with Implicit Guidance](./generative/anomaly_detection/2d_classifierfree_guidance_anomalydetection_tutorial.ipynb):
This tutorial shows how to use a DDPM to perform weakly supervised anomaly detection using classifier-free (implicit) guidance based on the
method proposed by Sanchez et al. [What is Healthy? Generative Counterfactual Diffusion for Lesion Localization](https://arxiv.org/abs/2207.12268). DGM 4 MICCAI 2022


### 2. Latent Diffusion Models

#### Image synthesis with Latent Diffusion Models

* [Training a 3D Latent Diffusion Model](./generative/3d_ldm/3d_ldm_tutorial.ipynb): This tutorial shows how to train a LDM on 3D medical
data. In this example, we use the BraTS dataset. We show how to train an AutoencoderKL and connect it to an LDM. We also
comment on the importance of the scaling factor in the LDM used to scale the latent representation of the AEKL to a suitable
range for the diffusion model. Finally, we show how to use the LatentDiffusionInferer class to simplify the training and sampling.

* [Training a 2D Latent Diffusion Model](./generative/2d_ldm/2d_ldm_tutorial.ipynb): This tutorial shows how to train an LDM on medical
on the MedNIST dataset. We show how to train an AutoencoderKL and connect it to an LDM.

* Training Autoencoder with KL-regularization: In this section, we focus on training an AutoencoderKL on [2D](./generative/2d_autoencoderkl/2d_autoencoderkl_tutorial.ipynb) and [3D](./generative/3d_autoencoderkl/3d_autoencoderkl_tutorial.ipynb) medical data,
that can be used as the compression model used in a Latent Diffusion Model.

#### Super-resolution with Latent Diffusion Models

* [Super-resolution using Stable Diffusion Upscalers method](./generative/2d_super_resolution/2d_stable_diffusion_v2_super_resolution.ipynb):
In this tutorial, we show how to perform super-resolution on 2D images from the MedNIST dataset using the Stable
Diffusion Upscalers method. In this example, we will show how to condition a latent diffusion model on a low-resolution image
as well as how to use the DiffusionModelUNet's class_labels conditioning to condition the model on the level of noise added to the image
(aka "noise conditioning augmentation")


### 3. VQ-VAE + Transformers

#### Image synthesis with VQ-VAE + Transformers

* [Training a 2D VQ-VAE + Autoregressive Transformers](./generative/2d_vqvae_transformer/2d_vqvae_transformer_tutorial.ipynb): This tutorial shows how to train
a Vector-Quantized Variation Autoencoder + Transformers on the MedNIST dataset.

* Training VQ-VAEs and VQ-GANs: In this section, we show how to train Vector Quantized Variation Autoencoder (on [2D](./generative/2d_vqvae/2d_vqvae_tutorial.ipynb) and [3D](./generative/3d_autoencoderkl/3d_autoencoderkl_tutorial.ipynb) data) and
show how to use the PatchDiscriminator class to train a [VQ-GAN](./generative/2d_vqgan/2d_vqgan_tutorial.ipynb) and improve the quality of the generated images.

#### Anomaly Detection with VQ-VAE + Transformers

* [Anomaly Detection with 2D VQ-VAE + Autoregressive Transformers](./generative/anomaly_detection/anomaly_detection_with_transformers.ipynb): This tutorial shows how to
 train a Vector-Quantized Variation Autoencoder + Transformers on the MedNIST dataset and use it to extract the likelihood of
testing images to be part of the in-distribution class (used during training).
