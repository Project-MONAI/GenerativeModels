<p align="center">
  <img src="https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/docs/images/MONAI-logo-color.png" width="50%" alt='project-monai'>
</p>

# MONAI Generative Models
Prototyping repository for generative models to be integrated into MONAI core, MONAI tutorials, and MONAI model zoo.
## Features
* Network architectures: Diffusion Model, Autoencoder-KL, VQ-VAE, Autoregressive transformers, (Multi-scale) Patch-GAN discriminator.
* Diffusion Model Noise Schedulers: DDPM, DDIM, and PNDM.
* Losses: Adversarial losses, Spectral losses, and Perceptual losses (for 2D and 3D data using LPIPS, RadImageNet, and 3DMedicalNet pre-trained models).
* Metrics: Multi-Scale Structural Similarity Index Measure (MS-SSIM) and Fréchet inception distance (FID).
* Diffusion Models, Latent Diffusion Models, and VQ-VAE + Transformer Inferers classes (compatible with MONAI style) containing methods to train, sample synthetic images, and obtain the likelihood of inputted data.
* MONAI-compatible trainer engine (based on Ignite) to train models with reconstruction and adversarial components.
* Tutorials including:
  * How to train VQ-VAEs, VQ-GANs, VQ-VAE + Transformers, AutoencoderKLs, Diffusion Models, and Latent Diffusion Models on 2D and 3D data.
  * Train diffusion model to perform conditional image generation with classifier-free guidance.
  * Comparison of different diffusion model schedulers.
  * Diffusion models with different parameterizations (e.g., v-prediction and epsilon parameterization).
  * Anomaly Detection using VQ-VAE + Transformers and Diffusion Models.
  * Inpainting with diffusion model (using Repaint method)
  * Super-resolution with Latent Diffusion Models (using Noise Conditioning Augmentation)

## Roadmap
Our short-term goals are available in the [Milestones](https://github.com/Project-MONAI/GenerativeModels/milestones)
section of the repository.

In the longer term, we aim to integrate the generative models into the MONAI core repository (supporting tasks such as,
image synthesis, anomaly detection, MRI reconstruction, domain transfer)

## Installation
To install the current release of MONAI Generative Models, you can run:
```
pip install monai-generative
```
To install the current main branch of the repository, run:
```
pip install git+https://github.com/Project-MONAI/GenerativeModels.git
```
Requires Python >= 3.8.

## Contributing
For guidance on making a contribution to MONAI, see the [contributing guidelines](https://github.com/Project-MONAI/GenerativeModels/blob/main/CONTRIBUTING.md).

## Community
Join the conversation on Twitter [@ProjectMONAI](https://twitter.com/ProjectMONAI) or join our [Slack channel](https://forms.gle/QTxJq3hFictp31UM9).

# Citation

If you use MONAI Generative in your research, please cite us! The citation can be exported from [the paper](https://arxiv.org/abs/2307.15208).

## Links
- Website: https://monai.io/
- Code: https://github.com/Project-MONAI/GenerativeModels
- Project tracker: https://github.com/Project-MONAI/GenerativeModels/projects
- Issue tracker: https://github.com/Project-MONAI/GenerativeModels/issues
