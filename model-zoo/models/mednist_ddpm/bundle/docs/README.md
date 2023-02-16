
# MedNIST DDPM Example Bundle

This implements roughly equivalent code to the "Denoising Diffusion Probabilistic Models with MedNIST Dataset" example notebook. This includes scripts for training with single or multiple GPUs and a visualisation notebook.

The files included here demonstrate how to use the bundle:
  * [2d_ddpm_bundle_tutorial.ipynb](./2d_ddpm_bundle_tutorial.ipynb) - demonstrates command line and in-code invocation of the bundle's training and inference scripts
  * [sub_train.sh](sub_train.sh) - SLURM submission script example for training
  * [sub_train_multigpu.sh](sub_train_multigpu.sh) - SLURM submission script example for training with multiple GPUs
