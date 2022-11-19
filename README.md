# GenerativeModels
Prototyping repo for generative models to be integrated into MONAI

* Target: alpha for November
* Priority: develop general purpose components and trial PR process into MONAI

## Objectives
* Develop models to reproduce instances of data on the same distribution as the training data but differentiated enough to preserve privacy/data protection
* The outputs should be valid inputs for training further models without bias and with robust results
* Develop new architectures, loss functions, layers and blocks, inferers, and other components to be integrated into MONAI, in particular focus on general purpose pieces first
* Develop specific networks train for tasks and distribute as bundles
* Develop the tutorials and other documentation to describe how to use and develop these classes of models
* Focus on modular components and network definitions for others to extend and use in different situations
* Reference implementations for generative models (VQVAE, diffusion) to demonstrate how to configure and use MONAI with these model types which vary from existing supervised model training schemes
* Provide examples on possibly shrunken decathlon data and using patch-based models
* Superresolution networks? In time and space?
* Unconditional and conditional networks using input images/text/other values as conditions
* Models for generating images and related reports
* Inpainting
* Anomaly detection
* MRI recon both in image and k space
