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

from torch import nn


# TODO: Discuss other methods that should be included
class LatentDiffusionModel(nn.Module):
    """
    Latent Diffusion Model class. This class takes care of storing all components for latent diffusion models
    and handles methods for sampling.

    Args:
        first_stage: First stage object to encode and decode images to and from latent representations.
        unet_network: Conditional U-Net architecture to denoise the encoded image latents.
        scheduler: Variance scheduler.
    """

    def __init__(
        self,
        first_stage: nn.Module,
        unet_network: nn.Module,
        scheduler: nn.Module,
    ) -> None:

        super().__init__()
        self.first_stage = first_stage
        self.unet_network = unet_network
        self.scheduler = scheduler

    # TODO: Implement sampling methods
    def sample_unconditioned(self):
        raise NotImplementedError
