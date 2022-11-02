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

import torch
import torch.nn as nn
from lpips import LPIPS


# TODO: Check difference between this and MONAI's DeepSupervisionLoss (when not using LPIPS)
#  https://github.com/Project-MONAI/MONAI/blob/06cb0fa3b4aa04744cbf9eff46f5860a7681b25f/monai/losses/ds_loss.py#L21
# TODO: Define model_path for lpips networks.
# TODO: Add MedicalNet for true 3D computation (https://github.com/Tencent/MedicalNet)
# TODO: Add RadImageNet for 2D computaion with networks pretrained using radiological images
#  (https://github.com/BMEII-AI/RadImageNet)
class PerceptualLoss(nn.Module):
    """
    Perceptual loss using features from pretrained deep neural networks trained. The function supports networks
    pretrained on ImageNet that use the LPIPS approach from: Zhang, et al. "The unreasonable effectiveness of deep
    features as a perceptual metric." https://arxiv.org/abs/1801.03924
    The fake 3D implementation is based on a 2.5D approach where we calculate the 2D perceptual on slices from the
    three axis.

    Args:
        spatial_dims: number of spatial dimensions.
        network_type: {``"alex"``, ``"vgg"``, ``"squeeze"``}
            Specifies the network architecture to use. Defaults to ``"alex"``.
        is_fake_3d: if True use 2.5D approach for a 3D perceptual loss.
        slices_per_axis_ratio: ratio of how many slices per axis are used in the 2.5D approach.
    """

    def __init__(
        self,
        spatial_dims: int,
        network_type: str = "alex",
        is_fake_3d: bool = True,
        slices_per_axis_ratio: float = 0.5,
    ):
        super().__init__()

        if spatial_dims not in [2, 3]:
            raise NotImplementedError("Perceptual loss is implemented only in 2D and 3D.")

        if spatial_dims == 3 and is_fake_3d is False:
            raise NotImplementedError("True 3D perceptual loss is not implemented.")

        self.spatial_dims = spatial_dims
        self.perceptual_function = LPIPS(
            pretrained=True,
            net=network_type,
            verbose=False,
        )
        self.is_fake_3d = is_fake_3d
        self.slices_per_axis_ratio = slices_per_axis_ratio

    def _calculate_fake_3d_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculating perceptual loss after one spatial axis is batchified according to permute dims.
        """

        # Sagittal axis
        input_2d_slices = input.float().permute(0, 2, 1, 3, 4).contiguous()
        input_2d_slices = input_2d_slices.view(-1, input.shape[1], input.shape[3], input.shape[4])

        target_2d_slices = target.float().permute(0, 2, 1, 3, 4).contiguous()
        target_2d_slices = target_2d_slices.view(-1, target.shape[1], target.shape[3], target.shape[4])

        num_slices = input_2d_slices.shape[0]
        indices = torch.randperm(num_slices)[: int(num_slices * self.slices_per_axis_ratio)]
        input_2d_slices = input_2d_slices[indices]
        target_2d_slices = target_2d_slices[indices]

        loss_sagital = torch.mean(self.perceptual_function(input_2d_slices, target_2d_slices))

        # Axial axis
        input_2d_slices = input.float().permute(0, 4, 1, 2, 3).contiguous()
        input_2d_slices = input_2d_slices.view(-1, input.shape[1], input.shape[2], input.shape[3])

        target_2d_slices = target.float().permute(0, 4, 1, 2, 3).contiguous()
        target_2d_slices = target_2d_slices.view(-1, target.shape[1], target.shape[2], target.shape[3])

        num_slices = input_2d_slices.shape[0]
        indices = torch.randperm(num_slices)[: int(num_slices * self.slices_per_axis_ratio)]
        input_2d_slices = input_2d_slices[indices]
        target_2d_slices = target_2d_slices[indices]

        loss_axial = torch.mean(self.perceptual_function(input_2d_slices, target_2d_slices))

        # Coronal axis
        input_2d_slices = input.float().permute(0, 3, 1, 2, 4).contiguous()
        input_2d_slices = input_2d_slices.view(-1, input.shape[1], input.shape[2], input.shape[4])

        target_2d_slices = target.float().permute(0, 3, 1, 2, 4).contiguous()
        target_2d_slices = target_2d_slices.view(-1, target.shape[1], target.shape[2], target.shape[4])

        num_slices = input_2d_slices.shape[0]
        indices = torch.randperm(num_slices)[: int(num_slices * self.slices_per_axis_ratio)]
        input_2d_slices = input_2d_slices[indices]
        target_2d_slices = target_2d_slices[indices]

        loss_coronal = torch.mean(self.perceptual_function(input_2d_slices, target_2d_slices))

        return loss_sagital + loss_axial + loss_coronal

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNHW[D].
            target: the shape should be BNHW[D].
        """
        if target.shape != input.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        if self.spatial_dims == 2:
            loss = self.perceptual_function(input, target)
        elif self.spatial_dims == 3 and self.is_fake_3d:
            loss = self._calculate_fake_3d_loss(input, target)

        return torch.mean(loss)
