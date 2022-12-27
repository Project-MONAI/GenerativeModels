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

from typing import Tuple

import torch
import torch.nn as nn
from lpips import LPIPS


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using features from pretrained deep neural networks trained. The function supports networks
    pretrained on ImageNet that use the LPIPS approach from: Zhang, et al. "The unreasonable effectiveness of deep
    features as a perceptual metric." https://arxiv.org/abs/1801.03924
    The fake 3D implementation is based on a 2.5D approach where we calculate the 2D perceptual on slices from the
    three axis.

    Args:
        spatial_dims: number of spatial dimensions.
        network_type: {``"alex"``, ``"vgg"``, ``"squeeze"``, ``"medicalnet_resnet10_23datasets"``,
            ``"medicalnet_resnet50_23datasets"``}
            Specifies the network architecture to use. Defaults to ``"alex"``.
        is_fake_3d: if True use 2.5D approach for a 3D perceptual loss.
        fake_3d_ratio: ratio of how many slices per axis are used in the 2.5D approach.
    """

    def __init__(
        self,
        spatial_dims: int,
        network_type: str = "alex",
        is_fake_3d: bool = True,
        fake_3d_ratio: float = 0.5,
    ):
        super().__init__()

        if spatial_dims not in [2, 3]:
            raise NotImplementedError("Perceptual loss is implemented only in 2D and 3D.")

        self.spatial_dims = spatial_dims
        if spatial_dims == 3 and is_fake_3d is False:
            self.perceptual_function = MedicalNetPerceptualComponent(net=network_type, verbose=False)
        else:
            self.perceptual_function = LPIPS(
                pretrained=True,
                net=network_type,
                verbose=False,
            )
        self.is_fake_3d = is_fake_3d
        self.fake_3d_ratio = fake_3d_ratio

    def _calculate_axis_loss(self, input: torch.Tensor, target: torch.Tensor, spatial_axis: int) -> torch.Tensor:
        """
        Calculate perceptual loss in one of the axis used in the 2.5D approach. After the slices of one spatial axis
        is transformed into different instances in the batch, we compute the loss using the 2D approach.

        Args:
            input: input 5D tensor. BNHWD
            target: target 5D tensor. BNHWD
            spatial_axis: spatial axis to obtain the 2D slices.
        """

        def batchify_axis(x: torch.Tensor, fake_3d_perm: Tuple) -> torch.Tensor:
            """
            Transform slices from one spatial axis into different instances in the batch.
            """
            slices = x.float().permute((0,) + fake_3d_perm).contiguous()
            slices = slices.view(-1, x.shape[fake_3d_perm[1]], x.shape[fake_3d_perm[2]], x.shape[fake_3d_perm[3]])

            return slices

        preserved_axes = [2, 3, 4]
        preserved_axes.remove(spatial_axis)

        channel_axis = 1
        input_slices = batchify_axis(
            x=input,
            fake_3d_perm=(
                spatial_axis,
                channel_axis,
            )
            + tuple(preserved_axes),
        )
        indices = torch.randperm(input_slices.shape[0])[: int(input_slices.shape[0] * self.fake_3d_ratio)].to(
            input_slices.device
        )
        input_slices = torch.index_select(input_slices, dim=0, index=indices)
        target_slices = batchify_axis(
            x=target,
            fake_3d_perm=(
                spatial_axis,
                channel_axis,
            )
            + tuple(preserved_axes),
        )
        target_slices = torch.index_select(target_slices, dim=0, index=indices)

        axis_loss = torch.mean(self.perceptual_function(input_slices, target_slices))

        return axis_loss

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
            # Compute 2.5D approach
            loss_sagittal = self._calculate_axis_loss(input, target, spatial_axis=2)
            loss_coronal = self._calculate_axis_loss(input, target, spatial_axis=3)
            loss_axial = self._calculate_axis_loss(input, target, spatial_axis=4)
            loss = loss_sagittal + loss_axial + loss_coronal
        if self.spatial_dims == 3 and self.is_fake_3d is False:
            loss = self.perceptual_function(input, target)

        return torch.mean(loss)


class MedicalNetPerceptualComponent(nn.Module):
    """
    Component to perform the perceptual evaluation with the networks pretrained by Chen, et al. "Med3D: Transfer
    Learning for 3D Medical Image Analysis". This class uses torch Hub to download the networks from
    "Warvito/MedicalNet-models".

    Args:
        net: {``"medicalnet_resnet10_23datasets"``, ``"medicalnet_resnet50_23datasets"``}
            Specifies the network architecture to use. Defaults to ``"medicalnet_resnet10_23datasets"``.
        verbose: if false, mute messages from torch Hub load function.
    """

    def __init__(
        self,
        net: str = "medicalnet_resnet10_23datasets",
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.model = torch.hub.load("Warvito/MedicalNet-models", model=net, verbose=verbose)
        self.eval()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss using MedicalNet 3D networks.
        The outputs are normalised across the channels, and we obtain the mean from the spatial dimensions (similar
        approach to the lpips package).
        """
        input = medicalnet_intensty_normalisation(input)
        target = medicalnet_intensty_normalisation(target)

        # Get model outputs
        outs_input = self.model.forward(input)
        outs_target = self.model.forward(target)

        # Normalise through the channels
        feats_input = normalize_tensor(outs_input)
        feats_target = normalize_tensor(outs_target)

        results = (feats_input - feats_target) ** 2
        results = spatial_average_3d(results.sum(dim=1, keepdim=True), keepdim=True)

        return results


def spatial_average_3d(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3, 4], keepdim=keepdim)


def normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def medicalnet_intensty_normalisation(volume):
    """Based on https://github.com/Tencent/MedicalNet/blob/18c8bb6cd564eb1b964bffef1f4c2283f1ae6e7b/datasets/brains18.py#L133"""
    mean = volume.mean()
    std = volume.std()
    return (volume - mean) / std
