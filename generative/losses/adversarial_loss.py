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

from __future__ import annotations

import warnings

import torch
from monai.networks.layers.utils import get_act_layer
from monai.utils import LossReduction
from monai.utils.enums import StrEnum
from torch.nn.modules.loss import _Loss


class AdversarialCriterions(StrEnum):
    BCE = "bce"
    HINGE = "hinge"
    LEAST_SQUARE = "least_squares"


class PatchAdversarialLoss(_Loss):
    """
    Calculates an adversarial loss on a Patch Discriminator or a Multi-scale Patch Discriminator.
    Warning: due to the possibility of using different criterions, the output of the discrimination
    mustn't be passed to a final activation layer. That is taken care of internally within the loss.

    Args:
        reduction: {``"none"``, ``"mean"``, ``"sum"``} Specifies the reduction to apply to the output.
        Defaults to ``"mean"``.
        - ``"none"``: no reduction will be applied.
        - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
        - ``"sum"``: the output will be summed.
        criterion: which criterion (hinge, least_squares or bce) you want to use on the discriminators outputs.
        Depending on the criterion, a different activation layer will be used. Make sure you don't run the outputs
        through an activation layer prior to calling the loss.
        no_activation_leastsq: if True, the activation layer in the case of least-squares is removed.
    """

    def __init__(
        self,
        reduction: LossReduction | str = LossReduction.MEAN,
        criterion: str = AdversarialCriterions.LEAST_SQUARE.value,
        no_activation_leastsq: bool = False,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)

        if criterion.lower() not in [m.value for m in AdversarialCriterions]:
            raise ValueError(
                "Unrecognised criterion entered for Adversarial Loss. Must be one in: %s"
                % ", ".join([m.value for m in AdversarialCriterions])
            )

        # Depending on the criterion, a different activation layer is used.
        self.real_label = 1.0
        self.fake_label = 0.0
        if criterion == AdversarialCriterions.BCE.value:
            self.activation = get_act_layer("SIGMOID")
            self.loss_fct = torch.nn.BCELoss(reduction=reduction)
        elif criterion == AdversarialCriterions.HINGE.value:
            self.activation = get_act_layer("TANH")
            self.fake_label = -1.0
        elif criterion == AdversarialCriterions.LEAST_SQUARE.value:
            if no_activation_leastsq:
                self.activation = None
            else:
                self.activation = get_act_layer(name=("LEAKYRELU", {"negative_slope": 0.05}))
            self.loss_fct = torch.nn.MSELoss(reduction=reduction)

        self.criterion = criterion
        self.reduction = reduction

    def get_target_tensor(self, input: torch.FloatTensor, target_is_real: bool) -> torch.Tensor:
        """
        Gets the ground truth tensor for the discriminator depending on whether the input is real or fake.

        Args:
            input: input tensor from the discriminator (output of discriminator, or output of one of the multi-scale
            discriminator). This is used to match the shape.
            target_is_real: whether the input is real or wannabe-real (1s) or fake (0s).
        Returns:
        """
        filling_label = self.real_label if target_is_real else self.fake_label
        label_tensor = torch.tensor(1).fill_(filling_label).type(input.type()).to(input[0].device)
        label_tensor.requires_grad_(False)
        return label_tensor.expand_as(input)

    def get_zero_tensor(self, input: torch.FloatTensor) -> torch.Tensor:
        """
        Gets a zero tensor.

        Args:
            input: tensor which shape you want the zeros tensor to correspond to.
        Returns:
        """

        zero_label_tensor = torch.tensor(0).type(input[0].type()).to(input[0].device)
        zero_label_tensor.requires_grad_(False)
        return zero_label_tensor.expand_as(input)

    def forward(
        self, input: torch.FloatTensor | list, target_is_real: bool, for_discriminator: bool
    ) -> torch.Tensor | list[torch.Tensor]:
        """

        Args:
            input: output of Multi-Scale Patch Discriminator or Patch Discriminator; being a list of
            tensors or a tensor; they shouldn't have gone through an activation layer.
            target_is_real: whereas the input corresponds to discriminator output for real or fake images
            for_discriminator: whereas this is being calculated for discriminator or generator loss. In the last
            case, target_is_real is set to True, as the generator wants the input to be dimmed as real.
        Returns: if reduction is None, returns a list with the loss tensors of each discriminator if multi-scale
        discriminator is active, or the loss tensor if there is just one discriminator. Otherwise, it returns the
        summed or mean loss over the tensor and discriminator/s.

        """

        if not for_discriminator and not target_is_real:
            target_is_real = True  # With generator, we always want this to be true!
            warnings.warn(
                "Variable target_is_real has been set to False, but for_discriminator is set"
                "to False. To optimise a generator, target_is_real must be set to True."
            )

        if type(input) is not list:
            input = [input]
        target_ = []
        for _, disc_out in enumerate(input):
            if self.criterion != AdversarialCriterions.HINGE.value:
                target_.append(self.get_target_tensor(disc_out, target_is_real))
            else:
                target_.append(self.get_zero_tensor(disc_out))

        # Loss calculation
        loss = []
        for disc_ind, disc_out in enumerate(input):
            if self.activation is not None:
                disc_out = self.activation(disc_out)
            if self.criterion == AdversarialCriterions.HINGE.value and not target_is_real:
                loss_ = self.forward_single(-disc_out, target_[disc_ind])
            else:
                loss_ = self.forward_single(disc_out, target_[disc_ind])
            loss.append(loss_)

        if loss is not None:
            if self.reduction == LossReduction.MEAN.value:
                loss = torch.mean(torch.stack(loss))
            elif self.reduction == LossReduction.SUM.value:
                loss = torch.sum(torch.stack(loss))

        return loss

    def forward_single(self, input: torch.FloatTensor, target: torch.FloatTensor) -> torch.Tensor | None:
        if (
            self.criterion == AdversarialCriterions.BCE.value
            or self.criterion == AdversarialCriterions.LEAST_SQUARE.value
        ):
            return self.loss_fct(input, target)
        elif self.criterion == AdversarialCriterions.HINGE.value:
            minval = torch.min(input - 1, self.get_zero_tensor(input))
            return -torch.mean(minval)
        else:
            return None
