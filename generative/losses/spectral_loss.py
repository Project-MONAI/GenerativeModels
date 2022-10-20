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

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.fft import fftn
from torch.nn.modules.loss import _Loss


# TODO: Check removed TBSummaryTypes
class SpectralLoss(_Loss):
    """
    Loss function that has a spectral component based on the amplitude and phase of FFT
    and a pixel component based on mean absolute error and mean squared error.

    References:
        [1] Takaki, S., Nakashika, T., Wang, X. and Yamagishi, J., 2019, May.
        STFT spectral loss for training a neural speech waveform model.
        In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
        (pp. 7065-7069). IEEE.

    Args:
        dimensions: Number of spatial dimensions.
        include_pixel_loss : If the loss includes the pixel component as well
        fft_kwargs: Dictionary hold all FFT arguments that are to be used when calling torch.fft.fftn.
            Defaults to:  {'s': None, 'dims': tuple(range(1, self.dimensions + 2)), 'norm': 'ortho'}
        size_average: Deprecated (see reduction). By default, the losses are averaged over each loss element
            in the batch. Note that for some losses, there are multiple elements per sample. If the field
            size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is
            False. Default: True
        reduce: Deprecated (see reduction). By default, the losses are averaged or summed over observations
            for each minibatch depending on size_average. When reduce is False, returns a loss per batch element
            instead and ignores size_average. Default: True
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no
            reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in
            the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being
            deprecated, and in the meantime, specifying either of those two args will override reduction.
            Default: 'mean'
    """

    def __init__(
        self,
        dimensions: int,
        include_pixel_loss: bool = True,
        fft_kwargs: Dict = None,
        size_average: bool = True,
        reduce: bool = True,
        reduction: str = "mean",
    ):
        super(SpectralLoss, self).__init__(size_average, reduce, reduction)

        self.dimensions = dimensions
        self.include_pixel_loss = include_pixel_loss
        self.fft_factor: float = 1.0
        self.fft_kwargs = (
            {"s": None, "dim": tuple(range(1, self.dimensions + 2)), "norm": "ortho"}
            if fft_kwargs is None
            else fft_kwargs
        )

    def forward(self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor) -> torch.Tensor:
        # Unpacking elements
        y = y.float()
        y_pred = network_output["reconstruction"][0].float()
        q_losses = network_output["quantization_losses"]

        # Calculating amplitudes and phases
        y_amplitude, y_phase = self._get_fft_amplitude_and_phase(y)
        yp_amplitude, yp_phase = self._get_fft_amplitude_and_phase(y_pred)

        # Ref 1 - Sec 2.2 - Equation 7
        amplitude_loss = 0.5 * F.mse_loss(yp_amplitude, y_amplitude)

        # Ref 1 - Sec 2.3 - Equation 10
        phase_loss = torch.mean(0.5 * torch.abs((1 - torch.exp(torch.abs(yp_phase - y_phase))) ** 2))
        fft_loss = (amplitude_loss + phase_loss) * self.fft_factor
        loss = fft_loss

        if self.include_pixel_loss:
            l2_loss = F.mse_loss(y_pred, y, reduction=self.reduction)
            loss = loss + l2_loss

        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()
            loss += q_loss

        return loss

    def _get_fft_amplitude_and_phase(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Manually calculating the amplitude and phase of the fourier transformations representation of the images

        Args:
            images (torch.Tensor): Images that are to undergo fftn

        Returns:
            torch.Tensor: fourier transformation amplitude
            torch.Tensor: fourier transformation phase
        """
        img_fft = fftn(input=images, **self.fft_kwargs)

        amplitude = torch.sqrt(img_fft.real**2 + img_fft.imag**2)
        phase = torch.atan2(img_fft.imag, img_fft.real)

        return amplitude, phase

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries

    def get_fft_factor(self) -> float:
        return self.fft_factor

    def set_fft_factor(self, fft_factor: float) -> float:
        self.fft_factor = fft_factor

        return self.get_fft_factor()
