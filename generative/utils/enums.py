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

from typing import TYPE_CHECKING

from monai.config import IgniteInfo
from monai.utils import StrEnum, min_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import EventEnum
else:
    EventEnum, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum", as_type="base"
    )


class AdversarialKeys(StrEnum):
    REALS = "reals"
    REAL_LOGITS = "real_logits"
    FAKES = "fakes"
    FAKE_LOGITS = "fake_logits"
    RECONSTRUCTION_LOSS = "reconstruction_loss"
    GENERATOR_LOSS = "generator_loss"
    DISCRIMINATOR_LOSS = "discriminator_loss"


class AdversarialIterationEvents(EventEnum):
    RECONSTRUCTION_LOSS_COMPLETED = "reconstruction_loss_completed"
    GENERATOR_FORWARD_COMPLETED = "generator_forward_completed"
    GENERATOR_DISCRIMINATOR_FORWARD_COMPLETED = "generator_discriminator_forward_completed"
    GENERATOR_LOSS_COMPLETED = "generator_loss_completed"
    GENERATOR_BACKWARD_COMPLETED = "generator_backward_completed"
    GENERATOR_MODEL_COMPLETED = "generator_model_completed"
    DISCRIMINATOR_REALS_FORWARD_COMPLETED = "discriminator_reals_forward_completed"
    DISCRIMINATOR_FAKES_FORWARD_COMPLETED = "discriminator_fakes_forward_completed"
    DISCRIMINATOR_LOSS_COMPLETED = "discriminator_loss_completed"
    DISCRIMINATOR_BACKWARD_COMPLETED = "discriminator_backward_completed"
    DISCRIMINATOR_MODEL_COMPLETED = "discriminator_model_completed"


class OrderingType(StrEnum):
    RASTER_SCAN = "raster_scan"
    S_CURVE = "s_curve"
    RANDOM = "random"


class OrderingTransformations(StrEnum):
    ROTATE_90 = "rotate_90"
    TRANSPOSE = "transpose"
    REFLECT = "reflect"
