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

from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from monai.config import IgniteInfo
from monai.engines.trainer import Trainer
from monai.engines.utils import CommonKeys as Keys
from monai.engines.utils import default_prepare_batch
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform
from monai.utils import min_version, optional_import
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from generative.utils.enums import AdversarialKeys

if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")


class AdversarialTrainer(Trainer):
    """
    Adversarial network training, inherits from ``Trainer`` and ``Workflow``.
    Args:
        device: an object representing the device on which to run.
        max_epochs: the total epoch number for engine to run.
        train_data_loader: Core ignite engines uses `DataLoader` for training loop batchdata.
        recon_loss_function: G loss fcuntion for reconstructions.
        g_network: ''generator'' (G) network architecture.
        g_optimizer: G optimizer function.
        g_loss_function: G loss function for adversarial training.
        d_network: discriminator (D) network architecture.
        d_optimizer: D optimizer function.
        d_loss_function: D loss function for adversarial training..
        epoch_length: number of iterations for one epoch, default to `len(train_data_loader)`.
        g_inferer: inference method to execute G model forward. Defaults to ``SimpleInferer()``.
            optimization step. It is expected to receive g_infer's output, device and non_blocking.
        d_inferer: inference method to execute D model forward. Defaults to ``SimpleInferer()``.
            receive the 'batchdata', g_inferer's output, device and non_blocking.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse image and label for current iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        post_transform: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_train_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_train_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        train_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.
        amp: whether to enable auto-mixed-precision training or inference, default is False.
    """

    def __init__(
        self,
        device: torch.device,
        max_epochs: int,
        train_data_loader: DataLoader,
        recon_loss_function: Callable,
        g_network: torch.nn.Module,
        g_optimizer: Optimizer,
        g_loss_function: Callable,
        d_network: torch.nn.Module,
        d_optimizer: Optimizer,
        d_loss_function: Callable,
        epoch_length: Optional[int] = None,
        g_inferer: Optional[Inferer] = None,
        d_inferer: Optional[Inferer] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        post_transform: Optional[Transform] = None,
        key_train_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        train_handlers: Optional[Sequence] = None,
        amp: bool = False,
        use_adversarial_adaptive_weight: bool = False,
        adaptive_adversarial_weight_threshold: int = 0,
        adaptive_adversarial_weight_value: float = 0,
    ):
        # set up Ignite engine and environments
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            data_loader=train_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            key_metric=key_train_metric,
            additional_metrics=additional_metrics,
            handlers=train_handlers,
            post_transform=post_transform,
            amp=amp,
        )
        self.g_network = g_network
        self.g_optimizer = g_optimizer
        self.g_loss_function = g_loss_function
        self.recon_loss_function = recon_loss_function

        self.d_network = d_network
        self.d_optimizer = d_optimizer
        self.d_loss_function = d_loss_function

        self.g_inferer = SimpleInferer() if g_inferer is None else g_inferer
        self.d_inferer = SimpleInferer() if d_inferer is None else d_inferer

        self.g_scaler = torch.cuda.amp.GradScaler() if self.amp else None
        self.d_scaler = torch.cuda.amp.GradScaler() if self.amp else None

        self.use_adversarial_adaptive_weight = use_adversarial_adaptive_weight
        self.adaptive_adversarial_weight_threshold = adaptive_adversarial_weight_threshold
        self.adaptive_adversarial_weight_value = adaptive_adversarial_weight_value

    def _iteration(
        self, engine: Engine, batchdata: Union[Dict, Sequence]
    ) -> Dict[str, Union[torch.Tensor, int, float, bool]]:
        """
        Callback function for Adversarial Training processing logic of 1 iteration in Ignite Engine.
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.
        Returns:
            Dict: Results of the iterations
                - REALS: image Tensor data for model input, already moved to device.
                - FAKES: output Tensor data corresponding to the image, already moved to device.
                - GLOSS: the loss of the g_network
                - DLOSS: the loss of the d_network
        Raises:
            ValueError: When ``batchdata`` is None.
        """
        if batchdata is None:
            raise ValueError("must provide batch data for current iteration.")

        batch = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)

        if len(batch) == 2:
            inputs, targets = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, targets, args, kwargs = batch

        batch_size = self.data_loader.batch_size  # type: ignore

        # Train generator
        self.g_network.train()
        self.g_optimizer.zero_grad(set_to_none=True)

        if self.amp and self.g_scaler is not None:
            with torch.cuda.amp.autocast():
                g_predictions = self.g_inferer(inputs, self.g_network, *args, **kwargs)

                logits_fake = self.d_inferer(
                    g_predictions["reconstruction"][0].float().contiguous(), self.d_network, *args, **kwargs
                )

                reconstruction_loss = self.recon_loss_function(g_predictions, targets).mean()

                generator_loss = self.g_loss_function(logits_fake).mean()
                adversarial_weight = self.adaptive_adversarial_weight(
                    reconstruction_loss=reconstruction_loss,
                    generator_loss=generator_loss,
                    global_step=self.state.epoch,
                    threshold=self.adaptive_adversarial_weight_threshold,
                    value=self.adaptive_adversarial_weight_value,
                )
                generator_loss = reconstruction_loss + generator_loss * adversarial_weight

            self.g_scaler.scale(generator_loss).backward()
            self.g_scaler.step(self.g_optimizer)
            self.g_scaler.update()
        else:
            g_predictions = self.g_inferer(inputs, self.g_network, *args, **kwargs)
            logits_fake = self.d_inferer(
                g_predictions["reconstruction"][0].float().contiguous(), self.d_network, *args, **kwargs
            )
            reconstruction_loss = self.recon_loss_function(g_predictions, targets).mean()

            generator_loss = self.g_loss_function(logits_fake).mean()
            adversarial_weight = self.adaptive_adversarial_weight(
                reconstruction_loss=reconstruction_loss,
                generator_loss=generator_loss,
                global_step=self.state.epoch,
                threshold=self.adaptive_adversarial_weight_threshold,
                value=self.adaptive_adversarial_weight_value,
            )
            generator_loss = reconstruction_loss + generator_loss * adversarial_weight
            generator_loss.backward()
            self.g_optimizer.step()

        # Train Discriminator
        self.d_network.train()
        self.d_network.zero_grad(set_to_none=True)

        if self.amp and self.d_scaler is not None:
            with torch.cuda.amp.autocast():
                logits_fake = self.d_inferer(
                    g_predictions["reconstruction"][0].float().contiguous().detach(), self.d_network, *args, **kwargs
                )

                logits_real = self.d_inferer(inputs.contiguous().detach(), self.d_network, *args, **kwargs)

                d_loss = self.d_loss_function(logits_fake, logits_real).mean() * adversarial_weight

            self.d_scaler.scale(d_loss).backward()
            self.d_scaler.step(self.d_optimizer)
            self.d_scaler.update()
        else:
            logits_fake = self.d_inferer(
                g_predictions["reconstruction"][0].float().contiguous().detach(), self.d_network, *args, **kwargs
            )

            logits_real = self.d_inferer(inputs.contiguous().detach(), self.d_network, *args, **kwargs)
            d_loss = self.d_loss_function(logits_fake, logits_real).mean() * adversarial_weight
            d_loss.backward()
            self.d_optimizer.step()

        return {
            Keys.IMAGE: inputs,
            Keys.LABEL: targets,
            Keys.PRED: g_predictions,
            Keys.LOSS: reconstruction_loss.item(),
            AdversarialKeys.REALS: inputs,
            AdversarialKeys.FAKES: g_predictions,
            AdversarialKeys.GLOSS: generator_loss.item(),
            AdversarialKeys.DLOSS: d_loss.item(),
        }

    def adaptive_adversarial_weight(
        self,
        reconstruction_loss: torch.nn.Module,
        generator_loss: torch.nn.Module,
        global_step: int,
        threshold: int = 0,
        value: float = 1,
    ):
        if self.use_adversarial_adaptive_weight:
            generator_last_layer = self.g_network.get_last_layer()
            nll_grads = torch.autograd.grad(reconstruction_loss, generator_last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(generator_loss, generator_last_layer, retain_graph=True)[0]

            weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
            weight = torch.clamp(weight, 0.0, 1e4).detach()

            if global_step < threshold:
                weight = value
        else:
            weight = 1

        return weight
