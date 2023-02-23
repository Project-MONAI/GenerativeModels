# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This example shows how to execute distributed training based on PyTorch native `DistributedDataParallel` module.
It can run on several nodes with multiple GPU devices on every node.

This example is based on the MedNIST Hand dataset.

If you do not have enough GPU memory, you can try to decrease the input parameter `cache_rate`.

Main steps to set up the distributed training:

- Execute `torchrun` to create processes on every node for every GPU.
  It receives parameters as below:
  `--nproc_per_node=NUM_GPUS_PER_NODE`
  `--nnodes=NUM_NODES`
  `--node_rank=INDEX_CURRENT_NODE`
  For more details, refer to https://pytorch.org/docs/stable/elastic/run.html.
- Wrap the model with `DistributedDataParallel` after moving to expected device.
- Partition dataset before training, so every rank process will only handle its own data partition.

Note:
    `torchrun` will launch `nnodes * nproc_per_node = world_size` processes in total.
    Suggest setting exactly the same software environment for every node, especially `PyTorch`, `nccl`, etc.
    A good practice is to use the same MONAI docker image for all nodes directly.
    Example script to execute this program on every node:
    torchrun --nproc_per_node=NUM_GPUS_PER_NODE
           --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE
           ddpm_training_ddp.py -d DIR_OF_TESTDATA

Referring to: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

This code is based on https://github.com/Project-MONAI/tutorials/blob/main/acceleration/distributed_training/brats_training_ddp.py

"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import monai.inferers
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai import transforms
from monai.apps import MedNISTDataset
from monai.data import DataLoader, ThreadDataLoader, partition_dataset
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler


class MedNISTCacheDataset(MedNISTDataset):
    """
    Enhance the MedNISTDataset to support distributed data parallel.

    """

    def __init__(
        self,
        root_dir: str,
        section: str,
        transform: Optional[transforms.Transform] = None,
        cache_rate: float = 1.0,
        num_workers: int = 0,
        shuffle: bool = False,
    ) -> None:
        if not os.path.isdir(root_dir):
            raise ValueError("root directory root_dir must be a directory.")
        self.section = section
        self.shuffle = shuffle
        self.val_frac = 0.2
        self.test_frac = 0.0
        self.set_random_state(seed=0)
        dataset_dir = Path(root_dir) / "MedNIST"
        if not os.path.exists(dataset_dir):
            raise RuntimeError(f"cannot find dataset directory: {dataset_dir}, please download it.")
        data = self._generate_data_list(dataset_dir)
        super(MedNISTDataset, self).__init__(data, transform, cache_rate=cache_rate, num_workers=num_workers)

    def _generate_data_list(self, dataset_dir: Path):
        data = super()._generate_data_list(dataset_dir)
        # only extract hand data
        data = [{"image": item["image"]} for item in data if item["class_name"] == "Hand"]
        # partition dataset based on current rank number, every rank trains with its own data
        # it can avoid duplicated caching content in each rank, but will not do global shuffle before every epoch
        return partition_dataset(
            data=data,
            num_partitions=dist.get_world_size(),
            shuffle=self.shuffle,
            seed=0,
            drop_last=False,
            even_divisible=self.shuffle,
        )[dist.get_rank()]


def main_worker(args):
    # disable logging for processes except 0 on every node
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"missing directory {args.data_dir}")

    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    # use amp to accelerate training
    scaler = GradScaler()
    torch.backends.cudnn.benchmark = True

    total_start = time.time()
    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            transforms.RandAffined(
                keys=["image"],
                rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
                translate_range=[(-1, 1), (-1, 1)],
                scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
                spatial_size=[64, 64],
                padding_mode="zeros",
                prob=0.5,
            ),
        ]
    )

    # create a training data loader

    train_ds = MedNISTCacheDataset(
        root_dir=args.data_dir,
        transform=train_transforms,
        section="training",
        num_workers=4,
        cache_rate=args.cache_rate,
        shuffle=True,
    )
    # ThreadDataLoader can be faster if no IO operations when caching all the data in memory
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=args.batch_size, shuffle=True)

    # validation transforms and dataset
    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        ]
    )
    val_ds = MedNISTCacheDataset(
        root_dir=args.data_dir,
        transform=val_transforms,
        section="validation",
        num_workers=4,
        cache_rate=args.cache_rate,
        shuffle=False,
    )
    # ThreadDataLoader can be faster if no IO operations when caching all the data in memory
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=args.batch_size, shuffle=False)

    # create network, loss function and optimizer
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
    )
    model = model.to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)

    inferer = DiffusionInferer(scheduler)
    # wrap the model with DistributedDataParallel module
    model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    # start a typical PyTorch training
    best_metric = 10000
    best_metric_epoch = 1000
    print(f"Time elapsed before training: {time.time() - total_start}")
    train_start = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.epochs}")
        epoch_loss = train(train_loader, model, optimizer, inferer, scaler, device)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % args.val_interval == 0:
            metric = evaluate(model, val_loader, inferer, device)

            if metric < best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                if dist.get_rank() == 0:
                    torch.save(model.module.state_dict(), Path(args.output_dir) / "best_metric_model.pth")
                    print(f"Saving model at epoch {epoch+1}")
            print(
                f"current epoch: {epoch + 1} current val loss: {metric:.4f}"
                f"\nbest MSE loss: {best_metric:.4f} at epoch: {best_metric_epoch}"
            )

        print(f"Training time for epoch {epoch + 1} was: {(time.time() - epoch_start):.4f}s")

    print(
        f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch},"
        f"Total train time: {(time.time() - train_start):.4f}"
    )
    dist.destroy_process_group()


def train(
    train_loader: DataLoader,
    model: torch.nn,
    optimizer: torch.optim.Optimizer,
    inferer: monai.inferers.Inferer,
    scaler: GradScaler,
    device: torch.device,
):
    model.train()
    step = 0
    epoch_len = len(train_loader)
    epoch_loss = 0
    step_start = time.time()
    for batch_data in train_loader:
        step += 1
        images = batch_data["image"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(images).to(device)

            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            # Get model prediction
            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)

            loss = F.mse_loss(noise_pred.float(), noise.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, step time: {(time.time() - step_start):.4f}")
        step_start = time.time()
    epoch_loss /= step

    return epoch_loss


def evaluate(model: torch.nn, val_loader: DataLoader, inferer: monai.inferers.Inferer, device: torch.device):
    model.eval()
    val_epoch_loss = 0
    with torch.no_grad():
        for step, batch_data in enumerate(val_loader):
            images = batch_data["image"].to(device)
            with autocast(enabled=True):
                noise = torch.randn_like(images).to(device)
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()
                noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                val_loss = F.mse_loss(noise_pred.float(), noise.float())

            val_epoch_loss += val_loss.item()
        val_epoch_loss = val_epoch_loss / (step + 1)
    return val_epoch_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", default="./testdata", type=str, help="directory of downloaded MedNIST dataset"
    )
    parser.add_argument("--output_dir", default="/project", type=str, help="directory to save outputs")
    parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("-b", "--batch_size", default=1, type=int, help="mini-batch size of every GPU")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training.")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="larger cache rate relies on enough GPU memory.")
    parser.add_argument("--val_interval", type=int, default=5)
    args = parser.parse_args()

    if args.seed is not None:
        set_determinism(seed=args.seed)
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    main_worker(args=args)


# usage example (refer to https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py):

# torchrun --nproc_per_node=NUM_GPUS_PER_NODE
#        --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE
#        ddpm_training_ddp.py -d DIR_OF_TESTDATA

if __name__ == "__main__":
    main()
