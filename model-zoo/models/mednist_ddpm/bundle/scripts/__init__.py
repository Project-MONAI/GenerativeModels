
from typing import Dict, Mapping, Optional, Union
import torch
from monai.engines import PrepareBatch, default_prepare_batch


class DiffusionPrepareBatch(PrepareBatch):
    """
    This class is used as a callable for the `prepare_batch` parameter of engine classes for diffusion training.

    Assuming a supervised training process, it will generate a noise field using `get_noise` for an input image, and
    return the image and noise field as the image/target pair plus the noise field the kwargs under the key "noise".
    This assumes the inferer being used in conjunction with this class expects a "noise" parameter to be provided.

    If the `condition_name` is provided, this must refer to a key in the input dictionary containing the condition
    field to be passed to the inferer. This will appear in the keyword arguments under the key "condition".

    """

    def __init__(self, num_train_timesteps: int, condition_name: Optional[str] = None):
        self.condition_name = condition_name
        self.num_train_timesteps = num_train_timesteps

    def get_noise(self, images: torch.Tensor) -> torch.Tensor:
        """Returns the noise tensor for input tensor `images`, override this for different noise distributions."""
        return torch.randn_like(images)

    def get_timesteps(self, images: torch.Tensor) -> torch.Tensor:
        return torch.randint(0, self.num_train_timesteps, (images.shape[0],), device=images.device).long()

    def __call__(
        self,
        batchdata: Dict[str, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
        **kwargs,
    ):
        images, _ = default_prepare_batch(batchdata, device, non_blocking, **kwargs)
        noise = self.get_noise(images).to(device, non_blocking=non_blocking, **kwargs)
        timesteps = self.get_timesteps(images).to(device, non_blocking=non_blocking, **kwargs)

        kwargs = {"noise": noise, "timesteps": timesteps}

        if self.condition_name is not None and isinstance(batchdata, Mapping):
            kwargs["conditioning"] = batchdata[self.condition_name].to(device, non_blocking=non_blocking, **kwargs)

        # return input, target, arguments, and keyword arguments where noise is the target and also a keyword value
        return images, noise, (), kwargs


def inv_metric_cmp_fn(current_metric: float, prev_best: float) -> bool:
    """
    This inverts comparison for those metrics which reduce like loss values, such that the lower one is better.

    Args:
        current_metric: metric value of current round computation.
        prev_best: the best metric value of previous rounds to compare with.
    """
    return current_metric < prev_best
