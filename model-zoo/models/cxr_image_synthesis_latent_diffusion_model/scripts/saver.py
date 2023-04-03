from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class JPGSaver:
    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = output_dir

    def save(self, image_data: torch.Tensor, file_name: str) -> None:
        image_data = np.clip(image_data.cpu().numpy(), 0, 1)
        image_data = (image_data * 255).astype(np.uint8)
        im = Image.fromarray(image_data[0, 0])
        im.save(self.output_dir + "/" + file_name + ".jpg")
