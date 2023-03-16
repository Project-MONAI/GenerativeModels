from __future__ import annotations

import nibabel as nib
import numpy as np
import torch


class NiftiSaver:
    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.affine = np.array(
            [
                [-1.0, 0.0, 0.0, 96.48149872],
                [0.0, 1.0, 0.0, -141.47715759],
                [0.0, 0.0, 1.0, -156.55375671],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def save(self, image_data: torch.Tensor, file_name: str) -> None:
        image_data = image_data.cpu().numpy()
        image_data = image_data[0, 0, 5:-5, 5:-5, :-15]
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
        image_data = (image_data * 255).astype(np.uint8)

        empty_header = nib.Nifti1Header()
        sample_nii = nib.Nifti1Image(image_data, self.affine, empty_header)
        nib.save(sample_nii, f"{str(self.output_dir)}/{file_name}.nii.gz")
