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

import unittest

import torch
from monai.networks import eval_mode
from parameterized import parameterized
from generative.networks.nets import AutoencoderKL
from generative.networks.nets import DiffusionModelUNet

AUTOENCODER_CASES = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, False),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "spade_norm": True,
            "label_nc": 3,
        },
        (1, 1, 16, 16),
        (1, 3, 16, 16),
        (1, 1, 16, 16),
        (1, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, True),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "spade_norm": True,
            "label_nc": 3,
            "spade_intermediate_channels": 32,
        },
        (1, 1, 16, 16),
        (1, 3, 16, 16),
        (1, 1, 16, 16),
        (1, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, True),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "spade_norm": True,
            "label_nc": 3,
        },
        (1, 1, 16, 16, 16),
        (1, 3, 16, 16, 16),
        (1, 1, 16, 16, 16),
        (1, 4, 4, 4, 4),
    ],
]
WRONG_AUTOENCODER_CASE = [
    [
    {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, False),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "spade_norm": True,
            "label_nc": None,
        },
    ]
]
# LDM CASES: 2D and 3D, with and without conditioning, with and without SPADE norm (8)
# Some parameters that are tweaked in test_difussion_model_unet are or not included in each case.

UNCOND_CASES_2D = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": (1, 1, 2),
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "spade_norm": True,
            "label_nc": 3,
        }
    ],
]
UNCOND_CASES_3D = [
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "resblock_updown": True,
            "spade_norm": True,
            "label_nc": 3,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
            "resblock_updown": True,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": (0, 0, 4),
            "norm_num_groups": 8,
        }
    ],
]

WRONG_DIFFUSION_CASE = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": (1, 1, 2),
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "spade_norm": True,
            "label_nc": None,
        }
    ],

]
COND_CASES_2D = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "resblock_updown": True,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "resblock_updown": True,
            "spade_norm": True,
            "label_nc": 3,
        }
    ],
]
COND_CASES_3D = [
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "resblock_updown": True
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "spade_norm": True,
            "label_nc": 3,

        }
    ],
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TestSPADEAutoEncoderKL(unittest.TestCase):
    @parameterized.expand(AUTOENCODER_CASES)
    def test_shape(self, input_param, input_shape, input_seg, expected_shape, expected_latent_shape):
        net = AutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device),
                                 torch.randn(input_seg).to(device))
            self.assertEqual(result[0].shape, expected_shape)
            self.assertEqual(result[1].shape, expected_latent_shape)

    @parameterized.expand(WRONG_AUTOENCODER_CASE)
    def test_wrong_spade_config(self, input_param):
        with self.assertRaises(ValueError):
            AutoencoderKL(
                **input_param
            )

class TestSPADEDiffusionModelUNet2D(unittest.TestCase):
    @parameterized.expand(UNCOND_CASES_2D)
    def test_shape_unconditioned_models(self, input_param):
        net = DiffusionModelUNet(**input_param)
        with eval_mode(net):
            if net.spade_norm:
                result = net.forward(torch.rand((1, 1, 16, 16)),
                                     torch.randint(0, 1000, (1,)).long(),
                                     seg = torch.rand((1, 3, 16, 16)))
            else:
                result = net.forward(torch.rand((1, 1, 16, 16)),
                                     torch.randint(0, 1000, (1,)).long())
            self.assertEqual(result.shape, (1, 1, 16, 16))

    @parameterized.expand(COND_CASES_2D)
    def test_shape_unconditioned_models(self, input_param):
        net = DiffusionModelUNet(**input_param)
        with eval_mode(net):
            if net.spade_norm:
                result = net.forward(torch.rand((1, 1, 16, 16)),
                                     torch.randint(0, 1000, (1,)).long(),
                                     seg = torch.rand((1, 3, 16, 16)),
                                     context=torch.rand((1, 1, 3)))
            else:
                result = net.forward(torch.rand((1, 1, 16, 16)),
                                     torch.randint(0, 1000, (1,)).long(),
                                     context=torch.rand((1, 1, 3)))
            self.assertEqual(result.shape, (1, 1, 16, 16))

    @parameterized.expand(WRONG_DIFFUSION_CASE)
    def test_wrong_spade_config(self, input_param):
        with self.assertRaises(ValueError):
            DiffusionModelUNet(
                **input_param
            )

class TestSPADEDiffusionModelUNet3D(unittest.TestCase):
    @parameterized.expand(UNCOND_CASES_3D)
    def test_shape_unconditioned_models(self, input_param):
        net = DiffusionModelUNet(**input_param)
        with eval_mode(net):
            if net.spade_norm:
                result = net.forward(torch.rand((1, 1, 16, 16, 16)),
                                     torch.randint(0, 1000, (1,)).long(),
                                     seg=torch.rand((1, 3, 16, 16, 16)))
            else:
                result = net.forward(torch.rand((1, 1, 16, 16, 16)),
                                     torch.randint(0, 1000, (1,)).long())
            self.assertEqual(result.shape, (1, 1, 16, 16, 16))

    @parameterized.expand(COND_CASES_3D)
    def test_shape_unconditioned_models(self, input_param):
        net = DiffusionModelUNet(**input_param)
        with eval_mode(net):
            if net.spade_norm:
                result = net.forward(torch.rand((1, 1, 16, 16, 16)),
                                     torch.randint(0, 1000, (1,)).long(),
                                     seg=torch.rand((1, 3, 16, 16, 16)),
                                     context=torch.rand((1, 1, 3)))
            else:
                result = net.forward(torch.rand((1, 1, 16, 16, 16)),
                                     torch.randint(0, 1000, (1,)).long(),
                                     context=torch.rand((1, 1, 3)))
            self.assertEqual(result.shape, (1, 1, 16, 16, 16))

if __name__ == "__main__":
    unittest.main()
