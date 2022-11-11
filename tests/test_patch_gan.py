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

import unittest

import torch
from monai.networks import eval_mode
from parameterized import parameterized

from generative.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator

TEST_2D_WITHOUT_INTERMEDIATES = [
    {
        "num_D": 2,
        "num_layers_D": 3,
        "spatial_dims": 2,
        "num_channels": 8,
        "in_channels": 3,
        "out_channels": 1,
        "kernel_size": 3,
        "activation": "LEAKYRELU",
        "norm": "instance",
        "bias": False,
        "dropout": 0.1,
        "minimum_size_im": 256,
    },
    False,
    torch.rand([1, 3, 256, 512]),
    [(1, 1, 32, 64), (1, 1, 4, 8)],
    [3, 6],
]
TEST_2D_WITH_INTERMEDIATES = [
    {
        "num_D": 2,
        "num_layers_D": 3,
        "spatial_dims": 2,
        "num_channels": 8,
        "in_channels": 3,
        "out_channels": 1,
        "kernel_size": 3,
        "activation": "LEAKYRELU",
        "norm": "instance",
        "bias": False,
        "dropout": 0.1,
        "minimum_size_im": 256,
    },
    True,
    torch.rand([1, 3, 256, 512]),
    [(1, 1, 32, 64), (1, 1, 4, 8)],
    [3, 6],
]
TEST_3D_WITHOUT_INTERMEDIATES = [
    {
        "num_D": 2,
        "num_layers_D": 3,
        "spatial_dims": 3,
        "num_channels": 8,
        "in_channels": 3,
        "out_channels": 1,
        "kernel_size": 3,
        "activation": "LEAKYRELU",
        "norm": "instance",
        "bias": False,
        "dropout": 0.1,
        "minimum_size_im": 256,
    },
    False,
    torch.rand([1, 3, 256, 512, 256]),
    [(1, 1, 32, 64, 32), (1, 1, 4, 8, 4)],
    [3, 6],
]
TEST_3D_WITH_INTERMEDIATES = [
    {
        "num_D": 2,
        "num_layers_D": 3,
        "spatial_dims": 3,
        "num_channels": 8,
        "in_channels": 3,
        "out_channels": 1,
        "kernel_size": 3,
        "activation": "LEAKYRELU",
        "norm": "instance",
        "bias": False,
        "dropout": 0.1,
        "minimum_size_im": 256,
    },
    True,
    torch.rand([1, 3, 256, 512, 256]),
    [(1, 1, 32, 64, 32), (1, 1, 4, 8, 4)],
    [3, 6],
]
TEST_TOO_SMALL_SIZE = [
    {
        "num_D": 2,
        "num_layers_D": 6,
        "spatial_dims": 2,
        "num_channels": 8,
        "in_channels": 3,
        "out_channels": 1,
        "kernel_size": 3,
        "activation": "LEAKYRELU",
        "norm": "instance",
        "bias": False,
        "dropout": 0.1,
        "minimum_size_im": 256,
    },
]

CASES = [
    TEST_2D_WITHOUT_INTERMEDIATES,
    TEST_2D_WITH_INTERMEDIATES,
    TEST_3D_WITH_INTERMEDIATES,
    TEST_3D_WITHOUT_INTERMEDIATES,
]


class TestPatchGAN(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(
        self,
        input_param,
        get_intermediate_features,
        input_data,
        expected_shape,
        features_lengths=None,
    ):
        net = MultiScalePatchDiscriminator(**input_param)
        with eval_mode(net):
            if not get_intermediate_features:
                result = net.forward(input_data, get_intermediate_features)
            else:
                result, features = net.forward(input_data, get_intermediate_features)
            for r_ind, r in enumerate(result):
                self.assertEqual(tuple(r.shape), expected_shape[r_ind])
            if get_intermediate_features:
                for o_d_ind, o_d in enumerate(features):
                    self.assertEqual(len(o_d), features_lengths[o_d_ind])

    def test_too_small_shape(self):
        with self.assertRaises(AssertionError):
            MultiScalePatchDiscriminator(**TEST_TOO_SMALL_SIZE[0])


if __name__ == "__main__":
    unittest.main()
