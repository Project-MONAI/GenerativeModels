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

from generative.networks.nets.patchgan_discriminator import MultiScaleDiscriminator

TEST_CASE_0 = [
    {
        "num_D": 2,
        "n_layers_D": 3,
        "spatial_dims": 2,
        "n_channels": 8,
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
    False,
]

TEST_CASE_1 = [
    {
        "num_D": 2,
        "n_layers_D": 3,
        "spatial_dims": 2,
        "n_channels": 8,
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
    False,
]

TEST_CASE_2 = [
    {
        "num_D": 2,
        "n_layers_D": 6,
        "spatial_dims": 2,
        "n_channels": 8,
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
    True,
]

CASES = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2]


class TestPatchGAN(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(
        self,
        input_param,
        get_intermediate_features,
        input_data,
        expected_shape,
        features_lengths=None,
        error_raised=False,
    ):

        if error_raised:
            with self.assertRaises(AssertionError):
                MultiScaleDiscriminator(**input_param)
        else:
            net = MultiScaleDiscriminator(**input_param)
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


if __name__ == "__main__":
    unittest.main()
