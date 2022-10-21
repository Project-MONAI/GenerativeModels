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

# from tests.utils import test_script_save
from generative.networks.nets import AutoencoderKL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TEST_CASE_0 = [
    {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "n_channels": 32,
        "latent_channels": 32,
        "ch_mult": [1, 1, 1],
        "num_res_blocks": 1,
        "resolution": (64, 64),
        "with_attention": False,
    },
    (1, 1, 128, 128),
    (1, 1, 128, 128),
]

CASES = [TEST_CASE_0]


class TestAutoEncoderKL(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = AutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result[0].shape, expected_shape)

    # def test_script(self):
    #     net = AutoencoderKL(
    #         spatial_dims= 2,
    #         in_channels= 1,
    #         out_channels= 1,
    #         n_channels= 32,
    #         z_channels= 32,
    #         embed_dim= 32,
    #         ch_mult= [1, 1, 1],
    #         num_res_blocks= 1,
    #         resolution= (64, 64),
    #         with_attention= False,
    #     )
    #     test_data = torch.randn(2, 1, 128, 128)
    #     test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
