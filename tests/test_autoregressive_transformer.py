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

from generative.networks.nets import AutoregressiveTransformer


class TestAutoregressiveTransformer(unittest.TestCase):
    def test_shape_unconditioned_models(self):
        net = AutoregressiveTransformer(
            num_tokens=10, max_seq_len=16, attn_layers_dim=8, attn_layers_depth=2, attn_layers_heads=2
        )
        with eval_mode(net):
            net.forward(torch.randint(0, 10, (1, 16)))


if __name__ == "__main__":
    unittest.main()
