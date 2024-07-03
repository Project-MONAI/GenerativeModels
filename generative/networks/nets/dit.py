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
#
# =========================================================================
# Adapted from https://github.com/huggingface/diffusers
# which has the following license:
# https://github.com/huggingface/diffusers/blob/main/LICENSE
#
# Copyright 2022 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import torch
import torch.nn as nn
import monai

class TransformerBlock(nn.Module):
    def __init__(self, ...):
        super().__init__()
        

    def forward(self, x):
        
        return x

class DiffusionBlock(nn.Module):
    def __init__(self, ...):
        super().__init__()
        

    def forward(self, x, t):
        
        return x

class DiTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = TransformerBlock(...)
        self.diffusion = DiffusionBlock(...)

    def forward(self, x, timesteps):
        x = self.transformer(x)
        x = self.diffusion(x, timesteps)
        return x

