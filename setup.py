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

from setuptools import find_packages, setup

setup(
    name="monai-generative",
    packages=find_packages(exclude=[]),
    version="0.2.1",
    description="Installer to help to use the prototypes from MONAI generative models in other projects.",
    install_requires=["monai-weekly==1.2.dev2304"],
)
