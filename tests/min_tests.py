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

import glob
import os
import sys
import unittest


def run_testsuit():
    """
    Load test cases by excluding those need external dependencies.
    The loaded cases should work with "requirements-min.txt"::

        # in the monai repo folder:
        pip install -r requirements-min.txt
        QUICKTEST=true python -m tests.min_tests

    :return: a test suite
    """
    exclude_cases = [  # these cases use external dependencies
        "test_autoencoderkl",
        "test_diffusion_inferer",
        "test_integration_workflows_adversarial",
        "test_latent_diffusion_inferer",
        "test_perceptual_loss",
        "test_transformer",
    ]
    assert sorted(exclude_cases) == sorted(set(exclude_cases)), f"Duplicated items in {exclude_cases}"

    files = glob.glob(os.path.join(os.path.dirname(__file__), "test_*.py"))

    cases = []
    for case in files:
        test_module = os.path.basename(case)[:-3]
        if test_module in exclude_cases:
            exclude_cases.remove(test_module)
            print(f"skipping tests.{test_module}.")
        else:
            cases.append(f"tests.{test_module}")
    assert not exclude_cases, f"items in exclude_cases not used: {exclude_cases}."
    test_suite = unittest.TestLoader().loadTestsFromNames(cases)
    return test_suite


if __name__ == "__main__":
    # testing import submodules
    from monai.utils.module import load_submodules

    _, err_mod = load_submodules(sys.modules["monai"], True)
    assert not err_mod, f"err_mod={err_mod} not empty"

    # testing all modules
    test_runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = test_runner.run(run_testsuit())
    sys.exit(int(not result.wasSuccessful()))
