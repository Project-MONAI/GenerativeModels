- [Introduction](#introduction)
- [The contribution process](#the-contribution-process)
  * [Preparing pull requests](#preparing-pull-requests)
  * [Submitting pull requests](#submitting-pull-requests)

## Introduction


Welcome to Project MONAI Generative Models! We're excited you're here and want to contribute. This documentation is intended for individuals and institutions interested in contributing to MONAI Generative Models. MONAI Generative Models is an open-source project and, as such, its success relies on its community of contributors willing to keep improving it. Your contribution will be a valued addition to the code base; we simply ask that you read this page and understand our contribution process, whether you are a seasoned open-source contributor or whether you are a first-time contributor.

### Communicate with us

We are happy to talk with you about your needs for MONAI Generative Models and your ideas for contributing to the project. One way to do this is to create an issue discussing your thoughts. It might be that a very similar feature is under development or already exists, so an issue is a great starting point. If you are looking for an issue to resolve that will help Project MONAI Generative Models, see the [*good first issue*](https://github.com/Project-MONAI/GenerativeModels/labels/good%20first%20issue) and [*Contribution wanted*](https://github.com/Project-MONAI/GenerativeModels/labels/Contribution%20wanted) labels.

## The contribution process

_Pull request early_

We encourage you to create pull requests early. It helps us track the contributions under development, whether they are ready to be merged or not. [Create a draft pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request) until it is ready for formal review.


### Preparing pull requests
To ensure the code quality, MONAI Generative Models relies on several linting tools ([flake8 and its plugins](https://gitlab.com/pycqa/flake8), [black](https://github.com/psf/black), [isort](https://github.com/timothycrosley/isort)),
static type analysis tools ([mypy](https://github.com/python/mypy), [pytype](https://github.com/google/pytype)), as well as a set of unit/integration tests.

This section highlights all the necessary preparation steps required before sending a pull request.
To collaborate efficiently, please read through this section and follow them.

* [Checking the coding style](#checking-the-coding-style)
* [Licensing information](#licensing-information)
* [Exporting modules](#exporting-modules)
* [Unit testing](#unit-testing)

#### Checking the coding style
>In progress.  Please wait for more instructions to follow
Before submitting Pull Request make sure basic CI checks are passed.

Install `pre-commit` if you have not already.
```
python -m pip install pre-commit
```
Run `pre-commit` check from MONAI Generative Models working directory.
```
cd GenerativeModels
python -m pre_commit run --all-files
```

#### Licensing information
All source code files should start with this paragraph:

```
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

```

#### Exporting modules

If you intend for any variables/functions/classes to be available outside of the file with the edited functionality, then:

- Create or append to the `__all__` variable (in the file in which functionality has been added), and
- Add to the `__init__.py` file.

#### Unit testing
>In progress.  Please wait for more instructions to follow

MONAI Generative Models tests are located under `tests/`.

##### Set environment
To use the tests already available at MONAI core, first we clone it:
```shell
git clone https://github.com/Project-MONAI/MONAI --branch main
```

Then we add it to PYTHONPATH
```shell
export PYTHONPATH="${PYTHONPATH}:./MONAI/"
```

##### Executing tests
To run tests, use the following command:

```shell script
 python -m unittest discover tests
```


### Submitting pull requests
All code changes to the main branch must be done via [pull requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests).
1. Create a new ticket or take a known ticket from [the issue list][monai issue list].
1. Check if there's already a branch dedicated to the task.
1. If the task has not been taken, [create a new branch in your fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)
of the codebase named `[ticket_id]-[task_name]`.
For example, branch name `19-ci-pipeline-setup` corresponds to [issue #19](https://github.com/Project-MONAI/MONAI/issues/19).
Ideally, the new branch should be based on the latest `main` branch.
1. Make changes to the branch ([use detailed commit messages if possible](https://chris.beams.io/posts/git-commit/)).
1. Make sure that new tests cover the changes and the changed codebase [passes all tests locally](#unit-testing).
1. [Create a new pull request](https://help.github.com/en/desktop/contributing-to-projects/creating-a-pull-request) from the task branch to the main branch, with detailed descriptions of the purpose of this pull request.
1. Wait for reviews; if there are reviews, make point-to-point responses, make further code changes if needed.
1. If there are conflicts between the pull request branch and the main branch, pull the changes from the main and resolve the conflicts locally.
1. Reviewer and contributor may have discussions back and forth until all comments addressed.
1. Wait for the pull request to be merged.
