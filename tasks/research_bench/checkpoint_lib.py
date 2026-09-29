# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""checkpoint formats for the model-PORTING tasks (STUBS).

Registers `FalconH1Format` and `RecurrentGemmaFormat` on the SHARED core
`CheckpointFormatRegistry` so a config with `init_ckpt_format='FalconH1Format'`
(resp. `'RecurrentGemmaFormat'`) resolves and the package imports. Both are
STUBS that raise `NotImplementedError`: implementing the checkpoint transform
(mapping the provided checkpoint's stored tensors onto the model's parameter
tree) is part of each porting task. `main.py` imports this
module so the registrations fire. NO core checkpoint_lib.py change is required.
"""

import dataclasses

from simply.utils import checkpoint_lib as _core

CheckpointFormat = _core.CheckpointFormat
CheckpointFormatRegistry = _core.CheckpointFormatRegistry
PyTree = _core.PyTree


@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class FalconH1Format(CheckpointFormat):
  """Falcon-H1 checkpoint format (STUB -- implement me).

  Kept REGISTERED on the shared core `CheckpointFormatRegistry` so a config with
  `init_ckpt_format='FalconH1Format'` resolves and the package imports.
  Implement `transforms` to map the provided checkpoint's stored tensors onto
  your FalconH1LM parameter tree (inspect the stored tensor names/shapes; see
  the task spec). No core checkpoint_lib.py change is required.
  """

  def transforms(
      self, stored_state: PyTree, target_abstract_state: PyTree = None
  ) -> PyTree:
    del stored_state, target_abstract_state
    raise NotImplementedError(
        'FalconH1Format is a STUB. Implement transforms() to map the '
        'published Falcon-H1-0.5B-Base safetensor tensors onto the FalconH1LM '
        'parameter tree (see the task spec / config_lib.port_falcon_h1_0p5b).'
    )


@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class RecurrentGemmaFormat(CheckpointFormat):
  """RecurrentGemma-2B checkpoint format (STUB -- implement me).

  Kept REGISTERED on the shared core `CheckpointFormatRegistry` so a config with
  `init_ckpt_format='RecurrentGemmaFormat'` resolves and the package imports.
  Implement `transforms` to map the provided checkpoint's stored tensors onto
  your RecurrentGemmaLM parameter tree (inspect the stored tensor names/shapes;
  see the task spec). No core checkpoint_lib.py change is required.
  """

  def transforms(
      self, stored_state: PyTree, target_abstract_state: PyTree = None
  ) -> PyTree:
    del stored_state, target_abstract_state
    raise NotImplementedError(
        'RecurrentGemmaFormat is a STUB. Implement transforms() to map the '
        'published RecurrentGemma-2B safetensor tensors onto the '
        'RecurrentGemmaLM parameter tree (see the task spec / '
        'config_lib.port_recurrentgemma_2b).'
    )
