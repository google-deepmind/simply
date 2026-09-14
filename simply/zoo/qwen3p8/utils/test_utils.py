# Copyright 2026 The Simply Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The HuggingFace oracle, loaded once for every test that reads it.

`testdata/golden_tiny.npz` is a tiny random-weight Qwen3.8 run through the
*unmodified* release code (`transformers.models.qwen3_5`, transformers 5.16.1)
on CPU in float32; `testdata/gen_golden.py` regenerates it and its docstring
says how. Every test that reads it wants the same four things -- the arrays,
the Simply config the fixture's HuggingFace config maps to, a single-device
mesh, and the parameters in Simply's tree -- so they are built here once
instead of being transcribed per test.

Two things this module deliberately does not do:

  * it does not map parameter names. The npz stores HuggingFace names under
    `param/`, and the conversion runs through `utils/ckpt_format.py`, i.e.
    through the code under test, so a mapping bug cannot be baked into the
    fixture and then confirmed by it.
  * it does not restate the architecture. The config comes from
    `config_lib.config_from_hf` applied to the fixture's own
    `config_kwargs`, which is the same function the released deployment uses.
"""

from collections.abc import Mapping
import dataclasses
import json
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from simply import config_lib
from simply.utils import sharding as sharding_lib
from simply.zoo.qwen3p8 import config_lib as qwen3p8_config_lib
from simply.zoo.qwen3p8.utils import ckpt_format

# The oracle lives with the model tests that compare against it, one level up;
# this module reaches it through the `:golden_tiny` filegroup.
_TESTDATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'testdata')

_NPZ = 'golden_tiny.npz'
_CONFIG_JSON = 'golden_tiny_config.json'

# The fixture's prefill is 20 tokens, so a chunk size of 8 crosses two chunk
# boundaries and leaves a 4-token remainder. The released 32 would make the
# whole prefill one padded chunk and never exercise the carry.
CHUNK_SIZE = 8

# The one function this module needs from the package's checkpoint format;
# named here so the error message can name it too.
_CONVERT = 'convert_from_mapping(state_dict, config, dtype=...)'


@dataclasses.dataclass(frozen=True)
class Golden:
  """The fixture, and the Simply model state that mirrors it.

  Attributes:
    data: the `.npz` arrays -- HuggingFace parameters under `param/`,
      activations under `act/`, `act_decode/` and `act_decode2/`, decode
      states under `cache_prefill/`, `cache_decode/` and `cache_decode2/`,
      the token ids, `meta/config_json`, and `mrope/` + `mrope_released/`,
      which are not part of the model run at all (see `gen_golden.py`: the
      release's rotary embedding for three *different* mRoPE position rows,
      the only thing in the fixture that separates `mrope_section` splits).
    meta: `golden_tiny_config.json`: `config_kwargs` (what `config_from_hf`
      is fed), `hf_config` (what the release made of it) and `meta` (how it
      was run).
    config: the Simply config `config_from_hf` maps `config_kwargs` to, with
      the test-only overrides applied.
    mesh: a single-device replicated mesh -- the fixture is a CPU reference.
    params: the fixture's parameters in Simply's tree, replicated on `mesh`.
  """

  data: Any
  meta: Mapping[str, Any]
  config: qwen3p8_config_lib.Qwen38ExperimentConfig
  mesh: jax.sharding.Mesh
  params: Any


def load_golden(**config_overrides: Any) -> Golden:
  """Loads the fixture and converts its parameters into Simply's tree.

  Args:
    **config_overrides: fields to override on the config, applied last, e.g.
      a longer `seq_len` for a decode test.

  Returns:
    The loaded `Golden`.

  Raises:
    NotImplementedError: if `utils/ckpt_format.py` does not export the
      conversion this loader needs.
    ValueError: if the conversion does not account for every stored tensor.
  """
  data = np.load(os.path.join(_TESTDATA, _NPZ))
  with open(os.path.join(_TESTDATA, _CONFIG_JSON)) as f:
    meta = json.load(f)
  hf_kwargs = meta['config_kwargs']
  # The test-only deployment choices, overridable: a caller that wants the
  # released dtype must be able to ask for it (`config_from_hf` would
  # otherwise be given the same keyword twice).
  overrides = dict(
      seq_len=hf_kwargs['max_position_embeddings'],
      batch_size=meta['meta']['batch_size'],
      linear_attention_chunk_size=CHUNK_SIZE,
      activation_dtype_name='float32',
      use_remat=False,
      sharding_config=dataclasses.replace(
          config_lib.BaseSharding(),
          embed_partition=None,
          activation_partition=None,
          data_partition=None,
          logits_partition=None,
      ),
  )
  overrides.update(config_overrides)
  config = qwen3p8_config_lib.config_from_hf(hf_kwargs, **overrides)
  mesh = sharding_lib.create_mesh(
      mesh_shape={'replica': 1, 'data': 1, 'model': 1},
      axis_names=('replica', 'data', 'model'),
  )
  state_dict = {
      k[len('param/') :]: data[k] for k in data.files if k.startswith('param/')
  }
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  params = jax.tree.map(
      lambda x: jax.device_put(jnp.asarray(x), replicated),
      _convert(state_dict, config),
  )
  return Golden(data=data, meta=meta, config=config, mesh=mesh, params=params)


def _convert(state_dict: Mapping[str, np.ndarray], config: Any) -> Any:
  """The fixture's parameters, mapped by the code under test.

  Args:
    state_dict: the `param/` arrays, under their HuggingFace names.
    config: the Simply config those tensors belong to.

  Returns:
    The Simply parameter tree.

  Raises:
    NotImplementedError: if `ckpt_format` does not export the conversion.
    ValueError: if the tree does not have exactly one leaf per stored tensor,
      which is what this fixture's release maps to -- every dropped or fused
      family is a mapping change that the model tests must see, not absorb.
  """
  convert = getattr(ckpt_format, 'convert_from_mapping', None)
  if convert is None:
    raise NotImplementedError(
        f'`utils/ckpt_format.py` must export `{_CONVERT}`, returning the'
        ' Simply parameter tree (no `params` level) for a HuggingFace state'
        ' dict. The golden fixture stores HuggingFace names on purpose, so'
        ' this loader will not map them itself.'
    )
  params = convert(state_dict, config, dtype=np.float32)
  n_leaves = len(jax.tree.leaves(params))
  if n_leaves != len(state_dict):
    raise ValueError(
        f'{_CONVERT} produced {n_leaves} leaves from {len(state_dict)} stored'
        ' tensors; the fixture maps one-to-one, so something was dropped,'
        ' fused or invented.'
    )
  return params
