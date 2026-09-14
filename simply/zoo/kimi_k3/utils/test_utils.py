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
"""Fixtures shared by the Kimi K3 tests.

`load_golden` loads the HuggingFace oracle: `testdata/k3_golden_tiny.npz` was
produced by running the *unmodified* HF release on CPU in float32 over a tiny
config that exercises every K3 feature (`testdata/gen_k3_golden.py`
regenerates it). Every test that reads it wants the same four things -- the
arrays, the Simply config the fixture's HF config maps to, a single-device
mesh, and the converted parameters -- so they are built here once rather than
transcribed per test.

`tiny_config` is the other shared fixture: the released topology at 1/1000 of
the width, for the tests that need a real model but no oracle.
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
from simply.zoo.kimi_k3 import config_lib as k3_config_lib
from simply.zoo.kimi_k3.utils import hf_params

# The oracle lives with the model tests that compare against it, one level
# up; this module reaches it through the `:k3_golden_tiny` filegroup.
_TESTDATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'testdata')


@dataclasses.dataclass(frozen=True)
class Golden:
  """The fixture, and the Simply model state that mirrors it.

  Attributes:
    data: the `.npz` arrays: HF parameters under `param/`, and activations and
      cache states under `act/`, `act_decode/` and `cache_decode/`.
    meta: `k3_golden_tiny_config.json`, whose `config_kwargs` is the HF config
      the fixture was generated from and whose `meta` records how.
    config: the Simply config `config_from_hf` maps `config_kwargs` to.
    mesh: a single-device replicated mesh -- the fixture is a CPU reference.
    params: the fixture's parameters in Simply's tree, replicated on `mesh`.
  """

  data: Any
  meta: Mapping[str, Any]
  config: k3_config_lib.KimiK3ExperimentConfig
  mesh: jax.sharding.Mesh
  params: Any


def load_golden(**config_overrides: Any) -> Golden:
  """Loads the fixture and converts its parameters into Simply's tree.

  Args:
    **config_overrides: fields to override on the config, e.g. a shorter
      `seq_len`.

  Returns:
    The loaded `Golden`.
  """
  data = np.load(os.path.join(_TESTDATA, 'k3_golden_tiny.npz'))
  with open(os.path.join(_TESTDATA, 'k3_golden_tiny_config.json')) as f:
    meta = json.load(f)
  hf_kwargs = meta['config_kwargs']
  config = k3_config_lib.config_from_hf(
      hf_kwargs,
      seq_len=hf_kwargs['max_position_embeddings'],
      batch_size=2,
      kda_chunk_size=8,
      activation_dtype_name='float32',
      use_scan=False,
      use_remat=False,
      sharding_config=dataclasses.replace(
          config_lib.BaseSharding(),
          embed_partition=None,
          activation_partition=None,
          data_partition=None,
          logits_partition=None,
      ),
      **config_overrides,
  )
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
      hf_params.convert_from_mapping(state_dict, config, dtype=np.float32),
  )
  return Golden(data=data, meta=meta, config=config, mesh=mesh, params=params)


def tiny_config(
    n_layers: int = 13,
    attn_res_block_size: int = 6,
) -> k3_config_lib.KimiK3ExperimentConfig:
  """The released topology at 1/1000 of the width, on one CPU device.

  Narrower than the registered `kimi_k3_tiny_test` and free in its depth, so a
  test can pick a layer count that makes its own point. That is the trade this
  one makes: it is for structural tests (parameter trees, checkpoint
  transforms), and it is deliberately too small and too short to sample from --
  `model_lib_test._tiny_config` is the one for the `LMInterface` path.

  Args:
    n_layers: decoder layers; the 3 KDA : 1 MLA schedule is derived from it.
    attn_res_block_size: layers per Block Attention Residual snapshot.

  Returns:
    The config.
  """
  return k3_config_lib.KimiK3ExperimentConfig(
      vocab_size=32,
      model_dim=16,
      n_layers=n_layers,
      attn_res_block_size=attn_res_block_size,
      n_heads=2,
      per_head_dim=8,
      q_lora_rank=8,
      kv_lora_rank=8,
      qk_nope_head_dim=8,
      qk_rope_head_dim=4,
      v_head_dim=8,
      kda_num_heads=2,
      kda_head_dim=8,
      kda_gate_lora_rank=8,
      kda_chunk_size=4,
      num_experts=4,
      num_experts_per_token=2,
      routed_expert_latent_dim=8,
      moe_intermediate_size=8,
      num_shared_experts=1,
      ffn_expand_dim=16,
      batch_size=2,
      seq_len=16,
      activation_dtype_name='float32',
      sharding_config=config_lib.gspmd_sharding(),
      mesh_shape=None,
      use_remat=False,
  )
