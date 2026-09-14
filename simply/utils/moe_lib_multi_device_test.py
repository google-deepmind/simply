# Copyright 2024 The Simply Authors
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
"""Multi-device tests for the pipelined MoE in `moe_lib`.

`model_lib_test.test_pipelined_moe_feed_forward_equivalence` runs every
`pipelined_*` case at `mesh_shape=(1, 1, 1, 1)`, where `moe_lib` takes its
`jax.lax.axis_size(axis_name) == 1` short circuit and never communicates at
all. These tests run the same routine on 4 forced CPU devices with an expert
axis of size 4, so the ragged-all-to-all / all-gather pipeline, the local
permutations around it and the dropless fallback are actually exercised.
"""

import functools
import os

# Must be set BEFORE `import jax`. JAX initializes its backend on first use
# and the device count cannot be changed afterwards.
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=4')

# pylint: disable=g-import-not-at-top
# Imports below jax-affecting env vars are intentional; do not reorder.
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from simply.utils import moe_lib

# pylint: enable=g-import-not-at-top

_AXIS = 'expert'
_NUM_DEVICES = 4
_NUM_EXPERTS = 8
_EXPERTS_PER_TOK = 2
_TOKENS_PER_DEVICE = 8
_LANES = 128  # `moe_lib` consumes tokens in the [tokens, dim / 128, 128] layout
_MODEL_DIM = 2 * _LANES
_HIDDEN_DIM = 256
_SPLITS = 2


def _ragged_all_to_all(
    operand,
    output,
    input_offsets,
    send_sizes,
    output_offsets,
    recv_sizes,
    *,
    axis_name
):
  """Pure-XLA stand-in for `jax.lax.ragged_all_to_all`.

  XLA:CPU has no `ragged-all-to-all` thunk ("HLO opcode `ragged-all-to-all` is
  not supported by XLA:CPU ThunkEmitter"), so the production collective cannot
  run here. `PipelinedMoEConfig.ra2a` is pluggable precisely so a portable
  implementation can be swapped in; everything else in the pipeline is the
  production code path.

  Args:
    operand: rows to send, `[send_rows, ...]`.
    output: receive buffer, `[recv_rows, ...]`; only its shape is used.
    input_offsets: `input_offsets[j]` is where this device's chunk for device j
      starts in `operand`.
    send_sizes: `send_sizes[j]` rows go to device j.
    output_offsets: `output_offsets[j]` is where this device's chunk lands in
      device j's `output`.
    recv_sizes: unused; implied by the all-gathered `send_sizes`.
    axis_name: the manual mesh axis to communicate over.

  Returns:
    This device's filled receive buffer.
  """
  del recv_sizes
  gather = functools.partial(
      jax.lax.all_gather, axis_name=axis_name, axis=0, tiled=False
  )
  operands, in_offsets = gather(operand), gather(input_offsets)
  sizes, out_offsets = gather(send_sizes), gather(output_offsets)
  self_idx, rows = jax.lax.axis_index(axis_name), jnp.arange(output.shape[0])
  out = jnp.zeros_like(output)
  for src_idx in range(jax.lax.axis_size(axis_name)):
    start, size = out_offsets[src_idx, self_idx], sizes[src_idx, self_idx]
    src_rows = jnp.clip(
        in_offsets[src_idx, self_idx] + rows - start, 0, operands.shape[1] - 1
    )
    from_src = (rows >= start) & (rows < start + size)
    out = jnp.where(
        jnp.expand_dims(from_src, range(1, out.ndim)),
        jnp.take(operands[src_idx], src_rows, axis=0),
        out,
    )
  return out


def _compute_block(
    tokens: jax.Array, local_group_sizes: jax.Array, *extra_args: jax.Array
) -> jax.Array:
  """Toy grouped FFN, in the 3D -> 2D -> 3D shape contract `moe_lib` expects."""
  ffn0_w, ffn1_w = extra_args
  shape = tokens.shape
  tokens = tokens.reshape(shape[0], -1)
  starts = jnp.cumsum(local_group_sizes) - local_group_sizes
  group = jnp.sum(jnp.arange(shape[0])[:, None] >= starts[None, :], axis=-1) - 1
  out = jnp.zeros_like(tokens)
  for expert in range(ffn0_w.shape[0]):
    ffn = jax.nn.relu(tokens @ ffn0_w[expert]) @ ffn1_w[expert]
    out = jnp.where((group == expert)[:, None], ffn, out)
  return out.reshape(shape)


def _make_inputs(seed=0):
  """Balanced routing keeps every expert's receive buffer well below capacity."""
  rng = np.random.default_rng(seed)
  num_tokens = _NUM_DEVICES * _TOKENS_PER_DEVICE
  normal = lambda *shape: np.asarray(rng.normal(size=shape) / 10, np.float32)
  x = normal(num_tokens, _MODEL_DIM // _LANES, _LANES)
  expert_idx = np.stack([
      rng.permutation(_NUM_EXPERTS)[:_EXPERTS_PER_TOK]
      for _ in range(num_tokens)
  ]).astype(np.int32)
  scales = np.asarray(rng.uniform(0.2, 0.8, size=expert_idx.shape), np.float32)
  ffn0_w = normal(_NUM_EXPERTS, _MODEL_DIM, _HIDDEN_DIM)
  ffn1_w = normal(_NUM_EXPERTS, _HIDDEN_DIM, _MODEL_DIM)
  return x, expert_idx, scales, ffn0_w, ffn1_w


def _reference(x, expert_idx, scales, ffn0_w, ffn1_w):
  """Dense MoE: run every token through each of its experts, unsharded."""
  tokens = x.reshape(x.shape[0], -1)
  out = np.zeros_like(tokens)
  for slot in range(_EXPERTS_PER_TOK):
    experts = expert_idx[:, slot]
    hidden = np.maximum(np.einsum('td,tdh->th', tokens, ffn0_w[experts]), 0)
    out += scales[:, slot, None] * np.einsum(
        'th,thd->td', hidden, ffn1_w[experts]
    )
  return out.reshape(x.shape)


class PipelinedMoeMultiDeviceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.assertEqual(jax.device_count(), _NUM_DEVICES)

  @parameterized.named_parameters(
      # `dropless_fallback` builds both pipelines into one `lax.cond`, so this
      # case covers the ra2a pipeline and the fallback's `lax.cond` at once.
      dict(
          testcase_name='_ra2a_dropless',
          ep_method='ra2a',
          dropless_fallback=True,
          use_pipelined_ra2a_barriers=True,
      ),
      dict(
          testcase_name='_ra2a',
          ep_method='ra2a',
          dropless_fallback=False,
          use_pipelined_ra2a_barriers=False,
      ),
      dict(
          testcase_name='_ag',
          ep_method='ag',
          dropless_fallback=False,
          use_pipelined_ra2a_barriers=True,
      ),
  )
  def test_matches_dense_reference(
      self, ep_method, dropless_fallback, use_pipelined_ra2a_barriers
  ):
    mesh = jax.make_mesh((_NUM_DEVICES,), (_AXIS,))
    spec = jax.sharding.PartitionSpec(_AXIS)
    config = moe_lib.PipelinedMoEConfig(
        ra2a=_ragged_all_to_all,
        ep_method=ep_method,
        dropless_fallback=dropless_fallback,
        use_pipelined_ra2a_barriers=use_pipelined_ra2a_barriers,
        # The SparseCore default would round this toy's buffers up to 1024 rows.
        pad_buffers_to_multiple=64,
    )

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(spec,) * 5,
        out_specs=(spec, jax.sharding.PartitionSpec()),
        check_vma=False,
    )
    def moe(x, expert_idx, scales, ffn0_w, ffn1_w):
      return moe_lib.run_moe_pipelined_shard_map(
          expert_idx,
          x,
          ffn0_w,
          ffn1_w,
          scales=scales,
          compute_block=_compute_block,
          axis_name=_AXIS,
          experts_per_tok=_EXPERTS_PER_TOK,
          num_experts=_NUM_EXPERTS,
          splits=_SPLITS,
          config=config,
      )

    x, expert_idx, scales, ffn0_w, ffn1_w = _make_inputs()
    args = jax.device_put(
        (x, expert_idx.reshape(-1), scales.reshape(-1), ffn0_w, ffn1_w),
        jax.sharding.NamedSharding(mesh, spec),
    )
    out, metrics = jax.jit(moe)(*args)

    # With a size-1 expert axis `moe_lib` short-circuits the collective and
    # never records this metric; a nonzero value proves tokens crossed devices.
    self.assertGreater(int(metrics['total_recv_size']), 0)
    self.assertFalse(bool(metrics['exceeds_buffer']))
    np.testing.assert_allclose(
        np.asarray(out),
        _reference(x, expert_idx, scales, ffn0_w, ffn1_w),
        rtol=1e-4,
        atol=1e-5,
    )


if __name__ == '__main__':
  absltest.main()
