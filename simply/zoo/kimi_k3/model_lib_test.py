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
"""Tests for the assembled Kimi K3 model, in four parts.

`HfEquivalenceTest` is the correctness claim: prefill logits, every layer's
residual stream, the Block Attention Residual snapshots and the caches, against
a fixture produced by the *unmodified* HuggingFace release (see
`utils/test_utils.py`, and `testdata/gen_k3_golden.py` to regenerate it).

`DecodeTest` covers the decode protocol -- the cached step against the same
fixture, and the cached path against a stateless re-run of the whole prefix,
which is a tighter bound than the HF comparison and catches cache bugs a single
step is too short to expose.

`LMInterfaceTest` runs K3 through `model_lib.LMInterface`, i.e. through what
`decode_eval` runs: the `prefill_position` protocol and the
`pad_block_decode_state` registrations, neither of which is called directly.
"""

import dataclasses
from typing import Any, cast

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from simply import config_lib
from simply import model_lib
from simply.utils import common
from simply.utils import evaluation_lib
from simply.utils import lm_format as lm_format_lib
from simply.utils import sampling_lib
from simply.utils import sharding as sharding_lib
from simply.utils import tokenization
from simply.zoo.kimi_k3 import config_lib as k3_config_lib
from simply.zoo.kimi_k3 import model_lib as k3_model_lib
from simply.zoo.kimi_k3.utils import attn_res as attn_res_lib
from simply.zoo.kimi_k3.utils import kda as kda_lib
from simply.zoo.kimi_k3.utils import mla as mla_lib
from simply.zoo.kimi_k3.utils import test_utils


def setUpModule():
  # `HfEquivalenceTest` and `DecodeTest` compare against an f32 CPU reference
  # and need f32 matmuls; pinning it for the module keeps the value independent
  # of which class a shard happens to run first.
  jax.config.update('jax_default_matmul_precision', 'float32')


# --- The HuggingFace oracle ---------------------------------------------------
class HfEquivalenceTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    golden = test_utils.load_golden()
    cls.data, cls.config = golden.data, golden.config
    cls.mesh, cls.params = golden.mesh, golden.params

  def setUp(self):
    super().setUp()
    self.model = k3_model_lib.KimiK3LM(
        config=self.config, sharding_config=self.config.sharding_config
    )

  def _prefill_inputs(self):
    input_ids = np.asarray(self.data['prefill_input_ids'], np.int32)
    batch, seq_len = input_ids.shape
    segment_ids = np.ones((batch, seq_len), np.int32)
    segment_positions = np.broadcast_to(
        np.arange(seq_len, dtype=np.int32), (batch, seq_len)
    )
    replicated = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec()
    )
    return tuple(
        jax.device_put(x, replicated)
        for x in (input_ids, segment_ids, segment_positions)
    )

  def test_param_tree_matches_model_init(self):
    with jax.set_mesh(self.mesh):
      init_params = self.model.init(jax.random.PRNGKey(0))
    shapes = lambda tree: jax.tree.map(np.shape, common.get_raw_arrays(tree))
    init_shapes, loaded_shapes = shapes(init_params), shapes(self.params)
    self.assertEqual(
        jax.tree.structure(init_shapes), jax.tree.structure(loaded_shapes)
    )
    jax.tree.map(self.assertEqual, init_shapes, loaded_shapes)

  def test_prefill_logits(self):
    input_ids, segment_ids, segment_positions = self._prefill_inputs()
    with jax.set_mesh(self.mesh):
      logits, _ = self.model.apply(
          self.params,
          input_ids,
          segment_ids=segment_ids,
          segment_positions=segment_positions,
      )
    np.testing.assert_allclose(
        np.asarray(logits), self.data['act/logits'], atol=2e-4, rtol=1e-3
    )

  def test_layerwise_activations(self):
    """Walks the layer stack, localizing any divergence to a sublayer."""
    input_ids, segment_ids, segment_positions = self._prefill_inputs()
    with jax.set_mesh(self.mesh):
      x = self.model.embed_linear.embed(self.params['embed_linear'], input_ids)
      np.testing.assert_allclose(
          np.asarray(x), self.data['act/embed'], atol=1e-6
      )
      state = attn_res_lib.init_attn_res_state(
          batch_size=x.shape[0],
          seq_len=x.shape[1],
          model_dim=x.shape[2],
          num_slots=self.model.num_slots,
          dtype=x.dtype,
      )
      for i, block in enumerate(self.model.blocks):
        x, extra = block.apply(
            self.params[f'block_{i}'],
            x,
            attn_res_state=state,
            segment_ids=segment_ids,
            segment_positions=segment_positions,
        )
        state = extra['attn_res_state']
        np.testing.assert_allclose(
            np.asarray(x),
            self.data[f'act/layer{i}/prefix_sum'],
            atol=2e-4,
            rtol=1e-3,
            err_msg=f'prefix_sum mismatch after layer {i}',
        )
        golden_snapshots = self.data[f'act/layer{i}/block_residual']
        num_valid = golden_snapshots.shape[1]
        self.assertEqual(int(state.num_valid), num_valid)
        snapshots = np.asarray(state.snapshots)[:, :, :num_valid, :]
        snapshots = snapshots.reshape(-1, num_valid, snapshots.shape[-1])
        np.testing.assert_allclose(
            snapshots,
            golden_snapshots,
            atol=2e-4,
            rtol=1e-3,
            err_msg=f'block_residual mismatch after layer {i}',
        )


# --- The decode protocol ------------------------------------------------------
_PREFILL_LEN = 16
# Room for the fixture's decode step plus the greedy-generation tests.
_DECODE_MAX_SEQ_LEN = 24


def _relative(a, b) -> tuple[float, float]:
  """`(max|a - b|, max|a - b| / max|b|)`; the relative part is 0 for `b == 0`."""
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  delta = float(np.max(np.abs(a - b)))
  scale = float(np.max(np.abs(b)))
  return delta, delta / scale if scale else 0.0


# Tolerances are RELATIVE to the magnitude of the reference tensor
# (`max|delta| / max|expected|`), with a small absolute floor for tensors that
# are exactly zero. An absolute bound would measure the residual stream's
# scale, which grows with depth, rather than the error: the fp32 deltas grow
# from 2e-6 at layer 0 to 2e-4 at layer 7 while the relative error stays at
# 1e-6 .. 3e-5.
_ATOL_FLOOR = 2e-5
# vs the HuggingFace fixture (both sides fp32, different summation order).
# Deliberately 10x above the observed 6e-5: XLA:CPU picks its reduction order
# from the machine it lands on, so the same binary on the same sources measures
# anywhere in 3e-5 .. 7e-5 at layer 7. Anything structural is orders away.
_HF_REL = 5e-4
# The model against itself: a cached pass vs a stateless one (observed 7e-6 ..
# 2e-5, same machine-dependent spread). Re-running the *same* chunking is
# bit-exact and asserted as such.
_SELF_REL = 1e-4


class DecodeTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    golden = test_utils.load_golden()
    cls.data, cls.config = golden.data, golden.config
    cls.mesh, cls.params = golden.mesh, golden.params
    cls.input_ids = np.asarray(cls.data['input_ids'], np.int32)
    # Prefilling the fixture is the bulk of this test's runtime and is
    # deterministic, so the tests share their prefills.
    cls.memo = {}

  def setUp(self):
    super().setUp()
    self.model = k3_model_lib.KimiK3LM(
        config=self.config, sharding_config=self.config.sharding_config
    )
    bf16_config = dataclasses.replace(
        self.config, activation_dtype_name='bfloat16'
    )
    self.bf16_model = k3_model_lib.KimiK3LM(
        config=bf16_config, sharding_config=bf16_config.sharding_config
    )

  # --- Helpers ----------------------------------------------------------------

  def _assert_close(
      self, actual, expected, name: str, rel: float, atol: float = _ATOL_FLOOR
  ) -> float:
    """Asserts `max|delta| <= atol + rel * max|expected|`, logging both."""
    delta, rel_delta = _relative(actual, expected)
    logging.info(
        '%-46s max|delta| = %.2e  rel = %.1e  (rel tol %.0e)',
        name,
        delta,
        rel_delta,
        rel,
    )
    scale = delta / rel_delta if rel_delta else 0.0
    self.assertLessEqual(
        delta,
        atol + rel * scale,
        msg=f'{name}: max|delta| = {delta:.2e}, rel = {rel_delta:.2e}',
    )
    return rel_delta

  def _assert_states_close(
      self, actual, expected, name: str, rel: float, atol: float = _ATOL_FLOOR
  ):
    """Per-leaf comparison of two decode states, reported leaf by leaf."""
    worst = 0.0
    for (path, a), b in zip(
        jax.tree.leaves_with_path(actual),
        jax.tree.leaves(expected),
        strict=True,
    ):
      delta, rel_delta = _relative(a, b)
      leaf = f'{name}{jax.tree_util.keystr(path)}'
      logging.info(
          '%-46s max|delta| = %.2e  rel = %.1e', leaf, delta, rel_delta
      )
      scale = delta / rel_delta if rel_delta else 0.0
      self.assertLessEqual(
          delta,
          atol + rel * scale,
          msg=f'{leaf}: max|delta| = {delta:.2e}, rel = {rel_delta:.2e}',
      )
      worst = max(worst, rel_delta)
    return worst

  def _tokens(self, start: int, end: int):
    """`input_ids[:, start:end]` with matching segment ids and positions."""
    ids = jnp.asarray(self.input_ids[:, start:end])
    batch = ids.shape[0]
    segment_ids = jnp.ones(ids.shape, jnp.int32)
    positions = jnp.broadcast_to(
        jnp.arange(start, end, dtype=jnp.int32), (batch, end - start)
    )
    return ids, segment_ids, positions

  def _apply(self, ids, segment_ids, positions, decode_state=None, model=None):
    return (model or self.model).apply(
        self.params,
        ids,
        segment_ids=segment_ids,
        segment_positions=positions,
        decode_state=decode_state,
    )

  def _prefill(
      self, length: int = _PREFILL_LEN, chunk: int | None = None, model=None
  ):
    """Runs `length` tokens into a fresh decode state, `chunk` tokens at a time."""
    model = model or self.model
    key = ('prefill', length, chunk, model.config.activation_dtype_name)
    if key not in self.memo:
      state = model.init_decode_state(_DECODE_MAX_SEQ_LEN)
      for start in range(0, length, chunk or length):
        end = min(start + (chunk or length), length)
        _, extra = self._apply(
            *self._tokens(start, end), decode_state=state, model=model
        )
        state = extra['decode_state']
      self.memo[key] = state
    return self.memo[key]

  def _walk_decode_step(self, model, state, perturbation: float = 0.0):
    """Yields `(layer_index, prefix_sum, attn_res_state)` for the decode step.

    Args:
      model: the model to step.
      state: its decode state, already holding the prefill.
      perturbation: scales a deterministic relative noise added to the
        embedding, which measures how much the stack amplifies an input of that
        size.

    Yields:
      One triple per layer, in order.
    """
    ids, segment_ids, positions = self._tokens(_PREFILL_LEN, _PREFILL_LEN + 1)
    x = model.embed_linear.embed(self.params['embed_linear'], ids)
    if perturbation:
      noise = jax.random.uniform(
          jax.random.PRNGKey(0), x.shape, minval=-1.0, maxval=1.0
      )
      x = x * (1.0 + perturbation * noise).astype(x.dtype)
    yield -1, x, None
    attn_res_state = attn_res_lib.init_attn_res_state(
        batch_size=x.shape[0],
        seq_len=x.shape[1],
        model_dim=x.shape[2],
        num_slots=model.num_slots,
        dtype=x.dtype,
    )
    for i, block in enumerate(model.blocks):
      x, extra = block.apply(
          self.params[f'block_{i}'],
          x,
          attn_res_state=attn_res_state,
          segment_ids=segment_ids,
          segment_positions=positions,
          decode_state=state[f'block_{i}'],
      )
      attn_res_state = extra['attn_res_state']
      yield i, x, attn_res_state

  # --- The HuggingFace fixture ------------------------------------------------

  def test_decode_step_logits_match_hf(self):
    with jax.set_mesh(self.mesh):
      state = self._prefill()
      logits, _ = self._apply(
          *self._tokens(_PREFILL_LEN, _PREFILL_LEN + 1), decode_state=state
      )
    self._assert_close(
        logits, self.data['act_decode/logits'], 'decode logits vs HF', _HF_REL
    )

  def test_decode_step_layerwise_matches_hf(self):
    """Walks the layer stack of the decode step, localizing any divergence."""
    with jax.set_mesh(self.mesh):
      state = self._prefill()
      for i, x, attn_res_state in self._walk_decode_step(self.model, state):
        if i < 0:
          self._assert_close(x, self.data['act_decode/embed'], 'embed', _HF_REL)
          continue
        self._assert_close(
            x,
            self.data[f'act_decode/layer{i}/prefix_sum'],
            f'layer{i} prefix_sum',
            _HF_REL,
        )
        golden = self.data[f'act_decode/layer{i}/block_residual']
        num_valid = golden.shape[1]
        self.assertEqual(int(attn_res_state.num_valid), num_valid)
        # A decode step's snapshots are built from that step's own residual
        # stream, so the buffer is per-step: [B, 1, slots, D].
        snapshots = np.asarray(attn_res_state.snapshots)[:, :, :num_valid, :]
        self.assertEqual(snapshots.shape[1], 1)
        self._assert_close(
            snapshots.reshape(-1, num_valid, snapshots.shape[-1]),
            golden,
            f'layer{i} block_residual',
            _HF_REL,
        )

  def test_bfloat16_decode_stays_within_the_models_conditioning(self):
    """The production dtype, bounded by what this fixture can actually say.

    A bf16 cached decode does NOT reproduce a bf16 stateless prefill on this
    fixture (measured: 0.5 relative on the logits), and that is a property of
    the *model*, not of the cache: with random weights the AttnRes softmax sits
    on near-tied scores and the MoE top-k on near-tied routes, so an input
    perturbation of bf16 size (2^-8) is amplified to the same O(0.1..1) by the
    fp32 model alone -- which is what this test asserts, and what makes the
    number interpretable. The exactness contract lives in the fp32 tests; the
    real-weight bf16 number has to come from the released checkpoint.
    """
    with jax.set_mesh(self.mesh):
      state = self._prefill(model=self.bf16_model)
      for i, x, _ in self._walk_decode_step(self.bf16_model, state):
        if i < 0:
          continue
        _, rel = _relative(x, self.data[f'act_decode/layer{i}/prefix_sum'])
        logging.info('bf16 layer%d prefix_sum vs fp32 HF: rel = %.1e', i, rel)
        self.assertTrue(np.all(np.isfinite(np.asarray(x, np.float32))))
      bf16_logits, _ = self._apply(
          *self._tokens(_PREFILL_LEN, _PREFILL_LEN + 1),
          decode_state=state,
          model=self.bf16_model,
      )
      bf16_prefill, _ = self._apply(
          *self._tokens(0, _PREFILL_LEN + 1), model=self.bf16_model
      )
      _, bf16_rel = _relative(bf16_logits, bf16_prefill[:, -1:, :])
      # The same stack in fp32, fed an input off by one bf16 ulp.
      base = self._decode_logits(self.model, self._prefill())
      perturbed = self._decode_logits(
          self.model, self._prefill(), perturbation=2**-8
      )
      _, conditioning = _relative(perturbed, base)
    logging.info(
        'bf16 decode vs bf16 prefill: rel = %.1e; fp32 response to a 2^-8'
        ' input perturbation: rel = %.1e',
        bf16_rel,
        conditioning,
    )
    self.assertLessEqual(bf16_rel, 10 * conditioning)

  @parameterized.named_parameters(
      ('prefill', 'cache_prefill', 0), ('decode', 'cache_decode', 1)
  )
  def test_kda_cache_matches_hf(self, prefix: str, extra_steps: int):
    """KDA conv window and recurrent state vs the HF `KimiDynamicCache`."""
    with jax.set_mesh(self.mesh):
      state = self._prefill()
      for step in range(extra_steps):
        pos = _PREFILL_LEN + step
        _, extra = self._apply(*self._tokens(pos, pos + 1), decode_state=state)
        state = extra['decode_state']
    for i, layer_type in enumerate(self.model.layer_types):
      if layer_type != k3_config_lib.LINEAR_ATTENTION:
        continue
      block_state = state[f'block_{i}']
      conv = np.asarray(block_state.conv_state)
      for branch, golden_key in enumerate(('conv_q', 'conv_k', 'conv_v')):
        golden = self.data[f'{prefix}/layer{i}/{golden_key}']
        self._assert_close(
            conv[:, branch],
            golden.reshape(conv.shape[0], *conv.shape[2:]),
            f'{prefix} layer{i} {golden_key}',
            _HF_REL,
        )
      # HF keeps the state V-first (`transpose_state_layout=True`); ours is
      # `[B, H, K, V]`.
      self._assert_close(
          block_state.recurrent_state,
          np.swapaxes(self.data[f'{prefix}/layer{i}/recurrent_state'], -1, -2),
          f'{prefix} layer{i} recurrent_state',
          _HF_REL,
      )

  def test_mla_cache_length_matches_hf(self):
    with jax.set_mesh(self.mesh):
      state = self._prefill()
      _, extra = self._apply(
          *self._tokens(_PREFILL_LEN, _PREFILL_LEN + 1), decode_state=state
      )
    for i, layer_type in enumerate(self.model.layer_types):
      if layer_type != k3_config_lib.FULL_ATTENTION:
        continue
      # Ours caches the 576-wide latent, HF the decompressed key/value, so only
      # the length is comparable; the values are pinned by the layer outputs.
      golden_len = self.data[f'cache_decode/layer{i}/key'].shape[2]
      np.testing.assert_array_equal(
          np.asarray(extra['decode_state'][f'block_{i}'].lengths),
          np.full((2,), golden_len, np.int32),
      )

  def _decode_logits(self, model, state, perturbation: float = 0.0):
    """Logits of the decode step, walking the stack so `x` can be perturbed."""
    _, x, attn_res_state = list(
        self._walk_decode_step(model, state, perturbation)
    )[-1]
    x = model.final_attn_res.apply(
        self.params['final_attn_res'], x, attn_res_state
    )
    x = model.final_ln.apply(self.params['final_ln'], x)
    return model.embed_linear.apply(self.params['embed_linear'], x)

  # --- Self-consistency -------------------------------------------------------

  def _stateless_logits(self, length: int):
    key = ('stateless_logits', length)
    if key not in self.memo:
      with jax.set_mesh(self.mesh):
        logits, _ = self._apply(*self._tokens(0, length))
      self.memo[key] = np.asarray(logits)
    return self.memo[key]

  @parameterized.named_parameters(
      ('one_shot', None),
      ('two_chunks', 8),
      ('four_chunks', 4),
      ('token_by_token', 1),
  )
  def test_prefill_then_decode_matches_long_prefill(self, chunk):
    """Cached prefill + one step == one 17-token stateless prefill."""
    with jax.set_mesh(self.mesh):
      state = self._prefill(chunk=chunk)
      logits, _ = self._apply(
          *self._tokens(_PREFILL_LEN, _PREFILL_LEN + 1), decode_state=state
      )
    reference = self._stateless_logits(_PREFILL_LEN + 1)[:, -1:, :]
    self._assert_close(
        logits, reference, f'decode logits vs prefill ({chunk=})', _SELF_REL
    )

  @parameterized.named_parameters(
      ('one_shot', None),
      ('two_chunks', 8),
      ('four_chunks', 4),
      ('token_by_token', 1),
  )
  def test_chunked_prefill_reaches_the_same_state(self, chunk):
    """Chunking the prefill is a no-op on the cache, up to fp32 rounding.

    `chunk=None` re-runs the identical computation and must be bit-exact, which
    separates a genuine chunking bug from summation-order noise.
    """
    with jax.set_mesh(self.mesh):
      one_shot = self._prefill()
      chunked = self._prefill(chunk=chunk)
    self._assert_states_close(
        chunked,
        one_shot,
        f'state ({chunk=})',
        rel=0.0 if chunk is None else _SELF_REL,
        atol=0.0 if chunk is None else _ATOL_FLOOR,
    )

  # --- Greedy generation ------------------------------------------------------

  def _greedy_incremental(self, num_steps: int):
    with jax.set_mesh(self.mesh):
      state = self._prefill()
      ids, segment_ids, positions = self._tokens(_PREFILL_LEN, _PREFILL_LEN + 1)
      tokens = []
      for step in range(num_steps):
        logits, extra = self._apply(
            ids, segment_ids, positions + step, decode_state=state
        )
        state = extra['decode_state']
        ids = jnp.argmax(logits[:, -1:, :], axis=-1).astype(jnp.int32)
        tokens.append(np.asarray(ids)[:, 0])
    return np.stack(tokens, axis=1)

  def _greedy_full_prefix(self, num_steps: int):
    prefix = self.input_ids[:, : _PREFILL_LEN + 1]
    tokens = []
    with jax.set_mesh(self.mesh):
      for _ in range(num_steps):
        length = prefix.shape[1]
        positions = np.broadcast_to(
            np.arange(length, dtype=np.int32), prefix.shape
        )
        logits, _ = self._apply(
            jnp.asarray(prefix),
            jnp.ones(prefix.shape, jnp.int32),
            jnp.asarray(positions),
        )
        nxt = np.asarray(jnp.argmax(logits[:, -1, :], axis=-1), np.int32)
        tokens.append(nxt)
        prefix = np.concatenate([prefix, nxt[:, None]], axis=1)
    return np.stack(tokens, axis=1)

  def test_greedy_decode_matches_full_prefix_rerun(self):
    num_steps = 4
    np.testing.assert_array_equal(
        self._greedy_incremental(num_steps), self._greedy_full_prefix(num_steps)
    )

  # --- The sampler's prefill pattern ------------------------------------------

  def test_prefill_position_matches_full_prefill(self):
    """`sampling_lib`'s pattern: prefill a window, then re-feed its tail.

    `LMInterface` prefills the whole padded window but only *commits* the first
    `prefill_position` tokens, and the sampling loop then supplies the rest one
    at a time. Absorbing the tail twice is unrecoverable for an order-indexed
    cache, so this must equal a plain 17-token prefill exactly.
    """
    window = _PREFILL_LEN + 1
    prefill_position = 12
    with jax.set_mesh(self.mesh):
      ids, segment_ids, positions = self._tokens(0, window)
      _, extra = self.model.apply(
          self.params,
          ids,
          segment_ids=segment_ids,
          segment_positions=positions,
          extra_inputs={
              'prefill_position': prefill_position,
              'decode_max_seq_len': _DECODE_MAX_SEQ_LEN,
          },
      )
      state = extra['decode_state']
      # Only the committed prefix is in the cache.
      self._assert_states_close(
          state,
          self._prefill(length=prefill_position),
          'committed',
          _SELF_REL,
      )
      for position in range(prefill_position, window - 1):
        _, extra = self._apply(
            *self._tokens(position, position + 1), decode_state=state
        )
        state = extra['decode_state']
      logits, _ = self._apply(
          *self._tokens(window - 1, window), decode_state=state
      )
    reference = self._stateless_logits(window)[:, -1:, :]
    self._assert_close(
        logits, reference, 'prefill_position + re-fed tail', _SELF_REL
    )

  def test_padded_decode_state_grows_the_cache_and_keeps_decoding(self):
    """`SamplingState.pad_to` runs before every decode chunk of `LMInterface`."""
    grown = _DECODE_MAX_SEQ_LEN + 8
    with jax.set_mesh(self.mesh):
      state = self._prefill()
      logits, _ = self._apply(
          *self._tokens(_PREFILL_LEN, _PREFILL_LEN + 1), decode_state=state
      )
      padded = cast(
          dict[str, Any],
          model_lib.pad_decode_state_to(dict(state), grown),
      )
      padded_logits, _ = self._apply(
          *self._tokens(_PREFILL_LEN, _PREFILL_LEN + 1), decode_state=padded
      )
    for i, layer_type in enumerate(self.model.layer_types):
      block = padded[f'block_{i}']
      if layer_type == k3_config_lib.FULL_ATTENTION:
        self.assertEqual(block.kv_cache.shape[1], grown)
      else:  # KDA's state does not grow with the sequence length.
        self.assertEqual(
            block.conv_state.shape, state[f'block_{i}'].conv_state.shape
        )
      np.testing.assert_array_equal(block.lengths, state[f'block_{i}'].lengths)
    np.testing.assert_array_equal(np.asarray(padded_logits), np.asarray(logits))

  def test_decode_without_positions_is_rejected(self):
    """A silent 0..T-1 default would restart both caches on every step."""
    with jax.set_mesh(self.mesh):
      ids, _, _ = self._tokens(_PREFILL_LEN, _PREFILL_LEN + 1)
      state = self.model.init_decode_state(_DECODE_MAX_SEQ_LEN)
      with self.assertRaisesRegex(ValueError, 'segment_positions is required'):
        self.model.apply(self.params, ids, decode_state=state)

  def test_prefill_longer_than_the_cache_is_rejected(self):
    with jax.set_mesh(self.mesh):
      ids, segment_ids, positions = self._tokens(0, _PREFILL_LEN)
      with self.assertRaisesRegex(ValueError, 'does not fit'):
        self.model.apply(
            self.params,
            ids,
            segment_ids=segment_ids,
            segment_positions=positions,
            extra_inputs={
                'prefill_position': _PREFILL_LEN,
                'decode_max_seq_len': _PREFILL_LEN - 1,
            },
        )

  # --- jit / scan -------------------------------------------------------------

  def test_jitted_scan_decode_matches_step_by_step(self):
    """The decode state is a fixed point of a jitted `lax.scan` loop."""
    num_steps = 4

    @jax.jit
    def run(params, prefill_ids, prefill_positions, first_id, first_position):
      state = self.model.init_decode_state(_DECODE_MAX_SEQ_LEN)
      segment_ids = jnp.ones(prefill_ids.shape, jnp.int32)
      _, extra = self.model.apply(
          params,
          prefill_ids,
          segment_ids=segment_ids,
          segment_positions=prefill_positions,
          decode_state=state,
      )

      def step(carry, _):
        state, ids, position = carry
        logits, extra = self.model.apply(
            params,
            ids,
            segment_ids=jnp.ones(ids.shape, jnp.int32),
            segment_positions=position,
            decode_state=state,
        )
        nxt = jnp.argmax(logits[:, -1:, :], axis=-1).astype(jnp.int32)
        return (extra['decode_state'], nxt, position + 1), nxt[:, 0]

      carry = (extra['decode_state'], first_id, first_position)
      _, tokens = jax.lax.scan(step, carry, length=num_steps)
      return jnp.swapaxes(tokens, 0, 1)

    prefill_ids, _, prefill_positions = self._tokens(0, _PREFILL_LEN)
    first_id, _, first_position = self._tokens(_PREFILL_LEN, _PREFILL_LEN + 1)
    with jax.set_mesh(self.mesh):
      tokens = run(
          self.params, prefill_ids, prefill_positions, first_id, first_position
      )
    np.testing.assert_array_equal(
        np.asarray(tokens), self._greedy_incremental(num_steps)
    )


# --- Remat -----------------------------------------------------------------
class RematTest(absltest.TestCase):
  """`use_remat` must be numerically transparent, and the policy must resolve.

  Nothing in the package turns it on -- the registered configs pin it off and
  training is not here -- so the branch that wraps a layer in `jax.remat`, and
  the `remat_policy` name it looks up on `jax.checkpoint_policies`, are only
  reachable from a config a user writes. A typo in either is silent otherwise.
  """

  def test_rematted_prefill_matches_the_plain_one(self):
    config = test_utils.tiny_config(n_layers=4, attn_res_block_size=2)
    plain = k3_model_lib.KimiK3LM(config=config)
    rematted = k3_model_lib.KimiK3LM(
        config=dataclasses.replace(config, use_remat=True)
    )
    params = plain.init(jax.random.key(0))
    tokens = jnp.zeros((config.batch_size, 8), jnp.int32)
    want, _ = plain.apply(params, tokens)
    got, _ = rematted.apply(params, tokens)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=0)


# --- The LMInterface path -----------------------------------------------------
MESH_AXES = ('replica', 'data', 'model')
_LM_MAX_SEQ_LEN = 512
_VOCAB_NAME = 'ByteVocabForK3Test'


if _VOCAB_NAME not in tokenization.TokenizerRegistry.keys():
  tokenization.TokenizerRegistry.register_value(
      tokenization.ByteVocab(), name=_VOCAB_NAME
  )


def _tiny_config(vocab_size: int) -> config_lib.BaseExperimentConfig:
  """`kimi_k3_tiny_test` with a vocab a byte tokenizer can drive.

  Not `test_utils.tiny_config`: this suite drives the *sampling* path, so it
  needs the registered config's width and a `seq_len` long enough to generate
  into, and it needs the vocabulary and input-processor fields to match the
  byte vocab it installs. `test_utils.tiny_config` is the opposite trade -- a
  quarter of the width and a free layer count, for the structural tests.

  Args:
    vocab_size: the byte vocabulary's size, which the model must match.

  Returns:
    The config.
  """
  return dataclasses.replace(
      k3_config_lib.kimi_k3_tiny_test(),
      vocab_name=_VOCAB_NAME,
      vocab_size=vocab_size,
      # The released processor renders XTML for the 160k vocab; a byte vocab
      # needs the plain text path.
      input_processor_name=None,
      seq_len=_LM_MAX_SEQ_LEN,
  )


class LMInterfaceTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Scoped to the class rather than the module: the other three classes below
    # carry their own meshes, and a leaked global one would make them depend on
    # which shard they land in.
    cls.enterClassContext(
        sharding_lib.set_mesh(
            {name: 1 for name in MESH_AXES}, axis_names=MESH_AXES
        )
    )
    cls.vocab = tokenization.TokenizerRegistry.get_instance(_VOCAB_NAME)
    cls.config = _tiny_config(cls.vocab.vocab_size)
    cls.model, _ = model_lib.create_model(cls.config)
    cls.params = cls.model.init(jax.random.key(0))
    # Eager `apply` recompiles every primitive per shape, which dominates this
    # test; one jit per sequence length instead.
    cls.logits_fn = staticmethod(
        jax.jit(lambda tokens: cls.model.apply(cls.params, tokens)[0])
    )

  def _lm_interface(self, **sampling_kwargs) -> model_lib.LMInterface:
    """`decode_eval.main`'s interface, minus the checkpoint load."""
    params = dict(
        temperature=0.0,
        max_seq_len=_LM_MAX_SEQ_LEN,
        max_decode_steps=6,
        num_samples=1,
    )
    params.update(sampling_kwargs)
    return model_lib.LMInterface(
        self.model,
        params=self.params,
        input_processor=sampling_lib.create_input_processor(
            self.config, vocab=self.vocab
        ),
        default_sampling_params=model_lib.SamplingParams(**params),
    )

  def _assert_is_greedy(self, prompt: str, output_token_ids: list[int]):
    """Teacher-forces the generation through one stateless pass.

    Greedy decoding means every generated token is the argmax given the true
    prefix, so a single pass over `prompt + output` checks them all -- and it
    shares no decode-state code with the sampler, so a cache bug cannot cancel
    out.

    Args:
      prompt: The prompt that was generated from.
      output_token_ids: The sampler's generated ids.
    """
    prompt_ids = [self.vocab.bos_id] + self.vocab.encode(prompt)
    logits = self.logits_fn(jnp.asarray([prompt_ids + output_token_ids]))
    predicted = np.argmax(np.asarray(logits[0]), axis=-1)
    np.testing.assert_array_equal(
        predicted[len(prompt_ids) - 1 : -1],
        np.asarray(output_token_ids),
        err_msg=f'{prompt=}',
    )

  @parameterized.named_parameters(
      # `prefill_size <= min_input_len - 1`: the sampler decodes from the end
      # of the prefill window, the case TransformerLM also sees.
      ('short_prefill', 2, None),
      # `prefill_size > min_input_len - 1`: the sampler restarts *inside* the
      # window, so the window tail must not be absorbed twice.
      ('long_prefill', 16, None),
      # Several decode chunks, so the state is padded (and the MLA cache grown)
      # between them.
      ('chunked_decode', 16, 2),
  )
  def test_generate_matches_greedy_reference(
      self, prefill_size, intermediate_decode_steps
  ):
    max_decode_steps = 4
    prompts = ['hi', 'hello there']
    lm_interface_ = self._lm_interface(
        prefill_size=prefill_size,
        max_decode_steps=max_decode_steps,
        intermediate_decode_steps=intermediate_decode_steps,
    )
    # A batch larger than the input exercises the all-padding rows too.
    outputs = cast(
        list[list[model_lib.SamplingOutput]],
        lm_interface_.generate(
            prompts, prng_key=0, batch_size=4, scoring_inputs=False
        ),
    )
    for prompt, sample_outputs in zip(prompts, outputs, strict=True):
      self.assertLen(sample_outputs, 1)
      self.assertEqual(
          sample_outputs[0].input_token_ids,
          [self.vocab.bos_id] + self.vocab.encode(prompt),
      )
      # A random-init model does not emit eos; anything shorter means the loop
      # stopped early, which `_assert_is_greedy` alone would not catch.
      self.assertLen(sample_outputs[0].output_token_ids, max_decode_steps)
      self._assert_is_greedy(prompt, sample_outputs[0].output_token_ids)

  def test_decode_eval_loop_scores_generations(self):
    """`decode_eval.main`'s per-example body on a canned example."""
    evaluation = evaluation_lib.EvaluationRegistry.get_instance(
        'ZeroShotDeepSeekQwenR1CoTBoxed'
    )
    lm_format = lm_format_lib.LMFormatRegistry.get_instance('SimplyV1Chat')
    example = {'question': 'What is 2+2?', 'short_answer': '4'}

    sampling_input = evaluation.get_sampling_input(example, lm_format)
    outputs = cast(
        list[list[model_lib.SamplingOutput]],
        self._lm_interface(prefill_size=128, max_decode_steps=4).generate(
            [sampling_input], prng_key=0, batch_size=1, scoring_inputs=False
        ),
    )
    response = outputs[0][0].output_text

    # The scorer runs on whatever the (random-init) model produced ...
    self.assertEqual(
        evaluation.evaluate(example, response),
        {'correct': False, 'reward': 0.0},
    )
    # ... and would credit the answer if the model had found it.
    self.assertEqual(
        evaluation.evaluate(example, r'so it is \boxed{4}'),
        {'correct': True, 'reward': 1.0},
    )

  def test_prefill_opens_the_cache_at_the_prefill_window(self):
    """Not at `config.seq_len`: the sampler grows it per chunk.

    Preallocating the sampler's whole horizon is 22.6 GiB/device of MLA cache
    at batch 64 / 32k on 4x4x8, most of it never written.
    """
    prefill_size = 8
    self.assertGreater(self.config.seq_len, prefill_size)
    tokens = jnp.zeros((2, prefill_size), jnp.int32)
    _, extra = jax.jit(self.model.apply)(
        self.params,
        tokens,
        segment_ids=jnp.ones_like(tokens),
        segment_positions=jnp.broadcast_to(
            jnp.arange(prefill_size, dtype=jnp.int32), tokens.shape
        ),
        extra_inputs={'prefill_position': prefill_size},
    )
    rows = {
        state.kv_cache.shape[1]
        for state in extra['decode_state'].values()
        if isinstance(state, mla_lib.KimiK3MLADecodeState)
    }
    self.assertEqual(rows, {prefill_size})

  def test_pad_decode_state_grows_the_latent_cache_only(self):
    state: dict[str, Any] = self.model.init_decode_state(
        max_seq_len=4, batch_size=2
    )
    # `pad_decode_state_to` rewrites its argument in place.
    grown = cast(dict[str, Any], model_lib.pad_decode_state_to(dict(state), 9))
    for name, block_state in state.items():
      if isinstance(block_state, mla_lib.KimiK3MLADecodeState):
        self.assertEqual(grown[name].kv_cache.shape[1], 9, msg=name)
        np.testing.assert_array_equal(
            grown[name].kv_cache[:, :4], block_state.kv_cache
        )
      else:
        self.assertIsInstance(block_state, kda_lib.KDADecodeState)
        self.assertEqual(
            jax.tree.structure(grown[name]),
            jax.tree.structure(block_state),
            msg=name,
        )

  def test_pad_decode_state_rejects_unknown_block_state(self):
    with self.assertRaises(TypeError):
      model_lib.pad_decode_state_to(cast(Any, {'block_0': object()}), 4)


if __name__ == '__main__':
  absltest.main()
