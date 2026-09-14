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
"""Tests for the reference (non-Pallas) ragged paged attention.

The reference implementation is what runs off TPU, inside the jitted decode
loops of `simply/serving` -- so these tests pin down that it is jittable and
that jit and eager agree, on CPU.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from simply.kernels import ragged_paged_attention as rpa


_NUM_Q_HEADS = 4
_NUM_KV_HEADS = 2
_HEAD_DIM = 128
_PAGE_SIZE = 4
_PAGES_PER_SEQ = 3


def _histories(rng, kv_lens, dtype):
  """Per-sequence K/V history, `[kv_len, num_kv_heads, head_dim]` each."""
  histories = []
  for kv_len in kv_lens:
    k = rng.normal(size=(kv_len, _NUM_KV_HEADS, _HEAD_DIM))
    v = rng.normal(size=(kv_len, _NUM_KV_HEADS, _HEAD_DIM))
    histories.append((jnp.asarray(k, dtype), jnp.asarray(v, dtype)))
  return histories


def _make_inputs(rng, q_lens, kv_lens, dtype=jnp.float32, max_num_tokens=None,
                 num_seqs=None, write_tail=True):
  """Builds RPA inputs; sequence `i` owns pages `[i * _PAGES_PER_SEQ, ...)`.

  Args:
    rng: numpy random generator.
    q_lens: per-sequence query length.
    kv_lens: per-sequence KV length, including the query tokens.
    dtype: activation/cache dtype.
    max_num_tokens: query buffer size; defaults to `sum(q_lens)` (no padding).
    num_seqs: value for `distribution[-1]`; defaults to all sequences.
    write_tail: if False, the cache already holds the query tokens (what a
      caller passing `update_kv_cache=False` would do).

  Returns:
    `(kwargs, histories)` where `histories` are the per-sequence K/V arrays
    the attention output should be computed against.
  """
  num_kv_seqs = len(kv_lens)
  max_num_tokens = sum(q_lens) if max_num_tokens is None else max_num_tokens
  histories = _histories(rng, kv_lens, dtype)

  packing = rpa.get_dtype_packing(dtype)
  num_kv_heads_x2 = rpa.align_to(_NUM_KV_HEADS * 2, packing)
  total_num_pages = num_kv_seqs * _PAGES_PER_SEQ
  kv_cache = jnp.zeros(
      (total_num_pages, _PAGE_SIZE, num_kv_heads_x2 // packing, packing,
       _HEAD_DIM), dtype)

  queries = jnp.asarray(
      rng.normal(size=(max_num_tokens, _NUM_Q_HEADS, _HEAD_DIM)), dtype)
  keys = jnp.zeros((max_num_tokens, _NUM_KV_HEADS, _HEAD_DIM), dtype)
  values = jnp.zeros_like(keys)

  q_start = 0
  for i, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
    k_hist, v_hist = histories[i]
    # The last `q_len` KV entries are the query tokens: they are passed in
    # `keys`/`values` and only reach the cache via `update_kv_cache`.
    num_cached = kv_len - q_len if write_tail else kv_len
    merged = rpa.merge_kv(k_hist[:num_cached], v_hist[:num_cached])
    for pos in range(num_cached):
      page = i * _PAGES_PER_SEQ + pos // _PAGE_SIZE
      kv_cache = kv_cache.at[page, pos % _PAGE_SIZE].set(merged[pos])
    if write_tail and q_len:
      keys = keys.at[q_start:q_start + q_len].set(k_hist[kv_len - q_len:])
      values = values.at[q_start:q_start + q_len].set(v_hist[kv_len - q_len:])
    q_start += q_len

  page_indices = jnp.asarray(
      np.arange(num_kv_seqs * _PAGES_PER_SEQ).reshape(
          num_kv_seqs, _PAGES_PER_SEQ), jnp.int32)
  kwargs = dict(
      queries=queries,
      keys=keys,
      values=values,
      kv_cache=kv_cache,
      kv_lens=jnp.asarray(kv_lens, jnp.int32),
      page_indices=page_indices,
      cu_q_lens=jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]),
                            jnp.int32),
      distribution=jnp.asarray(
          [0, 0, num_kv_seqs if num_seqs is None else num_seqs], jnp.int32),
  )
  return kwargs, histories


def _naive_attention(queries, q_lens, kv_lens, histories, num_seqs,
                     sliding_window=None, use_causal_mask=True):
  """Straightforward dense attention, one sequence at a time, in float32."""
  q = np.asarray(queries, np.float32)
  out = np.zeros_like(q)
  q_start = 0
  for i, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
    if i >= num_seqs:
      break
    k, v = (np.asarray(x, np.float32) for x in histories[i])
    k = np.repeat(k, _NUM_Q_HEADS // _NUM_KV_HEADS, axis=1)
    v = np.repeat(v, _NUM_Q_HEADS // _NUM_KV_HEADS, axis=1)
    for j in range(q_len):
      pos = kv_len - q_len + j
      logits = np.einsum('hd,khd->hk', q[q_start + j], k)
      if use_causal_mask:
        allowed = np.arange(kv_len) <= pos
        if sliding_window is not None:
          allowed &= np.arange(kv_len) > pos - sliding_window
        logits = np.where(allowed, logits, -np.inf)
      weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
      weights /= weights.sum(axis=-1, keepdims=True)
      out[q_start + j] = np.einsum('hk,khd->hd', weights, v)
    q_start += q_len
  return out


class RefRaggedPagedAttentionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # A ragged batch with a padded query buffer: 2 prefills and a decode.
      dict(testcase_name='ragged_with_padding', q_lens=(3, 1, 2),
           kv_lens=(3, 9, 5), max_num_tokens=8),
      dict(testcase_name='decode_only', q_lens=(1, 1, 1), kv_lens=(4, 9, 2)),
      dict(testcase_name='empty_sequence', q_lens=(2, 0, 1),
           kv_lens=(6, 0, 3)),
      dict(testcase_name='inactive_sequences', q_lens=(2, 2, 2),
           kv_lens=(6, 7, 8), num_seqs=2, max_num_tokens=6),
      dict(testcase_name='bfloat16', q_lens=(3, 1, 2), kv_lens=(3, 9, 5),
           dtype=jnp.bfloat16),
      dict(testcase_name='sliding_window', q_lens=(3, 1, 2),
           kv_lens=(3, 9, 5), kernel_kwargs=dict(sliding_window=3)),
      dict(testcase_name='no_causal_mask', q_lens=(3, 1, 2),
           kv_lens=(3, 9, 5), kernel_kwargs=dict(use_causal_mask=False)),
      dict(testcase_name='soft_cap', q_lens=(3, 1, 2), kv_lens=(3, 9, 5),
           kernel_kwargs=dict(soft_cap=5.0)),
      dict(testcase_name='no_kv_cache_update', q_lens=(3, 1, 2),
           kv_lens=(3, 9, 5), kernel_kwargs=dict(update_kv_cache=False)),
  )
  def test_jit_matches_eager(self, q_lens, kv_lens, dtype=jnp.float32,
                             max_num_tokens=None, num_seqs=None,
                             kernel_kwargs=None):
    kernel_kwargs = kernel_kwargs or {}
    update_kv_cache = kernel_kwargs.get('update_kv_cache', True)
    kwargs, _ = _make_inputs(
        np.random.default_rng(0), q_lens, kv_lens, dtype=dtype,
        max_num_tokens=max_num_tokens, num_seqs=num_seqs,
        write_tail=update_kv_cache)

    fn = lambda **kw: rpa.ref_ragged_paged_attention(**kw, **kernel_kwargs)
    eager_out, eager_cache, _ = fn(**kwargs)
    jit_out, jit_cache, _ = jax.jit(fn)(**kwargs)

    np.testing.assert_array_equal(np.asarray(jit_out), np.asarray(eager_out))
    if update_kv_cache:
      np.testing.assert_array_equal(
          np.asarray(jit_cache), np.asarray(eager_cache))
    else:
      self.assertIsNone(jit_cache)

  @parameterized.named_parameters(
      dict(testcase_name='prefill_and_decode', q_lens=(3, 1, 2),
           kv_lens=(3, 9, 5), max_num_tokens=8),
      dict(testcase_name='sliding_window', q_lens=(2, 2), kv_lens=(8, 5),
           kernel_kwargs=dict(sliding_window=3)),
      dict(testcase_name='no_causal_mask', q_lens=(2, 2), kv_lens=(8, 5),
           kernel_kwargs=dict(use_causal_mask=False)),
  )
  def test_matches_naive_attention(self, q_lens, kv_lens, max_num_tokens=None,
                                   kernel_kwargs=None):
    kernel_kwargs = kernel_kwargs or {}
    kwargs, histories = _make_inputs(
        np.random.default_rng(1), q_lens, kv_lens,
        max_num_tokens=max_num_tokens)

    out, _, _ = jax.jit(
        lambda **kw: rpa.ref_ragged_paged_attention(**kw, **kernel_kwargs)
    )(**kwargs)

    expected = _naive_attention(
        kwargs['queries'], q_lens, kv_lens, histories, len(kv_lens),
        **kernel_kwargs)
    num_q_tokens = sum(q_lens)
    np.testing.assert_allclose(
        np.asarray(out[:num_q_tokens], np.float32), expected[:num_q_tokens],
        atol=1e-4, rtol=1e-4)
    # Padding tokens get no output.
    np.testing.assert_array_equal(
        np.asarray(out[num_q_tokens:]),
        np.zeros_like(np.asarray(out[num_q_tokens:])))

  def test_update_kv_cache_appends_query_tokens(self):
    q_lens, kv_lens = (2, 3), (6, 3)
    kwargs, histories = _make_inputs(
        np.random.default_rng(2), q_lens, kv_lens)

    _, cache, _ = jax.jit(rpa.ref_ragged_paged_attention)(**kwargs)

    for i, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
      k_hist, v_hist = histories[i]
      expected = rpa.merge_kv(k_hist, v_hist)
      for pos in range(kv_len):
        page = i * _PAGES_PER_SEQ + pos // _PAGE_SIZE
        np.testing.assert_array_equal(
            np.asarray(cache[page, pos % _PAGE_SIZE]),
            np.asarray(expected[pos]),
            err_msg=f'sequence {i} position {pos} (q_len={q_len})')

  def test_inactive_sequences_are_untouched(self):
    q_lens, kv_lens = (2, 2), (6, 7)
    kwargs, _ = _make_inputs(
        np.random.default_rng(3), q_lens, kv_lens, num_seqs=1)

    out, cache, _ = jax.jit(rpa.ref_ragged_paged_attention)(**kwargs)

    # Sequence 1 contributes no output and its pages keep their old contents.
    np.testing.assert_array_equal(
        np.asarray(out[q_lens[0]:]),
        np.zeros_like(np.asarray(out[q_lens[0]:])))
    np.testing.assert_array_equal(
        np.asarray(cache[_PAGES_PER_SEQ:]),
        np.asarray(kwargs['kv_cache'][_PAGES_PER_SEQ:]))

  def test_dynamic_validation_rejects_bad_inputs_when_concrete(self):
    kwargs, _ = _make_inputs(np.random.default_rng(4), (2,), (6,))
    kwargs['kv_lens'] = jnp.asarray([1], jnp.int32)  # < q_len

    with self.assertRaises(ValueError):
      rpa.ref_ragged_paged_attention(**kwargs)

  def test_is_traced(self):
    x = jnp.arange(3)
    self.assertFalse(rpa.is_traced(x))
    self.assertTrue(jax.jit(rpa.is_traced)(x))


if __name__ == '__main__':
  absltest.main()
