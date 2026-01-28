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
import dataclasses
import functools

from absl.testing import absltest
import einops
import jax
import jax.numpy as jnp
import numpy as np
from simply.utils import common
from simply.utils import ragged_paged_attention as rpa
from simply.utils import sampling_lib
from simply.utils import sharding

RaggedArray = common.RaggedArray


def qkv_attn(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
  """Computes qkv attention (reference implementation)."""
  q_len = q.shape[-3]
  kv_len = k.shape[-3]
  n_kv_heads = k.shape[-2]
  q = einops.rearrange(q, ' ... (m g) h -> ... m g h', m=n_kv_heads)
  attn = jnp.einsum('...tmgh,...smh->...tmgs', q, k)
  q_span = jnp.arange(kv_len - q_len, kv_len)
  kv_span = jnp.arange(kv_len)
  mask = q_span[:, None, None, None] >= kv_span
  attn += (~mask) * (-0.7 * float(jnp.finfo(jnp.dtype('float32')).max))
  output = jnp.einsum(
      '...tmgs,...smh->...tmgh', jax.nn.softmax(attn, axis=-1), v
  )
  output = einops.rearrange(output, '... m g h -> ... (m g) h')
  return output


class DecodeStateTest(absltest.TestCase):

  def test_allocate(self):
    total_num_pages = 6
    page_size = 3
    config = rpa.DecodeStateConfig(
        total_num_pages=total_num_pages,
        page_size=page_size,
        n_kv_heads=1,
        per_head_dim=2,
        batch_size=3,
        dtype='float32',
        max_seq_len=total_num_pages * page_size + 1,
    )
    ds = config.init()

    ds = ds.allocate(q_lens=jnp.array([3, 0, 4]))
    np.testing.assert_array_equal(ds.kv_lens, np.array([3, 0, 4]))
    np.testing.assert_array_equal(
        ds.available_page_indices_np, np.array([3, 4, 5])
    )
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        ds.page_indices_nplist,
        [np.array([0]), np.array([]), np.array([1, 2])],
    )

    ds = dataclasses.replace(
        ds,
        page_indices=jnp.array([
            [3, -1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ]),
        kv_lens=jnp.array([3, 2, 0]),
        num_available_pages=jnp.array(4, dtype=jnp.int32),
        available_page_indices=jnp.array([5, 2, 4, 0, -1, -1]),
    )
    ds = jax.jit(ds.allocate)(q_lens=jnp.array([5, 0, 6]))
    np.testing.assert_array_equal(ds.kv_lens, jnp.array([8, 2, 6]))
    np.testing.assert_array_equal(ds.available_page_indices_np, np.array([]))
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        ds.page_indices_nplist,
        [np.array([3, 5, 2]), np.array([1]), np.array([4, 0])],
    )

  def test_insert(self):
    total_num_pages = 6
    page_size = 3
    n_kv_heads = 1
    per_head_dim = 2
    pages_shape = (total_num_pages, page_size, n_kv_heads * 2, per_head_dim)
    pages = jnp.reshape(jnp.arange(np.prod(pages_shape)), pages_shape)
    pages = jnp.pad(pages, ((0, 0), (0, 0), (0, 0), (0, 128 - per_head_dim)))
    page_indices = jnp.array([
        [5, 1, 3],
        [2, 0, -1],
        [4, -1, -1],
    ])
    active = (
        jnp.zeros(total_num_pages, dtype=jnp.bool)
        .at[jnp.ravel(page_indices)]
        .set(True, mode='drop', wrap_negative_indices=False)
    )
    available_page_indices = jnp.flatnonzero(~active, size=total_num_pages)
    num_available_pages = jnp.sum(page_indices < 0)
    kv_lens = jnp.array([3, 2, 1])
    q_lens = jnp.array([5, 2, 2])
    ds = rpa.DecodeState(
        pages=pages,
        page_indices=page_indices,
        available_page_indices=available_page_indices,
        num_available_pages=num_available_pages,
        kv_lens=kv_lens + q_lens,
        max_seq_len=8,
    )

    np.testing.assert_array_equal(
        ds.num_pages, jnp.sum(page_indices >= 0, axis=-1)
    )
    new_kv_shape = (jnp.sum(q_lens), n_kv_heads * 2, per_head_dim)
    new_kv = jnp.reshape(jnp.arange(np.prod(new_kv_shape)) * -1, new_kv_shape)
    ds = jax.jit(ds.insert)(new_kv[:, 0::2], new_kv[:, 1::2], q_lens)
    np.testing.assert_array_equal(
        ds.pages[:, :, :, :per_head_dim],
        np.array([
            [[[-24, -25], [-26, -27]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
            [[[0, -1], [-2, -3]], [[-4, -5], [-6, -7]], [[-8, -9], [-10, -11]]],
            [
                [[24, 25], [26, 27]],
                [[28, 29], [30, 31]],
                [[-20, -21], [-22, -23]],
            ],
            [
                [[-12, -13], [-14, -15]],
                [[-16, -17], [-18, -19]],
                [[44, 45], [46, 47]],
            ],
            [
                [[48, 49], [50, 51]],
                [[-28, -29], [-30, -31]],
                [[-32, -33], [-34, -35]],
            ],
            [[[60, 61], [62, 63]], [[64, 65], [66, 67]], [[68, 69], [70, 71]]],
        ]),
    )

  def test_update_decode_state_and_compute_attn(self):
    total_num_pages = 6
    page_size = 3
    max_seq_len = total_num_pages * page_size + 1
    n_kv_heads = 2
    n_q_heads = 4
    per_head_dim = 2
    batch_size = 4
    rk1, rk2, rk3 = jax.random.split(jax.random.key(0), 3)

    sharding.set_mesh([1, 1, 1])

    ragged_old_kv = RaggedArray(
        data=jax.random.normal(
            rk1, (max_seq_len, n_kv_heads * 2, per_head_dim)
        ),
        lens=jnp.array([3, 2, 0, 1]),
    )
    ragged_q = RaggedArray(
        data=jax.random.normal(rk2, (max_seq_len, n_q_heads, per_head_dim)),
        lens=jnp.array([5, 2, 0, 2]),
    )
    ragged_kv = RaggedArray(
        data=jax.random.normal(
            rk3, (max_seq_len, n_kv_heads * 2, per_head_dim)
        ),
        lens=ragged_q.lens,
    )
    updated_ragged_kv = ragged_old_kv.concat(ragged_kv)
    expected_attn_out_list = []
    for i in range(batch_size):
      q = jnp.reshape(ragged_q.row(i), (-1, n_q_heads, per_head_dim))
      k = updated_ragged_kv.row(i)[:, 0::2]
      v = updated_ragged_kv.row(i)[:, 1::2]
      o = qkv_attn(q, k, v)
      expected_attn_out_list.append(o)

    config = rpa.DecodeStateConfig(
        total_num_pages=total_num_pages,
        page_size=page_size,
        n_kv_heads=n_kv_heads,
        per_head_dim=per_head_dim,
        batch_size=batch_size,
        dtype='float32',
        max_seq_len=max_seq_len,
    )
    ds = (
        config.init()
        .allocate(ragged_old_kv.lens)
        .insert(
            ragged_old_kv.data[:, 0::2],
            ragged_old_kv.data[:, 1::2],
            ragged_old_kv.lens,
        )
    )

    ds, ragged_attn_out = ds.update_decode_state_and_compute_attn(
        q=ragged_q,
        k=ragged_kv.data[:, 0::2],
        v=ragged_kv.data[:, 1::2],
    )
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        ds.kv_nplist(per_head_dim),
        updated_ragged_kv.to_numpy_list(),
    )
    jax.tree_util.tree_map(
        functools.partial(np.testing.assert_allclose, atol=0.005),
        RaggedArray(ragged_attn_out, ragged_q.lens).to_numpy_list(),
        expected_attn_out_list,
    )


class SamplingStateTest(absltest.TestCase):

  def test_push_and_release(self):

    sampling_state = rpa.SamplingState.create(
        max_total_num_tokens=16,
        prng_key=jax.random.key(0),
        eos_ids=jnp.array([100]),
        decode_state=rpa.DecodeStateConfig(
            total_num_pages=6,
            page_size=3,
            n_kv_heads=1,
            per_head_dim=2,
            batch_size=3,
            max_seq_len=8,
            dtype='float32',
        ).init(),
    )
    inputs = np.arange(15).reshape(3, 5)
    lens = [3, 5, 1]
    for i in range(len(inputs)):
      sampling_state, idx = sampling_state.push(inputs[i], lens[i], 0)
      self.assertEqual(idx, i)
    sampling_state = dataclasses.replace(
        sampling_state, position=jnp.array(lens) - 1
    )
    np.testing.assert_array_equal(
        sampling_state.input_lens, np.array([3, 5, 1])
    )
    np.testing.assert_array_equal(
        sampling_state.rank, np.array([0, 1, 2])
    )
    outputs = sampling_state.get(np.array([True, True, True]))
    for i in range(len(outputs)):
      self.assertEqual(outputs[i]['index'], i)
      np.testing.assert_array_equal(outputs[i]['tokens'], inputs[i][:lens[i]])

    sampling_state = sampling_state.release(jnp.array([False, True, False]))
    np.testing.assert_array_equal(
        sampling_state.input_lens, np.array([3, 0, 1])
    )

    sampling_state, idx = sampling_state.push(np.array([3, 2, 1]), 3, 0)
    sampling_state = dataclasses.replace(
        sampling_state, position=sampling_state.position.at[1].set(2)
    )
    self.assertEqual(idx, 1)
    (output,) = sampling_state.get(np.array([False, True, False]))
    self.assertEqual(output['index'], 1)
    np.testing.assert_array_equal(output['tokens'], np.array([3, 2, 1]))
    np.testing.assert_array_equal(
        sampling_state.input_lens, np.array([3, 3, 1])
    )
    np.testing.assert_array_equal(
        sampling_state.rank, np.array([0, 2, 1])
    )

  def test_ragged_issue_tokens(self):
    sampling_state = rpa.SamplingState.create(
        max_total_num_tokens=8,
        prng_key=jax.random.key(0),
        eos_ids=jnp.array([100]),
        decode_state=rpa.DecodeStateConfig(
            total_num_pages=6,
            page_size=3,
            n_kv_heads=1,
            per_head_dim=2,
            batch_size=2,
            max_seq_len=8,
            dtype='float32',
        ).init(),
    )
    tokens = RaggedArray.from_numpy_list([
        np.array([1, 2, 3]),
        np.array([4, 5, 6, 7, 8, 9, 10]),
    ])
    sampling_state = dataclasses.replace(
        sampling_state,
        tokens=tokens.to_padded_dense(sampling_state.max_seq_len),
        token_logprobs=jnp.zeros_like(sampling_state.tokens, dtype=jnp.float32),
        token_scores=jnp.zeros_like(sampling_state.tokens, dtype=jnp.float32),
        input_lens=tokens.lens,
        position=jnp.array([0, 2]),
        max_decode_steps=jnp.array([5, 5]),
        rank=jnp.array([1, 0]),
    )
    np.testing.assert_array_equal(
        sampling_state.issue_lens(capacity=100), np.array([1, 5])
    )
    ragged_issue_tokens = sampling_state.ragged_issue_tokens(100)
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        ragged_issue_tokens.to_numpy_list(),
        [np.array([1]), np.array([6, 7, 8, 9, 10])],
    )

    ragged_output_tokens = dataclasses.replace(
        ragged_issue_tokens, data=-ragged_issue_tokens.data
    )
    sampling_state = sampling_state.update_with_ragged_output(
        ragged_output_tokens,
        token_logprobs=jnp.ones_like(ragged_output_tokens.data),
        token_scores=jnp.ones_like(ragged_output_tokens.data),
    )
    np.testing.assert_array_equal(
        sampling_state.position, np.array([1, 7])
    )
    outputs = sampling_state.get(np.array([True, True]))
    jax.tree_util.tree_map(
        np.testing.assert_almost_equal,
        outputs,
        [
            dict(
                index=0,
                input_len=tokens.lens[0],
                tokens=np.array([1, 2]),
                logprobs=np.zeros(2),
                scores=np.zeros(2),
            ),
            dict(
                index=1,
                input_len=tokens.lens[1],
                tokens=np.array([4, 5, 6, 7, 8, 9, 10, -10]),
                logprobs=np.pad(np.zeros(7), (0, 1), constant_values=1),
                scores=np.pad(np.zeros(7), (0, 1), constant_values=1),
            ),
        ],
    )

  def test_continue_decode(self):
    max_seq_len = 8
    batch_size = 2
    vocab_size = 10
    per_head_dim = 2

    sharding.set_mesh([1, 1, 1])

    sampling_state = rpa.SamplingState.create(
        max_total_num_tokens=max_seq_len * 100,
        prng_key=jax.random.key(0),
        eos_ids=jnp.array([100]),
        decode_state=rpa.DecodeStateConfig(
            total_num_pages=6,
            page_size=3,
            n_kv_heads=1,
            per_head_dim=per_head_dim,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            dtype='float32',
            head_partition='model',
        ).init(),
    )
    tokens = RaggedArray.from_numpy_list(
        [np.array([0, 1, 2]), np.array([4, 5, 6, 7, 8])]
    )
    sampling_state = dataclasses.replace(
        sampling_state,
        tokens=tokens.to_padded_dense(max_seq_len),
        input_lens=tokens.lens,
        position=jnp.array([0, 0]),
        max_decode_steps=jnp.array([0, 10]),
        rank=jnp.array([0, 1]),
    )

    n_heads = 2
    n_kv_heads = 1
    emb_key, q_key, k_key, v_key = jax.random.split(jax.random.key(0), 4)
    params = dict(
        emb=jax.random.normal(emb_key, (vocab_size, per_head_dim)),
        q_proj=jax.random.normal(q_key, (n_heads, per_head_dim, per_head_dim)),
        k_proj=jax.random.normal(
            k_key, (n_kv_heads, per_head_dim, per_head_dim)
        ),
        v_proj=jax.random.normal(
            v_key, (n_kv_heads, per_head_dim, per_head_dim)
        ),
    )

    def forward_fn(
        params: common.PyTree,
        tokens: jax.Array,
        segment_ids: jax.Array,
        segment_positions: jax.Array,
        extra_inputs: common.PyTree = None,
        decode_state: common.PyTree = None,
        ragged: bool = True,
    ) -> tuple[jax.Array, common.PyTree]:
      del segment_ids
      emb = jnp.take(params['emb'], tokens, axis=0)
      emb *= jnp.cos(segment_positions)[:, :, None]
      q = jnp.einsum('...d,ndh->...nh', emb, params['q_proj'])
      k = jnp.einsum('...d,ndh->...nh', emb, params['k_proj'])
      v = jnp.einsum('...d,ndh->...nh', emb, params['v_proj'])
      if ragged:
        q = einops.rearrange(q, '1 l ... -> l ...')
        k = einops.rearrange(k, '1 l ... -> l ...')
        v = einops.rearrange(v, '1 l ... -> l ...')
        decode_state, attn_out = (
            decode_state.update_decode_state_and_compute_attn(
                q=RaggedArray(q, extra_inputs['lens']),
                k=k,
                v=v,
            )
        )
        attn_out = einops.rearrange(attn_out, 'l ... -> 1 l ...')
      else:
        attn_out = qkv_attn(q, k, v)
      output = jnp.einsum(
          'vd,...d->...v', params['emb'], jnp.mean(attn_out, axis=-2)
      )
      return output, {'decode_state': decode_state}

    next_sampling_state = sampling_state.continue_decode(
        forward_fn=forward_fn,
        until_fn=lambda state: jnp.any(~state.is_pad_seq & state.has_ended),
        params=params,
    )
    np.testing.assert_array_equal(next_sampling_state.lens, np.array([3, 6]))
    np.testing.assert_array_equal(
        next_sampling_state.has_ended, np.array([True, False])
    )
    final_sampling_state = next_sampling_state.continue_decode(
        forward_fn=forward_fn,
        until_fn=lambda state: jnp.all(state.has_ended),
        params=params,
    )
    np.testing.assert_array_equal(final_sampling_state.lens, np.array([3, 8]))
    np.testing.assert_array_equal(
        final_sampling_state.has_ended, np.array([True, True])
    )
    outputs = final_sampling_state.get(np.array([True, True]))

    logits, _ = forward_fn(
        params,
        final_sampling_state.tokens[:, :-1],
        segment_ids=jnp.ones_like(final_sampling_state.tokens[:, :-1]),
        segment_positions=jax.lax.broadcasted_iota(
            jnp.int32, final_sampling_state.tokens[:, :-1].shape, 1
        ),
        ragged=False,
    )
    logprobs = sampling_lib.compute_log_likelihood(
        logits, final_sampling_state.tokens[:, 1:]
    )
    logprobs = jnp.pad(logprobs, ((0, 0), (1, 0)))

    for i in range(batch_size):
      input_len = outputs[i]['input_len']
      self.assertEqual(input_len, tokens.lens[i])
      output_slice = slice(input_len, len(outputs[i]['tokens']))

      np.testing.assert_allclose(
          outputs[i]['logprobs'][output_slice],
          logprobs[i][output_slice],
          rtol=5e-3,
      )


if __name__ == '__main__':
  absltest.main()
