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
"""Unit test for model_lib.py."""

import dataclasses
import functools
from typing import cast

from absl.testing import absltest
from absl.testing import parameterized
import einops
import jax
import jax.numpy as jnp
import numpy as np
from simply import config_lib
from simply import model_lib
from simply.utils import common
from simply.utils import optimizers as opt_lib
from simply.utils import pytree
from simply.utils import registry
from simply.utils import sampling_lib
from simply.utils import sharding as sharding_lib
from simply.utils import tokenization


jax.config.update('jax_threefry_partitionable', False)


def lm_test():
  """Returns a test config for TransformerLM."""
  config = config_lib.BaseExperimentConfig()
  config = dataclasses.replace(
      config,
      # Model config
      model_dim=16,
      per_head_dim=4,
      n_heads=8,
      n_layers=2,
      expand_factor=4,
      use_scan=True,
      use_flash_attention=False,
      activation_dtype_name='float32',
      ffn_expand_dim=None,
      # Data config
      num_train_steps=2000,
      batch_size=16,
      vocab_size=64,
      seq_len=10,
      dataset_name='simply_det:lm1b',
      lr=opt_lib.LinearWarmupCosineDecay(
          value=1e-3,
          warmup_steps=10,
          steps_after_decay=10,
          end_decay=0.1,
      ),
      clip_grad_norm=1.0,
      clip_update_norm=1.0,
      # Checkpoint and tensorboard config
      ckpt_interval=10,
      ckpt_max_to_keep=3,
      tb_log_interval=2,
  )
  return dataclasses.replace(config)


class ModelLibTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.config = lm_test()
    self.tfm_lm = model_lib.TransformerLM(self.config)

  def test_forward_pass(self):
    # Generate test data
    seed = 42
    prng_key = jax.random.key(seed)
    config = lm_test()
    config = dataclasses.replace(config, vocab_size=4)
    model = model_lib.TransformerLM(config)
    params = model.init(prng_key)
    batch_size = 2
    batch = jax.random.randint(
        prng_key, shape=(batch_size, 8), minval=0, maxval=config.vocab_size
    )
    logits, _ = model.apply(params, batch)
    logits = np.array(logits)
    print(f'config: {config}')
    print(f'params: {params}')
    # Ensure same behavior as the previous version
    # with `use_combined_qkv=False`.
    expected_logits = [
        [
            [2.3079278, 0.1556056, -1.5357848, 0.41470128],
            [2.307928, 0.15560551, -1.5357848, 0.41470116],
            [2.3079278, 0.15560555, -1.5357842, 0.41470134],
            [2.3079276, 0.15560544, -1.5357848, 0.41470134],
            [0.47243598, 0.7662277, -1.5105368, -0.16814126],
            [0.5188375, -1.5030693, -2.1286428, 0.19518685],
            [0.35362136, 0.6417981, -1.5894638, -0.29224548],
            [0.8483894, 0.42836407, 0.18231091, 0.42470744]
        ],
        [
            [-0.89131826, -0.3329473, 0.5497832, -0.16565481],
            [-0.8913184, -0.33294758, 0.5497832, -0.1656549],
            [0.5598074, 0.40017155, -1.3329586, -0.5925414],
            [-1.2523335, -2.1327658, -0.42040733, 0.64368564],
            [1.018697, -0.5962016, -1.3819861, -0.28444785],
            [1.18975, -0.5388045, -1.4259155, -0.28062773],
            [-0.23136477, 1.4716903, -0.66726977, -0.95534205],
            [0.0925856, -0.13153964, 0.85055715, -0.10545513]
        ]
    ]
    print(
        f'logits-np.array(expected_logits): {logits-np.array(expected_logits)}')
    print(f'logits: {logits}')
    print(f'expected_logits: {expected_logits}')
    self.assertTrue(
        np.allclose(np.array(expected_logits), logits, atol=1e-5))

  def test_identical_inputs_run_to_run_consistency(self):
    """Tests for deterministic outputs across multiple runs with identical inputs."""
    config = lm_test()
    model = model_lib.TransformerLM(config)

    prng_key = jax.random.key(4)
    batch_size, seq_len = 4, 8

    prng_key, subkey = jax.random.split(prng_key)
    x = jax.random.randint(
        subkey,
        shape=(batch_size, seq_len),
        minval=0,
        maxval=config.vocab_size,
    )

    params = model.init(prng_key)

    logits1, _ = model.apply(params, x)
    logits2, _ = model.apply(params, x)

    np.testing.assert_array_equal(logits1, logits2, strict=True)

  def test_identitical_inputs_batch_consistency(self):
    """Tests that identical inputs yield identical outputs regardless of batch position."""
    config = lm_test()
    model = model_lib.TransformerLM(config)

    prng_key = jax.random.key(4)
    batch_size, seq_len = 4, 8

    # Make all batch elements identical.
    prng_key, subkey = jax.random.split(prng_key)
    x = jnp.repeat(
        jax.random.randint(
            subkey,
            shape=(1, seq_len),
            minval=0,
            maxval=config.vocab_size,
        ),
        repeats=batch_size,
        axis=0,
    )

    params = model.init(prng_key)
    logits, _ = model.apply(params, x)

    for i in range(1, batch_size):
      with self.subTest(f'output_{i}'):
        np.testing.assert_allclose(logits[0], logits[i], rtol=1e-5, atol=1e-5)

  def test_backward_pass(self):
    # Generate test data
    seed = 42
    prng_key = jax.random.key(seed)
    config = lm_test()
    config = dataclasses.replace(config, vocab_size=4)
    model = model_lib.TransformerLM(config)
    params = model.init(prng_key)
    batch_size = 2
    inputs = jax.random.randint(
        prng_key, shape=(batch_size, 8), minval=0, maxval=config.vocab_size
    )
    opt = opt_lib.Adam(
        beta1=0.9, beta2=0.95, epsilon=1e-8)
    state = opt.init(params)
    lr = 0.01
    @functools.partial(jax.jit, static_argnames=['add_log_info'])
    def train_one_step_fn(state, batch, lr, add_log_info=False):
      return model_lib.train_one_step(
          state=state,
          batch=batch,
          lr=lr,
          model=model,
          opt=opt,
          clip_grad_norm=config.clip_grad_norm,
          clip_update_norm=config.clip_update_norm,
          clip_local_update_rms=config.clip_local_update_rms,
          weight_decay=config.weight_decay,
          add_log_info=add_log_info,
      )
    batch = {
        'decoder_input_tokens': inputs[:-1],
        'decoder_target_tokens': inputs[1:],
        'decoder_loss_weights': jnp.ones_like(inputs[1:]),
    }
    for _ in range(5):
      _, state, _ = train_one_step_fn(state, batch, lr)
    logits, _ = model.apply(state['params'], inputs)
    logits = np.array(logits)
    expected_logits = [
        [
            [2.2992256, 0.12030537, -1.4335514, 0.4669848],
            [2.2992253, 0.12030543, -1.4335512, 0.46698487],
            [2.2992256, 0.12030561, -1.4335513, 0.46698487],
            [2.2992246, 0.12030512, -1.4335514, 0.46698463],
            [0.5527477, 0.7296491, -1.3306446, -0.20080139],
            [0.5500062, -1.5798444, -2.0209923, 0.16794905],
            [0.39745516, 0.63439167, -1.4578781, -0.32599187],
            [0.7003035, 0.3446952, 0.46150324, 0.44432324]
        ],
        [
            [-0.8991864, -0.3413272, 0.5727337, -0.18962501],
            [-0.8991866, -0.34132737, 0.57273364, -0.18962495],
            [0.5205872, 0.37674674, -1.2782695, -0.5897207],
            [-1.2450306, -2.1522186, -0.40919983, 0.620265],
            [0.9932629, -0.6255955, -1.2935325, -0.2830235],
            [1.1882919, -0.5988288, -1.340022, -0.2543603],
            [-0.24534938, 1.4544814, -0.6483614, -0.99232906],
            [-0.00745952, -0.23477697, 0.95938516, -0.10769912]
        ]
    ]
    self.assertTrue(np.allclose(np.array(expected_logits), logits, atol=1e-5))

  def test_grad_accumulation(self):
    # Generate test data
    seed = 42
    prng_key = jax.random.key(seed)
    config = lm_test()
    config = dataclasses.replace(config, vocab_size=4)
    model = model_lib.TransformerLM(config)
    params = model.init(prng_key)
    batch_size = 4
    inputs = jax.random.randint(
        prng_key, shape=(batch_size, 8), minval=0, maxval=config.vocab_size
    )
    opt = opt_lib.Adam(
        beta1=0.9, beta2=0.95, epsilon=1e-8)
    init_state = opt.init(params)
    lr = 0.01
    @functools.partial(jax.jit, static_argnames=['add_log_info'])
    def train_one_step_fn(state, batch, lr, add_log_info=False):
      return model_lib.train_one_step(
          state=state,
          batch=batch,
          lr=lr,
          model=model,
          opt=opt,
          clip_grad_norm=config.clip_grad_norm,
          clip_update_norm=config.clip_update_norm,
          clip_local_update_rms=config.clip_local_update_rms,
          weight_decay=config.weight_decay,
          add_log_info=add_log_info,
      )
    @functools.partial(jax.jit, static_argnames=['add_log_info'])
    def train_one_step_fn_grad_accum(state, batch, lr, add_log_info=False):
      return model_lib.train_one_step(
          state=state,
          batch=batch,
          lr=lr,
          model=model,
          opt=opt,
          grad_accum_steps=2,
          clip_grad_norm=config.clip_grad_norm,
          clip_update_norm=config.clip_update_norm,
          clip_local_update_rms=config.clip_local_update_rms,
          weight_decay=config.weight_decay,
          add_log_info=add_log_info,
      )
    batch = {
        'decoder_input_tokens': inputs[:, :-1],
        'decoder_target_tokens': inputs[:, 1:],
        'decoder_loss_weights': jnp.ones_like(inputs[:, 1:]),
    }
    state = init_state
    for _ in range(2):
      _, state, _ = train_one_step_fn(state, batch, lr)
    logits, _ = model.apply(state['params'], inputs[:, :-1])
    logits = np.array(logits)

    state = init_state
    for _ in range(2):
      _, state, _ = train_one_step_fn_grad_accum(state, batch, lr)
    logits_grad_accum, _ = model.apply(state['params'], inputs[:, :-1])
    logits_grad_accum = np.array(logits_grad_accum)
    np.testing.assert_allclose(logits, logits_grad_accum, atol=1e-5)

  def test_replace_embeddings(self):
    seq_len, embedding_dim = 8, 10
    orig_embeddings = np.ones((2, seq_len, embedding_dim))
    mask = np.array([
        [0, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 0],
    ])
    replacement_embeddings = 7 * np.ones((2, 5, embedding_dim))
    new_embeddings = self.tfm_lm._replace_embeddings(
        orig_embeddings,
        replacement_embeddings,
        mask,
    )
    expected_new_embeddings = np.repeat(
        np.array([
            [1, 7, 7, 7, 1, 7, 7, 1],
            [1, 1, 7, 1, 7, 7, 1, 1],
        ])[..., np.newaxis],
        embedding_dim,
        axis=-1,
    )
    np.testing.assert_allclose(expected_new_embeddings, new_embeddings)

  def test_dim_annotation(self):
    # Generate test data
    seed = 42
    prng_key = jax.random.key(seed)
    config = lm_test()
    config = dataclasses.replace(config, vocab_size=4)
    model = model_lib.TransformerLM(config)
    params = model.init(prng_key)
    dim_annotations = jax.tree.map(
        lambda x: x.metadata['dim_annotation'],
        params,
        is_leaf=lambda x: isinstance(x, common.AnnotatedArray),
    )
    expected_dim_annotations = {
        'block_0': {
            'attn': {
                'k_proj': {'w': 'ioo'},
                'o_proj': {'w': 'oii'},
                'per_dim_scale': {'scale': 'h'},
                'q_proj': {'w': 'ioo'},
                'v_proj': {'w': 'ioo'},
            },
            'ffn': {
                'ffn_0': {'b': 'h', 'w': 'io'},
                'ffn_0_gate': {'b': 'h', 'w': 'io'},
                'ffn_1': {'b': 'h', 'w': 'io'},
            },
            'post_ln_0': {'scale': 'h'},
            'post_ln_1': {'scale': 'h'},
            'pre_ln_0': {'scale': 'h'},
            'pre_ln_1': {'scale': 'h'},
        },
        'block_1': {
            'attn': {
                'k_proj': {'w': 'ioo'},
                'o_proj': {'w': 'oii'},
                'per_dim_scale': {'scale': 'h'},
                'q_proj': {'w': 'ioo'},
                'v_proj': {'w': 'ioo'},
            },
            'ffn': {
                'ffn_0': {'b': 'h', 'w': 'io'},
                'ffn_0_gate': {'b': 'h', 'w': 'io'},
                'ffn_1': {'b': 'h', 'w': 'io'},
            },
            'post_ln_0': {'scale': 'h'},
            'post_ln_1': {'scale': 'h'},
            'pre_ln_0': {'scale': 'h'},
            'pre_ln_1': {'scale': 'h'},
        },
        'embed_linear': {'b': '.', 'w': '.i'},
        'final_ln': {'scale': 'h'},
    }
    print('=' * 50)
    print(dim_annotations)
    print('=' * 50)
    self.assertEqual(dim_annotations, expected_dim_annotations)

  @parameterized.parameters(
      {'activation_dtype_name': 'float8_e4m3fn'},
      {'activation_dtype_name': 'float8_e5m2'},
      {'activation_dtype_name': 'int8'},
      {'activation_dtype_name': 'bfloat16'},
      {'activation_dtype_name': 'float16'},
      {'activation_dtype_name': 'float32'},
  )
  def test_mixed_precision(self, activation_dtype_name):
    # Generate test data
    seed = 42
    prng_key = jax.random.key(seed)
    config = lm_test()
    new_config = dataclasses.replace(
        config, activation_dtype_name=activation_dtype_name)
    tfm_lm = model_lib.TransformerLM(new_config)
    params = tfm_lm.init(prng_key)
    batch_size = 8
    batch = jax.random.randint(
        prng_key, shape=(batch_size, self.config.seq_len),
        minval=0, maxval=self.config.vocab_size)
    logits, _ = tfm_lm.apply(params, batch)
    self.assertEqual(logits.dtype, jnp.dtype(activation_dtype_name))

  @parameterized.parameters(
      {'activation_dtype_name': 'float8_e4m3fn'},
      {'activation_dtype_name': 'float8_e5m2'},
      {'activation_dtype_name': 'int8'},
      {'activation_dtype_name': 'bfloat16'},
      {'activation_dtype_name': 'float16'},
      {'activation_dtype_name': 'float32'},
  )
  def test_quantization(self, activation_dtype_name):
    # Generate test data
    seed = 42
    prng_key = jax.random.key(seed)
    config = lm_test()
    new_config = dataclasses.replace(
        config, activation_dtype_name=activation_dtype_name)
    tfm_lm = model_lib.TransformerLM(new_config)
    params = tfm_lm.init(prng_key)
    batch_size = 8
    quant_params = model_lib.quantize_tfm_params(params)
    batch = jax.random.randint(
        prng_key, shape=(batch_size, self.config.seq_len),
        minval=0, maxval=self.config.vocab_size)
    logits, _ = tfm_lm.apply(quant_params, batch)
    self.assertEqual(logits.dtype, jnp.dtype(activation_dtype_name))

  def test_create_mask(self):
    segment_positions = np.array([
        [0, 1, 2, 3, 0, 1, 2, 0, 0, 0],
        [0, 1, 2, 0, 1, 2, 3, 4, 0, 0]])
    segment_ids = np.array([
        [1, 1, 1, 1, 2, 2, 2, 0, 0, 0],
        [1, 1, 1, 2, 2, 2, 2, 2, 0, 0]
    ])
    mask = model_lib.create_mask(
        segment_positions=segment_positions,
        kv_segment_positions=segment_positions,
        segment_ids=segment_ids,
        kv_segment_ids=segment_ids,
    )
    b, seq_len, *_ = segment_positions.shape
    assert mask.shape == (b, seq_len, seq_len)
    for i in range(b):
      for j in range(seq_len):
        for k in range(seq_len):
          same_segment = (segment_ids[i, j] == segment_ids[i, k])
          causal = (segment_positions[i, j] >= segment_positions[i, k])
          self.assertEqual(
              mask[i, j, k], same_segment and causal)

  def test_create_mask_with_window_size(self):
    segment_ids = np.array([
        [1, 1, 1, 1, 2, 2],
        [1, 1, 1, 2, 2, 2],
    ])
    segment_positions = np.array([
        [0, 1, 2, 3, 0, 1],
        [0, 1, 2, 0, 1, 2],
    ])
    mask = model_lib.create_mask(
        segment_positions=segment_positions,
        kv_segment_positions=segment_positions,
        segment_ids=segment_ids,
        kv_segment_ids=segment_ids,
        window_size=1,
    )
    np.testing.assert_equal(
        mask,
        np.array([
            [
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 1],
            ],
            [
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 1],
            ],
        ]),
    )

    segment_positions = np.arange(4).reshape(1, -1)
    segment_ids = np.ones((1, 4))
    mask = model_lib.create_mask(
        segment_positions=segment_positions,
        kv_segment_positions=segment_positions,
        segment_ids=segment_ids,
        kv_segment_ids=segment_ids,
        window_size=2,
    )
    np.testing.assert_equal(
        mask,
        np.array([[
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
        ]]),
    )

  def test_create_mask_decode(self):
    segment_positions = np.array([[3, 4, 5, 6, 7, 0, 0]])
    segment_ids = np.array([[1, 1, 1, 1, 1, 0, 0]])
    mask = model_lib.create_mask(
        segment_positions=np.array([[7]]),
        kv_segment_positions=segment_positions,
        segment_ids=np.array([[1]]),
        kv_segment_ids=segment_ids,
    )
    self.assertEqual(
        mask.tolist(), [[[True, True, True, True, True, False, False]]]
    )

  def test_chunked_local_attn(self):
    seq_len = 9
    window_size = 3
    q = np.random.randn(2, seq_len, 2, 3, 4)
    k = np.random.randn(2, seq_len, 3, 4)
    v = np.random.randn(2, seq_len, 3, 4)
    segment_ids = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 2, 2, 2, 2, 2, 3, 3],
    ])
    segment_positions = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [0, 1, 0, 1, 2, 3, 4, 0, 1],
    ])
    mask = model_lib.create_mask(
        segment_positions=segment_positions,
        kv_segment_positions=segment_positions,
        segment_ids=segment_ids,
        kv_segment_ids=segment_ids,
        window_size=window_size,
    )
    # Add group and head axis.
    mask = jnp.expand_dims(mask, range(1, len(q.shape) - len(mask.shape) + 1))
    output1, _ = model_lib.attn(q, k, v, mask, dtype=jnp.float32)
    output2 = model_lib.chunked_local_attn(
        q, k, v, mask, window_size=window_size, dtype=jnp.float32
    )
    self.assertEqual(output1.shape, q.shape)
    self.assertTrue(np.allclose(output1, output2, atol=1e-5))

  @parameterized.parameters(
      {'use_scan': True},
      {'use_scan': False},
  )
  def test_logits_with_kv_cache(self, use_scan):
    example = [0, 23, 32, 7, 10]
    batch = 2

    def compute_logits_decoding_from_k(model, params, k):
      inputs = np.tile(np.array([example[:k]]), (batch, 1))
      segment_positions = np.tile(np.expand_dims(np.arange(k), 0), (batch, 1))
      apply_fn = jax.jit(model.apply)
      init_logits, extra_output = apply_fn(
          params,
          inputs,
          segment_positions=segment_positions,
          extra_inputs=dict(prefill_position=k),
      )
      decode_state = extra_output['decode_state']
      decode_state = model_lib.pad_decode_state_to(decode_state, len(example))
      all_logits = [init_logits]
      for i in range(k, len(example)):
        inputs = np.tile(np.array([[example[i]]]), (batch, 1))
        segment_positions = np.tile(np.array([[i]]), (batch, 1))
        logits, extra_output = apply_fn(
            params, inputs, decode_state=decode_state,
            segment_positions=segment_positions)
        decode_state = extra_output['decode_state']
        all_logits.append(logits)
      result = np.concatenate(all_logits, axis=1)
      return result

    test_config = dataclasses.replace(self.config, use_scan=use_scan)
    model = model_lib.TransformerLM(test_config)
    params = model.init(jax.random.key(0))
    output_logits = compute_logits_decoding_from_k(model, params, 1)
    target_logits = compute_logits_decoding_from_k(model, params, len(example))
    self.assertEqual(output_logits.shape, (batch, len(example), 64))
    # This test works on CPU but will fail on GPU/TPU due to nonassociativity
    # and different calculation order in the reduction in Layernorm.
    # This test would also fail on cpu for bfloat16, but works for
    # float16, float32 and int8.
    self.assertTrue(np.allclose(output_logits, target_logits, atol=1e-5))

  def test_tree_norm(self):
    tree = {'a': np.array([3, 4]), 'b': 3, 'c': [{'d': np.array([4])}]}
    result = model_lib.tree_norm(tree)
    self.assertEqual(np.square(result), 50)

  def test_flatten_dict(self):
    tree = {'a': 1, 'b': 2, 'c': {'d': 3},
            'e': {'f': 4, 'g': {'h': 5}}}
    result = {'a': 1, 'b': 2, 'c/d': 3, 'e/f': 4, 'e/g/h': 5}
    self.assertEqual(result, model_lib.flatten_dict(tree))

  def test_lm_interface_generate(self):
    vocab = tokenization.TestVocab([str(i) for i in range(60)])
    prng_key = jax.random.key(0)
    params = self.tfm_lm.init(prng_key)
    max_decode_steps = 10
    sampling_params = model_lib.SamplingParams(
        top_k=-1, top_p=1.0, temperature=1.0,
        max_decode_steps=max_decode_steps)
    lm_interface = model_lib.LMInterface(self.tfm_lm, params, vocab)
    outputs = lm_interface.generate(
        input_text='1 2 3',
        prng_key=jax.random.key(seed=25),
        sampling_params=sampling_params,
    )
    outputs = cast(list[model_lib.SamplingOutput], outputs)
    for so in outputs:
      self.assertLen(so.output_token_ids, max_decode_steps)
      self.assertLen(so.input_token_ids, 4)
      self.assertLen(so.input_token_scores, 3)

  def test_lm_interface_batch(self):
    vocab = tokenization.TestVocab([str(i) for i in range(20)])
    prng_key = jax.random.key(0)
    params = self.tfm_lm.init(prng_key)
    max_decode_steps = 4
    sampling_params = model_lib.SamplingParams(
        top_k=-1,
        top_p=1.0,
        temperature=0.0,
        max_decode_steps=max_decode_steps,
        prefill_size=10,
    )
    lm_interface = model_lib.LMInterface(self.tfm_lm, params, vocab)
    outputs1 = lm_interface.generate(
        input_text='1 2 3',
        prng_key=jax.random.key(seed=25),
        sampling_params=sampling_params,
    )
    outputs2 = lm_interface.generate(
        input_text='1 2 3',
        prng_key=jax.random.key(seed=25),
        sampling_params=sampling_params,
        batch_size=2,
    )
    for so1, so2 in zip(outputs1, outputs2):
      self.assertEqual(so1.input_token_ids, so2.input_token_ids)
      self.assertEqual(so1.output_token_ids, so2.output_token_ids)
      self.assertEqual(so1.input_token_scores, so2.input_token_scores)
      self.assertEqual(so1.output_token_scores, so2.output_token_scores)

  def test_lm_interface_generate_without_scoring(self):
    vocab = tokenization.TestVocab([str(i) for i in range(10)])
    prng_key = jax.random.key(0)
    params = self.tfm_lm.init(prng_key)
    max_decode_steps = 10
    sampling_params = model_lib.SamplingParams(
        top_k=-1, top_p=1.0, temperature=1.0,
        max_decode_steps=max_decode_steps)
    lm_interface = model_lib.LMInterface(self.tfm_lm, params, vocab)
    outputs = lm_interface.generate(
        input_text='1 2 3',
        prng_key=jax.random.key(seed=25),
        sampling_params=sampling_params,
        scoring_inputs=False,
    )
    outputs = cast(list[model_lib.SamplingOutput], outputs)
    for so in outputs:
      scores = lm_interface.score_tokens(
          so.input_token_ids + so.output_token_ids
      )
      np.testing.assert_allclose(
          so.output_token_scores, scores[-len(so.output_token_ids) :], rtol=1e-5
      )

  @parameterized.named_parameters(
      dict(testcase_name='scan_prefill_3', use_scan=True, prefill_size=3),
      dict(testcase_name='scan_prefill_4', use_scan=True, prefill_size=4),
      dict(testcase_name='noscan_prefill_4', use_scan=False, prefill_size=4),
      dict(testcase_name='scan_prefill_5', use_scan=True, prefill_size=5),
      dict(testcase_name='scan_prefill_6', use_scan=False, prefill_size=6),
  )
  def test_lm_interface_generate_with_local_state(self, use_scan, prefill_size):
    vocab = tokenization.TestVocab([str(i) for i in range(15)])
    prng_key = jax.random.key(0)
    config = dataclasses.replace(
        cast(model_lib.SimplyConfig, self.config),
        use_scan=use_scan,
        block_attn_pattern=('local', 'global'),
        n_layers=3,
        n_heads=3,
        window_size=4,
        vocab_size=20,
    )
    tfm_lm = model_lib.TransformerLM(config)
    params = tfm_lm.init(prng_key)
    sampling_params = model_lib.SamplingParams(
        max_decode_steps=0, intermediate_decode_steps=1
    )
    lm_interface = model_lib.LMInterface(
        tfm_lm, params, vocab, default_sampling_params=sampling_params
    )
    text = '1 2 3 4 5 6 7'
    tokens = np.array([vocab.bos_id] + vocab.encode(text)).reshape([1, -1])
    token_scores = lm_interface.score_tokens(tokens=tokens)
    with jax.disable_jit():
      outputs = lm_interface.generate(
          input_text=text,
          prng_key=jax.random.key(seed=25),
          prefill_size=prefill_size,
      )
    outputs = cast(list[model_lib.SamplingOutput], outputs)
    for so in outputs:
      self.assertEqual(so.output_token_ids, [])
      self.assertLen(so.input_token_ids, 8)
      np.testing.assert_allclose(so.input_token_scores, token_scores, rtol=1e-5)

  def test_lm_interface_score(self):
    vocab = tokenization.TestVocab([str(i) for i in range(60)])
    prng_key = jax.random.key(0)
    params = self.tfm_lm.init(prng_key)
    scoring_params = model_lib.ScoringParams(
        temperature=1.0, top_k=-1, top_p=1.0
    )
    lm_interface = model_lib.LMInterface(self.tfm_lm, params, vocab)
    scoring_output = lm_interface.score(
        input_text='1 2 3', output_text='4 5', scoring_params=scoring_params
    )
    self.assertLen(scoring_output.input_token_scores, 3)  # 3 input tokens
    self.assertLen(scoring_output.output_token_scores, 2)  # 2 output tokens

  def test_lm_interface_score_tokens(self):
    vocab = tokenization.TestVocab([str(i) for i in range(60)])
    prng_key = jax.random.key(0)
    params = self.tfm_lm.init(prng_key)
    scoring_params = model_lib.ScoringParams(
        temperature=1.0, top_k=-1, top_p=1.0
    )
    lm_interface = model_lib.LMInterface(self.tfm_lm, params, vocab)
    text = '1 2 3'
    tokens = np.array([vocab.bos_id] + vocab.encode(text)).reshape([1, -1])
    token_scores = lm_interface.score_tokens(
        tokens=tokens, scoring_params=scoring_params
    )
    self.assertLen(token_scores, 3)  # 1 BOS + 3 encoded tokens, thus 3 scores

  def test_continue_decoding(self):
    vocab = tokenization.TestVocab([str(i) for i in range(60)])
    prng_key = jax.random.key(0)
    params = self.tfm_lm.init(prng_key)
    max_decode_steps = 10
    intermediate_decode_steps = 5
    sampling_prng_key = jax.random.key(seed=25)
    sampling_params_1 = model_lib.SamplingParams(
        top_k=-1, top_p=1.0, temperature=1.0,
        max_decode_steps=max_decode_steps)
    lm_interface_1 = model_lib.LMInterface(self.tfm_lm, params, vocab)
    outputs_1 = lm_interface_1.generate(
        input_text='1 2 3',
        prng_key=sampling_prng_key.copy(),
        sampling_params=sampling_params_1,
    )
    outputs_1 = cast(list[model_lib.SamplingOutput], outputs_1)

    sampling_params_2 = model_lib.SamplingParams(
        top_k=-1, top_p=1.0, temperature=1.0,
        max_decode_steps=max_decode_steps,
        intermediate_decode_steps=intermediate_decode_steps)
    lm_interface_2 = model_lib.LMInterface(self.tfm_lm, params, vocab)
    outputs_2 = lm_interface_2.generate(
        input_text='1 2 3',
        prng_key=sampling_prng_key.copy(),
        sampling_params=sampling_params_2,
    )
    outputs_2 = cast(list[model_lib.SamplingOutput], outputs_2)

    for so1, so2 in zip(outputs_1, outputs_2):
      # Note that this works on CPU but may fail on TPU or GPU.
      self.assertEqual(so1.output_token_ids, so2.output_token_ids)
      self.assertAlmostEqual(
          np.asarray(so1.avg_output_score, np.float32),
          np.asarray(so2.avg_output_score, np.float32),
          delta=1e-6,
      )

  @parameterized.named_parameters(
      dict(testcase_name='params_1', temperature=0.0, top_k=-1, top_p=1.0),
      dict(testcase_name='params_2', temperature=0.5, top_k=-1, top_p=1.0),
      dict(testcase_name='params_3', temperature=1.0, top_k=-1, top_p=1.0),
      dict(testcase_name='params_4', temperature=1.0, top_k=10, top_p=1.0),
      dict(testcase_name='params_5', temperature=1.0, top_k=-1, top_p=0.9),
  )
  def test_sampling_output_logprobs(
      self, temperature: float, top_k: int, top_p: float
  ):
    vocab_size = 60
    vocab = tokenization.TestVocab([str(i) for i in range(vocab_size)])
    prng_key = jax.random.key(0)
    params = self.tfm_lm.init(prng_key)
    max_decode_steps = 64
    input_len = 128
    input_text = ' '.join([str(i % vocab_size) for i in list(range(input_len))])
    sampling_params = model_lib.SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_decode_steps=max_decode_steps,
    )
    lm_interface = model_lib.LMInterface(self.tfm_lm, params, vocab)

    outputs = lm_interface.generate(
        input_text=input_text,
        prng_key=jax.random.key(seed=25),
        sampling_params=sampling_params,
    )
    outputs = cast(list[model_lib.SamplingOutput], outputs)

    for so in outputs:
      self.assertLen(so.input_token_ids, input_len + 1)
      self.assertLen(so.output_token_ids, max_decode_steps)

      all_tokens = np.array(
          [so.input_token_ids + so.output_token_ids])
      logits, _ = self.tfm_lm.apply(params, all_tokens[:, :-1])

      token_logprobs = sampling_lib.compute_log_likelihood(
          logits,
          all_tokens[:, 1:],
          temperature=sampling_params.temperature,
          top_k=sampling_params.top_k,
          top_p=sampling_params.top_p,
      )

      output_len = len(so.output_token_ids)
      np.testing.assert_allclose(
          np.asarray([so.output_token_logprobs]),
          token_logprobs[:, -output_len:],
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          so.sum_output_logprob,
          np.sum(token_logprobs[:, -output_len:]),
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          so.avg_output_logprob,
          np.mean(token_logprobs[:, -output_len:]),
          rtol=1e-5,
      )

  @parameterized.named_parameters(
      dict(testcase_name='params_1', temperature=0.0, top_k=-1, top_p=1.0),
      dict(testcase_name='params_2', temperature=0.5, top_k=-1, top_p=1.0),
      dict(testcase_name='params_3', temperature=1.0, top_k=-1, top_p=1.0),
      dict(testcase_name='params_4', temperature=1.0, top_k=10, top_p=1.0),
      dict(testcase_name='params_5', temperature=1.0, top_k=-1, top_p=0.9),
  )
  def test_sampling_token_scores(
      self, temperature: float, top_k: int, top_p: float
  ):
    vocab_size = 60
    vocab = tokenization.TestVocab([str(i) for i in range(vocab_size)])
    prng_key = jax.random.key(0)
    params = self.tfm_lm.init(prng_key)
    max_decode_steps = 64
    input_len = 128
    input_text = ' '.join([str(i % vocab_size) for i in list(range(input_len))])
    sampling_params = model_lib.SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_decode_steps=max_decode_steps,
    )
    scoring_params = model_lib.ScoringParams(
        temperature=1.0, top_k=-1, top_p=1.0
    )
    lm_interface = model_lib.LMInterface(self.tfm_lm, params, vocab)

    outputs = lm_interface.generate(
        input_text=input_text,
        prng_key=jax.random.key(seed=25),
        sampling_params=sampling_params,
        scoring_params=scoring_params,
    )
    outputs = cast(list[model_lib.SamplingOutput], outputs)

    for so in outputs:
      self.assertLen(so.input_token_ids, input_len + 1)
      self.assertLen(so.output_token_ids, max_decode_steps)

      all_tokens = np.array(
          [so.input_token_ids + so.output_token_ids])
      logits, _ = self.tfm_lm.apply(
          params, all_tokens[:, :-1], segment_ids=None, segment_positions=None
      )
      token_ll = sampling_lib.compute_log_likelihood(
          logits,
          all_tokens[:, 1:],
          temperature=scoring_params.temperature,
          top_k=scoring_params.top_k,
          top_p=scoring_params.top_p,
      )

      # Note that this test works on CPU but may fail on TPU or GPU.
      np.testing.assert_allclose(
          np.asarray([so.input_token_scores]),
          token_ll[:, : len(so.input_token_scores)],
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          so.sum_input_score,
          np.sum(token_ll[:, : len(so.input_token_scores)]),
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          so.avg_input_score,
          np.mean(token_ll[:, : len(so.input_token_scores)]),
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          np.asarray([so.output_token_scores]),
          token_ll[:, len(so.input_token_scores) :],
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          so.sum_output_score,
          np.sum(token_ll[:, len(so.input_token_scores) :]),
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          so.avg_output_score,
          np.mean(token_ll[:, len(so.input_token_scores) :]),
          rtol=1e-5,
      )

  def test_sampling_max_decode_steps_equals_prefill_size(self):
    vocab_size = 30
    vocab = tokenization.TestVocab([str(i) for i in range(vocab_size)])
    prng_key = jax.random.PRNGKey(0)
    params = self.tfm_lm.init(prng_key)
    max_decode_steps = 32
    input_len = 16
    input_text = ' '.join([str(i % vocab_size) for i in list(range(input_len))])
    sampling_params = model_lib.SamplingParams(
        max_decode_steps=max_decode_steps
    )
    lm_interface = model_lib.LMInterface(self.tfm_lm, params, vocab)

    outputs = lm_interface.generate(
        input_text=input_text,
        prng_key=jax.random.key(seed=25),
        prefill_size=max_decode_steps,
        sampling_params=sampling_params,
    )
    outputs = cast(list[model_lib.SamplingOutput], outputs)

    for so in outputs:
      self.assertLen(so.input_token_ids, input_len + 1)
      self.assertLen(so.output_token_ids, max_decode_steps)

      all_tokens = np.array(
          [so.input_token_ids + so.output_token_ids])
      logits, _ = self.tfm_lm.apply(
          params, all_tokens[:, :-1], segment_ids=None, segment_positions=None
      )
      token_ll = sampling_lib.compute_log_likelihood(
          logits,
          all_tokens[:, 1:],
          temperature=sampling_params.temperature,
          top_k=sampling_params.top_k,
          top_p=sampling_params.top_p,
      )

      # Note that this test works on CPU but may fail on TPU or GPU.
      np.testing.assert_allclose(
          np.asarray([so.input_token_scores]),
          token_ll[:, : len(so.input_token_scores)],
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          np.asarray([so.output_token_scores]),
          token_ll[:, len(so.input_token_scores):],
          rtol=1e-5,
      )

  def test_batch_sampling(self):
    vocab_size = 10
    vocab = tokenization.TestVocab([str(i) for i in range(vocab_size)])
    prng_key = jax.random.PRNGKey(0)
    params = self.tfm_lm.init(prng_key)
    max_decode_steps = 4
    batch_size = 3
    base_input_len = 2
    input_texts = []
    for i in range(batch_size):
      input_texts.append(
          ' '.join([
              str((j + i) % vocab_size) for j in list(range(base_input_len + i))
          ])
      )
    sampling_params = model_lib.SamplingParams(
        max_decode_steps=max_decode_steps,
        num_samples=2,
        sort_by='avg_output_logprob',
    )
    lm_interface = model_lib.LMInterface(self.tfm_lm, params, vocab)

    outputs = lm_interface.generate(
        input_text=input_texts,
        prng_key=jax.random.key(seed=25),
        sampling_params=sampling_params,
    )
    outputs = cast(list[list[model_lib.SamplingOutput]], outputs)

    for i, batch in enumerate(outputs):
      for so in batch:
        self.assertLen(so.input_token_ids, base_input_len + i + 1)
        self.assertLen(so.output_token_ids, max_decode_steps)

        all_tokens = np.array([so.input_token_ids + so.output_token_ids])
        logits, _ = self.tfm_lm.apply(params, all_tokens[:, :-1])
        token_ll = sampling_lib.compute_log_likelihood(
            logits,
            all_tokens[:, 1:],
            temperature=sampling_params.temperature,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
        )

        # Note that this test works on CPU but may fail on TPU or GPU.
        np.testing.assert_allclose(
            np.asarray([so.input_token_scores]),
            token_ll[:, : len(so.input_token_scores)],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray([so.output_token_scores]),
            token_ll[:, len(so.input_token_scores) :],
            rtol=1e-5,
        )

  def test_batch_sampling_with_max_seq_len(self):
    vocab_size = 10
    vocab = tokenization.TestVocab([str(i) for i in range(vocab_size)])
    prng_key = jax.random.PRNGKey(0)
    params = self.tfm_lm.init(prng_key)
    max_decode_steps = 4
    max_seq_len = 7
    batch_size = 3
    base_input_len = 2
    input_texts = []
    max_input_len = 3
    for i in range(batch_size):
      # input_len = 2, 3, 4
      # output_len = 6, 7, 7
      input_texts.append(
          ' '.join([
              str((j + i) % vocab_size) for j in list(range(base_input_len + i))
          ])
      )
    sampling_params = model_lib.SamplingParams(
        max_decode_steps=max_decode_steps,
        max_seq_len=max_seq_len,
        max_input_len=max_input_len,
        sort_by='avg_output_logprob',
    )
    lm_interface = model_lib.LMInterface(self.tfm_lm, params, vocab)
    input_as_chunks = lm_interface.input_processor.input_as_chunks
    input_tokens_no_truncation = [
        lm_interface.input_processor.encode(
            input_as_chunks(text)).tokens for text in input_texts]

    outputs = lm_interface.generate(
        input_text=input_texts,
        prng_key=jax.random.key(seed=25),
        sampling_params=sampling_params,
    )
    outputs = cast(list[list[model_lib.SamplingOutput]], outputs)

    for i, batch in enumerate(outputs):
      for so in batch:
        # Check whether input tokens are truncated correctly.
        self.assertLen(
            so.input_token_ids, min(max_input_len, base_input_len + i + 1))
        self.assertEqual(
            so.input_token_ids,
            input_tokens_no_truncation[i][-max_input_len:])
        self.assertLen(
            so.output_token_ids,
            min(max_decode_steps, max_seq_len - len(so.input_token_ids)),
        )

        all_tokens = np.array([so.input_token_ids + so.output_token_ids])
        logits, _ = self.tfm_lm.apply(params, all_tokens[:, :-1])
        token_ll = sampling_lib.compute_log_likelihood(
            logits,
            all_tokens[:, 1:],
            temperature=sampling_params.temperature,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
        )

        # Note that this test works on CPU but may fail on TPU or GPU.
        np.testing.assert_allclose(
            np.asarray([so.input_token_scores]),
            token_ll[:, : len(so.input_token_scores)],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray([so.output_token_scores]),
            token_ll[:, len(so.input_token_scores) :],
            rtol=1e-5,
        )

  def test_dump_load(self):
    py = pytree.dump_dataclasses(self.tfm_lm)
    lm_loaded = pytree.load_dataclasses(py)
    loaded_py = pytree.dump_dataclasses(lm_loaded)
    self.assertEqual(py, loaded_py)

  def test_segment_info(self):
    prng_key = jax.random.key(0)
    params = self.tfm_lm.init(prng_key)
    batch_size = 4
    seq_len = 128
    batch = {
        'decoder_input_tokens': np.random.randint(
            0, self.config.vocab_size, size=(batch_size, seq_len)),
        'decoder_target_tokens': np.random.randint(
            0, self.config.vocab_size, size=(batch_size, seq_len)),
        'decoder_loss_weights': np.ones(shape=(batch_size, seq_len)),
        'decoder_segment_ids': np.ones(shape=(batch_size, seq_len)),
        'decoder_positions': einops.repeat(
            np.arange(start=0, stop=seq_len), 'l -> b l', b=batch_size),
    }
    logits1, _ = self.tfm_lm.apply(params, batch['decoder_input_tokens'])
    logits2, _ = self.tfm_lm.apply(
        params,
        batch['decoder_input_tokens'],
        segment_ids=batch['decoder_segment_ids'],
        segment_positions=batch['decoder_positions'],
    )
    self.assertTrue(np.allclose(logits1, logits2))

  @parameterized.named_parameters(
      dict(testcase_name='expand_dim_10', ffn_expand_dim=10),
      dict(testcase_name='expand_dim_5', ffn_expand_dim=5),
  )
  def test_expand_dim_in_ffn(self, ffn_expand_dim: int):
    new_config = dataclasses.replace(
        lm_test(), ffn_expand_dim=ffn_expand_dim)
    self.tfm_lm = model_lib.TransformerLM(
        new_config, config_lib.gspmd_sharding()
    )
    for tfm_block in self.tfm_lm.blocks:
      self.assertEqual(tfm_block.expand_dim, ffn_expand_dim)

  def test_get_scaling_info(self):
    scaling_info_dict = model_lib.get_scaling_info(
        self.config, add_attn_flops=True)
    self.assertGreater(scaling_info_dict['num_flops'],
                       scaling_info_dict['num_attn_flops'])
    self.assertGreater(scaling_info_dict['num_attn_flops'], 0)

  def test_n_blocks(self):
    lm = model_lib.TransformerLM(self.config)
    self.assertLen(lm.blocks, 2)

    lm = model_lib.TransformerLM(
        dataclasses.replace(
            cast(config_lib.SimplyConfig, self.config),
            n_layers=3,
            block_attn_pattern=('global', 'local'),
        )
    )
    self.assertLen(lm.blocks, 3)


class TestNpArrayQuantizer(tokenization.SimplyVocab[np.ndarray]):

  vocab_size: int
  feature_dim: int

  def __init__(
      self,
      vocab_list,
      bos_id: int = 2,
      eos_id: int = -1,
      pad_id: int = 0,
      unk_id: int = 3,
      vocab_size: int = 64,
      feature_dim: int = 4,
  ):
    self.bos_id = bos_id
    self.eos_id = eos_id
    self.pad_id = pad_id
    self.unk_id = unk_id
    self.vocab_size = vocab_size
    self.feature_dim = feature_dim
    start_id = max(unk_id, pad_id, eos_id, bos_id) + 1
    self._vocab_dict = dict(
        [(w, (i + start_id)) for i, w in enumerate(vocab_list)]
    )
    self._rev_vocab_dict = {v: k for k, v in self._vocab_dict.items()}

  def encode(self, raw_sequence: np.ndarray) -> list[int]:
    assert raw_sequence.ndim == 2
    assert raw_sequence.shape[1] == self.feature_dim
    tokens = []
    for i in range(raw_sequence.shape[0]):
      w = np.sum(raw_sequence[i]) % self.vocab_size
      tokens.append(self._vocab_dict.get(w, self.unk_id))
    return tokens

  def decode(self, token_ids: list[int]) -> np.ndarray:
    output_raw = []
    for i in token_ids:
      output_raw.append(self._rev_vocab_dict.get(i, -1))
    output_raw = np.array(output_raw, dtype=np.int32)[None, :]
    output_raw = np.repeat(output_raw, self.feature_dim, axis=1)
    return output_raw


def simple_moe(
    params, inputs, inputs_mask,
    num_experts_per_token, num_experts,
    ffn_activation,
    use_gated_activation_in_ffn,
    activation_dtype):
  # Use a naive dense MoE implementation to check equivalence.
  params = model_lib.get_raw_arrays(params)
  router_w = jnp.asarray(params['router']['w'], activation_dtype)
  router_logits = jnp.einsum('ie,bsi->bse', router_w, inputs)
  router_probs = jax.nn.softmax(router_logits, axis=-1)
  _, selected_indices = jax.lax.top_k(
      router_probs, k=num_experts_per_token
  )
  mask = jnp.sum(
      jax.nn.one_hot(selected_indices, num_experts, dtype=router_probs.dtype),
      axis=-2,
  )
  effective_router_probs = router_probs * mask
  if num_experts_per_token > 1:
    # Re-normalize the effective router probs.
    effective_router_probs = effective_router_probs / jnp.sum(
        effective_router_probs, axis=-1, keepdims=True)
  effective_router_probs = jnp.asarray(
      effective_router_probs, dtype=activation_dtype)
  ffn0_w = jnp.asarray(params['ffn_0']['w'], activation_dtype)
  projected_inputs = jnp.einsum('eio,bsi->ebso', ffn0_w, inputs)
  activation_fn = registry.FunctionRegistry.get(ffn_activation)
  if use_gated_activation_in_ffn:
    ffn0_gate_w = jnp.asarray(
        params['ffn_0_gate']['w'], activation_dtype)
    gate = jnp.einsum('eio,bsi->ebso', ffn0_gate_w, inputs)
    middle = (
        jnp.asarray(activation_fn(gate), activation_dtype)
        * projected_inputs
    )
  else:
    middle = jnp.asarray(
        activation_fn(projected_inputs), activation_dtype
    )
  ffn1_w = jnp.asarray(params['ffn_1']['w'], activation_dtype)
  expert_outputs = jnp.einsum('eio,ebsi->ebso', ffn1_w, middle)
  outputs = jnp.einsum(
      'ebsd,bse->bsd', expert_outputs, effective_router_probs
  )
  outputs = outputs * inputs_mask[..., None]
  return outputs


class MoETest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='_no_gate_dropless',
          use_gated_activation_in_ffn=False,
          num_experts=8,
          expert_capacity_factor=None,
          num_experts_per_token=2,
      ),
      dict(
          testcase_name='_gate_dropless',
          use_gated_activation_in_ffn=True,
          num_experts=6,
          expert_capacity_factor=None,
          num_experts_per_token=2,
      ),
      dict(
          testcase_name='_gate_dropless_single_expert',
          use_gated_activation_in_ffn=True,
          num_experts=1,
          expert_capacity_factor=None,
          num_experts_per_token=1,
      ),
      dict(
          testcase_name='_gate_dropless_all_experts',
          use_gated_activation_in_ffn=True,
          num_experts=4,
          expert_capacity_factor=None,
          num_experts_per_token=4,
      ),
      dict(
          testcase_name='_no_gate',
          use_gated_activation_in_ffn=False,
          num_experts=8,
          expert_capacity_factor=5,
          num_experts_per_token=2,
      ),
      dict(
          testcase_name='_gate',
          use_gated_activation_in_ffn=True,
          num_experts=6,
          expert_capacity_factor=5,
          num_experts_per_token=2,
      ),
      dict(
          testcase_name='_gate_single_expert',
          use_gated_activation_in_ffn=True,
          num_experts=1,
          expert_capacity_factor=5,
          num_experts_per_token=1,
      ),
  )
  def test_moe_feed_forward_equivalence(
      self, use_gated_activation_in_ffn, num_experts, expert_capacity_factor,
      num_experts_per_token=2, activation_dtype='bfloat16',
  ):
    sharding_config = config_lib.moe_sharding()
    sharding_lib.set_default_mesh_shape(
        mesh_shape=(1, 1, 1, 1),
        axis_names=sharding_config.mesh_axis_names)
    batch_size, seq_len, model_dim, expand_factor = 2, 4, 4, 2
    segment_ids = jnp.array([[1, 2, 3, 0], [1, 0, 0, 1]])
    key = jax.random.PRNGKey(0)
    input_key, prng_key = jax.random.split(key)
    inputs = jax.random.normal(
        input_key, shape=(batch_size, seq_len, model_dim),
        dtype=activation_dtype
    )
    inputs_mask = segment_ids != 0

    moe_ffn = model_lib.MoEFeedForward(
        model_dim=model_dim,
        expand_factor=expand_factor,
        sharding_config=sharding_config,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        expert_capacity_factor=expert_capacity_factor,
        ffn_use_bias=False,
        use_gated_activation_in_ffn=use_gated_activation_in_ffn,
        activation_dtype=activation_dtype,
    )

    params = moe_ffn.init(prng_key)
    moe_output, _ = moe_ffn.apply(params, inputs, inputs_mask=inputs_mask)
    simple_moe_fn = functools.partial(
        simple_moe,
        num_experts_per_token=num_experts_per_token,
        num_experts=num_experts,
        ffn_activation=moe_ffn.ffn_activation,
        use_gated_activation_in_ffn=use_gated_activation_in_ffn,
        activation_dtype=activation_dtype,
    )
    simple_moe_output = simple_moe_fn(params, inputs, inputs_mask=inputs_mask)
    self.assertEqual(moe_output.shape, simple_moe_output.shape)
    self.assertEqual(moe_output.dtype, simple_moe_output.dtype)
    np.testing.assert_allclose(
        moe_output, simple_moe_output, rtol=1e-2, atol=1e-2)

    def loss1(params, inputs, inputs_mask):
      moe_output, _ = moe_ffn.apply(params, inputs, inputs_mask=inputs_mask)
      return jnp.sum(moe_output) / (batch_size * seq_len)

    def loss2(params, inputs, inputs_mask):
      simple_moe_output = simple_moe_fn(params, inputs, inputs_mask=inputs_mask)
      return jnp.sum(simple_moe_output) / (batch_size * seq_len)

    grad1 = jax.grad(loss1)(params, inputs, inputs_mask)
    grad2 = jax.grad(loss2)(params, inputs, inputs_mask)
    jax.tree.map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-2, atol=1e-2),
        grad1, grad2)


if __name__ == '__main__':
  absltest.main()
