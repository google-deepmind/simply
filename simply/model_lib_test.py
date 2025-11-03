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
from simply.utils import masked
from simply.utils import optimizers as opt_lib
from simply.utils import pytree
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
    model = model_lib.TransformerLM(
        config, sharding_config=config_lib.GSPMDSharding()
    )
    params = model.init(prng_key)
    batch_size = 2
    batch = jax.random.randint(
        prng_key, shape=(batch_size, 8), minval=0, maxval=config.vocab_size
    )
    logits, _ = model.apply(params, batch)
    logits = np.array(logits)
    expected_logits = [
        [
            [
                2.7519984245300293,
                -0.2155388593673706,
                -0.21683339774608612,
                1.0551955699920654,
            ],
            [
                2.75199818611145,
                -0.2155388742685318,
                -0.2168334424495697,
                1.0551949739456177,
            ],
            [
                2.75199818611145,
                -0.21553881466388702,
                -0.21683336794376373,
                1.0551953315734863,
            ],
            [
                2.751997947692871,
                -0.21553891897201538,
                -0.21683350205421448,
                1.0551954507827759,
            ],
            [
                0.06351777166128159,
                1.5503135919570923,
                -1.1891260147094727,
                0.8147411346435547,
            ],
            [
                0.5395821928977966,
                0.028114307671785355,
                0.3171342611312866,
                2.6201274394989014,
            ],
            [
                -0.7307804822921753,
                1.5626051425933838,
                -1.6585614681243896,
                0.4315919876098633,
            ],
            [
                -0.4943885803222656,
                -1.399471402168274,
                1.7614712715148926,
                1.8116846084594727,
            ],
        ],
        [
            [
                0.5078861713409424,
                -1.2493711709976196,
                2.3632442951202393,
                1.254542350769043,
            ],
            [
                0.5078865885734558,
                -1.2493711709976196,
                2.363243818283081,
                1.2545422315597534,
            ],
            [
                2.8932785987854004,
                -0.7085745334625244,
                0.521935760974884,
                1.2472620010375977,
            ],
            [
                0.6895977258682251,
                -0.4795836806297302,
                0.8725722432136536,
                2.448836088180542,
            ],
            [
                2.8496932983398438,
                -0.8699660897254944,
                0.2777295410633087,
                1.549817681312561,
            ],
            [
                2.7562153339385986,
                -0.6204763054847717,
                -0.2111501842737198,
                1.4128155708312988,
            ],
            [
                1.1002166271209717,
                0.8060446381568909,
                -0.527627170085907,
                1.3561891317367554,
            ],
            [
                0.6698483228683472,
                -1.1773743629455566,
                1.478686809539795,
                2.4011504650115967,
            ],
        ],
    ]
    self.assertTrue(
        np.allclose(np.array(expected_logits), logits, atol=1e-5))

  def test_backward_pass(self):
    # Generate test data
    seed = 42
    prng_key = jax.random.key(seed)
    config = lm_test()
    config = dataclasses.replace(config, vocab_size=4)
    model = model_lib.TransformerLM(
        config, sharding_config=config_lib.GSPMDSharding()
    )
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
            [
                2.694333076477051,
                -0.2943641245365143,
                -0.1179223582148552,
                1.0545716285705566,
            ],
            [
                2.694333076477051,
                -0.29436421394348145,
                -0.11792253702878952,
                1.054571509361267,
            ],
            [
                2.694333076477051,
                -0.29436415433883667,
                -0.11792244017124176,
                1.054571509361267,
            ],
            [
                2.694333076477051,
                -0.29436418414115906,
                -0.11792242527008057,
                1.0545717477798462,
            ],
            [
                0.4579642713069916,
                1.3849267959594727,
                -1.1025311946868896,
                0.6992211937904358,
            ],
            [
                0.6853991150856018,
                -0.047158315777778625,
                0.3695349097251892,
                2.5992767810821533,
            ],
            [
                -0.3630373477935791,
                1.4172710180282593,
                -1.5797486305236816,
                0.313672810792923,
            ],
            [
                -0.39001917839050293,
                -1.4826804399490356,
                1.8424708843231201,
                1.7501987218856812,
            ],
        ],
        [
            [
                0.4981890916824341,
                -1.2542948722839355,
                2.372939109802246,
                1.250625729560852,
            ],
            [
                0.49818921089172363,
                -1.254294753074646,
                2.372939109802246,
                1.250625729560852,
            ],
            [
                2.869960069656372,
                -0.7210258841514587,
                0.5573416948318481,
                1.2652201652526855,
            ],
            [
                0.7159809470176697,
                -0.5081028938293457,
                0.9096092581748962,
                2.4309475421905518,
            ],
            [
                2.833054304122925,
                -0.8901601433753967,
                0.32386356592178345,
                1.5564500093460083,
            ],
            [
                2.7338998317718506,
                -0.6527514457702637,
                -0.14567126333713531,
                1.418534278869629,
            ],
            [
                1.1498193740844727,
                0.779076099395752,
                -0.47674769163131714,
                1.340635061264038,
            ],
            [
                0.7024662494659424,
                -1.1868096590042114,
                1.5199283361434937,
                2.3696625232696533,
            ],
        ],
    ]
    self.assertTrue(
        np.allclose(np.array(expected_logits), logits, atol=1e-5))

  def test_grad_accumulation(self):
    # Generate test data
    seed = 42
    prng_key = jax.random.key(seed)
    config = lm_test()
    config = dataclasses.replace(config, vocab_size=4)
    model = model_lib.TransformerLM(
        config, sharding_config=config_lib.GSPMDSharding()
    )
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
    model = model_lib.TransformerLM(
        config, sharding_config=config_lib.GSPMDSharding()
    )
    params = model.init(prng_key)
    dim_annotations = jax.tree.map(
        lambda x: x.metadata['dim_annotation'], params,
        is_leaf=lambda x: isinstance(x, common.AnnotatedArray))
    expected_dim_annotations = {
        'block_0': {
            'attn': {
                'o_proj': 'oii',
                'per_dim_scale': {'scale': 'h'},
                'qkv_proj': '.ioo',
            },
            'ffn_0': {'b': 'h', 'w': 'io'},
            'ffn_0_gate': {'b': 'h', 'w': 'io'},
            'ffn_1': {'b': 'h', 'w': 'io'},
            'post_ln_0': {'scale': 'h'},
            'post_ln_1': {'scale': 'h'},
            'pre_ln_0': {'scale': 'h'},
            'pre_ln_1': {'scale': 'h'},
        },
        'block_1': {
            'attn': {
                'o_proj': 'oii',
                'per_dim_scale': {'scale': 'h'},
                'qkv_proj': '.ioo',
            },
            'ffn_0': {'b': 'h', 'w': 'io'},
            'ffn_0_gate': {'b': 'h', 'w': 'io'},
            'ffn_1': {'b': 'h', 'w': 'io'},
            'post_ln_0': {'scale': 'h'},
            'post_ln_1': {'scale': 'h'},
            'pre_ln_0': {'scale': 'h'},
            'pre_ln_1': {'scale': 'h'},
        },
        'embed': 'io',
        'final_ln': {'scale': 'h'},
        'output_layer': {'b': 'h'},
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
    tfm_lm = model_lib.TransformerLM(
        new_config, sharding_config=config_lib.GSPMDSharding())
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
    tfm_lm = model_lib.TransformerLM(
        new_config, sharding_config=config_lib.GSPMDSharding())
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
      decode_state = {}
      init_logits, extra_output = apply_fn(
          params, inputs, decode_state=decode_state,
          segment_positions=segment_positions)
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

  @parameterized.named_parameters(
      ('top_k_1', 1), ('top_k_5', 5), ('top_k_10', 10), ('top_k_12', 12)
  )
  def test_top_k_mask(self, top_k: int):
    logits = np.random.randn(5, 10)
    mask = model_lib.top_k_mask(logits, top_k=top_k)
    self.assertEqual(mask.shape, logits.shape)
    self.assertEqual(mask.dtype, jnp.bool)
    np.testing.assert_array_equal(jnp.sum(mask, axis=-1), min(top_k, 10))
    np.testing.assert_array_less(
        masked.masked_max(logits, mask=~mask, axis=-1),
        masked.masked_min(logits, mask=mask, axis=-1) + 1e-6,
    )

  @parameterized.named_parameters(
      ('top_p_0.0', 0.0),
      ('top_p_0.2', 0.2),
      ('top_p_0.5', 0.5),
      ('top_p_0.8', 0.8),
      ('top_p_1.0', 1.0),
  )
  def test_top_p_mask(self, top_p: float):
    logits = np.random.randn(5, 10)
    probs = jax.nn.softmax(logits, axis=-1)
    mask = model_lib.top_p_mask(logits, top_p=top_p)
    self.assertEqual(mask.shape, logits.shape)
    self.assertEqual(mask.dtype, jnp.bool)
    np.testing.assert_array_less(
        top_p, masked.masked_sum(probs, mask=mask, axis=-1) + 1e-6
    )
    np.testing.assert_array_less(
        masked.masked_max(logits, mask=~mask, axis=-1),
        masked.masked_min(logits, mask=mask, axis=-1) + 1e-6,
    )

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

      token_logprobs = model_lib.compute_log_likelihood(
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
      token_ll = model_lib.compute_log_likelihood(
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
      token_ll = model_lib.compute_log_likelihood(
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
        token_ll = model_lib.compute_log_likelihood(
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
        token_ll = model_lib.compute_log_likelihood(
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
        new_config, config_lib.GSPMDSharding()
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


if __name__ == '__main__':
  absltest.main()
