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
"""Tests for `moe.py` against a NumPy transcription of the HF reference.

The reference below is `modeling_kimi_linear.py`'s `SituAndMul`, `KimiMLP`,
`KimiMoEGate` and `KimiSparseMoeBlock` (latent variant) rewritten in NumPy --
deliberately naive (a Python loop over the routed experts), so it shares no
code, no dispatch and no fusion with the implementation under test.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from simply import config_lib
from simply.utils import sharding as sharding_lib
from simply.zoo.kimi_k3 import config_lib as k3_config_lib
from simply.zoo.kimi_k3.utils import moe

# Enough CPU devices for a mesh whose every axis is > 1; must precede the
# first use of the backend.
_EP_DEVICES = 8
jax.config.update('jax_num_cpu_devices', _EP_DEVICES)

# (replica, data, seq, model). The second one is the interesting case: the
# expert `shard_map` then has to compose with a `data`-sharded contraction and
# a `model`-sharded intermediate, which a `seq`-only mesh cannot exercise.
_EP_MESHES = (('seq_only', (1, 1, 8, 1)), ('all_axes', (1, 2, 2, 2)))


_BETA = 4.0
_LINEAR_BETA = 25.0
_EPSILON = 1e-5
_ATOL = 2e-5


# --- NumPy reference ---------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
  # `tanh` spelling of the logistic: identical, and it does not overflow on the
  # saturating-input cases below.
  return 0.5 * (1.0 + np.tanh(0.5 * x))


def _situ_glu(
    gate: np.ndarray,
    up: np.ndarray,
    beta: float = _BETA,
    linear_beta: float = _LINEAR_BETA,
) -> np.ndarray:
  """`SituAndMul.forward`."""
  activated = beta * np.tanh(gate / beta) * _sigmoid(gate)
  return activated * (linear_beta * np.tanh(up / linear_beta))


def _mlp(
    x: np.ndarray, p, beta: float = _BETA, linear_beta: float = _LINEAR_BETA
) -> np.ndarray:
  """`KimiMLP.forward` / `KimiBlockSparseMLP.forward`."""
  hidden = _situ_glu(
      x @ p['ffn_0_gate']['w'], x @ p['ffn_0']['w'], beta, linear_beta
  )
  return hidden @ p['ffn_1']['w']


def _rms_norm(x: np.ndarray, scale: np.ndarray, dtype=np.float32) -> np.ndarray:
  """`KimiRMSNorm.forward`: normalize in f32, cast, then apply the gain."""
  mean_square = np.mean(np.square(x.astype(np.float32)), -1, keepdims=True)
  normed = (x / np.sqrt(mean_square + _EPSILON)).astype(dtype)
  return normed * scale.astype(dtype)


def _route(
    x: np.ndarray,
    p,
    *,
    top_k: int,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    use_biased_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
  """`KimiMoEGate.forward` (K3 has one expert group, so no group masking).

  Args:
    x: `[N, D]` activations.
    p: `{'w': [D, E], 'bias': [E]}`.
    top_k: experts per token.
    renormalize: renormalize the gathered weights to sum 1.
    routed_scaling_factor: final multiplier.
    use_biased_weights: the BUG this test guards against -- taking the combine
      weights from the biased scores instead of the raw ones.

  Returns:
    `[N, top_k]` expert ids and combine weights.
  """
  scores = _sigmoid(x @ p['w'])
  biased = scores + p['bias']
  # `torch.topk`: largest first, lower index wins a tie.
  indices = np.argsort(-biased, axis=-1, kind='stable')[:, :top_k]
  weights = np.take_along_axis(
      biased if use_biased_weights else scores, indices, axis=-1
  )
  if top_k > 1 and renormalize:
    weights = weights / (np.sum(weights, -1, keepdims=True) + 1e-20)
  return indices, weights * routed_scaling_factor


def _latent_moe(
    x: np.ndarray,
    p,
    *,
    top_k: int,
    num_experts: int,
    beta: float = _BETA,
    linear_beta: float = _LINEAR_BETA,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    use_latent_norm: bool = True,
    use_biased_weights: bool = False,
    dtype=np.float32,
) -> np.ndarray:
  """`KimiSparseMoeBlock.forward` with `use_latent_moe=True`."""
  batch, seq_len, model_dim = x.shape
  flat = x.reshape(-1, model_dim)
  indices, weights = _route(
      flat,
      p['router'],
      top_k=top_k,
      renormalize=renormalize,
      routed_scaling_factor=routed_scaling_factor,
      use_biased_weights=use_biased_weights,
  )
  latent = flat @ p['down_proj']['w']
  experts = p['experts']
  y = np.zeros_like(latent)
  for token in range(latent.shape[0]):
    for slot in range(top_k):
      expert = indices[token, slot]
      assert 0 <= expert < num_experts
      expert_p = {
          name: {'w': experts[name]['w'][expert]}
          for name in ('ffn_0_gate', 'ffn_0', 'ffn_1')
      }
      y[token] += weights[token, slot] * _mlp(
          latent[token], expert_p, beta, linear_beta
      )
  if use_latent_norm:
    y = _rms_norm(y, p['latent_norm']['scale'], dtype).astype(np.float32)
  y = y @ p['up_proj']['w']
  if 'shared' in p:
    y += _mlp(flat, p['shared'], beta, linear_beta)
  return y.reshape(batch, seq_len, -1)


# --- Fixtures ----------------------------------------------------------------


class _Dims:
  """Small stand-ins for K3's (7168, 3584, 3072, 896, 16, 6144)."""

  batch = 2
  seq_len = 5
  model_dim = 16
  latent_dim = 8
  expert_dim = 6
  num_experts = 8
  top_k = 2
  shared_dim = 10


def _mlp_params(rng, model_dim: int, expand_dim: int, scale: float = 0.5):
  normal = lambda *shape: rng.normal(size=shape).astype(np.float32) * scale
  return {
      'ffn_0_gate': {'w': normal(model_dim, expand_dim)},
      'ffn_0': {'w': normal(model_dim, expand_dim)},
      'ffn_1': {'w': normal(expand_dim, model_dim)},
  }


def _moe_params(rng, dims=_Dims, *, with_shared: bool = True):
  """A plain-dict param tree, exactly what the checkpoint converter emits."""
  normal = lambda *shape: rng.normal(size=shape).astype(np.float32) * 0.5
  params = {
      'router': {
          'w': normal(dims.model_dim, dims.num_experts),
          'bias': rng.normal(size=dims.num_experts).astype(np.float32) * 0.1,
      },
      'down_proj': {'w': normal(dims.model_dim, dims.latent_dim)},
      'up_proj': {'w': normal(dims.latent_dim, dims.model_dim)},
      'latent_norm': {
          'scale': (
              (1.0 + 0.1 * rng.normal(size=dims.latent_dim)).astype(np.float32)
          )
      },
      'experts': {
          'ffn_0_gate': {
              'w': normal(dims.num_experts, dims.latent_dim, dims.expert_dim)
          },
          'ffn_0': {
              'w': normal(dims.num_experts, dims.latent_dim, dims.expert_dim)
          },
          'ffn_1': {
              'w': normal(dims.num_experts, dims.expert_dim, dims.latent_dim)
          },
      },
  }
  if with_shared:
    params['shared'] = _mlp_params(rng, dims.model_dim, dims.shared_dim)
  return params


def _make_moe(dims=_Dims, **overrides) -> moe.KimiK3LatentMoE:
  kwargs = dict(
      model_dim=dims.model_dim,
      latent_dim=dims.latent_dim,
      moe_intermediate_size=dims.expert_dim,
      num_experts=dims.num_experts,
      num_experts_per_token=dims.top_k,
      shared_expert_dim=dims.shared_dim,
      rms_norm_epsilon=_EPSILON,
      situ_beta=_BETA,
      situ_linear_beta=_LINEAR_BETA,
      activation_dtype='float32',
  )
  kwargs.update(overrides)
  return moe.KimiK3LatentMoE(**kwargs)


def _inputs(rng, dims=_Dims, scale: float = 1.0) -> np.ndarray:
  return (
      rng.normal(size=(dims.batch, dims.seq_len, dims.model_dim)) * scale
  ).astype(np.float32)


def _max_abs_diff(a, b) -> float:
  return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _max_rel_diff(a, b) -> float:
  """`_max_abs_diff` relative to the reference's scale.

  Saturating inputs push the FFN outputs to O(100), where f32 accumulation
  order alone costs more than an absolute 2e-5.

  Args:
    a: array under test.
    b: reference array, whose magnitude sets the scale.

  Returns:
    The largest absolute difference, divided by `max(1, max|b|)` so that
    small-output cases stay an absolute comparison.
  """
  return _max_abs_diff(a, b) / max(1.0, float(np.max(np.abs(np.asarray(b)))))


class SituGluTest(parameterized.TestCase):
  """`moe._situ_glu` against an inline transcription of `SituAndMul.forward`."""

  @parameterized.parameters((4.0, 25.0), (1.0, None))
  def test_situ_glu_matches_hf(self, beta, linear_beta):
    rng = np.random.default_rng(1)
    gate = rng.normal(scale=10.0, size=(4, 16)).astype(np.float32)
    up = rng.normal(scale=10.0, size=(4, 16)).astype(np.float32)
    situ = beta * np.tanh(gate / beta) / (1.0 + np.exp(-gate))
    capped = (
        up if linear_beta is None else linear_beta * np.tanh(up / linear_beta)
    )
    got = moe._situ_glu(
        jnp.asarray(gate),
        jnp.asarray(up),
        beta=beta,
        # HF's uncapped `up` branch: a cap this wide is a no-op in f32.
        linear_beta=linear_beta if linear_beta is not None else 1e30,
    )
    np.testing.assert_allclose(np.asarray(got), situ * capped, atol=2e-5)

  def test_situ_glu_is_bounded(self):
    """Both factors are soft capped, unlike SwiGLU (report S2.3.2)."""
    big = jnp.full((32,), 1e4, jnp.float32)
    got = moe._situ_glu(big, big, beta=4.0, linear_beta=25.0)
    np.testing.assert_allclose(np.asarray(got), 100.0, rtol=1e-4)


class KimiK3DenseMLPTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('small_inputs', 1.0),
      # 40x inputs drive both branches deep into their soft caps, so the test
      # fails if either `beta` or `linear_beta` is dropped.
      ('saturating_inputs', 40.0),
  )
  def test_matches_numpy_reference(self, input_scale: float):
    rng = np.random.default_rng(0)
    params = _mlp_params(rng, _Dims.model_dim, _Dims.shared_dim)
    x = _inputs(rng, scale=input_scale)
    mlp = moe.KimiK3DenseMLP(
        model_dim=_Dims.model_dim,
        expand_dim=_Dims.shared_dim,
        situ_beta=_BETA,
        situ_linear_beta=_LINEAR_BETA,
        activation_dtype='float32',
    )
    y, extra = mlp.apply(params, jnp.asarray(x))
    self.assertEmpty(extra)
    expected = _mlp(x, params)
    self.assertLess(_max_rel_diff(y, expected), _ATOL)

  def test_soft_caps_are_applied(self):
    rng = np.random.default_rng(1)
    params = _mlp_params(rng, _Dims.model_dim, _Dims.shared_dim)
    x = _inputs(rng, scale=40.0)
    mlp = moe.KimiK3DenseMLP(
        model_dim=_Dims.model_dim,
        expand_dim=_Dims.shared_dim,
        activation_dtype='float32',
    )
    y, _ = mlp.apply(params, jnp.asarray(x))
    uncapped = (
        _situ_glu(
            x @ params['ffn_0_gate']['w'],
            x @ params['ffn_0']['w'],
            beta=1e6,
            linear_beta=1e6,
        )
        @ params['ffn_1']['w']
    )
    self.assertGreater(_max_abs_diff(y, uncapped), 1.0)

  def test_param_tree_shapes(self):
    mlp = moe.KimiK3DenseMLP(model_dim=16, expand_dim=32)
    params = mlp.init(jax.random.PRNGKey(0))
    self.assertEqual(
        _shape_tree(params),
        {
            'ffn_0_gate': {'w': (16, 32)},
            'ffn_0': {'w': (16, 32)},
            'ffn_1': {'w': (32, 16)},
        },
    )


def _shape_tree(params):
  return jax.tree.map(
      lambda leaf: tuple(leaf.shape),
      params,
      is_leaf=lambda leaf: hasattr(leaf, 'shape'),
  )


class KimiK3LatentMoETest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('gmm', 'gmm'),
      ('dense', 'dense'),
  )
  def test_matches_numpy_reference(self, expert_dispatch: str):
    rng = np.random.default_rng(2)
    params = _moe_params(rng)
    x = _inputs(rng)
    layer = _make_moe(expert_dispatch=expert_dispatch)
    y, extra = layer.apply(params, jnp.asarray(x))
    expected = _latent_moe(
        x, params, top_k=_Dims.top_k, num_experts=_Dims.num_experts
    )
    self.assertEqual(y.shape, expected.shape)
    self.assertLess(_max_abs_diff(y, expected), _ATOL)
    self.assertIn('max_load', extra['metric'])

  @parameterized.named_parameters(
      ('non_default_caps', 2.0, 7.0, _Dims.top_k),
      ('top_1', _BETA, _LINEAR_BETA, 1),
  )
  def test_config_fields_are_honoured(
      self, beta: float, linear_beta: float, top_k: int
  ):
    """Caps and top-k come from the fields, not from hardcoded K3 defaults."""
    rng = np.random.default_rng(13)
    params = _moe_params(rng)
    x = _inputs(rng)
    layer = _make_moe(
        situ_beta=beta,
        situ_linear_beta=linear_beta,
        num_experts_per_token=top_k,
    )
    y, _ = layer.apply(params, jnp.asarray(x))
    expected = _latent_moe(
        x,
        params,
        top_k=top_k,
        num_experts=_Dims.num_experts,
        beta=beta,
        linear_beta=linear_beta,
    )
    self.assertLess(_max_abs_diff(y, expected), _ATOL)
    default = _latent_moe(
        x, params, top_k=_Dims.top_k, num_experts=_Dims.num_experts
    )
    self.assertGreater(_max_abs_diff(expected, default), 1e-2)

  def test_dispatch_paths_agree(self):
    rng = np.random.default_rng(3)
    params = _moe_params(rng)
    x = jnp.asarray(_inputs(rng))
    grouped, _ = _make_moe(expert_dispatch='gmm').apply(params, x)
    dense, _ = _make_moe(expert_dispatch='dense').apply(params, x)
    self.assertLess(_max_abs_diff(grouped, dense), _ATOL)

  @parameterized.named_parameters(
      ('renormalized', True, 1.0),
      ('unnormalized', False, 1.0),
      ('scaled', True, 2.5),
      ('unnormalized_scaled', False, 0.5),
  )
  def test_renormalization_and_scaling(
      self, renormalize: bool, routed_scaling_factor: float
  ):
    rng = np.random.default_rng(4)
    params = _moe_params(rng)
    del params['latent_norm']  # It would divide out both effects per token.
    x = _inputs(rng)
    layer = _make_moe(
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        use_latent_norm=False,
    )
    y, _ = layer.apply(params, jnp.asarray(x))
    expected = _latent_moe(
        x,
        params,
        top_k=_Dims.top_k,
        num_experts=_Dims.num_experts,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        use_latent_norm=False,
    )
    self.assertLess(_max_abs_diff(y, expected), _ATOL)
    # Both flags must actually move the output at this scale.
    baseline = _latent_moe(
        x,
        params,
        top_k=_Dims.top_k,
        num_experts=_Dims.num_experts,
        use_latent_norm=False,
    )
    if not renormalize or routed_scaling_factor != 1.0:
      self.assertGreater(_max_abs_diff(y, baseline), 1e-2)

  def test_router_bias_steers_selection_but_not_weights(self):
    """`e_score_correction_bias` reorders the top-k; the weights stay raw."""
    dims = _Dims
    rng = np.random.default_rng(5)
    params = _moe_params(rng)
    # One token whose scores are exactly sigmoid(logits): x is the first basis
    # vector and only the first row of the router matrix is non-zero.
    logits = np.zeros((dims.model_dim, dims.num_experts), np.float32)
    logits[0] = np.array(
        [2.0, 1.0, 0.5, 0.0, -1.0, -2.0, -3.0, -4.0], np.float32
    )
    params['router']['w'] = logits
    # Without the bias the winners would be experts 0 and 1; the bias promotes
    # expert 7 (the lowest raw score) over expert 1.
    bias = np.zeros(dims.num_experts, np.float32)
    bias[7] = 0.9
    params['router']['bias'] = bias
    x = np.zeros((1, 1, dims.model_dim), np.float32)
    x[0, 0, 0] = 1.0

    layer = _make_moe()
    # pylint: disable=protected-access
    indices, weights = layer._route(params['router'], jnp.asarray(x))
    # Biased: expert 7 (0.018 + 0.9) now outranks expert 0 (0.881).
    np.testing.assert_array_equal(np.asarray(indices)[0, 0], [7, 0])
    raw = _sigmoid(logits[0])
    expected_weights = np.array([raw[7], raw[0]]) / (raw[0] + raw[7] + 1e-20)
    self.assertLess(_max_abs_diff(weights[0, 0], expected_weights), _ATOL)

    y, _ = layer.apply(params, jnp.asarray(x))
    expected = _latent_moe(
        x, params, top_k=dims.top_k, num_experts=dims.num_experts
    )
    self.assertLess(_max_abs_diff(y, expected), _ATOL)
    # The same layer with biased combine weights is a materially different
    # function, so the test above is not vacuous.
    biased = _latent_moe(
        x,
        params,
        top_k=dims.top_k,
        num_experts=dims.num_experts,
        use_biased_weights=True,
    )
    self.assertGreater(_max_abs_diff(expected, biased), 1e-3)

  @parameterized.named_parameters(('gmm', 'gmm'), ('dense', 'dense'))
  def test_idle_experts(self, expert_dispatch: str):
    """Six of eight experts never win: empty groups must stay exact."""
    rng = np.random.default_rng(6)
    params = _moe_params(rng)
    bias = np.zeros(_Dims.num_experts, np.float32)
    bias[:2] = 10.0  # Dominates any sigmoid score, so the top-2 is {0, 1}.
    params['router']['bias'] = bias
    x = _inputs(rng)
    y, _ = _make_moe(expert_dispatch=expert_dispatch).apply(
        params, jnp.asarray(x)
    )
    expected = _latent_moe(
        x, params, top_k=_Dims.top_k, num_experts=_Dims.num_experts
    )
    self.assertLess(_max_abs_diff(y, expected), _ATOL)

  def test_production_expert_count(self):
    """896 experts / top-16 (K3's real routing width) on toy widths."""

    class Dims(_Dims):
      batch = 1
      seq_len = 4
      num_experts = 896
      top_k = 16

    rng = np.random.default_rng(12)
    params = _moe_params(rng, Dims)
    x = _inputs(rng, Dims)
    grouped, _ = _make_moe(Dims, expert_dispatch='gmm').apply(
        params, jnp.asarray(x)
    )
    expected = _latent_moe(
        x, params, top_k=Dims.top_k, num_experts=Dims.num_experts
    )
    self.assertLess(_max_abs_diff(grouped, expected), _ATOL)
    dense, _ = _make_moe(Dims, expert_dispatch='dense').apply(
        params, jnp.asarray(x)
    )
    self.assertLess(_max_abs_diff(grouped, dense), _ATOL)

  def test_no_latent_norm_and_no_shared_expert(self):
    rng = np.random.default_rng(7)
    params = _moe_params(rng, with_shared=False)
    del params['latent_norm']
    x = _inputs(rng)
    layer = _make_moe(use_latent_norm=False, shared_expert_dim=0)
    y, _ = layer.apply(params, jnp.asarray(x))
    expected = _latent_moe(
        x,
        params,
        top_k=_Dims.top_k,
        num_experts=_Dims.num_experts,
        use_latent_norm=False,
    )
    self.assertLess(_max_abs_diff(y, expected), _ATOL)

  def test_inputs_mask_isolates_padded_tokens(self):
    rng = np.random.default_rng(8)
    params = _moe_params(rng)
    x = _inputs(rng)
    x[:, 3:] = 1e3 * rng.normal(size=x[:, 3:].shape)  # Garbage in the padding.
    mask = np.ones((_Dims.batch, _Dims.seq_len), bool)
    mask[:, 3:] = False
    layer = _make_moe()
    y, _ = layer.apply(params, jnp.asarray(x), inputs_mask=jnp.asarray(mask))
    np.testing.assert_array_equal(np.asarray(y)[:, 3:], 0.0)
    # The layer is position-wise, so the valid prefix must be bit-comparable to
    # a run that never saw the padded positions at all.
    prefix, _ = layer.apply(params, jnp.asarray(x[:, :3]))
    self.assertLess(_max_abs_diff(y[:, :3], prefix), _ATOL)

  def test_quantized_expert_weights(self):
    """Weights may arrive as `{'quant_array', 'scale'}` dicts."""
    rng = np.random.default_rng(9)
    params = _moe_params(rng)
    dequantized = dict(params)
    quantized = dict(params)
    experts, dequant_experts = {}, {}
    for name, leaf in params['experts'].items():
      w = leaf['w']
      scale = np.max(np.abs(w), axis=-1, keepdims=True) / 127.0
      quant = np.round(w / scale).astype(np.int8)
      experts[name] = {'w': {'quant_array': quant, 'scale': scale}}
      dequant_experts[name] = {'w': (quant * scale).astype(np.float32)}
    quantized['experts'] = experts
    dequantized['experts'] = dequant_experts
    layer = _make_moe()
    x = jnp.asarray(_inputs(rng))
    y, _ = layer.apply(quantized, x)
    expected, _ = layer.apply(dequantized, x)
    self.assertLess(_max_abs_diff(y, expected), _ATOL)

  def test_param_tree_shapes(self):
    dims = _Dims
    layer = _make_moe()
    params = layer.init(jax.random.PRNGKey(0))
    self.assertEqual(
        _shape_tree(params),
        {
            'router': {
                'w': (dims.model_dim, dims.num_experts),
                'bias': (dims.num_experts,),
            },
            'down_proj': {'w': (dims.model_dim, dims.latent_dim)},
            'up_proj': {'w': (dims.latent_dim, dims.model_dim)},
            'latent_norm': {'scale': (dims.latent_dim,)},
            'experts': {
                'ffn_0_gate': {
                    'w': (dims.num_experts, dims.latent_dim, dims.expert_dim)
                },
                'ffn_0': {
                    'w': (dims.num_experts, dims.latent_dim, dims.expert_dim)
                },
                'ffn_1': {
                    'w': (dims.num_experts, dims.expert_dim, dims.latent_dim)
                },
            },
            'shared': {
                'ffn_0_gate': {'w': (dims.model_dim, dims.shared_dim)},
                'ffn_0': {'w': (dims.model_dim, dims.shared_dim)},
                'ffn_1': {'w': (dims.shared_dim, dims.model_dim)},
            },
        },
    )
    # `init`'s params (AnnotatedArray leaves) must run as-is.
    x = jnp.asarray(_inputs(np.random.default_rng(10)))
    y, _ = layer.apply(params, x)
    self.assertEqual(y.shape, x.shape)
    self.assertTrue(np.all(np.isfinite(np.asarray(y))))

  def test_router_dtype_is_float32_under_bfloat16_activations(self):
    rng = np.random.default_rng(11)
    params = _moe_params(rng)
    layer = _make_moe(activation_dtype='bfloat16')
    x = jnp.asarray(_inputs(rng), jnp.bfloat16)
    # pylint: disable=protected-access
    indices, weights = layer._route(params['router'], x)
    self.assertEqual(weights.dtype, jnp.float32)
    self.assertEqual(indices.shape, (_Dims.batch, _Dims.seq_len, _Dims.top_k))
    y, _ = layer.apply(params, x)
    self.assertEqual(y.dtype, jnp.bfloat16)
    # Loose, but it catches an f32 stage silently becoming bf16 or vice versa.
    expected = _latent_moe(
        np.asarray(x, np.float32),
        params,
        top_k=_Dims.top_k,
        num_experts=_Dims.num_experts,
        dtype=jnp.bfloat16,
    )
    self.assertLess(_max_rel_diff(np.asarray(y, np.float32), expected), 2e-2)


def _ep_dtype_cases() -> list[tuple[str, tuple[int, ...], str]]:
  """Returns `(name, mesh_shape, dtype)` for every EP mesh at both dtypes."""
  cases = []
  for name, shape in _EP_MESHES:
    for dtype in ('float32', 'bfloat16'):
      cases.append((f'{name}_{dtype}', shape, dtype))
  return cases


class ExpertParallelTest(parameterized.TestCase):
  """`expert_parallel_axis` must be an exact refactor of the grouped path.

  Run on a 4-device CPU mesh (`jax_num_cpu_devices`, set at import), which is
  enough to exercise the `shard_map`: the expert stacks are split 4 ways, each
  shard rolls its window of the sorted rows to the front, and the `psum`
  reassembles rows it does not own from zeros.
  """

  def _mesh(self, shape=(1, 1, _EP_DEVICES, 1)) -> jax.sharding.Mesh:
    # `sharding_lib.create_mesh`, not `jax.make_mesh`: the latter builds a mesh
    # with explicit axis types, under which `ragged_dot` demands an
    # `out_sharding`. Production meshes come from `create_mesh`.
    return sharding_lib.create_mesh(
        mesh_shape=shape, axis_names=('replica', 'data', 'seq', 'model')
    )

  @parameterized.named_parameters(*_ep_dtype_cases())
  def test_matches_the_replicated_grouped_path(
      self, mesh_shape, activation_dtype
  ):
    rng = np.random.default_rng(12)
    params = _moe_params(rng)
    x = jnp.asarray(_inputs(rng), activation_dtype)
    if mesh_shape[2] == _EP_DEVICES:
      sharding_config = None
    else:
      # The real decode placement, so the `shard_map` has to compose with a
      # sharded contraction and a sharded intermediate.
      sharding_config = k3_config_lib.kimi_k3_decoding_sharding()
    base = _make_moe(
        activation_dtype=activation_dtype, sharding_config=sharding_config
    )
    parallel = _make_moe(
        activation_dtype=activation_dtype,
        sharding_config=sharding_config,
        expert_parallel_axis='seq',
    )
    with jax.sharding.set_mesh(self._mesh(mesh_shape)):
      want, _ = jax.jit(base.apply)(params, x)
      got, _ = jax.jit(parallel.apply)(params, x)
    self.assertEqual(got.dtype, want.dtype)
    np.testing.assert_allclose(
        np.asarray(got, np.float32),
        np.asarray(want, np.float32),
        rtol=1e-6,
        atol=1e-6,
    )

  def test_no_expert_stack_all_gather_in_the_lowered_program(self):
    """The point of the path: the `[E, in, out]` stacks stay put."""
    rng = np.random.default_rng(13)
    params = _moe_params(rng)
    x = jnp.asarray(_inputs(rng))
    sharding = jax.sharding.NamedSharding
    spec = jax.sharding.PartitionSpec
    mesh = self._mesh()
    with jax.sharding.set_mesh(mesh):
      experts = jax.tree.map(
          lambda w: jax.device_put(w, sharding(mesh, spec('seq'))),
          params['experts'],
      )
      params = params | {'experts': experts}
      texts = {}
      for name, axis in (('base', None), ('parallel', 'seq')):
        layer = _make_moe(expert_parallel_axis=axis)
        texts[name] = (
            jax.jit(layer.apply).lower(params, x).compile().as_text() or ''
        )
    gathers = lambda text: sum(
        1
        for line in text.splitlines()
        if 'all-gather' in line and f',{_Dims.latent_dim},' in line
    )
    self.assertGreater(gathers(texts['base']), 0)
    self.assertEqual(gathers(texts['parallel']), 0)

  @parameterized.named_parameters(
      # A gradient is where this path is *known* to fail at scale, so the cases
      # are the ones that make a per-expert reduction degenerate: an expert
      # with no tokens at all, a shard with no tokens at all, and padded rows
      # (which route to the phantom expert and sort past every group).
      ('seq_only', (1, 1, _EP_DEVICES, 1), None, False),
      ('all_axes', (1, 2, 2, 2), None, False),
      ('all_axes_padded', (1, 2, 2, 2), None, True),
      ('all_axes_one_shard_starved', (1, 2, 2, 2), 'starve', False),
      ('all_axes_one_expert', (1, 2, 2, 2), 'one_expert', True),
  )
  def test_gradients_match_the_replicated_grouped_path(
      self, mesh_shape, routing, padded
  ):
    """`pure-grouse` measured NaN gradients from this path at 0.98 B."""
    rng = np.random.default_rng(14)
    params = _moe_params(rng)
    if routing:
      # A router bias of +-50 is a hard assignment: 'starve' sends every token
      # to the first shard's experts, 'one_expert' to a single expert.
      bias = np.full((_Dims.num_experts,), -50.0, np.float32)
      bias[: 1 if routing == 'one_expert' else _Dims.num_experts // 2] = 50.0
      params['router']['bias'] = bias
    x = jnp.asarray(_inputs(rng))
    inputs_mask = None
    if padded:
      mask = np.ones((_Dims.batch, _Dims.seq_len), bool)
      mask[:, _Dims.seq_len // 2 :] = False
      inputs_mask = jnp.asarray(mask)
    sharding_config = (
        None
        if mesh_shape[2] == _EP_DEVICES
        else k3_config_lib.kimi_k3_decoding_sharding()
    )

    def loss(layer, params):
      return jnp.sum(
          jnp.square(layer.apply(params, x, inputs_mask=inputs_mask)[0])
      )

    base = _make_moe(sharding_config=sharding_config)
    parallel = _make_moe(
        sharding_config=sharding_config, expert_parallel_axis='seq'
    )
    with jax.sharding.set_mesh(self._mesh(mesh_shape)):
      want = jax.jit(jax.grad(lambda p: loss(base, p)))(params)
      got = jax.jit(jax.grad(lambda p: loss(parallel, p)))(params)
    flat_want = jax.tree_util.tree_flatten_with_path(want)[0]
    flat_got = jax.tree_util.tree_flatten_with_path(got)[0]
    for (path, w), (_, g) in zip(flat_want, flat_got, strict=True):
      name = jax.tree_util.keystr(path)
      self.assertTrue(np.all(np.isfinite(np.asarray(g))), msg=f'{name} is nan')
      np.testing.assert_allclose(
          np.asarray(g), np.asarray(w), rtol=2e-5, atol=2e-5, err_msg=name
      )

  def test_matches_the_plain_path_without_a_kernel(self):
    """Values and gradients again, with no Mosaic call under the matmuls.

    `gmm_impl='dense_grouped'` is the kernel-free control that localized the
    NaN gradient of the hardware expert-parallel path; this
    case cross-validates the instrument against `ragged_dot` so it cannot rot
    while the follow-up is open.
    """
    rng = np.random.default_rng(15)
    params = _moe_params(rng)
    x = jnp.asarray(_inputs(rng))
    sharding_config = k3_config_lib.kimi_k3_decoding_sharding()
    base = _make_moe(sharding_config=sharding_config)
    parallel = _make_moe(
        sharding_config=sharding_config,
        expert_parallel_axis='seq',
        gmm_impl='dense_grouped',
    )
    loss = lambda layer, p: jnp.sum(jnp.square(layer.apply(p, x)[0]))
    with jax.sharding.set_mesh(self._mesh((1, 2, 2, 2))):
      want, want_grad = jax.jit(jax.value_and_grad(lambda p: loss(base, p)))(
          params
      )
      got, got_grad = jax.jit(jax.value_and_grad(lambda p: loss(parallel, p)))(
          params
      )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    for w, g in zip(jax.tree.leaves(want_grad), jax.tree.leaves(got_grad)):
      np.testing.assert_allclose(
          np.asarray(g), np.asarray(w), rtol=2e-5, atol=2e-5
      )

  def test_expert_parallelism_is_a_no_op_on_a_mesh_without_the_axis(self):
    """A mesh that does not name the axis leaves the layer replicated.

    The guard is what makes an EP config runnable on a small mesh -- a CPU
    test mesh, say -- instead of raising or silently sharding over nothing.
    """
    layer = _make_moe(
        expert_parallel_axis='seq',
        sharding_config=k3_config_lib.kimi_k3_decoding_sharding(),
    )
    with jax.sharding.set_mesh(
        sharding_lib.create_mesh(
            mesh_shape={'replica': 1, 'data': 8, 'model': 1},
            axis_names=('replica', 'data', 'model'),
        )
    ):
      self.assertEqual(layer._expert_parallel_shards(), 1)  # pylint: disable=protected-access

  def test_rejects_an_activation_axis(self):
    with self.assertRaisesRegex(ValueError, 'expert_parallel_axis'):
      _make_moe(
          expert_parallel_axis='seq',
          sharding_config=config_lib.moe_sharding(),
      )


if __name__ == '__main__':
  absltest.main()
