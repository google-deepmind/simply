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
"""What a Kimi K3 is, and which K3s we run.

`KimiK3ExperimentConfig` is the architecture as data, `config_from_hf` is the
one place a released `config.json` turns into one -- shared by the checkpoint
converter and the equivalence test so the two cannot drift -- and the rest of
the file is the registered deployments and the arithmetic that sizes them.

The architecture is specified by the HuggingFace release `moonshotai/Kimi-K3`
(commit 9f62e4e) -- `config.json`, `configuration_kimi_k3.py` and
`modeling_kimi_linear.py` -- and by the Kimi K3 report (arXiv 2607.24653).
The text backbone (`KimiLinearForCausalLM`) is 93 layers of 3 KDA (linear
attention) : 1 gated NoPE MLA with the last two layers both MLA, Block
Attention Residuals across layers (`attn_res_block_size` = 12), Stable
LatentMoE channel mixing (896 routed experts of latent width 3584, 16 active,
2 shared) with SiTU-GLU activations, and a dense SiTU-GLU MLP at layer 0.
Only the text backbone is modelled: the MoonViT-V2 vision tower is not part
of the text-only inference path (text tokens never enter it).

Registered configs (`--experiment_config=<name>`):

  `kimi_k3_2p8t`      the released 2.8T model, bf16, on `4x4x8`.
  `kimi_k3_decode`    inference/eval variant of it, 32k context.
  `kimi_k3_decode_ep` the same with the routed matmuls expert-parallel: 5.4x
                      the decode throughput, and what the benchmarks ran on.
  `kimi_k3_tiny_test` a few tiny layers, random init, CPU-runnable.

The chip the released model is deployed on is *split*: 2 JAX devices of
~95 GiB each, so the `4x4x8` slice it runs on is 128 chips = 256 devices.
Mesh axes are `('replica','data','seq','model')` = (replica, FSDP/DP, expert
parallelism, tensor parallelism) and every weight is sharded over all of them
by `moe_sharding()`: the routed experts `[896, 3584, 3072]` split 896/8 x
3584/8 x 3072/4 = 112 x 448 x 768 per device.

`param_counts()` and `hbm_budget()` answer "does this deployment fit" for a
given config, batch, context and mesh without an accelerator; both are pinned
in the test, `param_counts` against `jax.eval_shape(model.init)`. README.md
tabulates what they return for the deployments above. The two ways to lose:
dropping the batch shard (`data=1`) replicates the MLA cache and the AttnRes
buffers on all 256 devices, and prefilling a long prompt in one call
materialises an AttnRes buffer larger than the model, since the snapshots are
8x the residual stream over the *prefill chunk*.
"""

from collections.abc import Mapping
import dataclasses
import math
from typing import Any

from simply import config_lib

# The weights are user-supplied: run `convert_hf_checkpoint` on the release and
# pass `--ckpt_dir`. The configs carry the *format* rather than a path, because
# a 1.45 TiB conversion lives wherever its owner has quota; `decode_eval`
# refuses to start without one. `KimiK3Format` is `V2Format` plus the
# decode-and-transpose of the MXFP4 expert leaves on restore.
KIMI_K3_CKPT_FORMAT = 'KimiK3Format'

# 128 chips = 256 devices (a split chip: 2 devices of 95 GiB each).
HBM_BYTES_PER_DEVICE = 95 * 2**30
MESH_4X4X8 = {'replica': 1, 'data': 8, 'seq': 8, 'model': 4}

_BF16 = 2
_F32 = 4


# --- Layer schedule ----------------------------------------------------------

LINEAR_ATTENTION = 'linear_attention'
FULL_ATTENTION = 'full_attention'


def kimi_k3_layer_types(
    n_layers: int, full_attention_interval: int = 4
) -> tuple[str, ...]:
  """Returns the per-layer token-mixer schedule.

  Every `full_attention_interval`-th layer is MLA (1-indexed, matching the HF
  `linear_attn_config.full_attn_layers` list), and the final layer is always
  MLA, which makes the last two layers MLA when `n_layers` is a multiple of
  the interval plus one (93 = 23 * 4 + 1).

  Args:
    n_layers: total number of decoder layers.
    full_attention_interval: period of the MLA layers.
  """
  types = [
      FULL_ATTENTION
      if (i + 1) % full_attention_interval == 0
      else LINEAR_ATTENTION
      for i in range(n_layers)
  ]
  types[-1] = FULL_ATTENTION
  return tuple(types)


# --- Experiment config -------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class KimiK3ExperimentConfig(config_lib.BaseExperimentConfig):
  """Experiment configuration for the Kimi K3 text backbone.

  Defaults are the released 2.8T model (`config.json` of moonshotai/Kimi-K3);
  tests override the dimensions to build small models with the same topology.
  """

  model_name: str = 'KimiK3LM'
  # `KimiK3Format` is `V2Format` plus MXFP4 expert decoding on restore.
  init_ckpt_format: str = 'KimiK3Format'
  vocab_name: str = 'KimiK3'
  lm_format_name: str = 'KimiK3Chat'
  # The release tokenizes each XTML piece separately, and BPE merges across
  # those boundaries, so encoding the flat prompt string is not reference
  # exact; this processor restores the release's segmentation.
  input_processor_name: str | None = 'KimiK3InputProcessor'

  # Backbone.
  seq_len: int = 8192
  vocab_size: int = 163_840
  model_dim: int = 7168
  n_layers: int = 93
  layer_types: tuple[str, ...] = ()
  full_attention_interval: int = 4
  # HF's `rms_norm_eps`; base defaults to 1e-6. Every test builds its config
  # through `config_from_hf`, which sets this field, so only the registered
  # deployment configs read the default: losing it changes the real 2.8T model
  # with the whole suite still green.
  rms_norm_epsilon: float = 1e-5
  use_tied_embedding: bool = False
  embedding_lookup_scale: float | None = None
  output_layer_use_bias: bool = False
  output_logits_soft_cap: float = -1.0
  # Declared, not read: `KimiK3LM` hardcodes each flavour. Deleting a line does
  # not drop the field, it re-inherits core's value -- and `override_from` and
  # the logged config would then claim RoPE, +1 norm gains and a 50.0 attention
  # soft cap for a model that has none of them.
  norm_scale_plus_one: bool = False
  attn_soft_cap: float = -1.0
  position_encoding: None = None  # NoPE everywhere: MLA is NoPE, KDA recurrent.
  use_per_dim_scale: bool = False
  use_post_ln: bool = False

  # Block Attention Residuals. `attn_res_block_size` layers per block; a
  # snapshot of the residual stream is pushed at every layer whose index is a
  # multiple of it.
  attn_res_block_size: int = 12

  # Gated MLA (NoPE).
  n_heads: int = 96
  per_head_dim: int = 128  # v_head_dim; qk_nope_head_dim matches it for K3.
  q_lora_rank: int = 1536
  kv_lora_rank: int = 512
  qk_nope_head_dim: int = 128
  qk_rope_head_dim: int = 64  # Carried and attended, never rotated (NoPE).
  v_head_dim: int = 128
  mla_use_output_gate: bool = True

  # Kimi Delta Attention.
  kda_num_heads: int = 96
  kda_head_dim: int = 128
  kda_conv_kernel_dim: int = 4
  kda_gate_lower_bound: float = -5.0
  kda_gate_lora_rank: int = 128
  kda_chunk_size: int = 64

  # Stable LatentMoE.
  use_moe: bool = True
  num_experts: int = 896
  num_experts_per_token: int = 16
  routed_expert_latent_dim: int = 3584
  moe_intermediate_size: int = 3072
  num_shared_experts: int = 2
  routed_scaling_factor: float = 1.0
  moe_renormalize: bool = True
  latent_moe_use_norm: bool = True
  first_k_dense_replace: int = 1
  ffn_expand_dim: int | None = 33792  # Dense layer 0 only.
  moe_expert_dispatch: str = 'gmm'  # 'gmm' (sorted grouped matmul) or 'dense'.
  # Mesh axis the expert stacks are sharded over ('seq' in `moe_sharding`).
  # Set it to run the grouped matmuls under a `shard_map` instead of letting
  # GSPMD all-gather the stacks before each one -- 159 GiB per device per
  # decode step at 2.8T. Needs the decoding sharding config; see
  # `utils/moe.py`s `KimiK3LatentMoE._expert_parallel_matmuls`.
  moe_expert_parallel_axis: str | None = None
  # Deliberately not a `FunctionRegistry` name: SiTU-GLU caps both branches of
  # the gate (see `utils/moe.py`), so a core FFN path must fail looking this up
  # instead of silently building the inherited 'gelu'.
  ffn_activation: str = 'situ'
  situ_beta: float = 4.0
  situ_linear_beta: float = 25.0
  # Inference-only: the router bias is frozen, so no balancing loss is applied.
  lbl_loss_weight: float = 0.0
  ffn_use_bias: bool = False  # Declared, not read; see `norm_scale_plus_one`.

  # Execution.
  # Pinned off, unlike Simply's default: this port has no scan path -- the
  # layer stack is always unrolled -- so the inherited True would be a claim
  # about a capability that is not here. Scanning K3's irregular 3:1 schedule
  # is exact and compiles 9 layer bodies instead of 93, but it cost +124% step
  # time on hardware, so it lands separately once that is understood.
  use_scan: bool = False
  use_remat: bool = False
  activation_dtype_name: str = 'bfloat16'
  decoding_sharding_config: config_lib.ShardingConfig | None = None

  @property
  def shared_expert_intermediate_size(self) -> int:
    """Shared experts are fused into one MLP of this width."""
    return self.moe_intermediate_size * self.num_shared_experts

  @property
  def q_head_dim(self) -> int:
    return self.qk_nope_head_dim + self.qk_rope_head_dim

  def resolved_layer_types(self) -> tuple[str, ...]:
    return self.layer_types or kimi_k3_layer_types(
        self.n_layers, self.full_attention_interval
    )


# --- HF config mapping -------------------------------------------------------


def config_from_hf(
    text_config: Mapping[str, Any], **overrides: Any
) -> KimiK3ExperimentConfig:
  """Builds the Simply config from HF `config.json:text_config`.

  One mapping shared by the checkpoint converter and the equivalence test, so
  a real checkpoint and the golden fixture cannot drift apart.

  Args:
    text_config: the `text_config` dict of the HF `config.json` (or the config
      of a scaled-down variant with the same field names).
    **overrides: fields to override on the resulting config.

  Returns:
    The equivalent `KimiK3ExperimentConfig`.

  Raises:
    ValueError: if the HF config asks for a variant this port does not
      implement, or if its layer schedule is not the 3:1 pattern.
  """
  linear = text_config['linear_attn_config']
  config = KimiK3ExperimentConfig(
      vocab_size=text_config['vocab_size'],
      model_dim=text_config['hidden_size'],
      n_layers=text_config['num_hidden_layers'],
      attn_res_block_size=text_config['attn_res_block_size'],
      rms_norm_epsilon=text_config['rms_norm_eps'],
      n_heads=text_config['num_attention_heads'],
      per_head_dim=text_config['v_head_dim'],
      q_lora_rank=text_config['q_lora_rank'],
      kv_lora_rank=text_config['kv_lora_rank'],
      qk_nope_head_dim=text_config['qk_nope_head_dim'],
      qk_rope_head_dim=text_config['qk_rope_head_dim'],
      v_head_dim=text_config['v_head_dim'],
      mla_use_output_gate=text_config['mla_use_output_gate'],
      kda_num_heads=linear['num_heads'],
      kda_head_dim=linear['head_dim'],
      kda_conv_kernel_dim=linear['short_conv_kernel_size'],
      kda_gate_lower_bound=linear['gate_lower_bound'],
      # The decay bottleneck is a rank-`head_dim` LoRA (`f_a_proj` is
      # `[head_dim, hidden]` in the release).
      kda_gate_lora_rank=linear['head_dim'],
      num_experts=text_config['num_experts'],
      num_experts_per_token=text_config['num_experts_per_token'],
      routed_expert_latent_dim=text_config['routed_expert_hidden_size'],
      moe_intermediate_size=text_config['moe_intermediate_size'],
      num_shared_experts=text_config['num_shared_experts'],
      routed_scaling_factor=text_config['routed_scaling_factor'],
      moe_renormalize=text_config['moe_renormalize'],
      latent_moe_use_norm=text_config['latent_moe_use_norm'],
      first_k_dense_replace=text_config['first_k_dense_replace'],
      ffn_expand_dim=text_config['intermediate_size'],
      situ_beta=text_config['activation_situ_beta'],
      situ_linear_beta=text_config['activation_situ_linear_beta'],
      use_tied_embedding=text_config.get('tie_word_embeddings', False),
  )
  if not text_config.get('mla_use_nope', True):
    raise ValueError('Kimi K3 MLA is NoPE-only; rotary support is not built.')
  kda_layers = set(linear['kda_layers'])
  expected = tuple(
      LINEAR_ATTENTION if (i + 1) in kda_layers else FULL_ATTENTION
      for i in range(config.n_layers)
  )
  if expected != config.resolved_layer_types():
    config = dataclasses.replace(config, layer_types=expected)
  return dataclasses.replace(config, **overrides) if overrides else config


# --- Sharding ----------------------------------------------------------------


@config_lib.ShardingConfigRegistry.register
def kimi_k3_decoding_sharding() -> config_lib.BaseSharding:
  """Decode placement: same weights, but the token axis is never sharded.

  At decode the token axis is 1 (and during chunked prefill it is a ragged
  buffer of arbitrary length), so sharding it on the EP axis only buys
  padding and reshards. TP *is* kept on the feature axis, which is why core's
  `BaseSharding.to_decoding_sharding()` cannot express this: it nulls the
  *last* (feature) axis and keeps the token shard, the opposite trade. K3's
  AttnRes read sites materialize 8x the residual stream, so dividing them by
  TP is worth the reduce-scatter.
  """
  return dataclasses.replace(
      config_lib.moe_sharding(),
      activation_partition=(('replica', 'data'), None, 'model'),
      attn_activation_partition=(('replica', 'data'), None, 'model', None),
      ffn0_activation_partition=(('replica', 'data'), None, 'model'),
      logits_partition=(('replica', 'data'), None, 'model'),
      data_partition=(('replica', 'data'), None),
  )


# --- Parameter and HBM arithmetic --------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParamCounts:
  """Parameter counts of a `KimiK3ExperimentConfig`, by role."""

  total: int
  routed_experts: int
  # Parameters read per token: the embedding table is excluded (one row per
  # token), the untied head included, and only `num_experts_per_token` of the
  # routed experts fire.
  activated: int


def param_counts(config: KimiK3ExperimentConfig) -> ParamCounts:
  """Counts the parameters of `KimiK3LM.init` without building it."""
  d = config.model_dim
  layer_types = config.resolved_layer_types()

  kda_heads, kda_dim = config.kda_num_heads, config.kda_head_dim
  kda = (
      4 * d * kda_heads * kda_dim  # q, k, v, output gate
      + 3 * kda_heads * kda_dim * config.kda_conv_kernel_dim
      + d * config.kda_gate_lora_rank
      + config.kda_gate_lora_rank * kda_heads * kda_dim
      + kda_heads * kda_dim  # dt_bias
      + kda_heads  # a_log
      + d * kda_heads  # beta
      + kda_dim  # o_norm
      + kda_heads * kda_dim * d  # o_proj
  )
  heads, q_lora, kv_lora = (
      config.n_heads,
      config.q_lora_rank,
      config.kv_lora_rank,
  )
  mla = (
      d * q_lora
      + q_lora
      + q_lora * heads * config.q_head_dim
      + d * (kv_lora + config.qk_rope_head_dim)
      + kv_lora
      + kv_lora * heads * (config.qk_nope_head_dim + config.v_head_dim)
      + (d * heads * config.v_head_dim if config.mla_use_output_gate else 0)
      + heads * config.v_head_dim * d
  )
  # input/post-attention norms plus two AttnRes read sites of {scale, w}.
  per_block_norms = 6 * d

  experts = config.num_experts * config.routed_expert_latent_dim
  routed_per_layer = 3 * experts * config.moe_intermediate_size
  moe = (
      d * config.num_experts
      + config.num_experts  # router bias
      + 2 * d * config.routed_expert_latent_dim  # down_proj, up_proj
      + (config.routed_expert_latent_dim if config.latent_moe_use_norm else 0)
      + routed_per_layer
      + 3 * d * config.shared_expert_intermediate_size
  )
  active_per_layer = (
      3
      * config.num_experts_per_token
      * config.routed_expert_latent_dim
      * config.moe_intermediate_size
  )
  dense = 3 * d * (config.ffn_expand_dim or 0)

  n_moe_layers = max(config.n_layers - config.first_k_dense_replace, 0)
  n_dense_layers = config.n_layers - n_moe_layers
  n_mla = sum(t == FULL_ATTENTION for t in layer_types)
  n_kda = config.n_layers - n_mla

  embed = config.vocab_size * d * (1 if config.use_tied_embedding else 2)
  total = (
      embed
      + 3 * d  # final AttnRes read site {scale, w} and final norm
      + config.n_layers * per_block_norms
      + n_kda * kda
      + n_mla * mla
      + n_dense_layers * dense
      + (n_moe_layers * moe if config.use_moe else n_moe_layers * dense)
  )
  routed = n_moe_layers * routed_per_layer if config.use_moe else 0
  activated = (
      total
      - config.vocab_size * d  # embedding table
      - routed
      + (n_moe_layers * active_per_layer if config.use_moe else 0)
  )
  return ParamCounts(total=total, routed_experts=routed, activated=activated)


@dataclasses.dataclass(frozen=True)
class HbmBudget:
  """Per-device HBM footprint of one deployment, in bytes."""

  devices: int
  weights: int
  mla_cache: int
  kda_state: int
  kda_conv_state: int
  attn_res_snapshots: int
  attn_res_mixture: int

  @property
  def total(self) -> int:
    return (
        self.weights
        + self.mla_cache
        + self.kda_state
        + self.kda_conv_state
        + self.attn_res_snapshots
        + self.attn_res_mixture
    )

  def fits(self, hbm_bytes: int = HBM_BYTES_PER_DEVICE) -> bool:
    return self.total <= hbm_bytes


def hbm_budget(
    config: KimiK3ExperimentConfig,
    mesh_shape: Mapping[str, int],
    *,
    batch_size: int | None = None,
    max_seq_len: int | None = None,
    prefill_chunk_tokens: int = 8192,
    weight_bytes_per_param: float = _BF16,
    kda_state_batch_sharded: bool = True,
) -> HbmBudget:
  """Per-device HBM of the K3 decode state, weights and prefill buffers.

  Weights are divided by the whole device count: `moe_sharding()` covers every
  large tensor with a partition that spans all four axes (see the module
  docstring). It assumes those splits are exact, which is a property of the
  mesh -- 896 experts do not divide by 5.

  Args:
    config: the experiment config.
    mesh_shape: `{'replica','data','seq','model'}` device counts.
    batch_size: sequences in flight; defaults to `config.batch_size`.
    max_seq_len: KV cache depth; defaults to `config.seq_len`.
    prefill_chunk_tokens: tokens issued per prefill call, which is what sizes
      the AttnRes buffers (the mixture is per token, so chunking is exact).
    weight_bytes_per_param: 2 for bf16, 4.25/8 for mxfp4-packed experts.
    kda_state_batch_sharded: `utils/kda.py` follows the activations' batch axis;
      False prices the alternative of replicating the state over it.

  Returns:
    The per-device footprint.
  """
  batch_size = config.batch_size if batch_size is None else batch_size
  max_seq_len = config.seq_len if max_seq_len is None else max_seq_len
  devices = math.prod(mesh_shape.values())
  batch_shards = mesh_shape['replica'] * mesh_shape['data']
  tp = mesh_shape['model']
  layer_types = config.resolved_layer_types()
  n_mla = sum(t == FULL_ATTENTION for t in layer_types)
  n_kda = config.n_layers - n_mla

  local_batch = math.ceil(batch_size / batch_shards)
  local_heads = math.ceil(config.kda_num_heads / tp)
  local_dim = math.ceil(config.model_dim / tp)
  slots = math.ceil(config.n_layers / config.attn_res_block_size)
  kda_batch = local_batch if kda_state_batch_sharded else batch_size

  return HbmBudget(
      devices=devices,
      weights=int(
          param_counts(config).total * weight_bytes_per_param / devices
      ),
      mla_cache=(
          n_mla
          * local_batch
          * max_seq_len
          * (config.kv_lora_rank + config.qk_rope_head_dim)
          * _BF16
      ),
      kda_state=(
          n_kda * kda_batch * local_heads * config.kda_head_dim**2 * _F32
      ),
      kda_conv_state=(
          n_kda
          * kda_batch
          * 3
          * local_heads
          * config.kda_head_dim
          * config.kda_conv_kernel_dim
          * _F32
      ),
      attn_res_snapshots=(
          local_batch * prefill_chunk_tokens * slots * local_dim * _BF16
      ),
      attn_res_mixture=(
          local_batch * prefill_chunk_tokens * (slots + 1) * local_dim * _F32
      ),
  )


# --- Configs -----------------------------------------------------------------


def _released_deployment(
    config: KimiK3ExperimentConfig, *, batch_size: int, seq_len: int
) -> KimiK3ExperimentConfig:
  """Points a config at the released weights and the 4x4x8 placement."""
  return dataclasses.replace(
      config,
      batch_size=batch_size,
      seq_len=seq_len,
      init_ckpt_format=KIMI_K3_CKPT_FORMAT,
      init_ckpt_step=-1,
      # Core's `moe_sharding()` unchanged: the routed expert stacks
      # `[E, in, out]` need the rank-3 `ffn0_partition`/`ffn1_partition` it
      # introduces, which is what `utils/moe.py` reads for
      # `experts/{ffn_0_gate,ffn_0,ffn_1}`.
      sharding_config=config_lib.moe_sharding(),
      decoding_sharding_config=kimi_k3_decoding_sharding(),
      mesh_shape=MESH_4X4X8,
      decoding_mesh_shape=MESH_4X4X8,
      activation_dtype_name='bfloat16',
      use_scan=False,  # No scan path; see `KimiK3ExperimentConfig.use_scan`.
      use_remat=False,
  )


@config_lib.ExperimentConfigRegistry.register
def kimi_k3_2p8t() -> KimiK3ExperimentConfig:
  """The released Kimi K3 (2.8T total / 104B activated), bf16 on 4x4x8."""
  return _released_deployment(
      KimiK3ExperimentConfig(), batch_size=64, seq_len=32768
  )


@config_lib.ExperimentConfigRegistry.register
def kimi_k3_decode() -> KimiK3ExperimentConfig:
  """K3 for inference/eval at 32k context: B=64 on 4x4x8, 33.6 GiB/device.

  Identical to `kimi_k3_2p8t` today: decoding needs no config delta, because
  sampling is not a config field -- the eval drivers take it from flags
  (`--temperature`, `--top_p`, `--top_k`) and the release runs the reasoning
  model at temperature 1.0 with nucleus 0.95 (report S5.1).

  It is still its own registered name because the name is the user interface:
  `--experiment_config=kimi_k3_decode` is what `decode_eval.py` and the README
  launch commands pass, `kimi_k3_decode_ep` is defined as a delta on it, and a
  decode-only knob (paging, KV quantisation) belongs here rather than in the
  config that describes the released deployment.
  """
  return kimi_k3_2p8t()


@config_lib.ExperimentConfigRegistry.register
def kimi_k3_decode_ep() -> KimiK3ExperimentConfig:
  """`kimi_k3_decode` with the routed experts really expert-parallel.

  `ragged_dot` needs its group axis replicated, so GSPMD all-gathers the whole
  expert stack before each of the three grouped matmuls of each of the 92 MoE
  layers: 159 GiB of all-gather output per device per decode step against a
  20.2 GiB weight budget, which is what makes `t_step` flat in batch size.
  Running them under a `shard_map` over the expert axis removes it
  (`KimiK3LatentMoE._expert_parallel_matmuls`, `utils/moe.py`). Measured on 256
  v5p chips at 2.8T: a steady-state batch of 64 takes 349.8 s against the
  shipped path's 1879.8 s -- **5.4x** -- with
  identical accuracy.

  The path assumes the token axis is not sharded over the expert axis, so this
  config puts the *decoding* sharding in `sharding_config` too -- `create_model`
  reads that field, not `decoding_sharding_config`. The cost is that a prefill
  call no longer shards its token axis 8 ways, so its AttnRes buffers grow 8x
  (`[B/8, T, 8 slots, D]` bf16 plus an f32 mixture): ~3 GiB + ~7 GiB per device
  at batch 200 x 1024 tokens, which fits. Keep `batch_size x prefill_size` in
  the few tens of thousands of tokens in any case: above ~50k the non-paged
  prefill program has been seen to hit an XLA rematerialisation RET_CHECK.
  """
  return dataclasses.replace(
      kimi_k3_decode(),
      moe_expert_parallel_axis='seq',
      sharding_config=kimi_k3_decoding_sharding(),
  )


@config_lib.ExperimentConfigRegistry.register
def kimi_k3_tiny_test() -> KimiK3ExperimentConfig:
  """A 4-layer toy K3 with every feature, random init, CPU-runnable.

  Keeps the released topology (3 KDA : 1 MLA with the last layer MLA, a dense
  layer 0, LatentMoE with shared experts, AttnRes snapshots) at ~1/1000 the
  width. `gspmd_sharding()` because a single CPU device needs no mesh.
  """
  return dataclasses.replace(
      KimiK3ExperimentConfig(),
      vocab_size=256,
      model_dim=128,
      n_layers=4,
      attn_res_block_size=2,
      n_heads=4,
      per_head_dim=16,
      q_lora_rank=32,
      kv_lora_rank=16,
      qk_nope_head_dim=16,
      qk_rope_head_dim=8,
      v_head_dim=16,
      kda_num_heads=4,
      kda_head_dim=16,
      kda_gate_lora_rank=16,
      kda_chunk_size=8,
      num_experts=8,
      num_experts_per_token=2,
      routed_expert_latent_dim=32,
      moe_intermediate_size=16,
      num_shared_experts=2,
      ffn_expand_dim=64,
      batch_size=2,
      seq_len=32,
      activation_dtype_name='float32',
      sharding_config=config_lib.gspmd_sharding(),
      decoding_sharding_config=config_lib.gspmd_sharding(),
      mesh_shape=None,
      decoding_mesh_shape=None,
      init_ckpt_dir='',
      init_ckpt_format='',
      use_scan=False,
      use_remat=False,
  )
