# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained, task-agnostic RL training loop for the research bench.

This is a thin, readable RL loop that REUSES core simply primitives (model /
optimizer / state construction, the LMInterface sampler, create_train_batch,
train_one_step, ExperimentHelper) but OWNS the orchestration, so the whole
algorithm surface -- sampling, training reward, batch construction, and loss --
lives here + in `rl_algorithms.py` and is fully modifiable by the agent WITHOUT
touching core simply.

Task-agnostic: the base model, the training data + training reward, and the
FIXED held-out eval are all supplied via the config (`config.evaluation` for the
training reward, `config.validation_evaluation` + the validation data source for
the scored eval), so the same loop serves every research-bench RL task
(function-calling, math, ...). The reference algorithm is selected by
`config.rl_algorithm`.

Registered as the `research_bench_rl` train loop. The skeleton per step is:

    sample rollouts (algo.sampling_params)
      -> training reward (algo.train_reward)
      -> build batch (algo.build_batch)
      -> loss + optimizer update (algo.compute_loss via train_one_step)
    ... every validation_eval_interval steps (<= 0 disables the periodic eval),
    ... and always after the final step:
      -> FIXED held-out eval (config.validation_evaluation on the held-out set)
         -> eval_accuracy_history in final_result.json  [the scored metric]

The eval decoding is pinned by the config's `eval_*` fields ONLY -- see
`_held_out_eval` for exactly which inputs the scored metric depends on.

Deliberately readable and single-host (on-policy num_train_steps_per_batch=1).
Checkpoint/resume is supported (see the loop below). NOTE that most of core
`RLExperimentConfig`'s algorithm fields (gamma, use_grpo, ppo_clip_eps_high/low,
policy_ratio_cap, normalize_advantage, max_abs_advantage,
normalize_reward_method, filter_truncated, num_train_steps_per_batch,
max_num_samples_per_train_batch, use_policy_logp_as_sampler_logp, early_stop,
...) belong to core's `rl` loop and are NOT read here: this loop owns its
orchestration and delegates the algorithm to `rl_algorithms.py`, so those
behaviours must be implemented in an `RLAlgorithm` subclass instead of switched
on via the config.
"""

from collections.abc import Mapping
import dataclasses
import functools
import itertools
import time

from absl import logging
import jax
import jax.numpy as jnp
import jax.sharding as js
import numpy as np
from simply import data_lib
from simply import model_lib
from simply import rl_lib
from simply.utils import checkpoint_lib as ckpt_lib
from simply.utils import common
from simply.utils import experiment_helper as exp_helper
from simply.utils import lm_format as lm_format_lib
from simply.utils import sampling_lib
from simply.utils import sharding as sharding_lib
from simply.utils import tokenization
from tasks.research_bench import rl_algorithms

TrainLoopRegistry = model_lib.TrainLoopRegistry


def _resolve_source(ds_cfg):
  """Returns the raw data source (supports __len__/__getitem__) for a config."""
  source = ds_cfg.source
  if isinstance(source, str):
    source = data_lib.DataSourceRegistry.get_instance(source)
  return source


def _build_decoding_stack(config):
  """Builds the decoding model + mesh + abstract params (reused core stack).

  Args:
    config: the experiment config.

  Returns:
    (decoding_model, decoding_mesh, abstract_decoding_params); shared by the
    rollout and held-out-eval interfaces.
  """
  decoding_config = dataclasses.replace(
      config,
      use_scan=False,
      use_remat=False,
      mesh_shape=config.decoding_mesh_shape or config.mesh_shape,
      sharding_config=config.decoding_sharding_config or config.sharding_config,
  )
  axis_names = decoding_config.sharding_config.mesh_axis_names
  decoding_mesh = sharding_lib.create_mesh(
      decoding_config.mesh_shape,
      decoding_config.dcn_mesh_shape,
      axis_names=axis_names,
  )
  if config.decode_reshard == 'jit':
    # `mesh_utils.create_device_mesh` picks a device ORDER per mesh shape, so
    # the decoding mesh generally permutes the training mesh's devices, and a
    # single jit may not name shardings on both ('Received incompatible
    # devices'). Reusing the training mesh's order removes that. It is tied to
    # the 'jit' path deliberately: alignment on its own does not speed the move
    # up, it slows the per-array path down (~5% on the math task), and it only
    # exists to make the jit legal. Self-consistent either way: the decode
    # model, its abstract params and its sharding all come from THIS mesh.
    train_mesh = sharding_lib.create_mesh(
        config.mesh_shape, config.dcn_mesh_shape, axis_names=axis_names
    )
    decoding_mesh = js.Mesh(
        train_mesh.devices.reshape(decoding_mesh.devices.shape),
        axis_names=axis_names,
    )
  decoding_model, _ = model_lib.create_model(decoding_config)
  with js.set_mesh(decoding_mesh):
    abstract_decoding_params = common.eval_abstract_output(
        lambda: jax.tree_util.tree_map(
            lambda x: jnp.astype(x, config.decoding_quant_scheme),
            decoding_model.init(jax.random.key(0)),
        )
    )
  return decoding_model, decoding_mesh, abstract_decoding_params


def _build_lm_interface(
    config, decoding_model, decoding_mesh, tokenizer, lm_format, eos_tokens
):
  """Builds an LMInterface for one prompt format + stop-token set.

  Args:
    config: the experiment config.
    decoding_model: the decoding model (from `_build_decoding_stack`).
    decoding_mesh: the decoding mesh (from `_build_decoding_stack`).
    tokenizer: the vocab.
    lm_format: the prompt format supplying bos/pad ids and its own stop tokens.
    eos_tokens: extra stop tokens, unioned with the format's own.

  Returns:
    The LMInterface.
  """
  extra_eos_tokens = list(set(eos_tokens) | set(lm_format.extra_eos_tokens))
  input_processor = sampling_lib.create_input_processor(
      config,
      vocab=tokenizer,
      bos_id_override=lm_format.bos_id,
      pad_id_override=lm_format.pad_id,
      extra_eos_tokens=extra_eos_tokens,
  )
  with js.set_mesh(decoding_mesh):
    return model_lib.LMInterface(
        decoding_model,
        params=None,
        vocab=tokenizer,
        input_processor=input_processor,
        bos_id=lm_format.bos_id,
        pad_id=lm_format.pad_id,
        extra_eos_tokens=extra_eos_tokens,
    )


def _make_decode_resharder(config, decoding_mesh, abstract_decoding_params):
  """Returns `params -> decoding params` (cast to the decode dtype, resharded).

  The policy is moved from the training mesh to the decoding mesh once per
  step, so this is paid on every rollout, and it is the single largest
  host-side item in an RL step: 2.86 s of a 7.57 s step on one internal
  accelerator and 62.7 s of a 71.2 s step on another, on
  `rl_bfcl_gemma3_1b`, to move ~2 GB of weights
  that HBM bandwidth would move in ~3 ms.

  The cost is not per-array dispatch, it is the cross-mesh `jax.device_put`:
  when the decode target is fully replicated (an `n_kv_heads == 1` model
  collapses the decode `model` axis to 1) no source shard holds a whole array,
  so `device_put` falls back to fetching each array to the host and back.

  `config.decode_reshard` selects the implementation:

  * `per_array` -- the original: `common.convert_array_with_abstract` per
    array, i.e. an un-jitted `jnp.astype` and a `device_put` onto a sharding
    that lives on a different mesh.
  * `device_put` -- one jitted cast for the whole tree, then one batched
    `device_put`. Measured indistinguishable from `per_array`.
  * `jit` (default) -- one jitted cast with `out_shardings` on the decoding
    mesh, so the move compiles into on-device collectives. Requires the two
    meshes to span the devices in the same order, which
    `_build_decoding_stack` arranges for this mode.

  Args:
    config: the experiment config.
    decoding_mesh: the mesh the decode params must live on.
    abstract_decoding_params: their target dtype + sharding.

  Returns:
    A function mapping training params to decoding params.
  """
  mode = config.decode_reshard
  if mode == 'per_array':

    def per_array(params):
      with js.set_mesh(decoding_mesh):
        return jax.tree_util.tree_map(
            common.convert_array_with_abstract,
            params,
            abstract_decoding_params,
        )

    return per_array

  dtypes = jax.tree_util.tree_map(lambda a: a.dtype, abstract_decoding_params)
  out_shardings = jax.tree_util.tree_map(
      lambda a: a.sharding, abstract_decoding_params
  )
  cast = lambda p: jax.tree_util.tree_map(jnp.astype, p, dtypes)

  if mode == 'jit':
    return jax.jit(cast, out_shardings=out_shardings)

  if mode != 'device_put':
    raise ValueError(f'unknown decode_reshard mode {mode!r}')
  cast_fn = jax.jit(cast)

  def device_put(params):
    leaves = jax.tree_util.tree_leaves(params)
    if any(
        x.dtype != d
        for x, d in zip(leaves, jax.tree_util.tree_leaves(dtypes), strict=True)
    ):
      # Cast under the SOURCE mesh, as convert_array_with_abstract does.
      with js.set_mesh(leaves[0].sharding.mesh):
        params = cast_fn(params)
    return jax.device_put(params, out_shardings)

  return device_put


def _aggregate_eval(examples, per_example_acc):
  """Aggregates per-example accuracy into the scored metric + per-category breakdown.

  The scored `eval_accuracy` is the mean per-example accuracy (avg@k for k>1)
  over the FULL eval set -- pooled (micro-averaged) across ALL examples.
  The loop is task-agnostic: WHICH examples are evaluated is controlled entirely
  by the task's eval data source. If examples carry an optional 'category'
  field, per-category accuracies are also reported (diagnostics only; NOT
  scored).

  Args:
    examples: the eval examples (each a Mapping; optional 'category' field).
    per_example_acc: parallel list of per-example accuracy in [0, 1] (avg@k).

  Returns:
    {'eval_accuracy': float, 'by_category': {cat: {'n','accuracy'}}, 'n_scored'}
  """
  cat_correct: dict[str, float] = {}
  cat_count: dict[str, int] = {}
  total = 0.0
  for ex, acc in zip(examples, per_example_acc, strict=True):
    cat = str(ex.get('category', 'all'))
    cat_correct[cat] = cat_correct.get(cat, 0.0) + acc
    cat_count[cat] = cat_count.get(cat, 0) + 1
    total += acc
  n = len(per_example_acc)
  by_category = {
      c: {'n': cat_count[c], 'accuracy': cat_correct[c] / max(cat_count[c], 1)}
      for c in sorted(cat_count)
  }
  return {
      'eval_accuracy': total / max(n, 1),
      'by_category': by_category,
      'n_scored': n,
  }


def _held_out_eval(
    lm_interface,
    decoding_params,
    decoding_mesh,
    eval_ds,
    evaluation,
    lm_format,
    config,
    eval_base_key,
    steps,
):
  """Runs the FIXED held-out eval; returns scored accuracy + per-category breakdown.

  The scored metric is the mean per-example accuracy (avg@k) over the FULL eval
  set (see `_aggregate_eval`). The eval iterates the data source DIRECTLY and
  covers every example (no grain workers / dropped remainder); full coverage is
  asserted below.

  Held-out decoding is FIXED and separate from the training sampler: the eval
  `SamplingParams` are constructed from scratch below (not
  `dataclasses.replace`d from the algorithm's training params), so every
  decode-affecting field comes from an `eval_*` config field or an explicit
  constant -- temperature (`eval_temperature`, default 0 => greedy), sample
  count (`eval_num_samples`, default 1 => pass@1), decode budget
  (`eval_max_decode_steps`), input budget (`eval_max_input_len` /
  `eval_prefill_size`, larger than training so long tool-schema prompts are not
  front-truncated), and the truncation/ordering knobs (`top_k`, `top_p`,
  `sort_by`, `prefill_size`, `min_prefill_size`). The prompt format and stop
  tokens likewise come from the eval-side `validation_lm_format_name` /
  `eval_extra_eos_tokens` (see `run_experiment`). Retuning the training sampler
  or the training prompt format therefore leaves the scored number unchanged.

  Still shared with training, and part of each task's fixed spec: the model and
  its `activation_dtype_name` / `decoding_quant_scheme` / `use_flash_attention`,
  the decoding mesh, and the tokenizer (`vocab_name`).
  `validation_eval_batch_size` is fixed too -- it sets the generate() chunking,
  which for `eval_temperature > 0` changes the per-chunk PRNG draw (and, in low
  precision, the numerics).

  The eval PRNG key is `fold_in(eval_base_key, steps)`: deterministic given the
  step (=> the eval key is stable across preemption/resume) while inheriting
  `eval_base_key`'s layout (created under the decoding mesh), so the jit'd
  generate below does not hit an incompatible-devices error on multi-device
  meshes. (Greedy default decoding does not draw on this sampling key.)

  Args:
    lm_interface: the language-model interface used to decode.
    decoding_params: the model params bundle for decoding.
    decoding_mesh: the device mesh used for eval decoding.
    eval_ds: the held-out eval data source (iterated directly).
    evaluation: the Evaluation defining per-example scoring.
    lm_format: the LM prompt format.
    config: the experiment config (supplies the fixed `eval_*` decoding fields).
    eval_base_key: base PRNG key, folded with `steps` for the eval key.
    steps: current training step count (for the eval key + logging).

  Returns:
    A dict {'eval_accuracy', 'by_category', 'n_scored'} (see `_aggregate_eval`).
  """
  # FIXED eval input budget (larger than training, config-independent) so long
  # tool-schema prompts aren't front-truncated to a silent 0. Falls back to the
  # training input len only if the field is unset.
  eval_input_len = getattr(config, 'eval_max_input_len', 0) or (
      config.sampling_max_input_len
  )
  eval_prefill = getattr(config, 'eval_prefill_size', 0) or eval_input_len
  # Eval sampling, decoupled from training: temperature (0 => greedy),
  # #samples (avg@k; 1 => pass@1), and a FIXED decode budget. All fall back to
  # sane values if a field is unset, but shipped configs set them explicitly.
  eval_k = max(int(getattr(config, 'eval_num_samples', 1) or 1), 1)
  eval_temp = float(getattr(config, 'eval_temperature', 0.0))
  # FIXED eval decode budget (tokens). NOT derived from the training sampler and
  # NOT the raw max_decode_steps sentinel (often huge -> KV-cache OOM). Falls
  # back to the training decode WINDOW (from the config, never from the
  # algorithm's overridable sampling_params) only if the eval field is unset.
  eval_decode = int(getattr(config, 'eval_max_decode_steps', 0) or 0)
  if eval_decode <= 0:
    eval_decode = max(
        int(config.train_max_seq_len) + 1 - int(config.sampling_max_input_len),
        1,
    )
  # max_seq_len must cover the eval input budget PLUS the decode window, else
  # get_decoding_schedule collapses decode window for long prompts (silent 0).
  eval_max_seq_len = eval_input_len + eval_decode + 1
  # Built explicitly, field by field, rather than
  # `dataclasses.replace(algo.sampling_params(), ...)`: a replace() only sets
  # the fields it names, so every unnamed SamplingParams field (top_k, top_p,
  # sort_by, the prefill sizes -- and any field added to SamplingParams later)
  # would carry over from the training sampler. `top_k=1`, for instance, selects
  # the greedy branch of `sample_from_logits` (utils/sampling_lib.py) and would
  # change the temperature>0 evals. Constructing from scratch keeps the eval
  # decode a pure function of the `eval_*` config fields.
  eval_sampling_params = model_lib.SamplingParams(
      temperature=eval_temp,
      top_k=-1,  # FIXED: full-vocab sampling (top_k=1 would force greedy).
      top_p=1.0,  # FIXED: no nucleus truncation.
      max_decode_steps=eval_decode,
      intermediate_decode_steps=eval_decode,
      max_input_len=eval_input_len,
      max_seq_len=eval_max_seq_len,
      num_samples=eval_k,
      # Deterministic prefill: pin both the size and the inference fallback so
      # the decoding schedule cannot depend on the training sampler.
      prefill_size=eval_prefill,
      min_prefill_size=eval_prefill,
      sort_by=None,  # FIXED: keep sample order; avg@k is order-invariant.
      # Pad decode buffers to a reusable grid to avoid recompiling across
      # varying prompt lengths during evaluation.
      decode_buffer_multiple=config.eval_decode_buffer_multiple,
  )
  eval_batch_size = (
      config.validation_eval_batch_size
      if config.validation_eval_batch_size > 0
      else config.batch_size
  )
  examples = []
  per_example_acc = []
  n = len(eval_ds)
  with js.set_mesh(decoding_mesh):
    # Deterministic per-step eval key = fold_in(eval_base_key, steps), computed
    # UNDER the decoding mesh so its sharding matches generate's jit context
    # (the ambient module-level mesh is the training mesh, a different order).
    prng_key = jax.random.fold_in(eval_base_key, steps)
    # Chunk the eval set into eval_batch_size-sized generate() calls.
    for start in range(0, n, eval_batch_size):
      chunk = [
          eval_ds[i] for i in range(start, min(start + eval_batch_size, n))
      ]
      prompts = [evaluation.get_sampling_input(ex, lm_format) for ex in chunk]
      prng_key, subkey = jax.random.split(prng_key)
      outputs = lm_interface.generate(
          prompts,
          prng_key=subkey,
          params=decoding_params,
          sampling_params=eval_sampling_params,
          prefill_size=eval_prefill,
          scoring_inputs=False,
          batch_size=eval_batch_size,
      )
      for ex, outs in zip(chunk, outputs, strict=True):
        # avg@k: mean correctness over the k samples for this example.
        per_sample = [
            evaluation.evaluate(ex, so.output_text)['correct'] for so in outs
        ]
        examples.append(ex)
        per_example_acc.append(sum(per_sample) / max(len(per_sample), 1))
  # Full-coverage guarantee: every example in the source was scored (no dropped
  # remainder / worker sharding, unlike a grain eval iterator).
  assert len(examples) == n, f'eval covered {len(examples)}/{n} examples'
  return _aggregate_eval(examples, per_example_acc)


def _eval_protocol(config, evaluation, lm_format_name, eos_tokens, n_scored):
  """Describes the scored eval exactly as this run performed it.

  Written to `final_result.json` as `eval_protocol` so the reported number
  carries the setup it was produced under: the data source, the decode
  parameters, the prompt format and stop tokens, the tokenizer/dtypes, and the
  Evaluation class together with any constructor arguments that differ from its
  defaults (`few_shot`, `system_message`, ... are ordinary dataclass fields, so
  the metric is only comparable across runs that used the same ones).

  Args:
    config: the experiment config.
    evaluation: the Evaluation instance used for the scored eval.
    lm_format_name: the resolved eval prompt-format name.
    eos_tokens: the resolved eval stop tokens.
    n_scored: number of examples actually scored.

  Returns:
    A JSON-safe dict describing the eval protocol.
  """
  ctor_args = {}
  if dataclasses.is_dataclass(evaluation):
    for f in dataclasses.fields(evaluation):
      value = getattr(evaluation, f.name, None)
      if isinstance(value, (bool, int, float, str)) or value is None:
        ctor_args[f.name] = value
      else:
        ctor_args[f.name] = f'<{type(value).__name__}>'
  return {
      'evaluation': type(evaluation).__name__,
      'evaluation_args': ctor_args,
      'eval_source': str(getattr(config.validation_datasets[0], 'source', '')),
      'n_scored': n_scored,
      'lm_format_name': lm_format_name,
      'extra_eos_tokens': sorted(set(eos_tokens)),
      'vocab_name': config.vocab_name,
      'eval_temperature': float(getattr(config, 'eval_temperature', 0.0)),
      'eval_num_samples': int(getattr(config, 'eval_num_samples', 1)),
      'eval_max_decode_steps': int(getattr(config, 'eval_max_decode_steps', 0)),
      'eval_max_input_len': int(getattr(config, 'eval_max_input_len', 0)),
      'eval_prefill_size': int(getattr(config, 'eval_prefill_size', 0)),
      'validation_eval_batch_size': int(config.validation_eval_batch_size),
      'activation_dtype_name': str(config.activation_dtype_name),
      'decoding_quant_scheme': str(config.decoding_quant_scheme),
  }


def make_experiment_helper(config, experiment_dir: str):
  """The loop's `ExperimentHelper`, split out so a test can build one.

  Everything else in `run_experiment` needs a checkpoint and an accelerator, so
  a kwarg this package passes that core does not accept would otherwise only
  surface on a TPU, minutes into a run (it did, once: the internal helper's
  `write_to_datatable`).
  """
  return exp_helper.ExperimentHelper(
      experiment_dir,
      ckpt_interval=config.ckpt_interval,
      ckpt_max_to_keep=config.ckpt_max_to_keep,
      num_train_steps=config.num_train_steps,
      metric_log_interval=config.tb_log_interval,
      should_save_ckpt=config.should_save_ckpt,
  )


@functools.partial(TrainLoopRegistry.register, name='research_bench_rl')
def run_experiment(config, experiment_dir='', **kwargs):
  """Self-contained, task-agnostic research-bench RL loop (see module docstring)."""
  del kwargs
  # Single-host by design: the batch-assembly path builds the global training
  # batch from ONE process's rollouts.
  if jax.process_count() > 1:
    raise NotImplementedError(
        'research_bench_rl is single-host only, but jax.process_count() =='
        f' {jax.process_count()}. Use a single-host slice (e.g. v6e-4 or'
        ' v6e-8), or implement multi-host rollout sharding + a'
        ' process_allgather of per-process valid counts in the batch-build path'
        ' (rl_loop.py + rl_algorithms.build_batch; cf. core'
        ' rl_lib.create_train_batch).'
    )
  algo = rl_algorithms.RLAlgorithmRegistry.get(config.rl_algorithm)(config)

  sharding_lib.set_mesh(
      mesh_shape=config.mesh_shape,
      dcn_mesh_shape=config.dcn_mesh_shape,
      axis_names=config.sharding_config.mesh_axis_names,
  )
  helper = make_experiment_helper(config, experiment_dir)
  model, _ = model_lib.create_model(config, config.sharding_config)
  helper.save_config_info(config, config.sharding_config, model)
  opt = config.optimizer
  state = model_lib.get_init_state(
      config, config.sharding_config, helper.ckpt_mngr, helper.ckpt_dir
  )

  # Reference params (for the KL penalty) = the FROZEN init (pretrained) policy.
  # On a RESUME (a ckpt exists), state['params'] is the mid-training policy, so
  # we must reload the reference from init_ckpt_dir (mirrors core rl_lib), else
  # the KL objective would silently change after preemption. On a fresh run,
  # state['params'] is the init policy, so use it directly.
  ref_params = None
  if config.use_ref_params:
    if helper.ckpt_mngr and helper.ckpt_mngr.latest_step() is not None:
      abstract_state = {'params': ckpt_lib.get_abstract_params(model)}
      ref_state = ckpt_lib.load_checkpoint_from_dir(
          config.init_ckpt_dir,
          abstract_state,
          config.init_ckpt_step,
          ckpt_format=config.init_ckpt_format,
      )
      ref_params = ref_state['params']
    else:
      ref_params = state['params']
    ref_params = jax.tree_util.tree_map(
        lambda x: jnp.array(x, config.ref_params_dtype), ref_params
    )

  lr_fn = common.named_jit(model_lib.create_lr_schedule(config), 'lr_fn')

  @functools.partial(
      jax.jit, donate_argnames=['state'], static_argnames=['add_log_info']
  )
  def train_one_step_fn(state, batch, lr, add_log_info=False):
    return model_lib.train_one_step(
        state=state,
        batch=batch,
        lr=lr,
        model=model,
        opt=opt,
        custom_loss_fn=algo.compute_loss,
        grad_accum_steps=config.grad_accum_steps,
        clip_grad_norm=config.clip_grad_norm,
        clip_update_norm=config.clip_update_norm,
        clip_local_update_rms=config.clip_local_update_rms,
        weight_decay=config.weight_decay,
        add_log_info=add_log_info,
    )

  # Chunk the reference-policy (KL) forward pass into grad-accum-sized
  # microbatches, matching core rl_lib. Otherwise the second forward runs at the
  # full train_batch_size and blows HBM (the very thing the pf configs try to
  # bound). None when no accumulation.
  logprobs_microbatch_size = None
  if config.grad_accum_steps and config.grad_accum_steps > 1:
    logprobs_microbatch_size = (
        config.train_batch_size // config.grad_accum_steps
    )
  compute_logprobs_fn = common.named_jit(
      rl_lib.compute_logprobs,
      'compute_logprobs_fn',
      model=model,
      microbatch_size=logprobs_microbatch_size,
  )

  tokenizer = tokenization.TokenizerRegistry.get(config.vocab_name)()
  decoding_model, decoding_mesh, abstract_decoding_params = (
      _build_decoding_stack(config)
  )
  reshard_for_decode = _make_decode_resharder(
      config, decoding_mesh, abstract_decoding_params
  )
  # Two prompt formats / stop-token sets, one decoding stack:
  #  * rollouts use `lm_format_name` + `extra_eos_tokens`, which the training
  #    recipe may change;
  #  * the held-out eval uses `validation_lm_format_name` +
  #    `eval_extra_eos_tokens`, which the task fixes, so the scored prompt and
  #    stop set stay the same when the training prompt format changes. Each
  #    falls back to its training counterpart when unset, and the eval reuses
  #    the rollout interface when both are identical (the shipped baselines).
  lm_format = lm_format_lib.LMFormatRegistry.get(config.lm_format_name)()
  eval_lm_format_name = (
      getattr(config, 'validation_lm_format_name', '') or config.lm_format_name
  )
  eval_lm_format = lm_format_lib.LMFormatRegistry.get(eval_lm_format_name)()
  eval_eos_tokens = tuple(
      getattr(config, 'eval_extra_eos_tokens', ()) or config.extra_eos_tokens
  )
  lm_interface = _build_lm_interface(
      config,
      decoding_model,
      decoding_mesh,
      tokenizer,
      lm_format,
      config.extra_eos_tokens,
  )
  if eval_lm_format_name == config.lm_format_name and set(
      eval_eos_tokens
  ) == set(config.extra_eos_tokens):
    eval_lm_interface = lm_interface
  else:
    eval_lm_interface = _build_lm_interface(
        config,
        decoding_model,
        decoding_mesh,
        tokenizer,
        eval_lm_format,
        eval_eos_tokens,
    )
    logging.info(
        'held-out eval uses its own prompt format %s (rollouts use %s)',
        eval_lm_format_name,
        config.lm_format_name,
    )
  # Rollout decoding only -- the scored eval builds its own params, see
  # `_held_out_eval`.
  sampling_params = algo.sampling_params()

  evaluation = config.evaluation  # training reward source
  eval_evaluation = config.validation_evaluation  # FIXED scored eval
  train_ds = _resolve_source(config.dataset)
  # The scored eval is single-source by design: only validation_datasets[0] is
  # read (its `packing` / `lm_format_name` are unused -- the eval iterates the
  # raw source and formats with `lm_format`), and any further entry is ignored.
  eval_ds = _resolve_source(config.validation_datasets[0])

  # Create the sampling PRNG keys UNDER the DECODING mesh so they inherit its
  # device order. `lm_interface.generate` jnp.copy's the key inside a jit whose
  # context mesh is decoding mesh; a key made under the (differently ordered)
  # training mesh -- or with no mesh -- has a permuted device layout and crashes
  # with "Received incompatible devices ... key<fry>[]". This matches how core
  # rl_lib creates its key inside the decoding-mesh context.
  #  * prng_key      : evolving training-rollout key (split each step).
  #  * eval_base_key : SEPARATE, never-split base for the held-out eval; each
  #                    eval folds in `steps` for a deterministic, resume-
  #                    invariant key (not re-seeded on resume; fixes M2).
  with js.set_mesh(decoding_mesh):
    prng_key = jax.random.key(seed=config.model_seed)
    eval_base_key = jax.random.key(seed=config.model_seed + 100003)
  # Restore the eval-accuracy history on resume so the curve (and the scored
  # last-K-mean) isn't truncated after preemption.
  eval_history: list[float] = []
  if helper.ckpt_mngr and helper.ckpt_mngr.latest_step() is not None:
    data_state = ckpt_lib.load_data_state_from_dir(
        helper.ckpt_dir, helper.ckpt_mngr.latest_step()  # pyrefly: ignore[bad-argument-type]
    )
    if isinstance(data_state, Mapping):
      eval_history = list(data_state.get('eval_accuracy_history', []) or [])  # pyrefly: ignore[bad-argument-type, bad-assignment]
  final_result = {'eval_accuracy_history': eval_history}
  # Resume-aware: get_init_state restores from the latest ckpt (if any), so we
  # start from the restored step count. This lets a preempted run resume
  # instead of restarting from 0 (important under a fixed wall-clock budget).
  steps = int(state['steps'])
  n_train = len(train_ds)
  # Data order is a fixed permutation of `dataset_seed`, indexed by a
  # resume-safe cursor (a pure function of `steps`). NOTE: the submission
  # protocol for the RL tasks sweeps `model_seed` only (42/43/44), so the three
  # seeds share this ordering and differ through the rollout sampling key
  # (`prng_key` below); sweep `dataset_seed` too if you also want data-order
  # variation.
  order = np.random.default_rng(config.dataset_seed).permutation(n_train)
  cursor = (steps * config.batch_size) % max(n_train, 1)
  # Resume-safe eval history requires each saved checkpoint to land on an eval
  # step (ckpt interval a multiple of the eval interval), else the restored eval
  # curve could gap or duplicate across preemption.
  if config.should_save_ckpt and config.validation_eval_interval > 0:
    assert config.ckpt_interval % config.validation_eval_interval == 0, (
        f'ckpt_interval ({config.ckpt_interval}) must be a multiple of '
        f'validation_eval_interval ({config.validation_eval_interval})'
    )

  def _do_eval(label):
    """Runs the FIXED held-out eval on the CURRENT model, keyed by `label`."""
    # Deterministic key = fold_in(eval_base_key, label): reproducible across
    # resume, NOT the evolving training prng, and mesh-compatible via
    # eval_base_key's device layout.
    decoding_params = reshard_for_decode(state['params'])
    eval_res = _held_out_eval(
        eval_lm_interface,
        decoding_params,
        decoding_mesh,
        eval_ds,
        eval_evaluation,
        eval_lm_format,
        config,
        eval_base_key,
        label,
    )
    del decoding_params
    acc = eval_res['eval_accuracy']
    final_result['eval_accuracy'] = acc
    final_result['eval_accuracy_history'].append(acc)
    # Per-category diagnostics (call-required categories only); not scored.
    final_result['eval_accuracy_by_category'] = eval_res['by_category']
    scalars = {'eval_accuracy': acc}
    for _cat, _stats in eval_res['by_category'].items():
      scalars[f'eval_accuracy_cat/{_cat}'] = _stats['accuracy']
    helper.write_scalars(label, scalars)
    helper.flush()
    logging.info(
        'step %d: held-out eval_accuracy=%.4f (n_scored=%d) by_category=%s',
        label, acc, eval_res['n_scored'], eval_res['by_category'],
    )

  # Run exactly `num_train_steps` optimizer updates: `steps` counts COMPLETED
  # updates, so the body runs for steps = 0 .. num_train_steps - 1. (The former
  # `<=` ran one extra update past the declared budget.)
  while steps < config.num_train_steps:
    t0 = time.time()
    # ---- sample rollouts for a batch of prompts ----
    decoding_params = reshard_for_decode(state['params'])
    prompts, examples = [], []
    for _ in range(config.batch_size):
      ex = train_ds[int(order[cursor % n_train])]
      cursor += 1
      examples.append(ex)
      prompts.append(evaluation.get_sampling_input(ex, lm_format))
    # Split UNDER the decoding mesh: the ambient (module-level) mesh is the
    # TRAINING mesh, whose device order can differ from the decoding mesh; a
    # split done there yields a key whose sharding mismatches generate's jit
    # context mesh ("incompatible devices ... _threefry_split").
    with js.set_mesh(decoding_mesh):
      prng_key, subkey = jax.random.split(prng_key)
      outputs = lm_interface.generate(
          prompts,
          prng_key=subkey,
          params=decoding_params,
          sampling_params=sampling_params,  # num_samples_per_example rollouts.
          prefill_size=config.sampling_prefill_size,
          scoring_inputs=False,
      )
    del decoding_params

    # ---- training reward (algorithm hook) + assemble rewarded samples ----
    rewarded = {}
    for idx, (ex, outs) in enumerate(zip(examples, outputs, strict=True)):
      group = []
      for so in outs:
        r = algo.train_reward(ex, so.output_text)
        rs = rl_lib.RewardedSample(
            raw_example=ex,
            sampling_input=prompts[idx],
            step=steps,
            in_batch_example_index=idx,
            sampling_output=so,
            # `train_reward` MUST return 'reward' and 'correct' (see
            # RLAlgorithm.train_reward); update_with_evaluation_result below
            # requires both, so read 'reward' directly instead of defaulting it.
            is_valid_for_training=not bool(np.isnan(r['reward'])),
        ).update_with_evaluation_result(r)
        group.append(rs)
      rewarded[idx] = group
    all_rollouts = list(itertools.chain.from_iterable(rewarded.values()))
    num_valid = np.array(
        [sum(x.is_valid_for_training for x in all_rollouts)]
    )
    _valid = [x for x in all_rollouts if x.is_valid_for_training]
    if _valid:
      # Keep the two distinct: `reward/mean` is the (shapeable) TRAINING reward,
      # `train_accuracy` is the underlying binary correctness. They coincide for
      # the shipped exact-match rewards but diverge as soon as an algorithm
      # shapes the reward (partial credit, format bonus, ...).
      helper.add_metric(
          'reward/mean', float(np.mean([float(x.reward) for x in _valid]))
      )
      helper.add_metric(
          'train_accuracy', float(np.mean([float(x.correct) for x in _valid]))
      )
    sampling_time = time.time() - t0
    helper.add_metric('sampling_time', sampling_time)

    # ---- build training batch (algorithm hook) ----
    batch = algo.build_batch(
        rewarded,
        num_valid,
        train_batch_size=config.train_batch_size,
        max_seq_len=config.train_max_seq_len,
        ref_params=ref_params,
        compute_logprobs_fn=compute_logprobs_fn,
    )
    # Rows that actually contribute a gradient: `create_train_batch` pads up to
    # `train_batch_size` and keeps only the first `train_batch_size` valid rows.
    helper.add_metric(
        'effective_train_batch_size',
        float(np.sum(np.asarray(batch.is_valid_for_training))),
    )

    # ---- loss + optimizer update ----
    # Save a checkpoint before the update so a preempted run can resume
    # (no-op when should_save_ckpt=False). helper honours ckpt_interval.
    helper.save_ckpt(
        state,
        steps,
        # Copy: the list keeps growing during the step, and the checkpoint at
        # step N must record exactly the evals up to step N.
        data={
            'eval_accuracy_history': list(
                final_result['eval_accuracy_history']
            )
        },
    )
    lr = lr_fn(state['steps'])
    loss, state, log_dict = train_one_step_fn(state, batch, lr=lr)
    # After this update the model has `new_steps = steps + 1` completed updates;
    # ALL metrics from this iteration (training + held-out eval) are keyed by
    # new_steps, so "step N" consistently means "after N optimizer updates".
    new_steps = int(state['steps'])
    helper.add_metric('loss', float(loss))
    # Surface the algorithm's own metrics (entropy, kl, advantage, ...). Every
    # key is optional: a custom `compute_loss` only has to return 'loss_weight'
    # (required by core's grad accumulation). 'loss' is skipped because it was
    # just added from the returned scalar -- adding it twice would halve the
    # metric aggregator's rolling averaging window for it.
    for k, v in log_dict.items():
      if k == 'loss':
        continue
      try:
        helper.add_metric(k, float(v))
      except (TypeError, ValueError):
        pass
    if helper.should_log_metrics(new_steps):
      md = dict(lr=lr)
      md.update(helper.get_aggregated_metrics())
      helper.write_scalars(new_steps, md)
      helper.flush()

    # ---- FIXED held-out eval (on post-update model, keyed by new_steps) ----
    # `validation_eval_interval <= 0` DISABLES the periodic eval (matching the
    # resume-invariant assert above, which only applies when it is > 0); the
    # final-step eval always runs so a scored metric always exists.
    if new_steps == config.num_train_steps or (
        config.validation_eval_interval > 0
        and new_steps % config.validation_eval_interval == 0
    ):
      _do_eval(new_steps)

    steps = new_steps

  # Guarantee a scored metric even when the loop body never ran (e.g.
  # num_train_steps == 0): evaluate the current (pristine / resumed) model once.
  # On a resume that already reached the budget, reuse the last recorded eval
  # instead of appending a duplicate.
  if not final_result['eval_accuracy_history']:
    _do_eval(steps)
  elif 'eval_accuracy' not in final_result:
    final_result['eval_accuracy'] = final_result['eval_accuracy_history'][-1]  # pyrefly: ignore[bad-assignment]

  # Persist the FINAL model + full eval history: loop checkpoints BEFORE each
  # update, so without this the post-final-update state would not be
  # checkpointed. The helper's save policy is
  # AnySavePolicy(FixedInterval(ckpt_interval),
  # SpecificSteps([num_train_steps])) (see
  # utils/experiment_helper.ckpt_save_policy), so this final write always lands,
  # and re-saving an already-saved step is a no-op.
  helper.save_ckpt(
      state,
      steps,
      data={
          'eval_accuracy_history': list(final_result['eval_accuracy_history'])
      },
  )
  final_result['eval_protocol'] = _eval_protocol(
      config,
      eval_evaluation,
      eval_lm_format_name,
      eval_eos_tokens,
      len(eval_ds),
  )
  helper.close(final_result)
  return final_result
