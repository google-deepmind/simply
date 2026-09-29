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

"""model_lib fork (thin).

Imports core simply so the data/vocab registrations load. The compute
budget (`training_flops_xla` in final_result.json) is emitted by core
`run_experiment` itself via XLA HLO cost_analysis on the compiled train step,
so this fork adds no training-loop logic. Each seed of the 3-seed sweep is a
separate process with its own `--experiment_dir` (see `main.py`).

This fork also registers a `research_bench_ttt` train loop for the
`pretrain_optimizer_ttt` task: it runs the standard core train loop, then
appends the held-out C4 val-loss CURVE (read back from the run's tb_log
tfevents) into `final_result.json` as `validation_loss_curve`. The
time-to-target metric needs the whole curve (the step at which each loss target
is first reached), not just the final value. This is a TASK-SPECIFIC
customization and deliberately lives here in the experimental scaffolding rather
than in upstream core.
"""

import dataclasses
import functools
import json

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
from simply import model_lib as core
from simply.utils import module
from simply.utils import sharding as sharding_lib
from tasks.research_bench import data_lib


run_experiment = core.run_experiment
TrainLoopRegistry = core.TrainLoopRegistry

# Type aliases into core simply, used by the self-contained PORT model classes
# (FalconH1LM / RecurrentGemmaLM). Each port is a top-level model class
# registered on the shared core `module.ModuleRegistry`, so core `create_model`
# dispatches to it WITHOUT any edit to core model_lib.py.
SimplyConfig = core.SimplyConfig
SimplyModule = core.SimplyModule
PyTree = core.PyTree
PRNGKey = core.PRNGKey
Array = core.Array

# Held-out val-loss scalar metric written to tb_log every
# `validation_eval_interval` steps (C4 validation split). Core prefixes it with
# the validation source's name (see `_val_loss_tag`).
_VAL_LOSS_METRIC = 'eval_loss'


def _val_loss_tag(config) -> str:
  """The tb_log scalar tag core writes the held-out val loss under.

  Core names each validation dataset's metrics `<source name>/<metric>`, where
  the source name is the source's own `name` attribute or its class name (see
  core `run_experiment`). The internal baseline read `c4:3.1.0/eval_loss`; this
  port's C4 source is `data_lib.C4FileSource`, so the tag is derived rather
  than hardcoded.

  Args:
    config: The experiment config.

  Returns:
    The scalar tag of the first validation dataset's eval loss.
  """
  source = config.validation_datasets[0].source
  if isinstance(source, str):
    source = data_lib.DataSourceRegistry.get_instance(source)
  name = getattr(source, 'name', None) or type(source).__name__
  return f'{name}/{_VAL_LOSS_METRIC}'


def _read_val_loss_curve(
    experiment_dir: str, config
) -> list[list[float]] | None:
  """Reads the [[step, val_loss], ...] curve from the run's tb_log tfevents.

  Args:
    experiment_dir: The run's working directory (contains the `tb_log` subdir).
    config: The experiment config (names the validation source, hence the tag).

  Returns:
    The [[step, val_loss], ...] curve, or None (and logs) on any failure so
    curve extraction never breaks a run that otherwise trained and scored fine.
  """
  try:
    # Imported lazily so the core train path has no hard TF-events dependency.
    from tensorboard.backend.event_processing import event_accumulator  # pylint: disable=g-import-not-at-top

    tag = _val_loss_tag(config)
    logdir = (epath.Path(experiment_dir) / 'tb_log').as_posix()
    acc = event_accumulator.EventAccumulator(
        logdir, size_guidance={event_accumulator.SCALARS: 0}
    )
    acc.Reload()
    if tag not in acc.Tags().get('scalars', []):
      logging.warning('research_bench_ttt: val-loss tag %s not in %s', tag, logdir)
      return None
    return [[int(e.step), float(e.value)] for e in acc.Scalars(tag)]
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('research_bench_ttt: failed to read val-loss curve: %s', e)
    return None


@functools.partial(TrainLoopRegistry.register, name='research_bench_ttt')
def run_experiment_with_curve(config, experiment_dir, **kwargs):
  """Core train loop + append the val-loss curve to final_result.json.

  Task-specific loop for `pretrain_optimizer_ttt`: the time-to-target metric
  consumes the whole held-out C4 val-loss curve, so after the standard run we
  read it back from tb_log and rewrite final_result.json with an added
  `validation_loss_curve` field. Core is unchanged.

  Args:
    config: The experiment config.
    experiment_dir: The run's working directory.
    **kwargs: Forwarded to the core `run_experiment`.

  Returns:
    The final_result dict (with `validation_loss_curve` added when available).
  """
  final_result = core.run_experiment(
      config=config, experiment_dir=experiment_dir, **kwargs
  )
  curve = _read_val_loss_curve(experiment_dir, config)
  if final_result is not None:
    if curve is not None:
      final_result['validation_loss_curve'] = curve
    final_result['eval_protocol'] = _eval_protocol(config)
    _write_final_result(final_result, experiment_dir)
  return final_result


# =============================================================================
# bpb pretraining integrity: keep the fixed `training_flops_xla` cap faithful.
# =============================================================================
# Pallas/Mosaic TPU kernels lower to an opaque HLO `custom-call` with this
# target; XLA's cost_analysis does NOT count their interior FLOPs.
_PALLAS_MOSAIC_HLO_MARKER = 'tpu_custom_call'


def _dummy_train_batch(config):
  """A minimal (batch, seq)-shaped token batch to compile the train step."""
  bs, s = config.batch_size, config.seq_len
  toks = jnp.tile(
      jnp.arange(s, dtype=jnp.int32) % max(config.vocab_size - 1, 1), (bs, 1)
  )
  return {'decoder_input_tokens': toks, 'decoder_target_tokens': toks}


def _check_capped_compute_config(config) -> None:
  """Checks the config settings the fixed-compute budget depends on.

  The bpb pretraining tasks report `training_flops_xla` from XLA's HLO
  `cost_analysis`, which counts a `jax.lax.scan` body once regardless of trip
  count and does not count the interior of Pallas/Mosaic kernels. The task spec
  therefore requires `use_scan=False` and `use_flash_attention=False`; this
  fails fast so a run does not report FLOPs that leave out part of its compute.

  Args:
    config: The loaded experiment config.

  Raises:
    ValueError: if either setting is enabled.
  """
  if getattr(config, 'use_scan', False):
    raise ValueError(
        'The pretraining tasks require use_scan=False (unrolled layers): XLA '
        'cost_analysis counts a scan body only once, so a scanned layer stack '
        'reports the FLOPs of a single layer regardless of depth. The '
        'submitted config has use_scan=True. Set use_scan=False and stack '
        'layers without a scan/loop construct (see the task spec).'
    )
  if getattr(config, 'use_flash_attention', False):
    raise ValueError(
        'The pretraining tasks require use_flash_attention=False: splash/flash '
        'attention lowers to a Pallas/Mosaic custom-call whose interior FLOPs '
        'XLA cost_analysis does not count, so training_flops_xla would omit '
        'the attention compute. The submitted config has '
        'use_flash_attention=True. Use standard (XLA) attention (see the task '
        'spec).'
    )


def _assert_no_pallas_custom_call(config) -> str:
  """Checks the compiled train step for Pallas/Mosaic custom kernels.

  XLA `cost_analysis` does not count the interior FLOPs of a Pallas/Mosaic
  kernel -- it lowers to a `tpu_custom_call` that reads as ~0 FLOPs -- so a
  train step routed through one (splash/flash attention, a hand-written
  matmul/attention kernel) would report `training_flops_xla` that leaves out
  that compute. The task spec requires standard XLA ops; this compiles the real
  train step on the job's mesh and fails fast if any such call is present.

  If the check itself cannot run it proceeds anyway (the core run surfaces any
  genuine compile error) and reports that in its return value.

  Args:
    config: The experiment config.

  Returns:
    'ok' if the compiled train step is clean, or 'skipped:<reason>' if the check
    itself could not run (it fails OPEN, but the outcome is recorded in the
    run's `compute_integrity` stamp so a skipped check is never invisible).

  Raises:
    ValueError: if the compiled train step contains a Pallas/Mosaic custom-call.
  """
  try:
    sharding_lib.set_mesh(
        mesh_shape=config.mesh_shape,
        dcn_mesh_shape=config.dcn_mesh_shape,
        axis_names=config.sharding_config.mesh_axis_names,
    )
    model, _ = core.create_model(config, config.sharding_config)
    opt = config.optimizer
    state = opt.init(model.init(jax.random.key(config.model_seed)))

    @functools.partial(jax.jit, static_argnames=['add_log_info'])
    def _train_step(state, batch, lr, add_log_info=False):
      return core.train_one_step(
          state=state, batch=batch, lr=lr, model=model, opt=opt,
          grad_accum_steps=config.grad_accum_steps,
          clip_grad_norm=config.clip_grad_norm,
          clip_update_norm=config.clip_update_norm,
          clip_local_update_rms=config.clip_local_update_rms,
          weight_decay=config.weight_decay,
          add_log_info=add_log_info,
      )

    lr = jnp.asarray(0.01, dtype=jnp.float32)
    compiled = _train_step.lower(
        state, _dummy_train_batch(config), lr
    ).compile()
    hlo_text = compiled.as_text()
  except Exception as e:  # pylint: disable=broad-except
    logging.warning(
        'research_bench_pretrain: custom-call integrity check could not compile the train'
        ' step (%s); proceeding without it. The core run will surface any real '
        'compile error.', e
    )
    return f'skipped:{type(e).__name__}'
  n_custom_calls = (hlo_text or '').count(_PALLAS_MOSAIC_HLO_MARKER)
  if n_custom_calls > 0:
    raise ValueError(
        'The pretraining tasks require standard XLA ops in the train step: XLA'
        ' cost_analysis does not count the FLOPs inside a Pallas/Mosaic kernel,'
        ' so training_flops_xla would leave that compute out. The compiled '
        f'train step contains {n_custom_calls} `tpu_custom_call` op(s) (e.g. '
        'splash/flash attention or a hand-written kernel). Express the '
        'training compute in standard XLA operations (see the task spec).'
    )
  return 'ok'


# =============================================================================
# Protocol provenance: record HOW the number was produced.
# =============================================================================
# Every research-bench train loop (`research_bench_pretrain`, `research_bench_ttt`, `research_bench_rl`)
# stamps `eval_protocol` into final_result.json. `main.py` resolves
# `--experiment_config` in the task-private registry, so a run cannot pick up
# a core config by accident; this is the backstop for everything else -- a
# research-bench config that names another train loop, a hand-rolled loop, a future
# code path that forgets to stamp.
RESEARCH_BENCH_TRAIN_LOOPS = ('research_bench_pretrain', 'research_bench_ttt', 'research_bench_rl')


def record_run_provenance(
    config, final_result, experiment_dir='', config_name='', loop_name=''
):
  """Records how a run was produced into final_result.json, never raises.

  Args:
    config: The experiment config.
    final_result: The dict the train loop returned (may be None).
    experiment_dir: The run's working directory.
    config_name: The resolved `--experiment_config`.
    loop_name: The resolved train-loop name.

  Returns:
    The provenance dict that was recorded (empty if there was no result to
    write it into).
  """
  result = final_result
  if not isinstance(result, dict):
    result = _read_final_result(experiment_dir)
  if not isinstance(result, dict):
    logging.warning(
        'run provenance: no final_result to annotate; the submitted metric '
        'cannot be checked against a protocol'
    )
    return {}
  capped = float(getattr(config, 'fixed_compute_cap_flops', 0.0) or 0.0) > 0.0
  prov = {
      'experiment_config': config_name,
      # The seeds the run ACTUALLY used. A 3-seed submission is three processes
      # with three experiment dirs, so without this the validator can only
      # trust the launcher's directory names.
      'model_seed': getattr(config, 'model_seed', None),
      'dataset_seed': getattr(config, 'dataset_seed', None),
      'train_loop_name': loop_name or getattr(config, 'train_loop_name', None),
      'train_loop_is_research_bench': loop_name in RESEARCH_BENCH_TRAIN_LOOPS,
      'eval_protocol_present': 'eval_protocol' in result,
      'compute_integrity_present': (
          ('compute_integrity' in result) if capped else None
      ),
  }
  result['run_provenance'] = prov
  if not prov['eval_protocol_present'] or not prov['train_loop_is_research_bench']:
    logging.warning(
        'run provenance: %s -- the reported metric was NOT produced by a research-bench '
        'train loop with a pinned eval; it is not comparable without review',
        prov,
    )
  else:
    logging.info('run provenance: %s', prov)
  _write_final_result(result, experiment_dir)
  return prov


def _read_final_result(experiment_dir):
  """Reads back final_result.json, or None when it is absent/unreadable."""
  if not experiment_dir:
    return None
  try:
    fr_path = epath.Path(experiment_dir) / 'final_result.json'
    with fr_path.open('r') as f:
      return json.load(f)
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('failed to read final_result.json: %s', e)
    return None


def _write_final_result(final_result, experiment_dir) -> None:
  """Rewrites final_result.json after adding a provenance block."""
  try:
    fr_path = epath.Path(experiment_dir) / 'final_result.json'
    with fr_path.open('w') as f:
      f.write(json.dumps(final_result, indent=2))
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('failed to rewrite final_result.json: %s', e)


def _eval_protocol(config) -> dict[str, object]:
  """Describes the validation setup the reported metric was produced under.

  Written to `final_result.json` as `eval_protocol`: the validation data source
  and the settings the metric depends on (vocab, sequence length, eval extent
  and batching), so the number is only compared against runs that used the same
  ones.

  Args:
    config: the experiment config.

  Returns:
    A JSON-safe dict describing the eval protocol.
  """
  datasets = config.validation_datasets or ()
  sources = [str(getattr(ds, 'source', '')) for ds in datasets]
  return {
      'validation_sources': sources,
      'vocab_name': str(config.vocab_name),
      'vocab_size': int(config.vocab_size),
      'seq_len': int(config.seq_len),
      'batch_size': int(config.batch_size),
      'validation_num_eval_steps': int(config.validation_num_eval_steps),
      'validation_eval_interval': int(config.validation_eval_interval),
  }


def _stamp_compute_integrity(final_result, experiment_dir, config, checks):
  """Adds a `compute_integrity` block to final_result.json.

  Records which compute checks ran, the run's `training_flops_xla`, and how it
  compares with the task's cap, so the reported compute is self-describing.

  Args:
    final_result: The final_result dict returned by train loop (may be None).
    experiment_dir: The run's working directory.
    config: The experiment config.
    checks: Mapping check-name -> outcome string ('ok' / 'skipped:<reason>').

  Returns:
    The final_result dict with the `compute_integrity` block added.
  """
  if final_result is None:
    return final_result
  cap = float(getattr(config, 'fixed_compute_cap_flops', 0.0) or 0.0)
  flops = final_result.get('training_flops_xla')
  stamp = {
      'checks': dict(checks),
      'cap_flops': cap,
      'training_flops_xla': flops,
      'within_cap': (
          None if (flops is None or cap <= 0) else bool(float(flops) <= cap)
      ),
      # The settings the checks cover, recorded as run.
      'train_loop_name': getattr(config, 'train_loop_name', None) or 'default',
      'use_scan': bool(getattr(config, 'use_scan', False)),
      'use_flash_attention': bool(
          getattr(config, 'use_flash_attention', False)
      ),
  }
  final_result['compute_integrity'] = stamp
  final_result['eval_protocol'] = _eval_protocol(config)
  logging.info('compute_integrity=%s', stamp)
  _write_final_result(final_result, experiment_dir)
  return final_result


def resolve_train_loop(config, loop_name: str):
  """Returns the train loop to run, with the fixed-compute checks if applicable.

  For a task that declares a training-compute cap (`fixed_compute_cap_flops`,
  set by the bpb pretraining configs), the loop is wrapped with
  `with_compute_integrity`. The wrapping follows the config field rather than
  `train_loop_name` so that the checks still apply when a recipe supplies its
  own train loop.

  Args:
    config: The loaded experiment config.
    loop_name: The resolved train-loop name for this run.

  Returns:
    The registered train-loop callable, wrapped with the compute checks when the
    config declares a fixed-compute cap.
  """
  run_experiment_fn = TrainLoopRegistry.get(loop_name)
  if float(getattr(config, 'fixed_compute_cap_flops', 0.0) or 0.0) > 0.0:
    run_experiment_fn = with_compute_integrity(run_experiment_fn)
  return run_experiment_fn


def with_compute_integrity(run_experiment_fn):
  """Wraps a train loop with the fixed-compute checks + the result stamp.

  Idempotent: a loop that is already wrapped is returned unchanged.

  Args:
    run_experiment_fn: the train-loop callable to wrap.

  Returns:
    The wrapped train loop (guards -> run -> stamp).
  """
  if getattr(run_experiment_fn, 'has_compute_integrity', False):
    return run_experiment_fn

  @functools.wraps(run_experiment_fn)
  def wrapped(config, experiment_dir, **kwargs):
    checks = {}
    _check_capped_compute_config(config)
    checks['config_flags'] = 'ok'
    checks['no_pallas_custom_call'] = _assert_no_pallas_custom_call(config)
    final_result = run_experiment_fn(
        config=config, experiment_dir=experiment_dir, **kwargs
    )
    return _stamp_compute_integrity(
        final_result, experiment_dir, config, checks
    )

  wrapped.has_compute_integrity = True  # pyrefly: ignore[missing-attribute]
  return wrapped


@functools.partial(TrainLoopRegistry.register, name='research_bench_pretrain')
@with_compute_integrity
def run_experiment_pretrain(config, experiment_dir, **kwargs):
  """Core train loop plus the fixed-compute checks (bpb pretraining tasks).

  `with_compute_integrity` checks, before training, that the config and the
  compiled train step keep `training_flops_xla` complete (no scanned layer
  stack, no Pallas/Mosaic custom kernels) and records the outcome in
  final_result.json; the rest is the standard core run. Core is unchanged.

  Args:
    config: The experiment config.
    experiment_dir: The run's working directory.
    **kwargs: Forwarded to the core `run_experiment`.

  Returns:
    The final_result dict from the core run.
  """
  return core.run_experiment(
      config=config, experiment_dir=experiment_dir, **kwargs
  )


# =============================================================================
# Model-PORTING task models (STUBS).
#
# Each port is a self-contained top-level model class registered on the shared
# core `module.ModuleRegistry` under its model_name ('FalconH1LM' /
# 'RecurrentGemmaLM'), so core `create_model` dispatches to it via
# ModuleRegistry.get(...) WITHOUT any edit to core model_lib.py. They ship
# as STUBS: every method raises
# NotImplementedError. Implementing the architecture from the task spec (so it
# loads the provided checkpoint and reproduces the reference GSM8K score under
# the FIXED eval) IS the porting task. An implementation may reuse core building
# blocks available on `core` (e.g. rotary_positional_embedding,
# updated_decode_state, create_mask) with NO core change.
# =============================================================================

_FALCON_STUB_MSG = (
    'FalconH1LM is a STUB. Implement the model per the task spec so it loads'
    " the provided checkpoint (init_ckpt_format='FalconH1Format') and"
    ' reproduces the reference GSM8K score under the FIXED eval. See'
    ' config_lib.port_falcon_h1_0p5b for the config fields + checkpoint/vocab'
    ' wiring.'
)

_RG_STUB_MSG = (
    'RecurrentGemmaLM is a STUB. Implement the model per the task spec so it'
    " loads the provided checkpoint (init_ckpt_format='RecurrentGemmaFormat')"
    ' and reproduces the reference GSM8K score under the FIXED eval. See'
    ' config_lib.port_recurrentgemma_2b for the config fields + checkpoint'
    ' and vocab wiring.'
)


@module.ModuleRegistry.register
@dataclasses.dataclass
class FalconH1LM(SimplyModule):
  """Falcon-H1 decoder-only LM (STUB -- implement me).

  Kept REGISTERED on the shared core `module.ModuleRegistry` under
  model_name='FalconH1LM' so the package imports and
  `create_model`/`ModuleRegistry.get('FalconH1LM')` resolve; every method raises
  NotImplementedError until implemented. Implement `init` and `apply` (returning
  `(logits, {'decode_state': ...})`) to load the provided checkpoint and run
  inference.
  """

  config: SimplyConfig
  sharding_config: SimplyConfig | None = None

  def setup(self) -> None:
    raise NotImplementedError(_FALCON_STUB_MSG)

  def init(self, prng_key: PRNGKey) -> PyTree:
    del prng_key
    raise NotImplementedError(_FALCON_STUB_MSG)

  def apply(  # pyrefly: ignore[bad-override]
      self,
      params: PyTree,
      x: Array,
      *,
      segment_ids: Array | None = None,
      segment_positions: Array | None = None,
      extra_inputs: PyTree = None,
      decode_state: PyTree = None,
  ) -> tuple[Array, PyTree]:
    del params, x, segment_ids, segment_positions, extra_inputs, decode_state
    raise NotImplementedError(_FALCON_STUB_MSG)

  def init_decode_state(self, max_seq_len: int) -> PyTree:
    del max_seq_len
    raise NotImplementedError(_FALCON_STUB_MSG)


@module.ModuleRegistry.register
@dataclasses.dataclass
class RecurrentGemmaLM(SimplyModule):
  """RecurrentGemma-2B decoder-only LM (STUB -- implement me).

  Kept REGISTERED on the shared core `module.ModuleRegistry` under
  model_name='RecurrentGemmaLM' so the package imports and
  `create_model`/`ModuleRegistry.get('RecurrentGemmaLM')` resolve; every method
  raises NotImplementedError until implemented. Implement `init` and `apply`
  (returning `(logits, {'decode_state': ...})`) to load the provided checkpoint
  and run inference.
  """

  config: SimplyConfig
  sharding_config: SimplyConfig | None = None

  def setup(self) -> None:
    raise NotImplementedError(_RG_STUB_MSG)

  def init(self, prng_key: PRNGKey) -> PyTree:
    del prng_key
    raise NotImplementedError(_RG_STUB_MSG)

  def apply(  # pyrefly: ignore[bad-override]
      self,
      params: PyTree,
      x: Array,
      *,
      segment_ids: Array | None = None,
      segment_positions: Array | None = None,
      extra_inputs: PyTree = None,
      decode_state: PyTree = None,
  ) -> tuple[Array, PyTree]:
    del params, x, segment_ids, segment_positions, extra_inputs, decode_state
    raise NotImplementedError(_RG_STUB_MSG)

  def init_decode_state(self, max_seq_len: int) -> PyTree:
    del max_seq_len
    raise NotImplementedError(_RG_STUB_MSG)
