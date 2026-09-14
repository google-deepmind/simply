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
"""The released-weights parity gate: statistics, checks and the report.

`..:compare_to_hf_reference` is flags and `main` over this module; the
thresholds it defaults to, and the numbers the README quotes, are outputs of
`run` below.

`run` takes the model builder as an argument rather than calling
`build_model_and_params` itself, so that the reference is loaded and validated
before the multi-minute checkpoint restore is even attempted, and so that the
report and the checks can be tested against a stub model.

Registration of the Qwen3.8 config, model, checkpoint format, chat format and
tokenizer is the caller's job -- `build_model_and_params` looks the experiment
config up by name, as `eval/decode_eval.py` does.
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
import time
from typing import Any

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

from simply import config_lib as core_config_lib
from simply import model_lib
from simply.utils import checkpoint_lib
from simply.utils import common
from simply.utils import sharding
from simply.utils import tokenization


@dataclasses.dataclass(frozen=True)
class Thresholds:
  """Bounds a run must satisfy to pass; `--max_abs_diff` etc. override them.

  The defaults are the values measured on the released Qwen3.8-27B with
  `ref_full_bf16.npz` ("The capital of France is", bf16, 64 layers) plus
  margin -- see README. A longer prompt accumulates more bf16 error (the
  64-token chat golden reaches max|d| 4.8 / top-1 0.97), so relax them
  explicitly for those rather than loosening the defaults.
  """

  max_abs_diff: float = 0.5
  max_kl: float = 5e-3
  min_top1: float = 0.99
  min_hidden_cos: float = 0.999
  require_greedy_match: bool = True


@dataclasses.dataclass(frozen=True)
class Options:
  """What `run` reports on, beside the model itself.

  The defaults are the gate's shipped behaviour; the binary's flags expose
  them, so changing one here changes the flag's default with it.
  """

  ref_path: str
  experiment_config: str = 'qwen3p8_27b'
  activation_dtype: str = 'bfloat16'
  per_layer: bool = True
  greedy_steps: int = 0
  decode_text: bool = True
  output_path: str | None = None
  thresholds: Thresholds = dataclasses.field(default_factory=Thresholds)


@dataclasses.dataclass(frozen=True)
class Reference:
  """The golden dumped by `dump_hf_reference.py`."""

  path: str
  input_ids: np.ndarray
  logits: np.ndarray
  hidden_states: np.ndarray
  decoded: np.ndarray


def load_reference(path: str) -> Reference:
  """Reads a golden `.npz`; a missing continuation reads as an empty one."""
  with np.load(path) as ref:
    return Reference(
        path=path,
        input_ids=np.asarray(ref['input_ids'], dtype=np.int32),
        logits=np.asarray(ref['logits']),
        hidden_states=np.asarray(ref['hidden_states']),
        decoded=np.asarray(
            ref['decoded'] if 'decoded' in ref else np.zeros(0, dtype=np.int64)
        ),
    )


def check_greedy_reference(reference: Reference, greedy_steps: int) -> None:
  """Raises unless the golden can answer a `--greedy_steps` comparison.

  Comparing against an empty reference continuation would report a vacuous
  `match: True`, so this runs before the (multi-minute) checkpoint restore.

  Args:
    reference: the golden.
    greedy_steps: the requested number of greedy steps; 0 asks for no
      comparison and is always fine.

  Raises:
    ValueError: greedy steps were asked for and the golden has no continuation.
  """
  if greedy_steps and not len(reference.decoded):  # pylint: disable=g-explicit-length-test
    raise ValueError(
        f'--greedy_steps={greedy_steps} but {reference.path} carries no'
        ' reference continuation; re-dump the golden with `dump_hf_reference'
        ' --num_decode_steps=N`.'
    )


def diff_stats(got: np.ndarray, ref: np.ndarray) -> dict[str, float]:
  """Elementwise agreement of two same-shaped arrays."""
  # float64 reductions: in float32 the vocab-sized dot products lose ~1e-5 of
  # precision, enough to report cosine > 1 for identical inputs.
  got = got.astype(np.float64)
  ref = ref.astype(np.float64)
  d = np.abs(got - ref)
  denom = np.linalg.norm(got) * np.linalg.norm(ref)
  cos = float(np.sum(got * ref) / denom) if denom else float('nan')
  return {
      'max_abs': float(d.max()),
      'mean_abs': float(d.mean()),
      'rel_fro': float(np.linalg.norm(d) / (np.linalg.norm(ref) + 1e-12)),
      'cosine': cos,
  }


def logits_stats(got: np.ndarray, ref: np.ndarray) -> dict[str, float]:
  """Diff stats plus top-1 agreement and KL(ref || got) per position."""
  stats = diff_stats(got, ref)
  got64 = got.astype(np.float64)
  ref64 = ref.astype(np.float64)

  def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))

  log_p = _log_softmax(ref64)
  log_q = _log_softmax(got64)
  kl = np.sum(np.exp(log_p) * (log_p - log_q), axis=-1)
  top1_got = got64.argmax(axis=-1)
  top1_ref = ref64.argmax(axis=-1)
  stats.update({
      'top1_agreement': float((top1_got == top1_ref).mean()),
      'kl_mean': float(kl.mean()),
      'kl_max': float(kl.max()),
      'top5_agreement_last_pos': float(
          len(
              set(np.argsort(-got64[-1])[:5].tolist())
              & set(np.argsort(-ref64[-1])[:5].tolist())
          )
          / 5.0
      ),
  })
  return stats


# `() -> (model, params, config)`; `build_model_and_params` is the one the
# binary passes, and a stub is what the test passes.
BuildFn = Callable[[], tuple[Any, Any, Any]]


def build_model_and_params(
    config_name: str,
    ckpt_dir: str | None,
    ckpt_step: int,
    activation_dtype: str,
    mesh_shape: Sequence[int] | None,
    output_logits_soft_cap: float | None = None,
) -> tuple[Any, Any, Any]:
  """Creates the model and restores params exactly like `eval/decode_eval`."""
  config = core_config_lib.ExperimentConfigRegistry.get_instance(config_name)
  mesh: Mapping[str, int] | Sequence[int] = (
      mesh_shape
      if mesh_shape is not None
      else core_config_lib.get_default_mesh_shape(config, mode='decode')
  )
  sharding.set_mesh(mesh, axis_names=config.sharding_config.mesh_axis_names)
  decoding_sharding_config = getattr(config, 'decoding_sharding_config', None)
  if decoding_sharding_config is None:
    decoding_sharding_config = config.sharding_config.to_decoding_sharding()
  replace_kwargs: dict[str, Any] = {
      'use_scan': False,
      'use_remat': False,
      'batch_size': 1,
      'mesh_shape': mesh,
      'activation_dtype_name': activation_dtype,
      'sharding_config': decoding_sharding_config,
  }
  if ckpt_dir:
    replace_kwargs['init_ckpt_dir'] = ckpt_dir
    replace_kwargs['init_ckpt_step'] = ckpt_step
  if output_logits_soft_cap is not None:
    replace_kwargs['output_logits_soft_cap'] = output_logits_soft_cap
  config = dataclasses.replace(config, **replace_kwargs)

  model, _ = model_lib.create_model(config)

  def _init_fn():
    params = model.init(jax.random.key(0))
    if activation_dtype == 'bfloat16':
      params = jax.tree_util.tree_map(
          lambda x: jnp.astype(x, jnp.bfloat16), params
      )
    return {'params': params}

  abstract_state = common.eval_abstract_output(_init_fn)
  start = time.time()
  model_state = checkpoint_lib.load_checkpoint_from_dir(
      config.init_ckpt_dir,
      abstract_state,
      config.init_ckpt_step,
      ckpt_format=config.init_ckpt_format,
  )
  logging.info('Checkpoint restored in %.1f s', time.time() - start)
  return model, model_state['params'], config


def forward_per_layer(
    model: Any, params: Any, input_ids: jax.Array
) -> tuple[jax.Array, list[jax.Array]]:
  """Mirrors `Qwen38HybridLM.apply` (use_scan=False) but keeps every hidden."""
  batch, seq_len = input_ids.shape
  segment_positions = jnp.tile(jnp.arange(seq_len)[None, :], (batch, 1))
  segment_ids = jnp.ones_like(segment_positions)
  x = model.embed_linear.embed(params['embed_linear'], input_ids)
  hiddens = [x]
  for i, block in enumerate(model.blocks):
    x, _ = block.apply(
        params[f'block_{i}'],
        x,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
        extra_inputs={},
    )
    hiddens.append(x)
  final = model.final_ln.apply(params['final_ln'], x)
  hiddens.append(final)
  logits = model.embed_linear.apply(params['embed_linear'], final)
  return logits, hiddens


def align_hiddens(
    hiddens: Sequence[np.ndarray], num_ref: int
) -> list[np.ndarray]:
  """Reorders our hiddens to HF's `output_hidden_states` convention.

  HF appends the input of every layer plus the *final-normed* output, i.e.
  [embeddings, out(block_0), ..., out(block_{n-2}), norm(out(block_{n-1}))],
  which drops the un-normed output of the last block.

  Args:
    hiddens: our hiddens, [embeddings, out(block_0), ..., out(block_{n-1}),
      final_norm].
    num_ref: how many hiddens the HF reference holds.

  Returns:
    `num_ref` hiddens in HF's order.
  """
  if len(hiddens) == num_ref + 1:
    return [*hiddens[: num_ref - 1], hiddens[-1]]
  return list(hiddens[:num_ref])


def greedy_continue(
    forward_fn: Any,
    params: Any,
    input_ids: np.ndarray,
    num_steps: int,
    pad_id: int,
) -> np.ndarray:
  """Cache-free greedy decoding on a fixed-length buffer (one compilation).

  Positions past the current step are padding; causality makes them irrelevant
  to the logits we read.

  Args:
    forward_fn: `(params, tokens) -> logits`, jitted by the caller.
    params: model parameters passed straight to `forward_fn`.
    input_ids: the prompt.
    num_steps: how many tokens to generate.
    pad_id: fills the buffer positions not written yet.

  Returns:
    The `num_steps` generated token ids.
  """
  n = len(input_ids)
  buf = np.full((1, n + num_steps), pad_id, dtype=np.int32)
  buf[0, :n] = input_ids
  out = []
  for t in range(num_steps):
    logits = forward_fn(params, jnp.asarray(buf))
    next_id = int(np.asarray(logits[0, n + t - 1]).astype(np.float32).argmax())
    out.append(next_id)
    buf[0, n + t] = next_id
  return np.asarray(out, dtype=np.int64)


def evaluate_checks(
    report: Mapping[str, Any], thresholds: Thresholds
) -> list[dict[str, Any]]:
  """Turns the measured stats into pass/fail checks against the bounds."""
  logits = report['logits']
  checks: list[dict[str, Any]] = [
      {
          'name': 'logits_max_abs',
          'value': logits['max_abs'],
          'bound': f'<= {thresholds.max_abs_diff}',
          'passed': logits['max_abs'] <= thresholds.max_abs_diff,
      },
      {
          'name': 'logits_kl_mean',
          'value': logits['kl_mean'],
          'bound': f'<= {thresholds.max_kl}',
          'passed': logits['kl_mean'] <= thresholds.max_kl,
      },
      {
          'name': 'top1_agreement',
          'value': logits['top1_agreement'],
          'bound': f'>= {thresholds.min_top1}',
          'passed': logits['top1_agreement'] >= thresholds.min_top1,
      },
  ]
  if 'hidden' in report:
    worst = min(h['cosine'] for h in report['hidden'])
    checks.append({
        'name': 'min_hidden_cosine',
        'value': worst,
        'bound': f'>= {thresholds.min_hidden_cos}',
        'passed': worst >= thresholds.min_hidden_cos,
    })
  if 'greedy' in report and thresholds.require_greedy_match:
    greedy = report['greedy']
    checks.append({
        'name': f'greedy_match_{greedy["compared_steps"]}_steps',
        'value': greedy['match'],
        'bound': 'is True',
        'passed': greedy['match'],
    })
  return checks


def _print_flushed(line: str) -> None:
  # A 27B run takes minutes per section and is usually watched through nohup.
  print(line, flush=True)


def run(
    options: Options,
    build_fn: BuildFn,
    print_fn: Callable[[str], None] = _print_flushed,
) -> int:
  """Runs the gate and prints its report.

  Args:
    options: what to compare and how.
    build_fn: called after the golden has been read and validated, since
      restoring the released checkpoint takes minutes. Of the config it
      returns, the report reads `init_ckpt_dir` and `output_logits_soft_cap`,
      and the greedy continuation reads `pad_id` and `vocab_name`.
    print_fn: receives the report line by line.

  Returns:
    The process exit code: 0 if every check passed, 1 otherwise.
  """
  reference = load_reference(options.ref_path)
  check_greedy_reference(reference, options.greedy_steps)
  input_ids = reference.input_ids
  ref_logits = reference.logits
  ref_hidden = reference.hidden_states
  logging.info(
      'reference: input_ids=%s logits=%s hidden_states=%s',
      input_ids,
      ref_logits.shape,
      ref_hidden.shape,
  )

  model, params, config = build_fn()

  report: dict[str, Any] = {
      'experiment_config': options.experiment_config,
      'ckpt_dir': config.init_ckpt_dir,
      'activation_dtype': options.activation_dtype,
      'ref_path': options.ref_path,
      'output_logits_soft_cap': config.output_logits_soft_cap,
      'input_ids': input_ids.tolist(),
  }

  def _emit(line: str = '') -> None:
    print_fn(line)
    if options.output_path:  # Partial results survive a crash / a kill.
      with open(options.output_path, 'w') as f:
        json.dump(report, f, indent=2)

  _emit('==== Qwen3.8 real-weight parity vs HF golden ====')
  for key in (
      'experiment_config',
      'ckpt_dir',
      'activation_dtype',
      'ref_path',
      'output_logits_soft_cap',
      'input_ids',
  ):
    _emit(f'{key:18s}: {report[key]}')

  # jit is required: the modules apply sharding constraints, which are illegal
  # in eager mode under a set mesh.
  forward_fn = jax.jit(lambda p, ids: model.apply(p, ids)[0])

  x = jnp.asarray(input_ids[None, :], dtype=jnp.int32)
  start = time.time()
  logits = jax.block_until_ready(forward_fn(params, x))
  logits = np.asarray(logits[0]).astype(np.float32)
  report['forward_seconds'] = time.time() - start
  report['logits'] = logits_stats(logits, ref_logits)
  _emit(f'model.apply forward in {report["forward_seconds"]:.1f} s')
  _emit(
      'logits: ' + ' '.join(f'{k}={v:.6g}' for k, v in report['logits'].items())
  )

  if options.per_layer:
    start = time.time()
    per_layer_fn = jax.jit(lambda p, ids: forward_per_layer(model, p, ids))
    logits2, hiddens = jax.block_until_ready(per_layer_fn(params, x))
    hiddens = [np.asarray(h[0]).astype(np.float32) for h in hiddens]
    logits2 = np.asarray(logits2[0]).astype(np.float32)
    report['logits_per_layer_path'] = logits_stats(logits2, ref_logits)
    if len(hiddens) != ref_hidden.shape[0]:
      logging.warning(
          'hidden count mismatch: simply=%d ref=%d',
          len(hiddens),
          ref_hidden.shape[0],
      )
    report['logits_simply_path_delta'] = diff_stats(logits2, logits)
    report['hidden'] = [
        diff_stats(h, r)
        for h, r in zip(
            align_hiddens(hiddens, ref_hidden.shape[0]),
            ref_hidden,
            strict=True,
        )
    ]
    _emit(f'per-layer forward in {time.time() - start:.1f} s')
    _emit(
        'logits_per_layer_path: '
        + ' '.join(
            f'{k}={v:.6g}' for k, v in report['logits_per_layer_path'].items()
        )
    )
    _emit(
        'logits_simply_path_delta (model.apply vs per-layer, both Simply): '
        + ' '.join(
            f'{k}={v:.6g}'
            for k, v in report['logits_simply_path_delta'].items()
        )
    )
    report['top10_last_pos'] = [
        {
            'token': int(t),
            'ref': float(ref_logits[-1, t]),
            'model_apply': float(logits[-1, t]),
            'per_layer': float(logits2[-1, t]),
        }
        for t in np.argsort(-ref_logits[-1])[:10]
    ]
    _emit('top-10 reference logits at the last position:')
    for row in report['top10_last_pos']:
      _emit(
          f'  token={row["token"]:7d} ref={row["ref"]:9.4f}'
          f' model_apply={row["model_apply"]:9.4f}'
          f' per_layer={row["per_layer"]:9.4f}'
      )
    _emit(
        'per-layer hidden states (HF convention: 0=embeddings,'
        ' i=out(block_{i-1}), last=final norm):'
    )
    for i, s in enumerate(report['hidden']):
      _emit(
          f'  h[{i:3d}] max_abs={s["max_abs"]:.4g} mean_abs={s["mean_abs"]:.4g}'
          f' rel_fro={s["rel_fro"]:.4g} cos={s["cosine"]:.6f}'
      )

  if options.greedy_steps:
    start = time.time()
    got = greedy_continue(
        forward_fn,
        params,
        input_ids,
        options.greedy_steps,
        getattr(config, 'pad_id', 0),
    )
    ref_decoded = reference.decoded[: options.greedy_steps]
    n_overlap = len(ref_decoded)
    report['greedy'] = {
        'got': got.tolist(),
        'ref': ref_decoded.tolist(),
        'compared_steps': n_overlap,
        'requested_steps': options.greedy_steps,
        'match': bool(np.array_equal(got[:n_overlap], ref_decoded)),
    }
    _emit(f'greedy continuation in {time.time() - start:.1f} s')
    _emit(
        f'greedy match over {n_overlap}/{options.greedy_steps} steps:'
        f' {report["greedy"]["match"]}'
    )
    _emit(f'  got: {report["greedy"]["got"]}')
    _emit(f'  ref: {report["greedy"]["ref"]}')
    if options.decode_text:
      vocab = tokenization.TokenizerRegistry.get_instance(config.vocab_name)
      report['greedy']['prompt_text'] = vocab.decode(input_ids.tolist())
      report['greedy']['got_text'] = vocab.decode(got.tolist())
      _emit(f'prompt text     : {report["greedy"]["prompt_text"]!r}')
      _emit(f'continuation    : {report["greedy"]["got_text"]!r}')

  checks = evaluate_checks(report, options.thresholds)
  report['checks'] = checks
  _emit('checks:')
  for c in checks:
    _emit(
        f'  [{"PASS" if c["passed"] else "FAIL"}] {c["name"]}:'
        f' {c["value"]} vs {c["bound"]}'
    )
  failed = [c['name'] for c in checks if not c['passed']]
  report['passed'] = not failed
  _emit('PASS' if not failed else f'FAIL: {", ".join(failed)}')

  if options.output_path:
    _emit(f'wrote {options.output_path}')
  return 1 if failed else 0
