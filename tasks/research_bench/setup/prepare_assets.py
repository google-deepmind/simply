#!/usr/bin/env python3
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

r"""One CLI that fetches every asset the research-bench tasks need.

Run from the repository root. Everything lands in the canonical simply cache
(`SIMPLY_MODELS` / `SIMPLY_DATASETS` / `SIMPLY_VOCABS`); `--gcs-bucket` also
mirrors it to `gs://<bucket>/assets/{models,datasets,vocabs}` for Cloud TPU VMs
and prints the environment a VM should export.

Every subcommand is idempotent and resumable: an asset that is already present
and passes its check is skipped, so a re-run after an interruption costs only
the missing pieces.

  # everything for one task (the usual entry point)
  python -m tasks.research_bench.setup.prepare_assets task rl_gemma3_1b

  # everything for every task, mirrored to GCS
  python -m tasks.research_bench.setup.prepare_assets all \
      --gcs-bucket gs://my-bucket

  # individual groups
  python -m tasks.research_bench.setup.prepare_assets c4 --num-train-shards 32
  python -m tasks.research_bench.setup.prepare_assets vocabs --train-openmix
  python -m tasks.research_bench.setup.prepare_assets datasets
  python -m tasks.research_bench.setup.prepare_assets models --only gemma3_1b_pt

  # report what is staged and whether it still checks out
  python -m tasks.research_bench.setup.prepare_assets verify
  python -m tasks.research_bench.setup.prepare_assets plan
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Sequence

from tasks.research_bench.setup import asset_lib
from tasks.research_bench.setup import build_c4
from tasks.research_bench.setup import build_datasets
from tasks.research_bench.setup import build_models
from tasks.research_bench.setup import build_vocabs

# What each task needs. Asset ids are 'c4', 'vocab:<name>', 'dataset:<name>'
# and 'model:<ModelSpec.name>'.
TASK_ASSETS: dict[str, tuple[str, ...]] = {
    'pretrain_bpb_v32k': ('c4', 'vocab:nanodo_c4'),
    'pretrain_bpb_byte': ('c4',),
    'pretrain_optimizer_ttt': ('c4', 'vocab:openmix_substitute'),
    'rl_gemma3_1b': ('dataset:gsm8k', 'vocab:gemma3', 'model:gemma3_1b_pt'),
    'rl_qwen2p5_math_1p5b': (
        'dataset:math500',
        'dataset:deepscaler',
        'vocab:qwen2p5',
        'model:qwen2p5_math_1p5b',
    ),
    'rl_bfcl_qwen3_0p6b': (
        'dataset:bfcl',
        'dataset:toolace',
        'vocab:qwen3',
        'model:qwen3_0p6b_base',
    ),
    'rl_bfcl_gemma3_1b': (
        'dataset:bfcl',
        'dataset:toolace',
        'vocab:gemma3',
        'model:gemma3_1b_pt',
    ),
    'sampling_lcb': ('dataset:livecodebench', 'vocab:qwen3', 'model:qwen3_4b'),
    'decode_efficiency_vf': (
        'dataset:aime',
        'vocab:qwen3',
        'model:qwen3_30b_a3b_thinking_2507',
    ),
    'port_falcon_h1_0p5b': ('dataset:gsm8k', 'model:falcon_h1_0p5b_base'),
    'port_recurrentgemma_2b': ('dataset:gsm8k', 'model:recurrentgemma_2b'),
}

DATASET_BUILDERS = {
    'gsm8k': (build_datasets.build_gsm8k, 'gsm8k'),
    'math500': (build_datasets.build_math500, 'math500'),
    'deepscaler': (build_datasets.build_deepscaler, 'deepscaler'),
    'aime': (build_datasets.build_aime, 'aime'),
    'bfcl': (build_datasets.build_bfcl, 'tooluse_rlvr/bfcl'),
    'toolace': (build_datasets.build_toolace, 'tooluse_rlvr/toolace'),
    'livecodebench': (build_datasets.build_livecodebench, 'livecodebench'),
}

DATASET_SOURCES = {
    'gsm8k': build_datasets.SIMPLY_DATASETS_REPO,
    'math500': build_datasets.MATH500_REPO,
    'deepscaler': build_datasets.DEEPSCALER_REPO,
    'aime': f'{build_datasets.SIMPLY_DATASETS_REPO} + {build_datasets.AIME25_REPO}',
    'bfcl': build_datasets.BFCL_REPO,
    'toolace': build_datasets.TOOLACE_REPO,
    'livecodebench': build_datasets.LCB_REPO,
}


VOCAB_NAMES = (
    'nanodo_c4',
    'gemma3',
    'qwen2p5',
    'qwen3',
    'openmix_substitute',
)

# Asset id -> manifest key (they differ where the staged file is named after
# the registered vocab rather than the CLI selector).
MANIFEST_KEY = {
    'c4': 'c4_bin',
    'vocab:nanodo_c4': 'vocab_nanodo_c4',
    'vocab:gemma3': 'vocab_vb262144_gemma3',
    'vocab:qwen2p5': 'vocab_qwen2p5',
    'vocab:qwen3': 'vocab_qwen3',
    'vocab:openmix_substitute': 'vocab_openmix_substitute',
    **{f'dataset:{n}': f'dataset_{n}' for n in DATASET_BUILDERS},
    **{f'model:{n}': n for n in build_models.MODELS_BY_NAME},
}


def _c4_dir() -> str:
  return os.getenv('SIMPLY_C4_DIR', build_c4.DEFAULT_OUT_DIR)


def _reject_unknown(
    selected: Sequence[str] | None, known: Any, what: str
) -> None:
  unknown = sorted(set(selected or ()) - set(known))
  if unknown:
    raise SystemExit(
        f'unknown {what}(s) {unknown}; known: {sorted(known)}'
    )


# ---------------------------------------------------------------------------
# Groups.
# ---------------------------------------------------------------------------
def do_c4(args: argparse.Namespace) -> None:
  start = time.time()
  build_c4.build(
      num_train_shards=args.num_train_shards,
      validation=not args.no_validation,
      out_dir=_c4_dir(),
      workers=args.workers,
      keep_json=args.keep_json,
      force=args.force,
  )
  build_c4.verify(_c4_dir())
  stats = build_c4.stats(_c4_dir())
  asset_lib.record(
      'c4_bin',
      'datasets',
      os.path.relpath(_c4_dir(), asset_lib.DATASETS_DIR),
      source='https://huggingface.co/datasets/allenai/c4 (en)',
      extra={'splits': stats},
      seconds=time.time() - start,
  )
  for split, s in stats.items():
    asset_lib.log(
        f'c4 {split}: {s["shards"]} shards, {s["docs"]} docs,'
        f' {asset_lib.human(s["bytes"])}'
    )


def do_vocabs(args: argparse.Namespace) -> None:
  _reject_unknown(args.only, VOCAB_NAMES, 'vocab')
  wanted = set(args.only or ())

  def want(name: str) -> bool:
    return not wanted or name in wanted

  if want('nanodo_c4'):
    start = time.time()
    build_vocabs.build_nanodo_c4()
    asset_lib.record(
        'vocab_nanodo_c4',
        'vocabs',
        build_vocabs.NANODO_C4_NAME,
        source=build_vocabs.T5_VOCAB_URL + ' + <s> at index 2',
        seconds=time.time() - start,
    )
  if want('gemma3'):
    build_vocabs.build_gemma3_vocab(force=args.force)
    asset_lib.record(
        'vocab_vb262144_gemma3',
        'vocabs',
        build_vocabs.GEMMA3_VOCAB_NAME,
        source=f'{build_vocabs.GEMMA3_REPO}/tokenizer.model',
    )
  if want('qwen2p5'):
    build_vocabs.build_hf_vocab('Qwen2.5', build_vocabs.QWEN2P5_REPO)
    asset_lib.record(
        'vocab_qwen2p5', 'vocabs', 'Qwen2.5', source=build_vocabs.QWEN2P5_REPO
    )
  if want('qwen3'):
    build_vocabs.build_hf_vocab('Qwen3', build_vocabs.QWEN3_REPO)
    asset_lib.record(
        'vocab_qwen3', 'vocabs', 'Qwen3', source=build_vocabs.QWEN3_REPO
    )
  # Explicitly selecting it implies training it; otherwise it needs the flag,
  # since it is the one asset that costs ~15 min of CPU.
  if want('openmix_substitute') and (args.train_openmix or wanted):
    stats = build_vocabs.build_openmix_substitute(
        c4_dir=_c4_dir(),
        num_sentences=args.openmix_sentences,
        force=args.force,
    )
    stats['bytes_per_token_c4_val'] = round(
        build_vocabs.bytes_per_token(stats['path'], c4_dir=_c4_dir()), 4
    )
    asset_lib.record(
        'vocab_openmix_substitute',
        'vocabs',
        build_vocabs.OPENMIX_SUBSTITUTE_NAME,
        source=f'trained on C4 ({args.openmix_sentences} sentences)',
        extra=stats,
        seconds=stats['seconds'],
    )
  print(json.dumps(build_vocabs.check(), indent=2))


def do_datasets(args: argparse.Namespace) -> None:
  _reject_unknown(args.only, DATASET_BUILDERS, 'dataset')
  for name, (builder, rel) in DATASET_BUILDERS.items():
    if args.only and name not in args.only:
      continue
    start = time.time()
    builder(force=args.force)
    asset_lib.record(
        f'dataset_{name}',
        'datasets',
        rel,
        source=DATASET_SOURCES[name],
        seconds=time.time() - start,
    )
  print(json.dumps(build_datasets.check(), indent=2))


def do_models(args: argparse.Namespace) -> None:
  _reject_unknown(args.only, build_models.MODELS_BY_NAME, 'model')
  for spec in build_models.MODELS:
    if args.only and spec.name not in args.only:
      continue
    if args.max_gib and spec.approx_gib > args.max_gib:
      asset_lib.log(
          f'{spec.name}: skipped, ~{spec.approx_gib} GiB > --max-gib'
          f' {args.max_gib}'
      )
      continue
    build_models.build(spec, force=args.force)
    print(json.dumps(build_models.check(spec), indent=2))


def do_gemma3_selftest(args: argparse.Namespace) -> None:
  """Pins the Gemma-3 HF->flax mapping against the ORBAX simply already ships.

  `Gemma3pFormat` reads DeepMind *flax* parameter names, so the Gemma-3
  checkpoints have to be renamed on the way in (see `build_models`). This
  downloads `google/gemma-3-270m` and the matching `GEMMA-3.0-270M-PT-ORBAX`
  from `unkindledmonkey/simply-models` and asserts the mapping reproduces it
  tensor-for-tensor, bit for bit (~0.9 GiB, under a minute).
  """
  del args
  hf_dir = asset_lib.hf_snapshot(
      'google/gemma-3-270m',
      allow_patterns=['*.json', '*.safetensors'],
  )
  reference = os.path.join(
      asset_lib.MODELS_DIR, 'GEMMA-3.0-270M-PT-ORBAX', 'gemma-3-270m'
  )
  if not os.path.exists(reference):
    src = asset_lib.hf_snapshot(
        build_models.SIMPLY_MODELS_REPO,
        allow_patterns=['GEMMA-3.0-270M-PT-ORBAX/**'],
    )
    reference = os.path.join(src, 'GEMMA-3.0-270M-PT-ORBAX', 'gemma-3-270m')
  n = build_models.verify_gemma3_mapping(hf_dir, reference)
  print(f'gemma3 mapping self-test OK: {n} tensors bit-identical to {reference}')


def do_check_restore(args: argparse.Namespace) -> None:
  """Pre-flight: actually restore each task's checkpoint on CPU.

  `prepare_assets verify` proves the checkpoint is *where* the config looks;
  this proves it *fits* the model. `model_lib` re-initialises parameter
  branches the checkpoint does not supply and says so only at INFO level, so a
  half-mapped checkpoint yields a partly random model, a plausible-looking bad
  metric, and no error anywhere. The verdict here is
  `reinitialised_leaves == 0`.
  """
  tasks = list(args.task_ids) + [t for t in args.task if t not in args.task_ids]
  tasks = tasks or list(TASK_ASSETS)
  _reject_unknown(tasks, TASK_ASSETS, 'task')
  failures = []
  for task in tasks:
    models = [
        a.split(':', 1)[1] for a in TASK_ASSETS[task] if a.startswith('model:')
    ]
    if not models:
      print(f'{task:24s} no checkpoint to restore (trains from scratch)')
      continue
    for name in models:
      spec = build_models.MODELS_BY_NAME[name]
      if args.restore_max_gib and spec.approx_gib > args.restore_max_gib:
        print(
            f'{task:24s} {name}: SKIPPED, ~{spec.approx_gib} GiB >'
            f' --restore-max-gib {args.restore_max_gib} (a CPU restore'
            ' materialises every parameter in float32, so this one needs'
            ' ~4x its bf16 size in RAM; --restore-max-gib 0 forces it)'
        )
        continue
      report = build_models.restore_report(spec)
      if report['status'] == 'port_not_implemented':
        print(f'{task:24s} {name}: PORT NOT IMPLEMENTED YET (expected)')
        print(f'{"":24s}   {report["detail"]}')
      elif report['status'] == 'unsupported_on_cpu':
        print(f'{task:24s} {name}: NOT CHECKABLE ON CPU')
        print(f'{"":24s}   {report["detail"]}')
      elif report['ok']:
        print(
            f'{task:24s} {name}: OK  {report["restored_leaves"]}/'
            f'{report["leaves"]} leaves restored, 0 re-initialised,'
            f' {report["parameters"] / 1e9:.4f}B params'
            f' [{report["ckpt_format"]}]'
        )
      else:
        failures.append((task, name))
        print(f'{task:24s} {name}: FAILED  {report["detail"]}')
  if failures:
    raise SystemExit(
        f'check-restore failed for {failures}; do not launch these runs'
    )


def do_task(args: argparse.Namespace) -> None:
  """Stages exactly the assets one or more task ids need."""
  task_ids = list(args.task_ids) + [t for t in args.task if t not in args.task_ids]
  if not task_ids:
    raise SystemExit(f'no task given; known: {sorted(TASK_ASSETS)}')
  _reject_unknown(task_ids, TASK_ASSETS, 'task')
  needed: list[str] = []
  for task in task_ids:
    for asset in TASK_ASSETS[task]:
      if asset not in needed:
        needed.append(asset)
  asset_lib.log(f'assets for {task_ids}: {needed}')
  if 'c4' in needed:
    do_c4(args)
  vocabs = [a.split(':', 1)[1] for a in needed if a.startswith('vocab:')]
  if vocabs:
    do_vocabs(argparse.Namespace(**{**vars(args), 'only': vocabs}))
  datasets = [a.split(':', 1)[1] for a in needed if a.startswith('dataset:')]
  if datasets:
    do_datasets(argparse.Namespace(**{**vars(args), 'only': datasets}))
  models = [a.split(':', 1)[1] for a in needed if a.startswith('model:')]
  if models:
    do_models(argparse.Namespace(**{**vars(args), 'only': models}))


def do_all(args: argparse.Namespace) -> None:
  args.train_openmix = True
  do_c4(args)
  do_vocabs(argparse.Namespace(**{**vars(args), 'only': None}))
  do_datasets(argparse.Namespace(**{**vars(args), 'only': None}))
  do_models(argparse.Namespace(**{**vars(args), 'only': None}))


def do_verify(args: argparse.Namespace) -> None:
  """Structural checks by default; `--hash` recomputes every byte."""
  report: dict[str, Any] = {
      'c4': build_c4.stats(_c4_dir()),
      'vocabs': build_vocabs.check(),
      'datasets': build_datasets.check(),
      'models': [build_models.check(s) for s in build_models.MODELS],
  }
  build_c4.verify(_c4_dir())
  print(json.dumps(report, indent=2, default=str))
  if not args.hash:
    print(
        '\nverify: STRUCTURAL ONLY (counts, shapes, layout, formats). It does'
        ' NOT read the bytes, so it cannot see a truncated or corrupted file'
        ' that still has the right size. Run `verify --hash` after copying the'
        ' cache anywhere.'
    )
    return
  failed = check_hashes(bucket=args.gcs_bucket or '')
  if failed:
    raise SystemExit(
        f'verify --hash: {len(failed)} asset(s) failed: {sorted(failed)}'
    )


def check_hashes(*, bucket: str = '') -> list[str]:
  """Recomputes crc32c for every staged asset; returns the failing names.

  Args:
    bucket: if given, also compare the manifest against the mirrored objects'
      crc32c metadata, which proves the upload landed intact without
      downloading it.

  Returns:
    Manifest keys that failed.
  """
  failed = []
  for name, entry in asset_lib.load_manifest().items():
    recorded = entry.get('files')
    if not recorded:
      print(f'{name:30s} NO RECORDED HASHES -- re-run its builder to record')
      failed.append(name)
      continue
    if not os.path.exists(entry['path']):
      print(f'{name:30s} NOT STAGED at {entry["path"]}')
      failed.append(name)
      continue
    local = asset_lib.file_digests(entry['path'])
    diff = asset_lib.compare_digests(recorded, local)
    ok = not (diff['missing'] or diff['corrupt'])
    line = (
        f'{name:30s} local {len(recorded):5d} files'
        f' {asset_lib.human(entry["bytes"]):>10s}'
    )
    if bucket:
      remote = asset_lib.gcs_digests(bucket, entry['kind'], entry['rel'])
      rdiff = asset_lib.compare_digests(recorded, remote)
      ok = ok and not (rdiff['missing'] or rdiff['corrupt'])
      line += f'  gcs {len(remote):5d} objects'
      diff = {k: diff[k] + ['gcs:' + r for r in rdiff[k]] for k in diff}
    print(f'{line}  {"OK" if ok else "FAILED"}')
    if not ok:
      failed.append(name)
      for kind in ('corrupt', 'missing'):
        for rel in diff[kind][:5]:
          print(f'{"":30s}   {kind}: {rel}')
  return failed


def do_plan(args: argparse.Namespace) -> None:
  """Prints the per-task asset table without touching the network."""
  del args
  staged = asset_lib.load_manifest()
  print(f'{"task":24s} {"asset":34s} {"staged":8s} size')
  for task, assets in TASK_ASSETS.items():
    for asset in assets:
      entry = staged.get(MANIFEST_KEY[asset])
      size = asset_lib.human(entry['bytes']) if entry else '-'
      print(f'{task:24s} {asset:34s} {str(bool(entry)):8s} {size}')


def do_mirror(args: argparse.Namespace) -> None:
  """Mirrors the WHOLE staged cache to GCS."""
  if not args.gcs_bucket:
    raise SystemExit('mirror needs --gcs-bucket')
  mirror(args.gcs_bucket, dry_run=args.dry_run)


def mirror(
    bucket: str,
    *,
    only: Sequence[str] | None = None,
    dry_run: bool = False,
) -> None:
  """Rsyncs staged assets to `gs://<bucket>/assets/<kind>/<rel>`.

  Args:
    bucket: destination bucket.
    only: manifest keys to mirror; None mirrors everything staged so far.
    dry_run: print the gcloud commands instead of running them.
  """
  manifest = asset_lib.load_manifest()
  if not manifest:
    asset_lib.log('nothing staged yet; run a build subcommand first')
  for name, entry in manifest.items():
    if only is not None and name not in only:
      continue
    if not os.path.exists(entry['path']):
      continue
    asset_lib.mirror_to_gcs(
        bucket, entry['kind'], entry['rel'], dry_run=dry_run
    )
  asset_lib.print_tpu_env(bucket)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
# Flags shared by the top-level parser and every subparser, so they work on
# either side of the subcommand. Each caller gets a FRESH parser: `parents=`
# shares action *objects*, and `set_defaults` mutates them, so one instance
# would let the top-level defaults overwrite a value given before the
# subcommand. `argparse.SUPPRESS` keeps an unset subparser flag out of the
# namespace entirely; the real defaults come from `ap.set_defaults`.
def _shared_flags() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
  p.add_argument(
      '--gcs-bucket',
      help='after staging, mirror to gs://<bucket>/assets/{models,datasets,'
      'vocabs} and print the TPU VM environment',
  )
  p.add_argument(
      '--dry-run',
      action='store_true',
      help='print the gcloud commands instead of running them',
  )
  p.add_argument(
      '--task',
      action='append',
      metavar='TASK_ID',
      help='stage exactly what this task needs; repeatable. Implies the `task`'
      f' subcommand. One of: {", ".join(sorted(TASK_ASSETS))}',
  )
  p.add_argument(
      '--only',
      nargs='*',
      metavar='NAME',
      help='restrict a group subcommand to these assets',
  )
  p.add_argument(
      '--num-train-shards', type=int,
      help='C4 train shards to stage (~0.77 GB / 356317 docs each; 2 shards'
      ' already cover the 225M-token pretraining budget)',
  )
  p.add_argument('--no-validation', action='store_true')
  p.add_argument('--workers', type=int)
  p.add_argument('--keep-json', action='store_true')
  p.add_argument(
      '--train-openmix', action='store_true',
      help='train the 100864-piece C4 SPM that replaces the internal'
      ' vb100864_openmix_v1 (~15 min; needs the C4 shards)',
  )
  p.add_argument('--openmix-sentences', type=int)
  p.add_argument(
      '--max-gib', type=float,
      help='`models`: skip checkpoints larger than this (0 = no limit)',
  )
  p.add_argument(
      '--restore-max-gib', type=float,
      help='`check-restore`: skip checkpoints larger than this, since a CPU'
      ' restore materialises every parameter in float32 (0 = no limit)',
  )
  p.add_argument(
      '--force', action='store_true',
      help='rebuild even if the asset is already staged. This is how you'
      ' legitimately replace a bad copy: the builder rewrites the bytes AND'
      ' re-records their crc32c.',
  )
  p.add_argument(
      '--hash', action='store_true',
      help='`verify`: also recompute every file\'s crc32c and compare with the'
      ' manifest (and, with --gcs-bucket, with the mirrored objects\''
      ' metadata). Reads every byte; ~1 min per 100 GB.',
  )
  return p


SUBCOMMANDS = {
    'c4': (do_c4, 'download + repack the C4 shards'),
    'vocabs': (do_vocabs, 'stage the tokenizers'),
    'datasets': (do_datasets, 'stage the eval/train datasets'),
    'models': (do_models, 'stage the ORBAX base checkpoints'),
    'task': (do_task, 'stage everything one or more task ids need'),
    'all': (do_all, 'stage every asset for all 11 tasks'),
    'verify': (do_verify, 're-check everything staged'),
    'plan': (do_plan, 'per-task asset table (offline)'),
    'mirror': (do_mirror, 'mirror the whole staged cache to --gcs-bucket'),
    'check-restore': (
        do_check_restore,
        "restore each task's checkpoint on CPU and assert 0 re-initialised"
        ' parameter branches',
    ),
    'gemma3-selftest': (
        do_gemma3_selftest,
        'check the Gemma-3 HF->flax mapping against the shipped 270M ORBAX',
    ),
}


def build_parser() -> argparse.ArgumentParser:
  ap = argparse.ArgumentParser(
      prog='prepare_assets',
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter,
      parents=[_shared_flags()],
  )
  ap.set_defaults(
      command=None,
      func=do_task,
      gcs_bucket='',
      dry_run=False,
      task=None,
      task_ids=(),
      only=None,
      num_train_shards=32,
      no_validation=False,
      workers=8,
      keep_json=False,
      train_openmix=False,
      openmix_sentences=10_000_000,
      max_gib=0.0,
      restore_max_gib=16.0,
      hash=False,
      force=False,
  )
  sub = ap.add_subparsers(dest='command')
  for name, (func, help_text) in SUBCOMMANDS.items():
    p = sub.add_parser(name, help=help_text, parents=[_shared_flags()])
    if name in ('task', 'check-restore'):
      p.add_argument('task_ids', nargs='*', metavar='TASK_ID')
    p.set_defaults(func=func)
  return ap


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  if args.command is None and not args.task:
    parser.error('give a subcommand or --task=<task_id>')
  if args.task is None:
    args.task = []
  args.func(args)
  if args.gcs_bucket and args.command not in ('mirror', 'verify'):
    mirror(
        args.gcs_bucket,
        only=asset_lib.staged_this_run,
        dry_run=args.dry_run,
    )
  return 0


if __name__ == '__main__':
  sys.exit(main())
