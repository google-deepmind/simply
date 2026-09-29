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

"""Stages the base checkpoints the research-bench tasks start from.

Three provenance routes, in increasing order of work:

1. **Already ORBAX** in `unkindledmonkey/simply-models` -- downloaded verbatim
   (Qwen3-4B, Qwen3-30B-A3B-Thinking-2507).
2. **HF safetensors with a core format** -- converted by the repo's own
   `simply.tools.hf_to_orbax`, which writes the *stored* tensor names and only
   records the format; the rename happens in `CheckpointFormat.transforms` at
   restore time (Qwen2.5-Math-1.5B, Qwen3-0.6B-Base, and the two porting-task
   checkpoints, whose formats are deliberately unimplemented stubs).
3. **HF safetensors with no matching core format** -- Gemma-3. `Gemma3pFormat`
   reads the DeepMind *flax* parameter names (`transformer/layer_N/...`), which
   the HF export does not use, so `hf_to_orbax` alone produces an unloadable
   checkpoint. `convert_gemma3` does the flax renaming itself; the mapping is
   verified tensor-for-tensor (bit-exact) against the `GEMMA-3.0-270M-PT-ORBAX`
   checkpoint simply already ships -- see `verify_gemma3_mapping`.

Layout written: step-numbered directories under **the exact path the config's
`init_ckpt_dir` points at** -- which is NOT uniform, so every spec states it
explicitly in `ckpt_dir` rather than deriving it.

    <SIMPLY_MODELS>/<ckpt_dir>/1/{_CHECKPOINT_METADATA,state,metadata}
    <SIMPLY_MODELS>/<dest>/VOCAB/...      (porting tasks only)

Most constants end in `/ORBAX` (`Qwen3-4B/ORBAX`, ...), but the Gemma family
does not (`GEMMA-3.0-1B-PT-ORBAX`), so its step dirs sit directly under the
model directory. `check()` resolves each spec's `config_name` and fails if the
staged path and the config disagree, or if the staged path has no numeric step
child -- the two ways this silently breaks a TPU run.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any

from tasks.research_bench.setup import asset_lib

M = asset_lib.MODELS_DIR
SIMPLY_MODELS_REPO = 'unkindledmonkey/simply-models'


@dataclasses.dataclass(frozen=True)
class ModelSpec:
  """One base checkpoint a task starts from.

  Attributes:
    name: asset name (manifest key).
    tasks: task ids that need it.
    dest: path under `SIMPLY_MODELS` that is mirrored as a unit (holds the
      checkpoint and, for the porting tasks, `VOCAB/`).
    ckpt_dir: path under `SIMPLY_MODELS` that holds the numeric step dirs;
      must equal the config's `init_ckpt_dir`. Usually `<dest>/ORBAX`, but the
      Gemma constants have no `/ORBAX` level.
    config_name: registered experiment config that pins `ckpt_dir`, used by
      `check()` to catch drift between this table and the configs.
    ckpt_format: the `CheckpointFormat` recorded in the checkpoint metadata.
    hf_repo: HuggingFace repo to convert from ('' if route 1).
    orbax_subdir: path inside `simply-models` if it is already ORBAX.
    converter: 'hf_to_orbax' | 'gemma3' | 'download'.
    vocab_files: files to stage into `<dest>/VOCAB` (porting tasks).
    gated: the HF repo needs an accepted licence + `huggingface-cli login`.
    approx_gib: download/convert size, for planning.
  """

  name: str
  tasks: tuple[str, ...]
  dest: str
  ckpt_dir: str
  config_name: str
  ckpt_format: str
  hf_repo: str = ''
  orbax_subdir: str = ''
  converter: str = 'hf_to_orbax'
  vocab_files: tuple[str, ...] = ()
  gated: bool = False
  approx_gib: float = 0.0


HF_TOKENIZER_FILES = (
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'tokenizer.model',
)

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name='gemma3_1b_pt',
        tasks=('rl_gemma3_1b', 'rl_bfcl_gemma3_1b'),
        dest='GEMMA-3.0-1B-PT-ORBAX',
        # No `/ORBAX` level: `core.GEMMA3_1B_PT_CKPT_DIR` is the model dir itself.
        ckpt_dir='GEMMA-3.0-1B-PT-ORBAX',
        config_name='rl_gemma3_1b',
        ckpt_format='Gemma3pFormat',
        hf_repo='google/gemma-3-1b-pt',
        converter='gemma3',
        gated=True,
        approx_gib=2.0,
    ),
    ModelSpec(
        name='qwen2p5_math_1p5b',
        tasks=('rl_qwen2p5_math_1p5b',),
        dest='Qwen2.5-Math-1.5B',
        ckpt_dir='Qwen2.5-Math-1.5B/ORBAX',
        config_name='rl_qwen2p5_math_1p5b',
        ckpt_format='Qwen2Format',
        hf_repo='Qwen/Qwen2.5-Math-1.5B',
        approx_gib=3.1,
    ),
    ModelSpec(
        name='qwen3_0p6b_base',
        tasks=('rl_bfcl_qwen3_0p6b',),
        dest='Qwen3-0.6B-Base',
        ckpt_dir='Qwen3-0.6B-Base/ORBAX',
        config_name='rl_bfcl_qwen3_0p6b',
        ckpt_format='Qwen2Format',
        hf_repo='Qwen/Qwen3-0.6B-Base',
        approx_gib=1.4,
    ),
    ModelSpec(
        name='qwen3_4b',
        tasks=('sampling_lcb',),
        dest='Qwen3-4B',
        ckpt_dir='Qwen3-4B/ORBAX',
        config_name='qwen3_4b',
        ckpt_format='Qwen2Format',
        orbax_subdir='Qwen3-4B/ORBAX',
        converter='download',
        approx_gib=7.5,
    ),
    ModelSpec(
        name='qwen3_30b_a3b_thinking_2507',
        tasks=('decode_efficiency_vf',),
        dest='Qwen3-30B-A3B-Thinking-2507',
        ckpt_dir='Qwen3-30B-A3B-Thinking-2507/ORBAX',
        config_name='qwen3_30b_a3b_thinking_2507',
        ckpt_format='Qwen2Format',
        orbax_subdir='Qwen3-30B-A3B-Thinking-2507/ORBAX',
        converter='download',
        approx_gib=57.0,
    ),
    ModelSpec(
        name='falcon_h1_0p5b_base',
        tasks=('port_falcon_h1_0p5b',),
        dest='research_bench/Falcon-H1-0.5B-Base',
        ckpt_dir='research_bench/Falcon-H1-0.5B-Base/ORBAX',
        config_name='port_falcon_h1_0p5b',
        ckpt_format='FalconH1Format',
        hf_repo='tiiuae/Falcon-H1-0.5B-Base',
        vocab_files=HF_TOKENIZER_FILES,
        approx_gib=1.1,
    ),
    ModelSpec(
        name='recurrentgemma_2b',
        tasks=('port_recurrentgemma_2b',),
        dest='research_bench/RecurrentGemma-2B',
        ckpt_dir='research_bench/RecurrentGemma-2B/ORBAX',
        config_name='port_recurrentgemma_2b',
        ckpt_format='RecurrentGemmaFormat',
        hf_repo='google/recurrentgemma-2b',
        vocab_files=HF_TOKENIZER_FILES,
        gated=True,
        approx_gib=5.3,
    ),
)

MODELS_BY_NAME = {spec.name: spec for spec in MODELS}


def orbax_dir(spec: ModelSpec) -> str:
  """Absolute directory that must contain the numeric step dirs."""
  return os.path.join(M, spec.ckpt_dir)


def step_dirs(path: str) -> list[str]:
  """Numeric step subdirectories of `path`, as `checkpoint_lib` finds them."""
  if not os.path.isdir(path):
    return []
  return sorted(
      name for name in os.listdir(path)
      if name.isdigit() and os.path.isdir(os.path.join(path, name))
  )


def _is_built(path: str) -> bool:
  return any(
      os.path.exists(os.path.join(path, step, '_CHECKPOINT_METADATA'))
      for step in step_dirs(path)
  )


def config_ckpt_dir(config_name: str) -> str:
  """`init_ckpt_dir` of a registered config (research_bench registry, then core)."""
  return _get_config(config_name).init_ckpt_dir


# ---------------------------------------------------------------------------
# Route 3: Gemma-3 HF safetensors -> the flax names `Gemma3pFormat` reads.
# ---------------------------------------------------------------------------
def gemma3_flax_state(hf_dir: str) -> dict[str, Any]:
  """Reads Gemma-3 HF safetensors and returns the flax-named tensor tree.

  Shapes follow the DeepMind flax export, which is what `Gemma3pFormat`
  un-maps: `q_einsum/w` is (heads, model_dim, head_dim), `kv_einsum/w` is
  (2, kv_heads, model_dim, head_dim), `attn_vec_einsum/w` is
  (heads, head_dim, model_dim), `gating_einsum/w` is (2, ffn_dim, model_dim)
  and `linear/w` is (ffn_dim, model_dim).

  Args:
    hf_dir: a `google/gemma-3-*` snapshot directory.

  Returns:
    `{'transformer/...': np.ndarray}`.
  """
  import ml_dtypes  # pylint: disable=g-import-not-at-top,unused-import  # registers bfloat16 with numpy
  import numpy as np  # pylint: disable=g-import-not-at-top
  from safetensors import safe_open  # pylint: disable=g-import-not-at-top

  with open(os.path.join(hf_dir, 'config.json'), encoding='utf-8') as f:
    config = json.load(f)
  config = config.get('text_config', config)
  model_dim = config['hidden_size']
  head_dim = config['head_dim']
  num_heads = config['num_attention_heads']
  num_kv_heads = config['num_key_value_heads']

  tensors = {}
  for path in sorted(glob.glob(os.path.join(hf_dir, '*.safetensors'))):
    with safe_open(path, 'numpy') as f:
      for key in f.keys():  # pylint: disable=g-builtin-op
        tensors[key] = f.get_tensor(key)
  # Multimodal Gemma-3 exports prefix the text tower; the text tower is what
  # the research-bench configs instantiate.
  tensors = {
      k.removeprefix('language_model.').removeprefix('model.language_model.'): v
      for k, v in tensors.items()
  }

  out: dict[str, Any] = {
      'transformer/embedder/input_embedding': tensors['model.embed_tokens.weight'],
      'transformer/final_norm/scale': tensors['model.norm.weight'],
  }
  norms = {
      'input_layernorm': 'pre_attention_norm',
      'post_attention_layernorm': 'post_attention_norm',
      'pre_feedforward_layernorm': 'pre_ffw_norm',
      'post_feedforward_layernorm': 'post_ffw_norm',
  }
  for layer in range(config['num_hidden_layers']):
    src = f'model.layers.{layer}'
    dst = f'transformer/layer_{layer}'
    for hf_name, flax_name in norms.items():
      out[f'{dst}/{flax_name}/scale'] = tensors[f'{src}.{hf_name}.weight']
    out[f'{dst}/attn/_query_norm/scale'] = tensors[f'{src}.self_attn.q_norm.weight']
    out[f'{dst}/attn/_key_norm/scale'] = tensors[f'{src}.self_attn.k_norm.weight']
    out[f'{dst}/attn/q_einsum/w'] = (
        tensors[f'{src}.self_attn.q_proj.weight']
        .reshape(num_heads, head_dim, model_dim)
        .transpose(0, 2, 1)
    )
    out[f'{dst}/attn/kv_einsum/w'] = np.stack(
        [
            tensors[f'{src}.self_attn.{proj}_proj.weight']
            .reshape(num_kv_heads, head_dim, model_dim)
            .transpose(0, 2, 1)
            for proj in ('k', 'v')
        ],
        axis=0,
    )
    out[f'{dst}/attn/attn_vec_einsum/w'] = (
        tensors[f'{src}.self_attn.o_proj.weight']
        .reshape(model_dim, num_heads, head_dim)
        .transpose(1, 2, 0)
    )
    out[f'{dst}/mlp/gating_einsum/w'] = np.stack(
        [
            tensors[f'{src}.mlp.gate_proj.weight'],
            tensors[f'{src}.mlp.up_proj.weight'],
        ],
        axis=0,
    )
    out[f'{dst}/mlp/linear/w'] = tensors[f'{src}.mlp.down_proj.weight'].T
  return out


def write_orbax(state: dict[str, Any], out_dir: str, ckpt_format: str) -> str:
  """Writes `state` as an ORBAX checkpoint at step 1, tagged `ckpt_format`."""
  import orbax.checkpoint as ocp  # pylint: disable=g-import-not-at-top
  from etils import epath  # pylint: disable=g-import-not-at-top
  from simply.utils import checkpoint_lib as ckpt_lib  # pylint: disable=g-import-not-at-top

  path = epath.Path(out_dir)
  if path.exists():
    path.rmtree()
  registry = ocp.DefaultCheckpointHandlerRegistry()
  registry.add('state', ocp.args.PyTreeSave, ocp.PyTreeCheckpointHandler())
  registry.add('metadata', ocp.args.JsonSave, ocp.JsonCheckpointHandler())
  with ocp.CheckpointManager(path, handler_registry=registry) as manager:
    ckpt_lib.save_checkpoint(
        manager,
        ocp.tree.from_flat_dict(state, sep='/'),
        1,
        ckpt_lib.CheckpointFormatRegistry.get_instance(ckpt_format),
    )
  return out_dir


def convert_gemma3(hf_dir: str, out_dir: str) -> str:
  return write_orbax(gemma3_flax_state(hf_dir), out_dir, 'Gemma3pFormat')


def _bitwise_equal(a: Any, b: Any) -> bool:
  """Compares two arrays bit for bit, including non-numpy dtypes like bf16."""
  import numpy as np  # pylint: disable=g-import-not-at-top

  a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
  if a.shape != b.shape or a.dtype != b.dtype:
    return False
  return bool(np.array_equal(a.view(np.uint8), b.view(np.uint8)))


def verify_gemma3_mapping(hf_dir: str, reference_ckpt: str) -> int:
  """Asserts `gemma3_flax_state(hf_dir)` equals an existing ORBAX checkpoint.

  Used to pin `convert_gemma3` against `GEMMA-3.0-270M-PT-ORBAX`, the Gemma-3
  checkpoint simply already ships in ORBAX form.

  Args:
    hf_dir: HF snapshot of the SAME model as `reference_ckpt`.
    reference_ckpt: an ORBAX step directory in flax naming.

  Returns:
    The number of tensors compared.

  Raises:
    ValueError: on the first tensor that differs.
  """
  import numpy as np  # pylint: disable=g-import-not-at-top
  import orbax.checkpoint as ocp  # pylint: disable=g-import-not-at-top
  from etils import epath  # pylint: disable=g-import-not-at-top

  handler = ocp.PyTreeCheckpointHandler()
  path = epath.Path(reference_ckpt)
  reference = ocp.tree.to_flat_dict(
      handler.restore(path, args=ocp.args.PyTreeRestore(handler.metadata(path))),
      sep='/',
  )
  converted = gemma3_flax_state(hf_dir)
  if set(reference) != set(converted):
    raise ValueError(
        'gemma3 key mismatch:'
        f' missing={sorted(set(reference) - set(converted))[:5]}'
        f' extra={sorted(set(converted) - set(reference))[:5]}'
    )
  for key, want in reference.items():
    if not _bitwise_equal(converted[key], want):
      raise ValueError(f'gemma3 tensor {key} differs from {reference_ckpt}')
  return len(reference)


# ---------------------------------------------------------------------------
# Route 2: the repo's own converter.
# ---------------------------------------------------------------------------
# `hf_to_orbax` owns absl flags, so it runs in a subprocess; the porting-task
# formats live in the research_bench package, so import it first to register them.
_HF_TO_ORBAX_BOOTSTRAP = (
    'from absl import app;'
    ' import tasks.research_bench.checkpoint_lib;'
    ' from simply.tools import hf_to_orbax;'
    ' app.run(hf_to_orbax.main)'
)


def convert_hf_to_orbax(hf_dir: str, out_dir: str, ckpt_format: str) -> str:
  """Converts a safetensors repo with the repo's own `hf_to_orbax`."""
  cmd = [
      sys.executable,
      '-c',
      _HF_TO_ORBAX_BOOTSTRAP,
      f'--input_path={hf_dir}',
      f'--output_path={out_dir}',
      f'--format={ckpt_format}',
  ]
  asset_lib.log(f'hf_to_orbax {hf_dir} -> {out_dir} ({ckpt_format})')
  subprocess.run(cmd, check=True)
  return out_dir


# ---------------------------------------------------------------------------
# Build + check.
# ---------------------------------------------------------------------------
def build(spec: ModelSpec, *, force: bool = False) -> dict[str, Any]:
  """Stages one checkpoint (and its tokenizer, for the porting tasks)."""
  out = orbax_dir(spec)
  start = time.time()
  if _is_built(out) and not force:
    asset_lib.log(f'{spec.name}: already staged at {out}')
  elif spec.converter == 'download':
    src = asset_lib.hf_snapshot(
        SIMPLY_MODELS_REPO, allow_patterns=[f'{spec.orbax_subdir}/**']
    )
    staged = os.path.join(src, spec.orbax_subdir)
    asset_lib.ensure_dir(os.path.dirname(out))
    if os.path.abspath(staged) != os.path.abspath(out):
      if os.path.exists(out):
        shutil.rmtree(out)
      shutil.copytree(staged, out, symlinks=False)
  else:
    hf_dir = asset_lib.hf_snapshot(
        spec.hf_repo,
        allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'],
    )
    if spec.converter == 'gemma3':
      convert_gemma3(hf_dir, out)
    else:
      convert_hf_to_orbax(hf_dir, out, spec.ckpt_format)
  if spec.vocab_files:
    hf_dir = asset_lib.hf_snapshot(
        spec.hf_repo, allow_patterns=list(spec.vocab_files)
    )
    vocab_out = asset_lib.ensure_dir(os.path.join(M, spec.dest, 'VOCAB'))
    for name in spec.vocab_files:
      src_file = os.path.join(hf_dir, name)
      if os.path.exists(src_file):
        asset_lib.stage_file(src_file, os.path.join(vocab_out, name))
  return asset_lib.record(
      spec.name,
      'models',
      os.path.join(spec.dest),
      source=spec.hf_repo or f'{SIMPLY_MODELS_REPO}/{spec.orbax_subdir}',
      extra={'tasks': list(spec.tasks), 'ckpt_format': spec.ckpt_format},
      seconds=time.time() - start,
  )


def restore_report(spec: ModelSpec) -> dict[str, Any]:
  """Restores the staged checkpoint on CPU the way a run does, and counts.

  `model_lib` re-initialises *missing* parameter branches on a structural
  mismatch and only says so at INFO level, so a half-mapped checkpoint trains a
  partly random model and reports a plausible-looking bad number instead of
  failing. This reproduces that code path (`create_model` ->
  `load_checkpoint_from_dir`) and makes the branch count the verdict:
  `reinitialised_leaves` must be 0.

  Args:
    spec: the checkpoint to restore.

  Returns:
    `{'ok': bool, 'status': str, ...}`; `status` is 'restored',
    'port_not_implemented' (the porting-task model classes are stubs, which is
    the shipped state) or 'failed'.
  """
  import jax  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from simply import model_lib  # pylint: disable=g-import-not-at-top
  from simply.utils import checkpoint_lib as ckpt_lib  # pylint: disable=g-import-not-at-top
  from simply.utils import common  # pylint: disable=g-import-not-at-top
  # Registration side effect: the porting tasks' model classes (stubs as
  # shipped) live in the research_bench package, not in core's ModuleRegistry.
  from tasks.research_bench import model_lib as _rb_model_lib  # pylint: disable=g-import-not-at-top,unused-import

  report: dict[str, Any] = {'name': spec.name, 'tasks': list(spec.tasks)}
  config = _get_config(spec.config_name)
  report['ckpt_dir'] = config.init_ckpt_dir
  report['ckpt_format'] = config.init_ckpt_format or spec.ckpt_format
  try:
    model, _ = model_lib.create_model(config)
    abstract = {
        'params': common.eval_abstract_output(
            lambda: model.init(jax.random.key(0))
        )
    }
  except ValueError as e:
    # Some configs (the 30B MoE) declare a sharding mesh a single CPU device
    # cannot provide. That is an environment limit, not a bad checkpoint.
    if 'not found in mesh' not in str(e):
      raise
    report.update(
        ok=True,
        status='unsupported_on_cpu',
        detail=(
            f'{spec.config_name}: the model\'s sharding config needs a mesh'
            ' this host cannot build on one CPU device, so the checkpoint'
            ' cannot be restored here. Check it on the accelerator instead;'
            ' `verify` still covers its layout and format.'
        ),
        message=str(e).splitlines()[0][:200],
    )
    return report
  except NotImplementedError as e:
    report.update(
        ok=True,
        status='port_not_implemented',
        detail=(
            f'{spec.config_name}: the model class is a STUB -- writing it IS'
            ' the task. Nothing is wrong with the staged checkpoint (its'
            ' tensors are there to be inspected); rerun this check once the'
            ' port exists.'
        ),
        message=str(e).splitlines()[0][:200],
    )
    return report
  state = ckpt_lib.load_checkpoint_from_dir(
      config.init_ckpt_dir,
      abstract,
      config.init_ckpt_step,
      ckpt_format=config.init_ckpt_format,
  )
  leaves = jax.tree.leaves(state['params'])
  restored = [leaf for leaf in leaves if isinstance(leaf, jax.Array)]
  report.update(
      leaves=len(leaves),
      restored_leaves=len(restored),
      reinitialised_leaves=len(leaves) - len(restored),
      parameters=int(sum(np.prod(leaf.shape) for leaf in restored)),
  )
  report['ok'] = bool(leaves) and len(restored) == len(leaves)
  report['status'] = 'restored' if report['ok'] else 'failed'
  if not report['ok']:
    report['detail'] = (
        f'{len(leaves) - len(restored)} of {len(leaves)} parameter leaves were'
        ' NOT in the checkpoint; a run would silently re-initialise them'
        ' (model_lib logs "Checkpoint structural mismatch" at INFO) and train'
        ' a partly random model.'
    )
  return report


def _get_config(config_name: str):
  """Resolves a config by name from the research_bench registry, then core's."""
  from simply import config_lib as core_config  # pylint: disable=g-import-not-at-top
  from tasks.research_bench import config_lib as rb_config  # pylint: disable=g-import-not-at-top

  for registry in (rb_config.ExperimentConfigRegistry,
                   core_config.ExperimentConfigRegistry):
    try:
      return registry.get_config(config_name)
    except (ValueError, KeyError):
      continue
  raise ValueError(f'no registered config named {config_name!r}')


def check(spec: ModelSpec, *, strict: bool = True) -> dict[str, Any]:
  """Reads the staged checkpoint the way a run's restore would.

  Args:
    spec: the checkpoint to check.
    strict: raise when the staged layout would fail at restore time (no
      numeric step dir, or a path the config does not point at) instead of
      only reporting it.

  Returns:
    A summary dict.

  Raises:
    ValueError: in strict mode, on a layout a run would reject.
  """
  import orbax.checkpoint as ocp  # pylint: disable=g-import-not-at-top
  from etils import epath  # pylint: disable=g-import-not-at-top
  from simply.utils import checkpoint_lib as ckpt_lib  # pylint: disable=g-import-not-at-top

  out = orbax_dir(spec)
  wanted = config_ckpt_dir(spec.config_name)
  if os.path.normpath(wanted) != os.path.normpath(out):
    message = (
        f'{spec.name}: staged at {out} but config {spec.config_name!r} loads'
        f' from {wanted}'
    )
    if strict:
      raise ValueError(message)
    asset_lib.log(message)
  steps = step_dirs(out)
  if not steps:
    message = (
        f'{spec.name}: {out} has no numeric step directory, so'
        ' `checkpoint_lib.last_checkpoint_step` returns -1 and the run dies'
        f' with "No checkpoint found in {out}"'
    )
    if strict and os.path.isdir(out):
      raise ValueError(message)
    return {'name': spec.name, 'staged': False, 'path': out, 'error': message}
  if not _is_built(out):
    return {'name': spec.name, 'staged': False, 'path': out}
  step_path = epath.Path(ckpt_lib.get_checkpoint_path(out))
  handler = ocp.PyTreeCheckpointHandler()
  metadata = ocp.tree.to_flat_dict(handler.metadata(step_path / 'state'), sep='/')
  recorded = ''
  meta_file = step_path / 'metadata' / 'metadata'
  if meta_file.exists():
    recorded = json.loads(meta_file.read_text()).get(
        '__checkpoint_format__', {}
    ).get('__dataclass__', '')
  return {
      'name': spec.name,
      'staged': True,
      'path': out,
      'config_ckpt_dir': wanted,
      'step_dirs': steps,
      'step_dir': step_path.name,
      'tensors': len(metadata),
      'bytes': asset_lib.du_bytes(out),
      'recorded_format': recorded,
      'expected_format': f'CheckpointFormat:{spec.ckpt_format}',
      'sample_keys': sorted(metadata)[:3],
  }
