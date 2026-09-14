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
r"""Converts a HuggingFace Kimi K3 release into a Simply checkpoint.

    python -m simply.zoo.kimi_k3.convert_hf_checkpoint \\
      --hf_dir=<release> --out_dir=<yours> --step=0 --verify

Flags and `main` only; `utils/hf_convert.py` is the machinery, and its module
docstring explains how a 1.56 TB release is read without materializing it.
"""

from collections.abc import Sequence
import time

from absl import app
from absl import flags
from absl import logging
from simply.zoo.kimi_k3.utils import hf_convert

_GIB = 2**30


_HF_DIR = flags.DEFINE_string('hf_dir', '', 'HuggingFace checkpoint directory.')
_OUT_DIR = flags.DEFINE_string(
    'out_dir', None, 'Destination Orbax directory (local or gs://...).'
)
_STEP = flags.DEFINE_integer('step', 0, 'Checkpoint step to write.')
_DRY_RUN = flags.DEFINE_bool(
    'dry_run', False, 'Validate the plan and print the tree; write nothing.'
)
_VERIFY = flags.DEFINE_bool(
    'verify', False, 'After writing, re-read sampled leaves and compare bytes.'
)
_VERIFY_ONLY = flags.DEFINE_bool(
    'verify_only', False, 'Skip writing; verify an existing --out_dir.'
)
_VERIFY_SAMPLES = flags.DEFINE_integer(
    'verify_samples', 8, 'Leaves to verify; <= 0 verifies all of them.'
)
_LAYERS = flags.DEFINE_string(
    'layers',
    '',
    'Layers to convert, e.g. "0-3,8"; empty means all of them. On'
    ' --verify_only it must describe the layers the checkpoint HOLDS, since'
    ' its structure is what tells Orbax which leaves to skip.',
)
_VERIFY_LAYERS = flags.DEFINE_string(
    'verify_layers',
    '',
    'Restrict --verify to these layers (plus the embeddings and final norms),'
    ' e.g. "0" for a two-minute smoke check of a finished conversion.',
)
_DEQUANTIZE_EXPERTS = flags.DEFINE_bool(
    'dequantize_experts',
    None,
    'Decode the MXFP4 routed experts at conversion time. Defaults to whether'
    ' --expert_dtype names a dtype; setting it false forces the packed form.',
)
_EXPERT_DTYPE = flags.DEFINE_enum(
    'expert_dtype',
    'mxfp4',
    ['mxfp4', 'bfloat16', 'float32'],
    'Routed expert representation. "mxfp4" keeps the packed u8 pair verbatim'
    ' (1.45 TiB, decoded on restore by KimiK3Format); the dense alternatives'
    ' are 5.06 TiB in bf16 and twice that in f32.',
)
_PARAM_DTYPE = flags.DEFINE_enum(
    'param_dtype',
    'source',
    list(hf_convert.PARAM_DTYPES),
    'Dtype of the non-expert weights. "source" keeps what the release ships'
    ' (bf16 matrices; f32 for A_log, dt_bias, the conv kernels, o_norm and'
    ' the router bias).',
)
_CHUNK_MIB = flags.DEFINE_integer(
    'chunk_mib', 32, 'TensorStore chunk size, in MiB.'
)
_MAX_INFLIGHT_GIB = flags.DEFINE_integer(
    'max_inflight_gib', 24, 'Materialized bytes allowed to await a write.'
)
_MAX_RSS_GIB = flags.DEFINE_integer(
    'max_rss_gib', 100, 'Abort past this resident size, in GiB; 0 disables.'
)
_READ_THREADS = flags.DEFINE_integer(
    'read_threads', 8, 'Concurrent shard reads.'
)


def main(argv: Sequence[str]) -> None:
  del argv
  logging.set_verbosity(logging.INFO)
  logging.set_stderrthreshold('info')
  started = time.monotonic()
  if not _HF_DIR.value:
    raise app.UsageError('--hf_dir is required.')

  config = hf_convert.load_config(_HF_DIR.value)
  layers = hf_convert.parse_layers(_LAYERS.value, config.n_layers)
  index = hf_convert.read_index(_HF_DIR.value)
  converter = hf_convert.build_converter(
      config,
      index,
      expert_dtype=_EXPERT_DTYPE.value,
      param_dtype=_PARAM_DTYPE.value,
      dequantize_experts=_DEQUANTIZE_EXPERTS.value,
  )
  logging.info(
      'index: %d tensors, %d shards missing or truncated; converting %d/%d'
      ' layers, experts=%s',
      len(index.shard_of),
      len(index.missing_shards) + len(index.truncated_shards),
      len(layers),
      config.n_layers,
      _EXPERT_DTYPE.value,
  )
  try:
    plan = hf_convert.build_plan(converter, index, layers)
  except KeyError as missing:
    raise app.UsageError(
        f'cannot plan the conversion: {missing}. Shards not on disk:'
        f' {list(index.missing_shards[:5])}'
    ) from missing
  logging.info(
      'plan: %d leaves, %.2f GiB, built in %.1fs',
      len(plan.leaves),
      plan.output_bytes / _GIB,
      time.monotonic() - started,
  )

  if _DRY_RUN.value:
    problems = hf_convert.dry_run_report(plan, index, converter)
    for problem in problems:
      print(f'PROBLEM: {problem}')
    raise SystemExit(1 if problems else 0)

  if not _OUT_DIR.value:
    raise app.UsageError('--out_dir is required unless --dry_run.')
  unreadable = plan.unreadable_shards(index)
  if unreadable:
    raise app.UsageError(
        f'{len(unreadable)} shards this conversion reads are missing or'
        f' truncated: {list(unreadable[:5])}'
    )
  options = hf_convert.WriteOptions(
      chunk_bytes=_CHUNK_MIB.value * 2**20,
      max_inflight_bytes=_MAX_INFLIGHT_GIB.value * _GIB,
      max_rss_bytes=_MAX_RSS_GIB.value * _GIB,
  )
  with hf_convert.ShardReader(_HF_DIR.value) as reader:
    materializer = hf_convert.GroupMaterializer(
        converter, plan, index, reader, read_threads=_READ_THREADS.value
    )
    try:
      if not _VERIFY_ONLY.value:
        hf_convert.write_checkpoint(
            plan, materializer.leaf, _OUT_DIR.value, _STEP.value, options
        )
      if _VERIFY.value or _VERIFY_ONLY.value:
        problems = hf_convert.verify(
            plan,
            materializer.leaf,
            _OUT_DIR.value,
            _STEP.value,
            _VERIFY_SAMPLES.value,
            leaves=hf_convert.leaves_of_layers(
                plan,
                hf_convert.parse_layers(_VERIFY_LAYERS.value, config.n_layers),
            ),
        )
        for problem in problems:
          print(f'MISMATCH: {problem}')
        if problems:
          raise SystemExit(f'{len(problems)} leaves did not match the source.')
    finally:
      materializer.close()
  logging.info(
      'done in %s', hf_convert.format_duration(time.monotonic() - started)
  )


if __name__ == '__main__':
  app.run(main)
