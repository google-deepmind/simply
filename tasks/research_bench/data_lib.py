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

"""Research Bench data_lib.

Re-exports core simply data_lib (so the byte256 ByteVocab registration loads),
and registers the tool-use / function-calling RLVR datasets for the
`tool_use_bfcl_rl` task: BFCL (Berkeley Function Calling Leaderboard, AST
categories) as the held-out eval, plus BFCL-non-live and ToolACE as train
sources. Data are public (BFCL from the gorilla repo; ToolACE =
Team-ACE/ToolACE, Apache-2.0), staged as JSON under TOOLUSE_DATASETS_DIR.

The OSS port additionally owns the C4 pretraining stream and the `nanodo_c4`
vocabulary, neither of which is publicly reachable the way core simply reaches
them internally:

  * `C4FileSource` replaces `TFDSSource(name='c4:3.1.0')` -- TFDS has no public
    mirror of C4 and building it needs the full CommonCrawl pipeline, so the
    `allenai/c4` `en` release is repacked into flat `.bin` blobs + `.idx.npy`
    offset tables by `setup/build_c4.py`.
  * `nanodo_c4` is not present in core's vocab table (the model lived on
    internal storage), so it is registered here against the staged
    `cc_all.32000.100extra.bos.model`.
"""

import dataclasses
import functools
import json
import os
import shutil
from typing import Any

from etils import epath
import numpy as np
from simply import config_lib as _config
from simply import data_lib as _core
from simply.utils import tokenization as _tok

DataSourceRegistry = _core.DataSourceRegistry
DatasetConfigRegistry = _core.DatasetConfigRegistry
DatasetConfig = _core.DatasetConfig
PACKING_NONE = _core.PACKING_NONE
create_iter_dataset = _core.create_iter_dataset

# Public tool-use datasets (BFCL held-out eval + BFCL-non-live/ToolACE train),
# staged as JSON under the shared, canonical simply datasets dir.
TOOLUSE_DATASETS_DIR = os.path.join(_core.DATASETS_DIR, 'tooluse_rlvr')


def _bfcl_user_text(question) -> str:
  """Flattens BFCL's nested `question` ([[{role,content}]]) into a user string.

  Keeps every example field flat/hashable for the RL data pipeline; the eval
  reconstructs the prompt from this text.

  Args:
    question: BFCL's nested question ([[{role, content}]]).

  Returns:
    The flattened user-turn text.
  """
  if isinstance(question, str):
    return question
  if not question:
    return ''
  turns = question[0] if isinstance(question[0], list) else question
  return ' '.join(
      t['content']
      for t in turns
      if isinstance(t, dict) and t.get('role') == 'user'
  )


@functools.partial(DataSourceRegistry.register, name='simply:bfcl')
@dataclasses.dataclass(frozen=True)
class BFCLSource:
  """Berkeley Function Calling Leaderboard (AST categories) data source.

  Each example carries the BFCL `question` (message turns), `function` (tool
  schemas), and `ground_truth` (allowed-value sets per arg; None for the
  irrelevance/abstention categories). Used for function-calling RLVR with an
  AST-based verifiable reward (see BFCLFunctionCallEvaluation).

  `categories` selects which BFCL category JSON files to concatenate (files live
  under TOOLUSE_DATASETS_DIR/bfcl/<category>.json).
  """

  dir: str = os.path.join(TOOLUSE_DATASETS_DIR, 'bfcl')
  # Default: the non-live AST categories (clean, deterministic verifier).
  categories: tuple[str, ...] = (
      'simple',
      'multiple',
      'parallel',
      'parallel_multiple',
      'irrelevance',
  )
  start_index: int | None = None
  end_index: int | None = None

  @functools.cached_property
  def _examples(self) -> list[dict[str, Any]]:
    """Loads + flattens the BFCL examples for this source's categories."""
    examples = []
    for cat in self.categories:
      with epath.Path(os.path.join(self.dir, f'{cat}.json')).open('r') as f:
        data = json.load(f)
      # Tag each example with its source category (a hashable str; safe for the
      # grain RL pipeline) so the held-out eval can report per-category accuracy
      # and separate call-required examples from abstention ones.
      for example in data:
        example['category'] = cat
      examples.extend(data)
    for i, example in enumerate(examples):
      example['uid'] = f'bfcl-{example.get("id", i)}'
      # Serialize the nested fields to JSON strings so no dict/list flows as a
      # raw example field into the RL data pipeline (grain hashes example
      # fields -> 'unhashable type: dict/list' otherwise). The evaluation
      # (BFCLFunctionCallEvaluation) json-loads them transparently. `None`
      # ground_truth (irrelevance/abstention) becomes '' (abstention marker).
      gt = example.get('ground_truth')
      example['ground_truth'] = json.dumps(gt) if gt is not None else ''
      example['function'] = json.dumps(example.get('function', []))
      example['question'] = _bfcl_user_text(example.get('question'))
    return examples[self.start_index : self.end_index]

  def __len__(self) -> int:
    return len(self._examples)

  def __getitem__(self, index: int) -> dict[str, Any]:
    return self._examples[index]


@functools.partial(
    DataSourceRegistry.register, name='simply:bfcl_nonlive_train'
)
@dataclasses.dataclass(frozen=True)
class BFCLNonLiveTrainSource(BFCLSource):
  """Non-live AST categories, used as the RL TRAIN split."""

  categories: tuple[str, ...] = (
      'simple',
      'multiple',
      'parallel',
      'parallel_multiple',
      'irrelevance',
  )


@functools.partial(DataSourceRegistry.register, name='simply:bfcl_live_eval')
@dataclasses.dataclass(frozen=True)
class BFCLLiveEvalSource(BFCLSource):
  """Live AST categories, held out as the EVAL split (no train contamination).

  Only the CALL-REQUIRED (relevance) categories are included, so the scored
  metric measures function-calling accuracy on examples that must emit a call.
  The `live_irrelevance` (abstention) category is excluded: an empty output
  counts as correct there, so including it would mix in a category a
  non-calling model already scores 1.0 on. Dropping it also makes the eval ~40%
  cheaper.
  """

  categories: tuple[str, ...] = (
      'live_simple',
      'live_multiple',
      'live_parallel',
      'live_parallel_multiple',
  )


@functools.partial(DataSourceRegistry.register, name='simply:toolace_train')
@dataclasses.dataclass(frozen=True)
class ToolACETrainSource:
  """ToolACE function-calling data converted to the BFCL example schema.

  Public dataset (Team-ACE/ToolACE, Apache-2.0), converted from its multi-turn
  `[Func(args)]` chat format to BFCL-style {question, function, ground_truth}
  (single reference call -> allowed-set [value]; ~8.5k single-user-turn rows).
  """

  path: str = os.path.join(TOOLUSE_DATASETS_DIR, 'toolace/toolace_bfcl.json')
  start_index: int | None = None
  end_index: int | None = None

  @functools.cached_property
  def _examples(self) -> list[dict[str, Any]]:
    with epath.Path(self.path).open('r') as f:
      data = json.load(f)
    for i, ex in enumerate(data):
      ex['uid'] = f'toolace-{ex.get("id", i)}'
    return data[self.start_index : self.end_index]

  def __len__(self) -> int:
    return len(self._examples)

  def __getitem__(self, index: int) -> dict[str, Any]:
    return self._examples[index]


@functools.partial(
    DataSourceRegistry.register, name='simply:toolace_bfcl_mix_train'
)
@dataclasses.dataclass(frozen=True)
class ToolACEBFCLMixTrainSource:
  """Mixture of ToolACE + BFCL-non-live as the RL train split (concatenated)."""

  @functools.cached_property
  def _examples(self) -> list[dict[str, Any]]:
    ex = list(ToolACETrainSource()._examples) + list(  # pylint: disable=protected-access
        BFCLNonLiveTrainSource()._examples  # pylint: disable=protected-access
    )
    return ex

  def __len__(self) -> int:
    return len(self._examples)

  def __getitem__(self, index: int) -> dict[str, Any]:
    return self._examples[index]


@functools.partial(DataSourceRegistry.register, name='simply:math500_test_l45')
@dataclasses.dataclass(frozen=True)
class MATH500TestL45Source:
  """MATH500 test, levels 4-5 only (the 262 hardest problems).

  The FIXED held-out eval for the rl_qwen2p5_math_1p5b task. Wraps core
  `MATH500Source` (which maps the raw problem/answer/level fields to the
  question/short_answer schema the boxed-answer eval reads) and keeps only the
  level-4 and level-5 problems. Registered here so the RL tasks need no
  core data_lib change. Each example gets a `category` = `level_<n>` for the
  held-out eval's per-category diagnostics (the scored metric pools all levels).
  """

  @functools.cached_property
  def _examples(self) -> list[dict[str, Any]]:
    out = []
    for ex in _core.MATH500Source()._examples:  # pylint: disable=protected-access
      if ex.get('level') in (4, 5):
        out.append(dict(ex, category=f'level_{ex["level"]}'))
    return out

  def __len__(self) -> int:
    return len(self._examples)

  def __getitem__(self, index: int) -> dict[str, Any]:
    return self._examples[index]


# =============================================================================
# Model-PORTING task vocabularies (Falcon-H1 + RecurrentGemma).
#
# Each porting task loads the published model's own tokenizer, staged alongside
# its ORBAX checkpoint. Registered here on the shared TokenizerRegistry
# so a config with vocab_name='FalconH1' / 'RecurrentGemma' resolves it; NO
# core data_lib change is required.
# =============================================================================

# Falcon-H1-0.5B tokenizer (staged alongside the ORBAX checkpoint).
FALCON_H1_VOCAB = os.path.join(
    _config.MODELS_DIR, 'research_bench/Falcon-H1-0.5B-Base/VOCAB'
)

# RecurrentGemma-2B Gemma tokenizer (256000-token Gemma SPM staged to a
# tokenizer.json), alongside the ORBAX checkpoint.
RECURRENTGEMMA_VOCAB = os.path.join(
    _config.MODELS_DIR, 'research_bench/RecurrentGemma-2B/VOCAB'
)


def _register_falcon_h1_vocab():
  """Registers the FalconH1 tokenizer on the shared TokenizerRegistry."""
  if _tok.TokenizerRegistry.get('FalconH1', raise_error=False) is not None:
    return
  _tok.TokenizerRegistry.register(
      lambda: _tok.HuggingFaceVocab(FALCON_H1_VOCAB),
      name='FalconH1',
  )


def _register_recurrentgemma_vocab():
  """Registers the RecurrentGemma tokenizer on the shared registry."""
  if (
      _tok.TokenizerRegistry.get('RecurrentGemma', raise_error=False)
      is not None
  ):
    return
  _tok.TokenizerRegistry.register(
      lambda: _tok.HuggingFaceVocab(RECURRENTGEMMA_VOCAB),
      name='RecurrentGemma',
  )


_register_falcon_h1_vocab()
_register_recurrentgemma_vocab()

# =============================================================================
# OSS-port additions: the C4 pretraining stream + the `nanodo_c4` vocabulary.
#
# Everything above this line is a straight port of the internal data_lib.
# Everything below replaces an internal asset with its public
# equivalent; nothing else in the package reaches for those assets directly.
# =============================================================================

# Root of the repacked C4 corpus: `c4-{split}.{i:05d}-of-{n:05d}.bin` blobs
# with matching `.idx.npy` uint64 offset tables (see setup/build_c4.py). A
# `gs://` prefix works too -- shards are mirrored into `C4_CACHE_DIR` on first
# use, because the reader memory-maps them.
C4_DIR = os.getenv(
    'SIMPLY_C4_DIR',
    os.path.join(_core.DATASETS_DIR, 'c4_bin'),
)
C4_CACHE_DIR = os.getenv(
    'SIMPLY_C4_CACHE_DIR',
    os.path.join(os.path.expanduser('~/.cache/simply/datasets'), 'c4_bin'),
)


def _localize(path: epath.Path) -> str:
  """Returns a local filesystem path for `path`, copying it in if remote.

  `np.memmap` needs a real file, so a shard held on GCS (or any other epath
  backend) is mirrored into `C4_CACHE_DIR` once and reused from there.

  Args:
    path: the shard path, local or remote.

  Returns:
    A local filesystem path holding the shard's bytes.
  """
  if os.path.exists(path.as_posix()):
    return path.as_posix()
  local = os.path.join(C4_CACHE_DIR, path.name)
  if not os.path.exists(local):
    os.makedirs(C4_CACHE_DIR, exist_ok=True)
    tmp = f'{local}.tmp{os.getpid()}'
    with path.open('rb') as src, open(tmp, 'wb') as dst:
      shutil.copyfileobj(src, dst, length=1 << 24)
    os.replace(tmp, local)
  return local


class _Shard:
  """One `.bin` blob + its `.idx.npy` uint64 offset table, memory-mapped."""

  def __init__(self, bin_path: epath.Path):
    idx_path = bin_path.parent / (bin_path.name[: -len('.bin')] + '.idx.npy')
    self.offsets = np.load(_localize(idx_path), mmap_mode='r')
    self.blob = np.memmap(_localize(bin_path), dtype=np.uint8, mode='r')

  def __len__(self) -> int:
    return len(self.offsets) - 1

  def text(self, i: int) -> str:
    lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
    return self.blob[lo:hi].tobytes().decode('utf-8', 'replace')


@DataSourceRegistry.register
@dataclasses.dataclass(frozen=True)
class C4FileSource:
  """Random-access source over the repacked C4 (en) shards.

  Stands in for core's `TFDSSource(name='c4:3.1.0')`: TFDS has no public mirror
  of C4. The `.bin`/`.idx.npy` shards hold exactly the documents of the
  `allenai/c4` `en` release that TFDS c4/en is built from, in the file order of
  that release, so the stream is the same corpus read deterministically -- but
  NOT the same document order as TFDS (which shuffles by key hash when it
  writes shards). Absolute bpb is therefore only comparable within this port;
  the baseline recipes are re-measured here (see PORTING_NOTES.md).

  Attributes:
    split: 'train' or 'validation'.
    num_shards: how many shards (in file order) to expose; 0 => all present.
    root: directory holding the `.bin`/`.idx.npy` pairs (local or `gs://`).
  """

  split: str = 'train'
  num_shards: int = 0
  root: str = C4_DIR

  @functools.cached_property
  def _shards(self) -> list[_Shard]:
    paths = sorted(
        epath.Path(self.root).glob(f'c4-{self.split}.*.bin'),
        key=lambda p: p.name,
    )
    if not paths:
      raise FileNotFoundError(
          f'no C4 shards for split={self.split!r} under {self.root!r}; run '
          'setup/build_c4.py or set SIMPLY_C4_DIR.'
      )
    if self.num_shards:
      paths = paths[: self.num_shards]
    return [_Shard(p) for p in paths]

  @functools.cached_property
  def _starts(self) -> np.ndarray:
    """Global index of the first document of each shard (+ total at the end)."""
    return np.cumsum([0] + [len(s) for s in self._shards])

  def __str__(self) -> str:
    """A location-independent name, for the `eval_protocol` stamp.

    `model_lib._eval_protocol` records `str(source)` in final_result.json and a
    validator compares it across runs; the dataclass repr would pin the stamp
    to one machine's `root`.

    Returns:
      The source identity without its (environment-specific) root.
    """
    return f'C4FileSource(split={self.split!r}, num_shards={self.num_shards})'

  def __getstate__(self) -> dict[str, Any]:
    """Drops the shard cache so each grain worker re-opens its own maps.

    grain pickles the data source into its worker processes, and pickle
    serializes an `np.memmap` BY VALUE -- the whole corpus, once per worker.
    Re-mapping in the worker is O(1) and shares the OS page cache instead.

    Returns:
      The instance dict without the memory-mapped caches.
    """
    cached = ('_shards', '_starts')
    return {k: v for k, v in self.__dict__.items() if k not in cached}

  def __len__(self) -> int:
    return int(self._starts[-1])

  def __getitem__(self, index: int) -> dict[str, Any]:
    if index < 0 or index >= len(self):
      raise IndexError('list index out of range')
    shard = int(np.searchsorted(self._starts, index, side='right')) - 1
    return {'text': self._shards[shard].text(index - int(self._starts[shard]))}


# The internal `nanodo_c4` vocab, `cc_all.32000.100extra.bos.model`: the public
# T5 `cc_all.32000.100extra` sentencepiece model with one `<s>` CONTROL piece
# inserted at index 2 (bos_id=2, 32101 pieces). setup/prepare_assets.py
# regenerates it bit-exactly from the public download and stages it here.
NANODO_C4_VOCAB = os.getenv(
    'SIMPLY_NANODO_C4_VOCAB',
    os.path.join(_core.VOCABS_DIR, 'cc_all.32000.100extra.bos.model'),
)


def _register_nanodo_c4_vocab():
  """Registers the `nanodo_c4` tokenizer on the shared TokenizerRegistry.

  Core simply strips this registration out of its public data_lib (the vocab
  lived on internal storage), but `pretrain_bpb_v32k` is defined by it, so the benchmark
  package owns it.
  """
  if _tok.TokenizerRegistry.get('nanodo_c4', raise_error=False) is not None:
    return
  _tok.TokenizerRegistry.register(
      lambda: _tok.SimplySentencePieceVocab(NANODO_C4_VOCAB),
      name='nanodo_c4',
  )


_register_nanodo_c4_vocab()


# Stand-in for the internal `vb100864_openmix_v1` (pretrain_optimizer_ttt): a
# 100864-piece unigram SPM trained on the public C4 itself, so the model shape
# and step FLOPs of the task are unchanged. setup/prepare_assets.py trains and
# stages it; see ASSETS.md for the recipe and PORTING_NOTES.md for what the
# substitution costs (4.878 bytes/token here vs 4.547 for the internal vocab,
# so absolute losses are on a slightly different scale).
C4_SPM100864_VOCAB = os.getenv(
    'SIMPLY_C4_SPM100864_VOCAB',
    os.path.join(_core.VOCABS_DIR, 'spm-100864-c4-r100-v1.model'),
)


def _register_c4_spm100864_vocab():
  """Registers the 100864-piece C4 tokenizer on the shared registry."""
  if _tok.TokenizerRegistry.get('c4_spm100864', raise_error=False) is not None:
    return
  _tok.TokenizerRegistry.register(
      lambda: _tok.SimplySentencePieceVocab(C4_SPM100864_VOCAB),
      name='c4_spm100864',
  )


_register_c4_spm100864_vocab()
