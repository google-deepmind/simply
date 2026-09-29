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

"""Stages the vocabularies the research-bench tasks need.

| registered name        | file staged under `SIMPLY_VOCABS`            | source |
|------------------------|----------------------------------------------|--------|
| `nanodo_c4`            | `cc_all.32000.100extra.bos.model`            | derived from the public t5-data SPM (below) |
| `vb262144_gemma3`      | `gemma3_cleaned_262144_v2.spiece.model`      | `google/gemma-3-1b-pt` `tokenizer.model` |
| `Qwen2.5`              | `Qwen2.5/{tokenizer.json,tokenizer_config.json}` | `Qwen/Qwen2.5-Math-1.5B` |
| `Qwen3`                | `Qwen3/{tokenizer.json,tokenizer_config.json}`   | `Qwen/Qwen3-0.6B-Base` |
| `byte256`              | none (`tokenization.ByteVocab`, code only)   | -- |
| `vb100864_openmix_v1`  | no public twin -> `spm-100864-c4-r100-v1.model` | trained here on C4 |

`nanodo_c4` provenance chain (reproducible outside Google):

  https://storage.googleapis.com/t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model
    sha256 839ffa4b9afae8d77834a88b87781849aa021975d6063dec6085633fcaf7171c
    32100 pieces, pad=0 eos=1 unk=2, NO bos
  -> insert a CONTROL piece `<s>` (score 0.0) at index 2, leaving `trainer_spec`
     untouched (SentencePiece derives bos/unk ids from the piece strings)
  -> sha256 15d8dfa11996e0d1e8645897ec4d70b1dec6a7cb6d87df0daf560a3cde479006
     32101 pieces, pad=0 eos=1 bos=2 unk=3

That output is byte-identical to the internal `nanodo_c4`
(`cc_all.32000.100extra.bos.model`), so `pretrain_bpb_v32k` runs on exactly the
internal tokenizer and its reference numbers carry over unchanged.
"""

from __future__ import annotations

import glob
import os
import random
import time
import urllib.request
from typing import Any

import numpy as np

from tasks.research_bench.setup import asset_lib

V = asset_lib.VOCABS_DIR

T5_VOCAB_URL = (
    'https://storage.googleapis.com/t5-data/vocabs/cc_all.32000.100extra/'
    'sentencepiece.model'
)
T5_VOCAB_NAME = 'cc_all.32000.100extra.sentencepiece.model'
T5_VOCAB_SHA256 = (
    '839ffa4b9afae8d77834a88b87781849aa021975d6063dec6085633fcaf7171c'
)
NANODO_C4_NAME = 'cc_all.32000.100extra.bos.model'
NANODO_C4_SHA256 = (
    '15d8dfa11996e0d1e8645897ec4d70b1dec6a7cb6d87df0daf560a3cde479006'
)
NANODO_C4_PIECES = 32101

GEMMA3_VOCAB_NAME = 'gemma3_cleaned_262144_v2.spiece.model'
GEMMA3_REPO = 'google/gemma-3-1b-pt'
QWEN2P5_REPO = 'Qwen/Qwen2.5-Math-1.5B'
QWEN3_REPO = 'Qwen/Qwen3-0.6B-Base'
HF_VOCAB_FILES = ('tokenizer.json', 'tokenizer_config.json')

# Public replacement for the internal `vb100864_openmix_v1` (OpenMix is not
# public). Trained here on the same C4 corpus the task trains on, at the same
# vocab size, so the model shape and FLOP accounting are unchanged.
OPENMIX_SUBSTITUTE_NAME = 'spm-100864-c4-r100-v1.model'
OPENMIX_SUBSTITUTE_VOCAB_SIZE = 100_864


# ---------------------------------------------------------------------------
# nanodo_c4 (32k subword, pretrain_bpb_v32k).
# ---------------------------------------------------------------------------
def build_nanodo_c4() -> str:
  """Downloads the t5-data SPM and derives the 32101-piece `nanodo_c4` twin."""
  from sentencepiece import sentencepiece_model_pb2 as sp_pb2  # pylint: disable=g-import-not-at-top

  raw = os.path.join(asset_lib.ensure_dir(V), T5_VOCAB_NAME)
  if not os.path.exists(raw) or asset_lib.sha256(raw) != T5_VOCAB_SHA256:
    with urllib.request.urlopen(T5_VOCAB_URL) as r, open(raw + '.part', 'wb') as f:
      f.write(r.read())
    os.replace(raw + '.part', raw)
  got = asset_lib.sha256(raw)
  if got != T5_VOCAB_SHA256:
    raise ValueError(f'{T5_VOCAB_URL}: sha256 {got} != {T5_VOCAB_SHA256}')

  model = sp_pb2.ModelProto()
  model.ParseFromString(open(raw, 'rb').read())
  model.pieces.insert(2, sp_pb2.ModelProto.SentencePiece(
      piece='<s>', score=0.0, type=sp_pb2.ModelProto.SentencePiece.CONTROL
  ))
  dst = os.path.join(V, NANODO_C4_NAME)
  with open(dst + '.part', 'wb') as f:
    f.write(model.SerializeToString())
  os.replace(dst + '.part', dst)
  got = asset_lib.sha256(dst)
  if got != NANODO_C4_SHA256:
    raise ValueError(f'{dst}: sha256 {got} != {NANODO_C4_SHA256}')
  return dst


# ---------------------------------------------------------------------------
# Published-model tokenizers.
# ---------------------------------------------------------------------------
def build_gemma3_vocab(force: bool = False) -> str:
  """Stages the Gemma-3 262144-piece SPM under the name core config expects."""
  src = asset_lib.hf_file(GEMMA3_REPO, 'tokenizer.model')
  return asset_lib.stage_file(
      src, os.path.join(V, GEMMA3_VOCAB_NAME), overwrite=force
  )


def build_hf_vocab(name: str, repo: str) -> str:
  """Stages a HuggingFace tokenizer directory as `SIMPLY_VOCABS/<name>`."""
  out = asset_lib.ensure_dir(os.path.join(V, name))
  src = asset_lib.hf_snapshot(repo, allow_patterns=list(HF_VOCAB_FILES))
  for filename in HF_VOCAB_FILES:
    path = os.path.join(src, filename)
    if os.path.exists(path):
      asset_lib.stage_file(path, os.path.join(out, filename), overwrite=True)
  return out


# ---------------------------------------------------------------------------
# vb100864_openmix_v1 substitute: a 100864-piece unigram SPM trained on C4.
# ---------------------------------------------------------------------------
def sample_c4_sentences(
    out_path: str,
    *,
    c4_dir: str,
    num_sentences: int,
    stride: int = 7,
    min_chars: int = 8,
) -> tuple[str, int]:
  """Writes one C4 line per output line, strided over the train shards.

  A deterministic stride (rather than a random sample) keeps the corpus
  reproducible and spreads it over every shard without loading them.

  Args:
    out_path: file to write.
    c4_dir: directory of `c4-train.*.bin` / `.idx.npy` pairs.
    num_sentences: stop after this many lines.
    stride: take every `stride`-th document of each shard.
    min_chars: drop shorter lines (boilerplate/navigation crumbs).

  Returns:
    `(path, num_lines)`.
  """
  paths = sorted(glob.glob(os.path.join(c4_dir, 'c4-train.*.bin')))
  if not paths:
    raise FileNotFoundError(f'no c4 train shards under {c4_dir}')
  written = 0
  asset_lib.ensure_dir(os.path.dirname(out_path))
  with open(out_path + '.part', 'w', encoding='utf-8') as out:
    for path in paths:
      offsets = np.load(path[:-4] + '.idx.npy', mmap_mode='r')
      blob = np.memmap(path, dtype=np.uint8, mode='r')
      for i in range(0, len(offsets) - 1, stride):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        text = blob[lo:hi].tobytes().decode('utf-8', 'replace')
        for line in text.split('\n'):
          line = line.strip()
          if len(line) >= min_chars:
            out.write(line + '\n')
            written += 1
        if written >= num_sentences:
          break
      if written >= num_sentences:
        break
  os.replace(out_path + '.part', out_path)
  return out_path, written


def train_spm(
    *,
    corpus: str,
    model_prefix: str,
    vocab_size: int,
    input_sentence_size: int,
    num_threads: int = 0,
    character_coverage: float = 0.9999,
    extremely_large_corpus: bool = False,
) -> dict[str, Any]:
  """Trains a unigram SentencePiece model; returns timing + peak RSS.

  Args:
    corpus: one sentence per line.
    model_prefix: writes `<prefix>.model` and `<prefix>.vocab`.
    vocab_size: number of pieces.
    input_sentence_size: sentences to sample from `corpus`.
    num_threads: EM threads (default: every core).
    character_coverage: SentencePiece `character_coverage`.
    extremely_large_corpus: widens the suffix-array node index to 64 bit;
      required above ~2^31 characters (the trainer aborts otherwise).

  Returns:
    `{'seconds', 'peak_rss_gib'}`.
  """
  import resource  # pylint: disable=g-import-not-at-top
  import sentencepiece as spm  # pylint: disable=g-import-not-at-top

  start = time.time()
  spm.SentencePieceTrainer.train(
      input=corpus,
      model_prefix=model_prefix,
      model_type='unigram',
      vocab_size=vocab_size,
      character_coverage=character_coverage,
      byte_fallback=True,
      input_sentence_size=input_sentence_size,
      shuffle_input_sentence=True,
      num_threads=num_threads or (os.cpu_count() or 8),
      train_extremely_large_corpus=extremely_large_corpus,
      # Match the `nanodo_c4` id layout so the two pretraining vocabs differ
      # only in size.
      pad_id=0,
      eos_id=1,
      bos_id=2,
      unk_id=3,
  )
  return {
      'seconds': round(time.time() - start, 1),
      'peak_rss_gib': round(
          resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 1
      ),
  }


def build_openmix_substitute(
    *,
    c4_dir: str,
    num_sentences: int = 10_000_000,
    vocab_size: int = OPENMIX_SUBSTITUTE_VOCAB_SIZE,
    keep_corpus: bool = False,
    force: bool = False,
) -> dict[str, Any]:
  """Trains + stages the 100864-piece C4 SPM that replaces `openmix_v1`."""
  dst = os.path.join(asset_lib.ensure_dir(V), OPENMIX_SUBSTITUTE_NAME)
  if os.path.exists(dst) and not force:
    asset_lib.log(f'openmix substitute already trained: {dst}')
    return {'seconds': 0.0, 'peak_rss_gib': 0.0, **check_vocab(dst)}
  corpus = os.path.join(asset_lib.DATASETS_DIR, '_spm_c4_corpus.txt')
  if not os.path.exists(corpus):
    _, lines = sample_c4_sentences(
        corpus, c4_dir=c4_dir, num_sentences=num_sentences
    )
    asset_lib.log(f'spm corpus: {lines} lines -> {corpus}')
  prefix = dst[: -len('.model')]
  stats = train_spm(
      corpus=corpus,
      model_prefix=prefix,
      vocab_size=vocab_size,
      input_sentence_size=num_sentences,
      extremely_large_corpus=num_sentences > 2_000_000,
  )
  if not keep_corpus:
    os.remove(corpus)
  stats.update(check_vocab(dst))
  return stats


# ---------------------------------------------------------------------------
# Checks.
# ---------------------------------------------------------------------------
# Two probes: the t5-derived `nanodo_c4` has no byte fallback, so it cannot
# round-trip characters outside its training distribution -- that is a property
# of the published vocab, not a staging error, and the two fields keep it
# visible instead of looking like a failure.
ROUND_TRIP_ASCII = 'The quick brown fox jumps over 13 lazy dogs.'
ROUND_TRIP_UNICODE = 'naïve café résumé — 日本語 ✅'


def check_vocab(path: str) -> dict[str, Any]:
  """Loads an SPM model and returns its piece count, special ids, round trip."""
  import sentencepiece as spm  # pylint: disable=g-import-not-at-top

  sp = spm.SentencePieceProcessor(model_file=path)
  return {
      'path': path,
      'pieces': sp.GetPieceSize(),
      'pad_id': sp.pad_id(),
      'eos_id': sp.eos_id(),
      'bos_id': sp.bos_id(),
      'unk_id': sp.unk_id(),
      'round_trip_ascii': sp.decode(sp.encode(ROUND_TRIP_ASCII))
      == ROUND_TRIP_ASCII,
      'round_trip_unicode': sp.decode(sp.encode(ROUND_TRIP_UNICODE))
      == ROUND_TRIP_UNICODE,
      'sha256': asset_lib.sha256(path),
  }


def bytes_per_token(
    vocab_path: str,
    *,
    c4_dir: str,
    num_docs: int = 2000,
    seed: int = 0,
) -> float:
  """Measures utf-8 bytes per token on held-out C4 validation documents."""
  import sentencepiece as spm  # pylint: disable=g-import-not-at-top

  paths = sorted(glob.glob(os.path.join(c4_dir, 'c4-validation.*.bin')))
  if not paths:
    raise FileNotFoundError(f'no c4 validation shards under {c4_dir}')
  sp = spm.SentencePieceProcessor(model_file=vocab_path)
  offsets = np.load(paths[0][:-4] + '.idx.npy', mmap_mode='r')
  blob = np.memmap(paths[0], dtype=np.uint8, mode='r')
  rng = random.Random(seed)
  indices = rng.sample(range(len(offsets) - 1), num_docs)
  total_bytes = total_tokens = 0
  for i in indices:
    lo, hi = int(offsets[i]), int(offsets[i + 1])
    text = blob[lo:hi].tobytes().decode('utf-8', 'replace')
    total_bytes += len(text.encode('utf-8'))
    total_tokens += len(sp.encode(text))
  return total_bytes / total_tokens


def check() -> dict[str, Any]:
  """Checks every staged vocab; returns a per-vocab summary."""
  out: dict[str, Any] = {}
  nanodo = os.path.join(V, NANODO_C4_NAME)
  if os.path.exists(nanodo):
    info = check_vocab(nanodo)
    if info['pieces'] != NANODO_C4_PIECES or info['bos_id'] != 2:
      raise ValueError(f'nanodo_c4 vocab is wrong: {info}')
    info['matches_internal_sha256'] = info['sha256'] == NANODO_C4_SHA256
    out['nanodo_c4'] = info
  gemma3 = os.path.join(V, GEMMA3_VOCAB_NAME)
  if os.path.exists(gemma3):
    out['vb262144_gemma3'] = check_vocab(gemma3)
  for name in ('Qwen2.5', 'Qwen3'):
    d = os.path.join(V, name)
    if os.path.isdir(d):
      out[name] = {
          'path': d,
          'files': sorted(os.listdir(d)),
          'bytes': asset_lib.du_bytes(d),
      }
  openmix = os.path.join(V, OPENMIX_SUBSTITUTE_NAME)
  if os.path.exists(openmix):
    out['vb100864_openmix_v1_substitute'] = check_vocab(openmix)
  return out
