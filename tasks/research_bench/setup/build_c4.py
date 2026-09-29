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

r"""Downloads `allenai/c4` (en) and repacks it into `.bin` + `.idx.npy` pairs.

Why not TFDS `c4:3.0.1`: TFDS has no public mirror of C4 and building it needs
the full CommonCrawl pipeline. The `allenai/c4` `en` release holds exactly the
documents TFDS c4/en is built from, in that release's file order, so the eval
stream is the same corpus read deterministically -- but NOT the same document
order as TFDS (which shuffles by key hash when it writes shards).

Format (what `data_lib.C4FileSource` reads):
  `c4-{split}.{i:05d}-of-{n:05d}.bin`      utf-8 text of the shard's documents,
                                           concatenated in file order.
  `c4-{split}.{i:05d}-of-{n:05d}.idx.npy`  uint64 offsets, len == ndocs + 1.

O(1) random access, no dataset framework, mmap-friendly.

Usage:
  python -m tasks.research_bench.setup.build_c4 \
      --num_train_shards=32 --validation
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import gzip
import json
import os
import time
import urllib.request

import numpy as np

from tasks.research_bench.setup import asset_lib

# `allenai/c4` `en` file layout, fixed by the published release.
C4_BASE_URL = 'https://huggingface.co/datasets/allenai/c4/resolve/main/en'
NUM_TRAIN_FILES = 1024
NUM_VALIDATION_FILES = 8

# Where C4FileSource looks by default (`SIMPLY_C4_DIR` overrides).
DEFAULT_OUT_DIR = os.path.join(asset_lib.DATASETS_DIR, 'c4_bin')
DEFAULT_SRC_DIR = os.path.join(asset_lib.DATASETS_DIR, 'c4_json')


def shard_name(split: str, index: int) -> str:
  n = NUM_TRAIN_FILES if split == 'train' else NUM_VALIDATION_FILES
  return f'c4-{split}.{index:05d}-of-{n:05d}'


def download(split: str, index: int, src_dir: str) -> str:
  """Fetches one `json.gz` shard if absent; returns its local path."""
  name = shard_name(split, index) + '.json.gz'
  dst = os.path.join(asset_lib.ensure_dir(src_dir), name)
  if os.path.exists(dst) and os.path.getsize(dst) > 0:
    return dst
  url = f'{C4_BASE_URL}/{name}'
  tmp = dst + '.part'
  with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
    while chunk := r.read(1 << 22):
      f.write(chunk)
  os.replace(tmp, dst)
  return dst


def repack(src: str, out_dir: str) -> str:
  """Repacks one `json.gz` shard into `.bin` + `.idx.npy`; returns the `.bin`."""
  name = os.path.basename(src)
  for suffix in ('.json.gz', '.jsonl.gz'):
    if name.endswith(suffix):
      name = name[: -len(suffix)]
  bin_path = os.path.join(asset_lib.ensure_dir(out_dir), name + '.bin')
  idx_path = bin_path[: -len('.bin')] + '.idx.npy'
  if os.path.exists(bin_path) and os.path.exists(idx_path):
    return bin_path
  offsets = [0]
  with gzip.open(src, 'rt', encoding='utf-8') as f, open(
      bin_path + '.part', 'wb'
  ) as out:
    for line in f:
      blob = json.loads(line)['text'].encode('utf-8')
      out.write(blob)
      offsets.append(offsets[-1] + len(blob))
  os.replace(bin_path + '.part', bin_path)
  np.save(idx_path, np.asarray(offsets, dtype=np.uint64))
  return bin_path


def build_shard(
    split: str, index: int, src_dir: str, out_dir: str, force: bool = False
) -> str:
  """Downloads + repacks one shard; skips whatever already exists."""
  bin_path = os.path.join(out_dir, shard_name(split, index) + '.bin')
  if force:
    for path in (bin_path, bin_path[:-4] + '.idx.npy'):
      if os.path.exists(path):
        os.remove(path)
  elif os.path.exists(bin_path) and os.path.exists(bin_path[:-4] + '.idx.npy'):
    return bin_path
  return repack(download(split, index, src_dir), out_dir)


def build(
    *,
    num_train_shards: int,
    validation: bool,
    src_dir: str = DEFAULT_SRC_DIR,
    out_dir: str = DEFAULT_OUT_DIR,
    workers: int = 8,
    keep_json: bool = False,
    force: bool = False,
) -> dict[str, list[str]]:
  """Builds the requested shards in parallel; returns the paths per split."""
  jobs = [('train', i) for i in range(num_train_shards)]
  if validation:
    jobs += [('validation', i) for i in range(NUM_VALIDATION_FILES)]
  out: dict[str, list[str]] = {'train': [], 'validation': []}
  with concurrent.futures.ThreadPoolExecutor(workers) as pool:
    futures = {
        pool.submit(build_shard, split, i, src_dir, out_dir, force): (split, i)
        for split, i in jobs
    }
    for fut in concurrent.futures.as_completed(futures):
      split, _ = futures[fut]
      out[split].append(fut.result())
  if not keep_json:
    for split, i in jobs:
      src = os.path.join(src_dir, shard_name(split, i) + '.json.gz')
      if os.path.exists(src):
        os.remove(src)
  for split in out:
    out[split].sort()
  return out


def stats(out_dir: str = DEFAULT_OUT_DIR) -> dict[str, dict[str, int]]:
  """Per-split document/byte counts of the shards present in `out_dir`."""
  result = {}
  for split in ('train', 'validation'):
    paths = sorted(glob.glob(os.path.join(out_dir, f'c4-{split}.*.bin')))
    docs = sum(
        len(np.load(p[:-4] + '.idx.npy', mmap_mode='r')) - 1 for p in paths
    )
    result[split] = {
        'shards': len(paths),
        'docs': docs,
        'bytes': sum(os.path.getsize(p) for p in paths),
    }
  return result


def verify(out_dir: str = DEFAULT_OUT_DIR, num_docs: int = 3) -> None:
  """Decodes a few documents per split and asserts the index is consistent."""
  for split in ('train', 'validation'):
    paths = sorted(glob.glob(os.path.join(out_dir, f'c4-{split}.*.bin')))
    if not paths:
      asset_lib.log(f'c4 verify: no {split} shards in {out_dir}')
      continue
    for path in (paths[0], paths[-1]):
      offsets = np.load(path[:-4] + '.idx.npy', mmap_mode='r')
      assert int(offsets[-1]) == os.path.getsize(path), (
          f'{path}: idx tail {int(offsets[-1])} != file size'
          f' {os.path.getsize(path)}'
      )
      blob = np.memmap(path, dtype=np.uint8, mode='r')
      for i in range(min(num_docs, len(offsets) - 1)):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        text = blob[lo:hi].tobytes().decode('utf-8')
        assert text, f'{path}: empty doc {i}'
      asset_lib.log(
          f'c4 verify {os.path.basename(path)}: {len(offsets) - 1} docs,'
          f' {os.path.getsize(path)} bytes, first doc'
          f' {blob[0:60].tobytes().decode("utf-8", "replace")!r}'
      )


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--num_train_shards', type=int, default=32)
  ap.add_argument(
      '--no_validation', dest='validation', action='store_false',
      help='skip the (full, 8-shard) validation split',
  )
  ap.add_argument('--src_dir', default=DEFAULT_SRC_DIR)
  ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
  ap.add_argument('--workers', type=int, default=8)
  ap.add_argument(
      '--keep_json', action='store_true',
      help='keep the downloaded json.gz (default: delete after repacking)',
  )
  ap.add_argument('--force', action='store_true')
  ap.add_argument('--verify_only', action='store_true')
  args = ap.parse_args()

  if not args.verify_only:
    t0 = time.time()
    build(
        num_train_shards=args.num_train_shards,
        validation=args.validation,
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        workers=args.workers,
        keep_json=args.keep_json,
        force=args.force,
    )
    asset_lib.log(f'c4 build: {time.time() - t0:.0f}s')
  verify(args.out_dir)
  for split, s in stats(args.out_dir).items():
    asset_lib.log(
        f'c4 {split}: {s["shards"]} shards, {s["docs"]} docs,'
        f' {asset_lib.human(s["bytes"])}'
    )


if __name__ == '__main__':
  main()
