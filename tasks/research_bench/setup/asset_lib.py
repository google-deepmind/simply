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

"""Shared plumbing for the research-bench asset preparation scripts.

Every asset lands in the canonical simply cache
(`SIMPLY_MODELS` / `SIMPLY_DATASETS` / `SIMPLY_VOCABS`, default
`~/.cache/simply/{models,datasets,vocabs}`) so the ported task code finds it
with no path flags, and can be mirrored verbatim to
`gs://<bucket>/assets/{models,datasets,vocabs}` for Cloud TPU VMs.

Everything here is idempotent and resumable: a step that finds its output
already present and passing its check is skipped.
"""

from __future__ import annotations

import base64
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import time
from typing import Any, Sequence

MODELS_DIR = os.getenv(
    'SIMPLY_MODELS', os.path.expanduser('~/.cache/simply/models/')
)
DATASETS_DIR = os.getenv(
    'SIMPLY_DATASETS', os.path.expanduser('~/.cache/simply/datasets/')
)
VOCABS_DIR = os.getenv(
    'SIMPLY_VOCABS', os.path.expanduser('~/.cache/simply/vocabs/')
)

# Subdirectory of the mirror bucket; `<bucket>/assets/{models,datasets,vocabs}`
# maps 1:1 onto the three cache roots above.
GCS_PREFIX = 'assets'

_KIND_TO_DIR = {
    'models': MODELS_DIR,
    'datasets': DATASETS_DIR,
    'vocabs': VOCABS_DIR,
}


def kind_dir(kind: str) -> str:
  """Returns the local cache root for 'models' / 'datasets' / 'vocabs'."""
  return _KIND_TO_DIR[kind]


def ensure_dir(path: str) -> str:
  os.makedirs(path, exist_ok=True)
  return path


def log(msg: str) -> None:
  print(f'[assets] {msg}', flush=True)


def sha256(path: str, chunk: int = 1 << 20) -> str:
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    while block := f.read(chunk):
      h.update(block)
  return h.hexdigest()


def crc32c(path: str, chunk: int = 1 << 22) -> str:
  """Returns the base64 crc32c of a file, the way GCS reports it.

  crc32c rather than sha256 so the same value can be compared against an
  object's `crc32c_hash` metadata without downloading it.

  Args:
    path: file to hash.
    chunk: read size.

  Returns:
    Base64 of the big-endian crc32c digest, e.g. 'k2Y6aw=='.
  """
  try:
    import google_crc32c  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    raise ImportError(
        'crc32c hashing needs `google-crc32c` (pip install google-crc32c). It'
        ' arrives with google-cloud-storage, which is not pinned by this'
        " repo's requirements.txt. crc32c specifically, not zlib's CRC-32:"
        ' only crc32c matches what GCS stores in object metadata.'
    ) from e

  checksum = google_crc32c.Checksum()
  with open(path, 'rb') as f:
    while block := f.read(chunk):
      checksum.update(block)
  return base64.b64encode(checksum.digest()).decode()


def file_digests(path: str, workers: int = 16) -> dict[str, dict[str, Any]]:
  """Sizes + crc32c of every file under `path`, keyed by relative path.

  Args:
    path: a file or a directory tree.
    workers: threads (the C crc32c runs at several GB/s, so this is IO bound).

  Returns:
    `{relative_path: {'bytes': int, 'crc32c': str}}`; a single file is keyed
    by its basename.
  """
  if os.path.isfile(path):
    names = {os.path.basename(path): path}
  else:
    names = {
        os.path.relpath(os.path.join(root, name), path): os.path.join(
            root, name
        )
        for root, _, files in os.walk(path)
        for name in files
        if not os.path.islink(os.path.join(root, name))
    }
  with concurrent.futures.ThreadPoolExecutor(workers) as pool:
    digests = dict(
        zip(names, pool.map(crc32c, names.values()))
    )
  return {
      rel: {'bytes': os.path.getsize(names[rel]), 'crc32c': digests[rel]}
      for rel in sorted(names)
  }


def gcs_digests(
    bucket: str, kind: str, rel: str
) -> dict[str, dict[str, Any]]:
  """Sizes + crc32c of the mirrored objects, read from GCS metadata.

  No bytes are downloaded, so this is a cheap way to prove an upload landed
  intact.

  Args:
    bucket: mirror bucket.
    kind: 'models' | 'datasets' | 'vocabs'.
    rel: path relative to that cache root.

  Returns:
    `{relative_path: {'bytes': int, 'crc32c': str}}`, keyed the same way as
    `file_digests`.
  """
  uri = gcs_uri(bucket, kind, rel)
  prefix = uri[len('gs://') :].split('/', 1)[1]
  out = subprocess.run(
      [
          'gcloud', 'storage', 'objects', 'list', f'{uri}**',
          "--format=value[separator='|'](name,size,crc32c_hash)",
      ],
      capture_output=True,
      text=True,
      env=gcloud_env(),
  )
  digests = {}
  for line in out.stdout.splitlines():
    name, size, digest = line.rsplit('|', 2)
    key = os.path.relpath(name, prefix) if name != prefix else os.path.basename(
        name
    )
    digests[key] = {'bytes': int(size), 'crc32c': digest}
  return digests


def compare_digests(
    want: dict[str, dict[str, Any]], got: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
  """Returns `{'missing': [...], 'extra': [...], 'corrupt': [...]}`."""
  return {
      'missing': sorted(set(want) - set(got)),
      'extra': sorted(set(got) - set(want)),
      'corrupt': sorted(
          rel for rel in set(want) & set(got)
          if want[rel]['crc32c'] != got[rel]['crc32c']
          or want[rel]['bytes'] != got[rel]['bytes']
      ),
  }


def du_bytes(path: str) -> int:
  """Total size of a file or directory tree, in bytes."""
  if os.path.isfile(path):
    return os.path.getsize(path)
  total = 0
  for root, _, files in os.walk(path):
    for name in files:
      fp = os.path.join(root, name)
      if not os.path.islink(fp):
        total += os.path.getsize(fp)
  return total


def human(nbytes: float) -> str:
  for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
    if abs(nbytes) < 1024 or unit == 'TiB':
      return f'{nbytes:.1f} {unit}'
    nbytes /= 1024
  raise AssertionError('unreachable')


@dataclasses.dataclass
class Timer:
  """Wall-clock timer that logs what it timed."""

  what: str
  start: float = 0.0
  elapsed: float = 0.0

  def __enter__(self) -> 'Timer':
    self.start = time.time()
    return self

  def __exit__(self, *exc: Any) -> None:
    self.elapsed = time.time() - self.start
    log(f'{self.what}: {self.elapsed:.1f}s')


# ---------------------------------------------------------------------------
# HuggingFace downloads.
# ---------------------------------------------------------------------------
def hf_snapshot(
    repo_id: str,
    *,
    repo_type: str = 'model',
    allow_patterns: Sequence[str] | None = None,
    local_dir: str | None = None,
) -> str:
  """Downloads (or reuses) a HuggingFace repo snapshot; returns its path."""
  from huggingface_hub import snapshot_download  # pylint: disable=g-import-not-at-top

  return snapshot_download(
      repo_id,
      repo_type=repo_type,
      allow_patterns=list(allow_patterns) if allow_patterns else None,
      local_dir=local_dir,
      max_workers=16,
  )


def hf_file(
    repo_id: str, filename: str, *, repo_type: str = 'model'
) -> str:
  """Downloads (or reuses) one file from a HuggingFace repo."""
  from huggingface_hub import hf_hub_download  # pylint: disable=g-import-not-at-top

  return hf_hub_download(repo_id, filename, repo_type=repo_type)


def stage_file(src: str, dst: str, *, overwrite: bool = False) -> str:
  """Copies `src` to `dst` (creating parents); returns `dst`."""
  ensure_dir(os.path.dirname(dst))
  if os.path.exists(dst) and not overwrite:
    return dst
  tmp = dst + '.part'
  shutil.copyfile(src, tmp)
  os.replace(tmp, dst)
  return dst


def write_json(obj: Any, dst: str, *, indent: int | None = None) -> str:
  """Atomically writes `obj` as JSON to `dst`."""
  ensure_dir(os.path.dirname(dst))
  tmp = dst + '.part'
  with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(obj, f, indent=indent, ensure_ascii=False)
  os.replace(tmp, dst)
  return dst


def read_jsonl(path: str) -> list[dict[str, Any]]:
  with open(path, 'r', encoding='utf-8') as f:
    return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# GCS mirroring.
# ---------------------------------------------------------------------------
def gcloud_env() -> dict[str, str]:
  """Environment for `gcloud storage`, with the ADC access token if needed.

  Context-Aware-Access blocks the normal gcloud credential path on some
  corp workstations; an explicit `CLOUDSDK_AUTH_ACCESS_TOKEN` works there and
  is harmless elsewhere.
  """
  env = dict(os.environ)
  if 'CLOUDSDK_AUTH_ACCESS_TOKEN' in env:
    return env
  try:
    token = subprocess.run(
        ['gcloud', 'auth', 'application-default', 'print-access-token'],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.strip()
  except (subprocess.SubprocessError, FileNotFoundError):
    return env
  if token:
    env['CLOUDSDK_AUTH_ACCESS_TOKEN'] = token
  return env


def gcs_uri(bucket: str, kind: str, rel: str = '') -> str:
  bucket = bucket.rstrip('/')
  if not bucket.startswith('gs://'):
    bucket = 'gs://' + bucket
  uri = f'{bucket}/{GCS_PREFIX}/{kind}'
  return f'{uri}/{rel}' if rel else uri


def mirror_to_gcs(
    bucket: str,
    kind: str,
    rel: str,
    *,
    dry_run: bool = False,
) -> str:
  """Rsyncs `<cache>/<kind>/<rel>` to `gs://<bucket>/assets/<kind>/<rel>`.

  Args:
    bucket: destination bucket (`gs://...` or bare name).
    kind: 'models' | 'datasets' | 'vocabs'.
    rel: path relative to that cache root (file or directory).
    dry_run: print the command instead of running it.

  Returns:
    The destination URI.
  """
  local = os.path.join(kind_dir(kind), rel)
  dst = gcs_uri(bucket, kind, rel)
  if os.path.isdir(local):
    # A true mirror: drop objects the local tree no longer has, so a restaged
    # asset cannot leave a stale layout behind for a TPU VM to rsync down.
    cmd = [
        'gcloud', 'storage', 'rsync', '--recursive',
        '--delete-unmatched-destination-objects', local, dst,
    ]
  else:
    cmd = ['gcloud', 'storage', 'cp', local, dst]
  log(('DRY-RUN ' if dry_run else '') + ' '.join(cmd))
  if not dry_run:
    subprocess.run(cmd, check=True, env=gcloud_env())
  return dst


def print_tpu_env(bucket: str) -> None:
  """Prints the env a TPU VM needs to read the mirrored cache."""
  print('\n# On the Cloud TPU VM, after `gcloud storage rsync`-ing the mirror:')
  for kind in ('models', 'datasets', 'vocabs'):
    print(
        f'#   gcloud storage rsync --recursive {gcs_uri(bucket, kind)}'
        f' $HOME/.cache/simply/{kind}'
    )
  print('export SIMPLY_MODELS=$HOME/.cache/simply/models/')
  print('export SIMPLY_DATASETS=$HOME/.cache/simply/datasets/')
  print('export SIMPLY_VOCABS=$HOME/.cache/simply/vocabs/')
  print(
      '# Or read straight from GCS (slower, no local disk needed):\n'
      f'# export SIMPLY_MODELS={gcs_uri(bucket, "models")}/\n'
      f'# export SIMPLY_DATASETS={gcs_uri(bucket, "datasets")}/\n'
      f'# export SIMPLY_VOCABS={gcs_uri(bucket, "vocabs")}/'
  )


# ---------------------------------------------------------------------------
# Manifest: what was staged, how big, how long.
# ---------------------------------------------------------------------------
MANIFEST_PATH = os.path.join(DATASETS_DIR, 'research_bench_manifest.json')

# Asset names `record`ed by this process, so `--gcs-bucket` mirrors what the
# command just staged rather than the whole cache.
staged_this_run: list[str] = []


def record(
    name: str,
    kind: str,
    rel: str,
    *,
    source: str,
    extra: dict[str, Any] | None = None,
    seconds: float | None = None,
    hash_files: bool = True,
) -> dict[str, Any]:
  """Appends one staged asset to the on-disk manifest and returns its entry.

  Args:
    name: manifest key.
    kind: 'models' | 'datasets' | 'vocabs'.
    rel: path relative to that cache root.
    source: human-readable provenance.
    extra: extra fields to merge into the entry.
    seconds: build wall clock.
    hash_files: record a per-file crc32c, so `verify --hash` can prove the
      bytes are still the bytes that were staged. Reading the asset back at
      several GB/s costs seconds; it is what catches a corrupt copy that has
      the right size.

  Returns:
    The manifest entry.
  """
  path = os.path.join(kind_dir(kind), rel)
  entry = {
      'name': name,
      'kind': kind,
      'rel': rel,
      'path': path,
      'bytes': du_bytes(path) if os.path.exists(path) else 0,
      'source': source,
      'staged_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
  }
  if hash_files and os.path.exists(path):
    entry['files'] = file_digests(path)
  if seconds is not None:
    entry['seconds'] = round(seconds, 1)
  if extra:
    entry.update(extra)
  manifest = {}
  if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
      manifest = json.load(f)
  manifest[name] = entry
  write_json(manifest, MANIFEST_PATH, indent=2)
  if name not in staged_this_run:
    staged_this_run.append(name)
  log(
      f'staged {name}: {entry["path"]} ({human(entry["bytes"])})'
      + (f' in {entry["seconds"]}s' if seconds is not None else '')
  )
  return entry


def load_manifest() -> dict[str, Any]:
  if not os.path.exists(MANIFEST_PATH):
    return {}
  with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
    return json.load(f)
