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
r"""Streams a HuggingFace Kimi K3 release into a Simply checkpoint.

The release is 1.56 TB across 96 safetensors shards and the converted tree is
1.45 TiB, so nothing here materializes either: `build_plan` decides what every
target leaf is made of, `ShardReader` reads byte ranges out of the shards, and
`GroupMaterializer` builds one parameter group at a time and drops it. The
routed experts stay MXFP4-packed -- `KimiK3Format` decodes them on restore --
which is what keeps the write at 1.45 TiB instead of 5.06 TiB of dense bf16.

`convert_hf_checkpoint.py` is the binary over this: it owns the flags and
nothing else. Simply's own `tools/hf_to_orbax.py` cannot do this job -- it
loads every tensor into memory and writes them under HuggingFace names, leaving
the mapping to restore time.
"""

import collections
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import concurrent.futures
import dataclasses
import gc
import json
import os
import random
import re
import struct
import threading
import time
from typing import Any
from absl import logging
from etils import epath
import jax
import ml_dtypes
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint import type_handlers
from simply.utils import checkpoint_lib
from simply.utils import pytree as pytree_lib
from simply.zoo.kimi_k3 import config_lib as k3_config_lib
from simply.zoo.kimi_k3.utils import ckpt_format
from simply.zoo.kimi_k3.utils import hf_params

PyTree = Any
Path = tuple[str, ...]

# Tensors the text backbone intentionally does not consume. MoonViT-V2 and its
# projector are part of the multimodal wrapper, not of `KimiLinearForCausalLM`.
IGNORED_SOURCE_PREFIXES = ('vision_tower.', 'mm_projector.')

SAFETENSORS_DTYPES: Mapping[str, Any] = {
    'BOOL': np.bool_,
    'U8': np.uint8,
    'I8': np.int8,
    'I16': np.int16,
    'U16': np.uint16,
    'F16': np.float16,
    'BF16': ml_dtypes.bfloat16,
    'I32': np.int32,
    'U32': np.uint32,
    'F32': np.float32,
    'F64': np.float64,
    'I64': np.int64,
    'U64': np.uint64,
    'F8_E4M3': ml_dtypes.float8_e4m3fn,
    'F8_E5M2': ml_dtypes.float8_e5m2,
}

PARAM_DTYPES: Mapping[str, Any] = {
    'source': None,
    'bfloat16': ml_dtypes.bfloat16,
    'float32': np.float32,
}

_GIB = 2**30


# --- the HuggingFace side ----------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SourceTensor:
  """One HF tensor, addressed as a byte range in its shard."""

  name: str
  shard: str
  start: int
  end: int
  dtype: np.dtype
  shape: tuple[int, ...]

  @property
  def nbytes(self) -> int:
    return self.end - self.start


@dataclasses.dataclass(frozen=True)
class HfIndex:
  """`model.safetensors.index.json` plus the parsed shard headers."""

  hf_dir: str
  shard_of: Mapping[str, str]
  tensors: Mapping[str, SourceTensor]
  total_size: int
  missing_shards: tuple[str, ...]
  # Shards whose file is shorter than their own header says. This is the only
  # honest integrity check on the download: the shards are not all the same
  # size, so no constant works.
  truncated_shards: tuple[str, ...]

  def __getitem__(self, name: str) -> SourceTensor:
    tensor = self.tensors.get(name)
    if tensor is not None:
      return tensor
    shard = self.shard_of.get(name)
    if shard is None:
      raise KeyError(f'{name}: not in model.safetensors.index.json')
    raise KeyError(f'{name}: lives in {shard}, which was not read')

  def __contains__(self, name: str) -> bool:
    return name in self.shard_of


def _read_shard_header(
    path: epath.Path, shard: str
) -> tuple[dict[str, SourceTensor], bool]:
  """Parses one safetensors header: `<u64 length><json><data>`.

  Args:
    path: the shard.
    shard: its basename, recorded on every tensor.

  Returns:
    Its tensors, and whether the file is at least as long as the header says.
  """
  with path.open('rb') as f:
    (header_len,) = struct.unpack('<Q', f.read(8))
    header = json.loads(f.read(header_len))
  base = 8 + header_len
  tensors = {}
  for name, spec in header.items():
    if name == '__metadata__':
      continue
    dtype = SAFETENSORS_DTYPES.get(spec['dtype'])
    if dtype is None:
      raise ValueError(f'{shard}:{name}: unsupported dtype {spec["dtype"]}')
    start, end = spec['data_offsets']
    tensors[name] = SourceTensor(
        name=name,
        shard=shard,
        start=base + start,
        end=base + end,
        dtype=np.dtype(dtype),
        shape=tuple(spec['shape']),
    )
  complete = path.stat().length >= max(
      (t.end for t in tensors.values()), default=base
  )
  return tensors, complete


def read_index(
    hf_dir: str, shards: Iterable[str] | None = None, threads: int = 16
) -> HfIndex:
  """Reads the shard index and the headers of the shards it points at.

  Args:
    hf_dir: directory holding `model.safetensors.index.json` and the shards.
    shards: shard basenames to parse; `None` parses all of them. Headers are
      only needed for shapes and byte ranges, so a partial download can still be
      checked for name coverage.
    threads: concurrent header reads (each is one open + one small read).

  Returns:
    The index; `missing_shards` lists the shards that were asked for but are
    not on disk, `truncated_shards` those still shorter than their header.
  """
  root = epath.Path(hf_dir)
  index = json.loads((root / 'model.safetensors.index.json').read_text())
  shard_of: Mapping[str, str] = index['weight_map']
  total_size = int(index.get('metadata', {}).get('total_size', 0))
  wanted = sorted(set(shard_of.values()) if shards is None else set(shards))

  present, missing = [], []
  for shard in wanted:
    (present if (root / shard).exists() else missing).append(shard)
  tensors: dict[str, SourceTensor] = {}
  truncated = []
  with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
    for shard, (parsed, complete) in zip(
        present,
        pool.map(lambda s: _read_shard_header(root / s, s), present),
    ):
      tensors.update(parsed)
      if not complete:
        truncated.append(shard)
  return HfIndex(
      hf_dir=hf_dir,
      shard_of=shard_of,
      tensors=tensors,
      total_size=total_size,
      missing_shards=tuple(missing),
      truncated_shards=tuple(truncated),
  )


class ShardReader:
  """Reads tensors out of the shards by byte range.

  Not the `safetensors` library: `read_index` needs every tensor's shape,
  dtype and byte range before anything is materialized, and the truncation
  check has no `safe_open` equivalent, so `_read_shard_header` stays either way
  and the dependency would only replace the slicing below. Handles are per
  thread because a file object carries a seek position, so sharing one would
  serialize every read behind a lock; a single `safe_open` handle per shard has
  the same problem. (Its numpy
  backend does return bfloat16 once `ml_dtypes` is imported, but nothing
  upstream tests that combination.)
  """

  def __init__(self, hf_dir: str, max_open: int = 16):
    self._root = epath.Path(hf_dir)
    self._max_open = max_open
    self._local = threading.local()
    self._lock = threading.Lock()
    self._all_handles: list[Any] = []

  def _handle(self, shard: str) -> Any:
    """Returns the calling thread's handle for `shard`, opening it if needed.

    Args:
      shard: shard file name, relative to the release directory.

    Returns:
      An open file object owned by this thread. A thread keeps at most
      `max_open` of them and closes the least recently used one to stay under
      that bound, because the release has hundreds of shards.
    """
    handles = getattr(self._local, 'handles', None)
    if handles is None:
      handles = self._local.handles = collections.OrderedDict()
    if shard in handles:
      handles.move_to_end(shard)
      return handles[shard]
    if len(handles) >= self._max_open:
      _, stale = handles.popitem(last=False)
      stale.close()
    handle = (self._root / shard).open('rb')
    handles[shard] = handle
    with self._lock:
      self._all_handles.append(handle)
    return handle

  def read(self, tensor: SourceTensor) -> np.ndarray:
    """Returns one tensor, bit-exact, in its source dtype."""
    handle = self._handle(tensor.shard)
    handle.seek(tensor.start)
    buf = handle.read(tensor.nbytes)
    if len(buf) != tensor.nbytes:
      raise IOError(
          f'{tensor.name}: read {len(buf)} of {tensor.nbytes} bytes from'
          f' {tensor.shard} at {tensor.start}'
      )
    # frombuffer aliases a read-only buffer; Orbax and the transforms want a
    # writable array.
    return np.frombuffer(buf, dtype=tensor.dtype).reshape(tensor.shape).copy()

  def close(self) -> None:
    with self._lock:
      for handle in self._all_handles:
        handle.close()
      self._all_handles.clear()

  def __enter__(self) -> 'ShardReader':
    return self

  def __exit__(self, *exc: Any) -> None:
    self.close()


# --- the plan ----------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TargetLeaf:
  """One array in the Simply tree, and the HF tensors that produce it."""

  path: Path
  shape: tuple[int, ...]
  dtype: np.dtype
  group: Path

  @property
  def nbytes(self) -> int:
    return (
        int(np.prod(self.shape)) * self.dtype.itemsize
        if self.shape
        else (self.dtype.itemsize)
    )

  @property
  def name(self) -> str:
    return '/'.join(self.path)


@dataclasses.dataclass(frozen=True)
class Plan:
  """Every target leaf, plus what each conversion group reads."""

  leaves: tuple[TargetLeaf, ...]
  group_sources: Mapping[Path, tuple[str, ...]]
  layers: tuple[int, ...]

  @property
  def output_bytes(self) -> int:
    return sum(leaf.nbytes for leaf in self.leaves)

  @property
  def sources(self) -> tuple[str, ...]:
    names: list[str] = []
    for group_names in self.group_sources.values():
      names.extend(group_names)
    return tuple(names)

  def unreadable_shards(self, index: 'HfIndex') -> tuple[str, ...]:
    """Shards this plan reads that are missing or shorter than their header."""
    broken = set(index.missing_shards) | set(index.truncated_shards)
    return tuple(
        sorted({index.shard_of[name] for name in self.sources} & broken)
    )


def flatten_tree(tree: PyTree, prefix: Path = ()) -> Iterator[tuple[Path, Any]]:
  """Yields `(path, leaf)` for every leaf of a nested dict, in insertion order.

  Not `pytree.to_flat_dict`: it goes through `jax.tree_util`, which sorts dict
  keys at every level, and the conversion order is what bounds this module's
  memory (`GroupMaterializer`).

  Args:
    tree: a nested dict.
    prefix: path of `tree` within its root.

  Yields:
    `(path, leaf)` per leaf.
  """
  if isinstance(tree, Mapping):
    for key, value in tree.items():
      yield from flatten_tree(value, prefix + (key,))
  else:
    yield prefix, tree


class _ZeroTensors:
  """Zero-filled stand-ins for HF tensors, recording what was asked for.

  `np.zeros` is calloc, so the source pages are never faulted in; the plan
  pass costs one transposed copy per leaf and no I/O at all.
  """

  def __init__(self, index: HfIndex):
    self._index = index
    self.requested: list[str] = []

  def get(self, name: str) -> np.ndarray:
    tensor = self._index[name]
    self.requested.append(name)
    return np.zeros(tensor.shape, tensor.dtype)


def _expand_expert_sources(names: Sequence[str], num_experts: int) -> list[str]:
  """Repeats an expert-0 read pattern over all experts, in converter order.

  The plan pass runs with `num_experts=1` (see `build_plan`), so it only ever
  asks for expert 0. The converter's loop is expert-major, so the real read
  order is this pattern repeated with the index substituted.

  Args:
    names: the sources of one expert group, as read with `num_experts=1`.
    num_experts: the number of experts the real model has.

  Returns:
    The read order of the group for all `num_experts` experts.
  """
  expanded = []
  for expert in range(num_experts):
    for name in names:
      expanded.append(re.sub(r'(\.experts\.)0\.', rf'\g<1>{expert}.', name))
  return expanded


def _is_routed_expert(path: Path) -> bool:
  """Whether a group or leaf path names routed experts, not the shared one."""
  return 'experts' in path and 'shared' not in path


def build_plan(
    converter: hf_params.KimiK3HfConverter,
    index: HfIndex,
    layers: Sequence[int],
) -> Plan:
  """Runs the converter over zero-filled tensors to learn the target tree.

  The routed expert groups are planned with a single expert -- planning a real
  one would build the whole 20 GiB stack out of zeros -- and their leading
  dimension is scaled back up to `config.num_experts`. `GroupMaterializer.leaf`
  rejects any array that disagrees with its planned shape or dtype, so the
  shortcut cannot drift silently.

  Args:
    converter: the converter the conversion will use (defines the mapping).
    index: the HF index; supplies every source shape and dtype.
    layers: layer indices to convert.

  Returns:
    The plan.

  Raises:
    KeyError: if the converter asks for a tensor the index does not have.
  """
  num_experts = converter.config.num_experts
  planner = hf_params.KimiK3HfConverter(
      dataclasses.replace(converter.config, num_experts=1),
      prefix=converter.prefix,
      dequantize_experts=converter.dequantize_experts,
      dtype=converter.dtype,
      expert_dtype=converter.expert_dtype,
  )
  leaves: list[TargetLeaf] = []
  group_sources: dict[Path, tuple[str, ...]] = {}
  for group in converter.group_paths(layers):
    zeros = _ZeroTensors(index)
    subtree = planner.convert_group(zeros.get, group)
    expert_group = _is_routed_expert(group)
    for suffix, array in flatten_tree(subtree):
      shape = tuple(array.shape)
      if expert_group:
        shape = (num_experts,) + shape[1:]
      leaves.append(
          TargetLeaf(
              path=('params',) + group + suffix,
              shape=shape,
              dtype=np.dtype(array.dtype),
              group=group,
          )
      )
    sources = zeros.requested
    if expert_group:
      sources = _expand_expert_sources(sources, num_experts)
    group_sources[group] = tuple(sources)
  return Plan(
      leaves=tuple(leaves),
      group_sources=group_sources,
      layers=tuple(layers),
  )


# --- materialization ---------------------------------------------------------


class _PrefetchingGet:
  """Serves one group's sources, reading ahead in a thread pool.

  The converter pulls tensors one at a time, so without a read-ahead the
  896 x 2 reads of an expert stack are strictly serialized against the
  dequantization of the previous one. The plan knows the exact read order, so
  the window is simply the next `lookahead` sources.
  """

  def __init__(
      self,
      reader: ShardReader,
      index: HfIndex,
      sources: Sequence[str],
      pool: concurrent.futures.ThreadPoolExecutor,
      lookahead: int = 16,
  ):
    self._reader = reader
    self._index = index
    self._sources = sources
    self._pool = pool
    self._lookahead = lookahead
    self._next = 0
    self._pending: dict[str, concurrent.futures.Future[np.ndarray]] = {}

  def _submit_through(self, limit: int) -> None:
    while self._next < min(limit, len(self._sources)):
      name = self._sources[self._next]
      self._next += 1
      if name not in self._pending:
        self._pending[name] = self._pool.submit(
            self._reader.read, self._index[name]
        )

  def get(self, name: str) -> np.ndarray:
    """Returns one source tensor, from the read-ahead window when it is there.

    Args:
      name: HF tensor name.

    Returns:
      The tensor, in its source dtype. A name the window does not hold is read
      synchronously.
    """
    self._submit_through(self._next + self._lookahead)
    future = self._pending.pop(name, None)
    if future is None:
      # Either a probe the converter expects to fail (it tries the dense expert
      # name before the packed pair) or a read out of plan order; correctness
      # never depends on the prediction, only speed.
      tensor = self._index[name]
      logging.log_first_n(
          logging.WARNING, 'read %s outside the planned order', 3, name
      )
      return self._reader.read(tensor)
    self._submit_through(self._next + 1)
    return future.result()

  def drain(self) -> None:
    for future in self._pending.values():
      future.cancel()
    self._pending.clear()


class GroupMaterializer:
  """Builds conversion groups on demand and hands out their leaves.

  Orbax serializes in sorted-key order, which walks a group's own leaves
  contiguously but interrupts a group with the nested groups of its subtree --
  `block_i/ffn` by its three expert stacks. A group is therefore kept until a
  group it does not contain is built, and its leaves are freed one by one as
  they are served: every group is built exactly once, and residency
  (`resident_bytes`) stays at one group plus the groups it is nested in -- one
  routed expert stack (19.7 GiB at bf16) and the small projections above it.
  """

  def __init__(
      self,
      converter: hf_params.KimiK3HfConverter,
      plan: Plan,
      index: HfIndex,
      reader: ShardReader,
      read_threads: int = 8,
  ):
    self._converter = converter
    self._plan = plan
    self._index = index
    self._reader = reader
    self._pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=read_threads, thread_name_prefix='shard_read'
    )
    # Leaves not yet served, by the group that built them.
    self._pending: dict[Path, dict[Path, np.ndarray]] = {}
    self._built: set[Path] = set()
    self.groups_built = 0
    self.groups_rebuilt = 0
    self.peak_resident_bytes = 0

  @property
  def resident_bytes(self) -> int:
    """Bytes of leaves that have been built and not yet served."""
    return sum(
        sum(array.nbytes for array in arrays.values())
        for arrays in self._pending.values()
    )

  def _build(self, group: Path) -> None:
    """Converts one group, dropping every group that does not contain it.

    Args:
      group: the group to build; its leaves are then served by `leaf`.
    """
    if group in self._built:
      # The cache assumes a group is only ever interrupted by its own nested
      # groups; a rebuild means the write order stopped satisfying that, and
      # costs a re-read of the group's sources rather than correctness.
      self.groups_rebuilt += 1
      logging.warning(
          'rebuilding %s: reading its sources again', '/'.join(group)
      )
    self._built.add(group)
    self._pending = {
        path: arrays
        for path, arrays in self._pending.items()
        if group[: len(path)] == path
    }
    gc.collect()
    started = time.monotonic()
    prefetch = _PrefetchingGet(
        self._reader, self._index, self._plan.group_sources[group], self._pool
    )
    subtree = self._converter.convert_group(prefetch.get, group)
    prefetch.drain()
    arrays: dict[Path, np.ndarray] = {
        ('params',) + group + suffix: array
        for suffix, array in flatten_tree(subtree)
    }
    self._pending[group] = arrays
    self.groups_built += 1
    self.peak_resident_bytes = max(
        self.peak_resident_bytes, self.resident_bytes
    )
    elapsed = max(time.monotonic() - started, 1e-9)
    nbytes = sum(a.nbytes for a in arrays.values())
    logging.info(
        'built %s: %d leaves, %.2f GiB in %.1fs (%.0f MiB/s)',
        '/'.join(group),
        len(arrays),
        nbytes / _GIB,
        elapsed,
        nbytes / 2**20 / elapsed,
    )

  def leaf(self, leaf: TargetLeaf) -> np.ndarray:
    if leaf.path not in self._pending.get(leaf.group, {}):
      self._build(leaf.group)
    array = self._pending[leaf.group].pop(leaf.path)
    if array.shape != leaf.shape or array.dtype != leaf.dtype:
      raise ValueError(
          f'{leaf.name}: planned {leaf.shape} {leaf.dtype} but the converter'
          f' produced {array.shape} {array.dtype}'
      )
    return array

  def close(self) -> None:
    self._pending = {}
    self._pool.shutdown(wait=False, cancel_futures=True)


# --- the Orbax write ---------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WriteOptions:
  """Knobs that bound memory and set the on-disk read granularity."""

  # The chunk shape is the restore granularity: an unchunked 20 GiB expert
  # stack forces every host to read the whole thing.
  chunk_bytes: int = 32 * 2**20
  # Bytes of materialized parameters that may be awaiting a write. Draining
  # each write before starting the next bounds memory perfectly and costs a
  # full storage round trip per parameter.
  max_inflight_bytes: int = 24 * _GIB
  # Abort rather than let the OOM killer pick a victim; 0 disables.
  max_rss_bytes: int = 100 * _GIB


def _rss_bytes() -> int:
  """Returns this process's resident size; /proc directly, so it stays cheap."""
  try:
    with open('/proc/self/statm', 'rb') as f:
      return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
  except (OSError, IndexError, ValueError):
    return 0


class MemoryCeilingExceededError(RuntimeError):
  """Resident memory passed `WriteOptions.max_rss_bytes`."""


class _LazyLeaf:
  """A promise for one parameter, resolved when Orbax serializes it.

  Orbax inspects `shape`/`dtype` to build the checkpoint metadata before
  serializing anything, and the plan knows both exactly, so nothing has to be
  read to answer.
  """

  def __init__(
      self,
      leaf: TargetLeaf,
      materialize: Callable[[TargetLeaf], np.ndarray],
      options: WriteOptions,
  ):
    self.leaf = leaf
    self.materialize = materialize
    self.options = options
    self.shape = tuple(leaf.shape)
    self.dtype = leaf.dtype
    self.ndim = len(self.shape)
    self.size = int(np.prod(self.shape)) if self.shape else 1
    self.nbytes = leaf.nbytes

  def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
    del copy  # The materialized array is always freshly built.
    array = self.materialize(self.leaf)
    if dtype is not None and np.dtype(dtype) != array.dtype:
      array = array.astype(dtype)
    return array

  def __repr__(self) -> str:
    return f'_LazyLeaf({self.leaf.name}, {self.shape}, {self.dtype.name})'


# Progress lives in a module global because Orbax owns the serialization loop
# and hands the registered handler instance no per-run state.
_PROGRESS = {'leaves': 0, 'bytes': 0, 'start': 0.0}


def _log_progress(leaf: _LazyLeaf, total_bytes: int) -> None:
  _PROGRESS['leaves'] += 1
  _PROGRESS['bytes'] += leaf.nbytes
  elapsed = max(time.monotonic() - _PROGRESS['start'], 1e-9)
  done = _PROGRESS['bytes']
  logging.info(
      'wrote %s (%.2f GiB); %.1f/%.1f GiB, %.0f MiB/s, rss %.1f GiB, eta %s',
      leaf.leaf.name,
      leaf.nbytes / _GIB,
      done / _GIB,
      total_bytes / _GIB,
      done / 2**20 / elapsed,
      _rss_bytes() / _GIB,
      format_duration((total_bytes - done) / max(done / elapsed, 1.0)),
  )


class _LazyLeafHandler(type_handlers.NumpyHandler):
  """Serializes `_LazyLeaf`s with a bounded amount of data in flight.

  Orbax hands a type handler every leaf of that type in one `serialize` call,
  so materializing them all up front would materialize the whole checkpoint.
  Draining each write before starting the next is also wrong: it costs a
  storage round trip per parameter (~23 s to a remote CNS cell). The bound
  that matters is bytes resident, so this keeps a window of writes open and
  drains the oldest only when the window is full.
  """

  async def serialize(self, values, infos, args=None):  # pylint: disable=invalid-overridden-method
    options = values[0].options if values else WriteOptions()
    total = sum(v.nbytes for v in values)
    window: collections.deque[tuple[Any, list[Any], int]] = collections.deque()
    inflight = 0

    def drain_one() -> None:
      nonlocal inflight
      array, futures, nbytes = window.popleft()
      for future in futures:
        future.result()
      inflight -= nbytes
      del futures, array

    for i, value in enumerate(values):
      # Drain before materializing, so the new array is never what pushes us
      # over the window.
      while window and inflight + value.nbytes > options.max_inflight_bytes:
        drain_one()
      rss = _rss_bytes()
      if options.max_rss_bytes and rss > options.max_rss_bytes:
        raise MemoryCeilingExceededError(
            f'resident memory {rss / _GIB:.1f} GiB exceeded the'
            f' {options.max_rss_bytes / _GIB:.1f} GiB ceiling after'
            f' {_PROGRESS["leaves"]} leaves, about to build {value.leaf.name}'
        )
      array = np.asarray(value)
      futures = await super().serialize(
          [array], [infos[i]], None if args is None else [args[i]]
      )
      window.append((array, list(futures), value.nbytes))
      inflight += value.nbytes
      del array, futures
      _log_progress(value, total)

    while window:
      drain_one()
    gc.collect()
    return []


type_handlers.register_type_handler(
    _LazyLeaf, _LazyLeafHandler(), override=True
)


def write_checkpoint(
    plan: Plan,
    materialize: Callable[[TargetLeaf], np.ndarray],
    out_dir: str,
    step: int,
    options: WriteOptions,
) -> None:
  """Writes the plan as a Simply-native Orbax checkpoint at `out_dir/step`.

  The tree is already in Simply layout, so it is tagged `KimiK3Format`, which
  passes matching leaves straight through at restore time and decodes the MXFP4
  expert pairs (if any) to dense `[E, in, out]`.

  This mirrors `checkpoint_lib.save_checkpoint`, down to the format-tag
  convention, and diverges only in passing `save_args`: the per-leaf chunking
  below is the restore granularity of a 20 GiB leaf, and core's saver forwards
  its `**kwargs` to `CheckpointManager.save`, not to `PyTreeSave`.

  Args:
    plan: the target tree.
    materialize: builds one leaf; called once per leaf, in Orbax's serialization
      order.
    out_dir: destination directory (local or any remote filesystem Orbax can
      write).
    step: checkpoint step to write under `out_dir`.
    options: chunking and memory bounds.
  """
  biggest = max(plan.leaves, key=lambda leaf: leaf.nbytes)
  logging.info(
      'writing %d leaves (%.2f GiB) to %s/%d; largest leaf %s at %.2f GiB',
      len(plan.leaves),
      plan.output_bytes / _GIB,
      out_dir,
      step,
      biggest.name,
      biggest.nbytes / _GIB,
  )
  tree = ocp.tree.from_flat_dict(
      {leaf.path: _LazyLeaf(leaf, materialize, options) for leaf in plan.leaves}
  )
  save_args = ocp.tree.from_flat_dict({
      leaf.path: ocp.SaveArgs(
          # Orbax clamps the chunk size to the array size.
          chunk_byte_size=min(options.chunk_bytes, leaf.nbytes),
          dtype=None,  # Never let Orbax cast: the plan resolved every dtype.
      )
      for leaf in plan.leaves
  })
  _PROGRESS.update(leaves=0, bytes=0, start=time.monotonic())
  with ocp.CheckpointManager(out_dir) as manager:
    manager.save(
        step,
        args=ocp.args.Composite(
            state=ocp.args.PyTreeSave(tree, save_args=save_args),
            metadata=ocp.args.JsonSave(
                pytree_lib.dump({  # pyrefly: ignore[bad-argument-type]
                    checkpoint_lib.CHECKPOINT_FORMAT_KEY: (
                        ckpt_format.KimiK3Format()
                    )
                })
            ),
        ),
    )
    manager.wait_until_finished()
  elapsed = max(time.monotonic() - _PROGRESS['start'], 1e-9)
  logging.info(
      'wrote %.2f GiB in %s (%.0f MiB/s)',
      _PROGRESS['bytes'] / _GIB,
      format_duration(elapsed),
      _PROGRESS['bytes'] / 2**20 / elapsed,
  )


def restore_leaves(
    out_dir: str,
    step: int,
    plan: Plan,
    wanted: Sequence[TargetLeaf] | None = None,
) -> dict[Path, np.ndarray]:
  """Restores `wanted` (default: everything) from a written checkpoint.

  Not `checkpoint_lib.load_checkpoint_from_path`: that runs the format's
  transforms and rebuilds the tree from a target abstract state, so it decodes,
  reshards and casts every leaf -- the three things `verify`'s bytewise
  comparison against the source exists to check.

  Args:
    out_dir: the checkpoint directory that was written.
    step: the step under it.
    plan: the plan that was written; it supplies the tree structure, so the
      restore target can be built without reading the checkpoint's metadata.
    wanted: leaves to read. Everything else is left on disk behind
      `ocp.PLACEHOLDER`, the mechanism Simply's loader uses to skip leaves its
      transform never reads.

  Returns:
    The restored arrays, keyed by tree path.
  """
  paths = {leaf.path for leaf in (plan.leaves if wanted is None else wanted)}
  item: dict[Path, Any] = {}
  restore_args: dict[Path, Any] = {}
  for leaf in plan.leaves:
    if leaf.path in paths:
      item[leaf.path] = jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
      restore_args[leaf.path] = ocp.RestoreArgs(restore_type=np.ndarray)
    else:
      item[leaf.path] = ocp.PLACEHOLDER
      restore_args[leaf.path] = ocp.RestoreArgs()
  with ocp.CheckpointManager(out_dir) as manager:
    restored = manager.restore(
        step,
        args=ocp.args.Composite(
            state=ocp.args.PyTreeRestore(
                item=ocp.tree.from_flat_dict(item),
                restore_args=ocp.tree.from_flat_dict(restore_args),
            )
        ),
    ).state
  return {
      path: np.asarray(value)
      for path, value in flatten_tree(restored)
      if path in paths
  }


def read_checkpoint_format(
    out_dir: str, step: int
) -> checkpoint_lib.CheckpointFormat:
  """Returns the format tag recorded in a written checkpoint's metadata."""
  with ocp.CheckpointManager(out_dir) as manager:
    metadata = manager.restore(
        step, args=ocp.args.Composite(metadata=ocp.args.JsonRestore())
    ).metadata
  return pytree_lib.load(metadata)[checkpoint_lib.CHECKPOINT_FORMAT_KEY]


def leaf_stats(leaf: TargetLeaf, array: np.ndarray) -> str:
  """Returns a one-line sanity summary of a materialized leaf.

  Bitwise equality with the source already proves the write; this catches the
  other half -- a source that is itself wrong (all zeros, NaN, denormal soup).

  Args:
    leaf: the leaf being described.
    array: its value.
  """
  if array.dtype == np.uint8:
    # E8M0 exponents: 0xFF is the only NaN.
    if leaf.path[-1] == hf_params.MXFP4_SCALE_KEY:
      return (
          f'e8m0 codes {array.min()}..{array.max()},'
          f' NaN(0xFF)={int((array == 255).sum())}'
      )
    return (
        f'packed nibbles, {int((array == 0).sum()) / array.size:.3f} zero bytes'
    )
  values = array.astype(np.float32)
  finite = np.isfinite(values)
  return (
      f'rms {float(np.sqrt(np.mean(np.square(values[finite])))):.4g},'
      f' absmax {float(np.abs(values[finite]).max()):.4g},'
      f' nonfinite {int((~finite).sum())}'
  )


def verify(
    plan: Plan,
    materialize: Callable[[TargetLeaf], np.ndarray],
    out_dir: str,
    step: int,
    samples: int,
    seed: int = 0,
    leaves: Sequence[TargetLeaf] | None = None,
) -> list[str]:
  """Re-reads sampled leaves and compares them bytewise with the source.

  Args:
    plan: the plan that was written. Its structure is what tells Orbax which
      leaves to skip, so it must cover the whole checkpoint even when only a few
      leaves are being checked.
    materialize: rebuilds a leaf from the shards.
    out_dir: the checkpoint directory.
    step: the step under it.
    samples: how many leaves to check; <= 0 checks all of them.
    seed: sampling seed.
    leaves: the leaves to sample from; the whole plan by default.

  Returns:
    One string per problem; empty means everything matched.
  """
  candidates = list(plan.leaves if leaves is None else leaves)
  if 0 < samples < len(candidates):
    # Always include the largest leaf: the expert stacks are the part with a
    # layout to get wrong.
    biggest = max(candidates, key=lambda leaf: leaf.nbytes)
    candidates = [biggest] + random.Random(seed).sample(
        [leaf for leaf in candidates if leaf is not biggest], samples - 1
    )
  problems = []
  for leaf in candidates:
    want = materialize(leaf)
    got = restore_leaves(out_dir, step, plan, [leaf])[leaf.path]
    if got.shape != want.shape:
      problems.append(f'{leaf.name}: shape {got.shape} != {want.shape}')
    elif got.dtype != want.dtype:
      problems.append(f'{leaf.name}: dtype {got.dtype} != {want.dtype}')
    elif not np.array_equal(
        got.view(np.uint8), np.ascontiguousarray(want).view(np.uint8)
    ):
      problems.append(f'{leaf.name}: bytes differ')
    else:
      logging.info(
          'verified %s %s %s: %s',
          leaf.name,
          leaf.shape,
          leaf.dtype.name,
          leaf_stats(leaf, got),
      )
    del want, got
  return problems


# --- reporting ---------------------------------------------------------------


def format_duration(seconds: float) -> str:
  seconds = int(max(seconds, 0))
  return f'{seconds // 3600}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s'


def _category(leaf: TargetLeaf) -> str:
  """Buckets a leaf for the dry-run totals."""
  path = leaf.path
  if _is_routed_expert(path):
    return 'moe routed experts'
  if len(path) > 2 and path[2] == 'ffn':
    return 'moe dense parts' if len(path) > 3 else 'ffn'
  if len(path) > 2 and path[2] == 'token_mixer':
    return 'token mixer'
  if path[1] == 'embed_linear':
    return 'embeddings'
  return 'norms and attn_res'


def dense_equivalent_bytes(plan: Plan) -> tuple[int, bool]:
  """Returns the size of the same tree with the other expert representation.

  Args:
    plan: the plan to convert.

  Returns:
    The alternative total in bytes, and whether the plan is the packed one. A
    packed byte becomes `MXFP4_CODES_PER_BYTE` bf16 values and needs no scale
    plane; the reverse packs the values and adds one scale byte per group.
  """
  bf16_bytes = np.dtype(ml_dtypes.bfloat16).itemsize
  packed = any(
      leaf.path[-1] == hf_params.MXFP4_PACKED_KEY for leaf in plan.leaves
  )
  total = 0
  for leaf in plan.leaves:
    if leaf.path[-1] == hf_params.MXFP4_SCALE_KEY:
      continue
    if leaf.path[-1] == hf_params.MXFP4_PACKED_KEY:
      total += hf_params.MXFP4_CODES_PER_BYTE * bf16_bytes * leaf.nbytes
    elif _is_routed_expert(leaf.path):
      values = leaf.nbytes // leaf.dtype.itemsize
      total += (
          values // hf_params.MXFP4_CODES_PER_BYTE
          + values // hf_params.MXFP4_GROUP_SIZE
      )
    else:
      total += leaf.nbytes
  return total, packed


def dry_run_report(
    plan: Plan, index: HfIndex, converter: hf_params.KimiK3HfConverter
) -> list[str]:
  """Prints the target tree and checks it against the index.

  Args:
    plan: the plan to report on.
    index: the source index.
    converter: the converter (for the HF key prefix).

  Returns:
    Problems that make the conversion invalid; empty means it is safe to run.
  """
  print(f'TARGET TREE ({len(plan.leaves)} leaves)')
  for leaf in plan.leaves:
    print(
        f'  {leaf.name:<58} {str(leaf.shape):<24} {leaf.dtype.name:<9}'
        f' {leaf.nbytes / _GIB:10.3f} GiB'
    )
  totals = collections.Counter()
  counts = collections.Counter()
  for leaf in plan.leaves:
    totals[_category(leaf)] += leaf.nbytes
    counts[_category(leaf)] += 1
  print('\nTARGET BYTES BY CATEGORY')
  for name, nbytes in totals.most_common():
    print(f'  {name:<24} {counts[name]:>6} leaves {nbytes / _GIB:12.2f} GiB')
  print(
      f'  {"TOTAL":<24} {len(plan.leaves):>6} leaves'
      f' {plan.output_bytes / _GIB:12.2f} GiB'
  )
  alternative, packed = dense_equivalent_bytes(plan)
  print(
      f'  {"same tree, experts in":<24}'
      f' {"bf16" if packed else "mxfp4":>6}        '
      f' {alternative / _GIB:12.2f} GiB'
  )

  consumed = set(plan.sources)
  ignored = {
      name
      for name in index.shard_of
      if name.startswith(
          tuple(converter.prefix + p for p in IGNORED_SOURCE_PREFIXES)
      )
      or name.startswith(IGNORED_SOURCE_PREFIXES)
  }
  missing = sorted(name for name in consumed if name not in index)
  unconsumed = sorted(set(index.shard_of) - consumed - ignored)
  source_bytes = sum(
      index.tensors[name].nbytes for name in consumed if name in index.tensors
  )
  print('\nSOURCE')
  print(
      f'  index                 {len(index.shard_of):>8} tensors,'
      f' {index.total_size / _GIB:.2f} GiB'
  )
  print(
      f'  consumed              {len(consumed):>8} tensors,'
      f' {source_bytes / _GIB:.2f} GiB (of the shards that were read)'
  )
  print(f'  skipped (vision)      {len(ignored):>8} tensors')
  print(f'  unconsumed            {len(unconsumed):>8} tensors')
  for name in unconsumed[:20]:
    print(f'    {name}')
  if len(unconsumed) > 20:
    print(f'    ... and {len(unconsumed) - 20} more')

  problems = []
  if missing:
    problems.append(
        f'{len(missing)} tensors the converter needs are not in the index,'
        f' e.g. {missing[:5]}'
    )
  if unconsumed and len(plan.layers) == converter.config.n_layers:
    problems.append(
        f'{len(unconsumed)} index tensors are neither consumed nor explicitly'
        f' ignored, e.g. {unconsumed[:5]}'
    )
  broken = index.missing_shards + index.truncated_shards
  if broken:
    print(
        f'\n{len(broken)} shards are missing or shorter than their own header'
        f' (still downloading?): {list(broken[:5])}'
    )
  unreadable = plan.unreadable_shards(index)
  if unreadable:
    problems.append(
        f'{len(unreadable)} shards this plan reads are missing or truncated:'
        f' {list(unreadable[:5])}'
    )
  return problems


# --- flags and main ----------------------------------------------------------


# Not `required=True`: that makes the flag mandatory for every binary that
# links this module in, including the test.


def parse_layers(spec: str, n_layers: int) -> tuple[int, ...]:
  """Parses a `0-3,8` style layer subset."""
  if not spec.strip():
    return tuple(range(n_layers))
  layers: set[int] = set()
  for part in spec.split(','):
    part = part.strip()
    if not part:
      continue
    if '-' in part.lstrip('-'):
      low, high = part.split('-', 1)
      layers.update(range(int(low), int(high) + 1))
    else:
      layers.add(int(part))
  out_of_range = sorted(i for i in layers if not 0 <= i < n_layers)
  if out_of_range:
    raise ValueError(f'--layers={spec}: {out_of_range} outside [0, {n_layers})')
  return tuple(sorted(layers))


def leaves_of_layers(plan: Plan, layers: Sequence[int]) -> list[TargetLeaf]:
  """Returns the leaves of `layers`, plus the ones outside any block."""
  blocks = {f'block_{layer}' for layer in layers}
  return [
      leaf
      for leaf in plan.leaves
      if not leaf.path[1].startswith('block_') or leaf.path[1] in blocks
  ]


def load_config(hf_dir: str) -> k3_config_lib.KimiK3ExperimentConfig:
  """Builds the Simply config from the release's `config.json`."""
  config = json.loads((epath.Path(hf_dir) / 'config.json').read_text())
  return k3_config_lib.config_from_hf(config.get('text_config', config))


def build_converter(
    config: k3_config_lib.KimiK3ExperimentConfig,
    index: HfIndex,
    expert_dtype: str = 'mxfp4',
    param_dtype: str = 'source',
    dequantize_experts: bool | None = None,
) -> hf_params.KimiK3HfConverter:
  """Builds the converter, detecting the release's HF key prefix.

  Args:
    config: the model config.
    index: the source index, used to detect the `language_model.` prefix the
      multimodal release nests the text backbone under.
    expert_dtype: `mxfp4` (keep the packed pair), `bfloat16` or `float32`.
    param_dtype: dtype of the non-expert weights, or `source` to keep theirs.
    dequantize_experts: decode MXFP4; `None` follows `expert_dtype`.

  Returns:
    A converter that reads the release's keys and emits the requested dtypes.

  Raises:
    ValueError: if the two expert flags contradict each other.
  """
  dequantize = (
      expert_dtype != 'mxfp4'
      if dequantize_experts is None
      else dequantize_experts
  )
  if dequantize and expert_dtype == 'mxfp4':
    raise ValueError(
        '--dequantize_experts needs --expert_dtype=bfloat16 or float32.'
    )
  return hf_params.KimiK3HfConverter(
      config,
      prefix=hf_params.detect_prefix(index.shard_of),
      dequantize_experts=dequantize,
      dtype=PARAM_DTYPES[param_dtype],
      expert_dtype=PARAM_DTYPES.get(expert_dtype, np.float32),
  )
