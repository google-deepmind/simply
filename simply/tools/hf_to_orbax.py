# Copyright 2024 The Simply Authors
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
r"""Converts HuggingFace safetensors to orbax.

A safetensors repository is indexed into a PyTree of `TensorRef` leaves, which
carry the shape/dtype metadata Orbax needs to describe a tensor but read the
bytes only on demand. Writes are fed by those lazy reads, so converting a
checkpoint costs a few times `--max_bytes_in_flight` of memory instead of one
full copy of the model -- measured on 8 GiB and 24 GiB checkpoints, resident
memory was ~1 GiB of interpreter overhead plus ~2.3x the limit, and did not
grow with the checkpoint. The limit defaults to what fits in the free memory
of the machine.

Converting a multi-TiB checkpoint is bound by the bandwidth to the
checkpoint's filesystem rather than by this binary, so run large conversions
close to the storage that holds the checkpoint.

Example:
python -m simply.tools.hf_to_orbax \
    --input_path=${HF_DIR}/Qwen3-0.6B/ \
    --output_path=${HF_DIR}/Qwen3-0.6B/ORBAX/ \
    --format=Qwen2Format
"""

import asyncio
import collections
from collections.abc import AsyncIterator, Mapping, Sequence
import concurrent.futures
import contextlib
import dataclasses
import inspect
import json
import math
import struct
import sys
import time
from typing import Any

from absl import app
from absl import flags
from absl import logging
from etils import epath
import ml_dtypes
import numpy as np
import orbax.checkpoint as ocp
from simply.utils import checkpoint_lib as ckpt_lib

DEFAULT_FILE_PATTERN = 'model*.safetensors'

# Resident memory is roughly this multiple of the bytes kept in flight:
# Tensorstore stages its own copy of every tensor it is writing.
_MEMORY_OVERHEAD_FACTOR = 2.5
# Fraction of the free memory the conversion is allowed to occupy.
_MEMORY_FRACTION = 0.75
_MIN_MAX_BYTES_IN_FLIGHT = 1 * 1024**3
_MAX_MAX_BYTES_IN_FLIGHT = 64 * 1024**3

# Reads are latency- rather than CPU-bound, so they run on more threads than
# the machine has cores (and than asyncio's default executor would give).
DEFAULT_READ_THREADS = 256

# Orbax defaults to 20 minutes for the whole background write, which a
# multi-TiB conversion exceeds; this is only a stuck-job safety net.
DEFAULT_TIMEOUT_SECS = 24 * 60 * 60

_LOG_EVERY_SECS = 60
_HEADER_SIZE_BYTES = 8
_HEADER_METADATA_KEY = '__metadata__'
_READ_CHUNK_BYTES = 64 * 1024**2

# https://github.com/huggingface/safetensors: dtype codes stored in the header.
_DTYPES: Mapping[str, np.dtype] = {
    'BOOL': np.dtype(np.bool_),
    'U8': np.dtype(np.uint8),
    'I8': np.dtype(np.int8),
    'F8_E4M3': np.dtype(ml_dtypes.float8_e4m3fn),
    'F8_E5M2': np.dtype(ml_dtypes.float8_e5m2),
    'U16': np.dtype(np.uint16),
    'I16': np.dtype(np.int16),
    'F16': np.dtype(np.float16),
    'BF16': np.dtype(ml_dtypes.bfloat16),
    'U32': np.dtype(np.uint32),
    'I32': np.dtype(np.int32),
    'F32': np.dtype(np.float32),
    'U64': np.dtype(np.uint64),
    'I64': np.dtype(np.int64),
    'F64': np.dtype(np.float64),
}


_INPUT_PATH = flags.DEFINE_string(
    'input_path', None, 'HuggingFace repo path.', required=True
)

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', None, 'Orbax output checkpoint path.'
)

_FORMAT = flags.DEFINE_string(
    'format', None, 'Checkpoint format.', required=True
)

_MAX_BYTES_IN_FLIGHT = flags.DEFINE_integer(
    'max_bytes_in_flight',
    None,
    'Upper bound on the tensor bytes held in memory while converting.'
    ' Defaults to what fits in the free memory of this machine.',
)

_READ_THREADS = flags.DEFINE_integer(
    'read_threads',
    DEFAULT_READ_THREADS,
    'Number of threads reading tensors from the safetensors files.',
)

_TIMEOUT_SECS = flags.DEFINE_integer(
    'timeout_secs',
    DEFAULT_TIMEOUT_SECS,
    'Deadline for the background Orbax write.',
)


@dataclasses.dataclass(frozen=True)
class TensorRef:
  """A tensor that is still on disk, inside a safetensors file.

  Mimics enough of the `np.ndarray` interface (`shape`, `dtype`, `size`,
  `nbytes`) for Orbax to plan the write of a tensor it has not read yet.
  """

  path: epath.Path
  key: str
  shape: tuple[int, ...]
  dtype: np.dtype
  offset: int  # Byte offset of the tensor data from the start of the file.

  @property
  def size(self) -> int:
    return math.prod(self.shape)

  @property
  def nbytes(self) -> int:
    return self.size * self.dtype.itemsize

  def read(self) -> np.ndarray:
    """Reads the tensor into a freshly allocated array."""
    with self.path.open('rb') as f:
      f.seek(self.offset)
      buffer = _read_exact(f, self.nbytes, f'{self.key} in {self.path}')
    return buffer.view(self.dtype).reshape(self.shape)


def _read_exact(f: Any, nbytes: int, what: str) -> np.ndarray:
  """Reads `nbytes` into a uint8 array, chunk by chunk."""
  buffer = np.empty(nbytes, dtype=np.uint8)
  view = memoryview(buffer.data)
  offset = 0
  while offset < nbytes:
    chunk = f.read(min(_READ_CHUNK_BYTES, nbytes - offset))
    if not chunk:
      raise EOFError(f'Truncated read of {what}: {offset} of {nbytes} bytes.')
    view[offset : offset + len(chunk)] = chunk
    offset += len(chunk)
  return buffer


def read_header(path: epath.PathLike) -> Mapping[str, TensorRef]:
  """Reads the tensor index of a single safetensors file."""
  if sys.byteorder != 'little':
    raise NotImplementedError('safetensors data is little-endian.')
  path = epath.Path(path)
  with path.open('rb') as f:
    (header_size,) = struct.unpack('<Q', f.read(_HEADER_SIZE_BYTES))
    header = json.loads(f.read(header_size))
  data_start = _HEADER_SIZE_BYTES + header_size
  refs = {}
  for key, entry in header.items():
    if key == _HEADER_METADATA_KEY:
      continue
    dtype = _DTYPES.get(entry['dtype'])
    if dtype is None:
      raise NotImplementedError(f'Unsupported dtype {entry["dtype"]} for {key}')
    ref = TensorRef(
        path=path,
        key=key,
        shape=tuple(entry['shape']),
        dtype=dtype,
        offset=data_start + entry['data_offsets'][0],
    )
    expected = entry['data_offsets'][1] - entry['data_offsets'][0]
    if ref.nbytes != expected:
      raise ValueError(
          f'{key} in {path} spans {expected} bytes but {ref.shape} of'
          f' {ref.dtype} needs {ref.nbytes}.'
      )
    refs[key] = ref
  return refs


def index_checkpoint(
    path: epath.PathLike, file_pattern: str = DEFAULT_FILE_PATTERN
) -> dict[str, TensorRef]:
  """Indexes a HuggingFace repo directory as a flat dict of `TensorRef`."""
  path = epath.Path(path)
  tensor_paths = sorted(path.glob(file_pattern))
  if not tensor_paths:
    raise ValueError(f'No file matching {file_pattern} found in {path}')
  refs = {}
  for tensor_path in tensor_paths:
    for key, ref in read_header(tensor_path).items():
      if key in refs:
        raise ValueError(
            f'Duplicate key {key} in {tensor_path} and {refs[key].path}'
        )
      refs[key] = ref
  logging.info(
      'Indexed %d tensors (%.2f GiB) from %d files in %s.',
      len(refs),
      total_bytes(refs) / 1024**3,
      len(tensor_paths),
      path,
  )
  return refs


def total_bytes(refs: Mapping[str, TensorRef]) -> int:
  return sum(ref.nbytes for ref in refs.values())


def _free_memory_bytes() -> int | None:
  """Memory that can be allocated without swapping, or None if unknown."""
  try:
    with open('/proc/meminfo') as f:
      for line in f:
        if line.startswith('MemAvailable:'):
          return int(line.split()[1]) * 1024
  except (OSError, IndexError, ValueError):
    pass
  return None


def default_max_bytes_in_flight() -> int:
  """Bytes to keep in flight, sized to the free memory of this machine."""
  free = _free_memory_bytes()
  if free is None:
    return _MIN_MAX_BYTES_IN_FLIGHT
  budget = int(free * _MEMORY_FRACTION / _MEMORY_OVERHEAD_FACTOR)
  return min(max(budget, _MIN_MAX_BYTES_IN_FLIGHT), _MAX_MAX_BYTES_IN_FLIGHT)


class TensorRefHandler(ocp.type_handlers.NumpyHandler):
  """Serializes `TensorRef` leaves, reading each one just in time.

  Orbax builds every write spec from the leaf metadata alone; the only step
  that touches data is the copy into Tensorstore, which is overridden here to
  read the tensor under a byte limiter and release it as soon as the write
  completes. Peak memory is therefore `max_bytes_in_flight` (or the largest
  single tensor, whichever is larger) instead of the size of the checkpoint.

  An instance is single-use: it accumulates `peak_bytes_in_flight` for one
  save and owns the threads that read the tensors, which `close` releases.
  """

  def __init__(
      self,
      max_bytes_in_flight: int | None = None,
      read_threads: int = DEFAULT_READ_THREADS,
      bytes_to_write: int = 0,
  ):
    # Skipping the deep copy of host arrays is what keeps peak memory at
    # `max_bytes_in_flight`; the knob is newer than the latest orbax release.
    if 'deepcopy_host_arrays' in inspect.signature(super().__init__).parameters:
      super().__init__(deepcopy_host_arrays=False)
    else:
      super().__init__()
    max_bytes_in_flight = max_bytes_in_flight or default_max_bytes_in_flight()
    if max_bytes_in_flight <= 0:
      raise ValueError(
          f'max_bytes_in_flight must be positive, got {max_bytes_in_flight}.'
      )
    logging.info(
        'Keeping up to %.1f GiB in flight, read by %d threads.',
        max_bytes_in_flight / 1024**3,
        read_threads,
    )
    self._limit = max_bytes_in_flight
    self._in_flight = 0
    self._waiters: collections.deque[tuple[int, asyncio.Future[None]]] = (
        collections.deque()
    )
    # asyncio's default executor caps reads at 32 threads, which leaves a
    # high-latency filesystem such as CNS far from its achievable throughput.
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=read_threads, thread_name_prefix='tensor_read'
    )
    self._bytes_to_write = bytes_to_write
    self._bytes_written = 0
    self._first_write_time: float | None = None
    self._last_log_time = time.monotonic()
    self.peak_bytes_in_flight = 0

  def close(self) -> None:
    """Releases the reader threads."""
    self._executor.shutdown(wait=False)

  def _fits(self, nbytes: int) -> bool:
    # An oversized tensor waits for exclusive use instead of deadlocking.
    return self._in_flight == 0 or self._in_flight + nbytes <= self._limit

  def _admit(self, nbytes: int) -> None:
    self._in_flight += nbytes
    self.peak_bytes_in_flight = max(self.peak_bytes_in_flight, self._in_flight)

  def _admit_waiters(self) -> None:
    """Admits the tensors that now fit, oldest first."""
    while self._waiters and self._fits(self._waiters[0][0]):
      nbytes, waiter = self._waiters.popleft()
      self._admit(nbytes)
      waiter.set_result(None)

  @contextlib.asynccontextmanager
  async def _reserve(self, nbytes: int) -> AsyncIterator[None]:
    """Reserves `nbytes` of the budget, granted in arrival order.

    Waiters are handed the budget by whoever releases it rather than woken to
    contend for it: Orbax hands every tensor to the handler at once, so all
    but the few in flight are waiting, and waking them all on each release
    costs a wake-up per waiting tensor -- quadratic overall, and the busiest
    thing in the event loop for a checkpoint of half a million tensors.
    """
    if self._fits(nbytes):
      self._admit(nbytes)
    else:
      entry = (nbytes, asyncio.get_running_loop().create_future())
      self._waiters.append(entry)
      try:
        await entry[1]  # Released by `_admit_waiters`, which also admits us.
      except asyncio.CancelledError:
        if entry in self._waiters:
          self._waiters.remove(entry)
          raise
        self._in_flight -= nbytes  # Admitted just before being cancelled.
        self._admit_waiters()
        raise
    try:
      yield
    finally:
      self._in_flight -= nbytes
      self._admit_waiters()

  async def _open_and_write(
      self,
      value: TensorRef | np.ndarray,
      tspec: dict[str, Any],
      ts_context: Any,
  ) -> None:
    nbytes = value.nbytes
    async with self._reserve(nbytes):
      if isinstance(value, TensorRef):
        value = await asyncio.get_running_loop().run_in_executor(
            self._executor, value.read
        )
      await super()._open_and_write(value, tspec, ts_context)
    self._record_write(nbytes)

  def _record_write(self, nbytes: int) -> None:
    """Counts a written tensor, logging progress once every minute."""
    now = time.monotonic()
    if self._first_write_time is None:
      # Timed from here rather than from construction: the rate should not be
      # diluted by the time Orbax spends preparing the write of every tensor.
      self._first_write_time = now
      self._last_log_time = now
    self._bytes_written += nbytes
    if now - self._last_log_time < _LOG_EVERY_SECS:
      return
    self._last_log_time = now
    rate = self._bytes_written / max(now - self._first_write_time, 1e-6)
    remaining = max(self._bytes_to_write - self._bytes_written, 0)
    logging.info(
        'Wrote %.1f GiB of %.1f GiB at %.0f MiB/s, %.0f min left.',
        self._bytes_written / 1024**3,
        self._bytes_to_write / 1024**3,
        rate / 1024**2,
        remaining / rate / 60 if rate > 0 else 0,
    )


def _type_handler_registry(handler: TensorRefHandler):
  """Returns the default registry with `TensorRef` mapped to `handler`."""
  defaults = [
      (ty, ocp.type_handlers.get_type_handler(ty))
      for ty in ocp.type_handlers.supported_types()
  ]
  # Registered last so that it also takes over `np.ndarray`, whose `typestr`
  # it shares: a checkpoint may mix lazy refs and in-memory arrays.
  return ocp.type_handlers.create_type_handler_registry(
      *defaults, (TensorRef, handler)
  )


def checkpoint_manager(
    directory: epath.PathLike,
    handler: TensorRefHandler,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> ocp.CheckpointManager:
  """Returns a manager that writes `TensorRef` leaves through `handler`."""
  handler_registry = ocp.DefaultCheckpointHandlerRegistry()
  handler_registry.add(
      'state',
      ocp.args.PyTreeSave,
      ocp.PyTreeCheckpointHandler(
          type_handler_registry=_type_handler_registry(handler)
      ),
  )
  handler_registry.add(
      'metadata', ocp.args.JsonSave, ocp.JsonCheckpointHandler()
  )
  return ocp.CheckpointManager(
      directory,
      handler_registry=handler_registry,
      options=ocp.CheckpointManagerOptions(
          async_options=ocp.AsyncOptions(timeout_secs=timeout_secs)
      ),
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  input_path = epath.Path(_INPUT_PATH.value)
  if _OUTPUT_PATH.value:
    output_path = epath.Path(_OUTPUT_PATH.value)
  else:
    output_path = input_path / 'ORBAX'
  assert input_path != output_path
  if output_path.exists():
    output_path.rmtree()

  state = index_checkpoint(input_path)
  handler = TensorRefHandler(
      _MAX_BYTES_IN_FLIGHT.value, _READ_THREADS.value, total_bytes(state)
  )
  try:
    with checkpoint_manager(output_path, handler, _TIMEOUT_SECS.value) as mngr:
      ckpt_lib.save_checkpoint(
          mngr,
          state,  # pyrefly: ignore[bad-argument-type]
          1,
          ckpt_lib.CheckpointFormatRegistry.get_instance(_FORMAT.value),
      )
  finally:
    handler.close()
  logging.info(
      'Converted %d tensors, peak %.2f GiB in flight.',
      len(state),
      handler.peak_bytes_in_flight / 1024**3,
  )


if __name__ == '__main__':
  app.run(main)
