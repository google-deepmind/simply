"""Utilities for pipelined MoE implementations."""

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import

import contextlib
import dataclasses
from functools import partial
import itertools
from typing import Any, Callable, Literal, Protocol, NamedTuple, Sequence

import jax
from jax.experimental import xla_metadata
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp


SENTINEL_VALUE = 2 ** 31 - 1
SPARSECORE_PAD_SIZE = 1024
AsyncCopyDescriptor = Any


class RaggedAllToAllCallable(Protocol):
  def __call__(self, operand: jax.Array, output: jax.Array,
               input_offsets: jax.Array, send_sizes: jax.Array,
               output_offsets: jax.Array, recv_sizes: jax.Array,
               *, axis_name: str) -> jax.Array:
    ...


class ComputeBlockCallable(Protocol):
  def __call__(self, tokens: jax.Array, local_group_sizes: jax.Array,
               *extra_args) -> jax.Array:
    ...


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class PipelinedMoEConfig:
  """A performance hyperparameters config for the MoE routines."""
  gathers: Literal["builtin", "custom", "custom_sc"] = "custom"
  safety_factor: float = 1.25
  ra2a: RaggedAllToAllCallable | None = None
  pad_buffers_to_multiple: int | None = None
  use_scheduling_groups: bool = False
  use_pipelined_ra2a_barriers: bool = False
  ep_method: Literal["ra2a", "ag"] = "ra2a"
  fine_grained_ra2a: bool = False
  dropless_fallback: bool = True

  def __post_init__(self):
    if self.pad_buffers_to_multiple is None:
      object.__setattr__(self, "pad_buffers_to_multiple", SPARSECORE_PAD_SIZE)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class _RA2AMeta:
  """Holds sizes and offsets for ragged all-to-all communication."""
  input_offsets: jax.Array
  send_sizes: jax.Array
  output_offsets: jax.Array
  recv_sizes: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MoEInfo:
  batch_size: int = dataclasses.field(metadata=dict(static=True))
  experts_per_tok: int = dataclasses.field(metadata=dict(static=True))
  num_experts: int = dataclasses.field(metadata=dict(static=True))
  buffer_size: int = dataclasses.field(metadata=dict(static=True))


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class LocalPermuteMetadata:
  group_idx: jax.Array
  group_counts: jax.Array
  sort_idx: jax.Array
  isort_idx: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MoEMetaRA2A:
  info: MoEInfo
  local_ra2a_sort: jax.Array
  local_ra2a_isort: jax.Array
  preamble: _RA2AMeta
  epilogue: _RA2AMeta
  local_permute: LocalPermuteMetadata
  buffer_overflow: jax.Array
  scales: jax.Array | None


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MoEMetaAG:
  info: MoEInfo
  gather_idx: jax.Array  # Indices into the global (all-gathered) token buffer
  group_counts: jax.Array  # Counts for the compute block (local experts)
  output_mask: jax.Array  # Mask for valid computed tokens (handling padding)
  scales: jax.Array | None


# private data structures


@dataclasses.dataclass
class _MoEMethods:
  compute_meta: (
      Callable[[jax.Array, jax.Array], MoEMetaRA2A]
      | Callable[[jax.Array, jax.Array], MoEMetaAG]
  )
  load_fn: Callable  # pylint: disable=g-bare-generic
  compute_fn: (
      Callable[[jax.Array, MoEMetaRA2A, Any], jax.Array]
      | Callable[[jax.Array, MoEMetaAG, Any], jax.Array]
  )
  unload_fn: Callable  # pylint: disable=g-bare-generic


@dataclasses.dataclass
class _RDMACopy:
  copy: AsyncCopyDescriptor | None
  start: Callable[[], None]
  wait: Callable[[], None]


################################################################################
# utilities ####################################################################
################################################################################


def maybe_tpu_sublane_size():
  try:
    return pltpu.get_tpu_info().num_sublanes
  except ValueError:  # we don't have TPU
    return 1


def maybe_pad_size(size: int, pad_to_multiple: int | None):
  if pad_to_multiple is None:
    return size
  return ((size + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple


def compute_local_permute_metadata(
    group_idx: jax.Array, groups: int, group_counts: jax.Array | None = None
):
  """Compute metadata for sorting tokens according to group_idx."""

  if group_counts is None:
    group_counts = jnp.bincount(group_idx, length=groups)

  sort_idx = jnp.argsort(group_idx)
  isort_idx = jnp.argsort(sort_idx)[:group_idx.shape[0]]

  return LocalPermuteMetadata(group_idx, group_counts, sort_idx, isort_idx)


@partial(jax.custom_vjp, nondiff_argnames=("empty_scatter",))
def unique_gather(x: jax.Array, idx: jax.Array,
                  inv_idx: jax.Array | None = None,
                  empty_scatter: bool = True):
  del empty_scatter, inv_idx
  return x[idx, ...]


def unique_gather_fwd(x: jax.Array, idx: jax.Array, inv_idx: jax.Array,
                      empty_scatter: bool):
  return (unique_gather(x, idx, inv_idx, empty_scatter=empty_scatter),
          (x.shape, idx, inv_idx))


def unique_gather_bwd(empty_scatter: bool,
                      res: tuple[Any, jax.Array, jax.Array], g: jax.Array):
  (x_shape, idx, inv_idx) = res
  if g.shape[0] == x_shape[0]:
    inv_idx = inv_idx if inv_idx is not None else jnp.argsort(idx)
    grad = g[inv_idx, ...]
  elif g.shape[0] != x_shape[0] and inv_idx is not None:  # padded gather
    raise NotImplementedError("Gather with padding not yet implemented")
  else:  # scatter
    _init = jax.lax.empty if empty_scatter else jnp.zeros
    grad = _init(x_shape, dtype=g.dtype).at[idx, ...].set(g, mode="drop")
  return (grad, None, None)


unique_gather.defvjp(unique_gather_fwd, unique_gather_bwd)


################################################################################
# ragged_all_to_all utilities ##################################################
################################################################################


def _ra2a_3d_kernel_async(
    src_ref, out_ref, input_offsets, send_sizes, output_offsets, recv_sizes,
    dst_ref, sems,  # outputs
    *,
    axis_name: str, start: bool = True, multiple: int = 1,  # config
):
  del out_ref  # aliased in dst_ref
  idx, n_devices = jax.lax.axis_index(axis_name), jax.lax.axis_size(axis_name)
  # TODO: maybe support multiple > 1 to allow direct 2D ra2a
  assert multiple == 1, "multiple > 1 not supported"
  mult_of = lambda x: multiple * (x // multiple)

  def make_dma():
    sem_id = jax.lax.rem(idx + idx, n_devices)
    src_slc = pl.ds(mult_of(input_offsets[idx]), mult_of(send_sizes[idx]))
    dst_slc = pl.ds(mult_of(output_offsets[idx]), mult_of(send_sizes[idx]))
    src, dst = src_ref.at[src_slc, ...], dst_ref.at[dst_slc, ...]
    copy = pltpu.make_async_copy(src, dst, sems.at[sem_id, 0, 1])
    return _RDMACopy(copy, copy.start, copy.wait)

  def make_rdma(other_id, send: bool = True):
    src_id, dst_id = (idx, other_id) if send else (other_id, idx)
    size = jax.lax.select(idx == src_id, send_sizes[dst_id], recv_sizes[src_id])
    src_slc = pl.ds(mult_of(input_offsets[dst_id]), mult_of(size))
    dst_slc = pl.ds(mult_of(output_offsets[dst_id]), mult_of(size))
    src, dst = src_ref.at[src_slc, ...], dst_ref.at[dst_slc, ...]
    sem_id = jax.lax.rem(idx + other_id, n_devices)
    direction_id = (src_id > dst_id).astype(jnp.int32)
    send_sem = sems.at[sem_id, direction_id, 0]
    recv_sem = sems.at[sem_id, direction_id, 1]
    copy = pltpu.make_async_remote_copy(
        src, dst, send_sem, recv_sem, device_id={axis_name: dst_id}
    )
    start_fn = copy.start if send else (lambda: None)
    wait_fn = copy.wait_send if send else copy.wait_recv
    return _RDMACopy(copy, start_fn, wait_fn)

  barrier_sem = pltpu.get_barrier_semaphore()
  for i in range(1, n_devices):
    neigh_id = jax.lax.rem(idx + i, n_devices)
    pl.semaphore_signal(barrier_sem, inc=1, device_id={axis_name: neigh_id})
  pl.semaphore_wait(barrier_sem, n_devices - 1)

  dma_copy = make_dma()

  send_rdmas, recv_rdmas = [], []
  for i in range(1, n_devices):
    other_id = jax.lax.rem(idx + i, n_devices)
    send_rdmas.append(make_rdma(other_id, send=True))
    recv_rdmas.append(make_rdma(other_id, send=False))

  if start:
    _ = [rdma.start() for rdma in (send_rdmas + recv_rdmas)]
    dma_copy.start()
  else:
    _ = [rdma.wait() for rdma in (send_rdmas + recv_rdmas)]
    dma_copy.wait()

  @partial(pl.run_scoped, second_barrier=pltpu.SemaphoreType.REGULAR)
  def _final_barrier(second_barrier):
    for i in range(1, n_devices):
      neigh_id = jax.lax.rem(idx + i, n_devices)
      pl.semaphore_signal(
          second_barrier, inc=1, device_id={axis_name: neigh_id}
      )
    pl.semaphore_wait(second_barrier, n_devices - 1)


_RA2A_COLLECTIVE_ID = 23  # needs to be globally (per-program) unique
# 23 is an arbitrary guess value, `make_ra2a_3d` function is work-in-progress


def make_ra2a_3d(axis_name: str = "x", multiple: int = 1):
  global _RA2A_COLLECTIVE_ID
  _RA2A_COLLECTIVE_ID += 1
  def start(
      src, output, input_offsets, send_sizes, output_offsets, recv_sizes,
      collective_id=_RA2A_COLLECTIVE_ID
  ):
    n_devices = jax.lax.axis_size(axis_name)

    def ra2a_kernel_start(
        src_ref, out_ref, input_offsets, send_sizes, output_offsets, recv_sizes,
        dst_ref, sems
    ):
      return _ra2a_3d_kernel_async(
          src_ref, out_ref,
          input_offsets, send_sizes, output_offsets, recv_sizes,
          dst_ref, sems,
          axis_name=axis_name, multiple=multiple, start=True
      )

    sems_spec = pltpu.SemaphoreType.DMA((n_devices, 2, 2))
    out, sems = pl.pallas_call(
        ra2a_kernel_start,
        out_shape=[output, sems_spec],
        in_specs=(
            2 * [pl.BlockSpec(memory_space=pl.ANY)]        # src, out
            + 4 * [pl.BlockSpec(memory_space=pltpu.SMEM)]  # ra2a meta
        ),
        out_specs=[
            pl.BlockSpec(memory_space=pl.ANY),           # out
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),  # sems
        ],
        input_output_aliases={1: 0},
        interpret=False,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
    )(src, output, input_offsets, send_sizes, output_offsets, recv_sizes)
    future = (
        src, out, sems, input_offsets, send_sizes, output_offsets, recv_sizes
    )
    return future

  def wait(future, collective_id=0):
    (
        src, output, sems, input_offsets, send_sizes, output_offsets, recv_sizes
    ) = future

    def ra2a_kernel_wait(src_ref, out_ref,
                         input_offsets, send_sizes, output_offsets, recv_sizes,
                         sems, dst_ref):
      return _ra2a_3d_kernel_async(
          src_ref, out_ref,
          input_offsets, send_sizes, output_offsets, recv_sizes,
          dst_ref, sems, axis_name=axis_name, multiple=multiple, start=False
      )

    out = pl.pallas_call(
        ra2a_kernel_wait,
        out_shape=output,
        in_specs=(
            2 * [pl.BlockSpec(memory_space=pl.ANY)]         # src, out
            + 4 * [pl.BlockSpec(memory_space=pltpu.SMEM)]   # ra2a metadata
            + [pl.BlockSpec(memory_space=pltpu.SEMAPHORE)]  # sems
        ),
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        input_output_aliases={1: 0},
        interpret=False,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
    )(src, output, input_offsets, send_sizes, output_offsets, recv_sizes, sems)
    return out

  return start, wait


def surround_compute_with_ag(
    axis_name: str, compute_fn, use_barriers: bool = False,
    *,
    collective_methods: list[Callable],  # pylint: disable=g-bare-generic
):
  def fn(payloads, args):
    if use_barriers:
      args, payloads = jax.lax.optimization_barrier((args, payloads))
    outs = [collective_method(payload, axis_name=axis_name)
            if payload is not None else None
            for payload, collective_method
            in zip(payloads, collective_methods, strict=True)]
    y = compute_fn(*args) if compute_fn is not None else None
    return outs, y

  return fn


def surround_compute_with_ra2a(
    axis_name: str, compute_fn, multiple: int = 1,
    ra2a: RaggedAllToAllCallable | None = None, use_barriers: bool = False
):
  if ra2a is not None:
    def fn(payloads, args):
      if use_barriers:
        args, payloads = jax.lax.optimization_barrier((args, payloads))
      outs = []
      for payload in payloads:
        if payload is None:
          outs.append(None)  # does not participate in the collective
        elif all(x is None for x in payload[1:]):
          outs.append(payload[0])  # short-circuit communication
        else:
          outs.append(ra2a(*payload, axis_name=axis_name))  # communicate
      y = compute_fn(*args) if compute_fn is not None else None
      return outs, y

    return fn

  split_start_fn, split_wait_fn = make_ra2a_3d(
      axis_name=axis_name, multiple=multiple
  )

  # define utility functions for starting and waiting on individual ra2a futures
  def start_fn(src, output,
               input_offsets, send_sizes, output_offsets, recv_sizes,
               *, collective_id):
    if output is None:  # short-circuit ra2a
      return src
    return split_start_fn(
        src, output, input_offsets, send_sizes, output_offsets, recv_sizes,
        collective_id=collective_id
    )

  def wait_fn(future, collective_id):
    if isinstance(future, jax.Array):  # not a future, the result already
      return future
    return split_wait_fn(future, collective_id=collective_id)

  # define ra2a_split function with a custom backward pass
  def _ra2a_split(payloads, args):
    futures = []
    for i, payload in enumerate(payloads):
      future = None
      if payload is not None:
        (src, output,
         input_offsets, send_sizes, output_offsets, recv_sizes) = payload
        with jax.named_scope(f"ra2a_split_{i}_start"):
          future = start_fn(src, output, input_offsets, send_sizes,
                            output_offsets, recv_sizes, collective_id=i)
      futures.append(future)

    # use the futures barrier only if we're using the custom split ra2a kernel
    if use_barriers:
      args, futures = jax.lax.optimization_barrier((args, futures))
    if compute_fn is not None:
      (y, vjp_fn) = jax.vjp(compute_fn, *args)
    else:
      (y, vjp_fn) = (None, None)
    if use_barriers:
      y, futures = jax.lax.optimization_barrier((y, futures))

    outs = []
    for i, future in enumerate(futures):
      out = None
      with jax.named_scope(f"ra2a_split_{i}_wait"):
        if future is not None:
          out = wait_fn(future, collective_id=len(payloads) + i)
      outs.append(out)

    return (outs, y), vjp_fn

  @jax.custom_vjp
  def ra2a_split(payloads, args):
    return _ra2a_split(payloads, args)[0]

  def ra2a_split_fwd(payloads, args):
    ret, vjp_fn = _ra2a_split(payloads, args)
    res = ([[payload[0].shape] + list(payload[2:])
            if payload is not None else None for payload in payloads],
           args, vjp_fn)
    return ret, res

  def ra2a_split_bwd(res, g):
    payloads, args, vjp_fn = res
    tangents = g[0]
    dout = g[1]

    futures = []
    for i, (tangent, payload) in enumerate(zip(tangents, payloads,
                                               strict=True)):
      if payload is not None:
        (
            src_shape, input_offsets, send_sizes, output_offsets, recv_sizes
        ) = payload
        if input_offsets is not None:
          inv_send_sizes, inv_recv_sizes = recv_sizes, send_sizes
          inv_input_offsets = jax.lax.all_to_all(
              output_offsets, axis_name=axis_name, split_axis=0, concat_axis=0)
          inv_output_offsets = jax.lax.all_to_all(
              input_offsets, axis_name=axis_name, split_axis=0, concat_axis=0)
          buf = jax.lax.empty(src_shape, dtype=tangent.dtype)
        else:  # short-circuit ra2a
          buf = None
          inv_input_offsets, inv_send_sizes = None, None
          inv_output_offsets, inv_recv_sizes = None, None
        future = start_fn(
            tangent, buf, inv_input_offsets, inv_send_sizes, inv_output_offsets,
            inv_recv_sizes, collective_id=i
        )
      else:
        future = None
      futures.append(future)

    if use_barriers:
      dout, futures = jax.lax.optimization_barrier((dout, futures))
    dcompute = jax.tree.map(lambda _: None, args)
    if compute_fn is not None:
      dcompute = vjp_fn(dout)
    if use_barriers:
      dcompute, futures = jax.lax.optimization_barrier((dcompute, futures))

    dins = []
    for i, future in enumerate(futures):
      if future is not None:
        din = wait_fn(future, collective_id=i)
        dins.append(tuple([din] + [None] * 5))
      else:
        dins.append(None)

    return tuple(dins), dcompute

  ra2a_split.defvjp(ra2a_split_fwd, ra2a_split_bwd)
  return ra2a_split

################################################################################
# pipelined MoE ################################################################
################################################################################


def _compute_chunked_ra2a(
    all_sizes: jax.Array, shard_idxs: jax.Array,
    *, num_shards: int, shard_idx: int | jax.Array,
    axis_name: str, num_experts: int, config: PipelinedMoEConfig):
  experts_per_shard = num_experts // num_shards
  all_shard_sizes = jnp.sum(
      all_sizes.reshape((num_shards, num_shards, -1)), axis=-1)
  actual_local_token_num = shard_idxs.shape[-1]

  # compute the ra2a communication #########################################

  # batch * expert_per_token * safety factor
  safety_factor = min(config.safety_factor, num_shards)
  buffer_size = round(actual_local_token_num * safety_factor)
  buffer_size = min(buffer_size, shard_idxs.size * num_shards)
  buffer_size = maybe_pad_size(buffer_size, config.pad_buffers_to_multiple)

  # cumsums from 0 are `jnp.cumsum(x) - x`
  all_input_offsets = jnp.cumsum(all_shard_sizes, axis=-1) - all_shard_sizes
  all_output_offsets = jnp.cumsum(all_shard_sizes, axis=0) - all_shard_sizes
  send_sizes = all_shard_sizes[shard_idx, :]
  recv_sizes = all_shard_sizes[:, shard_idx]
  input_offsets = all_input_offsets[shard_idx, :]
  output_offsets = all_output_offsets[shard_idx, :]
  preamble = _RA2AMeta(input_offsets, send_sizes, output_offsets, recv_sizes)
  # we send back chunks starting where we received them
  inv_input_offsets = all_output_offsets[:, shard_idx]
  inv_output_offsets = all_input_offsets[:, shard_idx]
  inv_send_sizes, inv_recv_sizes = recv_sizes, send_sizes
  epilogue = _RA2AMeta(inv_input_offsets, inv_send_sizes,
                       inv_output_offsets, inv_recv_sizes)
  # compute per expert shard sort; receiving several locally sorted chunks

  # get gather indices for pre-ra2a by-expert-shard organization
  local_ra2a_sort = jnp.argsort(shard_idxs)
  local_ra2a_isort = jnp.argsort(local_ra2a_sort)[:actual_local_token_num]

  all_local_expert_idxs = jax.lax.all_gather(
      shard_idxs[local_ra2a_sort], axis=0, tiled=False, axis_name=axis_name)

  # compute local expert idxs after the transfer (technically a ra2a, but
  # more efficient via AG and dynamic slice)
  def _update_fn(i, local_expert_idxs):
    update = jnp.roll(
        all_local_expert_idxs[i, :], -all_input_offsets[i, shard_idx]
    )
    return jax.lax.dynamic_update_slice_in_dim(
        local_expert_idxs, update, all_output_offsets[i, shard_idx], 0
    )

  local_expert_idxs = jax.lax.empty(
      (buffer_size + all_local_expert_idxs.shape[-1],), jnp.int32)
  local_expert_idxs = jax.lax.pcast(
      local_expert_idxs,
      tuple(jax.typeof(all_local_expert_idxs).manual_axis_type.varying),
      to="varying"
  )
  local_expert_idxs = jax.lax.fori_loop(0, num_shards, _update_fn,
                                        local_expert_idxs)[:buffer_size]
  local_pack_mask = jnp.arange(local_expert_idxs.size) < jnp.sum(
      preamble.recv_sizes)
  local_expert_idxs = jnp.where(local_pack_mask,
                                local_expert_idxs, SENTINEL_VALUE)

  # compute the local permutation
  local_expert_idxs_ = jnp.where(
      local_pack_mask,
      local_expert_idxs - shard_idx * experts_per_shard,
      SENTINEL_VALUE
  )
  local_group_counts = jnp.sum(all_sizes.reshape(
      (num_shards, num_shards, experts_per_shard))[:, shard_idx, :], 0)
  local_permute = compute_local_permute_metadata(
      local_expert_idxs_, experts_per_shard, group_counts=local_group_counts
  )
  return (
      preamble, epilogue, local_ra2a_sort, local_ra2a_isort, local_permute,
      buffer_size
  )


def ra2a_with_gather(all_sizes: jax.Array, num_shards: int,
                     shard_idx: jax.Array):
  all_offsets = jnp.cumsum(all_sizes, axis=-1) - all_sizes  # [S, E]
  send_sizes = all_sizes.reshape((num_shards, num_shards, -1))  # [S, S, EpS]
  send_sizes = send_sizes.swapaxes(1, 2)  # [S, EpS, S]
  input_offsets = all_offsets.reshape((num_shards, num_shards, -1)
                                      ).swapaxes(1, 2)  # [S, EpS, S]

  recv_sizes = send_sizes.swapaxes(0, 2)  # [S, EpS, S]
  _recv_flat = recv_sizes.reshape((num_shards, -1))
  write_offset_receiver = (jnp.cumsum(_recv_flat, axis=-1) - _recv_flat
                           ).reshape((num_shards, -1, num_shards))
  output_offsets = write_offset_receiver.swapaxes(0, 2)

  preamble = _RA2AMeta(*jax.tree.map(
      lambda x: x[shard_idx, ...].T.reshape(-1),
      (input_offsets, send_sizes, output_offsets, recv_sizes)))
  inv_send_sizes, inv_recv_sizes = recv_sizes, send_sizes
  inv_input_offsets = output_offsets.swapaxes(0, 2)
  inv_output_offsets = input_offsets.swapaxes(0, 2)
  epilogue = _RA2AMeta(*jax.tree.map(
      lambda x: x[shard_idx, ...].T.reshape(-1),
      (inv_input_offsets, inv_send_sizes, inv_output_offsets, inv_recv_sizes)))
  return preamble, epilogue


def _create_pipelined_ra2a_moe(
    compute_block: ComputeBlockCallable | None = None,
    *, axis_name: str, experts_per_tok: int, num_experts: int,
    config: PipelinedMoEConfig = PipelinedMoEConfig(),
) -> _MoEMethods:
  assert config.gathers in ("builtin", "custom")
  if config.fine_grained_ra2a and config.ra2a is not jax.lax.ragged_all_to_all:
    raise NotImplementedError("Fine grained RA2A only supported with the XLA's "
                              "jax.lax.ragged_all_to_all.")

  # metadata computation #######################################################
  shard_idx = jax.lax.axis_index(axis_name)
  num_shards = jax.lax.axis_size(axis_name)

  def compute_meta(shard_idxs: jax.Array, scales: jax.Array | None = None):
    # experts_per_shard = num_experts // num_shards
    if shard_idxs.ndim != 1:
      raise ValueError(f"{jax.typeof(shard_idxs)=} should be 1D.")

    with jax.named_scope("compute_metadata"):
      actual_local_token_num = shard_idxs.shape[-1]
      local_sizes = jnp.bincount(shard_idxs, length=num_experts)
      all_sizes = jax.lax.all_gather(local_sizes, axis=0, tiled=False,
                                     axis_name=axis_name)
      opts = dict(num_shards=num_shards, shard_idx=shard_idx,
                  axis_name=axis_name, num_experts=num_experts, config=config)
      (preamble, epilogue, local_ra2a_sort, local_ra2a_isort, local_permute,
       buffer_size) = _compute_chunked_ra2a(all_sizes, shard_idxs, **opts)

      if config.fine_grained_ra2a:
        preamble, epilogue = ra2a_with_gather(all_sizes, num_shards, shard_idx)

      info = MoEInfo(batch_size=actual_local_token_num // experts_per_tok,
                     experts_per_tok=experts_per_tok, num_experts=num_experts,
                     buffer_size=buffer_size)
      necessary_recv_buffer_size = jnp.sum(preamble.recv_sizes)[None]
      necessary_recv_buffer_sizes = jax.lax.all_gather(
          necessary_recv_buffer_size, axis=0, tiled=True, axis_name=axis_name)
      buffer_overflow = jnp.any(necessary_recv_buffer_sizes > buffer_size)
      return MoEMetaRA2A(info, local_ra2a_sort, local_ra2a_isort, preamble,
                         epilogue, local_permute, buffer_overflow, scales)

  # compute ####################################################################

  def load_fn(metrics: dict[str, Any] | None = None):
    def prepare_fn(x, meta: MoEMetaRA2A):
      # step 1: gather local tokens for every expert per token
      with jax.named_scope("tokens_to_experts_gather"):
        if config.gathers == "custom":
          x_sort = jnp.repeat(x, experts_per_tok, axis=0)
          x_sort = unique_gather(x_sort, meta.local_ra2a_sort,
                                 inv_idx=meta.local_ra2a_isort)
        else:
          x_sort = x[meta.local_ra2a_sort // experts_per_tok, ...]

      if jax.lax.axis_size(axis_name) == 1:  # short-circuit ra2a
        return (x_sort, None,
                *[None for _ in dataclasses.astuple(meta.preamble)])

      # step 2: communicate expert-gathered-tokens to their corresponding shards
      with jax.named_scope("ra2a_tokens"):
        # TODO: check that:
        # (total_recv_size / x.shape[0] * experts_per_tok) < safety_factor
        buffer = jax.lax.empty(
            (meta.info.buffer_size,) + x.shape[1:], dtype=x.dtype)
        if metrics is not None:
          total_recv_size = jnp.sum(meta.preamble.recv_sizes)
          metrics.setdefault("total_recv_size", []).append(total_recv_size)
          metrics.setdefault("load_factor", []).append(
              total_recv_size / (x.shape[0] * meta.info.experts_per_tok))
          metrics.setdefault("exceeds_buffer", []).append(
              total_recv_size > meta.info.buffer_size)

        # to be communicated as:
        # ra2a(y, out, *dataclasses.astuple(meta.epilogue), axis_name=axis_name)
        return (x_sort, buffer, *dataclasses.astuple(meta.preamble))

    def finalize_fn(y, meta: MoEMetaRA2A):
      # y comes from:
      # ra2a(y, out, *dataclasses.astuple(meta.prologue), axis_name=axis_name)
      if config.fine_grained_ra2a:
        return y

      if jax.lax.axis_size(axis_name) == 1:  # short-circuit ra2a
        return y

      # step 3: gather tokens locally so they're expert-contiguous
      with jax.named_scope("local_gather_before"):
        if config.gathers == "custom":
          y = unique_gather(y, meta.local_permute.sort_idx,
                            inv_idx=meta.local_permute.isort_idx)
        else:
          y = y[meta.local_permute.sort_idx, ...]
      return y

    return prepare_fn, finalize_fn

  def compute_fn(y: jax.Array, meta: MoEMetaRA2A, *args):
    # step 4: perform gmm computation
    local_group_counts = meta.local_permute.group_counts
    with jax.named_scope("compute"):
      if compute_block is not None:
        y = compute_block(y, local_group_counts, *args)
    return y

  def unload_fn():
    def prepare_fn(y, meta: MoEMetaRA2A):
      if jax.lax.axis_size(axis_name) == 1:  # short-circuit ra2a
        return (y, None, *[None for _ in dataclasses.astuple(meta.preamble)])

      # step 5: unpermute tokens locally to organize them into chunks in which
      # they arrived
      with jax.named_scope("local_gather_after"):
        if not config.fine_grained_ra2a:
          if config.gathers == "custom":
            y = unique_gather(y, meta.local_permute.isort_idx)
          else:
            y = y[meta.local_permute.isort_idx, ...]

      # step 6: communincate the chunks back to their origins
      with jax.named_scope("ra2a_results"):
        # an important performance optmization consideration
        # we use empty to avoid blocking the send, but need to make sure
        # uninitialized values don't poison either the forward or backward pass
        # so we immediately mask the communication result in finalize_fn
        # the alternative is to use zeros here, but at a cost of performance
        out = jax.lax.empty(  # TODO: if unexpected NaNs, switch to zeros
            (meta.info.batch_size * meta.info.experts_per_tok, *y.shape[1:]),
            dtype=y.dtype
        )

      # to be communicated as:
      # ra2a(y, out, *dataclasses.astuple(meta.epilogue), axis_name=axis_name)
      return (y, out, *dataclasses.astuple(meta.epilogue))

    def finalize_fn(x_sort, meta: MoEMetaRA2A):
      # x_sort comes from:
      # ra2a(y, out, *dataclasses.astuple(meta.epilogue), axis_name=axis_name)

      # early masking of potentially uninitialized garbage
      num_active = jnp.sum(meta.preamble.send_sizes)
      mask = jnp.arange(x_sort.shape[0]) < num_active
      mask = jnp.expand_dims(mask, axis=range(1, x_sort.ndim))
      x_sort = jnp.where(mask, x_sort, 0.0)

      # step 7: gather so each token repeats are next to each other
      with jax.named_scope("expert_to_tokens_gather"):
        if config.gathers == "custom":
          y = unique_gather(x_sort, meta.local_ra2a_isort,
                            inv_idx=meta.local_ra2a_sort)
        else:
          y = x_sort[meta.local_ra2a_isort, ...]

      # step 8: weigh by expert weights
      with jax.named_scope("reduction_across_experts"):
        if meta.scales is not None:
          y *= jnp.expand_dims(meta.scales, axis=range(1, y.ndim))
        y = y.reshape(
            (y.shape[0] // experts_per_tok, experts_per_tok) + y.shape[1:]
        )
        y = jnp.sum(y, axis=1)
      return y
    return prepare_fn, finalize_fn

  return _MoEMethods(compute_meta, load_fn, compute_fn, unload_fn)


def _ag_tokens(x, *, axis_name: str, axis: int, tiled: bool):
  with jax.named_scope("ag_tokens"):
    return jax.lax.all_gather(x, axis_name=axis_name, axis=axis, tiled=tiled)


def _rs_tokens(x, *, axis_name: str, scatter_dimension: int, tiled: bool):
  with jax.named_scope("reduce_scatter_tokens"):
    return jax.lax.psum_scatter(x, axis_name=axis_name, tiled=tiled,
                                scatter_dimension=scatter_dimension)

AG_PIPELINE_COLLECTIVES = [
    partial(_ag_tokens, axis=0, tiled=True),
    partial(_rs_tokens, scatter_dimension=0, tiled=True)
]


def _create_pipelined_ag_moe(
    compute_block: ComputeBlockCallable | None = None,
    *,
    axis_name: str, experts_per_tok: int, num_experts: int,
    config: PipelinedMoEConfig = PipelinedMoEConfig(),
) -> _MoEMethods:
  assert config.gathers in ("builtin", "custom", "custom_sc")
  shard_idx = jax.lax.axis_index(axis_name)
  num_shards = jax.lax.axis_size(axis_name)

  def compute_meta(shard_idxs: jax.Array, scales: jax.Array | None = None):
    experts_per_shard = num_experts // num_shards

    # Calculate batch size derived from indices
    local_batch_size = shard_idxs.size // experts_per_tok
    if shard_idxs.ndim != 1:
      raise ValueError(f"{jax.typeof(shard_idxs)=} should be 1D")
    with jax.named_scope("ag_scales"):
      all_gather = partial(jax.lax.all_gather, axis_name=axis_name, tiled=True)
      all_idxs = all_gather(shard_idxs, axis=0)
      all_scales = all_gather(scales, axis=0)

    with jax.named_scope("compute_metadata"):
      if all_idxs.ndim != 1:
        all_idxs = all_idxs.reshape(-1)
      valid_mask = ((all_idxs >= shard_idx * experts_per_shard)
                    & (all_idxs < (shard_idx + 1) * experts_per_shard))
      valid_idxs = jnp.where(valid_mask, all_idxs, SENTINEL_VALUE)
      all_sizes = jnp.bincount(all_idxs, length=num_experts)
      all_expert_counts = all_sizes.reshape((num_shards, experts_per_shard))
      local_expert_counts = all_expert_counts[shard_idx, :]

      safety_factor = min(config.safety_factor, num_shards)
      buffer_size = round(local_batch_size * experts_per_tok * safety_factor)
      buffer_size = maybe_pad_size(buffer_size, config.pad_buffers_to_multiple)
      buffer_size = min(buffer_size, shard_idxs.size * num_shards)

      local_sort = jnp.argsort(valid_idxs)[:buffer_size]
      total_valid_tokens = jnp.sum(valid_mask)
      output_mask = jnp.arange(buffer_size) < total_valid_tokens
      info = MoEInfo(local_batch_size, experts_per_tok, num_experts,
                     buffer_size)
      return MoEMetaAG(info=info, gather_idx=local_sort,
                       group_counts=local_expert_counts,
                       output_mask=output_mask, scales=all_scales)

  # compute

  def load_fn(metrics: dict[str, Any] | None = None):
    def prepare_fn(x, meta: MoEMetaAG):
      total_recv_size = jnp.sum(meta.group_counts)
      if metrics is not None:
        metrics.setdefault("total_recv_size", []).append(total_recv_size)
        metrics.setdefault("load_factor", []).append(
            total_recv_size / (x.shape[0] * experts_per_tok))
        metrics.setdefault("exceeds_buffer", []).append(
            total_recv_size > meta.info.buffer_size)
      return x

    def finalize_fn(all_x, meta: MoEMetaAG):
      # all_x comes from:
      # all_gather(x, tiled=True, axis=0, axis_name=axis_name)
      with jax.named_scope("local_gather_before"):
        if config.gathers == "custom":
          x_sort = jnp.repeat(all_x, experts_per_tok, axis=0)
          x_sort = unique_gather(x_sort, meta.gather_idx, empty_scatter=False)
        else:
          x_sort = all_x[meta.gather_idx // experts_per_tok, ...]
      mask = jnp.expand_dims(meta.output_mask, axis=tuple(range(1, all_x.ndim)))
      x_sort = jnp.where(mask, x_sort, 0)
      return x_sort

    return prepare_fn, finalize_fn

  def compute_fn(y: jax.Array, meta: MoEMetaAG, *args):
    with jax.named_scope("compute"):
      if compute_block is not None:
        y = compute_block(y, meta.group_counts, *args)
      if meta.scales is not None:
        all_scales = jax.lax.all_gather(
            meta.scales, axis_name=axis_name, tiled=True)
        y *= jnp.expand_dims(all_scales[meta.gather_idx], axis=range(1, y.ndim))
      mask = jnp.expand_dims(meta.output_mask, tuple(range(1, y.ndim)))
      y = jnp.where(mask, y, 0.0)
    return y

  def unload_fn():
    def prepare_fn(y, meta: MoEMetaAG):
      with jax.named_scope("local_scatter_after"):
        total_global_tokens = meta.info.batch_size * num_shards
        global_buffer_shape = (total_global_tokens,) + y.shape[1:]
        scatter_indices = meta.gather_idx // experts_per_tok
        # scatter-add reduce tokens from AG tokens to local tokens
        out_buffer = jnp.zeros(global_buffer_shape, dtype=y.dtype)
        out_buffer = out_buffer.at[scatter_indices, ...].add(y, mode="drop")
      return out_buffer

    def finalize_fn(y, meta: MoEMetaAG):
      # y comes from:
      # psum_scatter(y, out, tiled=True, scatter_dimension=0,
      #              axis_name=axis_name)
      del meta
      return y

    return prepare_fn, finalize_fn

  return _MoEMethods(compute_meta, load_fn, compute_fn, unload_fn)


_SCHEDULING_GROUP_ID = itertools.count()


def _overlap_fn(
    y1, y2, meta1, meta2, meta3, x_next, *extra_args,
    axis_name: str, moe_methods: _MoEMethods | Any, i: int, splits: int,
    config: PipelinedMoEConfig, metrics: dict[str, Any] | None = None,
):
  """Overlap stages 1 & 2 communication, with stage 2 computation."""
  ctx = contextlib.nullcontext()
  if config.use_scheduling_groups:
    ctx = xla_metadata.set_xla_metadata(
        _scheduling_group_id=next(_SCHEDULING_GROUP_ID)
    )

  with ctx:
    # perform initial non-communication tasks for y1 and y3
    fut1, fut3, finalize_fn1, finalize_fn3 = None, None, None, None
    if 0 <= i < splits:
      # capture communication metrics for the ragged-all-to-all
      prepare_fn1, finalize_fn1 = moe_methods.load_fn(metrics=metrics)
      fut1 = prepare_fn1(x_next, meta1)

    if 2 <= i < splits + 2:
      prepare_fn3, finalize_fn3 = moe_methods.unload_fn()
      fut3 = prepare_fn3(y2, meta3)

    # overlap communication from y1 and y3 with compute on y2
    compute_fn = moe_methods.compute_fn if 1 <= i < splits + 1 else None
    if config.ep_method == "ra2a":
      split_fn = surround_compute_with_ra2a(
          axis_name, compute_fn, ra2a=config.ra2a,
          use_barriers=config.use_pipelined_ra2a_barriers)
    elif config.ep_method == "ag":
      split_fn = surround_compute_with_ag(
          axis_name, compute_fn, collective_methods=AG_PIPELINE_COLLECTIVES,  # pyrefly: ignore[bad-argument-type]
          use_barriers=config.use_pipelined_ra2a_barriers)
    else:
      raise ValueError(f"Unsupported ep_method: `{config.ep_method}`")
    (y1_next, y3_next), y2_next = split_fn(
        (fut1, fut3), (y1, meta2, *extra_args))

    # perform remaining non-communication tasks for y1 and y3
    y1_next = finalize_fn1(y1_next, meta1) if 0 <= i < splits else y1_next  # pyrefly: ignore[not-callable]
    y3_next = finalize_fn3(y3_next, meta3) if 2 <= i < splits + 2 else y3_next  # pyrefly: ignore[not-callable]

  return y1_next, y2_next, y3_next


def run_moe_pipelined_shard_map(
    shard_idxs: jax.Array,
    x: jax.Array,
    *extra_args,
    scales: jax.Array | None = None,
    compute_block: ComputeBlockCallable | None = None,
    axis_name: str, experts_per_tok: int, num_experts: int, splits: int = 1,
    config: PipelinedMoEConfig = PipelinedMoEConfig()
):
  """Execute a piplined MoE layer in full assuming expert axis is manual."""
  assert x.ndim == 3, f"Tokens must be at 3D, but got x = {jax.typeof(x)}"

  pipeline_args = dict(compute_block=compute_block, axis_name=axis_name,
                       experts_per_tok=experts_per_tok, num_experts=num_experts)

  if config.ep_method == "ra2a":
    moe_methods = _create_pipelined_ra2a_moe(**pipeline_args, config=config)
  else:
    moe_methods = _create_pipelined_ag_moe(**pipeline_args, config=config)
  if x.shape[0] % splits != 0:
    raise ValueError(f"{x.shape[0]=} is not divisible by {splits=}")
  if shard_idxs.size % splits != 0:
    raise ValueError(f"{shard_idxs.size=} is not divisible by {splits=}")
  if shard_idxs.shape[0] != x.shape[0] * experts_per_tok:
    raise ValueError(f"{shard_idxs.shape[0]=} is not the same as"
                     f" {experts_per_tok * x.shape[0]=}")

  x_ = x.reshape((splits, x.shape[0] // splits, *x.shape[1:]))
  shard_idxs_ = shard_idxs.reshape((splits, shard_idxs.size // splits))
  if scales is not None:
    scales_ = scales.reshape((splits, -1))
  else:
    scales_ = [None for _ in range(splits)]

  def make_pipeline(config: PipelinedMoEConfig):
    def run_pipeline(x_, scales_, shard_idxs_, ra2a_metas=None):
      if config.ep_method == "ra2a":
        moe_methods = _create_pipelined_ra2a_moe(**pipeline_args, config=config)
      else:
        moe_methods = _create_pipelined_ag_moe(**pipeline_args, config=config)

      if config.ep_method == "ra2a":
        all_metas = ra2a_metas
      else:
        all_metas = [moe_methods.compute_meta(shard_idxs_[i, ...], scales_[i])
                     for i in range(splits)]

      x_next = x_[0, ...]
      y1s, y2s, y3s = [], [], []
      metrics = {}
      overlap_fn_ = partial(
          _overlap_fn, moe_methods=moe_methods, splits=splits,
          axis_name=axis_name, config=config, metrics=metrics
      )

      for i in range(splits + 2):
        overlap_fn_ = partial(overlap_fn_, i=i)
        meta1 = all_metas[i] if 0 <= i < splits else 0  # pyrefly: ignore[unsupported-operation]
        y1, meta2 = None, None
        if 1 <= i < splits + 1:
          y1, meta2 = (y1s[i - 1], all_metas[i - 1])  # pyrefly: ignore[unsupported-operation]
        y2, meta3 = None, None
        if 2 <= i < splits + 2:
          y2, meta3 = (y2s[i - 2], all_metas[i - 2])  # pyrefly: ignore[unsupported-operation]
        y1, y2, y3 = overlap_fn_(y1, y2, meta1, meta2, meta3, x_next,
                                 *extra_args)
        x_next = x_[i + 1, ...] if (i < splits - 1) else None

        # ensure that the pipeline stages don't overlap
        if config.use_pipelined_ra2a_barriers:
          y1, y2, y3, x_next = jax.lax.optimization_barrier(
              (y1, y2, y3, x_next))

        _ = y1s.append(y1) if y1 is not None else None
        _ = y2s.append(y2) if y2 is not None else None
        _ = y3s.append(y3) if y3 is not None else None

      metrics["total_recv_size"] = jnp.sum(jnp.array(
          metrics.get("total_recv_size", [0])))
      metrics["load_factor"] = jnp.max(jnp.array(
          metrics.get("load_factor", [0.0])))
      metrics["exceeds_buffer"] = jnp.any(jnp.array(
          metrics.get("exceeds_buffer", [False])))
      return jnp.concat(y3s, axis=0), metrics

    return run_pipeline

  ra2a_pipeline = make_pipeline(dataclasses.replace(config, ep_method="ra2a"))
  ag_pipeline = make_pipeline(dataclasses.replace(config, ep_method="ag"))

  if config.ep_method == "ag":
    y, metrics_ = ag_pipeline(x_, scales_, shard_idxs_)
  else:
    ra2a_metas = [moe_methods.compute_meta(shard_idxs_[i, ...], scales_[i])  # pyrefly: ignore[bad-argument-type]
                  for i in range(splits)]
    if config.dropless_fallback:
      no_dropping = jnp.all(~jnp.array(
          [meta.buffer_overflow for meta in ra2a_metas]))  # pytype: disable=attribute-error
      y, metrics_ = jax.lax.cond(no_dropping, ra2a_pipeline, ag_pipeline,
                                 x_, scales_, shard_idxs_, ra2a_metas)
    else:  # ra2a without fallback
      y, metrics_ = ra2a_pipeline(x_, scales_, shard_idxs_, ra2a_metas)

  return y, metrics_
