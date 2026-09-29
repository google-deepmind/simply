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

r"""Local (CPU) training-FLOPs ESTIMATE for the research-bench task.

This is a convenience for AGENTS to cheaply estimate a config's training FLOPs
**locally, before launching**, so they can size `num_train_steps` to stay within
the compute budget. It compiles the *real* jitted train step (fwd + bwd +
optimizer) on a **1-device CPU mock mesh** and reads XLA's HLO cost model:

    training_flops_xla_1cpu(config) =
        cost_analysis()['flops'] * mesh_devices * grad_accum_steps * steps

cost_analysis reports PER-PARTITION FLOPs, so core `run_experiment` (see
model_lib.py's `train_flops_per_step`) scales it by the device count of the
mesh it ran on. Here the mesh is deliberately TRIVIAL -- every axis is 1 -- so
one partition holds the whole batch and `mesh_devices` is 1; the factor is kept
explicit so the formula still matches core's if this ever compiles on a real
mesh. It must NOT be `jax.device_count()`: that reports the live backend, which
is unrelated to the mesh used here, and on a backend with 4-8 visible devices it
would inflate the estimate by that factor. The grad-accum factor is real: the
step is a `jax.lax.scan` whose body is counted once, so without it a
`grad_accum_steps=2` config reads 2x too low.

IMPORTANT: this 1-CPU value is only an ESTIMATE. The number that counts against
the budget is `training_flops_xla`, written to `final_result.json` by core
`run_experiment` -- the FLOPs of the actual compiled train step on the pinned
TPU topology (the 2-chip reference slice). The two differ by
a roughly constant factor (CPU vs TPU kernel selection); the 1-CPU estimate is
fine for staying safely under budget, but do not treat it as exact. Both count
what 6ND misses -- attention's quadratic term, multi-pass forwards, MoE
routing, optimizer compute (e.g. Muon Newton-Schulz), and remat.

WARNING: it is a LOCAL, standalone estimation helper: never import it from a
training/eval binary, or that job may run on CPU.
"""

import functools
import logging
import os
import sys

# Force a single CPU device so the FLOPs number is hardware-independent and
# reproducible. Must be set BEFORE jax is imported (jax reads these at import),
# hence the module-level side effect. If jax is ALREADY imported we must not
# touch the environment (it would have no effect on the live backend anyway) and
# we warn loudly, because that means this local helper was pulled into a real
# job -- the estimate will then be computed on whatever backend that job uses.
if 'jax' in sys.modules:
  logging.warning(
      'tpu_mfu imported AFTER jax: cannot pin the FLOPs estimate to 1 CPU '
      'device. The estimate will use the live backend and is NOT comparable to '
      'the documented 1-CPU numbers. tpu_mfu is a local estimation helper and '
      'should not be imported from a training/eval binary.'
  )
else:
  os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=1')
  os.environ.setdefault('JAX_PLATFORMS', 'cpu')

# pylint: disable=g-import-not-at-top
import jax
import numpy as np

from simply import model_lib
from simply.utils import sharding as sharding_lib
# pylint: enable=g-import-not-at-top


def _mesh_device_count(mesh_shape) -> int:
  """Devices in the mesh the estimate compiles on.

  This is the correct scale factor for a PER-PARTITION cost_analysis figure.
  Deliberately NOT `jax.device_count()`, which reports the live backend: the
  two agree only when the process happens to see exactly one device, and when
  they disagree (jax imported before this module, or XLA_FLAGS already set) the
  estimate is silently inflated by the backend's device count.

  Args:
    mesh_shape: axis name -> axis size for the mesh used to compile.

  Returns:
    The number of devices in that mesh, at least 1.
  """
  count = 1
  for size in (mesh_shape or {}).values():
    count *= max(1, int(size))
  return max(1, count)


def _make_dummy_batch(config):
  bs, s = config.batch_size, config.seq_len
  toks = np.tile(
      np.arange(s, dtype=np.int32) % max(config.vocab_size - 1, 1), (bs, 1))
  toks = jax.numpy.asarray(toks)
  return {'decoder_input_tokens': toks, 'decoder_target_tokens': toks}


def flops_per_step_1cpu(config) -> float:
  """GLOBAL XLA cost_analysis FLOPs for one train step, on a 1-CPU mock mesh.

  Mirrors core `run_experiment`'s `train_flops_per_step`: per-partition FLOPs x
  the compiling mesh's device count x grad_accum_steps (the grad-accum scan body
  is counted once). The mock mesh is trivial, so the first factor is 1.

  Args:
    config: the experiment config to estimate.

  Returns:
    Estimated global FLOPs for one optimizer step.
  """
  axis_names = config.sharding_config.mesh_axis_names
  trivial = {name: 1 for name in axis_names} if axis_names else {}
  sharding_lib.set_mesh(
      mesh_shape=trivial, dcn_mesh_shape=None, axis_names=axis_names)

  model, _ = model_lib.create_model(config, config.sharding_config)
  opt = config.optimizer
  state = opt.init(model.init(jax.random.key(config.model_seed)))

  @functools.partial(jax.jit, static_argnames=['add_log_info'])
  def train_one_step_fn(state, batch, lr, add_log_info=False):
    return model_lib.train_one_step(
        state=state, batch=batch, lr=lr, model=model, opt=opt,
        grad_accum_steps=config.grad_accum_steps,
        clip_grad_norm=config.clip_grad_norm,
        clip_update_norm=config.clip_update_norm,
        clip_local_update_rms=config.clip_local_update_rms,
        weight_decay=config.weight_decay,
        add_log_info=add_log_info)

  batch = _make_dummy_batch(config)
  lr = jax.numpy.asarray(0.01, dtype=jax.numpy.float32)
  compiled = train_one_step_fn.lower(state, batch, lr).compile()
  ca = compiled.cost_analysis()
  if isinstance(ca, (list, tuple)):
    ca = ca[0]
  # Same two corrections core applies (model_lib.train_flops_per_step):
  # cost_analysis is PER-PARTITION -- scaled by the devices in the mesh we
  # compiled on, which is the trivial 1-device mock mesh built above, NOT the
  # live backend's device count -- and a grad-accum step is a scan whose body is
  # counted once. Omitting the grad-accum factor made this estimate read
  # `grad_accum_steps` times too low.
  grad_accum = max(1, int(config.grad_accum_steps))
  return (float(ca.get('flops', float('nan')))  # pyrefly: ignore[missing-attribute]
          * _mesh_device_count(trivial) * grad_accum)


def training_flops_xla_1cpu(config) -> float:
  """Total canonical training FLOPs = per-step XLA FLOPs * steps executed.

  Core's train loop runs `num_train_steps + 1` optimizer steps (its condition is
  `while steps <= config.num_train_steps`) and reports
  `training_flops_xla = per_step * final_state_steps`, so the estimate uses the
  same step count -- slightly conservative rather than slightly under budget.

  Args:
    config: the experiment config to estimate.

  Returns:
    Estimated total training FLOPs for the whole run.
  """
  return flops_per_step_1cpu(config) * (int(config.num_train_steps) + 1)


def within_budget(config, baseline_flops: float, budget_multiple: float = 1.5):
  """Returns (ok, flops, cap) for the fixed-compute task constraint."""
  flops = training_flops_xla_1cpu(config)
  cap = baseline_flops * budget_multiple
  return (flops <= cap, flops, cap)
