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
"""`eval/decode_eval.py` with the Kimi K3 configs registered.

`eval:decode_eval` only links the configs in `simply/config_lib.py`, so
`--experiment_config=kimi_k3_*` fails there with
`ValueError: Unknown name: kimi_k3_decode`. Run this module instead
(`python -m simply.zoo.kimi_k3.eval.decode_eval`, or point a launcher's
`--binary` at it).
"""

from absl import app
from absl import flags
from absl import logging
import jax
from simply import config_lib
from simply.eval import decode_eval

# Imported for their registration side effects, as `simply/main.py` does: the
# `kimi_k3_*` experiment configs, the `KimiK3LM` module, the `KimiK3` vocab,
# `KimiK3Chat` + `KimiK3InputProcessor`, the `KimiK3Format` checkpoint format
# and the `KimiK3GPQADiamond` evaluation. `config_lib` deliberately does not
# import the model, so the binary is where they are pulled together.
from simply.zoo.kimi_k3 import config_lib as k3_config_lib  # pylint: disable=unused-import
from simply.zoo.kimi_k3 import model_lib as k3_model_lib  # pylint: disable=unused-import
from simply.zoo.kimi_k3.utils import ckpt_format  # pylint: disable=unused-import
from simply.zoo.kimi_k3.utils import evaluation  # pylint: disable=unused-import
from simply.zoo.kimi_k3.utils import lm_format  # pylint: disable=unused-import
from simply.zoo.kimi_k3.utils import tokenization  # pylint: disable=unused-import

_JIT_CACHE_DIR = flags.DEFINE_string(
    'k3_jit_cache_dir',
    '',
    "Directory for JAX's persistent compilation cache. Unset disables it."
    ' One program shape costs 10-45 min unrolled (less with a `*_scan`'
    ' config); the cache turns a rerun into ~1 min. Not'
    ' `--extra_flags=jax_compilation_cache_dir=...`:'
    ' that is a jax *config* option, not an absl flag, and the launchers'
    ' forward absl flags only.',
)


def _enable_jit_cache() -> None:
  """Points JAX's persistent compilation cache at `--k3_jit_cache_dir`."""
  if not _JIT_CACHE_DIR.value:
    return
  jax.config.update('jax_compilation_cache_dir', _JIT_CACHE_DIR.value)
  # Every K3 program is far above this; the floor only keeps trivia out.
  jax.config.update('jax_persistent_cache_min_compile_time_secs', 60)
  logging.info(
      'JAX compilation cache: %s', jax.config.jax_compilation_cache_dir
  )


def _check_batch_size() -> None:
  """Refuses a batch the data axes cannot split, before the 15-minute startup.

  K3's mesh is `(replica, data, seq, model)` and the batch axis is
  `('replica', 'data')`. An indivisible batch does not fail at the mesh, it
  fails deep inside the first `prefill_fn` call, *after* the checkpoint restore
  and the prefill compile -- 15 minutes and an alloc later -- with
  `IndivisibleError: shape=[4, 1, 64] ... dim_size=4 is not divisible by
  axis_size=8`.

  Raises:
    ValueError: if `--batch_size` is not a multiple of `replica * data`.
  """
  mesh_shape = flags.FLAGS.mesh_shape
  if not mesh_shape or len(mesh_shape) != 4:
    return
  batch_shards = int(mesh_shape[0]) * int(mesh_shape[1])
  batch_size = flags.FLAGS.batch_size
  if batch_size % batch_shards:
    raise ValueError(
        f'--batch_size={batch_size} is not a multiple of the'
        f' replica*data={batch_shards} shards of --mesh_shape={mesh_shape}.'
        ' The batch axis is sharded over both, and the sampler pads the batch'
        ' to --batch_size, so pick a multiple (the padding rows cost nothing'
        ' but a slot).'
    )


def _check_checkpoint() -> None:
  """Refuses to start without weights, rather than at the first restore.

  The K3 configs carry `init_ckpt_format` but no `init_ckpt_dir`: a 1.45 TiB
  conversion lives wherever its owner has quota, so the path is the caller's.
  Without this check the run reaches `load_checkpoint_from_dir('')` only after
  the mesh is up and the prefill program has compiled.

  Raises:
    ValueError: if neither `--ckpt_dir` nor the config supplies a checkpoint.
  """
  config = config_lib.ExperimentConfigRegistry.get_config(
      flags.FLAGS.experiment_config
  )
  if flags.FLAGS.ckpt_dir or config.init_ckpt_dir:
    return
  if not config.init_ckpt_format:  # A random-init config wants no weights.
    return
  raise ValueError(
      f'--experiment_config={flags.FLAGS.experiment_config} expects converted'
      ' weights and no --ckpt_dir was given. Produce them with'
      ' `python -m simply.zoo.kimi_k3.convert_hf_checkpoint'
      ' --hf_dir=<release> --out_dir=<yours> --step=0`, then pass'
      ' --ckpt_dir=<yours>.'
  )


def main(argv):
  _check_checkpoint()
  _check_batch_size()
  _enable_jit_cache()
  decode_eval.main(argv)


if __name__ == '__main__':
  app.run(main)
