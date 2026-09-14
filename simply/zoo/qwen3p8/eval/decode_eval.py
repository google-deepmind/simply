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
"""`eval/decode_eval.py` with the Qwen3.8 configs registered.

`eval:decode_eval` only links the configs in `simply/config_lib.py`, so
`--experiment_config=qwen3p8_27b` fails there with
`ValueError: Unknown name: qwen3p8_27b`. Run this module instead
(`python -m simply.zoo.qwen3p8.eval.decode_eval`, or point a launcher's
`--binary` at it).

README.md has the AIME-25 command and score this binary was accepted with.
"""

from absl import app
from absl import flags
from absl import logging
from etils import epath
import jax
from simply import config_lib
from simply.eval import decode_eval

# Imported for their registration side effects, as `simply/main.py` does: the
# `qwen3p8_*` experiment configs, the `Qwen38HybridLM` module, the `Qwen3.8`
# vocab, the `Qwen38Chat*` chat formats and the `Qwen38Format` checkpoint
# format. `config_lib` deliberately does not import the model, so the binary is
# where they are pulled together.
from simply.zoo.qwen3p8 import config_lib as qwen3p8_config_lib  # pylint: disable=unused-import
from simply.zoo.qwen3p8 import model_lib as qwen3p8_model_lib  # pylint: disable=unused-import
from simply.zoo.qwen3p8.utils import ckpt_format  # pylint: disable=unused-import
from simply.zoo.qwen3p8.utils import lm_format  # pylint: disable=unused-import
from simply.zoo.qwen3p8.utils import tokenization  # pylint: disable=unused-import


_JIT_CACHE_DIR = flags.DEFINE_string(
    'qwen3p8_jit_cache_dir',
    '',
    "Directory for JAX's persistent compilation cache. Unset disables it. A"
    ' 64-layer program costs minutes to compile and the sampler compiles one"'
    ' shape per decode-buffer size, so a rerun with the cache is ~1 min. Not'
    ' `--extra_flags=jax_compilation_cache_dir=...`: that is a jax *config*'
    ' option, not an absl flag, and the launchers forward absl flags only.',
)


def _enable_jit_cache() -> None:
  """Points JAX's persistent compilation cache at `--qwen3p8_jit_cache_dir`."""
  if not _JIT_CACHE_DIR.value:
    return
  jax.config.update('jax_compilation_cache_dir', _JIT_CACHE_DIR.value)
  # Every Qwen3.8 program is far above this; the floor only keeps trivia out.
  jax.config.update('jax_persistent_cache_min_compile_time_secs', 60)
  logging.info(
      'JAX compilation cache: %s', jax.config.jax_compilation_cache_dir
  )


def _check_mesh() -> None:
  """Refuses a mesh the model cannot use, before the 15-minute startup.

  Three ways to lose, all of which otherwise fail deep inside the first
  compile: a batch the batch axes cannot split; a `model` axis that does not
  divide the 4 key-value heads; and a `model` axis wider than them, which
  shards the attention cache to nothing.

  Raises:
    ValueError: on any of the three.
  """
  mesh_shape = flags.FLAGS.mesh_shape
  config = config_lib.ExperimentConfigRegistry.get_config(
      flags.FLAGS.experiment_config
  )
  if not mesh_shape or len(mesh_shape) != 3:
    raise ValueError(
        '--mesh_shape must have exactly three values (replica, data, model);'
        f' got {mesh_shape}. The default leaves the 27B unsharded.'
    )
  replica, data, model = (int(i) for i in mesh_shape)
  batch_size = flags.FLAGS.batch_size
  if batch_size % (replica * data):
    raise ValueError(
        f'--batch_size={batch_size} is not a multiple of the'
        f' replica*data={replica * data} shards of --mesh_shape={mesh_shape}.'
    )
  if config.n_kv_heads % model:
    raise ValueError(
        f'--mesh_shape={mesh_shape} shards the {config.n_kv_heads} key-value'
        f' heads {model} ways, which does not divide. Move the parallelism to'
        ' `data`.'
    )


def _check_checkpoint() -> None:
  """Refuses to start without weights, rather than at the first restore.

  A 27B restore is reached only after the mesh is up and the prefill program
  has compiled; an empty `--ckpt_dir` against a config that expects weights
  should fail in the first second instead.

  Raises:
    ValueError: if neither `--ckpt_dir` nor the config supplies a checkpoint.
  """
  config = config_lib.ExperimentConfigRegistry.get_config(
      flags.FLAGS.experiment_config
  )
  if not config.init_ckpt_format:  # A random-init config wants no weights.
    return
  ckpt_dir = flags.FLAGS.ckpt_dir or config.init_ckpt_dir
  if ckpt_dir:
    # Existence, not just truthiness: the config carries a default path, so
    # the case this check exists for is a path that is not there.
    if epath.Path(ckpt_dir).exists():
      return
    raise ValueError(
        f'--ckpt_dir={ckpt_dir} does not exist. Convert the release with'
        ' `python -m simply.tools.hf_to_orbax` (README.md, "Getting the'
        ' weights").'
    )
  raise ValueError(
      f'--experiment_config={flags.FLAGS.experiment_config} expects converted'
      ' weights and no --ckpt_dir was given. Convert the release with'
      ' `python -m simply.tools.hf_to_orbax` (README.md, "Getting the'
      ' weights") and pass --ckpt_dir=<yours>.'
  )


def main(argv):
  _check_checkpoint()
  _check_mesh()
  _enable_jit_cache()
  decode_eval.main(argv)


if __name__ == '__main__':
  app.run(main)
