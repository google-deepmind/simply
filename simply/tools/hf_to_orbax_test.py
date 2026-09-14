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
from collections.abc import Mapping

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import ml_dtypes
import numpy as np
import orbax.checkpoint as ocp
import safetensors.numpy
from simply.tools import hf_to_orbax
from simply.utils import checkpoint_lib as ckpt_lib

# The binary under test has required flags; the tests call its functions
# directly, so give the flags values to parse with.
flags.FLAGS.set_default('input_path', 'unused')
flags.FLAGS.set_default('format', 'Qwen2Format')


def write_safetensors(
    path: epath.Path, tensors: Mapping[str, np.ndarray]
) -> epath.Path:
  safetensors.numpy.save_file(dict(tensors), path.as_posix())
  return path


def hf_state(num_layers: int, dim: int) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(0)
  state = {
      'model.embed_tokens.weight': (
          rng.normal(size=(8, dim)).astype(ml_dtypes.bfloat16)
      )
  }
  for layer in range(num_layers):
    for proj in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
      state[f'model.layers.{layer}.self_attn.{proj}.weight'] = rng.normal(
          size=(dim, dim)
      ).astype(np.float32)
    state[f'model.layers.{layer}.input_layernorm.weight'] = np.ones(
        dim, np.float32
    )
  return state


def restore(directory: epath.Path, step: int = 1):
  """Restores a step the way `checkpoint_lib` resolves checkpoints."""
  path = epath.Path(directory) / str(step)
  handler = ckpt_lib.resolve_checkpoint_handler_from_path(path.as_posix())
  with ocp.Checkpointer(handler) as checkpointer:
    return checkpointer.restore(path)


class TensorRefTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('float32', np.float32),
      ('float16', np.float16),
      ('bfloat16', ml_dtypes.bfloat16),
      ('float8_e4m3fn', ml_dtypes.float8_e4m3fn),
      ('int8', np.int8),
      ('int64', np.int64),
      ('bool', np.bool_),
  )
  def test_read_matches_written_tensor(self, dtype):
    tensors = {
        'scalar': np.asarray(1).astype(dtype),
        'vector': np.arange(5).astype(dtype),
        'matrix': np.arange(6).reshape(2, 3).astype(dtype),
    }
    path = write_safetensors(
        epath.Path(self.create_tempdir().full_path) / 'model.safetensors',
        tensors,
    )
    refs = hf_to_orbax.read_header(path)

    self.assertCountEqual(tensors, refs)
    for key, expected in tensors.items():
      with self.subTest(key):
        self.assertEqual(refs[key].shape, expected.shape)
        self.assertEqual(refs[key].dtype, expected.dtype)
        self.assertEqual(refs[key].nbytes, expected.nbytes)
        np.testing.assert_array_equal(refs[key].read(), expected)

  def test_read_matches_safetensors_reader(self):
    tensors = hf_state(num_layers=1, dim=4)
    path = write_safetensors(
        epath.Path(self.create_tempdir().full_path) / 'model.safetensors',
        tensors,
    )
    refs = hf_to_orbax.read_header(path)

    with safetensors.safe_open(path.as_posix(), framework='np') as f:
      for key in f.keys():
        np.testing.assert_array_equal(refs[key].read(), f.get_tensor(key))

  def test_index_checkpoint_merges_shards(self):
    directory = epath.Path(self.create_tempdir().full_path)
    tensors = hf_state(num_layers=2, dim=4)
    shards = [dict(list(tensors.items())[:3]), dict(list(tensors.items())[3:])]
    for i, shard in enumerate(shards):
      write_safetensors(
          directory / f'model-0000{i}-of-00002.safetensors', shard
      )

    refs = hf_to_orbax.index_checkpoint(directory)

    self.assertCountEqual(tensors, refs)
    self.assertEqual(
        hf_to_orbax.total_bytes(refs),
        sum(t.nbytes for t in tensors.values()),
    )

  def test_index_checkpoint_rejects_duplicate_keys(self):
    directory = epath.Path(self.create_tempdir().full_path)
    tensors = {'w': np.zeros(4, np.float32)}
    write_safetensors(directory / 'model-00000-of-00002.safetensors', tensors)
    write_safetensors(directory / 'model-00001-of-00002.safetensors', tensors)

    with self.assertRaisesRegex(ValueError, 'Duplicate key w'):
      hf_to_orbax.index_checkpoint(directory)

  def test_index_checkpoint_without_safetensors(self):
    with self.assertRaisesRegex(ValueError, 'No file matching'):
      hf_to_orbax.index_checkpoint(self.create_tempdir().full_path)


class StreamingSaveTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tensors = hf_state(num_layers=2, dim=8)
    self.input_dir = epath.Path(self.create_tempdir().full_path)
    write_safetensors(self.input_dir / 'model.safetensors', self.tensors)
    self.refs = hf_to_orbax.index_checkpoint(self.input_dir)

  def convert(self, max_bytes_in_flight: int, ckpt_format=None):
    directory = epath.Path(self.create_tempdir().full_path) / 'orbax'
    handler = hf_to_orbax.TensorRefHandler(max_bytes_in_flight)
    self.addCleanup(handler.close)
    with hf_to_orbax.checkpoint_manager(directory, handler) as mngr:
      ckpt_lib.save_checkpoint(
          mngr,
          self.refs,  # pyrefly: ignore[bad-argument-type]
          1,
          ckpt_format or ckpt_lib.Qwen2Format(),
      )
    return directory, handler

  def test_matches_in_memory_save(self):
    streamed, _ = self.convert(max_bytes_in_flight=1024)
    eager = epath.Path(self.create_tempdir().full_path) / 'orbax'
    with ocp.CheckpointManager(eager) as mngr:
      ckpt_lib.save_checkpoint(
          mngr,
          self.tensors,  # pyrefly: ignore[bad-argument-type]
          1,
          ckpt_lib.Qwen2Format(),
      )

    streamed_ckpt, eager_ckpt = restore(streamed), restore(eager)

    self.assertEqual(streamed_ckpt.metadata, eager_ckpt.metadata)
    self.assertCountEqual(self.tensors, streamed_ckpt.state)
    for key, expected in self.tensors.items():
      with self.subTest(key):
        np.testing.assert_array_equal(streamed_ckpt.state[key], expected)
        np.testing.assert_array_equal(
            streamed_ckpt.state[key], eager_ckpt.state[key]
        )

  def test_bounds_bytes_in_flight(self):
    limit = max(ref.nbytes for ref in self.refs.values())
    _, handler = self.convert(max_bytes_in_flight=limit)

    self.assertLessEqual(handler.peak_bytes_in_flight, limit)
    self.assertLess(limit, hf_to_orbax.total_bytes(self.refs))

  def test_loads_through_checkpoint_lib(self):
    # HuggingFace keys pass through `LegacyFormat` unchanged, so the restored
    # state can be compared to the safetensors input directly.
    directory, _ = self.convert(
        max_bytes_in_flight=1024, ckpt_format=ckpt_lib.LegacyFormat()
    )
    shapes = {
        key: jax.ShapeDtypeStruct(value.shape, value.dtype)
        for key, value in self.tensors.items()
    }
    abstract_state = ckpt_lib.construct_restore_item(shapes)  # pyrefly: ignore[bad-argument-type]

    restored = ckpt_lib.load_checkpoint_from_dir(
        directory.as_posix(), abstract_state
    )

    for key, expected in self.tensors.items():
      with self.subTest(key):
        np.testing.assert_array_equal(restored[key], expected)


if __name__ == '__main__':
  absltest.main()
