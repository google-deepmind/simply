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

Example:
python -m simply.tools.hf_to_orbax  \
    --input_path=${HF_DIR}/Qwen3-0.6B/ \
    --output_path=${HF_DIR}/Qwen3-0.6B/ORBAX/ \
    --format=Qwen2Format
"""

import asyncio
from collections.abc import Mapping, Sequence

from absl import app
from absl import flags
from absl import logging
from etils import epath
import jax
import orbax.checkpoint as ocp
import safetensors

from simply.utils import checkpoint_lib as ckpt_lib


_INPUT_PATH = flags.DEFINE_string(
    'input_path', None, 'HuggingFace repo path.', required=True
)

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', None, 'Orbax output checkpoint path.', required=True
)

_FORMAT = flags.DEFINE_enum(
    'format', 'Qwen2Format', ['Qwen2Format'], 'Checkpoint format.'
)


def load_tensors_from_path(path: epath.PathLike) -> Mapping[str, jax.Array]:
  """Adds tensors from a path to the output dict."""
  output = {}
  with safetensors.safe_open(path, framework='jax') as f:
    for key in f.keys():
      tensor = f.get_tensor(key)
      logging.info(
          'Loaded %s, where dtype is %s, shape is %s',
          key,
          tensor.dtype,
          tensor.shape,
      )
      output[key] = tensor
    logging.info('Loaded %s', path)
  return output


async def load_hf_checkpoint(path: str):
  """Loads a HuggingFace checkpoint as a PyTree."""
  path = epath.Path(path)
  tensor_paths = path.glob('model*.safetensors')
  if not tensor_paths:
    raise ValueError(f'No safetensors found in {path}')
  output_futures = []
  for tensor_path in tensor_paths:
    output_futures.append(
        asyncio.to_thread(load_tensors_from_path, tensor_path)
    )
  output = {}
  for output_future in asyncio.as_completed(output_futures):
    suboutput = await output_future
    for key, tensor in suboutput.items():
      if key in output:
        raise ValueError(f'Duplicate key {key} found in {path}')
      output[key] = tensor
  return output


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  assert _INPUT_PATH.value != _OUTPUT_PATH.value
  output_path = epath.Path(_OUTPUT_PATH.value)
  if output_path.exists():
    output_path.rmtree()

  state = asyncio.run(load_hf_checkpoint(_INPUT_PATH.value))

  logging.info('Loading finished, start saving...')

  with ocp.CheckpointManager(_OUTPUT_PATH.value) as mngr:
    ckpt_lib.save_checkpoint(
        mngr, state, 1, ckpt_lib.CheckpointFormatRegistry.get(_FORMAT.value)
    )


if __name__ == '__main__':
  app.run(main)
