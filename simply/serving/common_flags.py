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
"""Common flags shared between serving related binaries."""

from absl import flags
import numpy as np

EXPERIMENT_CONFIG = flags.DEFINE_string(
    'experiment_config',
    None,
    'Experiment config that contains the model config to use.',
    required=True,
)

MESH_SHAPE = flags.DEFINE_list(
    'mesh_shape', None, 'Mesh shape to use. If none, use default mesh shape.'
)

CKPT_DIR = flags.DEFINE_string(
    'ckpt_dir', None, 'Path to the checkpoints directory.'
)

CKPT_STEP = flags.DEFINE_integer(
    'ckpt_step',
    -1,
    'Checkpoint step to use. By default, use the latest checkpoint step.',
)

CKPT_FORMAT = flags.DEFINE_string(
    'ckpt_format', None, 'Checkpoint format to use. (Optional)'
)

VOCAB_NAME = flags.DEFINE_string(
    'vocab_name',
    None,
    'Name of the vocab. If not provided, use the vocab name in the experiment'
    ' config.',
)

BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 1, 'Batch size to use for decoding.'
)

ACTIVATION_DTYPE = flags.DEFINE_enum(
    'activation_dtype',
    'bfloat16',
    ['float32', 'bfloat16'],
    'Dtype of the activation.',
)

MAX_SEQ_LEN = flags.DEFINE_integer(
    'max_seq_len', 65537, 'Max sequence length for the model.'
)

MAX_DECODE_STEPS = flags.DEFINE_integer(
    'max_decode_steps',
    np.iinfo(np.int32).max // 2,
    'Max decode steps for the model.',
)

PAGE_SIZE = flags.DEFINE_integer(
    'page_size', 128, 'Page size to use for decoding.'
)

TEMPERATURE = flags.DEFINE_float(
    'temperature', 1.0, 'Temperature for sampling.'
)

TOP_K = flags.DEFINE_integer('top_k', -1, 'Top-k for sampling.')

TOP_P = flags.DEFINE_float('top_p', 1.0, 'Top-p for sampling.')

INTERMEDIATE_STEPS = flags.DEFINE_integer(
    'intermediate_steps', 1024, 'Intermediate steps for decoding.'
)

MAX_NUM_ISSUE_TOKENS = flags.DEFINE_integer(
    'max_num_issue_tokens',
    0,
    'Maximum number of newly-issued tokens dispatched per prefill pass.',
)

RESPONSE_ASAP = flags.DEFINE_boolean(
    'response_asap',
    False,
    'Whether to return the response as soon as any sequence in the batch'
    ' finishes decoding.',
)

LM_FORMAT = flags.DEFINE_string(
    'lm_format',
    None,
    'LM format to use. If not provided, use the default LM format in the'
    ' experiment config.',
)

ENABLE_PREFIX_CACHING = flags.DEFINE_boolean(
    'enable_prefix_caching',
    False,
    'When True, enables the page batcher to reuse paged KV cache across'
    ' requests that share an input prefix. The cache is held in host (CPU)'
    ' RAM, per-process. False (default) disables the prefix cache entirely.'
    ' See simply.serving.prefix_cache for details.',
)

FFN_WEIGHT_QUANT = flags.DEFINE_string(
    'ffn_weight_quant',
    None,
    'FFN (MoE expert) weight-only quantization for decode, fused spec'
    " '<dtype>[:<block_size>]' (e.g. 'int8', 'int4:128'; '' disables). dtype:"
    " 'int8' (W8A16) or 'int4' (W4A16); optional block_size is the K-block"
    ' group size (omitted = per-channel).',
)

KV_CACHE_QUANT = flags.DEFINE_string(
    'kv_cache_quant',
    None,
    'Low-precision (fp8) paged KV cache for decode. Spec: <dtype>[:<scale>] or'
    " <dtype>:<k_scale>,<v_scale>. dtype: 'fp8'/'fp8_e4m3' -> float8_e4m3fn,"
    " 'fp8_e5m2' -> float8_e5m2; '' disables.",
)


def build_launch_args() -> dict[str, object]:
  """Returns `{flag_name: value}` for every flag defined in this module."""
  args: dict[str, object] = {}
  for flag in flags.FLAGS.get_flags_for_module(__name__):
    value = flag.value
    if isinstance(value, list):
      value = ','.join(value) if value is not None else None
    args[flag.name] = value
  return args
