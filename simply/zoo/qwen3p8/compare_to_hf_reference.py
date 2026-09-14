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
r"""Real-weight parity check: Simply Qwen3.8 vs a golden HF-torch reference.

Restores the Orbax checkpoint through the standard Simply path
(`checkpoint_lib.load_checkpoint_from_dir` with `config.init_ckpt_format`),
runs a forward pass on the reference `input_ids`, and reports logits / hidden
state agreement with the `.npz` written by `dump_hf_reference.py`.

A 27B forward on CPU needs ~60 GB RAM and several minutes; run it with nohup.

Flags and `main` only; `utils/parity.py` is the machinery, and `utils:
parity_test` is what covers it.

Example:
python -m simply.zoo.qwen3p8.compare_to_hf_reference \
    --ref_path=/tmp/ref_full_bf16.npz --experiment_config=qwen3p8_27b
"""

from collections.abc import Sequence
import functools

from absl import app
from absl import flags

# Imported for their registration side effects, as `eval/decode_eval.py` does:
# `parity.build_model_and_params` looks the experiment config up by name.
from simply.zoo.qwen3p8 import config_lib  # pylint: disable=unused-import
from simply.zoo.qwen3p8 import model_lib  # pylint: disable=unused-import
from simply.zoo.qwen3p8.utils import ckpt_format  # pylint: disable=unused-import
from simply.zoo.qwen3p8.utils import lm_format  # pylint: disable=unused-import
from simply.zoo.qwen3p8.utils import parity
from simply.zoo.qwen3p8.utils import tokenization  # pylint: disable=unused-import

# The defaults live in `parity`, which is where they are documented, measured
# and tested; these flags only expose them.
_DEFAULTS = parity.Options(ref_path='')
_BOUNDS = _DEFAULTS.thresholds

_REF_PATH = flags.DEFINE_string(
    'ref_path', None, 'Golden .npz from dump_hf_reference.', required=True
)
_EXPERIMENT_CONFIG = flags.DEFINE_string(
    'experiment_config',
    _DEFAULTS.experiment_config,
    'Registered experiment config.',
)
_CKPT_DIR = flags.DEFINE_string(
    'ckpt_dir', None, 'Overrides the config init_ckpt_dir.'
)
_CKPT_STEP = flags.DEFINE_integer('ckpt_step', -1, 'Checkpoint step.')
_ACTIVATION_DTYPE = flags.DEFINE_enum(
    'activation_dtype',
    _DEFAULTS.activation_dtype,
    ['float32', 'bfloat16'],
    'Param/act dtype.',
)
_MESH_SHAPE = flags.DEFINE_list('mesh_shape', None, 'Mesh shape.')
_PER_LAYER = flags.DEFINE_bool(
    'per_layer',
    _DEFAULTS.per_layer,
    'Also run the blocks one by one for per-layer diffs.',
)
_GREEDY_STEPS = flags.DEFINE_integer(
    'greedy_steps',
    _DEFAULTS.greedy_steps,
    'Greedily continue for this many tokens (cache-free re-prefill) and'
    ' compare with the reference continuation (over the overlap).',
)
_DECODE_TEXT = flags.DEFINE_bool(
    'decode_text',
    _DEFAULTS.decode_text,
    'Detokenize the prompt and the greedy continuation with the config vocab.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', None, 'Optional JSON report path.'
)
_OUTPUT_LOGITS_SOFT_CAP = flags.DEFINE_float(
    'output_logits_soft_cap',
    None,
    'Overrides config.output_logits_soft_cap. HF Qwen3.5/3.8 have no logit'
    ' soft-cap, so -1 is the HF-faithful setting.',
)

# Pass/fail thresholds; `parity.Thresholds` records where the defaults come
# from and what a longer prompt costs.
_MAX_ABS_DIFF = flags.DEFINE_float(
    'max_abs_diff',
    _BOUNDS.max_abs_diff,
    'Fail if the max abs logit diff exceeds this.',
)
_MAX_KL = flags.DEFINE_float(
    'max_kl',
    _BOUNDS.max_kl,
    'Fail if mean KL(HF || Simply) exceeds this.',
)
_MIN_TOP1 = flags.DEFINE_float(
    'min_top1',
    _BOUNDS.min_top1,
    'Fail if top-1 agreement falls below this.',
)
_MIN_HIDDEN_COS = flags.DEFINE_float(
    'min_hidden_cos',
    _BOUNDS.min_hidden_cos,
    'Fail if any per-layer hidden-state cosine falls below this'
    ' (--per_layer only).',
)
_REQUIRE_GREEDY_MATCH = flags.DEFINE_bool(
    'require_greedy_match',
    _BOUNDS.require_greedy_match,
    'Fail if the greedy continuation differs from the reference'
    ' (--greedy_steps only).',
)


def main(argv: Sequence[str]) -> None:
  del argv
  mesh_shape = (
      [int(i) for i in _MESH_SHAPE.value] if _MESH_SHAPE.value else None
  )
  options = parity.Options(
      ref_path=_REF_PATH.value,
      experiment_config=_EXPERIMENT_CONFIG.value,
      activation_dtype=_ACTIVATION_DTYPE.value,
      per_layer=_PER_LAYER.value,
      greedy_steps=_GREEDY_STEPS.value,
      decode_text=_DECODE_TEXT.value,
      output_path=_OUTPUT_PATH.value,
      thresholds=parity.Thresholds(
          max_abs_diff=_MAX_ABS_DIFF.value,
          max_kl=_MAX_KL.value,
          min_top1=_MIN_TOP1.value,
          min_hidden_cos=_MIN_HIDDEN_COS.value,
          require_greedy_match=_REQUIRE_GREEDY_MATCH.value,
      ),
  )
  exit_code = parity.run(
      options,
      functools.partial(
          parity.build_model_and_params,
          _EXPERIMENT_CONFIG.value,
          _CKPT_DIR.value,
          _CKPT_STEP.value,
          _ACTIVATION_DTYPE.value,
          mesh_shape,
          _OUTPUT_LOGITS_SOFT_CAP.value,
      ),
  )
  if exit_code:
    raise SystemExit(exit_code)


if __name__ == '__main__':
  app.run(main)
