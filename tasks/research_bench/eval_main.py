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

r"""Eval-only entry point for the research-bench decode/sampling tasks.

Unlike `main.py` (the TRAINING entry point), this runs the page decode-eval
harness: it loads a FIXED checkpoint and evaluates a registered `Evaluation` on
a registered data source, writing `accuracy`, `avg_generation_time` and `seed`
to `final_result.json`. It does NOT train.

Two tasks use it:

  sampling_lcb -- Qwen3-4B on LiveCodeBench v5 (167 problems), maximise
  `accuracy`::

      python -m tasks.research_bench.eval_main \
          --experiment_config=qwen3_4b --lm_format=QwenV2Chat \
          --evaluation=LcbBaseline \
          --datasource_name=simply_json:livecodebench_v5 \
          --temperature=0.6 --top_p=0.95 --top_k=20 \
          --batch_size=48 --n_repeats=1 --max_seq_len=12000 \
          --mesh_shape=1,1,4 --num_eval_threads=96 --seed=42 \
          --experiment_dir=gs://<bucket>/<experiment_name>/seed_42

  decode_efficiency_vf -- Qwen3-30B-A3B-Thinking-2507 on AIME-2025, minimise
  `avg_generation_time` subject to `accuracy >= 0.75`::

      python -m tasks.research_bench.eval_main \
          --experiment_config=qwen3_30b_a3b_thinking_2507 --lm_format=QwQChat \
          --evaluation=ZeroShotDeepSeekQwenR1CoTBoxed \
          --datasource_name=simply:aime25 \
          --temperature=0.6 --top_p=0.95 --top_k=20 \
          --batch_size=40 --n_repeats=4 --max_seq_len=32000 \
          --mesh_shape=1,1,8 --seed=42 \
          --experiment_dir=gs://<bucket>/<experiment_name>/seed_42

One process per seed, each with its own `--experiment_dir` (the launcher gives
each seed `.../seed_<seed>/`), which is why -- unlike the internal original --
there is no work-unit subdirectory logic here. The other internal-only
piece that is gone is the gVisor/CEO wiring: model-generated code now runs in
`code_exec_lib`'s subprocess sandbox, whose actual isolation is reported at
startup (and is weaker than gVisor -- see EVAL_TASKS_NOTES.md).

Every flag the task prompts quote is defined by the harness itself:
`--experiment_config --lm_format --mesh_shape --batch_size --max_seq_len
--temperature --top_p --top_k` in `simply/serving/common_flags.py`, and
`--evaluation --datasource_name --experiment_dir --n_repeats
--num_eval_threads --seed` in `simply/eval/page_decode_eval.py`.
"""

from typing import Sequence

from absl import app
from simply.eval import page_decode_eval
from simply.utils import experiment_helper
# Imported for registration side-effects: the LiveCodeBench data source
# (`simply_json:livecodebench_v5`) and the sampling evals (`LcbBaseline` and
# any subclass you add next to it). AIME-2025 + ZeroShotDeepSeekQwenR1CoTBoxed
# for decode_efficiency_vf are already registered by core simply.
from tasks.research_bench import code_exec_lib
from tasks.research_bench import lcb_data_lib  # pylint: disable=unused-import
from tasks.research_bench import lcb_sampling_lib  # pylint: disable=unused-import


def main(argv: Sequence[str]) -> None:
  # Printed, not logged: it must be the first greppable line of the work-unit
  # log, before absl logging is configured, so a run that silently fell back to
  # the unisolated runner is obvious from the top of the log.
  print(code_exec_lib.sandbox_report(), flush=True)
  # Everything else -- checkpoint load, batcher, decode loop, timing,
  # final_result.json -- is the UNCHANGED measurement harness.
  page_decode_eval.main(argv)


if __name__ == '__main__':
  experiment_helper.set_env_based_flags()
  app.run(main)
