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


"""Per-task launch defaults: entry point, accelerator and fixed eval flags.

Everything here is a *default*; every field has a CLI override. The table is
the single place that knows a task_id needs, say, the decode-eval entry point
instead of the training one -- `--task=sampling_lcb` alone must produce a
runnable command line.
"""

import dataclasses


# The submission sweep every task requires (see the task prompts).
DEFAULT_SEEDS = (42, 43, 44)

TRAIN_ENTRY = 'tasks.research_bench.main'
EVAL_ENTRY = 'tasks.research_bench.eval_main'

# `math-eval` (sympy, pylatexenc) is not optional in practice: the boxed-answer
# reward and every maths grader import it. `serving` is needed on top of that by
# the eval entry point, whose `page_decode_eval` imports the serving stubs.
BASE_EXTRAS = 'tpu,gcloud,math-eval'
EVAL_EXTRAS = f'{BASE_EXTRAS},serving'


@dataclasses.dataclass(frozen=True)
class TaskDefaults:
  """How one task_id turns into a remote command line.

  Attributes:
    tpu_type: Accelerator to request (`--tpu-type`).
    entry_module: Module run as `python -m <entry_module>`.
    experiment_config: `--experiment_config` value; None means the task_id
      itself (the benchmark registers one config per task under its own name).
    seed_flag: How the per-seed seed is passed. 'config_overlay' injects
      {"model_seed": s, "dataset_seed": s} into `--config_overlay`; any other
      value is a flag name, e.g. 'seed' -> `--seed=<s>`.
    flags: Flags appended verbatim to the command line, as 'name=value'.
      These are the protocol constants of the task, not tuning knobs.
    config_overlay: Config fields the task's launch command always overrides,
      merged into `--config_overlay` under the seeds.
    runtime_min: Reference wall-clock of one seed on the task's accelerator,
      measured on the internal 4-chip bundle. Sets the default `--timeout-min`.
    seeds: The sweep the submission must contain (validator/task_specs.py).
    apt_packages: System packages the VM needs on top of the base image.
    pip_extras: The repo's optional-dependency groups installed on the VM.
  """

  tpu_type: str
  entry_module: str = TRAIN_ENTRY
  experiment_config: str | None = None
  seed_flag: str = 'config_overlay'
  flags: tuple[str, ...] = ()
  config_overlay: dict[str, object] = dataclasses.field(default_factory=dict)
  runtime_min: int = 60
  seeds: tuple[int, ...] = DEFAULT_SEEDS
  apt_packages: tuple[str, ...] = ()
  pip_extras: str = BASE_EXTRAS


# The shipped bundle is hardware-homogenized: every task runs on one 4-chip
# host except `decode_efficiency_vf`, whose metric is wall-clock and whose
# 8-chip slice is therefore part of the task definition, not a choice.
_DECODE_BUFFERS = {
    'sampling_decode_buffer_multiple': 128,
    'eval_decode_buffer_multiple': 128,
}

TASKS: dict[str, TaskDefaults] = {
    'pretrain_bpb_v32k': TaskDefaults(tpu_type='v6e-4', runtime_min=20),
    'pretrain_bpb_byte': TaskDefaults(tpu_type='v6e-4', runtime_min=18),
    'pretrain_optimizer_ttt': TaskDefaults(tpu_type='v6e-4', runtime_min=35),
    'rl_gemma3_1b': TaskDefaults(
        tpu_type='v6e-4', config_overlay=dict(_DECODE_BUFFERS), runtime_min=20),
    'rl_qwen2p5_math_1p5b': TaskDefaults(
        tpu_type='v6e-4', config_overlay=dict(_DECODE_BUFFERS),
        runtime_min=120),
    'rl_bfcl_qwen3_0p6b': TaskDefaults(
        tpu_type='v6e-4', runtime_min=45,
        config_overlay={**_DECODE_BUFFERS, 'validation_eval_batch_size': 128}),
    'rl_bfcl_gemma3_1b': TaskDefaults(
        tpu_type='v6e-4', config_overlay=dict(_DECODE_BUFFERS), runtime_min=25),
    'port_falcon_h1_0p5b': TaskDefaults(
        tpu_type='v6e-4', config_overlay=dict(_DECODE_BUFFERS), runtime_min=30),
    'port_recurrentgemma_2b': TaskDefaults(
        tpu_type='v6e-4', config_overlay=dict(_DECODE_BUFFERS), runtime_min=30),
    # Eval-only tasks: a released checkpoint is sampled, nothing is trained, so
    # they run the decode-eval entry point with the task's fixed decoding
    # protocol. The agent's submission overrides `--evaluation` (and only that).
    'sampling_lcb': TaskDefaults(
        tpu_type='v6e-4',
        entry_module=EVAL_ENTRY,
        experiment_config='qwen3_4b',
        seed_flag='seed',
        runtime_min=30,
        # The grader executes model-written Python; bubblewrap is its sandbox.
        apt_packages=('bubblewrap',),
        pip_extras=EVAL_EXTRAS,
        flags=(
            'lm_format=QwenV2Chat',
            'evaluation=LcbBaseline',
            'datasource_name=simply_json:livecodebench_v5',
            'top_p=0.95',
            'temperature=0.6',
            'top_k=20',
            'batch_size=48',
            'n_repeats=1',
            'max_seq_len=12000',
            'mesh_shape=1,1,4',
            'num_eval_threads=96',
        ),
    ),
    'decode_efficiency_vf': TaskDefaults(
        # 8 chips: the metric is wall-clock, so the slice is part of the task.
        tpu_type='v6e-8',
        entry_module=EVAL_ENTRY,
        experiment_config='qwen3_30b_a3b_thinking_2507',
        seed_flag='seed',
        runtime_min=60,
        pip_extras=EVAL_EXTRAS,
        seeds=(42,),  # wall-clock metric: one run, not a mean over three
        flags=(
            'lm_format=QwQChat',
            'evaluation=ZeroShotDeepSeekQwenR1CoTBoxed',
            'datasource_name=simply:aime25',
            'temperature=0.6',
            'top_p=0.95',
            'top_k=20',
            'batch_size=40',
            'n_repeats=4',
            'max_seq_len=32000',
            'mesh_shape=1,1,2,4',
        ),
    ),
}


def get(task: str) -> TaskDefaults:
  if task not in TASKS:
    raise KeyError(
        f'unknown --task={task!r}; known tasks: {sorted(TASKS)}. Use '
        '--entry-module/--tpu-type/--experiment_config to launch something '
        'that is not a benchmark task.'
    )
  return TASKS[task]
