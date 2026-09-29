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


"""Tests for the Cloud TPU launcher's pure logic (no GCP calls)."""

import json

from tasks.research_bench.launch import launch_gcp
from tasks.research_bench.launch import remote
from tasks.research_bench.launch import task_defaults


def _args(argv):
  return launch_gcp.build_parser().parse_args(argv)


def _run_args(*extra):
  return _args(['run', '--bucket=gs://b', '--experiment_name=e', *extra])


def _command(*extra):
  args = _run_args(*extra)
  defaults = launch_gcp.resolve_defaults(args)
  return launch_gcp.build_command(args, defaults, 'gs://b/e/seed_42', 42)


class TestBuildCommand:

  def test_training_task_passes_seeds_through_the_config_overlay(self):
    cmd = _command('--task=pretrain_bpb_byte')
    assert cmd[:4] == ['python', '-u', '-m', 'tasks.research_bench.main']
    assert '--experiment_config=pretrain_bpb_byte' in cmd
    assert '--experiment_dir=gs://b/e/seed_42' in cmd
    overlay = json.loads(
        next(c for c in cmd if c.startswith('--config_overlay=')).split('=', 1)[1])
    assert overlay == {'model_seed': 42, 'dataset_seed': 42}

  def test_task_config_overlay_is_merged_under_the_seeds(self):
    cmd = _command('--task=rl_bfcl_qwen3_0p6b')
    overlay = json.loads(
        next(c for c in cmd if c.startswith('--config_overlay=')).split('=', 1)[1])
    assert overlay['validation_eval_batch_size'] == 128
    assert overlay['model_seed'] == 42

  def test_user_overlay_wins_over_the_task_default(self):
    cmd = _command('--task=rl_gemma3_1b',
                   '--config-overlay={"sampling_decode_buffer_multiple": 8}')
    overlay = json.loads(
        next(c for c in cmd if c.startswith('--config_overlay=')).split('=', 1)[1])
    assert overlay['sampling_decode_buffer_multiple'] == 8

  def test_eval_task_uses_a_seed_flag_and_the_eval_entry_point(self):
    cmd = _command('--task=sampling_lcb')
    assert cmd[3] == 'tasks.research_bench.eval_main'
    assert '--experiment_config=qwen3_4b' in cmd
    assert '--seed=42' in cmd
    assert not any(c.startswith('--config_overlay=') for c in cmd)

  def test_extra_flag_replaces_the_task_default_of_the_same_name(self):
    cmd = _command('--task=sampling_lcb', '--extra-flag=evaluation=MyDecoder')
    assert '--evaluation=MyDecoder' in cmd
    assert '--evaluation=LcbBaseline' not in cmd

  def test_defaults_come_from_the_task(self):
    args = _run_args('--task=decode_efficiency_vf')
    launch_gcp.resolve_defaults(args)
    assert args.tpu_type == 'v6e-8'
    assert args.entry_module == 'tasks.research_bench.eval_main'
    assert args.timeout_min == 4 * task_defaults.get('decode_efficiency_vf').runtime_min


class TestLayout:

  def test_experiment_paths(self):
    exp = launch_gcp.Experiment('gs://b/', 'exp')
    assert exp.root == 'gs://b/exp'
    assert exp.manifest_url == 'gs://b/exp/launch_manifest.json'
    assert exp.result_url(42) == 'gs://b/exp/seed_42/final_result.json'
    assert exp.log_url(42) == 'gs://b/exp/seed_42/log.txt'

  def test_vm_prefix_is_a_legal_tpu_node_name(self):
    name = launch_gcp.vm_prefix('9_Baseline/Pretrain_BPB v32k!')
    assert name == 'x9-baseline-pretrain-bpb-v32k'
    assert len(launch_gcp.vm_prefix('x' * 200)) <= 50


class TestJobScript:

  def _script(self, **kwargs):
    spec = remote.JobSpec(
        job_id='j1', seed=42, run_dir='gs://b/e/seed_42',
        status_url='gs://b/e/_ctl/j1.status', code_url='gs://b/e/c.tar.gz',
        command=('python', '-m', 'x', '--flag=a b'), **kwargs)
    return remote.job_script(spec)

  def test_quotes_the_command(self):
    assert "python -m x '--flag=a b'" in self._script()

  def test_writes_status_json_for_the_validator(self):
    script = self._script()
    assert 'RUN_DIR=gs://b/e/seed_42' in script
    assert '"$RUN_DIR/status.json"' in script
    assert 'COMPLETED' in script and 'FAILED' in script

  def test_mirrors_only_the_requested_assets(self):
    script = self._script(assets_url='gs://b/assets',
                          assets_include=('datasets/c4_bin',))
    assert 'for part in datasets/c4_bin;' in script
    assert 'export SIMPLY_DATASETS=/opt/simply_assets/datasets' in script

  def test_direct_assets_skip_the_mirror(self):
    script = self._script(assets_url='gs://b/assets', assets_mode='direct')
    assert 'export SIMPLY_VOCABS=gs://b/assets/vocabs' in script
    assert 'gcloud storage rsync' not in script

  def test_apt_packages_are_opt_in(self):
    assert 'apt_get install bubblewrap' not in self._script()
    assert 'apt_get install bubblewrap' in self._script(
        apt_packages=('bubblewrap',))


class TestTaskDefaults:

  def test_every_task_has_defaults(self):
    assert len(task_defaults.TASKS) == 11

  def test_decode_efficiency_is_pinned_to_its_hardware_and_one_seed(self):
    spec = task_defaults.get('decode_efficiency_vf')
    assert spec.tpu_type == 'v6e-8'
    assert spec.seeds == (42,)
    assert 'lm_format=QwQChat' in spec.flags
