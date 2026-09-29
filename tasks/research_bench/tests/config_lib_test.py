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

"""Tests for research-bench task baseline configs."""

import dataclasses
import json
import os

from absl import logging
from absl.testing import absltest
from simply import config_lib as core
from tasks.research_bench import checkpoint_lib
from tasks.research_bench import config_lib
from tasks.research_bench import data_lib
from tasks.research_bench import model_lib
from tasks.research_bench import port_eval_lib
from tasks.research_bench import tool_use_eval
from simply.utils import evaluation_lib
from simply.utils import module
from simply.utils import optimizers as opt_lib


class ConfigLibTest(absltest.TestCase):

  def test_pretrain_optimizer_ttt_resolves(self):
    cfg = config_lib.ExperimentConfigRegistry.get('pretrain_optimizer_ttt')()
    # 41M substrate, 100k-wide embedding/logit matmuls (NOT the 32k / byte
    # substrates). The tokenizer is the port's substitute for the internal
    # `vb100864_openmix_v1`; the SHAPES -- which is what this fixed-compute
    # task measures -- are the internal ones (see PORTING_NOTES.md).
    self.assertEqual(cfg.vocab_size, 100_864)
    self.assertEqual(cfg.vocab_name, config_lib.TTT_VOCAB_NAME)
    self.assertEqual(cfg.model_dim, 256)
    # Short horizon + curve sampling for time-to-target.
    self.assertEqual(cfg.num_train_steps, 1200)
    self.assertEqual(cfg.validation_eval_interval, 120)
    self.assertFalse(cfg.should_save_ckpt)
    # WSD schedule (non-collapsing: end_decay=0, decay starts at 80%).
    self.assertIsInstance(cfg.lr, opt_lib.LinearWarmupCosineDecay)
    self.assertEqual(cfg.lr.end_decay, 0.0)
    self.assertEqual(cfg.lr.decay_start_fraction, 0.8)
    # Baseline optimizer = tuned AdamW.
    self.assertIsInstance(cfg.optimizer, opt_lib.Adam)
    self.assertEqual(cfg.weight_decay, 0.1)
    # Task-specific train loop that appends the val-loss curve.
    self.assertEqual(cfg.train_loop_name, 'research_bench_ttt')

  def test_pretrain_optimizer_ttt_anchor_fixes_everything_but_the_optimizer(
      self,
  ):
    # `time_to_target_speedup` is measured against the anchor's val-loss curve,
    # so the anchor must differ from the baseline ONLY in the optimizer recipe;
    # a drift in model/data/budget would make the targets meaningless.
    base = config_lib.ExperimentConfigRegistry.get('pretrain_optimizer_ttt')()
    anchor = config_lib.ExperimentConfigRegistry.get(
        'pretrain_optimizer_ttt_anchor'
    )()
    for field in (
        'model_dim', 'n_layers', 'seq_len', 'batch_size', 'vocab_name',
        'vocab_size', 'num_train_steps', 'validation_eval_interval',
        'dataset', 'validation_datasets', 'train_loop_name',
    ):
      self.assertEqual(
          getattr(base, field), getattr(anchor, field), f'{field} drifted'
      )
    # ... and it must actually be a WEAKER recipe than the tuned baseline.
    self.assertLess(anchor.lr.value, base.lr.value)

  def test_ttt_train_loop_registered(self):
    self.assertIsNotNone(model_lib.TrainLoopRegistry.get('research_bench_ttt'))

  def test_pretrain_train_loop_registered(self):
    # The bpb tasks run under the guarded pretraining loop (rejects Pallas/
    # Mosaic custom kernels that would hide compute from the FLOP cap).
    self.assertIsNotNone(model_lib.TrainLoopRegistry.get('research_bench_pretrain'))

  def test_bpb_baselines_still_resolve(self):
    for name in ('pretrain_bpb_v32k', 'pretrain_bpb_byte'):
      cfg = config_lib.ExperimentConfigRegistry.get(name)()
      self.assertGreater(cfg.num_train_steps, 0)

  def test_pretraining_configs_read_the_repacked_c4_shards(self):
    # Core's `TFDSSource('c4:3.1.0')` has no public mirror; every pretraining
    # config must read the repacked shards instead, train AND validation, or
    # the run dies at the first batch on a machine without internal TFDS.
    for name in (
        'pretrain_bpb_v32k',
        'pretrain_bpb_byte',
        'pretrain_optimizer_ttt',
        'pretrain_optimizer_ttt_anchor',
    ):
      cfg = config_lib.ExperimentConfigRegistry.get(name)()
      self.assertIsInstance(cfg.dataset.source, data_lib.C4FileSource, name)
      self.assertEqual(cfg.dataset.source.split, 'train', name)
      self.assertLen(cfg.validation_datasets, 1, name)
      eval_source = cfg.validation_datasets[0].source
      self.assertIsInstance(eval_source, data_lib.C4FileSource, name)
      self.assertEqual(eval_source.split, 'validation', name)

  def test_ttt_val_loss_curve_tag_matches_the_configured_source(self):
    # The time-to-target metric is read back out of tb_log by tag; a mismatch
    # silently produces a run with no `validation_loss_curve`.
    cfg = config_lib.ExperimentConfigRegistry.get('pretrain_optimizer_ttt')()
    self.assertEqual(
        model_lib._val_loss_tag(cfg),  # pylint: disable=protected-access
        'C4FileSource/eval_loss',
    )

  def test_bpb_compute_accounting_is_faithful(self):
    # The fixed training_flops_xla cap is only meaningful if the layer stack is
    # unrolled (use_scan=False; a layer scan is counted once by cost_analysis)
    # and attention is not a Pallas/Mosaic custom-call (use_flash_attention=
    # False; its FLOPs are uncounted). Both must hold on the bpb baselines, and
    # the run must dispatch to the guarded loop that also rejects hand-rolled
    # custom kernels.
    for name in ('pretrain_bpb_v32k', 'pretrain_bpb_byte'):
      cfg = config_lib.ExperimentConfigRegistry.get(name)()
      self.assertFalse(cfg.use_scan, f'{name}: use_scan must be False')
      self.assertFalse(
          cfg.use_flash_attention,
          f'{name}: use_flash_attention must be False',
      )
      self.assertEqual(cfg.train_loop_name, 'research_bench_pretrain')
      # The cap marker is what actually ARMS the guards (see
      # main._resolve_train_loop): they follow this field, not the loop name,
      # so a recipe that registers its own train loop still runs under them.
      self.assertGreater(cfg.fixed_compute_cap_flops, 0.0, name)

  def test_compute_integrity_guard_rejects_uncounted_compute(self):
    # The two config settings that make XLA cost_analysis undercount the real
    # training compute must be rejected before the run starts.
    cfg = config_lib.ExperimentConfigRegistry.get('pretrain_bpb_v32k')()
    model_lib._check_capped_compute_config(cfg)  # pylint: disable=protected-access  # baseline is clean.
    for field in ('use_scan', 'use_flash_attention'):
      with self.subTest(field):
        with self.assertRaises(ValueError):
          model_lib._check_capped_compute_config(  # pylint: disable=protected-access
              dataclasses.replace(cfg, **{field: True})
          )

  def test_compute_integrity_wraps_any_train_loop(self):
    # Regression: the guards used to be keyed on train_loop_name, so a
    # submission could skip BOTH of them (and still write a normal-looking
    # final_result.json with undercounted FLOPs) just by naming another loop.
    cfg = config_lib.ExperimentConfigRegistry.get('pretrain_bpb_v32k')()
    for loop_name in ('research_bench_pretrain', 'default'):
      with self.subTest(loop_name):
        loop = model_lib.resolve_train_loop(
            dataclasses.replace(cfg, train_loop_name=loop_name), loop_name
        )
        self.assertTrue(getattr(loop, 'has_compute_integrity', False))
    # A task with no fixed-compute cap is left alone (`pretrain_optimizer_ttt`
    # is scored on data efficiency at a FIXED step budget, not a FLOP cap).
    ttt_cfg = config_lib.ExperimentConfigRegistry.get(
        'pretrain_optimizer_ttt'
    )()
    self.assertEqual(getattr(ttt_cfg, 'fixed_compute_cap_flops', 0.0), 0.0)
    ttt_loop = model_lib.resolve_train_loop(ttt_cfg, ttt_cfg.train_loop_name)
    self.assertFalse(getattr(ttt_loop, 'has_compute_integrity', False))
    for name in ('rl_bfcl_qwen3_0p6b', 'rl_gemma3_1b', 'port_falcon_h1_0p5b'):
      rl_cfg = config_lib.ExperimentConfigRegistry.get(name)()
      self.assertEqual(getattr(rl_cfg, 'fixed_compute_cap_flops', 0.0), 0.0)

  def test_compute_integrity_stamp_records_checks_and_cap(self):
    cfg = config_lib.ExperimentConfigRegistry.get('pretrain_bpb_byte')()
    result = model_lib._stamp_compute_integrity(  # pylint: disable=protected-access
        {'validation_bpb': 1.83, 'training_flops_xla': 6.12e15},
        self.create_tempdir().full_path,
        cfg,
        {'config_flags': 'ok', 'no_pallas_custom_call': 'ok'},
    )
    stamp = result['compute_integrity']
    self.assertEqual(stamp['checks']['no_pallas_custom_call'], 'ok')
    self.assertEqual(stamp['cap_flops'], cfg.fixed_compute_cap_flops)
    self.assertTrue(stamp['within_cap'])
    self.assertFalse(stamp['use_scan'])
    # Over-cap runs are recorded as such rather than silently accepted.
    over = model_lib._stamp_compute_integrity(  # pylint: disable=protected-access
        {'training_flops_xla': 10 * cfg.fixed_compute_cap_flops},
        self.create_tempdir().full_path,
        cfg,
        {},
    )
    self.assertFalse(over['compute_integrity']['within_cap'])

  def test_benchmark_configs_live_in_a_private_registry(self):
    # Benchmark config names must not be able to collide with core's: a config
    # defined in both would resolve to core's, which runs a loop with no
    # full-coverage assert on the held-out eval and no eval_protocol stamp.
    self.assertEqual(
        config_lib.ExperimentConfigRegistry.namespace, 'ExperimentV0p3'
    )
    for name in ('pretrain_bpb_v32k', 'rl_qwen2p5_math_1p5b'):
      with self.subTest(name):
        # Resolvable here, and NOT occupying core's keyspace.
        self.assertIsNotNone(config_lib.ExperimentConfigRegistry.get(name))
        self.assertIsNone(core.ExperimentConfigRegistry.get(name, False))
    # The collision that started it all: the same name in core no longer
    # conflicts, so there is no error to silence and nothing to shadow.
    shared = 'pretrain_bpb_v32k'
    core.ExperimentConfigRegistry.register(lambda: None, name=shared)
    self.addCleanup(core.ExperimentConfigRegistry.unregister, shared)
    self.assertIsNot(
        config_lib.ExperimentConfigRegistry.get(shared),
        core.ExperimentConfigRegistry.get(shared),
    )
    # A core-only config is invisible to the benchmark binary by design.
    self.assertIsNone(
        config_lib.ExperimentConfigRegistry.get('qwen3_4b', False)
    )

  def test_run_provenance_is_recorded_not_enforced(self):
    # Deliberately evidence, not a gate: this runs after the work is done, so
    # failing here would destroy both the compute and the artifact a reviewer
    # needs. The hard checks run before training instead.
    capped = config_lib.ExperimentConfigRegistry.get('pretrain_bpb_v32k')()
    rl_cfg = config_lib.ExperimentConfigRegistry.get('rl_qwen2p5_math_1p5b')()
    d = self.create_tempdir().full_path
    # A result from a loop that is not ours, with no stamp: recorded, no raise.
    prov = model_lib.record_run_provenance(
        rl_cfg, {'eval_accuracy': 0.53}, d, config_name='x', loop_name='rl')
    self.assertFalse(prov['eval_protocol_present'])
    self.assertFalse(prov['train_loop_is_research_bench'])
    self.assertIsNone(prov['compute_integrity_present'])
    # ... and it is written into the artifact for the reviewer to find.
    with open(f'{d}/final_result.json') as f:
      self.assertFalse(json.load(f)['run_provenance']['eval_protocol_present'])
    # A clean run records the positive case too.
    prov = model_lib.record_run_provenance(
        rl_cfg, {'eval_accuracy': 0.53, 'eval_protocol': {'n_scored': 262}},
        self.create_tempdir().full_path,
        config_name='rl_qwen2p5_math_1p5b', loop_name='research_bench_rl')
    self.assertTrue(prov['eval_protocol_present'])
    self.assertTrue(prov['train_loop_is_research_bench'])
    # Under a compute cap the compute stamp is tracked as well.
    prov = model_lib.record_run_provenance(
        capped, {'validation_bpb': 1.27, 'eval_protocol': {}},
        self.create_tempdir().full_path, loop_name='research_bench_pretrain')
    self.assertFalse(prov['compute_integrity_present'])
    # No result at all: still no raise.
    self.assertEqual(model_lib.record_run_provenance(rl_cfg, None, ''), {})

  def test_rl_baselines_pin_clipping_to_the_measured_regime(self):
    # The published RL baselines/anchors were measured with clipping OFF; the
    # loop now honours these fields, so the configs must pin them explicitly
    # (core's default is 1.0) or every anchor would shift.
    for name in (
        'rl_bfcl_qwen3_0p6b',
        'rl_bfcl_gemma3_1b',
        'rl_gemma3_1b',
        'rl_qwen2p5_math_1p5b',
    ):
      cfg = config_lib.ExperimentConfigRegistry.get(name)()
      self.assertLessEqual(cfg.clip_grad_norm, 0.0, name)
      self.assertLessEqual(cfg.clip_local_update_rms, 0.0, name)

  def test_pretrain_eval_protocol_records_the_scored_setup(self):
    # The metric depends on the validation source, vocab and seq_len; record
    # them next to the number.
    cfg = config_lib.ExperimentConfigRegistry.get('pretrain_bpb_byte')()
    got = model_lib._eval_protocol(cfg)  # pylint: disable=protected-access
    self.assertEqual(got['vocab_name'], 'byte256')
    self.assertEqual(got['vocab_size'], 259)
    self.assertEqual(got['seq_len'], cfg.seq_len)
    self.assertTrue(got['validation_sources'])
    json.dumps(got)  # written into final_result.json

  def test_rl_tasks_pin_the_eval_prompt_format(self):
    # The eval prompt format + stop tokens must not follow the training-side
    # ones (which the recipe may change): every RL task pins its own, equal to
    # the training values in the shipped baselines.
    for name in (
        'rl_bfcl_qwen3_0p6b',
        'rl_bfcl_gemma3_1b',
        'rl_gemma3_1b',
        'rl_qwen2p5_math_1p5b',
        'port_falcon_h1_0p5b',
        'port_recurrentgemma_2b',
    ):
      cfg = config_lib.ExperimentConfigRegistry.get(name)()
      self.assertEqual(cfg.validation_lm_format_name, cfg.lm_format_name, name)
      self.assertEqual(
          tuple(cfg.eval_extra_eos_tokens), tuple(cfg.extra_eos_tokens), name
      )

  def test_decode_buffer_multiples_are_overridable(self):
    # The reference RL/port runs are launched with these at 128 via
    # --config_overlay (the configs keep the internal default of 0), so the
    # fields have to exist and be settable on every RL config.
    for name in (
        'rl_bfcl_qwen3_0p6b',
        'rl_bfcl_gemma3_1b',
        'rl_gemma3_1b',
        'rl_qwen2p5_math_1p5b',
        'port_falcon_h1_0p5b',
        'port_recurrentgemma_2b',
    ):
      cfg = config_lib.ExperimentConfigRegistry.get(name)()
      self.assertEqual(cfg.sampling_decode_buffer_multiple, 0, name)
      self.assertEqual(cfg.eval_decode_buffer_multiple, 0, name)
      overridden = dataclasses.replace(
          cfg,
          sampling_decode_buffer_multiple=128,
          eval_decode_buffer_multiple=128,
      )
      self.assertEqual(overridden.eval_decode_buffer_multiple, 128, name)

  def test_tool_use_baselines_resolve(self):
    for name, algo in (
        ('rl_bfcl_qwen3_0p6b', 'simple_grpo'),
        ('rl_bfcl_gemma3_1b', 'simple_grpo'),
    ):
      cfg = config_lib.ExperimentConfigRegistry.get(name)()
      # Self-contained tool-use RL loop + selected algorithm.
      self.assertEqual(cfg.train_loop_name, 'research_bench_rl')
      self.assertEqual(cfg.rl_algorithm, algo)
      # FIXED held-out eval + its data (call-required BFCL-live split), and the
      # shapeable training reward + its (non-live) train split -- kept distinct.
      self.assertIsInstance(
          cfg.validation_evaluation, tool_use_eval.BFCLFunctionCallEvaluation
      )
      self.assertEqual(
          cfg.validation_datasets[0].source, 'simply:bfcl_live_eval'
      )
      self.assertEqual(cfg.dataset.source, 'simply:bfcl_nonlive_train')
      # Held-out eval sampling is FIXED and decoupled from the training sampler:
      # greedy (temp 0), pass@1, fixed decode budget, larger input budget.
      self.assertEqual(cfg.eval_temperature, 0.0)
      self.assertEqual(cfg.eval_num_samples, 1)
      self.assertEqual(cfg.eval_max_decode_steps, 160)
      self.assertEqual(cfg.eval_max_input_len, 3072)
      self.assertEqual(cfg.eval_prefill_size, 3072)
    # (The `research_bench_rl` loop is registered at entry point / by rl_loop,
    # not by config_lib; its registration is covered in rl_loop_test.)

  def test_rl_gemma3_1b_resolves(self):
    cfg = config_lib.ExperimentConfigRegistry.get('rl_gemma3_1b')()
    self.assertEqual(cfg.train_loop_name, 'research_bench_rl')
    self.assertEqual(cfg.rl_algorithm, 'simple_grpo')
    # R1-Zero-style naive GRPO: no KL, no reference policy.
    self.assertFalse(cfg.use_ref_params)
    self.assertEqual(cfg.kl_coeff, 0.0)
    # Gemma-3-1B PRETRAINED base + fixed 0-shot boxed eval on full GSM8K test.
    self.assertEqual(cfg.vocab_name, 'vb262144_gemma3')
    self.assertIn('GEMMA-3.0-1B-PT', cfg.init_ckpt_dir)
    self.assertEqual(cfg.dataset.source, 'simply:gsm8k_train')
    self.assertEqual(cfg.validation_datasets[0].source, 'simply:gsm8k_test')
    self.assertIsInstance(
        cfg.validation_evaluation,
        evaluation_lib.ZeroShotBoxedInQuestionEvaluation,
    )
    # Greedy pass@1 eval, decoupled from training.
    self.assertEqual(cfg.eval_temperature, 0.0)
    self.assertEqual(cfg.eval_num_samples, 1)

  def test_rl_qwen2p5_math_1p5b_resolves(self):
    cfg = config_lib.ExperimentConfigRegistry.get('rl_qwen2p5_math_1p5b')()
    self.assertEqual(cfg.train_loop_name, 'research_bench_rl')
    self.assertEqual(cfg.rl_algorithm, 'simple_grpo')
    self.assertFalse(cfg.use_ref_params)
    self.assertEqual(cfg.kl_coeff, 0.0)
    # Qwen2.5-Math-1.5B base + MATH500 L4-5 avg@8 (temp 0.6) fixed eval.
    self.assertEqual(cfg.vocab_name, 'Qwen2.5')
    self.assertEqual(cfg.dataset.source, 'simply:dsr40k_train')
    self.assertEqual(
        cfg.validation_datasets[0].source, 'simply:math500_test_l45'
    )
    self.assertIsInstance(
        cfg.validation_evaluation,
        evaluation_lib.ZeroShotBoxedInQuestionEvaluation,
    )
    self.assertEqual(cfg.eval_num_samples, 8)  # avg@8
    self.assertEqual(cfg.eval_temperature, 0.6)
    # float32 REQUIRED (bf16 NaNs on this model).
    self.assertEqual(cfg.activation_dtype_name, 'float32')
    self.assertEqual(cfg.ref_params_dtype, 'float32')

  def test_port_falcon_h1_0p5b_resolves(self):
    cfg = config_lib.ExperimentConfigRegistry.get('port_falcon_h1_0p5b')()
    # Unified train+eval on the research_bench_rl loop; default eval-only.
    self.assertEqual(cfg.train_loop_name, 'research_bench_rl')
    self.assertEqual(cfg.num_train_steps, 0)
    self.assertEqual(cfg.model_name, 'FalconH1LM')
    # Kept ANCHORS (valid values): model_dim disambiguates 2D weight shapes;
    # per_head_dim splits attention heads; mamba_d_state/n_groups pin the SSM
    # conv split; the scalars can't be recovered from any tensor.
    self.assertEqual(cfg.model_dim, 1024)
    self.assertEqual(cfg.per_head_dim, 64)
    self.assertEqual(cfg.n_heads, 8)
    self.assertEqual(cfg.n_kv_heads, 2)
    self.assertEqual(cfg.mamba_d_state, 128)
    self.assertEqual(cfg.mamba_n_groups, 1)
    self.assertEqual(cfg.falcon_rope_theta, 1e11)
    self.assertEqual(cfg.rms_norm_epsilon, 1e-5)
    # BLANKED (sentinel -1): the participant must recover these from the
    # checkpoint/vocab ('?' in the spec table). No circular dependency.
    self.assertEqual(cfg.vocab_size, -1)
    self.assertEqual(cfg.n_layers, -1)
    self.assertEqual(cfg.ffn_expand_dim, -1)
    self.assertEqual(cfg.mamba_d_ssm, -1)
    self.assertEqual(cfg.mamba_n_heads, -1)
    self.assertEqual(cfg.mamba_d_head, -1)
    self.assertEqual(cfg.mamba_d_conv, -1)
    self.assertEqual(cfg.vocab_name, 'FalconH1')
    self.assertEqual(cfg.lm_format_name, 'FalconH1GSM8K')
    self.assertEqual(cfg.init_ckpt_format, 'FalconH1Format')
    self.assertIn('Falcon-H1-0.5B-Base', cfg.init_ckpt_dir)
    # FIXED full-GSM8K-test 5-shot-strict eval, greedy pass@1.
    self.assertEqual(cfg.validation_datasets[0].source, 'simply:gsm8k_test')
    self.assertIsInstance(
        cfg.validation_evaluation,
        port_eval_lib.FewShotGSM8K5ShotStrictNoTrailSpaceEvaluation,
    )
    self.assertEqual(cfg.eval_temperature, 0.0)
    self.assertEqual(cfg.eval_num_samples, 1)
    self.assertFalse(cfg.use_ref_params)

  def test_port_recurrentgemma_2b_resolves(self):
    cfg = config_lib.ExperimentConfigRegistry.get('port_recurrentgemma_2b')()
    self.assertEqual(cfg.train_loop_name, 'research_bench_rl')
    self.assertEqual(cfg.num_train_steps, 0)
    self.assertEqual(cfg.model_name, 'RecurrentGemmaLM')
    # Kept ANCHORS (valid): model_dim labels the 2-D weight axes; n_kv_heads=1
    # (MQA) makes k_proj = head_dim; scalars aren't in any tensor.
    self.assertEqual(cfg.model_dim, 2560)
    self.assertEqual(cfg.n_kv_heads, 1)
    self.assertEqual(cfg.window_size, 2048)
    self.assertEqual(cfg.rms_norm_epsilon, 1e-6)
    self.assertEqual(
        cfg.block_type, ('recurrent', 'recurrent', 'attention')
    )
    self.assertEqual(cfg.output_logits_soft_cap, 30.0)
    self.assertTrue(cfg.norm_scale_plus_one)
    # BLANKED (sentinel -1): recovered from the checkpoint tensor shapes
    # ('?' in the spec table). No circular dependency.
    self.assertEqual(cfg.vocab_size, -1)
    self.assertEqual(cfg.n_layers, -1)
    self.assertEqual(cfg.n_heads, -1)
    self.assertEqual(cfg.per_head_dim, -1)
    self.assertEqual(cfg.ffn_expand_dim, -1)
    self.assertEqual(cfg.lru_width, -1)
    self.assertEqual(cfg.conv1d_width, -1)
    self.assertEqual(cfg.rglru_num_heads, -1)
    self.assertEqual(cfg.vocab_name, 'RecurrentGemma')
    self.assertEqual(cfg.lm_format_name, 'RecurrentGemmaGSM8K')
    self.assertEqual(cfg.init_ckpt_format, 'RecurrentGemmaFormat')
    self.assertIn('RecurrentGemma-2B', cfg.init_ckpt_dir)
    # Model-sharded mesh: single-host 4-chip slice, 4-way model axis.
    self.assertEqual(cfg.mesh_shape, {'model': 4})
    # FIXED full-GSM8K-test 5-shot-strict eval, single sample at temp 0.4.
    self.assertEqual(cfg.validation_datasets[0].source, 'simply:gsm8k_test')
    self.assertIsInstance(
        cfg.validation_evaluation, port_eval_lib.FewShot5StrictGSM8KEvaluation
    )
    self.assertEqual(cfg.eval_temperature, 0.4)
    self.assertEqual(cfg.eval_num_samples, 1)
    self.assertFalse(cfg.use_ref_params)

  def test_port_model_and_format_stubs_registered(self):
    # The port model classes + checkpoint formats ship REGISTERED (so configs
    # resolve + the package imports) but STUBBED (the port IS the task).
    #
    # What is asserted here has to hold in BOTH states of this package: as
    # shipped, and once the port is implemented. Being registered and exposing
    # the interface is such a property; RAISING NotImplementedError is not --
    # a completed port makes that false. Whether the port works is measured by
    # the scored run, not here.
    for model_name in ('FalconH1LM', 'RecurrentGemmaLM'):
      self.assertIsNotNone(module.ModuleRegistry.get(model_name))
    for fmt in ('FalconH1Format', 'RecurrentGemmaFormat'):
      fmt_cls = checkpoint_lib.CheckpointFormatRegistry.get(fmt)
      self.assertIsNotNone(fmt_cls)
      self.assertTrue(callable(getattr(fmt_cls, 'transforms', None)))

  def test_checkpoint_dirs_hold_step_subdirectories(self):
    """A staged checkpoint must look like simply expects: `<dir>/<step>/`.

    The path in a config is only half the contract; the other half is where the
    asset actually lands. A checkpoint one directory level off loads fine in
    every CPU test (nothing reads it) and then fails minutes into an
    accelerator run with `No checkpoint found in ...`, which is how this test
    came to exist.
    """
    checked = 0
    for name in sorted(config_lib.ExperimentConfigRegistry.keys()):
      ckpt_dir = getattr(
          config_lib.ExperimentConfigRegistry.get_config(name),
          'init_ckpt_dir', '')
      if not ckpt_dir or not os.path.isdir(ckpt_dir):
        continue  # asset not staged in this environment
      steps = [d for d in os.listdir(ckpt_dir) if d.isdigit()]
      self.assertNotEmpty(
          steps,
          f'{name}: init_ckpt_dir={ckpt_dir} has no numeric step subdirectory;'
          f' found {sorted(os.listdir(ckpt_dir))[:5]}')
      checked += 1
    logging.info('checked %d staged checkpoint dirs', checked)


if __name__ == '__main__':
  absltest.main()
