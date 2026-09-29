# Golden artifacts from real runs

Written by `python -m tasks.research_bench.main` on CPU with tiny
`--config_overlay` shapes (`seq_len=128`, `batch_size=2`, 4-20 steps), so the
metric VALUES are junk but every stamp the validator matches on
(`eval_protocol`, `compute_integrity`, `run_provenance`, `validation_loss_curve`)
has its genuine production shape. `validator_test.py` asserts the specs accept
them, so a future change to either side that breaks the contract fails a test
rather than a submission.

The three pretrain stamps come from end-to-end CPU runs. The six RL/port
stamps come from `rl_loop._eval_protocol` + `model_lib.record_run_provenance`
driven at their real call sites against the real staged datasets -- that
function is pure in (config, Evaluation, len(eval_ds)), so no checkpoint is
needed and the stamp is production-exact; only `eval_accuracy` (0.0) is
synthetic. Their `n_scored` (1319 / 262 / 1351) is the open-source data
pipeline reproducing the internal cardinalities.

Source: A1's smoke runs, 2026-09-29 (`$AMPLIO_ARTIFACT_DIR/a1_evidence/`).
