## Add expert-choice routing mode for MoEFeedForward

Adds expert-choice routing (Zhou et al., 2022) where each expert selectsits top-C tokens, providing natural load balancing without auxiliary losses.

### Changes

-   `model_lib.py`: `routing_mode` field, `_apply_expert_choice_moe()`,restructured `apply()` with early routing-mode branch
-   `config_lib.py`: `routing_mode` in `BaseExperimentConfig`,`lm_moe_test` and `lm_moe_expert_choice_test` configs
-   `model_lib_test.py`: `simple_expert_choice_moe()` reference impl,forward/gradient equivalence tests

### How to test

```bash
# Expert-choice MoE local testpython -m simply.main     --experiment_config lm_moe_expert_choice_test     --experiment_dir /tmp/moe_ec_test --alsologtostderr# All MoE unit tests (including existing + new)pytest simply/model_lib_test.py::MoETest -v
```

### Design decisions

-   Follows `_apply_dense_moe` dispatch pattern (einsums, not sparse GMM)
-   Capacity: C = num_experts_per_token x num_tokens / num_experts
-   `lbl_loss` is skipped when `routing_mode='expert_choice'` (load isbalanced by construction, so the auxiliary loss is unnecessary)
-   `routing_mode` is validated; unknown values raise `ValueError`
-   Token-choice path is unchanged: all 7 existing `simple_moe()`equivalence tests pass with the same tolerances as before