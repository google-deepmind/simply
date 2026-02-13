## [Feature] Expert-Choice Routing for MoEFeedForward

The current `MoEFeedForward` supports token-choice routing where each
token selects its top-k experts. I'd like to add expert-choice routing
(Zhou et al., 2022 -- https://arxiv.org/abs/2202.09368) where each expert
selects its top-C tokens.

### Motivation

Expert-choice routing provides natural load balancing -- each expert
processes exactly C tokens by construction, eliminating the need for
auxiliary losses. The Google paper showed it outperforms token-choice on
language modeling and downstream tasks.

### Proposed changes

- Add `routing_mode` field (`'token_choice'` | `'expert_choice'`) to
  `MoEFeedForward`, `TransformerBlock`, and `BaseExperimentConfig`
- Add `_apply_expert_choice_moe()` following the `_apply_dense_moe`
  dispatch pattern (einsums, no shard_map)
- Add test configs `lm_moe_test` and `lm_moe_expert_choice_test`
- Add equivalence and gradient tests to `MoETest`

### Design notes

- Backward compatible -- default remains `'token_choice'`
- No new dependencies or files
- Initial implementation uses dense dispatch (not sparse/GMM) for
  simplicity; can be extended to sparse dispatch in a follow-up

Would this be a welcome addition?
