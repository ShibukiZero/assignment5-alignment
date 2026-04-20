# GRPO Off-Policy Sweep Experiment Log

## Setup

- Fixed `rollout_batch_size = 256` for all broad and focused off-policy runs.
- Fixed `learning_rate = 4e-5`, `loss_type = grpo_clip`, `loss_normalization = masked_mean`, and `use_std_normalization = false` based on earlier on-policy GRPO ablations.
- Broad sweep varied `epochs_per_rollout_batch in {1, 2, 4}` and `train_batch_size in {256, 128}`.
- We kept the `e1, tb256` point as an on-policy-style control inside the broad sweep because it uses the off-policy infrastructure but performs no rollout reuse.
- These settings span optimizer-update counts of `1`, `2`, `4`, and `8` per rollout batch.

## Broad Findings

- The control run `broad_e1_tb256_control` reached 61.52% by step 40.
- Among the true off-policy broad runs, `broad_e2_tb128` was the strongest peak performer at 46.09%.
- Both `tb128` settings collapsed by the end of the 40-step broad sweep, while the `tb256` settings remained finite.

## Focused Choice

- We selected `focused_e2_tb256` for the 200-step focused run because it was the most stable true off-policy configuration in the broad sweep.
- The on-policy comparison run is `on_policy_reference_e1_tb256`, which uses `epochs_per_rollout_batch = 1`, `train_batch_size = 256`, and `reinforce_with_baseline` at the same `4e-5` learning rate.

## Focused Outcome

- `focused_e2_tb256` peaked at 34.38% on step 65, then suffered a late collapse and finished at 0.00%.
- The on-policy reference peaked at 80.08% and finished at 77.64%.
- Diagnostics show that the off-policy run stayed competitive for much of training but eventually hit a numerical blow-up on the second update of the rollout batch, followed by zero-reward, max-length degenerate generations.
