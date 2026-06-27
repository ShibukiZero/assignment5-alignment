# GRPO Group Standard Deviation Summary

| run | use std normalization | status | best answer | best step | final answer | final format | final rollout answer | final avg length | final grad norm |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `std_normalization` | `True` | completed | 88.18% | 190 | 88.18% | 98.05% | 80.86% | 466.9 | 0.14 |
| `no_std_normalization` | `False` | completed | 87.21% | 200 | 87.21% | 97.66% | 79.30% | 453.1 | 0.05 |

Default source logs:

- `std_normalization`: `runs/logs/ch7/grpo_on_policy_ablations/length_normalization_rerun_staging_single_gpu/masked_mean_lr4e-5`
- `no_std_normalization`: `runs/logs/reruns/grpo_group_standard_deviation_retry_single_gpu/grpo_group_standard_deviation/no_std_normalization`

Raw run files are archived under `artifacts/experiments/ch7/grpo_group_standard_deviation/runs/`. `sample_rollouts.jsonl` files are intentionally omitted because aggregate rollout summaries are sufficient for the writeup.
