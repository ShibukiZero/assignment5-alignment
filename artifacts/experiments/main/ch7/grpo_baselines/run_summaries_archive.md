# GRPO Baselines Summary

| run | loss type | status | best answer | best step | final answer | final format | final rollout answer | final avg length |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `reinforce_with_baseline` | `reinforce_with_baseline` | completed | 88.18% | 190 | 88.18% | 98.05% | 80.86% | 466.9 |
| `no_baseline` | `no_baseline` | completed | 26.17% | 90 | 18.46% | 85.55% | 22.66% | 103.6 |

Default source logs:

- `reinforce_with_baseline`: `runs/logs/ch7/grpo_on_policy_ablations/length_normalization_rerun_staging_single_gpu/masked_mean_lr4e-5`
- `no_baseline`: `runs/logs/reruns/grpo_ablation_repairs_single_gpu/grpo_baselines/no_baseline`

Raw run files are archived under `artifacts/experiments/ch7/grpo_baselines/runs/`. `sample_rollouts.jsonl` files are intentionally omitted because aggregate rollout summaries are sufficient for the writeup.
