# GRPO Length Normalization Summary

This archive summarizes the repaired single-GPU length-normalization rerun used for the `grpo_length_normalization` writeup section. It intentionally ignores older pre-repair artifacts.

| run | normalization | status | best answer | best step | final answer | final format | final rollout answer | final avg length | source log |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `masked_mean` | `masked_mean` | completed | 88.18% | 190 | 88.18% | 98.05% | 80.86% | 466.9 | `.agents/logs/ch7/grpo_on_policy_ablations/length_normalization_rerun_staging_single_gpu/masked_mean_lr4e-5` |
| `masked_normalize` | `masked_normalize` | completed | 87.11% | 165 | 87.11% | 96.48% | 79.69% | 512.1 | `.agents/logs/ch7/grpo_on_policy_ablations/length_normalization_rerun_staging_single_gpu/runs/masked_normalize_lr4e-5/log` |
| `batch_token_mean` | `batch_token_mean` | completed | 86.91% | 110 | 86.43% | 98.83% | 78.12% | 375.3 | `.agents/logs/ch7/grpo_on_policy_ablations/length_normalization_rerun_staging_single_gpu/runs/batch_token_mean_lr4e-5/log` |

Generated artifacts:

- `artifacts/experiments/ch7/grpo_length_normalization/grpo_length_normalization_validation_reward.svg`
- `artifacts/experiments/ch7/grpo_length_normalization/grpo_length_normalization_format_accuracy.svg`
- `artifacts/experiments/ch7/grpo_length_normalization/grpo_length_normalization_response_length.svg`
- `artifacts/experiments/ch7/grpo_length_normalization/grpo_length_normalization_grad_norm.svg`
- `artifacts/experiments/ch7/grpo_length_normalization/grpo_length_normalization_token_entropy.svg`
- `artifacts/experiments/ch7/grpo_length_normalization/grpo_length_normalization_eval_points.csv`
- `artifacts/experiments/ch7/grpo_length_normalization/grpo_length_normalization_rollout_points.csv`
- `artifacts/experiments/ch7/grpo_length_normalization/grpo_length_normalization_train_points.csv`
- `artifacts/experiments/ch7/grpo_length_normalization/grpo_length_normalization_summary.csv`
- `artifacts/experiments/ch7/grpo_length_normalization/run_summaries.json`
- `artifacts/experiments/ch7/grpo_length_normalization/archive_manifest.json`

Raw run files are archived under `artifacts/experiments/ch7/grpo_length_normalization/runs/`. `sample_rollouts.jsonl` files are intentionally omitted because aggregate rollout summaries are sufficient for the writeup. Each run archive directory is removed before copying so stale files from older runs cannot remain mixed with the repaired rerun logs.
