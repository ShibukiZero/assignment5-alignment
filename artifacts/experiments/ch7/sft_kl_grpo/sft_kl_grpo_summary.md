# SFT warm-start + KL GRPO summary

| role | run | KL coef | best answer | best step | final answer | final step | final format | final rollout answer | final avg length | final reference KL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT warm start | `sft_256_step100` |  | 39.26% | 100 | 39.26% | 100 | 88.87% |  |  |  |
| GRPO after SFT + KL | `grpo_sft256_step100_kl0p01_190` | 0.01 | 76.86% | 190 | 76.86% | 190 | 98.05% | 57.81% | 346.8 | 0.2036 |
| Direct GRPO reference | `direct_grpo_lr4e-5` | 0 | 88.18% | 190 | 88.18% | 200 | 98.05% | 80.86% | 466.9 |  |

Generated artifacts:

- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_staged_validation_reward.svg`
- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_validation_reward.svg`
- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_validation_reward_wall_clock.svg`
- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_format_accuracy.svg`
- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_response_length.svg`
- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_token_entropy.svg`
- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_reference_kl.svg`
- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_summary.csv`
- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_eval_points.csv`
- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_train_points.csv`
- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_rollout_points.csv`
