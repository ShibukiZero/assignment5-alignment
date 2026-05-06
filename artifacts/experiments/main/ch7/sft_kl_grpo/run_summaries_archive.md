# SFT warm-start + KL GRPO run archive

This archive summarizes the SFT warm-start + KL-regularized GRPO experiment.

| role | run | source log | best answer | best step | final answer | final step |
|---|---|---|---:|---:|---:|---:|
| SFT warm start | `sft_256_step100` | `.agents/logs/reruns/sft_kl_grpo/sft_256_step100` | 39.26% | 100 | 39.26% | 100 |
| GRPO after SFT + KL | `grpo_sft256_step100_kl0p01_190` | `.agents/logs/reruns/sft_kl_grpo/grpo_sft256_step100_kl0p01_190` | 76.86% | 190 | 76.86% | 190 |
| Direct GRPO reference | `direct_grpo_lr4e-5` | `artifacts/experiments/ch7/grpo_length_normalization/runs/masked_mean` | 88.18% | 190 | 88.18% | 200 |

Raw run files are archived under `artifacts/experiments/ch7/sft_kl_grpo/runs/`.
`sample_rollouts.jsonl` files are intentionally omitted because aggregate rollout summaries are sufficient for the writeup.
