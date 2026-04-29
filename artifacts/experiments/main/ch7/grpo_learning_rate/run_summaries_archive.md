# GRPO Learning Rate Run Summaries Archive

This archive summarizes the repaired-infra learning-rate runs used for the `grpo_learning_rate` writeup section. It intentionally ignores the older pre-repair sweep artifacts.

| learning rate | status | best answer | best step | best format | final answer | final step | final format | source log |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| `3e-6` | stopped early | 0.1005859375 | 50 | 0.3632812500 | 0.1005859375 | 50 | 0.3632812500 | `.agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr3e-6` |
| `5e-6` | completed | 0.4707031250 | 200 | 0.8193359375 | 0.4707031250 | 200 | 0.8193359375 | `.agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr5e-6` |
| `1e-5` | completed | 0.6777343750 | 195 | 0.9648437500 | 0.6728515625 | 200 | 0.9560546875 | `.agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr1e-5` |
| `2e-5` | completed | 0.8583984375 | 200 | 0.9833984375 | 0.8583984375 | 200 | 0.9833984375 | `.agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr2e-5` |
| `3e-5` | completed | 0.8642578125 | 200 | 0.9775390625 | 0.8642578125 | 200 | 0.9775390625 | `.agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr3e-5` |
| `4e-5` | completed | 0.8818359375 | 190 | 0.9765625000 | 0.8818359375 | 200 | 0.9804687500 | `.agents/logs/ch7/grpo_on_policy_ablations/length_normalization_rerun_staging_single_gpu/masked_mean_lr4e-5` |
| `5e-5` | completed | 0.8710937500 | 120 | 0.9511718750 | 0.8427734375 | 200 | 0.9462890625 | `.agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr5e-5` |
| `7e-5` | stopped early | 0.2128906250 | 50 | 1.0000000000 | 0.2128906250 | 50 | 1.0000000000 | `.agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr7e-5` |
| `2e-4` | collapsed | 0.0380859375 | 0 | 0.1806640625 | 0.0000000000 | 50 | 0.0000000000 | `.agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr2e-4` |

Generated artifacts:

- `artifacts/experiments/ch7/grpo_learning_rate/grpo_learning_rate_validation_reward.svg`
- `artifacts/experiments/ch7/grpo_learning_rate/grpo_learning_rate_format_accuracy.svg`
- `artifacts/experiments/ch7/grpo_learning_rate/grpo_learning_rate_eval_points.csv`
- `artifacts/experiments/ch7/grpo_learning_rate/grpo_learning_rate_summary.csv`
- `artifacts/experiments/ch7/grpo_learning_rate/grpo_learning_rate_summary.md`
- `artifacts/experiments/ch7/grpo_learning_rate/run_summaries.json`
