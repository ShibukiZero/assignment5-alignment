# GRPO Learning Rate Run Summaries

This archive summarizes every learning-rate run used for the `grpo_learning_rate` writeup section. Completed runs with a `run_summary.json` also keep their raw summaries under the corresponding `.agents/logs/ch7/...` log directory. Early-stopped runs are summarized from their `metrics.jsonl` eval rows.

| learning rate | status | best answer reward | best step | best format accuracy | final answer reward | final step | final format accuracy | source log dir |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| `3e-6` | stopped early | 0.0449218750 | 35 | 0.2509765625 | 0.0449218750 | 35 | 0.2509765625 | `.agents/logs/ch7/grpo_learning_rate_manual_sweep/lr3e-6` |
| `5e-6` | completed | 0.1464843750 | 200 | 0.5390625000 | 0.1464843750 | 200 | 0.5390625000 | `.agents/logs/ch7/grpo_learning_rate_manual_sweep/lr5e-6` |
| `1e-5` | completed | 0.2929687500 | 180 | 0.7519531250 | 0.2714843750 | 200 | 0.7333984375 | `.agents/logs/ch7/grpo_on_policy_lr1e-5` |
| `2e-5` | completed | 0.3720703125 | 155 | 0.7617187500 | 0.3417968750 | 200 | 0.7529296875 | `.agents/logs/ch7/grpo_learning_rate_manual_sweep/lr2e-5` |
| `3e-5` | completed | 0.5576171875 | 190 | 0.8535156250 | 0.5498046875 | 200 | 0.8496093750 | `.agents/logs/ch7/grpo_learning_rate_manual_sweep/lr3e-5` |
| `4e-5` | completed | 0.7441406250 | 75 | 0.9501953125 | 0.7001953125 | 200 | 0.8769531250 | `.agents/logs/ch7/grpo_learning_rate_aggressive_grid/lr4e-5` |
| `5e-5` | completed | 0.6494140625 | 160 | 0.9052734375 | 0.6240234375 | 200 | 0.8378906250 | `.agents/logs/ch7/grpo_learning_rate_aggressive_grid/lr5e-5` |
| `7e-5` | stopped early | 0.1308593750 | 5 | 0.8417968750 | 0.0751953125 | 20 | 0.3642578125 | `.agents/logs/ch7/grpo_learning_rate_aggressive_grid/lr7e-5` |
| `2e-4` | collapsed | 0.0380859375 | 0 | 0.1806640625 | 0.0000000000 | 5 | 0.0000000000 | `.agents/logs/ch7/grpo_learning_rate_aggressive_grid/lr2e-4` |

Additional generated artifacts:

- `artifacts/experiments/ch7/grpo_learning_rate/grpo_learning_rate_validation_reward.svg`
- `artifacts/experiments/ch7/grpo_learning_rate/grpo_learning_rate_format_accuracy.svg`
- `artifacts/experiments/ch7/grpo_learning_rate/grpo_learning_rate_eval_points.csv`
- `artifacts/experiments/ch7/grpo_learning_rate/grpo_learning_rate_summary.csv`
- `artifacts/experiments/ch7/grpo_learning_rate/grpo_learning_rate_summary.md`
- `artifacts/experiments/ch7/grpo_learning_rate/run_summaries.json`
