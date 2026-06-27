# GRPO Off-Policy Sweep Run Archive

| run | phase | role | status | best answer | best step | final answer | final step | source log |
|---|---|---|---|---:|---:|---:|---:|---|
| `broad_e2_tb256` | broad | true_off_policy | completed | 74.22% | 40 | 74.22% | 40 | `runs/logs/reruns/off_policy_sweep_retry_single_gpu_std_norm/broad_e2_tb256` |
| `broad_e2_tb128` | broad | true_off_policy | completed | 83.20% | 35 | 81.54% | 40 | `runs/logs/reruns/off_policy_sweep_retry_single_gpu_std_norm/broad_e2_tb128` |
| `broad_e4_tb256` | broad | true_off_policy | completed | 73.73% | 35 | 71.19% | 40 | `runs/logs/reruns/off_policy_sweep_retry_single_gpu_std_norm/broad_e4_tb256` |
| `broad_e4_tb128` | broad | true_off_policy | completed | 75.00% | 40 | 75.00% | 40 | `runs/logs/reruns/off_policy_sweep_retry_single_gpu_std_norm/broad_e4_tb128` |
| `broad_e2_tb64` | broad | true_off_policy | completed | 77.44% | 40 | 77.44% | 40 | `runs/logs/reruns/off_policy_sweep_remaining_broad_focus_single_gpu_std_norm/broad_e2_tb64` |
| `broad_e2_tb32` | broad | true_off_policy | completed | 74.71% | 40 | 74.71% | 40 | `runs/logs/reruns/off_policy_sweep_remaining_broad_focus_single_gpu_std_norm/broad_e2_tb32` |
| `broad_e8_tb256` | broad | true_off_policy | late collapse | 48.44% | 15 | 0.00% | 40 | `runs/logs/reruns/off_policy_sweep_remaining_broad_focus_single_gpu_std_norm/broad_e8_tb256` |
| `broad_e16_tb256` | broad | true_off_policy | collapsed | 3.81% | 0 | 0.00% | 40 | `runs/logs/reruns/off_policy_sweep_remaining_broad_focus_single_gpu_std_norm/broad_e16_tb256` |
| `on_policy_reference_e1_tb256_std_norm` | focused | on_policy_reference | completed | 88.18% | 190 | 88.18% | 200 | `runs/logs/ch7/grpo_on_policy_ablations/length_normalization_rerun_staging_single_gpu/masked_mean_lr4e-5` |
| `focused_e2_tb256` | focused | true_off_policy | completed | 86.33% | 145 | 82.32% | 200 | `runs/logs/reruns/off_policy_sweep_remaining_broad_focus_single_gpu_std_norm/focused_e2_tb256` |
| `focused_e2_tb128` | focused | true_off_policy | completed | 86.13% | 120 | 85.64% | 200 | `runs/logs/reruns/off_policy_sweep_remaining_broad_focus_single_gpu_std_norm/focused_e2_tb128` |
| `focused_e4_tb256` | focused | true_off_policy | completed | 86.43% | 145 | 82.23% | 200 | `runs/logs/reruns/off_policy_sweep_remaining_broad_focus_single_gpu_std_norm/focused_e4_tb256` |

Raw run files are archived under `artifacts/experiments/ch7/grpo_off_policy_sweep/runs/`. `sample_rollouts.jsonl` files are intentionally omitted because aggregate rollout summaries are sufficient for the writeup.
