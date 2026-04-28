# GRPO Off-Policy Clip Ablation Run Archive

| run | variant | loss | clip low | clip high | status | best answer | best step | final answer | final step | source log |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| `GRPO-Clip 0.20/0.20 (e2/tb256)` | `symmetric_clip` | `grpo_clip` | 0.20 | 0.20 | completed | 86.33% | 145 | 82.32% | 200 | `.agents/logs/reruns/off_policy_sweep_remaining_broad_focus_single_gpu_std_norm/focused_e2_tb256` |
| `GRPO-No-Clip (e2/tb256)` | `no_clip` | `grpo_no_clip` | 0.20 | 0.20 | late collapse | 24.61% | 125 | 0.00% | 200 | `.agents/logs/reruns/off_policy_clip_ablation_e2_tb256_single_gpu_std_norm/no_clip_e2_tb256` |
| `GRPO-Clip 0.20/0.28 (e2/tb256)` | `asymmetric_clip` | `grpo_clip` | 0.20 | 0.28 | completed | 83.79% | 150 | 78.91% | 200 | `.agents/logs/reruns/off_policy_clip_ablation_e2_tb256_single_gpu_std_norm/clip_low0p2_high0p28_e2_tb256` |

Raw run files are archived under `artifacts/experiments/ch7/grpo_off_policy_clip_ablation/runs/`.
