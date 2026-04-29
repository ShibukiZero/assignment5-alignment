# GRPO Off-Policy Clip Ablation Summary

| run | variant | loss | clip low | clip high | status | best answer | best step | final answer | final format | final avg length | last finite grad norm |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `GRPO-Clip 0.20/0.20 (e2/tb256)` | `symmetric_clip` | `grpo_clip` | 0.20 | 0.20 | completed | 86.33% | 145 | 82.32% | 98.14% | 474.8 | 0.0908 |
| `GRPO-No-Clip (e2/tb256)` | `no_clip` | `grpo_no_clip` | 0.20 | 0.20 | late collapse | 24.61% | 125 | 0.00% | 0.00% | 155.8 | 0.0000 |
| `GRPO-Clip 0.20/0.28 (e2/tb256)` | `asymmetric_clip` | `grpo_clip` | 0.20 | 0.28 | completed | 83.79% | 150 | 78.91% | 94.73% | 587.0 | 0.2852 |
