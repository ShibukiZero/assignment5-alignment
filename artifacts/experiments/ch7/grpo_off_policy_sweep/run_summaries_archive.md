# GRPO Off-Policy Sweep Run Archive

| run | phase | role | status | best answer | best step | final answer | final step |
|---|---|---|---|---:|---:|---:|---:|
| `broad_e1_tb256_control` | broad | on_policy_style_control | completed | 61.52% | 40 | 61.52% | 40 |
| `broad_e2_tb256` | broad | true_off_policy | completed | 18.55% | 20 | 13.96% | 40 |
| `broad_e2_tb128` | broad | true_off_policy | late collapse | 46.09% | 10 | 0.00% | 40 |
| `broad_e4_tb256` | broad | true_off_policy | completed | 22.17% | 10 | 16.99% | 40 |
| `broad_e4_tb128` | broad | true_off_policy | completed | 29.88% | 5 | 0.00% | 40 |
| `on_policy_reference_e1_tb256` | focused | on_policy_reference | completed | 80.08% | 95 | 77.64% | 200 |
| `focused_e2_tb256` | focused | true_off_policy | late collapse | 34.38% | 65 | 0.00% | 200 |

Raw run files are archived under `artifacts/experiments/ch7/grpo_off_policy_sweep/runs/`.
