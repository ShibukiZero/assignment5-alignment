# GRPO Learning Rate Sweep Summary

| learning rate | status | best answer | best step | final answer | final step | final format |
|---:|---|---:|---:|---:|---:|---:|
| `3e-6` | stopped early | 4.49% | 35 | 4.49% | 35 | 25.10% |
| `5e-6` | completed | 14.65% | 200 | 14.65% | 200 | 53.91% |
| `1e-5` | completed | 29.30% | 180 | 27.15% | 200 | 73.34% |
| `2e-5` | completed | 37.21% | 155 | 34.18% | 200 | 75.29% |
| `3e-5` | completed | 55.76% | 190 | 54.98% | 200 | 84.96% |
| `4e-5` | completed | 74.41% | 75 | 70.02% | 200 | 87.70% |
| `5e-5` | completed | 64.94% | 160 | 62.40% | 200 | 83.79% |
| `7e-5` | stopped early | 13.09% | 5 | 7.52% | 20 | 36.43% |
| `2e-4` | collapsed | 3.81% | 0 | 0.00% | 5 | 0.00% |

Use `lr=4e-5` as the selected on-policy learning rate. It achieved the best validation answer reward and remained substantially better than the lower learning rates at the final checkpoint. Larger rates showed unstable or collapsed behavior.
