# Ch5 Expert Iteration Summary

Source: prefix-cache-repaired single-GPU rerun; old EI artifacts are intentionally ignored.

| run | best answer acc | best step | final answer acc | final step | final accepted frac | final entropy |
|---|---:|---:|---:|---:|---:|---:|
| D_b=512, G=2, epochs=1 | 0.2216 | 5 | 0.2216 | 5 | 0.2734 | 0.6721 |
| D_b=512, G=2, epochs=2 | 0.3570 | 4 | 0.3392 | 5 | 0.2812 | 0.6600 |
| D_b=512, G=4, epochs=1 | 0.2607 | 2 | 0.2123 | 5 | 0.1748 | 2.6597 |
| D_b=512, G=4, epochs=2 | 0.3407 | 3 | 0.2466 | 5 | 0.2080 | 1.0129 |
| D_b=1024, G=4, epochs=2 | 0.3092 | 3 | 0.2073 | 5 | 0.1760 | 1.0949 |
| D_b=1024, G=8, epochs=2 | 0.3514 | 2 | 0.2082 | 5 | 0.1671 | 1.3029 |
| D_b=2048, G=4, epochs=2 | 0.3467 | 2 | 0.2269 | 5 | 0.1810 | 1.3813 |
| D_b=2048, G=4, epochs=3 | 0.4189 | 2 | 0.2304 | 5 | 0.2125 | 0.7737 |
