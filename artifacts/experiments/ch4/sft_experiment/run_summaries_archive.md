# Ch4 SFT Run Summaries

This archive records the run summary data used by the Ch4 `sft_experiment` writeup section.

The raw per-run data needed to reproduce the writeup tables and curves are archived under `artifacts/experiments/ch4/sft_experiment/runs/`.

| run | experiment | train examples | effective epochs | best answer accuracy | best step | final answer accuracy | final step | final format accuracy | source run summary |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `size128_bs16_micro4_lr5e-5` | noisy size sweep | 128 | 112.50 | 0.337293 | 200 | 0.013129 | 900 | 0.128478 | `artifacts/experiments/ch4/sft_experiment/runs/size128_bs16_micro4_lr5e-5/run_summary.json` |
| `size256_bs16_micro4_lr5e-5` | noisy size sweep | 256 | 56.25 | 0.366364 | 100 | 0.055330 | 900 | 0.381994 | `artifacts/experiments/ch4/sft_experiment/runs/size256_bs16_micro4_lr5e-5/run_summary.json` |
| `size512_bs16_micro4_lr5e-5` | noisy size sweep | 512 | 28.12 | 0.393560 | 600 | 0.342294 | 900 | 0.936543 | `artifacts/experiments/ch4/sft_experiment/runs/size512_bs16_micro4_lr5e-5/run_summary.json` |
| `size1024_bs16_micro4_lr5e-5` | noisy size sweep | 1024 | 14.06 | 0.373554 | 200 | 0.363551 | 900 | 0.932791 | `artifacts/experiments/ch4/sft_experiment/runs/size1024_bs16_micro4_lr5e-5/run_summary.json` |
| `sizefull_bs16_micro4_lr5e-5` | noisy size sweep | 4866 | 2.96 | 0.355111 | 700 | 0.311660 | 900 | 0.967490 | `artifacts/experiments/ch4/sft_experiment/runs/sizefull_bs16_micro4_lr5e-5/run_summary.json` |
| `filtered_full_bs16_micro4_lr5e-5` | filtered full | 4136 | 3.48 | 0.396374 | 900 | 0.396374 | 900 | 0.961238 | `artifacts/experiments/ch4/sft_experiment/runs/filtered_full_bs16_micro4_lr5e-5/run_summary.json` |

Additional archived artifacts:

- `artifacts/experiments/ch4/sft_experiment/sft_results_summary.csv`
- `artifacts/experiments/ch4/sft_experiment/sft_results_summary.md`
- `artifacts/experiments/ch4/sft_experiment/sft_result_table.csv`
- `artifacts/experiments/ch4/sft_experiment/sft_result_table.json`
- `artifacts/experiments/ch4/sft_experiment/sft_size_sweep_table.md`
- `artifacts/experiments/ch4/sft_experiment/sft_filtered_comparison_table.md`
- `artifacts/experiments/ch4/sft_experiment/run_summaries.json`
- `artifacts/experiments/ch4/sft_experiment/runs/`
