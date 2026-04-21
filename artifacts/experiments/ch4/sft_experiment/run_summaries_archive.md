# Ch4 SFT Run Summaries

This archive records the run summary data used by the Ch4 `sft_experiment` writeup section. It covers the noisy SFT dataset-size sweep and the filtered-full SFT comparison.

The raw per-run data needed to reproduce the writeup tables and curves are
archived under `artifacts/experiments/ch4/sft_experiment/runs/`. Each run
directory contains `config.json`, `run_summary.json`, `metrics.jsonl`, and all
`eval_summary_step_*.json` files used by the plots.

| run | experiment | train examples | effective epochs | best answer accuracy | best step | final answer accuracy | final step | final format accuracy | source run summary |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `size128_bs16_micro4_lr5e-5` | noisy size sweep | 128 | 112.50 | 0.411379 | 200 | 0.314786 | 900 | 0.912160 | `.agents/logs/ch4/sft_noisy_size_sweep_bs16_lr5e-5/size128_bs16_micro4_lr5e-5/run_summary.json` |
| `size256_bs16_micro4_lr5e-5` | noisy size sweep | 256 | 56.25 | 0.422007 | 500 | 0.382620 | 900 | 0.976555 | `.agents/logs/ch4/sft_noisy_size_sweep_bs16_lr5e-5/size256_bs16_micro4_lr5e-5/run_summary.json` |
| `size512_bs16_micro4_lr5e-5` | noisy size sweep | 512 | 28.12 | 0.416380 | 900 | 0.416380 | 900 | 0.981557 | `.agents/logs/ch4/sft_noisy_size_sweep_bs16_lr5e-5/size512_bs16_micro4_lr5e-5/run_summary.json` |
| `size1024_bs16_micro4_lr5e-5` | noisy size sweep | 1024 | 14.06 | 0.378556 | 900 | 0.378556 | 900 | 0.972179 | `.agents/logs/ch4/sft_noisy_size_sweep_bs16_lr5e-5/size1024_bs16_micro4_lr5e-5/run_summary.json` |
| `sizefull_bs16_micro4_lr5e-5` | noisy size sweep | 4866 | 2.96 | 0.260706 | 700 | 0.220069 | 900 | 0.923726 | `.agents/logs/ch4/sft_noisy_size_sweep_bs16_lr5e-5/sizefull_bs16_micro4_lr5e-5/run_summary.json` |
| `filtered_full_bs16_micro4_lr5e-5` | filtered full | 4136 | 3.48 | 0.357924 | 800 | 0.351047 | 900 | 0.951860 | `.agents/logs/ch4/sft_noisy_filtered_bs16_lr5e-5/filtered_full_bs16_micro4_lr5e-5/run_summary.json` |

Additional archived artifacts:

- `artifacts/experiments/ch4/sft_experiment/sft_results_summary.csv`
- `artifacts/experiments/ch4/sft_experiment/sft_results_summary.md`
- `artifacts/experiments/ch4/sft_experiment/sft_result_table.csv`
- `artifacts/experiments/ch4/sft_experiment/sft_result_table.json`
- `artifacts/experiments/ch4/sft_experiment/sft_size_sweep_table.md`
- `artifacts/experiments/ch4/sft_experiment/sft_filtered_comparison_table.md`
- `artifacts/experiments/ch4/sft_experiment/run_summaries.json`
- `artifacts/experiments/ch4/sft_experiment/runs/`
