# Ch5 Expert Iteration Run Summaries

This archive records the run summary data used by the Ch5 `expert_iteration_experiment` writeup section. It covers the four initial `D_b=512` runs and the four supplement runs used in the final table.

The raw per-run data needed to reproduce the writeup tables and curves are
archived under `artifacts/experiments/ch5/expert_iteration/runs/`. Each run
directory contains `config.json`, `run_summary.json`, `metrics.jsonl`, all
`eval_summary_ei_step_*.json` files, and each EI step's `rollout_summary.json`.

| run | rollout batch | G | SFT epochs | best answer accuracy | best EI step | final answer accuracy | final EI step | final rollout entropy | source run summary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `db512_g2_epoch1` | 512 | 2 | 1 | 0.281651 | 5 | 0.281651 | 5 | 1.251425 | `.agents/logs/ch5/ei_first_grid_db512_lr5e-5_bs16/db512_g2_epoch1/run_summary.json` |
| `db512_g2_epoch2` | 512 | 2 | 2 | 0.338543 | 5 | 0.338543 | 5 | 0.674401 | `.agents/logs/ch5/ei_first_grid_db512_lr5e-5_bs16/db512_g2_epoch2/run_summary.json` |
| `db512_g4_epoch1` | 512 | 4 | 1 | 0.256330 | 4 | 0.226008 | 5 | 1.368774 | `.agents/logs/ch5/ei_first_grid_db512_lr5e-5_bs16/db512_g4_epoch1/run_summary.json` |
| `db512_g4_epoch2` | 512 | 4 | 2 | 0.359175 | 4 | 0.353548 | 5 | 0.796369 | `.agents/logs/ch5/ei_first_grid_db512_lr5e-5_bs16/db512_g4_epoch2/run_summary.json` |
| `db1024_g4_epoch2` | 1024 | 4 | 2 | 0.331354 | 5 | 0.331354 | 5 | 0.679569 | `.agents/logs/ch5/ei_supplement_db1024_2048_budget4_lr5e-5_bs16/db1024_g4_epoch2/run_summary.json` |
| `db1024_g8_epoch2` | 1024 | 8 | 2 | 0.323539 | 2 | 0.278525 | 5 | 0.973193 | `.agents/logs/ch5/ei_supplement_db1024_2048_budget4_lr5e-5_bs16/db1024_g8_epoch2/run_summary.json` |
| `db2048_g4_epoch2` | 2048 | 4 | 2 | 0.346358 | 4 | 0.313535 | 5 | 0.672091 | `.agents/logs/ch5/ei_supplement_db1024_2048_budget4_lr5e-5_bs16/db2048_g4_epoch2/run_summary.json` |
| `db2048_g4_epoch3` | 2048 | 4 | 3 | 0.409503 | 3 | 0.405439 | 5 | 0.364994 | `.agents/logs/ch5/ei_supplement_db1024_2048_budget4_lr5e-5_bs16/db2048_g4_epoch3/run_summary.json` |

Additional archived artifacts:

- `artifacts/experiments/ch5/expert_iteration/ei_results_summary.csv`
- `artifacts/experiments/ch5/expert_iteration/ei_results_summary.md`
- `artifacts/experiments/ch5/expert_iteration/ei_validation_accuracy.svg`
- `artifacts/experiments/ch5/expert_iteration/ei_rollout_entropy.svg`
- `artifacts/experiments/ch5/expert_iteration/run_summaries.json`
- `artifacts/experiments/ch5/expert_iteration/runs/`
