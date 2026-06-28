# Leaderboard Run Archive

Source run: `/workspace/a5-alignment/runs/leaderboard/log`

Archived lightweight run files under `/workspace/a5-alignment/artifacts_new/runs/final_on_policy_lr4e-5_400_full_eval`.

The archive intentionally copies only `config.json`, `run_summary.json`, `metrics.jsonl`, validation summaries, and per-step `rollout_summary.json` files.
Checkpoints, model weights, optimizer state, scheduler state, and sample rollout dumps are intentionally omitted.

| run | validation examples | best answer | best step | final answer | final step | final rollout answer | final avg length |
|---|---:|---:|---:|---:|---:|---:|---:|
| `final_on_policy_lr4e-5_400_full_eval` | 3199 | 73.80% | 240 | 73.55% | 360 | 74.22% | 516.4 |
