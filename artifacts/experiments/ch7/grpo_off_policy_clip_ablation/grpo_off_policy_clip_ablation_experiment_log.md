# GRPO Off-Policy Clip Ablation Experiment Log

## Setup

- All runs use the same true off-policy `e2/tb256` configuration: `epochs_per_rollout_batch = 2`, `train_batch_size = 256`, `rollout_batch_size = 256`, `learning_rate = 4e-5`, `loss_normalization = masked_mean`, and `use_std_normalization = true`.
- The ablation compares symmetric GRPO-Clip with bounds `[0.8, 1.2]`, unclipped GRPO, and asymmetric GRPO-Clip with bounds `[0.8, 1.28]`.
- All runs were evaluated every 5 GRPO steps on 1024 validation examples with the same sampling settings.

## Findings

- `GRPO-Clip 0.20/0.20 (e2/tb256)` peaked at 86.33% on step 145 and finished at 82.32%.
- `GRPO-No-Clip (e2/tb256)` peaked at 24.61% on step 125 and finished at 0.00%.
- `GRPO-Clip 0.20/0.28 (e2/tb256)` peaked at 83.79% on step 150 and finished at 78.91%.
- Final rollout lengths were 474.8 for symmetric clipping, 155.8 for no clipping, and 587.0 for asymmetric clipping.
- The summary CSV and diagnostics CSV include `variant`, `cliprange_low`, and `cliprange_high` so the symmetric and asymmetric clipped objectives are distinguishable even though both use `loss_type = grpo_clip`.
