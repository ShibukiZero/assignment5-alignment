# GRPO Off-Policy Clip Ablation Experiment Log

## Setup

- Both runs use the same true off-policy configuration from the previous sweep: `epochs_per_rollout_batch = 2`, `train_batch_size = 256`, `rollout_batch_size = 256`, `learning_rate = 4e-5`, `loss_normalization = masked_mean`, and `use_std_normalization = false`.
- The ablation changes only the policy-gradient loss: `GRPO-Clip` versus `GRPO-No-Clip`.
- Both runs were evaluated every 5 GRPO steps on 1024 validation examples with the same sampling settings.

## Findings

- `GRPO-Clip (e2/tb256)` peaked at 34.38% on step 65, then entered a late collapse and finished at 0.00%.
- `GRPO-No-Clip (e2/tb256)` peaked at 78.42% on step 195 and remained stable through the final evaluation at 78.22%.
- The clipped run ended with rollout length 1024.0 and zero format / answer reward, while the unclipped run ended with rollout length 491.9 and 94.53% final format accuracy.
- Tail-train diagnostics for the clipped run show NaN loss, NaN gradient norm, and `clip_fraction = 1.0`, whereas the unclipped run keeps finite entropy and gradient norm values.
