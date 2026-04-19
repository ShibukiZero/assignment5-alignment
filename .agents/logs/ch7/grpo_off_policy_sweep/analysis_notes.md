# Off-Policy GRPO Notes

## Requirement Summary

- Implement multi-update GRPO on a single rollout batch.
- Cache `old_log_probs` immediately after rollout generation and before gradient updates.
- Use `GRPO-Clip` for off-policy training.
- Sweep `epochs_per_rollout_batch` and `train_batch_size` with `rollout_batch_size=256`.
- Keep memory roughly constant by changing `gradient_accumulation_steps` together with `train_batch_size`.
- Compare the final curves to the on-policy setting with `epochs_per_rollout_batch=1` and `train_batch_size=256`.

## Implementation Notes

- `scripts/grpo_experiment.py` now tokenizes the full rollout batch once and reuses those padded tensors for both old-log-prob caching and training. This avoids shape mismatch between cached `old_log_probs` and current policy log-probs.
- For `loss_type=grpo_clip`, `old_log_probs` are computed under `torch.inference_mode()` before entering the inner update loop.
- The inner loop shuffles the rollout indices once per rollout epoch, splits them into `train_batch_size` chunks, and applies gradient accumulation inside each train batch.
- Logged train metrics now include `rollout_epoch`, `train_batch_index`, `num_train_batches`, `clip_fraction`, `approx_kl`, and whether `old_log_probs` were cached.
- `run_summary.json` records the expected optimizer-step count so the sweep runner can verify complete landing.

## Grid Rationale

- Use the best current on-policy settings from the previous ablations: `learning_rate=4e-5`, `masked_mean`, and `use_std_normalization=False`.
- Broad sweep uses 40 GRPO steps to stay under the assignment's `<50` requirement and covers 1, 2, 4, and 8 optimizer updates per rollout batch.
- Focused sweep uses 200 GRPO steps for three pre-registered candidates: 2 updates per rollout, 4 updates via smaller train batches, and 4 updates via more epochs.
- Every real sweep run keeps `rollout_batch_size=256`; the smoke run intentionally uses a smaller rollout batch and is only an infra check.
