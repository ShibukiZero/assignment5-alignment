# GRPO Off-Policy Sweep Experiment Log

## Setup

- All runs use a single GPU with `policy_device = cuda:0` and `vllm_device = cuda:0`.
- Fixed `rollout_batch_size = 256`, `group_size = 8`, and `learning_rate = 4e-5`.
- Fixed `loss_type = grpo_clip`, `loss_normalization = masked_mean`, and `use_std_normalization = true`.
- The broad sweep varied `epochs_per_rollout_batch` and `train_batch_size` to change the number and character of optimizer updates per rollout batch.
- The focused sweep extended `e2/tb256`, `e2/tb128`, and `e4/tb256` to 200 GRPO steps.
- The on-policy reference is `on_policy_reference_e1_tb256_std_norm`, using the matched `masked_mean`, `use_std_normalization = true`, `lr = 4e-5` configuration.

## Broad Findings

- `broad_e2_tb128` was the strongest broad run, peaking at 83.20% on step 35 and finishing at 81.54%.
- Moderate off-policy reuse worked best. The best broad setting used two epochs and `train_batch_size = 128`, for four optimizer updates per rollout batch.
- Smaller train batches under `e2` remained usable, but their 40-step rewards were lower than `e2/tb128`.
- Repeating the same full rollout batch for many epochs was unstable.
- The aggressive full-batch settings `broad_e8_tb256`, `broad_e16_tb256` collapsed to zero validation reward by the end of the broad sweep.

## Focused Outcome

- Among off-policy focused runs, `focused_e4_tb256` had the highest peak, reaching 86.43% on step 145.
- Among off-policy focused runs, `focused_e2_tb128` was the most stable by final validation answer reward, ending at 85.64%.
- The on-policy reference peaked at 88.18% and finished at 88.18%.
- The off-policy focused peaks are close to each other, but none exceeds the matched on-policy reference.
- Token entropy and rollout response length are plotted for both broad and focused runs because the unstable settings show clear entropy/length pathologies near collapse.
