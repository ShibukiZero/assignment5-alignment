# GRPO Train Loop Experiment Log

## Setup

- Run label: `grpo_on_policy_lr1e-5`.
- Learning rate: `1e-05`.
- GRPO steps: `200`.
- Rollout batch size: `256` with group size `8`.
- Loss type: `reinforce_with_baseline` and std normalization set to `True`.

## Outcome

- Validation answer reward improved from 3.81% at step 0 to a best of 67.77% at step 195.
- The run finished at 67.29% validation answer reward and 95.61% validation format accuracy.
- Final rollout answer reward was 60.16% with average response length 207.2 tokens.
- Final train token entropy was 0.1335.
