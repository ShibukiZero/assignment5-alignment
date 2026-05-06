# GRPO Prompt Ablation Experiment Log

## Setup

- We compare the R1-Zero on-policy reference line used in the off-policy sweep with the prefix-cache-repaired question-only rerun.
- Both runs use a 200-step GRPO budget with `rollout_batch_size = 256`, `train_batch_size = 256`, `learning_rate = 4e-5`, `group_size = 8`, `epochs_per_rollout_batch = 1`, and `use_std_normalization = true`.
- The reference run uses `cs336_alignment/prompts/r1_zero.prompt` with `r1_zero_reward_fn` and the standard `</answer>` stop condition.
- The ablation run uses `cs336_alignment/prompts/question_only.prompt` with `question_only_reward_fn` and no `</answer>` stop condition.
- The reference uses `reinforce_with_baseline`; the question-only rerun uses `grpo_clip`, so this remains a prompt-and-evaluation contract comparison rather than a pure prompt-wording ablation.

## Findings

- The question-only contract starts much stronger at step 0, reaching 56.74% before any RL updates.
- The question-only run peaks at 85.16% on step 200 and finishes at 85.16%.
- The R1-Zero reference peaks at 88.18% and finishes at 88.18%.
- The question-only run also maintains high final format accuracy (97.66%), while ending with a longer average rollout response length (460.5 vs. 466.9).
- Since the prompt, reward function, and stop contract all change together, this should be interpreted as a comparison between two prompt-and-evaluation contracts rather than a pure prompt-wording ablation.
