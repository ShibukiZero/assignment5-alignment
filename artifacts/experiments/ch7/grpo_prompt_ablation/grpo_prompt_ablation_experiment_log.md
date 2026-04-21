# GRPO Prompt Ablation Experiment Log

## Setup

- We compare two prompt-and-evaluation contracts under the same on-policy training budget (`n_grpo_steps = 200`, `rollout_batch_size = 256`, `train_batch_size = 256`, `learning_rate = 4e-5`, `group_size = 8`, and `use_std_normalization = false`).
- The reference run uses `cs336_alignment/prompts/r1_zero.prompt` with `r1_zero_reward_fn` and the standard `</answer>` stop condition.
- The ablation run uses `cs336_alignment/prompts/question_only.prompt` with `question_only_reward_fn` and no `</answer>` stop condition.

## Findings

- The question-only contract starts much stronger at step 0, reaching 56.74% before any RL updates.
- The question-only run peaks at 82.81% on step 155 and finishes at 80.27%.
- The R1-Zero reference peaks at 80.08% and finishes at 77.64%.
- The question-only run also maintains high final format accuracy (96.58%), while ending with a longer average rollout response length (475.8 vs. 332.7).
- Since the prompt, reward function, and stop contract all change together, this should be interpreted as a comparison between two prompt-and-evaluation contracts rather than a pure prompt-wording ablation.
