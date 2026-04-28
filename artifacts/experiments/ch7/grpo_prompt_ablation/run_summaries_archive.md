# GRPO Prompt Ablation Run Archive

| run | prompt | reward | stop | best answer | best step | final answer | final format |
|---|---|---|---|---:|---:|---:|---:|
| `r1_zero_reference` | `cs336_alignment/prompts/r1_zero.prompt` | `r1_zero` | `</answer>` | 88.18% | 190 | 88.18% | 98.05% |
| `question_only` | `cs336_alignment/prompts/question_only.prompt` | `question_only` | `none` | 85.16% | 200 | 85.16% | 97.66% |

Raw run files are archived under `artifacts/experiments/ch7/grpo_prompt_ablation/runs/`. `sample_rollouts.jsonl` files are intentionally omitted because aggregate rollout summaries are sufficient for the writeup.
