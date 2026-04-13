# Part 1: Reasoning RL Assignment

## Problem `math_baseline`: Zero-Shot MATH Baseline (4 points)

### (a)
**Question:** Write a script that evaluates Qwen2.5-Math-1.5B zero-shot performance on the MATH validation set using the R1-Zero prompt, vLLM generation, and the provided MATH reward function.

**Deliverable:** A baseline evaluation script that loads validation examples, formats prompts, generates model outputs, computes metrics, and serializes examples, generations, and scores.

**Answer:** TODO.

### (b)
**Question:** Analyze generated outputs by grouping them into correct formatted answers, formatted but incorrect answers, and unformatted incorrect answers. Inspect examples from the failure categories and decide whether the issue is model behavior or parser behavior.

**Deliverable:** Counts for the requested reward categories plus commentary with examples from each relevant category.

**Answer:** TODO.

### (c)
**Question:** Report the zero-shot MATH performance of Qwen2.5-Math-1.5B.

**Deliverable:** A 1-2 sentence summary with evaluation metrics.

**Answer:** TODO.

---

## Problem `tokenize_prompt_and_output`: Prompt and Output Tokenization (2 points)

### Implementation
**Question:** Implement prompt/output tokenization by separately tokenizing prompts and responses, concatenating them, shifting labels, and constructing a response-token mask.

**Deliverable:** A method compatible with `adapters.run_tokenize_prompt_and_output` and the corresponding unit test.

**Answer:** TODO.

---

## Problem `compute_entropy`: Per-Token Entropy (1 point)

### Implementation
**Question:** Compute the per-token entropy of next-token predictions from model logits over the vocabulary dimension using a numerically stable method.

**Deliverable:** A method compatible with `adapters.run_compute_entropy` and the corresponding unit test.

**Answer:** TODO.

---

## Problem `get_response_log_probs`: Response Log-Probabilities and Entropy (2 points)

### Implementation
**Question:** Compute per-token conditional log-probabilities for labels given causal LM input IDs, optionally returning token entropy.

**Deliverable:** A method compatible with `adapters.run_get_response_log_probs` and the corresponding unit test.

**Answer:** TODO.

---

## Problem `masked_normalize`: Masked Normalize (1 point)

### Implementation
**Question:** Sum tensor values over masked positions and divide by a supplied normalization constant, optionally along a specified dimension.

**Deliverable:** A method compatible with `adapters.run_masked_normalize` and the corresponding unit test.

**Answer:** TODO.

---

## Problem `sft_microbatch_train_step`: SFT Microbatch Train Step (3 points)

### Implementation
**Question:** Implement a single SFT microbatch backward step using response-token log-probabilities, a response mask, normalization, and gradient accumulation scaling.

**Deliverable:** A method compatible with `adapters.run_sft_microbatch_train_step` and the corresponding unit test.

**Answer:** TODO.

---

## Problem `log_generations`: Logging Generations (1 point)

### Implementation
**Question:** Implement in-loop generation logging for prompts, model responses, ground-truth answers, reward information, entropy, and response-length statistics.

**Deliverable:** A reusable `log_generations` helper for SFT/RL experiments.

**Answer:** TODO.

---

## Problem `sft_experiment`: Run SFT on MATH (2 points, 2 H100 hrs)

### (1)
**Question:** Run SFT on reasoning traces using Qwen2.5-Math-1.5B, varying the number of unique SFT examples over `{128, 256, 512, 1024}` and the full dataset. Tune learning rate and batch size to reach at least 15% validation accuracy with the full dataset.

**Deliverable:** Validation accuracy curves for different dataset sizes.

**Answer:** TODO.

### (2)
**Question:** Filter the reasoning SFT examples to examples whose responses produce the correct answer, then train on the filtered dataset and compare against the unfiltered SFT run.

**Deliverable:** Filtered dataset size, validation accuracy curve, and comparison to the previous SFT experiment.

**Answer:** TODO.

---

## Problem `expert_iteration_experiment`: Run Expert Iteration on MATH (2 points, 6 H100 hrs)

### Experiment
**Question:** Run expert iteration with Qwen2.5-Math-1.5B on MATH train questions, varying rollout count, SFT epoch count, and batch size enough to draw conclusions. Use `n_ei_steps = 5`, stop rollouts at the answer tag, and log response entropy over training.

**Deliverable:** Validation accuracy curves for at least two rollout-count and epoch-count configurations, a model reaching at least 15% MATH validation accuracy, a short comparison to SFT, and an entropy plot.

**Answer:** TODO.

---

## Problem `compute_group_normalized_rewards`: Group Normalization (2 points)

### Implementation
**Question:** Compute raw rewards for rollout responses, group-normalize them within each prompt group, and optionally normalize by group standard deviation.

**Deliverable:** A method returning normalized rewards, raw rewards, and useful reward metadata.

**Answer:** TODO.

---

## Problem `compute_naive_policy_gradient_loss`: Naive Policy Gradient (1 point)

### Implementation
**Question:** Implement the per-token naive policy-gradient loss using raw rewards or advantages and policy log-probabilities.

**Deliverable:** A method compatible with the relevant adapter and unit test.

**Answer:** TODO.

---

## Problem `compute_grpo_clip_loss`: GRPO-Clip Loss (2 points)

### Implementation
**Question:** Implement the per-token GRPO-Clip loss using advantages, current policy log-probabilities, old policy log-probabilities, and the clip range.

**Deliverable:** A method returning per-token loss and metadata such as clipping statistics.

**Answer:** TODO.

---

## Problem `compute_policy_gradient_loss`: Policy-Gradient Wrapper (1 point)

### Implementation
**Question:** Implement a wrapper that dispatches to the requested policy-gradient loss type, including no-baseline, reinforce-with-baseline, and GRPO-Clip variants.

**Deliverable:** A method returning per-token loss and aggregated metadata.

**Answer:** TODO.

---

## Problem `masked_mean`: Masked Mean (1 point)

### Implementation
**Question:** Compute masked means over all elements or over a specified dimension while ignoring masked-out entries.

**Deliverable:** A method compatible with `adapters.run_masked_mean` and the corresponding unit test.

**Answer:** TODO.

---

## Problem `grpo_microbatch_train_step`: GRPO Microbatch Train Step (3 points)

### Implementation
**Question:** Implement one GRPO microbatch backward step, including policy-gradient loss selection, mask-based aggregation, normalization, metadata logging, and gradient-accumulation scaling.

**Deliverable:** A method compatible with `adapters.run_grpo_microbatch_train_step` and the corresponding unit test.

**Answer:** TODO.

---

## Problem `grpo_train_loop`: GRPO Train Loop (5 points)

### Implementation and Experiment
**Question:** Implement the complete GRPO training loop using R1-Zero prompting, vLLM rollouts, reward computation, policy updates, validation evaluation, and useful debugging logs.

**Deliverable:** A train loop, validation reward curve over steps, and example rollouts showing sensible behavior over time.

**Answer:** TODO.

---

## Problem `grpo_learning_rate`: Tune the Learning Rate (2 points, 6 H100 hrs)

### Experiment
**Question:** Starting from the suggested GRPO hyperparameters, sweep learning rates and report final validation answer rewards or divergence.

**Deliverable:** Validation reward curves for multiple learning rates, a model reaching at least 25% MATH validation accuracy, and a brief discussion of other logged metric trends.

**Answer:** TODO.

---

## Problem `grpo_baselines`: Effect of Baselining (2 points, 2 H100 hrs)

### Experiment
**Question:** Compare on-policy GRPO training with `no_baseline` and `reinforce_with_baseline`, using the tuned learning rate.

**Deliverable:** Validation reward curves for each loss type and a brief discussion of other logged metric trends.

**Answer:** TODO.

---

## Problem `think_about_length_normalization`: Think About Length Normalization (1 point)

### Written
**Question:** Compare masked mean aggregation with masked normalization by a fixed constant before running the empirical length-normalization experiment.

**Deliverable:** A conceptual comparison of the pros and cons of the two approaches and cases where one may be preferable.

**Answer:** TODO.

---

## Problem `grpo_length_normalization`: Effect of Length Normalization (2 points, 2 H100 hrs)

### Experiment
**Question:** Empirically compare GRPO training with masked mean aggregation versus fixed-constant masked normalization.

**Deliverable:** Validation answer reward curves and commentary on findings, including relevant stability metrics such as gradient norm.

**Answer:** TODO.

---

## Problem `grpo_group_standard_deviation`: Effect of Standard Deviation Normalization (2 points, 2 H100 hrs)

### Experiment
**Question:** Compare GRPO runs with group standard-deviation normalization enabled and disabled.

**Deliverable:** Validation answer reward curves and commentary on metric trends such as stability and gradient norm.

**Answer:** TODO.

---

## Problem `grpo_off_policy`: Implement Off-Policy GRPO

### Implementation
**Question:** Extend GRPO to support multiple epochs and optimizer updates per rollout batch, compute old log-probabilities once after rollout generation, and use GRPO-Clip for off-policy updates.

**Deliverable:** Off-policy GRPO training support.

**Answer:** TODO.

---

## Problem `grpo_off_policy_sweep`: Off-Policy GRPO Hyperparameter Sweep (4 points, 12 H100 hrs)

### Experiment
**Question:** With rollout batch size fixed to 256, sweep epochs per rollout batch and train batch size. Start with a broad short sweep, then run a focused 200-step sweep and compare against the on-policy configuration.

**Deliverable:** Experiment log explaining the sweep ranges, validation reward curves by validation step and wall-clock time, and commentary on trends such as entropy and response length.

**Answer:** TODO.

---

## Problem `grpo_off_policy_clip_ablation`: Off-Policy GRPO-Clip Ablation (2 points, 2 H100 hrs)

### Experiment
**Question:** Implement an unclipped off-policy loss type and compare it to GRPO-Clip using the best off-policy hyperparameters.

**Deliverable:** Validation reward curves and commentary comparing clipped versus unclipped training, including metrics such as entropy, response length, and gradient norm.

**Answer:** TODO.

---

## Problem `grpo_prompt_ablation`: Prompt Ablation (2 points, 2 H100 hrs)

### Experiment
**Question:** Compare GRPO training and validation using the R1-Zero prompt versus the question-only prompt, with the corresponding reward function for each.

**Deliverable:** Validation answer reward curves for both prompts and commentary on metrics such as entropy, response length, and gradient norm.

**Answer:** TODO.

---

## Problem `leaderboard`: Leaderboard (16 points, 16 H100 hrs)

### Final Experiment
**Question:** Obtain the highest validation reward possible within 4 hours of training on 2 H100 GPUs, using Qwen2.5-Math-1.5B and the allowed MATH train/validation setup.

**Deliverable:** Validation accuracy over the full MATH validation set, a screenshot of validation accuracy versus wall-clock time ending at no more than 4 hours, and confirmation that validation used R1-Zero prompting, temperature 1.0, max tokens 1024, and `r1_zero_reward_fn` answer rewards.

**Answer:** TODO.

---

# Part 2: Instruction Tuning and RLHF Assignment

## Problem `mmlu_baseline`: Zero-Shot MMLU Baseline (4 points)

### (a)
**Question:** Implement a parser that extracts the predicted multiple-choice letter from MMLU model outputs, returning `None` when parsing fails.

**Deliverable:** A parser compatible with `adapters.run_parse_mmlu_response` and the corresponding unit test.

**Answer:** TODO.

### (b)-(f)
**Question:** Evaluate Llama-3.1-8B zero-shot on MMLU, serialize generations and scores, report parse failures, throughput, accuracy, and an error analysis of random incorrect examples.

**Deliverable:** Evaluation script, metrics, throughput, parse-failure analysis, and 2-4 sentence error analysis.

**Answer:** TODO.

---

## Problem `gsm8k_baseline`: Zero-Shot GSM8K Baseline (4 points)

### (a)
**Question:** Implement a parser that extracts the final numeric answer from GSM8K model outputs, returning `None` when parsing fails.

**Deliverable:** A parser compatible with `adapters.run_parse_gsm8k_response` and the corresponding unit test.

**Answer:** TODO.

### (b)-(f)
**Question:** Evaluate Llama-3.1-8B zero-shot on GSM8K, serialize generations and scores, report parse failures, throughput, accuracy, and an error analysis of random incorrect examples.

**Deliverable:** Evaluation script, metrics, throughput, parse-failure analysis, and 2-4 sentence error analysis.

**Answer:** TODO.

---

## Problem `alpaca_eval_baseline`: Zero-Shot AlpacaEval Baseline (4 points)

### (a)-(d)
**Question:** Collect Llama-3.1-8B zero-shot predictions on AlpacaEval, serialize outputs in AlpacaEval-compatible JSON format, evaluate winrate with Llama-3.3-70B-Instruct as annotator, and inspect dispreferred examples.

**Deliverable:** Prediction-generation script, throughput estimate, winrate and length-controlled winrate, and a short error analysis.

**Answer:** TODO.

---

## Problem `sst_baseline`: Zero-Shot SimpleSafetyTests Baseline (4 points)

### (a)-(d)
**Question:** Collect Llama-3.1-8B zero-shot predictions on SimpleSafetyTests, serialize JSONL outputs, evaluate safe-output proportion with Llama-3.3-70B-Instruct as annotator, and inspect unsafe examples.

**Deliverable:** Prediction-generation script, throughput estimate, safe-output proportion, and a short error analysis.

**Answer:** TODO.

---

## Problem `look_at_sft`: Looking at Instruction-Tuning Data (4 points)

### Written
**Question:** Inspect ten random examples from the instruction-tuning training dataset and describe the represented task types and data quality.

**Deliverable:** A 2-4 sentence description with concrete examples.

**Answer:** TODO.

---

## Problem `data_loading`: Implement Data Loading (3 points)

### (a)
**Question:** Implement a PyTorch `Dataset` for packed instruction-tuning examples using an Alpaca-style prompt template, tokenizer, sequence length, and document shuffling option.

**Deliverable:** A dataset class compatible with `adapters.get_packed_sft_dataset` and the corresponding unit test.

**Answer:** TODO.

### (b)
**Question:** Implement batching over the dataset for a single epoch, with optional shuffling.

**Deliverable:** A batch iterator compatible with `adapters.run_iterate_batches` and the corresponding unit test.

**Answer:** TODO.

---

## Problem `sft_script`: Training Script for Instruction Tuning (4 points)

### Implementation
**Question:** Write a configurable instruction-tuning training loop for Llama-3.1-8B using gradient accumulation and periodic train/validation logging.

**Deliverable:** A training script with configurable model and optimizer hyperparameters, larger effective batch support, and logging.

**Answer:** TODO.

---

## Problem `sft`: Instruction Tuning (6 points, 24 H100 hrs)

### Experiment
**Question:** Fine-tune Llama-3.1-8B for one epoch on the provided instruction-tuning data, save the model and tokenizer, and report training setup and validation loss.

**Deliverable:** Training setup description, final validation loss, learning curve, and serialized model/tokenizer for later evaluation.

**Answer:** TODO.

---

## Problem `mmlu_sft`: MMLU After SFT (4 points)

### (a)-(c)
**Question:** Evaluate the instruction-tuned model on MMLU with the training prompt format, compare throughput and accuracy to the zero-shot baseline, and inspect incorrect examples.

**Deliverable:** Throughput comparison, accuracy comparison, and 2-4 sentence error analysis.

**Answer:** TODO.

---

## Problem `gsm8k_sft`: GSM8K After SFT (4 points)

### (a)-(c)
**Question:** Evaluate the instruction-tuned model on GSM8K with the training prompt format, compare throughput and accuracy to the zero-shot baseline, and inspect incorrect examples.

**Deliverable:** Throughput comparison, accuracy comparison, and 2-4 sentence error analysis.

**Answer:** TODO.

---

## Problem `alpaca_eval_sft`: AlpacaEval After SFT (4 points)

### (a)-(c)
**Question:** Collect AlpacaEval predictions from the instruction-tuned model, evaluate with Llama-3.3-70B-Instruct as annotator, compare to the zero-shot baseline, and inspect dispreferred examples.

**Deliverable:** Throughput estimate, winrate and length-controlled winrate comparison, and 2-4 sentence error analysis.

**Answer:** TODO.

---

## Problem `sst_sft`: SimpleSafetyTests After SFT (4 points)

### (a)-(c)
**Question:** Collect SimpleSafetyTests predictions from the instruction-tuned model, evaluate safe-output proportion with Llama-3.3-70B-Instruct as annotator, compare to baseline, and inspect unsafe examples.

**Deliverable:** Throughput estimate, safe-output comparison, and 2-4 sentence error analysis.

**Answer:** TODO.

---

## Problem `red_teaming`: Red-Teaming the Instruction-Tuned Model (4 points)

### (a)
**Question:** Name three possible misuse cases for language models beyond the examples in the handout.

**Deliverable:** A 1-3 sentence list or description.

**Answer:** TODO.

### (b)
**Question:** Try prompting the fine-tuned model for three different potentially malicious applications and describe the methodology and observed results.

**Deliverable:** For each application, a 2-4 sentence description of the red-teaming procedure and result.

**Answer:** TODO.

---

## Problem `look_at_hh`: Looking at HH Preference Data (2 points)

### (1)
**Question:** Load the Anthropic HH datasets, combine the four training files, remove multi-turn conversations, split each example into instruction, chosen response, rejected response, and source file.

**Deliverable:** A Python data-loading function suitable for DPO training.

**Answer:** TODO.

### (2)
**Question:** Inspect random helpful and harmless preference examples and compare chosen versus rejected responses.

**Deliverable:** Commentary on differences between chosen and rejected responses and whether the annotator choices seem reasonable.

**Answer:** TODO.

---

## Problem `dpo_loss`: DPO Loss (2 points)

### Implementation
**Question:** Implement the per-instance DPO loss for a trainable policy, reference policy, tokenizer, prompt, preferred response, rejected response, and beta.

**Deliverable:** A method compatible with `adapters.per_instance_dpo` and the corresponding unit test.

**Answer:** TODO.

---

## Problem `dpo_training`: DPO Training (4 points)

### (1)
**Question:** Train the instruction-tuned Llama model with DPO on HH for one epoch, using a reference model and validation classification accuracy tracking.

**Deliverable:** A DPO training script and screenshot of the validation accuracy curve.

**Answer:** TODO.

### (2)
**Question:** Evaluate the DPO-trained model on AlpacaEval and compare to the SFT model.

**Deliverable:** A 1-2 sentence report with winrate and length-controlled winrate.

**Answer:** TODO.

### (3)
**Question:** Evaluate the DPO-trained model on SimpleSafetyTests and compare to the SFT model.

**Deliverable:** A 1-2 sentence safety evaluation summary.

**Answer:** TODO.

### (4)
**Question:** Evaluate the DPO-trained model on GSM8K and MMLU to check for alignment tax.

**Deliverable:** A 2-3 sentence report with GSM8K and MMLU evaluations.

**Answer:** TODO.
