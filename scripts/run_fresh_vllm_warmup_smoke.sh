#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-/root/autodl-tmp/a5-alignment/runs/fresh_vllm_warmup_smoke}"
LOG_ROOT="${2:-.agents/logs/smoke/fresh_vllm_warmup_smoke}"
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

CS336_FRESH_VLLM_WARMUP=1 \
uv run scripts/sft_experiment.py \
  --output-dir "${OUTPUT_ROOT}/sft_filtered_full_smoke" \
  --log-dir "${LOG_ROOT}/sft_filtered_full_smoke" \
  --train-path /root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric_noisy/filtered_sft.jsonl \
  --val-path /root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric_noisy/validation.jsonl \
  --policy-device "${GPU_DEVICE}" \
  --vllm-device "${GPU_DEVICE}" \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --train-batch-size 16 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-5 \
  --max-steps 100 \
  --eval-every 100

CS336_FRESH_VLLM_WARMUP=1 \
uv run scripts/expert_iteration_experiment.py \
  --output-dir "${OUTPUT_ROOT}/ei_db2048_g4_epoch3_smoke" \
  --log-dir "${LOG_ROOT}/ei_db2048_g4_epoch3_smoke" \
  --train-path /root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric_noisy/train.jsonl \
  --val-path /root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric_noisy/validation.jsonl \
  --policy-device "${GPU_DEVICE}" \
  --vllm-device "${GPU_DEVICE}" \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --n-ei-steps 2 \
  --rollout-batch-size 2048 \
  --rollouts-per-question 4 \
  --sft-epochs-per-step 3 \
  --train-batch-size 16 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-5

CS336_FRESH_VLLM_WARMUP=1 \
uv run scripts/grpo_experiment.py \
  --output-dir "${OUTPUT_ROOT}/grpo_on_policy_no_std_smoke" \
  --log-dir "${LOG_ROOT}/grpo_on_policy_no_std_smoke" \
  --train-path /root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric/train.jsonl \
  --val-path /root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric/validation.jsonl \
  --policy-device "${GPU_DEVICE}" \
  --vllm-device "${GPU_DEVICE}" \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --n-grpo-steps 20 \
  --learning-rate 4e-5 \
  --rollout-batch-size 256 \
  --group-size 8 \
  --epochs-per-rollout-batch 1 \
  --train-batch-size 256 \
  --gradient-accumulation-steps 128 \
  --loss-type reinforce_with_baseline \
  --no-use-std-normalization \
  --eval-every 5 \
  --eval-max-examples 1024 \
  --save-checkpoints
