#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

USE_SUDO_SHUTDOWN=0
NO_SHUTDOWN=0
DRY_RUN=0
GRPO_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_grpo_and_shutdown.sh [wrapper options] -- [grpo_experiment.py args]
  bash scripts/run_grpo_and_shutdown.sh [wrapper options] [grpo_experiment.py args]

Wrapper options:
  --sudo-shutdown   Use sudo shutdown -h now instead of shutdown -h now
  --no-shutdown     Verify outputs but do not power off
  --dry-run         Print the parsed command and verification targets only
  -h, --help        Show this help message

Examples:
  bash scripts/run_grpo_and_shutdown.sh \
    --output-dir /root/autodl-tmp/a5-alignment/runs/grpo_prompt_ablation/question_only_e1_tb256 \
    --log-dir .agents/logs/ch7/grpo_prompt_ablation/question_only_e1_tb256 \
    --model /root/autodl-tmp/a5-alignment/models/Qwen2.5-Math-1.5B \
    --train-path /root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric/train.jsonl \
    --val-path /root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric/validation.jsonl \
    --prompt-template cs336_alignment/prompts/question_only.prompt \
    --reward-fn question_only \
    --n-grpo-steps 200 \
    --learning-rate 4e-5 \
    --advantage-eps 1e-6 \
    --rollout-batch-size 256 \
    --group-size 8 \
    --sampling-temperature 1.0 \
    --sampling-top-p 1.0 \
    --min-new-tokens 4 \
    --max-new-tokens 1024 \
    --epochs-per-rollout-batch 1 \
    --train-batch-size 256 \
    --gradient-accumulation-steps 128 \
    --loss-type grpo_clip \
    --cliprange 0.2 \
    --loss-normalization masked_mean \
    --loss-normalize-constant 1.0 \
    --no-use-std-normalization \
    --eval-every 5 \
    --eval-max-examples 1024 \
    --eval-temperature 1.0 \
    --eval-top-p 1.0 \
    --sample-rollouts-to-log 8 \
    --seed 0 \
    --policy-device cuda:0 \
    --vllm-device cuda:1 \
    --vllm-gpu-memory-utilization 0.85 \
    --save-checkpoints
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sudo-shutdown)
      USE_SUDO_SHUTDOWN=1
      shift
      ;;
    --no-shutdown)
      NO_SHUTDOWN=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        GRPO_ARGS+=("$1")
        shift
      done
      break
      ;;
    *)
      GRPO_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#GRPO_ARGS[@]} -eq 0 ]]; then
  echo "No GRPO arguments supplied."
  usage
  exit 2
fi

OUTPUT_DIR=""
LOG_DIR=""
SAVE_CHECKPOINTS=0
N_GRPO_STEPS=""
ROLLOUT_BATCH_SIZE=""
TRAIN_BATCH_SIZE=""
EPOCHS_PER_ROLLOUT_BATCH=""

for ((i=0; i<${#GRPO_ARGS[@]}; i++)); do
  arg="${GRPO_ARGS[$i]}"
  next=""
  if (( i + 1 < ${#GRPO_ARGS[@]} )); then
    next="${GRPO_ARGS[$((i + 1))]}"
  fi
  case "$arg" in
    --output-dir)
      OUTPUT_DIR="$next"
      ;;
    --output-dir=*)
      OUTPUT_DIR="${arg#*=}"
      ;;
    --log-dir)
      LOG_DIR="$next"
      ;;
    --log-dir=*)
      LOG_DIR="${arg#*=}"
      ;;
    --n-grpo-steps)
      N_GRPO_STEPS="$next"
      ;;
    --n-grpo-steps=*)
      N_GRPO_STEPS="${arg#*=}"
      ;;
    --rollout-batch-size)
      ROLLOUT_BATCH_SIZE="$next"
      ;;
    --rollout-batch-size=*)
      ROLLOUT_BATCH_SIZE="${arg#*=}"
      ;;
    --train-batch-size)
      TRAIN_BATCH_SIZE="$next"
      ;;
    --train-batch-size=*)
      TRAIN_BATCH_SIZE="${arg#*=}"
      ;;
    --epochs-per-rollout-batch)
      EPOCHS_PER_ROLLOUT_BATCH="$next"
      ;;
    --epochs-per-rollout-batch=*)
      EPOCHS_PER_ROLLOUT_BATCH="${arg#*=}"
      ;;
    --save-checkpoints)
      SAVE_CHECKPOINTS=1
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" || -z "$LOG_DIR" ]]; then
  echo "Both --output-dir and --log-dir are required so the wrapper can verify outputs."
  exit 2
fi

CONFIG_PATH="$LOG_DIR/config.json"
METRICS_PATH="$LOG_DIR/metrics.jsonl"
RUN_SUMMARY_PATH="$LOG_DIR/run_summary.json"
STDOUT_STDERR_PATH="$LOG_DIR/stdout_stderr.log"
FINAL_POLICY_DIR="$OUTPUT_DIR/final_policy"

expected_optimizer_steps() {
  if [[ -z "$N_GRPO_STEPS" || -z "$ROLLOUT_BATCH_SIZE" || -z "$TRAIN_BATCH_SIZE" || -z "$EPOCHS_PER_ROLLOUT_BATCH" ]]; then
    return 1
  fi
  echo $(( N_GRPO_STEPS * (ROLLOUT_BATCH_SIZE / TRAIN_BATCH_SIZE) * EPOCHS_PER_ROLLOUT_BATCH ))
}

extract_final_optimizer_step() {
  local summary_path="$1"
  grep -o '"final_optimizer_step"[[:space:]]*:[[:space:]]*[0-9]\+' "$summary_path" \
    | grep -o '[0-9]\+' \
    | head -n 1
}

count_train_rows() {
  local metrics_path="$1"
  grep -c '"type"[[:space:]]*:[[:space:]]*"train"' "$metrics_path"
}

verify_outputs() {
  local missing=0

  for required_file in "$CONFIG_PATH" "$METRICS_PATH" "$RUN_SUMMARY_PATH"; do
    if [[ ! -s "$required_file" ]]; then
      echo "Missing or empty file: $required_file"
      missing=1
    fi
  done

  if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Missing output directory: $OUTPUT_DIR"
    missing=1
  fi

  if [[ "$SAVE_CHECKPOINTS" == "1" && ! -d "$FINAL_POLICY_DIR" ]]; then
    echo "Missing final checkpoint directory: $FINAL_POLICY_DIR"
    missing=1
  fi

  if [[ "$missing" != "0" ]]; then
    return 1
  fi

  local expected_steps
  if expected_steps="$(expected_optimizer_steps)"; then
    local observed_steps
    observed_steps="$(extract_final_optimizer_step "$RUN_SUMMARY_PATH")"
    if [[ -z "$observed_steps" || "$observed_steps" != "$expected_steps" ]]; then
      echo "Unexpected final_optimizer_step: got '${observed_steps:-missing}', expected '$expected_steps'"
      return 1
    fi

    local train_rows
    train_rows="$(count_train_rows "$METRICS_PATH")"
    if [[ "$train_rows" != "$expected_steps" ]]; then
      echo "Unexpected train row count: got '$train_rows', expected '$expected_steps'"
      return 1
    fi
  fi

  echo "Verified outputs:"
  echo "  $CONFIG_PATH"
  echo "  $METRICS_PATH"
  echo "  $RUN_SUMMARY_PATH"
  if [[ "$SAVE_CHECKPOINTS" == "1" ]]; then
    echo "  $FINAL_POLICY_DIR"
  fi
}

check_for_partial_outputs() {
  if verify_outputs >/dev/null 2>&1; then
    echo "Run already completed and verified."
    return 0
  fi

  if [[ -e "$OUTPUT_DIR" || -e "$LOG_DIR" ]]; then
    echo "Refusing to continue because partial outputs already exist."
    echo "Output dir: $OUTPUT_DIR"
    echo "Log dir: $LOG_DIR"
    echo "Move or remove them before rerunning."
    return 1
  fi

  return 2
}

COMMAND=(uv run python scripts/grpo_experiment.py "${GRPO_ARGS[@]}")

echo "Repository root: $ROOT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Log dir: $LOG_DIR"
echo "Command:"
printf '  %q' "${COMMAND[@]}"
printf '\n'

partial_status=0
if check_for_partial_outputs; then
  partial_status=0
else
  partial_status=$?
fi

if [[ "$partial_status" == "1" ]]; then
  exit 3
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only; not launching training."
  exit 0
fi

if [[ "$partial_status" == "0" ]]; then
  echo "Skipping launch because outputs are already verified."
else
  mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
  "${COMMAND[@]}" > >(tee "$STDOUT_STDERR_PATH") 2>&1
fi

echo "Verifying output files..."
if ! verify_outputs; then
  echo "Output verification failed. Not shutting down."
  exit 4
fi

echo "Flushing filesystem buffers..."
sync
echo "sync complete."

if [[ "$NO_SHUTDOWN" == "1" ]]; then
  echo "--no-shutdown set; leaving the machine running."
  exit 0
fi

if [[ "$USE_SUDO_SHUTDOWN" == "1" ]]; then
  echo "Powering off with sudo shutdown -h now..."
  sudo shutdown -h now
else
  echo "Powering off with shutdown -h now..."
  shutdown -h now
fi
