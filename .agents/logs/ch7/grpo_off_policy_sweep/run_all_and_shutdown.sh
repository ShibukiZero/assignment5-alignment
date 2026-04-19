#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$ROOT_DIR"

GRID=".agents/logs/ch7/grpo_off_policy_sweep/grid.json"
PHASE="${PHASE:-broad}"

COMMAND=(
  uv run python scripts/run_grpo_off_policy_sweep.py
  --grid "$GRID"
  --phase "$PHASE"
  --skip-existing
  --shutdown
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  COMMAND+=(--dry-run)
fi

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  COMMAND+=(--check-only)
fi

if [[ "${USE_SUDO_SHUTDOWN:-0}" == "1" ]]; then
  COMMAND+=(--sudo-shutdown)
fi

echo "Running from $ROOT_DIR"
echo "Grid: $GRID"
echo "Phase: $PHASE"
echo "${COMMAND[@]}"
"${COMMAND[@]}"
