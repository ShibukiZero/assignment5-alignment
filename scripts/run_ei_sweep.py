#!/usr/bin/env python3
"""Run Expert Iteration experiment sweeps from a JSON config."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Expert Iteration sweeps.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an EI sweep JSON config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop the sweep when one run fails.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def append_arg(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag_name(key))
        return
    command.extend([flag_name(key), str(value)])


def default_run_name(run_config: dict[str, Any], index: int) -> str:
    rollout_batch_size = run_config.get("rollout_batch_size", "common")
    rollouts_per_question = run_config.get("rollouts_per_question", "common")
    sft_epochs = run_config.get("sft_epochs_per_step", "common")
    return f"run{index:02d}_db{rollout_batch_size}_g{rollouts_per_question}_epochs{sft_epochs}"


def build_command(
    config: dict[str, Any],
    run_config: dict[str, Any],
    run_dir: Path,
    log_dir: Path,
) -> list[str]:
    python_command = config.get("python", sys.executable)
    if isinstance(python_command, list):
        command = [str(part) for part in python_command]
    else:
        command = [str(python_command)]
    command.append(config.get("script", "scripts/expert_iteration_experiment.py"))

    for key, value in config.get("common_args", {}).items():
        append_arg(command, key, value)

    append_arg(command, "output_dir", run_dir)
    append_arg(command, "log_dir", log_dir)

    for key, value in run_config.items():
        if key == "name":
            continue
        append_arg(command, key, value)

    if config.get("save_checkpoints", False):
        command.append("--save-checkpoints")

    return command


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = Path(config["output_root"])
    log_root = Path(config["log_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    runs = config.get("runs")
    if not runs:
        raise ValueError("EI sweep config must contain a non-empty `runs` list.")

    sweep_records: list[dict[str, Any]] = []
    for index, run_config in enumerate(runs, start=1):
        name = run_config.get("name", default_run_name(run_config, index))
        run_dir = output_root / name
        run_log_dir = log_root / name
        run_log_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(
            config=config,
            run_config=run_config,
            run_dir=run_dir,
            log_dir=run_log_dir,
        )
        record = {
            "run_name": name,
            **run_config,
            "output_dir": str(run_dir),
            "log_dir": str(run_log_dir),
            "command": command,
        }
        sweep_records.append(record)
        print(" ".join(command), flush=True)

        if args.dry_run:
            continue

        start = time.time()
        stdout_path = run_log_dir / "stdout_stderr.log"
        with stdout_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        record["returncode"] = completed.returncode
        record["seconds"] = time.time() - start
        record["stdout_stderr"] = str(stdout_path)

        summary_path = run_log_dir / "run_summary.json"
        summary_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if completed.returncode != 0 and args.stop_on_failure:
            break

    (log_root / "sweep_manifest.json").write_text(
        json.dumps(sweep_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
