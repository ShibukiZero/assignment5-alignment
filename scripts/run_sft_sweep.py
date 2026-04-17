#!/usr/bin/env python3
"""Run an SFT hyperparameter sweep from a JSON config."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT batch-size/LR sweep.")
    parser.add_argument(
        "--config",
        default=".agents/logs/ch4/sft_capacity_probe/config.json",
        help="Path to the sweep JSON config.",
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


RUN_NAME_KEY_ALIASES = {
    "train_batch_size": "bs",
    "gradient_accumulation_steps": "ga",
    "learning_rate": "lr",
    "num_train_examples": "n",
    "seed": "seed",
}
RUN_NAME_KEY_ORDER = {
    "train_batch_size": 0,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2,
    "num_train_examples": 3,
    "seed": 4,
}


def format_name_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    text = text.replace("+", "")
    text = text.replace(".", "p")
    return re.sub(r"[^A-Za-z0-9-]+", "_", text).strip("_")


def default_run_name(run_config: dict[str, Any], config: dict[str, Any]) -> str:
    common_args = dict(config.get("common_args", {}))
    batch_size = int(run_config["train_batch_size"])
    grad_by_bs = config.get("gradient_accumulation_steps_by_batch_size", {})
    grad_steps = run_config.get(
        "gradient_accumulation_steps",
        grad_by_bs.get(str(batch_size), common_args.get("gradient_accumulation_steps")),
    )

    name_params: dict[str, Any] = {
        "train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_steps,
        "learning_rate": run_config["learning_rate"],
        "seed": run_config.get("seed", common_args.get("seed", 0)),
    }
    for key, value in run_config.items():
        if key in {"name", "train_batch_size", "gradient_accumulation_steps", "learning_rate"}:
            continue
        name_params[key] = value

    parts: list[str] = []
    for key in sorted(name_params, key=lambda item: (RUN_NAME_KEY_ORDER.get(item, 100), item)):
        value = name_params[key]
        if value is None:
            continue
        label = RUN_NAME_KEY_ALIASES.get(key, key)
        parts.append(f"{label}{format_name_value(value)}")
    return "_".join(parts)


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
    command.append(config["script"])

    common_args = dict(config.get("common_args", {}))
    batch_size = int(run_config["train_batch_size"])
    grad_by_bs = config.get("gradient_accumulation_steps_by_batch_size", {})
    grad_steps = run_config.get(
        "gradient_accumulation_steps",
        grad_by_bs.get(str(batch_size), common_args.get("gradient_accumulation_steps")),
    )
    if grad_steps is None:
        raise ValueError(
            "Missing gradient accumulation steps. Set common_args.gradient_accumulation_steps "
            "or per-run gradient_accumulation_steps."
        )

    for key, value in common_args.items():
        if key == "gradient_accumulation_steps":
            continue
        append_arg(command, key, value)

    append_arg(command, "output_dir", run_dir)
    append_arg(command, "log_dir", log_dir)
    append_arg(command, "train_batch_size", batch_size)
    append_arg(command, "gradient_accumulation_steps", grad_steps)
    append_arg(command, "learning_rate", run_config["learning_rate"])

    for key, value in run_config.items():
        if key in {"train_batch_size", "gradient_accumulation_steps", "learning_rate", "name"}:
            continue
        append_arg(command, key, value)

    if config.get("filter_correct_sft", False):
        command.append("--filter-correct-sft")
    if config.get("save_checkpoints", False):
        command.append("--save-checkpoints")

    # Full dataset is the default. Only pass this flag if the config requests a cap.
    if "num_train_examples" in config:
        append_arg(command, "num_train_examples", config["num_train_examples"])

    return command


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_root = Path(config["output_root"])
    log_root = Path(config["log_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    sweep_records: list[dict[str, Any]] = []
    seen_run_names: set[str] = set()
    runs = config.get("runs")
    if runs is None:
        batch_sizes = [int(value) for value in config["batch_sizes"]]
        learning_rates = [float(value) for value in config["learning_rates"]]
        runs = [
            {"train_batch_size": batch_size, "learning_rate": learning_rate}
            for batch_size, learning_rate in product(batch_sizes, learning_rates)
        ]

    for run_config in runs:
        name = run_config.get(
            "name",
            default_run_name(run_config=run_config, config=config),
        )
        if name in seen_run_names:
            raise ValueError(
                f"Duplicate SFT sweep run name {name!r}. Provide explicit names or vary "
                "the run parameters included in the default name."
            )
        seen_run_names.add(name)
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
