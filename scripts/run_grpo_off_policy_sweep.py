#!/usr/bin/env python3
"""Run and verify the off-policy GRPO sweep from a JSON grid."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


META_KEYS = {
    "name",
    "phase",
    "problem",
    "notes",
    "comparison",
    "expected_optimizer_updates_per_rollout_batch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run off-policy GRPO sweep jobs.")
    parser.add_argument(
        "--grid",
        default=".agents/logs/ch7/grpo_off_policy_sweep/grid.json",
        help="Path to the off-policy sweep JSON grid.",
    )
    parser.add_argument(
        "--phase",
        choices=["smoke", "broad", "focused", "all"],
        default="all",
        help="'all' runs broad and focused jobs, but not the smoke job.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify selected run outputs; do not launch jobs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip selected runs that already pass output verification.",
    )
    parser.add_argument(
        "--stop-on-failure",
        dest="stop_on_failure",
        action="store_true",
        help="Stop immediately when a run fails or fails verification (default).",
    )
    parser.add_argument(
        "--no-stop-on-failure",
        dest="stop_on_failure",
        action="store_false",
        help="Continue through the selected sweep even if a run fails verification.",
    )
    parser.set_defaults(stop_on_failure=True)
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Power off after every selected run completes and verifies.",
    )
    parser.add_argument(
        "--sudo-shutdown",
        action="store_true",
        help="Use sudo shutdown -h now instead of shutdown -h now.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def append_arg(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if key == "use_std_normalization":
            command.append("--use-std-normalization" if value else "--no-use-std-normalization")
        elif value:
            command.append(flag_name(key))
        return
    command.extend([flag_name(key), str(value)])


def selected_runs(config: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    runs = list(config["runs"])
    if phase == "all":
        return [run for run in runs if run.get("phase") in {"broad", "focused"}]
    return [run for run in runs if run.get("phase") == phase]


def run_paths(config: dict[str, Any], run_config: dict[str, Any]) -> tuple[Path, Path]:
    name = run_config["name"]
    output_dir = Path(run_config.get("output_dir", Path(config["output_root"]) / name))
    log_dir = Path(run_config.get("log_dir", Path(config["log_root"]) / name))
    return output_dir, log_dir


def build_command(config: dict[str, Any], run_config: dict[str, Any]) -> list[str]:
    python_command = config.get("python", sys.executable)
    if isinstance(python_command, list):
        command = [str(part) for part in python_command]
    else:
        command = [str(python_command)]
    command.append(config["script"])

    output_dir, log_dir = run_paths(config, run_config)
    merged_args = dict(config.get("common_args", {}))
    for key, value in run_config.items():
        if key not in META_KEYS and key not in {"output_dir", "log_dir"}:
            merged_args[key] = value

    merged_args["output_dir"] = str(output_dir)
    merged_args["log_dir"] = str(log_dir)

    for key, value in merged_args.items():
        append_arg(command, key, value)

    return command


def expected_optimizer_steps(config: dict[str, Any], run_config: dict[str, Any]) -> int:
    merged_args = dict(config.get("common_args", {}))
    for key, value in run_config.items():
        if key not in META_KEYS and key not in {"output_dir", "log_dir"}:
            merged_args[key] = value
    updates_per_rollout = int(merged_args["epochs_per_rollout_batch"]) * (
        int(merged_args["rollout_batch_size"]) // int(merged_args["train_batch_size"])
    )
    return int(merged_args["n_grpo_steps"]) * updates_per_rollout


def should_save_checkpoints(config: dict[str, Any], run_config: dict[str, Any]) -> bool:
    merged_args = dict(config.get("common_args", {}))
    merged_args.update(
        {
            key: value
            for key, value in run_config.items()
            if key not in META_KEYS and key not in {"output_dir", "log_dir"}
        }
    )
    return bool(merged_args.get("save_checkpoints", False))


def count_train_metrics(metrics_path: Path) -> int:
    count = 0
    with metrics_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "train":
                count += 1
    return count


def verify_run(config: dict[str, Any], run_config: dict[str, Any]) -> tuple[bool, list[str]]:
    name = run_config["name"]
    output_dir, log_dir = run_paths(config, run_config)
    messages: list[str] = []
    required_files = [
        log_dir / "config.json",
        log_dir / "metrics.jsonl",
        log_dir / "run_summary.json",
    ]
    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            messages.append(f"{name}: missing or empty {path}")

    if not output_dir.is_dir():
        messages.append(f"{name}: missing output directory {output_dir}")

    if should_save_checkpoints(config, run_config):
        final_policy = output_dir / "final_policy"
        if not final_policy.is_dir():
            messages.append(f"{name}: missing final checkpoint directory {final_policy}")

    summary_path = log_dir / "run_summary.json"
    metrics_path = log_dir / "metrics.jsonl"
    if summary_path.is_file() and metrics_path.is_file():
        summary = load_json(summary_path)
        expected_steps = expected_optimizer_steps(config, run_config)
        final_step = int(summary.get("final_optimizer_step", -1))
        if final_step != expected_steps:
            messages.append(
                f"{name}: final_optimizer_step={final_step}, expected {expected_steps}"
            )
        train_metric_count = count_train_metrics(metrics_path)
        if train_metric_count != expected_steps:
            messages.append(f"{name}: train metric rows={train_metric_count}, expected {expected_steps}")

    return len(messages) == 0, messages


def has_partial_outputs(config: dict[str, Any], run_config: dict[str, Any]) -> bool:
    output_dir, log_dir = run_paths(config, run_config)
    return output_dir.exists() or log_dir.exists()


def maybe_shutdown(use_sudo: bool) -> None:
    print("Flushing filesystem buffers...", flush=True)
    subprocess.run(["sync"], check=False)
    command = ["sudo", "shutdown", "-h", "now"] if use_sudo else ["shutdown", "-h", "now"]
    print("Powering off: " + " ".join(command), flush=True)
    subprocess.run(command, check=False)


def main() -> None:
    args = parse_args()
    config = load_json(Path(args.grid))
    runs = selected_runs(config, args.phase)
    if not runs:
        raise ValueError(f"No runs selected for phase {args.phase!r}.")

    if not args.dry_run and not args.check_only:
        Path(config["output_root"]).mkdir(parents=True, exist_ok=True)
    Path(config["log_root"]).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for run_config in runs:
        name = run_config["name"]
        output_dir, log_dir = run_paths(config, run_config)
        command = build_command(config, run_config)
        record = {
            "name": name,
            "phase": run_config.get("phase"),
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
            "command": command,
        }
        manifest.append(record)

        print(" ".join(command), flush=True)
        verified, messages = verify_run(config, run_config)
        if args.check_only:
            record["verified"] = verified
            record["verification_messages"] = messages
            for message in messages:
                print(message, flush=True)
            if not verified and args.stop_on_failure:
                write_json(Path(config["log_root"]) / f"sweep_manifest_{args.phase}.json", manifest)
                raise SystemExit(4)
            continue

        if args.dry_run:
            record["dry_run"] = True
            continue

        if verified and args.skip_existing:
            print(f"{name}: already verified; skipping.", flush=True)
            record["skipped"] = True
            continue
        if has_partial_outputs(config, run_config) and not verified:
            for message in messages:
                print(message, flush=True)
            print(f"{name}: partial outputs found; inspect or move them before rerunning.", flush=True)
            write_json(Path(config["log_root"]) / f"sweep_manifest_{args.phase}.json", manifest)
            raise SystemExit(3)

        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout_stderr.log"
        start = time.time()
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

        verified, messages = verify_run(config, run_config)
        record["verified"] = verified
        record["verification_messages"] = messages
        for message in messages:
            print(message, flush=True)
        if completed.returncode != 0 or not verified:
            write_json(Path(config["log_root"]) / f"sweep_manifest_{args.phase}.json", manifest)
            if args.stop_on_failure:
                raise SystemExit(4)

    manifest_path = Path(config["log_root"]) / f"sweep_manifest_{args.phase}.json"
    write_json(manifest_path, manifest)
    print(f"Wrote sweep manifest: {manifest_path}", flush=True)

    if args.shutdown and not args.dry_run and not args.check_only:
        maybe_shutdown(use_sudo=args.sudo_shutdown)


if __name__ == "__main__":
    main()
