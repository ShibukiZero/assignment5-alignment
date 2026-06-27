#!/usr/bin/env python3
"""Create writeup-ready artifacts for the final leaderboard-style GRPO run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plot_grpo_train_loop import render_svg_line_plot


DEFAULT_RUN_DIR = "runs/logs/leaderboard/final_on_policy_lr4e-5_400_full_eval"
DEFAULT_OUTPUT_DIR = "artifacts/experiments/ch7/leaderboard"
DEFAULT_RUN_NAME = "final_on_policy_lr4e-5_400_full_eval"
LARGE_FILE_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "optimizer*",
    "scheduler*",
    "checkpoint*",
    "best_policy",
    "final_policy",
    "sample_rollouts.jsonl",
)


@dataclass(frozen=True)
class EvalPoint:
    step: int
    answer_accuracy: float
    format_accuracy: float
    reward: float
    eval_seconds: float
    num_examples: int | None


@dataclass(frozen=True)
class TrainPoint:
    step: int
    optimizer_step: int
    token_entropy: float
    grad_norm: float
    loss: float
    train_answer_accuracy: float
    train_format_accuracy: float


@dataclass(frozen=True)
class RolloutPoint:
    step: int
    answer_accuracy: float
    format_accuracy: float
    avg_response_token_length: float
    rollout_seconds: float


@dataclass(frozen=True)
class StepSummary:
    step: int
    optimizer_step: int
    step_seconds: float
    best_answer_accuracy: float | None


@dataclass
class RunData:
    label: str
    run_dir: Path
    config: dict[str, Any]
    run_summary: dict[str, Any]
    eval_points: list[EvalPoint]
    train_points: list[TrainPoint]
    rollout_points: list[RolloutPoint]
    step_summaries: list[StepSummary]

    @property
    def learning_rate(self) -> float:
        return float(self.config["learning_rate"])

    @property
    def n_grpo_steps(self) -> int:
        return int(self.config["n_grpo_steps"])

    @property
    def first_eval(self) -> EvalPoint:
        return min(self.eval_points, key=lambda point: point.step)

    @property
    def best_eval(self) -> EvalPoint:
        return max(self.eval_points, key=lambda point: point.answer_accuracy)

    @property
    def final_eval(self) -> EvalPoint:
        return max(self.eval_points, key=lambda point: point.step)

    @property
    def final_train(self) -> TrainPoint:
        return max(self.train_points, key=lambda point: point.optimizer_step)

    @property
    def final_rollout(self) -> RolloutPoint:
        return max(self.rollout_points, key=lambda point: point.step)

    def cumulative_step_seconds(self) -> dict[int, float]:
        elapsed = 0.0
        by_step: dict[int, float] = {0: 0.0}
        for point in sorted(self.step_summaries, key=lambda item: item.step):
            elapsed += point.step_seconds
            by_step[point.step] = elapsed
        return by_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-run-archive",
        action="store_true",
        help="Generate plots and summaries without copying lightweight run logs.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def load_run(label: str, run_dir: Path) -> RunData:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    config = read_json(run_dir / "config.json")
    run_summary = read_json(run_dir / "run_summary.json")
    eval_points: list[EvalPoint] = []
    train_points: list[TrainPoint] = []
    rollout_points: list[RolloutPoint] = []
    step_summaries: list[StepSummary] = []

    for row in read_jsonl(metrics_path):
        row_type = row.get("type")
        if row_type == "eval":
            eval_points.append(
                EvalPoint(
                    step=int(row["grpo_step"]),
                    answer_accuracy=float(row["answer_accuracy"]),
                    format_accuracy=float(row["format_accuracy"]),
                    reward=float(row.get("reward", row["answer_accuracy"])),
                    eval_seconds=float(row["eval_seconds"]),
                    num_examples=optional_int(row.get("num_examples")),
                )
            )
        elif row_type == "train":
            train_points.append(
                TrainPoint(
                    step=int(row["grpo_step"]),
                    optimizer_step=int(row["optimizer_step"]),
                    token_entropy=float(row["token_entropy"]),
                    grad_norm=float(row["grad_norm"]),
                    loss=float(row["loss"]),
                    train_answer_accuracy=float(row["train_answer_accuracy"]),
                    train_format_accuracy=float(row["train_format_accuracy"]),
                )
            )
        elif row_type == "rollout":
            rollout_points.append(
                RolloutPoint(
                    step=int(row["grpo_step"]),
                    answer_accuracy=float(row["answer_accuracy"]),
                    format_accuracy=float(row["format_accuracy"]),
                    avg_response_token_length=float(row["avg_response_token_length"]),
                    rollout_seconds=float(row["rollout_seconds"]),
                )
            )
        elif row_type == "grpo_step_summary":
            step_summaries.append(
                StepSummary(
                    step=int(row["grpo_step"]),
                    optimizer_step=int(row["optimizer_step"]),
                    step_seconds=float(row["step_seconds"]),
                    best_answer_accuracy=optional_float(row.get("best_answer_accuracy")),
                )
            )

    if not eval_points:
        raise ValueError(f"No eval points found in {metrics_path}")
    if not train_points:
        raise ValueError(f"No train points found in {metrics_path}")
    if not rollout_points:
        raise ValueError(f"No rollout points found in {metrics_path}")
    if not step_summaries:
        raise ValueError(f"No step summaries found in {metrics_path}")

    return RunData(
        label=label,
        run_dir=run_dir,
        config=config,
        run_summary=run_summary,
        eval_points=sorted(eval_points, key=lambda point: point.step),
        train_points=sorted(train_points, key=lambda point: point.optimizer_step),
        rollout_points=sorted(rollout_points, key=lambda point: point.step),
        step_summaries=sorted(step_summaries, key=lambda point: point.step),
    )


def write_eval_points(run: RunData, output_dir: Path) -> None:
    elapsed_by_step = run.cumulative_step_seconds()
    rows: list[list[Any]] = []
    for point in run.eval_points:
        elapsed_seconds = elapsed_by_step.get(point.step)
        rows.append(
            [
                run.label,
                point.step,
                point.answer_accuracy,
                point.format_accuracy,
                point.reward,
                point.num_examples,
                point.eval_seconds,
                elapsed_seconds,
                elapsed_seconds / 3600.0 if elapsed_seconds is not None else None,
            ]
        )
    write_csv(
        output_dir / "leaderboard_eval_points.csv",
        [
            "run",
            "grpo_step",
            "answer_accuracy",
            "format_accuracy",
            "reward",
            "num_examples",
            "eval_seconds",
            "elapsed_step_seconds",
            "elapsed_step_hours",
        ],
        rows,
    )


def write_train_points(run: RunData, output_dir: Path) -> None:
    write_csv(
        output_dir / "leaderboard_train_points.csv",
        [
            "run",
            "grpo_step",
            "optimizer_step",
            "loss",
            "grad_norm",
            "token_entropy",
            "train_answer_accuracy",
            "train_format_accuracy",
        ],
        [
            [
                run.label,
                point.step,
                point.optimizer_step,
                point.loss,
                point.grad_norm,
                point.token_entropy,
                point.train_answer_accuracy,
                point.train_format_accuracy,
            ]
            for point in run.train_points
        ],
    )


def write_rollout_points(run: RunData, output_dir: Path) -> None:
    write_csv(
        output_dir / "leaderboard_rollout_points.csv",
        [
            "run",
            "grpo_step",
            "answer_accuracy",
            "format_accuracy",
            "avg_response_token_length",
            "rollout_seconds",
        ],
        [
            [
                run.label,
                point.step,
                point.answer_accuracy,
                point.format_accuracy,
                point.avg_response_token_length,
                point.rollout_seconds,
            ]
            for point in run.rollout_points
        ],
    )


def write_step_points(run: RunData, output_dir: Path) -> None:
    elapsed = 0.0
    rows: list[list[Any]] = []
    for point in run.step_summaries:
        elapsed += point.step_seconds
        rows.append(
            [
                run.label,
                point.step,
                point.optimizer_step,
                point.step_seconds,
                elapsed,
                elapsed / 3600.0,
                point.best_answer_accuracy,
            ]
        )
    write_csv(
        output_dir / "leaderboard_step_points.csv",
        [
            "run",
            "grpo_step",
            "optimizer_step",
            "step_seconds",
            "elapsed_step_seconds",
            "elapsed_step_hours",
            "best_answer_accuracy_so_far",
        ],
        rows,
    )


def write_summary(run: RunData, output_dir: Path) -> None:
    elapsed_by_step = run.cumulative_step_seconds()
    best_elapsed_hours = elapsed_by_step[run.best_eval.step] / 3600.0
    final_elapsed_hours = elapsed_by_step[run.final_eval.step] / 3600.0
    payload = {
        "run": run.label,
        "source_log_dir": str(run.run_dir),
        "train_path": run.config.get("train_path"),
        "val_path": run.config.get("val_path"),
        "model": run.config.get("model"),
        "learning_rate": run.learning_rate,
        "n_grpo_steps": run.n_grpo_steps,
        "eval_num_examples": run.final_eval.num_examples,
        "eval_temperature": run.config.get("eval_temperature"),
        "eval_top_p": run.config.get("eval_top_p"),
        "eval_max_tokens": run.config.get("max_new_tokens"),
        "best_eval": {
            "grpo_step": run.best_eval.step,
            "answer_accuracy": run.best_eval.answer_accuracy,
            "format_accuracy": run.best_eval.format_accuracy,
            "elapsed_step_hours": best_elapsed_hours,
        },
        "final_eval": {
            "grpo_step": run.final_eval.step,
            "answer_accuracy": run.final_eval.answer_accuracy,
            "format_accuracy": run.final_eval.format_accuracy,
            "elapsed_step_hours": final_elapsed_hours,
        },
        "final_rollout": {
            "grpo_step": run.final_rollout.step,
            "answer_accuracy": run.final_rollout.answer_accuracy,
            "format_accuracy": run.final_rollout.format_accuracy,
            "avg_response_token_length": run.final_rollout.avg_response_token_length,
        },
        "final_train": {
            "grpo_step": run.final_train.step,
            "optimizer_step": run.final_train.optimizer_step,
            "token_entropy": run.final_train.token_entropy,
            "grad_norm": run.final_train.grad_norm,
            "loss": run.final_train.loss,
        },
    }
    (output_dir / "leaderboard_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Leaderboard-style GRPO summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| validation examples | {run.final_eval.num_examples} |",
        f"| initial validation answer reward | {percent(run.first_eval.answer_accuracy)} |",
        f"| best validation answer reward | {percent(run.best_eval.answer_accuracy)} at step {run.best_eval.step} |",
        f"| final validation answer reward | {percent(run.final_eval.answer_accuracy)} at step {run.final_eval.step} |",
        f"| final validation format accuracy | {percent(run.final_eval.format_accuracy)} |",
        f"| best elapsed step time | {best_elapsed_hours:.2f} h |",
        f"| final elapsed step time | {final_elapsed_hours:.2f} h |",
        f"| final rollout answer reward | {percent(run.final_rollout.answer_accuracy)} |",
        f"| final rollout format reward | {percent(run.final_rollout.format_accuracy)} |",
        f"| final average response token length | {run.final_rollout.avg_response_token_length:.1f} |",
        f"| final token entropy | {run.final_train.token_entropy:.4f} |",
        "",
        "The archived result uses the substitute `competition_math_numeric` validation set, not the official course MATH validation files.",
        "",
        "Generated artifacts:",
        "",
        "- `artifacts/experiments/ch7/leaderboard/leaderboard_validation_reward.svg`",
        "- `artifacts/experiments/ch7/leaderboard/leaderboard_validation_reward_elapsed_time.svg`",
        "- `artifacts/experiments/ch7/leaderboard/leaderboard_format_accuracy.svg`",
        "- `artifacts/experiments/ch7/leaderboard/leaderboard_rollout_answer_reward.svg`",
        "- `artifacts/experiments/ch7/leaderboard/leaderboard_response_length.svg`",
        "- `artifacts/experiments/ch7/leaderboard/leaderboard_token_entropy.svg`",
        "- `artifacts/experiments/ch7/leaderboard/leaderboard_eval_points.csv`",
        "- `artifacts/experiments/ch7/leaderboard/leaderboard_train_points.csv`",
        "- `artifacts/experiments/ch7/leaderboard/leaderboard_rollout_points.csv`",
        "- `artifacts/experiments/ch7/leaderboard/leaderboard_step_points.csv`",
    ]
    (output_dir / "leaderboard_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def render_plots(run: RunData, output_dir: Path) -> None:
    elapsed_by_step = run.cumulative_step_seconds()
    eval_elapsed_time_points = [
        (elapsed_by_step[point.step] / 3600.0, point.answer_accuracy)
        for point in run.eval_points
        if point.step in elapsed_by_step
    ]
    render_svg_line_plot(
        series=[(run.label, [(point.step, point.answer_accuracy) for point in run.eval_points])],
        title="Leaderboard validation answer reward",
        x_label="GRPO step",
        y_label="Answer reward",
        output_path=output_dir / "leaderboard_validation_reward.svg",
        y_percent_axis=True,
    )
    render_svg_line_plot(
        series=[(run.label, eval_elapsed_time_points)],
        title="Leaderboard validation answer reward vs elapsed step time",
        x_label="Elapsed step hours",
        y_label="Answer reward",
        output_path=output_dir / "leaderboard_validation_reward_elapsed_time.svg",
        x_tick_decimals=1,
        y_percent_axis=True,
    )
    render_svg_line_plot(
        series=[(run.label, [(point.step, point.format_accuracy) for point in run.eval_points])],
        title="Leaderboard validation format accuracy",
        x_label="GRPO step",
        y_label="Format accuracy",
        output_path=output_dir / "leaderboard_format_accuracy.svg",
        y_percent_axis=True,
    )
    render_svg_line_plot(
        series=[(run.label, [(point.step, point.answer_accuracy) for point in run.rollout_points])],
        title="Leaderboard rollout answer reward",
        x_label="GRPO step",
        y_label="Rollout answer reward",
        output_path=output_dir / "leaderboard_rollout_answer_reward.svg",
        y_percent_axis=True,
    )
    render_svg_line_plot(
        series=[
            (
                run.label,
                [(point.step, point.avg_response_token_length) for point in run.rollout_points],
            )
        ],
        title="Leaderboard rollout response length",
        x_label="GRPO step",
        y_label="Average response tokens",
        output_path=output_dir / "leaderboard_response_length.svg",
        y_percent_axis=False,
    )
    render_svg_line_plot(
        series=[(run.label, [(point.step, point.token_entropy) for point in run.train_points])],
        title="Leaderboard token entropy",
        x_label="GRPO step",
        y_label="Mean token entropy",
        output_path=output_dir / "leaderboard_token_entropy.svg",
        y_percent_axis=False,
    )


def copy_file_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def archive_lightweight_run(run: RunData, output_dir: Path, skip_run_archive: bool) -> None:
    archive_dir = output_dir / "runs" / run.label
    copied_files: list[str] = []
    skipped_patterns = list(LARGE_FILE_PATTERNS)
    if not skip_run_archive:
        if archive_dir.exists():
            shutil.rmtree(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)

        for filename in ["config.json", "run_summary.json", "metrics.jsonl"]:
            source = run.run_dir / filename
            destination = archive_dir / filename
            if copy_file_if_exists(source, destination):
                copied_files.append(str(destination.relative_to(output_dir)))

        for source in sorted(run.run_dir.glob("eval_summary_grpo_step_*.json")):
            destination = archive_dir / source.name
            if copy_file_if_exists(source, destination):
                copied_files.append(str(destination.relative_to(output_dir)))

        for step_dir in sorted(run.run_dir.glob("grpo_step_*")):
            if not step_dir.is_dir():
                continue
            source = step_dir / "rollout_summary.json"
            destination = archive_dir / step_dir.name / "rollout_summary.json"
            if copy_file_if_exists(source, destination):
                copied_files.append(str(destination.relative_to(output_dir)))

    manifest = {
        "run": run.label,
        "source_log_dir": str(run.run_dir),
        "archived_run_dir": str(archive_dir),
        "copied_file_count": len(copied_files),
        "large_files_omitted": True,
        "sample_rollouts_omitted": True,
        "skipped_patterns": skipped_patterns,
    }
    (output_dir / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Leaderboard Run Archive",
        "",
        f"Source run: `{run.run_dir}`",
        "",
        f"Archived lightweight run files under `{archive_dir}`.",
        "",
        "The archive intentionally copies only `config.json`, `run_summary.json`, `metrics.jsonl`, validation summaries, and per-step `rollout_summary.json` files.",
        "Checkpoints, model weights, optimizer state, scheduler state, and sample rollout dumps are intentionally omitted.",
        "",
        "| run | validation examples | best answer | best step | final answer | final step | final rollout answer | final avg length |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| `{run.label}` | {run.final_eval.num_examples} | "
        f"{percent(run.best_eval.answer_accuracy)} | {run.best_eval.step} | "
        f"{percent(run.final_eval.answer_accuracy)} | {run.final_eval.step} | "
        f"{percent(run.final_rollout.answer_accuracy)} | "
        f"{run.final_rollout.avg_response_token_length:.1f} |",
        "",
    ]
    (output_dir / "run_summaries_archive.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = load_run(args.run_name, Path(args.run_dir))
    write_eval_points(run, output_dir)
    write_train_points(run, output_dir)
    write_rollout_points(run, output_dir)
    write_step_points(run, output_dir)
    write_summary(run, output_dir)
    render_plots(run, output_dir)
    archive_lightweight_run(run, output_dir, args.skip_run_archive)


if __name__ == "__main__":
    main()
