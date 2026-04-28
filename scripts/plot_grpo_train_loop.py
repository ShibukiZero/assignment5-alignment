#!/usr/bin/env python3
"""Create writeup-ready artifacts for the Ch7 on-policy GRPO train-loop run."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = ".agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr1e-5"
DEFAULT_OUTPUT_DIR = "artifacts/experiments/ch7/grpo_train_loop"
DEFAULT_RUN_NAME = "grpo_on_policy_lr1e-5"
SELECTED_ROLLOUT_STEPS = (1, 50, 100, 150, 200)

COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
]
ARCHIVE_IGNORE = shutil.ignore_patterns("sample_rollouts.jsonl")


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
    rollout_epoch: int
    token_entropy: float
    grad_norm: float
    loss: float


@dataclass(frozen=True)
class RolloutPoint:
    step: int
    answer_accuracy: float
    format_accuracy: float
    avg_response_token_length: float


@dataclass(frozen=True)
class StepSummary:
    step: int
    step_seconds: float


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
    def best_eval(self) -> EvalPoint:
        return max(self.eval_points, key=lambda point: point.answer_accuracy)

    @property
    def first_eval(self) -> EvalPoint:
        return min(self.eval_points, key=lambda point: point.step)

    @property
    def final_eval(self) -> EvalPoint:
        return max(self.eval_points, key=lambda point: point.step)

    @property
    def first_rollout(self) -> RolloutPoint:
        return min(self.rollout_points, key=lambda point: point.step)

    @property
    def final_rollout(self) -> RolloutPoint:
        return max(self.rollout_points, key=lambda point: point.step)

    @property
    def tail_train_points(self) -> list[TrainPoint]:
        by_step: dict[int, TrainPoint] = {}
        for point in self.train_points:
            previous = by_step.get(point.step)
            if previous is None or (point.optimizer_step, point.rollout_epoch) > (
                previous.optimizer_step,
                previous.rollout_epoch,
            ):
                by_step[point.step] = point
        return [by_step[step] for step in sorted(by_step)]

    @property
    def final_train(self) -> TrainPoint:
        return self.tail_train_points[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--skip-run-archive",
        action="store_true",
        help="Generate plots and summaries without copying the raw run into runs/.",
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


def as_float(value: Any) -> float:
    return float(value)


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
                    answer_accuracy=as_float(row["answer_accuracy"]),
                    format_accuracy=as_float(row["format_accuracy"]),
                    reward=as_float(row.get("reward", row["answer_accuracy"])),
                    eval_seconds=as_float(row["eval_seconds"]),
                    num_examples=(
                        int(row["num_examples"])
                        if row.get("num_examples") is not None
                        else None
                    ),
                )
            )
        elif row_type == "train":
            train_points.append(
                TrainPoint(
                    step=int(row["grpo_step"]),
                    optimizer_step=int(row["optimizer_step"]),
                    rollout_epoch=int(row.get("rollout_epoch", 1)),
                    token_entropy=as_float(row["token_entropy"]),
                    grad_norm=as_float(row["grad_norm"]),
                    loss=as_float(row["loss"]),
                )
            )
        elif row_type == "rollout":
            rollout_points.append(
                RolloutPoint(
                    step=int(row["grpo_step"]),
                    answer_accuracy=as_float(row["answer_accuracy"]),
                    format_accuracy=as_float(row["format_accuracy"]),
                    avg_response_token_length=as_float(
                        row["avg_response_token_length"]
                    ),
                )
            )
        elif row_type == "grpo_step_summary":
            step_summaries.append(
                StepSummary(
                    step=int(row["grpo_step"]),
                    step_seconds=as_float(row["step_seconds"]),
                )
            )

    if not eval_points:
        raise ValueError(f"No eval metrics found in {metrics_path}")
    if not train_points:
        raise ValueError(f"No train metrics found in {metrics_path}")
    if not rollout_points:
        raise ValueError(f"No rollout metrics found in {metrics_path}")

    return RunData(
        label=label,
        run_dir=run_dir,
        config=config,
        run_summary=run_summary,
        eval_points=sorted(eval_points, key=lambda point: point.step),
        train_points=sorted(
            train_points,
            key=lambda point: (point.step, point.optimizer_step, point.rollout_epoch),
        ),
        rollout_points=sorted(rollout_points, key=lambda point: point.step),
        step_summaries=sorted(step_summaries, key=lambda point: point.step),
    )


def nice_y_max(max_value: float) -> float:
    if max_value <= 0:
        return 1.0
    return math.ceil(max_value * 10) / 10


def point_to_svg_xy(
    *,
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
) -> tuple[float, float]:
    px = left if x_max == x_min else left + (x - x_min) / (x_max - x_min) * plot_width
    py = (
        top + plot_height
        if y_max == y_min
        else top + (y_max - y) / (y_max - y_min) * plot_height
    )
    return px, py


def points_to_polyline(
    *,
    points: list[tuple[float, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
) -> str:
    coords: list[str] = []
    for x, y in points:
        px, py = point_to_svg_xy(
            x=x,
            y=y,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            left=left,
            top=top,
            plot_width=plot_width,
            plot_height=plot_height,
        )
        coords.append(f"{px:.1f},{py:.1f}")
    return " ".join(coords)


def render_svg_line_plot(
    *,
    series: list[tuple[str, list[tuple[float, float]]]],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    y_min: float = 0.0,
    y_max: float | None = None,
    x_tick_decimals: int = 0,
    y_percent_axis: bool = True,
) -> None:
    width = 1100
    height = 660
    left = 95
    right = 300
    top = 74
    bottom = 82
    plot_width = width - left - right
    plot_height = height - top - bottom

    cleaned_series = [
        (label, [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)])
        for label, points in series
    ]
    cleaned_series = [(label, points) for label, points in cleaned_series if points]
    if not cleaned_series:
        raise ValueError(f"No plottable data for {title}")

    all_points = [point for _, points in cleaned_series for point in points]
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min = min(x_values)
    x_max = max(x_values)
    if y_max is None:
        y_max = nice_y_max(max(y_values))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" '
        'font-family="Arial" font-size="22" font-weight="700">'
        f"{html.escape(title)}</text>",
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        'stroke="#222" stroke-width="1.5"/>',
    ]

    for tick in range(6):
        ratio = tick / 5
        y_value = y_min + (y_max - y_min) * ratio
        py = top + plot_height - ratio * plot_height
        tick_label = percent(y_value) if y_percent_axis else f"{y_value:.2f}"
        parts.append(
            f'<line x1="{left - 5}" y1="{py:.1f}" x2="{left + plot_width}" '
            f'y2="{py:.1f}" stroke="#ddd" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{py + 4:.1f}" text-anchor="end" '
            'font-family="Arial" font-size="12">'
            f"{tick_label}</text>"
        )

    for tick in range(6):
        ratio = tick / 5
        x_value = x_min + (x_max - x_min) * ratio
        px = left + ratio * plot_width
        tick_label = (
            str(int(round(x_value)))
            if x_tick_decimals == 0
            else f"{x_value:.{x_tick_decimals}f}"
        )
        parts.append(
            f'<line x1="{px:.1f}" y1="{top + plot_height}" x2="{px:.1f}" '
            f'y2="{top + plot_height + 5}" stroke="#222" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px:.1f}" y="{top + plot_height + 24}" text-anchor="middle" '
            'font-family="Arial" font-size="12">'
            f"{tick_label}</text>"
        )

    for idx, (label, points) in enumerate(cleaned_series):
        color = COLORS[idx % len(COLORS)]
        polyline = points_to_polyline(
            points=points,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            left=left,
            top=top,
            plot_width=plot_width,
            plot_height=plot_height,
        )
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.3" '
            f'points="{polyline}"/>'
        )
        for x, y in points:
            px, py = point_to_svg_xy(
                x=x,
                y=y,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                left=left,
                top=top,
                plot_width=plot_width,
                plot_height=plot_height,
            )
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>')

        legend_x = left + plot_width + 28
        legend_y = top + 22 + idx * 24
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="2.5"/>'
        )
        parts.append(
            f'<text x="{legend_x + 30}" y="{legend_y + 4}" '
            'font-family="Arial" font-size="13">'
            f"{html.escape(label)}</text>"
        )

    parts.append(
        f'<text x="{left + plot_width / 2}" y="{height - 22}" text-anchor="middle" '
        'font-family="Arial" font-size="15">'
        f"{html.escape(x_label)}</text>"
    )
    parts.append(
        f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" '
        'font-family="Arial" font-size="15" '
        f'transform="rotate(-90 24 {top + plot_height / 2})">'
        f"{html.escape(y_label)}</text>"
    )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def first_correct_rollout(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rows = read_jsonl(path)
    return next(
        (row for row in rows if row.get("reward") == 1),
        rows[0] if rows else None,
    )


def truncate_response(response: str, max_chars: int = 700) -> str:
    if len(response) <= max_chars:
        return response
    return f"{response[:max_chars].strip()}..."


def archive_run(run: RunData, output_dir: Path) -> Path:
    archive_dir = output_dir / "runs" / run.label
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    if run.run_dir.resolve() != archive_dir.resolve():
        shutil.copytree(run.run_dir, archive_dir, dirs_exist_ok=True, ignore=ARCHIVE_IGNORE)
    return archive_dir


def write_eval_points(run: RunData, output_dir: Path) -> None:
    write_csv(
        output_dir / "grpo_train_loop_eval_points.csv",
        [
            "run",
            "grpo_step",
            "answer_accuracy",
            "format_accuracy",
            "reward",
            "num_examples",
            "eval_seconds",
        ],
        [
            [
                run.label,
                point.step,
                point.answer_accuracy,
                point.format_accuracy,
                point.reward,
                point.num_examples if point.num_examples is not None else "",
                point.eval_seconds,
            ]
            for point in run.eval_points
        ],
    )


def write_rollout_examples(run: RunData, output_dir: Path) -> None:
    rollout_examples: list[dict[str, Any]] = []
    for step in SELECTED_ROLLOUT_STEPS:
        sample_path = run.run_dir / f"grpo_step_{step:06d}" / "sample_rollouts.jsonl"
        rollout = first_correct_rollout(sample_path)
        if rollout is not None:
            rollout_examples.append({"step": step, **rollout})

    write_csv(
        output_dir / "grpo_train_loop_rollout_examples.csv",
        [
            "grpo_step",
            "question_index",
            "rollout_index",
            "reward",
            "format_reward",
            "answer_reward",
            "response_token_length",
            "ground_truth",
            "question",
            "response",
        ],
        [
            [
                row["step"],
                row["question_index"],
                row["rollout_index"],
                row["reward"],
                row["format_reward"],
                row["answer_reward"],
                row["response_token_length"],
                row["ground_truth"],
                row["question"],
                row["response"],
            ]
            for row in rollout_examples
        ],
    )

    lines = ["# GRPO Train Loop Rollout Examples", ""]
    for row in rollout_examples:
        lines.extend(
            [
                f"## Step {row['step']}",
                "",
                f"Question: {row['question']}",
                "",
                f"Ground truth: `{row['ground_truth']}`",
                "",
                (
                    f"Reward: {row['reward']}, format reward: {row['format_reward']}, "
                    f"answer reward: {row['answer_reward']}, response tokens: "
                    f"{row['response_token_length']}"
                ),
                "",
                "Response:",
                "",
                "```text",
                truncate_response(row["response"]),
                "```",
                "",
            ]
        )
    (output_dir / "grpo_train_loop_rollout_examples.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_run_summaries(run: RunData, output_dir: Path, archive_dir: Path) -> None:
    payload = [
        {
            "run_name": run.label,
            "archived_run_dir": str(archive_dir),
            "learning_rate": run.learning_rate,
            "n_grpo_steps": run.n_grpo_steps,
            "rollout_batch_size": run.config["rollout_batch_size"],
            "group_size": run.config["group_size"],
            "loss_type": run.config["loss_type"],
            "use_std_normalization": run.config["use_std_normalization"],
            "first_eval": {
                "grpo_step": run.first_eval.step,
                "answer_accuracy": run.first_eval.answer_accuracy,
                "format_accuracy": run.first_eval.format_accuracy,
            },
            "best_eval": {
                "grpo_step": run.best_eval.step,
                "answer_accuracy": run.best_eval.answer_accuracy,
                "format_accuracy": run.best_eval.format_accuracy,
            },
            "final_eval": {
                "grpo_step": run.final_eval.step,
                "answer_accuracy": run.final_eval.answer_accuracy,
                "format_accuracy": run.final_eval.format_accuracy,
            },
            "first_rollout": {
                "grpo_step": run.first_rollout.step,
                "answer_accuracy": run.first_rollout.answer_accuracy,
                "format_accuracy": run.first_rollout.format_accuracy,
                "avg_response_token_length": run.first_rollout.avg_response_token_length,
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
            "run_summary": run.run_summary,
        }
    ]
    (output_dir / "run_summaries.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# GRPO Train Loop Run Archive",
        "",
        "| run | learning rate | steps | initial answer | best answer | final answer | final format | final rollout answer | final avg length |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| `{run.label}` | `{run.learning_rate}` | {run.n_grpo_steps} | "
        f"{percent(run.first_eval.answer_accuracy)} | "
        f"{percent(run.best_eval.answer_accuracy)} @ {run.best_eval.step} | "
        f"{percent(run.final_eval.answer_accuracy)} | "
        f"{percent(run.final_eval.format_accuracy)} | "
        f"{percent(run.final_rollout.answer_accuracy)} | "
        f"{run.final_rollout.avg_response_token_length:.1f} |",
        "",
        (
            f"Raw run files are archived under `{archive_dir}/`. "
            "`sample_rollouts.jsonl` files are intentionally omitted because aggregate "
            "rollout summaries are sufficient for the writeup."
        ),
        "",
    ]
    (output_dir / "run_summaries_archive.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_summary(run: RunData, output_dir: Path) -> None:
    lines = [
        "# GRPO Train Loop Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| initial validation answer reward | {percent(run.first_eval.answer_accuracy)} |",
        f"| best validation answer reward | {percent(run.best_eval.answer_accuracy)} at step {run.best_eval.step} |",
        f"| final validation answer reward | {percent(run.final_eval.answer_accuracy)} |",
        f"| final validation format accuracy | {percent(run.final_eval.format_accuracy)} |",
        f"| final rollout answer reward | {percent(run.final_rollout.answer_accuracy)} |",
        f"| final rollout format accuracy | {percent(run.final_rollout.format_accuracy)} |",
        f"| final average response token length | {run.final_rollout.avg_response_token_length:.1f} |",
        f"| final token entropy | {run.final_train.token_entropy:.4f} |",
        f"| final gradient norm | {run.final_train.grad_norm:.4f} |",
        f"| final loss | {run.final_train.loss:.4f} |",
        "",
    ]
    (output_dir / "grpo_train_loop_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_experiment_log(run: RunData, output_dir: Path) -> None:
    lines = [
        "# GRPO Train Loop Experiment Log",
        "",
        "## Setup",
        "",
        f"- Run label: `{run.label}`.",
        f"- Learning rate: `{run.learning_rate}`.",
        f"- GRPO steps: `{run.n_grpo_steps}`.",
        f"- Rollout batch size: `{run.config['rollout_batch_size']}` with group size `{run.config['group_size']}`.",
        f"- Loss type: `{run.config['loss_type']}` and std normalization set to `{run.config['use_std_normalization']}`.",
        "",
        "## Outcome",
        "",
        f"- Validation answer reward improved from {percent(run.first_eval.answer_accuracy)} at step {run.first_eval.step} to a best of {percent(run.best_eval.answer_accuracy)} at step {run.best_eval.step}.",
        f"- The run finished at {percent(run.final_eval.answer_accuracy)} validation answer reward and {percent(run.final_eval.format_accuracy)} validation format accuracy.",
        f"- Final rollout answer reward was {percent(run.final_rollout.answer_accuracy)} with average response length {run.final_rollout.avg_response_token_length:.1f} tokens.",
        f"- Final train token entropy was {run.final_train.token_entropy:.4f}.",
        "",
    ]
    (output_dir / "grpo_train_loop_experiment_log.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = load_run(args.run_name, Path(args.run_dir))
    archive_dir = output_dir / "runs" / run.label
    if not args.skip_run_archive:
        archive_dir = archive_run(run, output_dir)

    write_eval_points(run, output_dir)
    write_rollout_examples(run, output_dir)
    write_run_summaries(run, output_dir, archive_dir)
    write_summary(run, output_dir)
    write_experiment_log(run, output_dir)

    render_svg_line_plot(
        series=[
            (
                run.label,
                [(point.step, point.answer_accuracy) for point in run.eval_points],
            )
        ],
        title="On-policy GRPO train loop: validation answer reward",
        x_label="GRPO step",
        y_label="Validation answer reward",
        output_path=output_dir / "grpo_train_loop_validation_reward.svg",
        y_min=0.0,
        y_max=0.75,
        x_tick_decimals=0,
        y_percent_axis=True,
    )
    render_svg_line_plot(
        series=[
            (
                run.label,
                [(point.step, point.format_accuracy) for point in run.eval_points],
            )
        ],
        title="On-policy GRPO train loop: validation format accuracy",
        x_label="GRPO step",
        y_label="Validation format accuracy",
        output_path=output_dir / "grpo_train_loop_format_accuracy.svg",
        y_min=0.0,
        y_max=1.0,
        x_tick_decimals=0,
        y_percent_axis=True,
    )
    render_svg_line_plot(
        series=[
            (
                run.label,
                [
                    (point.step, point.token_entropy)
                    for point in run.tail_train_points
                    if math.isfinite(point.token_entropy)
                ],
            )
        ],
        title="On-policy GRPO train loop: token entropy",
        x_label="GRPO step",
        y_label="Token entropy (nats)",
        output_path=output_dir / "grpo_train_loop_token_entropy.svg",
        y_min=0.0,
        y_max=None,
        x_tick_decimals=0,
        y_percent_axis=False,
    )
    render_svg_line_plot(
        series=[
            (
                run.label,
                [
                    (point.step, point.avg_response_token_length)
                    for point in run.rollout_points
                    if math.isfinite(point.avg_response_token_length)
                ],
            )
        ],
        title="On-policy GRPO train loop: rollout response length",
        x_label="GRPO step",
        y_label="Average rollout response tokens",
        output_path=output_dir / "grpo_train_loop_response_length.svg",
        y_min=0.0,
        y_max=None,
        x_tick_decimals=0,
        y_percent_axis=False,
    )

    print(f"Wrote GRPO train-loop artifacts to {output_dir}")


if __name__ == "__main__":
    main()
