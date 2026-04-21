#!/usr/bin/env python3
"""Create writeup-ready artifacts for the Ch7 off-policy GRPO sweep."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = "artifacts/experiments/ch7/grpo_off_policy_sweep"

BROAD_RUN_SPECS = [
    (
        "broad_e1_tb256_control",
        ".agents/logs/ch7/grpo_off_policy_sweep/broad_e1_tb256",
        "broad",
        "on_policy_style_control",
    ),
    (
        "broad_e2_tb256",
        ".agents/logs/ch7/grpo_off_policy_sweep/broad_e2_tb256",
        "broad",
        "true_off_policy",
    ),
    (
        "broad_e2_tb128",
        ".agents/logs/ch7/grpo_off_policy_sweep/broad_e2_tb128",
        "broad",
        "true_off_policy",
    ),
    (
        "broad_e4_tb256",
        ".agents/logs/ch7/grpo_off_policy_sweep/broad_e4_tb256",
        "broad",
        "true_off_policy",
    ),
    (
        "broad_e4_tb128",
        ".agents/logs/ch7/grpo_off_policy_sweep/broad_e4_tb128",
        "broad",
        "true_off_policy",
    ),
]

FOCUSED_RUN_SPECS = [
    (
        "on_policy_reference_e1_tb256",
        ".agents/logs/ch7/grpo_on_policy_ablations/std_normalization/no_std_lr4e-5",
        "focused",
        "on_policy_reference",
    ),
    (
        "focused_e2_tb256",
        ".agents/logs/ch7/grpo_off_policy_sweep/focused_e2_tb256",
        "focused",
        "true_off_policy",
    ),
]

COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


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
    clip_fraction: float | None
    approx_kl: float | None


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
    log_dir: Path
    phase: str
    role: str
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
    def epochs_per_rollout_batch(self) -> int:
        return int(self.config["epochs_per_rollout_batch"])

    @property
    def rollout_batch_size(self) -> int:
        return int(self.config["rollout_batch_size"])

    @property
    def train_batch_size(self) -> int:
        return int(self.config["train_batch_size"])

    @property
    def optimizer_updates_per_rollout_batch(self) -> int:
        return (
            self.epochs_per_rollout_batch
            * self.rollout_batch_size
            // self.train_batch_size
        )

    @property
    def loss_type(self) -> str:
        return str(self.config["loss_type"])

    @property
    def best_eval(self) -> EvalPoint:
        return max(self.eval_points, key=lambda point: point.answer_accuracy)

    @property
    def final_eval(self) -> EvalPoint:
        return max(self.eval_points, key=lambda point: point.step)

    @property
    def last_finite_eval(self) -> EvalPoint:
        finite = [
            point
            for point in self.eval_points
            if math.isfinite(point.answer_accuracy) and math.isfinite(point.format_accuracy)
        ]
        return max(finite, key=lambda point: point.step)

    @property
    def final_rollout(self) -> RolloutPoint:
        return max(self.rollout_points, key=lambda point: point.step)

    @property
    def last_finite_train(self) -> TrainPoint:
        finite = [
            point
            for point in self.train_points
            if math.isfinite(point.token_entropy)
            and math.isfinite(point.grad_norm)
            and math.isfinite(point.loss)
        ]
        return max(finite, key=lambda point: point.optimizer_step)

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
    def status(self) -> str:
        final = self.final_eval
        best = self.best_eval
        if final.answer_accuracy == 0.0 and final.format_accuracy == 0.0:
            if final.step >= self.n_grpo_steps and best.answer_accuracy >= 0.2:
                return "late collapse"
            return "collapsed"
        if final.step < self.n_grpo_steps:
            return "stopped early"
        return "completed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-run-archive",
        action="store_true",
        help="Generate plots and summaries without copying raw logs into runs/.",
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


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def load_run(label: str, log_dir: Path, phase: str, role: str) -> RunData:
    metrics_path = log_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    config = read_json(log_dir / "config.json")
    run_summary = read_json(log_dir / "run_summary.json")

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
                        int(row["num_examples"]) if row.get("num_examples") is not None else None
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
                    clip_fraction=maybe_float(row.get("clip_fraction")),
                    approx_kl=maybe_float(row.get("approx_kl")),
                )
            )
        elif row_type == "rollout":
            rollout_points.append(
                RolloutPoint(
                    step=int(row["grpo_step"]),
                    answer_accuracy=as_float(row["answer_accuracy"]),
                    format_accuracy=as_float(row["format_accuracy"]),
                    avg_response_token_length=as_float(row["avg_response_token_length"]),
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
    if not step_summaries and max(point.step for point in eval_points) > 0:
        raise ValueError(f"No grpo_step_summary metrics found in {metrics_path}")

    return RunData(
        label=label,
        log_dir=log_dir,
        phase=phase,
        role=role,
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


def load_runs() -> list[RunData]:
    runs: list[RunData] = []
    for label, log_dir, phase, role in BROAD_RUN_SPECS + FOCUSED_RUN_SPECS:
        runs.append(load_run(label, Path(log_dir), phase=phase, role=role))
    return runs


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


def format_x_tick(value: float, decimals: int) -> str:
    if decimals == 0:
        return str(int(round(value)))
    return f"{value:.{decimals}f}"


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
        parts.append(
            f'<line x1="{px:.1f}" y1="{top + plot_height}" x2="{px:.1f}" '
            f'y2="{top + plot_height + 5}" stroke="#222" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px:.1f}" y="{top + plot_height + 24}" text-anchor="middle" '
            'font-family="Arial" font-size="12">'
            f"{format_x_tick(x_value, x_tick_decimals)}</text>"
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


def archive_runs(runs: list[RunData], output_dir: Path) -> None:
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        archive_dir = runs_dir / run.label
        if run.log_dir.resolve() == archive_dir.resolve():
            continue
        shutil.copytree(run.log_dir, archive_dir, dirs_exist_ok=True)


def cumulative_eval_minutes(run: RunData) -> list[tuple[float, float]]:
    cumulative_seconds = 0.0
    previous_step = 0
    step_seconds = {row.step: row.step_seconds for row in run.step_summaries}
    points: list[tuple[float, float]] = []
    for eval_point in run.eval_points:
        if eval_point.step == 0:
            cumulative_seconds += eval_point.eval_seconds
            previous_step = 0
        else:
            for step in range(previous_step + 1, eval_point.step + 1):
                cumulative_seconds += step_seconds.get(step, 0.0)
            cumulative_seconds += eval_point.eval_seconds
            previous_step = eval_point.step
        points.append((cumulative_seconds / 60.0, eval_point.answer_accuracy))
    return points


def broad_runs(runs: Iterable[RunData]) -> list[RunData]:
    return [run for run in runs if run.phase == "broad"]


def focused_runs(runs: Iterable[RunData]) -> list[RunData]:
    return [run for run in runs if run.phase == "focused"]


def write_run_summaries(runs: list[RunData], output_dir: Path) -> None:
    payload: list[dict[str, Any]] = []
    for run in runs:
        payload.append(
            {
                "run_name": run.label,
                "phase": run.phase,
                "role": run.role,
                "archived_run_dir": str(output_dir / "runs" / run.label),
                "learning_rate": run.learning_rate,
                "epochs_per_rollout_batch": run.epochs_per_rollout_batch,
                "train_batch_size": run.train_batch_size,
                "optimizer_updates_per_rollout_batch": run.optimizer_updates_per_rollout_batch,
                "loss_type": run.loss_type,
                "status": run.status,
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
                "final_rollout": {
                    "grpo_step": run.final_rollout.step,
                    "answer_accuracy": run.final_rollout.answer_accuracy,
                    "format_accuracy": run.final_rollout.format_accuracy,
                    "avg_response_token_length": run.final_rollout.avg_response_token_length,
                },
                "last_finite_train": {
                    "optimizer_step": run.last_finite_train.optimizer_step,
                    "token_entropy": run.last_finite_train.token_entropy,
                    "grad_norm": run.last_finite_train.grad_norm,
                    "loss": run.last_finite_train.loss,
                },
                "run_summary": run.run_summary,
            }
        )
    (output_dir / "run_summaries.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# GRPO Off-Policy Sweep Run Archive",
        "",
        "| run | phase | role | status | best answer | best step | final answer | final step |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(
            f"| `{run.label}` | {run.phase} | {run.role} | {run.status} | "
            f"{percent(run.best_eval.answer_accuracy)} | {run.best_eval.step} | "
            f"{percent(run.final_eval.answer_accuracy)} | {run.final_eval.step} |"
        )
    lines.extend(
        [
            "",
            "Raw run files are archived under `artifacts/experiments/ch7/grpo_off_policy_sweep/runs/`.",
            "",
        ]
    )
    (output_dir / "run_summaries_archive.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_broad_tables(runs: list[RunData], output_dir: Path) -> None:
    rows: list[list[Any]] = []
    for run in runs:
        rows.append(
            [
                run.label,
                run.role,
                run.learning_rate,
                run.epochs_per_rollout_batch,
                run.train_batch_size,
                run.optimizer_updates_per_rollout_batch,
                run.status,
                run.best_eval.answer_accuracy,
                run.best_eval.step,
                run.final_eval.answer_accuracy,
                run.final_eval.step,
                run.final_eval.format_accuracy,
                run.final_rollout.avg_response_token_length,
            ]
        )
    write_csv(
        output_dir / "grpo_off_policy_broad_summary.csv",
        [
            "run",
            "role",
            "learning_rate",
            "epochs_per_rollout_batch",
            "train_batch_size",
            "optimizer_updates_per_rollout_batch",
            "status",
            "best_answer_accuracy",
            "best_step",
            "final_answer_accuracy",
            "final_step",
            "final_format_accuracy",
            "final_avg_response_token_length",
        ],
        rows,
    )

    lines = [
        "# GRPO Off-Policy Broad Sweep Summary",
        "",
        "| run | role | updates / rollout batch | status | best answer | best step | final answer | final format |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(
            f"| `{run.label}` | {run.role} | {run.optimizer_updates_per_rollout_batch} | "
            f"{run.status} | {percent(run.best_eval.answer_accuracy)} | {run.best_eval.step} | "
            f"{percent(run.final_eval.answer_accuracy)} | {percent(run.final_eval.format_accuracy)} |"
        )
    lines.append("")
    (output_dir / "grpo_off_policy_broad_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_focused_tables(runs: list[RunData], output_dir: Path) -> None:
    rows: list[list[Any]] = []
    for run in runs:
        rows.append(
            [
                run.label,
                run.role,
                run.learning_rate,
                run.epochs_per_rollout_batch,
                run.train_batch_size,
                run.optimizer_updates_per_rollout_batch,
                run.loss_type,
                run.status,
                run.best_eval.answer_accuracy,
                run.best_eval.step,
                run.final_eval.answer_accuracy,
                run.final_eval.step,
                run.final_eval.format_accuracy,
                run.final_rollout.avg_response_token_length,
                run.last_finite_train.token_entropy,
                run.last_finite_train.grad_norm,
            ]
        )
    write_csv(
        output_dir / "grpo_off_policy_focused_summary.csv",
        [
            "run",
            "role",
            "learning_rate",
            "epochs_per_rollout_batch",
            "train_batch_size",
            "optimizer_updates_per_rollout_batch",
            "loss_type",
            "status",
            "best_answer_accuracy",
            "best_step",
            "final_answer_accuracy",
            "final_step",
            "final_format_accuracy",
            "final_avg_response_token_length",
            "last_finite_token_entropy",
            "last_finite_grad_norm",
        ],
        rows,
    )

    lines = [
        "# GRPO Off-Policy Focused Comparison Summary",
        "",
        "| run | role | status | best answer | best step | final answer | final format | final avg length |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(
            f"| `{run.label}` | {run.role} | {run.status} | "
            f"{percent(run.best_eval.answer_accuracy)} | {run.best_eval.step} | "
            f"{percent(run.final_eval.answer_accuracy)} | {percent(run.final_eval.format_accuracy)} | "
            f"{run.final_rollout.avg_response_token_length:.1f} |"
        )
    lines.append("")
    (output_dir / "grpo_off_policy_focused_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_eval_points(runs: list[RunData], output_dir: Path) -> None:
    rows: list[list[Any]] = []
    for run in runs:
        wall_clock = dict(cumulative_eval_minutes(run))
        for point in run.eval_points:
            rows.append(
                [
                    run.label,
                    run.phase,
                    run.role,
                    run.learning_rate,
                    run.epochs_per_rollout_batch,
                    run.train_batch_size,
                    run.optimizer_updates_per_rollout_batch,
                    point.step,
                    wall_clock.get(point.step, ""),
                    point.answer_accuracy,
                    point.format_accuracy,
                    point.reward,
                    point.eval_seconds,
                ]
            )
    write_csv(
        output_dir / "grpo_off_policy_eval_points.csv",
        [
            "run",
            "phase",
            "role",
            "learning_rate",
            "epochs_per_rollout_batch",
            "train_batch_size",
            "optimizer_updates_per_rollout_batch",
            "grpo_step",
            "cumulative_eval_minutes",
            "answer_accuracy",
            "format_accuracy",
            "reward",
            "eval_seconds",
        ],
        rows,
    )


def write_diagnostics(runs: list[RunData], output_dir: Path) -> None:
    rows: list[list[Any]] = []
    for run in runs:
        rollout_by_step = {point.step: point for point in run.rollout_points}
        for train_point in run.tail_train_points:
            rollout_point = rollout_by_step.get(train_point.step)
            rows.append(
                [
                    run.label,
                    run.phase,
                    run.role,
                    train_point.step,
                    train_point.optimizer_step,
                    train_point.rollout_epoch,
                    train_point.token_entropy,
                    train_point.grad_norm,
                    train_point.loss,
                    train_point.clip_fraction if train_point.clip_fraction is not None else "",
                    train_point.approx_kl if train_point.approx_kl is not None else "",
                    rollout_point.avg_response_token_length if rollout_point is not None else "",
                    rollout_point.answer_accuracy if rollout_point is not None else "",
                    rollout_point.format_accuracy if rollout_point is not None else "",
                ]
            )
    write_csv(
        output_dir / "grpo_off_policy_diagnostics.csv",
        [
            "run",
            "phase",
            "role",
            "grpo_step",
            "optimizer_step",
            "rollout_epoch",
            "token_entropy",
            "grad_norm",
            "loss",
            "clip_fraction",
            "approx_kl",
            "rollout_avg_response_token_length",
            "rollout_answer_accuracy",
            "rollout_format_accuracy",
        ],
        rows,
    )


def write_experiment_log(all_runs: list[RunData], output_dir: Path) -> None:
    broad = broad_runs(all_runs)
    focused = focused_runs(all_runs)
    best_true_off_policy_broad = max(
        (run for run in broad if run.role == "true_off_policy"),
        key=lambda run: run.best_eval.answer_accuracy,
    )
    focused_off_policy = next(
        run for run in focused if run.role == "true_off_policy"
    )
    on_policy_reference = next(
        run for run in focused if run.role == "on_policy_reference"
    )

    lines = [
        "# GRPO Off-Policy Sweep Experiment Log",
        "",
        "## Setup",
        "",
        "- Fixed `rollout_batch_size = 256` for all broad and focused off-policy runs.",
        "- Fixed `learning_rate = 4e-5`, `loss_type = grpo_clip`, `loss_normalization = masked_mean`, and `use_std_normalization = false` based on earlier on-policy GRPO ablations.",
        "- Broad sweep varied `epochs_per_rollout_batch in {1, 2, 4}` and `train_batch_size in {256, 128}`.",
        "- We kept the `e1, tb256` point as an on-policy-style control inside the broad sweep because it uses the off-policy infrastructure but performs no rollout reuse.",
        "- These settings span optimizer-update counts of `1`, `2`, `4`, and `8` per rollout batch.",
        "",
        "## Broad Findings",
        "",
        f"- The control run `broad_e1_tb256_control` reached {percent(next(run for run in broad if run.label == 'broad_e1_tb256_control').best_eval.answer_accuracy)} by step {next(run for run in broad if run.label == 'broad_e1_tb256_control').best_eval.step}.",
        f"- Among the true off-policy broad runs, `{best_true_off_policy_broad.label}` was the strongest peak performer at {percent(best_true_off_policy_broad.best_eval.answer_accuracy)}.",
        "- Both `tb128` settings collapsed by the end of the 40-step broad sweep, while the `tb256` settings remained finite.",
        "",
        "## Focused Choice",
        "",
        f"- We selected `{focused_off_policy.label}` for the 200-step focused run because it was the most stable true off-policy configuration in the broad sweep.",
        f"- The on-policy comparison run is `{on_policy_reference.label}`, which uses `epochs_per_rollout_batch = 1`, `train_batch_size = 256`, and `reinforce_with_baseline` at the same `4e-5` learning rate.",
        "",
        "## Focused Outcome",
        "",
        f"- `{focused_off_policy.label}` peaked at {percent(focused_off_policy.best_eval.answer_accuracy)} on step {focused_off_policy.best_eval.step}, then suffered a late collapse and finished at {percent(focused_off_policy.final_eval.answer_accuracy)}.",
        f"- The on-policy reference peaked at {percent(on_policy_reference.best_eval.answer_accuracy)} and finished at {percent(on_policy_reference.final_eval.answer_accuracy)}.",
        "- Diagnostics show that the off-policy run stayed competitive for much of training but eventually hit a numerical blow-up on the second update of the rollout batch, followed by zero-reward, max-length degenerate generations.",
        "",
    ]
    (output_dir / "grpo_off_policy_experiment_log.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    broad = broad_runs(runs)
    focused = focused_runs(runs)

    if not args.skip_run_archive:
        archive_runs(runs, output_dir)

    write_run_summaries(runs, output_dir)
    write_broad_tables(broad, output_dir)
    write_focused_tables(focused, output_dir)
    write_eval_points(runs, output_dir)
    write_diagnostics(focused, output_dir)
    write_experiment_log(runs, output_dir)

    broad_answer_series = [
        (
            run.label,
            [(point.step, point.answer_accuracy) for point in run.eval_points],
        )
        for run in broad
    ]
    render_svg_line_plot(
        series=broad_answer_series,
        title="Off-policy GRPO broad sweep: validation answer reward",
        x_label="GRPO step",
        y_label="Validation answer reward",
        output_path=output_dir / "grpo_off_policy_broad_validation_reward.svg",
        y_min=0.0,
        y_max=0.7,
        x_tick_decimals=0,
        y_percent_axis=True,
    )

    focused_answer_series = [
        (
            run.label,
            [(point.step, point.answer_accuracy) for point in run.eval_points],
        )
        for run in focused
    ]
    render_svg_line_plot(
        series=focused_answer_series,
        title="On-policy vs off-policy focused comparison: validation answer reward",
        x_label="GRPO step",
        y_label="Validation answer reward",
        output_path=output_dir / "grpo_off_policy_focused_validation_reward.svg",
        y_min=0.0,
        y_max=0.85,
        x_tick_decimals=0,
        y_percent_axis=True,
    )

    focused_wall_clock_series = [
        (run.label, cumulative_eval_minutes(run))
        for run in focused
    ]
    render_svg_line_plot(
        series=focused_wall_clock_series,
        title="On-policy vs off-policy focused comparison: answer reward vs wall-clock time",
        x_label="Wall-clock time (minutes)",
        y_label="Validation answer reward",
        output_path=output_dir / "grpo_off_policy_focused_validation_reward_wall_clock.svg",
        y_min=0.0,
        y_max=0.85,
        x_tick_decimals=1,
        y_percent_axis=True,
    )

    focused_entropy_series = [
        (
            run.label,
            [
                (point.step, point.token_entropy)
                for point in run.tail_train_points
                if math.isfinite(point.token_entropy)
            ],
        )
        for run in focused
    ]
    render_svg_line_plot(
        series=focused_entropy_series,
        title="On-policy vs off-policy focused comparison: token entropy",
        x_label="GRPO step",
        y_label="Token entropy (nats)",
        output_path=output_dir / "grpo_off_policy_focused_token_entropy.svg",
        y_min=0.0,
        y_max=None,
        x_tick_decimals=0,
        y_percent_axis=False,
    )

    focused_length_series = [
        (
            run.label,
            [
                (point.step, point.avg_response_token_length)
                for point in run.rollout_points
                if math.isfinite(point.avg_response_token_length)
            ],
        )
        for run in focused
    ]
    render_svg_line_plot(
        series=focused_length_series,
        title="On-policy vs off-policy focused comparison: rollout response length",
        x_label="GRPO step",
        y_label="Average rollout response tokens",
        output_path=output_dir / "grpo_off_policy_focused_response_length.svg",
        y_min=0.0,
        y_max=None,
        x_tick_decimals=0,
        y_percent_axis=False,
    )

    print(f"Wrote GRPO off-policy sweep artifacts to {output_dir}")


if __name__ == "__main__":
    main()
