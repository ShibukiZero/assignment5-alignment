#!/usr/bin/env python3
"""Create writeup-ready artifacts for the Ch7 GRPO baseline ablation."""

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


DEFAULT_OUTPUT_DIR = "artifacts/experiments/ch7/grpo_baselines"
TARGET_STEPS = 200

DEFAULT_RUN_SPECS = [
    (
        "reinforce_with_baseline",
        ".agents/logs/ch7/grpo_on_policy_ablations/length_normalization_rerun_staging_single_gpu/masked_mean_lr4e-5",
    ),
    (
        "no_baseline",
        ".agents/logs/reruns/grpo_ablation_repairs_single_gpu/grpo_baselines/no_baseline",
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
ARCHIVE_IGNORE = shutil.ignore_patterns("sample_rollouts.jsonl")


@dataclass(frozen=True)
class EvalPoint:
    step: int
    answer_accuracy: float
    format_accuracy: float
    reward: float
    num_examples: int | None


@dataclass(frozen=True)
class TrainPoint:
    step: int
    loss: float
    grad_norm: float
    token_entropy: float
    train_answer_accuracy: float
    train_format_accuracy: float


@dataclass(frozen=True)
class RolloutPoint:
    step: int
    answer_accuracy: float
    format_accuracy: float
    avg_response_token_length: float


@dataclass
class RunData:
    label: str
    log_dir: Path
    config: dict[str, Any]
    run_summary: dict[str, Any]
    eval_points: list[EvalPoint]
    train_points: list[TrainPoint]
    rollout_points: list[RolloutPoint]

    @property
    def loss_type(self) -> str:
        return str(self.config.get("loss_type", self.label))

    @property
    def learning_rate(self) -> float:
        return float(self.config["learning_rate"])

    @property
    def best_eval(self) -> EvalPoint:
        return max(self.eval_points, key=lambda point: point.answer_accuracy)

    @property
    def final_eval(self) -> EvalPoint:
        return max(self.eval_points, key=lambda point: point.step)

    @property
    def final_train(self) -> TrainPoint:
        return max(self.train_points, key=lambda point: point.step)

    @property
    def final_rollout(self) -> RolloutPoint:
        return max(self.rollout_points, key=lambda point: point.step)

    @property
    def status(self) -> str:
        if self.final_eval.step >= TARGET_STEPS:
            return "completed"
        return "stopped early"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-run",
        action="append",
        default=None,
        help=(
            "Optional label=log_dir entry. "
            "Example: --include-run no_baseline=.agents/logs/..."
        ),
    )
    parser.add_argument(
        "--skip-run-archive",
        action="store_true",
        help="Generate plots and summaries without copying raw runs into runs/.",
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


def parse_extra_run(spec: str) -> tuple[str, str]:
    parts = spec.split("=", maxsplit=1)
    if len(parts) != 2:
        raise ValueError("--include-run must have the form label=log_dir")
    return parts[0], parts[1]


def default_run_specs(output_dir: Path) -> list[tuple[str, str]]:
    del output_dir
    return list(DEFAULT_RUN_SPECS)


def load_run(label: str, log_dir: Path) -> RunData:
    metrics_path = log_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    config = read_json(log_dir / "config.json")
    run_summary = read_json(log_dir / "run_summary.json")
    eval_points: list[EvalPoint] = []
    train_points: list[TrainPoint] = []
    rollout_points: list[RolloutPoint] = []

    for row in read_jsonl(metrics_path):
        row_type = row.get("type")
        if row_type == "eval":
            eval_points.append(
                EvalPoint(
                    step=int(row["grpo_step"]),
                    answer_accuracy=float(row["answer_accuracy"]),
                    format_accuracy=float(row["format_accuracy"]),
                    reward=float(row.get("reward", row["answer_accuracy"])),
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
                    loss=float(row["loss"]),
                    grad_norm=float(row["grad_norm"]),
                    token_entropy=float(row["token_entropy"]),
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
        log_dir=log_dir,
        config=config,
        run_summary=run_summary,
        eval_points=sorted(eval_points, key=lambda point: point.step),
        train_points=sorted(train_points, key=lambda point: point.step),
        rollout_points=sorted(rollout_points, key=lambda point: point.step),
    )


def load_runs(output_dir: Path, extra_specs: list[str] | None) -> list[RunData]:
    run_specs = default_run_specs(output_dir)
    if extra_specs:
        run_specs.extend(parse_extra_run(spec) for spec in extra_specs)
    return [load_run(label, Path(log_dir)) for label, log_dir in run_specs]


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def nice_y_max(max_value: float) -> float:
    if max_value <= 0:
        return 1.0
    return math.ceil(max_value * 10) / 10


def points_to_polyline(
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


def point_to_svg_xy(
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


def render_svg_line_plot(
    *,
    series: list[tuple[str, list[tuple[float, float]]]],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    y_min: float = 0.0,
    y_max: float | None = None,
) -> None:
    width = 1040
    height = 640
    left = 95
    right = 260
    top = 74
    bottom = 82
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_points = [point for _, points in series for point in points]
    if not all_points:
        raise ValueError(f"No data to plot for {title}")

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
        parts.append(
            f'<line x1="{left - 5}" y1="{py:.1f}" x2="{left + plot_width}" '
            f'y2="{py:.1f}" stroke="#ddd" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{py + 4:.1f}" text-anchor="end" '
            'font-family="Arial" font-size="12">'
            f"{y_value:.2f}</text>"
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
            f"{int(round(x_value))}</text>"
        )

    for idx, (label, points) in enumerate(series):
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
        legend_y = top + 22 + idx * 25
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
        shutil.copytree(run.log_dir, archive_dir, dirs_exist_ok=True, ignore=ARCHIVE_IGNORE)


def write_eval_points(runs: list[RunData], output_dir: Path) -> None:
    rows: list[list[Any]] = []
    for run in runs:
        for point in run.eval_points:
            rows.append(
                [
                    run.label,
                    run.loss_type,
                    run.learning_rate,
                    point.step,
                    point.answer_accuracy,
                    point.format_accuracy,
                    point.reward,
                    point.num_examples,
                ]
            )
    write_csv(
        output_dir / "grpo_baselines_eval_points.csv",
        [
            "run",
            "loss_type",
            "learning_rate",
            "grpo_step",
            "answer_accuracy",
            "format_accuracy",
            "reward",
            "num_examples",
        ],
        rows,
    )


def write_train_points(runs: list[RunData], output_dir: Path) -> None:
    rows: list[list[Any]] = []
    for run in runs:
        for point in run.train_points:
            rows.append(
                [
                    run.label,
                    run.loss_type,
                    run.learning_rate,
                    point.step,
                    point.loss,
                    point.grad_norm,
                    point.token_entropy,
                    point.train_answer_accuracy,
                    point.train_format_accuracy,
                ]
            )
    write_csv(
        output_dir / "grpo_baselines_train_points.csv",
        [
            "run",
            "loss_type",
            "learning_rate",
            "grpo_step",
            "loss",
            "grad_norm",
            "token_entropy",
            "train_answer_accuracy",
            "train_format_accuracy",
        ],
        rows,
    )


def write_rollout_points(runs: list[RunData], output_dir: Path) -> None:
    rows: list[list[Any]] = []
    for run in runs:
        for point in run.rollout_points:
            rows.append(
                [
                    run.label,
                    run.loss_type,
                    run.learning_rate,
                    point.step,
                    point.answer_accuracy,
                    point.format_accuracy,
                    point.avg_response_token_length,
                ]
            )
    write_csv(
        output_dir / "grpo_baselines_rollout_points.csv",
        [
            "run",
            "loss_type",
            "learning_rate",
            "grpo_step",
            "rollout_answer_accuracy",
            "rollout_format_accuracy",
            "avg_response_token_length",
        ],
        rows,
    )


def write_summary_files(runs: list[RunData], output_dir: Path) -> None:
    summary_rows: list[list[Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for run in runs:
        summary_rows.append(
            [
                run.label,
                run.loss_type,
                run.learning_rate,
                run.status,
                run.best_eval.answer_accuracy,
                run.best_eval.step,
                run.final_eval.answer_accuracy,
                run.final_eval.step,
                run.final_eval.format_accuracy,
                run.final_rollout.answer_accuracy,
                run.final_rollout.format_accuracy,
                run.final_rollout.avg_response_token_length,
                run.final_train.token_entropy,
                run.final_train.grad_norm,
            ]
        )
        run_summaries.append(
            {
                "run_name": run.label,
                "loss_type": run.loss_type,
                "learning_rate": run.learning_rate,
                "status": run.status,
                "source_log_dir": str(run.log_dir),
                "archived_run_dir": str(output_dir / "runs" / run.label),
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
                    "avg_response_token_length": (
                        run.final_rollout.avg_response_token_length
                    ),
                },
                "final_train": {
                    "grpo_step": run.final_train.step,
                    "token_entropy": run.final_train.token_entropy,
                    "grad_norm": run.final_train.grad_norm,
                    "loss": run.final_train.loss,
                },
                "run_summary": run.run_summary,
            }
        )

    write_csv(
        output_dir / "grpo_baselines_summary.csv",
        [
            "run",
            "loss_type",
            "learning_rate",
            "status",
            "best_answer_accuracy",
            "best_step",
            "final_answer_accuracy",
            "final_step",
            "final_format_accuracy",
            "final_rollout_answer_accuracy",
            "final_rollout_format_accuracy",
            "final_rollout_avg_response_token_length",
            "final_token_entropy",
            "final_grad_norm",
        ],
        summary_rows,
    )
    (output_dir / "run_summaries.json").write_text(
        json.dumps(run_summaries, indent=2) + "\n",
        encoding="utf-8",
    )

    markdown_lines = [
        "# GRPO Baselines Summary",
        "",
        (
            "| run | loss type | status | best answer | best step | final answer | "
            "final format | final rollout answer | final avg length |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        markdown_lines.append(
            (
                f"| `{run.label}` | `{run.loss_type}` | {run.status} | "
                f"{percent(run.best_eval.answer_accuracy)} | {run.best_eval.step} | "
                f"{percent(run.final_eval.answer_accuracy)} | "
                f"{percent(run.final_eval.format_accuracy)} | "
                f"{percent(run.final_rollout.answer_accuracy)} | "
                f"{run.final_rollout.avg_response_token_length:.1f} |"
            )
        )
    markdown_lines.extend(
        [
            "",
            "Default source logs:",
            "",
            *[
                f"- `{label}`: `{log_dir}`"
                for label, log_dir in DEFAULT_RUN_SPECS
            ],
            "",
            (
                "Raw run files are archived under `artifacts/experiments/ch7/grpo_baselines/runs/`. "
                "`sample_rollouts.jsonl` files are intentionally omitted because aggregate "
                "rollout summaries are sufficient for the writeup."
            ),
        ]
    )
    (output_dir / "run_summaries_archive.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(output_dir, args.include_run)

    if not args.skip_run_archive:
        archive_runs(runs, output_dir)

    write_eval_points(runs, output_dir)
    write_train_points(runs, output_dir)
    write_rollout_points(runs, output_dir)
    write_summary_files(runs, output_dir)

    answer_series = [
        (
            run.label,
            [(point.step, point.answer_accuracy) for point in run.eval_points],
        )
        for run in runs
    ]
    format_series = [
        (
            run.label,
            [(point.step, point.format_accuracy) for point in run.eval_points],
        )
        for run in runs
    ]
    entropy_series = [
        (
            run.label,
            [(point.step, point.token_entropy) for point in run.train_points],
        )
        for run in runs
    ]
    response_length_series = [
        (
            run.label,
            [
                (point.step, point.avg_response_token_length)
                for point in run.rollout_points
            ],
        )
        for run in runs
    ]

    render_svg_line_plot(
        series=answer_series,
        title="GRPO validation answer reward by baseline choice",
        x_label="GRPO step",
        y_label="validation answer reward",
        output_path=output_dir / "grpo_baselines_validation_reward.svg",
    )
    render_svg_line_plot(
        series=format_series,
        title="GRPO validation format accuracy by baseline choice",
        x_label="GRPO step",
        y_label="validation format accuracy",
        output_path=output_dir / "grpo_baselines_format_accuracy.svg",
        y_max=1.0,
    )
    render_svg_line_plot(
        series=entropy_series,
        title="GRPO token entropy by baseline choice",
        x_label="GRPO step",
        y_label="token entropy (nats)",
        output_path=output_dir / "grpo_baselines_token_entropy.svg",
        y_min=0.0,
    )
    render_svg_line_plot(
        series=response_length_series,
        title="GRPO rollout response length by baseline choice",
        x_label="GRPO step",
        y_label="average rollout response tokens",
        output_path=output_dir / "grpo_baselines_response_length.svg",
        y_min=0.0,
    )


if __name__ == "__main__":
    main()
