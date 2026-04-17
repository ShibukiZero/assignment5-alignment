#!/usr/bin/env python3
"""Create writeup-ready plots and tables for the Ch5 Expert Iteration experiments."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LOG_ROOTS = [
    ".agents/logs/ch5/ei_first_grid_db512_lr5e-5_bs16",
    ".agents/logs/ch5/ei_supplement_db1024_2048_budget4_lr5e-5_bs16",
]
DEFAULT_OUTPUT_DIR = "artifacts/ch5/expert_iteration"

COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
]


@dataclass(frozen=True)
class EvalPoint:
    step: int
    answer_accuracy: float
    format_accuracy: float
    reward: float
    num_examples: int


@dataclass(frozen=True)
class RolloutPoint:
    step: int
    num_questions: int
    rollouts_per_question: int
    num_rollouts: int
    num_accepted: int
    accepted_fraction: float
    question_coverage_with_accept: float
    answer_accuracy: float
    format_accuracy: float
    avg_token_entropy: float | None
    avg_accepted_token_entropy: float | None
    avg_response_token_length: float | None
    avg_accepted_response_token_length: float | None


@dataclass
class RunData:
    run_name: str
    label: str
    rollout_batch_size: int | None
    rollouts_per_question: int | None
    sft_epochs_per_step: int | None
    eval_points: list[EvalPoint]
    rollout_points: list[RolloutPoint]

    @property
    def best_eval(self) -> EvalPoint:
        return max(self.eval_points, key=lambda point: point.answer_accuracy)

    @property
    def final_eval(self) -> EvalPoint:
        return max(self.eval_points, key=lambda point: point.step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-root",
        action="append",
        dest="log_roots",
        default=None,
        help="EI sweep log root. May be passed multiple times.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-run",
        action="append",
        default=None,
        help="Optional run name filter. May be passed multiple times.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def parse_run_name(run_name: str) -> tuple[int | None, int | None, int | None]:
    match = re.search(r"db(\d+)_g(\d+)_epochs?(\d+)", run_name)
    if not match:
        return None, None, None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def label_for_run(
    run_name: str,
    db: int | None = None,
    g: int | None = None,
    epochs: int | None = None,
) -> str:
    parsed_db, parsed_g, parsed_epochs = parse_run_name(run_name)
    db = parsed_db if db is None else db
    g = parsed_g if g is None else g
    epochs = parsed_epochs if epochs is None else epochs
    if db is None or g is None or epochs is None:
        return run_name
    return f"D_b={db}, G={g}, epochs={epochs}"


def as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(float(value))


def sort_key(run: RunData) -> tuple[int, int, int, str]:
    db = run.rollout_batch_size if run.rollout_batch_size is not None else 10**9
    g = run.rollouts_per_question if run.rollouts_per_question is not None else 10**9
    epochs = run.sft_epochs_per_step if run.sft_epochs_per_step is not None else 10**9
    return db, g, epochs, run.run_name


def load_run(run_dir: Path) -> RunData | None:
    eval_paths = sorted(run_dir.glob("eval_summary_ei_step_*.json"))
    metrics_path = run_dir / "metrics.jsonl"
    if not eval_paths or not metrics_path.exists():
        return None

    db, g, epochs = parse_run_name(run_dir.name)
    config_path = run_dir / "config.json"
    if config_path.exists():
        run_config = read_json(config_path)
        if db is None:
            db = as_optional_int(run_config.get("rollout_batch_size"))
        if g is None:
            g = as_optional_int(run_config.get("rollouts_per_question"))
        if epochs is None:
            epochs = as_optional_int(run_config.get("sft_epochs_per_step"))

    eval_points: list[EvalPoint] = []
    for path in eval_paths:
        step_text = path.stem.removeprefix("eval_summary_ei_step_")
        summary = read_json(path)
        eval_points.append(
            EvalPoint(
                step=int(step_text),
                answer_accuracy=float(summary["answer_accuracy"]),
                format_accuracy=float(summary["format_accuracy"]),
                reward=float(summary.get("reward", summary["answer_accuracy"])),
                num_examples=int(summary["num_examples"]),
            )
        )

    rollout_points: list[RolloutPoint] = []
    for row in read_jsonl(metrics_path):
        if row.get("type") != "rollout":
            continue
        rollout_points.append(
            RolloutPoint(
                step=int(row["ei_step"]),
                num_questions=int(row["num_questions"]),
                rollouts_per_question=int(row["rollouts_per_question"]),
                num_rollouts=int(row["num_rollouts"]),
                num_accepted=int(row["num_accepted"]),
                accepted_fraction=float(row["accepted_fraction"]),
                question_coverage_with_accept=float(row["question_coverage_with_accept"]),
                answer_accuracy=float(row["answer_accuracy"]),
                format_accuracy=float(row["format_accuracy"]),
                avg_token_entropy=as_optional_float(row.get("avg_token_entropy")),
                avg_accepted_token_entropy=as_optional_float(
                    row.get("avg_accepted_token_entropy")
                ),
                avg_response_token_length=as_optional_float(
                    row.get("avg_response_token_length")
                ),
                avg_accepted_response_token_length=as_optional_float(
                    row.get("avg_accepted_response_token_length")
                ),
            )
        )

    if g is None and rollout_points:
        g = rollout_points[0].rollouts_per_question

    return RunData(
        run_name=run_dir.name,
        label=label_for_run(run_dir.name, db=db, g=g, epochs=epochs),
        rollout_batch_size=db,
        rollouts_per_question=g,
        sft_epochs_per_step=epochs,
        eval_points=sorted(eval_points, key=lambda point: point.step),
        rollout_points=sorted(rollout_points, key=lambda point: point.step),
    )


def load_runs(log_roots: list[Path], include_runs: set[str] | None) -> list[RunData]:
    runs: list[RunData] = []
    seen: set[str] = set()
    for root in log_roots:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name in seen:
                continue
            if include_runs is not None and child.name not in include_runs:
                continue
            run = load_run(child)
            if run is not None:
                runs.append(run)
                seen.add(child.name)
    return sorted(runs, key=sort_key)


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
        px = left if x_max == x_min else left + (x - x_min) / (x_max - x_min) * plot_width
        py = (
            top + plot_height
            if y_max == y_min
            else top + (y_max - y) / (y_max - y_min) * plot_height
        )
        coords.append(f"{px:.1f},{py:.1f}")
    return " ".join(coords)


def render_svg_line_plot(
    series: list[tuple[str, list[tuple[float, float]]]],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    y_min: float = 0.0,
    y_max: float | None = None,
) -> None:
    width = 1040
    height = 620
    left = 90
    right = 280
    top = 70
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    series = [(label, points) for label, points in series if points]
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
        f'<text x="{width / 2}" y="34" text-anchor="middle" '
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
            points,
            x_min,
            x_max,
            y_min,
            y_max,
            left,
            top,
            plot_width,
            plot_height,
        )
        parts.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" '
            'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y in points:
            point_svg = points_to_polyline(
                [(x, y)],
                x_min,
                x_max,
                y_min,
                y_max,
                left,
                top,
                plot_width,
                plot_height,
            )
            px, py = point_svg.split(",")
            parts.append(f'<circle cx="{px}" cy="{py}" r="3" fill="{color}"/>')

        legend_x = left + plot_width + 30
        legend_y = top + idx * 25
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{legend_x + 34}" y="{legend_y + 4}" '
            'font-family="Arial" font-size="13">'
            f"{html.escape(label)}</text>"
        )

    parts.append(
        f'<text x="{left + plot_width / 2}" y="{height - 24}" text-anchor="middle" '
        'font-family="Arial" font-size="15">'
        f"{html.escape(x_label)}</text>"
    )
    parts.append(
        f'<text transform="translate(24 {top + plot_height / 2}) rotate(-90)" '
        'text-anchor="middle" font-family="Arial" font-size="15">'
        f"{html.escape(y_label)}</text>"
    )
    parts.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_summary_csv(runs: list[RunData], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "rollout_batch_size",
                "rollouts_per_question",
                "sft_epochs_per_step",
                "best_step",
                "best_answer_accuracy",
                "final_step",
                "final_answer_accuracy",
                "final_format_accuracy",
                "final_accepted_fraction",
                "final_rollout_entropy",
            ],
        )
        writer.writeheader()
        for run in runs:
            best = run.best_eval
            final = run.final_eval
            final_rollout = run.rollout_points[-1] if run.rollout_points else None
            writer.writerow(
                {
                    "run": run.run_name,
                    "rollout_batch_size": run.rollout_batch_size,
                    "rollouts_per_question": run.rollouts_per_question,
                    "sft_epochs_per_step": run.sft_epochs_per_step,
                    "best_step": best.step,
                    "best_answer_accuracy": f"{best.answer_accuracy:.6f}",
                    "final_step": final.step,
                    "final_answer_accuracy": f"{final.answer_accuracy:.6f}",
                    "final_format_accuracy": f"{final.format_accuracy:.6f}",
                    "final_accepted_fraction": (
                        f"{final_rollout.accepted_fraction:.6f}" if final_rollout else ""
                    ),
                    "final_rollout_entropy": (
                        f"{final_rollout.avg_token_entropy:.6f}"
                        if final_rollout and final_rollout.avg_token_entropy is not None
                        else ""
                    ),
                }
            )


def write_summary_markdown(runs: list[RunData], output_path: Path) -> None:
    lines = [
        "# Ch5 Expert Iteration Summary",
        "",
        "| run | best answer acc | best step | final answer acc | final step | final accepted frac | final entropy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        best = run.best_eval
        final = run.final_eval
        final_rollout = run.rollout_points[-1] if run.rollout_points else None
        accepted = final_rollout.accepted_fraction if final_rollout else None
        entropy = final_rollout.avg_token_entropy if final_rollout else None
        accepted_text = f"{accepted:.4f}" if accepted is not None else ""
        entropy_text = f"{entropy:.4f}" if entropy is not None else ""
        lines.append(
            f"| {run.label} | {best.answer_accuracy:.4f} | {best.step} | "
            f"{final.answer_accuracy:.4f} | {final.step} | "
            f"{accepted_text} | {entropy_text} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_roots = [Path(path) for path in (args.log_roots or DEFAULT_LOG_ROOTS)]
    include_runs = set(args.include_run) if args.include_run else None
    runs = load_runs(log_roots, include_runs)
    if not runs:
        raise SystemExit("No EI runs found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_series = [
        (run.label, [(point.step, point.answer_accuracy) for point in run.eval_points])
        for run in runs
    ]
    render_svg_line_plot(
        accuracy_series,
        title="Expert Iteration Validation Accuracy",
        x_label="EI step",
        y_label="Validation answer accuracy",
        output_path=output_dir / "ei_validation_accuracy.svg",
        y_min=0.0,
        y_max=0.45,
    )

    entropy_series = [
        (
            run.label,
            [
                (point.step, point.avg_token_entropy)
                for point in run.rollout_points
                if point.avg_token_entropy is not None
            ],
        )
        for run in runs
    ]
    render_svg_line_plot(
        entropy_series,
        title="Expert Iteration Response Entropy",
        x_label="EI step",
        y_label="Average rollout token entropy",
        output_path=output_dir / "ei_rollout_entropy.svg",
        y_min=0.0,
        y_max=None,
    )

    accepted_series = [
        (
            run.label,
            [(point.step, point.accepted_fraction) for point in run.rollout_points],
        )
        for run in runs
    ]
    render_svg_line_plot(
        accepted_series,
        title="Expert Iteration Accepted Rollout Fraction",
        x_label="EI step",
        y_label="Accepted rollout fraction",
        output_path=output_dir / "ei_accepted_fraction.svg",
        y_min=0.0,
        y_max=0.40,
    )

    write_summary_csv(runs, output_dir / "ei_results_summary.csv")
    write_summary_markdown(runs, output_dir / "ei_results_summary.md")

    print(f"Wrote EI plots and tables to {output_dir}")


if __name__ == "__main__":
    main()
