#!/usr/bin/env python3
"""Create writeup-ready plots and tables for the Ch4 SFT experiments."""

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


DEFAULT_SIZE_SWEEP_LOG_ROOT = ".agents/logs/ch4/sft_noisy_size_sweep_bs16_lr5e-5"
DEFAULT_FILTERED_LOG_ROOT = ".agents/logs/ch4/sft_noisy_filtered_bs16_lr5e-5"
DEFAULT_OUTPUT_DIR = "artifacts/experiments/ch4/sft_experiment"

COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
]


@dataclass(frozen=True)
class EvalPoint:
    step: int
    answer_accuracy: float
    format_accuracy: float
    reward: float
    num_examples: int


@dataclass
class RunData:
    run_name: str
    label: str
    num_train_examples: int | None
    eval_points: list[EvalPoint]
    train_losses: list[tuple[int, float]]
    final_train_loss: float | None

    @property
    def best_eval(self) -> EvalPoint:
        return max(self.eval_points, key=lambda point: point.answer_accuracy)

    @property
    def final_eval(self) -> EvalPoint:
        return max(self.eval_points, key=lambda point: point.step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-sweep-log-root", default=DEFAULT_SIZE_SWEEP_LOG_ROOT)
    parser.add_argument("--filtered-log-root", default=DEFAULT_FILTERED_LOG_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
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


def label_for_run(run_name: str, num_train_examples: int | None) -> str:
    if run_name.startswith("filtered_full"):
        return "Filtered full"
    if run_name.startswith("sizefull"):
        return "Noisy full"
    match = re.match(r"size(\d+)_", run_name)
    if match:
        return f"{match.group(1)} examples"
    if num_train_examples is not None:
        return f"{num_train_examples} examples"
    return run_name


def sort_key(run: RunData) -> tuple[int, int, str]:
    if run.run_name.startswith("filtered_full"):
        return (1, 0, run.run_name)
    if run.run_name.startswith("sizefull"):
        return (0, 10**9, run.run_name)
    if run.num_train_examples is not None:
        return (0, run.num_train_examples, run.run_name)
    return (0, 10**8, run.run_name)


def load_run(run_dir: Path) -> RunData | None:
    eval_paths = sorted(run_dir.glob("eval_summary_step_*.json"))
    metrics_path = run_dir / "metrics.jsonl"
    if not eval_paths or not metrics_path.exists():
        return None

    eval_points: list[EvalPoint] = []
    for path in eval_paths:
        step_text = path.stem.removeprefix("eval_summary_step_")
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

    metrics = read_jsonl(metrics_path)
    train_rows = [row for row in metrics if row.get("type") == "train"]
    train_losses = [(int(row["train_step"]), float(row["loss"])) for row in train_rows]
    final_train_loss = train_losses[-1][1] if train_losses else None
    num_train_examples = None
    if train_rows:
        num_train_examples = int(train_rows[-1]["num_train_examples"])

    return RunData(
        run_name=run_dir.name,
        label=label_for_run(run_dir.name, num_train_examples),
        num_train_examples=num_train_examples,
        eval_points=sorted(eval_points, key=lambda point: point.step),
        train_losses=train_losses,
        final_train_loss=final_train_loss,
    )


def load_runs(log_root: Path) -> list[RunData]:
    runs: list[RunData] = []
    for child in sorted(log_root.iterdir()):
        if child.is_dir():
            run = load_run(child)
            if run is not None:
                runs.append(run)
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
        if x_max == x_min:
            px = left
        else:
            px = left + (x - x_min) / (x_max - x_min) * plot_width
        if y_max == y_min:
            py = top + plot_height
        else:
            py = top + (y_max - y) / (y_max - y_min) * plot_height
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
    width = 960
    height = 600
    left = 90
    right = 230
    top = 70
    bottom = 80
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
            f'<line x1="{left - 5}" y1="{py:.1f}" x2="{left + plot_width}" y2="{py:.1f}" '
            'stroke="#ddd" stroke-width="1"/>'
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
        legend_y = top + idx * 26
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
                "label",
                "num_train_examples",
                "best_step",
                "best_answer_accuracy",
                "best_format_accuracy",
                "final_step",
                "final_answer_accuracy",
                "final_format_accuracy",
                "final_train_loss",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for run in runs:
            best = run.best_eval
            final = run.final_eval
            writer.writerow(
                {
                    "run": run.run_name,
                    "label": run.label,
                    "num_train_examples": run.num_train_examples,
                    "best_step": best.step,
                    "best_answer_accuracy": f"{best.answer_accuracy:.6f}",
                    "best_format_accuracy": f"{best.format_accuracy:.6f}",
                    "final_step": final.step,
                    "final_answer_accuracy": f"{final.answer_accuracy:.6f}",
                    "final_format_accuracy": f"{final.format_accuracy:.6f}",
                    "final_train_loss": (
                        f"{run.final_train_loss:.6f}" if run.final_train_loss is not None else ""
                    ),
                }
            )


def write_summary_markdown(runs: list[RunData], output_path: Path) -> None:
    lines = [
        "# Ch4 SFT Experiment Summary",
        "",
        "| run | train examples | best answer acc | best step | final answer acc | final step |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        best = run.best_eval
        final = run.final_eval
        num_examples = run.num_train_examples if run.num_train_examples is not None else "full"
        lines.append(
            f"| {run.label} | {num_examples} | {best.answer_accuracy:.4f} | {best.step} | "
            f"{final.answer_accuracy:.4f} | {final.step} |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    size_runs = load_runs(Path(args.size_sweep_log_root))
    filtered_runs = load_runs(Path(args.filtered_log_root))
    all_runs = sorted(size_runs + filtered_runs, key=sort_key)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    size_series = [
        (
            run.label,
            [(point.step, point.answer_accuracy) for point in run.eval_points],
        )
        for run in size_runs
    ]
    render_svg_line_plot(
        series=size_series,
        title="SFT Validation Accuracy by Dataset Size",
        x_label="Optimizer step",
        y_label="Validation answer accuracy",
        output_path=output_dir / "sft_size_sweep_accuracy.svg",
        y_min=0.0,
        y_max=0.45,
    )

    full_noisy = [run for run in size_runs if run.run_name.startswith("sizefull")]
    comparison_runs = full_noisy + filtered_runs
    comparison_series = [
        (
            run.label,
            [(point.step, point.answer_accuracy) for point in run.eval_points],
        )
        for run in comparison_runs
    ]
    render_svg_line_plot(
        series=comparison_series,
        title="Filtered SFT vs Noisy Full SFT",
        x_label="Optimizer step",
        y_label="Validation answer accuracy",
        output_path=output_dir / "sft_filtered_vs_noisy_full_accuracy.svg",
        y_min=0.0,
        y_max=0.40,
    )

    write_summary_csv(all_runs, output_dir / "sft_results_summary.csv")
    write_summary_markdown(all_runs, output_dir / "sft_results_summary.md")

    print(f"Wrote SFT plots and tables to {output_dir}")


if __name__ == "__main__":
    main()
