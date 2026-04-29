#!/usr/bin/env python3
"""Create writeup-ready plots and archives for the repaired GRPO LR sweep."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "artifacts/experiments/ch7/grpo_learning_rate"
TARGET_STEPS = 200

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

DEFAULT_RUNS = [
    (
        "lr=3e-6",
        3e-6,
        ".agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr3e-6",
    ),
    (
        "lr=5e-6",
        5e-6,
        ".agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr5e-6",
    ),
    (
        "lr=1e-5",
        1e-5,
        ".agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr1e-5",
    ),
    (
        "lr=2e-5",
        2e-5,
        ".agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr2e-5",
    ),
    (
        "lr=3e-5",
        3e-5,
        ".agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr3e-5",
    ),
    (
        "lr=4e-5",
        4e-5,
        ".agents/logs/ch7/grpo_on_policy_ablations/length_normalization_rerun_staging_single_gpu/masked_mean_lr4e-5",
    ),
    (
        "lr=5e-5",
        5e-5,
        ".agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr5e-5",
    ),
    (
        "lr=7e-5",
        7e-5,
        ".agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr7e-5",
    ),
    (
        "lr=2e-4",
        2e-4,
        ".agents/logs/reruns/prefix_cache_repair_single_gpu/grpo/lr2e-4",
    ),
]


@dataclass(frozen=True)
class EvalPoint:
    step: int
    answer_accuracy: float
    format_accuracy: float
    reward: float
    num_examples: int | None


@dataclass
class RunData:
    label: str
    learning_rate: float
    log_dir: Path
    eval_points: list[EvalPoint]

    @property
    def best_eval(self) -> EvalPoint:
        return max(self.eval_points, key=lambda point: point.answer_accuracy)

    @property
    def final_eval(self) -> EvalPoint:
        return max(self.eval_points, key=lambda point: point.step)

    @property
    def status(self) -> str:
        if self.final_eval.step >= TARGET_STEPS:
            return "completed"
        if self.final_eval.answer_accuracy == 0 and self.final_eval.format_accuracy == 0:
            return "collapsed"
        return "stopped early"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-steps",
        type=int,
        default=TARGET_STEPS,
        help="Expected full-run GRPO step count, used only for status labels.",
    )
    parser.add_argument(
        "--include-run",
        action="append",
        default=None,
        help=(
            "Optional label=learning_rate=log_dir entry. "
            "Example: --include-run extra-lr=4e-5=.agents/logs/..."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_extra_run(spec: str) -> tuple[str, float, str]:
    parts = spec.rsplit("=", maxsplit=2)
    if len(parts) != 3:
        raise ValueError(
            "--include-run must have the form label=learning_rate=log_dir"
        )
    label, learning_rate, log_dir = parts
    return label, float(learning_rate), log_dir


def load_run(label: str, learning_rate: float, log_dir: Path) -> RunData | None:
    metrics_path = log_dir / "metrics.jsonl"
    if not metrics_path.exists():
        print(f"Skipping missing metrics file: {metrics_path}")
        return None

    eval_points: list[EvalPoint] = []
    for row in read_jsonl(metrics_path):
        if row.get("type") != "eval":
            continue
        eval_points.append(
            EvalPoint(
                step=int(row["grpo_step"]),
                answer_accuracy=float(row["answer_accuracy"]),
                format_accuracy=float(row["format_accuracy"]),
                reward=float(row.get("reward", row["answer_accuracy"])),
                num_examples=(
                    int(row["num_examples"]) if row.get("num_examples") is not None else None
                ),
            )
        )

    if not eval_points:
        print(f"Skipping run without eval rows: {metrics_path}")
        return None

    return RunData(
        label=label,
        learning_rate=learning_rate,
        log_dir=log_dir,
        eval_points=sorted(eval_points, key=lambda point: point.step),
    )


def load_runs(extra_specs: list[str] | None) -> list[RunData]:
    run_specs = list(DEFAULT_RUNS)
    if extra_specs:
        run_specs.extend(parse_extra_run(spec) for spec in extra_specs)

    runs: list[RunData] = []
    for label, learning_rate, log_dir in run_specs:
        run = load_run(label=label, learning_rate=learning_rate, log_dir=Path(log_dir))
        if run is not None:
            runs.append(run)

    return sorted(runs, key=lambda run: run.learning_rate)


def percent(value: float) -> float:
    return 100.0 * value


def nice_y_max(max_value: float) -> float:
    if max_value <= 0:
        return 0.1
    return min(1.0, max(0.1, math.ceil(max_value * 10) / 10))


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


def write_eval_points_csv(runs: list[RunData], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "label",
                "learning_rate",
                "status",
                "grpo_step",
                "answer_accuracy",
                "format_accuracy",
                "reward",
                "num_examples",
                "log_dir",
            ]
        )
        for run in runs:
            for point in run.eval_points:
                writer.writerow(
                    [
                        run.label,
                        f"{run.learning_rate:.8g}",
                        run.status,
                        point.step,
                        f"{point.answer_accuracy:.10f}",
                        f"{point.format_accuracy:.10f}",
                        f"{point.reward:.10f}",
                        point.num_examples if point.num_examples is not None else "",
                        run.log_dir,
                    ]
                )


def write_summary(runs: list[RunData], csv_path: Path, md_path: Path) -> None:
    headers = [
        "label",
        "learning_rate",
        "status",
        "best_answer_accuracy",
        "best_step",
        "best_format_accuracy",
        "final_answer_accuracy",
        "final_step",
        "final_format_accuracy",
        "log_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(headers)
        for run in runs:
            best = run.best_eval
            final = run.final_eval
            writer.writerow(
                [
                    run.label,
                    f"{run.learning_rate:.8g}",
                    run.status,
                    f"{best.answer_accuracy:.10f}",
                    best.step,
                    f"{best.format_accuracy:.10f}",
                    f"{final.answer_accuracy:.10f}",
                    final.step,
                    f"{final.format_accuracy:.10f}",
                    run.log_dir,
                ]
            )

    best_checkpoint_run = max(runs, key=lambda run: run.best_eval.answer_accuracy)
    best_final_run = max(runs, key=lambda run: run.final_eval.answer_accuracy)

    lines = [
        "# GRPO Learning Rate Sweep Summary",
        "",
        "| learning rate | status | best answer | best step | final answer | final step | final format |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        best = run.best_eval
        final = run.final_eval
        lines.append(
            "| "
            f"`{run.label.removeprefix('lr=')}` | "
            f"{run.status} | "
            f"{percent(best.answer_accuracy):.2f}% | "
            f"{best.step} | "
            f"{percent(final.answer_accuracy):.2f}% | "
            f"{final.step} | "
            f"{percent(final.format_accuracy):.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Best checkpoint: `{best_checkpoint_run.label}` at "
            f"{percent(best_checkpoint_run.best_eval.answer_accuracy):.2f}% "
            f"answer reward on step {best_checkpoint_run.best_eval.step}.",
            f"Best final checkpoint: `{best_final_run.label}` at "
            f"{percent(best_final_run.final_eval.answer_accuracy):.2f}% "
            f"answer reward on step {best_final_run.final_eval.step}.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def write_run_summaries(runs: list[RunData], json_path: Path, archive_path: Path) -> None:
    payload: list[dict[str, Any]] = []
    for run in runs:
        best = run.best_eval
        final = run.final_eval
        payload.append(
            {
                "learning_rate": run.learning_rate,
                "label": run.label,
                "status": run.status,
                "best_answer_reward": best.answer_accuracy,
                "best_step": best.step,
                "best_format_accuracy": best.format_accuracy,
                "final_answer_reward": final.answer_accuracy,
                "final_step": final.step,
                "final_format_accuracy": final.format_accuracy,
                "source_log_dir": str(run.log_dir),
            }
        )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    lines = [
        "# GRPO Learning Rate Run Summaries Archive",
        "",
        "This archive summarizes the repaired-infra learning-rate runs used for "
        "the `grpo_learning_rate` writeup section. It intentionally ignores "
        "the older pre-repair sweep artifacts.",
        "",
        "| learning rate | status | best answer | best step | best format | final answer | final step | final format | source log |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run_summary in payload:
        lines.append(
            "| "
            f"`{run_summary['label'].removeprefix('lr=')}` | "
            f"{run_summary['status']} | "
            f"{run_summary['best_answer_reward']:.10f} | "
            f"{run_summary['best_step']} | "
            f"{run_summary['best_format_accuracy']:.10f} | "
            f"{run_summary['final_answer_reward']:.10f} | "
            f"{run_summary['final_step']} | "
            f"{run_summary['final_format_accuracy']:.10f} | "
            f"`{run_summary['source_log_dir']}` |"
        )
    lines.extend(
        [
            "",
            "Generated artifacts:",
            "",
            "- `artifacts/experiments/ch7/grpo_learning_rate/grpo_learning_rate_validation_reward.svg`",
            "- `artifacts/experiments/ch7/grpo_learning_rate/grpo_learning_rate_format_accuracy.svg`",
            "- `artifacts/experiments/ch7/grpo_learning_rate/grpo_learning_rate_eval_points.csv`",
            "- `artifacts/experiments/ch7/grpo_learning_rate/grpo_learning_rate_summary.csv`",
            "- `artifacts/experiments/ch7/grpo_learning_rate/grpo_learning_rate_summary.md`",
            "- `artifacts/experiments/ch7/grpo_learning_rate/run_summaries.json`",
        ]
    )
    archive_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    global TARGET_STEPS
    TARGET_STEPS = args.target_steps

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(extra_specs=args.include_run)
    if not runs:
        raise SystemExit("No GRPO learning-rate runs found.")

    answer_series = [
        (
            f"{run.label}{' (stopped)' if run.final_eval.step < TARGET_STEPS else ''}",
            [(point.step, point.answer_accuracy) for point in run.eval_points],
        )
        for run in runs
    ]
    format_series = [
        (
            f"{run.label}{' (stopped)' if run.final_eval.step < TARGET_STEPS else ''}",
            [(point.step, point.format_accuracy) for point in run.eval_points],
        )
        for run in runs
    ]

    render_svg_line_plot(
        series=answer_series,
        title="GRPO Learning-Rate Sweep: Validation Answer Reward",
        x_label="GRPO step",
        y_label="Validation answer reward",
        output_path=output_dir / "grpo_learning_rate_validation_reward.svg",
        y_min=0.0,
    )
    render_svg_line_plot(
        series=format_series,
        title="GRPO Learning-Rate Sweep: Validation Format Accuracy",
        x_label="GRPO step",
        y_label="Validation format accuracy",
        output_path=output_dir / "grpo_learning_rate_format_accuracy.svg",
        y_min=0.0,
        y_max=1.0,
    )
    write_eval_points_csv(
        runs=runs,
        output_path=output_dir / "grpo_learning_rate_eval_points.csv",
    )
    write_summary(
        runs=runs,
        csv_path=output_dir / "grpo_learning_rate_summary.csv",
        md_path=output_dir / "grpo_learning_rate_summary.md",
    )
    write_run_summaries(
        runs=runs,
        json_path=output_dir / "run_summaries.json",
        archive_path=output_dir / "run_summaries_archive.md",
    )

    print(f"Wrote GRPO learning-rate artifacts to {output_dir}")


if __name__ == "__main__":
    main()
