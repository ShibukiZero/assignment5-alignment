#!/usr/bin/env python3
"""Create writeup-ready plots and tables for supplement Chapter 3 SFT."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = "runs/logs/ch3/sft"
DEFAULT_OUTPUT_DIR = "artifacts/experiments/supplement/ch3/sft"


@dataclass(frozen=True)
class EvalPoint:
    step: int
    loss: float
    perplexity: float
    num_tokens: int
    eval_seconds: float
    final: bool


@dataclass(frozen=True)
class TrainPoint:
    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-train-points",
        type=int,
        default=1200,
        help="Maximum train points to draw in the SVG curve.",
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_points(metrics_path: Path) -> tuple[list[TrainPoint], list[EvalPoint]]:
    train_points: list[TrainPoint] = []
    eval_points: list[EvalPoint] = []
    for row in read_jsonl(metrics_path):
        if row.get("type") == "train":
            train_points.append(
                TrainPoint(
                    step=int(row["step"]),
                    loss=float(row["loss"]),
                    learning_rate=float(row["learning_rate"]),
                    grad_norm=float(row["grad_norm"]),
                    elapsed_seconds=float(row["elapsed_seconds"]),
                )
            )
        elif row.get("type") == "eval":
            eval_points.append(
                EvalPoint(
                    step=int(row["step"]),
                    loss=float(row["loss"]),
                    perplexity=float(row["perplexity"]),
                    num_tokens=int(row["num_tokens"]),
                    eval_seconds=float(row["eval_seconds"]),
                    final=bool(row.get("final", False)),
                )
            )
    if not train_points:
        raise ValueError(f"No train rows found in {metrics_path}")
    if not eval_points:
        raise ValueError(f"No eval rows found in {metrics_path}")
    return train_points, eval_points


def downsample_train_points(points: list[TrainPoint], max_points: int) -> list[TrainPoint]:
    if max_points <= 0 or len(points) <= max_points:
        return points
    stride = math.ceil(len(points) / max_points)
    sampled = points[::stride]
    if sampled[-1].step != points[-1].step:
        sampled.append(points[-1])
    return sampled


def nice_range(values: list[float]) -> tuple[float, float]:
    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        pad = 0.1 if min_value == 0 else abs(min_value) * 0.05
        return min_value - pad, max_value + pad
    pad = (max_value - min_value) * 0.08
    return min_value - pad, max_value + pad


def to_svg_polyline(
    points: list[tuple[float, float]],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: int,
    top: int,
    width: int,
    height: int,
) -> str:
    coords: list[str] = []
    for x, y in points:
        px = left if x_max == x_min else left + (x - x_min) / (x_max - x_min) * width
        py = top if y_max == y_min else top + (y_max - y) / (y_max - y_min) * height
        coords.append(f"{px:.1f},{py:.1f}")
    return " ".join(coords)


def render_svg(
    title: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float]], str]],
    *,
    width: int = 900,
    height: int = 520,
) -> str:
    left, right, top, bottom = 82, 32, 48, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_x = [x for _, points, _ in series for x, _ in points]
    all_y = [y for _, points, _ in series for _, y in points]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = nice_range(all_y)
    x_ticks = [x_min + (x_max - x_min) * i / 4 for i in range(5)]
    y_ticks = [y_min + (y_max - y_min) * i / 4 for i in range(5)]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="26" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222" stroke-width="1.2"/>',
    ]
    for tick in x_ticks:
        px = left if x_max == x_min else left + (tick - x_min) / (x_max - x_min) * plot_width
        lines.extend(
            [
                f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_height}" stroke="#eee" stroke-width="1"/>',
                f'<text x="{px:.1f}" y="{top + plot_height + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">{tick:.0f}</text>',
            ]
        )
    for tick in y_ticks:
        py = top if y_max == y_min else top + (y_max - tick) / (y_max - y_min) * plot_height
        lines.extend(
            [
                f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_width}" y2="{py:.1f}" stroke="#eee" stroke-width="1"/>',
                f'<text x="{left - 10}" y="{py + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{tick:.3f}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 20}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Optimizer step</text>',
            f'<text transform="translate(22,{top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">{html.escape(y_label)}</text>',
        ]
    )
    legend_y = top + 18
    for index, (label, points, color) in enumerate(series):
        polyline = to_svg_polyline(
            points,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            left=left,
            top=top,
            width=plot_width,
            height=plot_height,
        )
        lines.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        lx = left + 18 + index * 190
        lines.extend(
            [
                f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 24}" y2="{legend_y}" stroke="{color}" stroke-width="2.4"/>',
                f'<text x="{lx + 32}" y="{legend_y + 4}" font-family="Arial, sans-serif" font-size="13">{html.escape(label)}</text>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_eval_tables(output_dir: Path, eval_points: list[EvalPoint]) -> None:
    json_rows = [
        {
            "step": point.step,
            "validation_loss": point.loss,
            "validation_perplexity": point.perplexity,
            "num_tokens": point.num_tokens,
            "eval_seconds": point.eval_seconds,
            "final": point.final,
        }
        for point in eval_points
    ]
    write_json(output_dir / "sft_eval_points.json", json_rows)

    with (output_dir / "sft_eval_points.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "validation_loss",
                "validation_perplexity",
                "num_tokens",
                "eval_seconds",
                "final",
            ],
        )
        writer.writeheader()
        writer.writerows(json_rows)

    lines = [
        "| step | validation loss | validation perplexity | final |",
        "|---:|---:|---:|:---:|",
    ]
    for point in eval_points:
        lines.append(
            f"| {point.step} | {point.loss:.4f} | {point.perplexity:.4f} | {'yes' if point.final else 'no'} |"
        )
    (output_dir / "sft_eval_points.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(output_dir: Path, run_summary: dict[str, Any], eval_points: list[EvalPoint]) -> None:
    best_point = min(eval_points, key=lambda point: point.loss)
    summary = {
        **run_summary,
        "best_eval_step": best_point.step,
        "best_eval_loss": best_point.loss,
        "best_eval_perplexity": best_point.perplexity,
        "num_eval_points": len(eval_points),
    }
    write_json(output_dir / "sft_run_summary.json", summary)
    lines = [
        "# Supplement Chapter 3 SFT Run Summary",
        "",
        f"- Model: `{run_summary['model']}`",
        f"- Train data: `{run_summary['train_path']}`",
        f"- Validation data: `{run_summary['val_path']}`",
        f"- Sequence length: `{run_summary['seq_length']}`",
        f"- Effective batch size: `{run_summary['effective_batch_size']}`",
        f"- Optimizer steps: `{run_summary['optimizer_steps']}`",
        f"- Learning rate: `{run_summary['learning_rate']}`",
        f"- Warmup steps: `{run_summary['warmup_steps']}`",
        f"- Best validation loss: `{best_point.loss:.6f}` at step `{best_point.step}`",
        f"- Final validation loss: `{run_summary['final_val_loss']:.6f}`",
        f"- Final validation perplexity: `{run_summary['final_val_perplexity']:.6f}`",
        f"- Output directory: `{run_summary['output_dir']}`",
    ]
    (output_dir / "sft_run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_points, eval_points = load_points(log_dir / "metrics.jsonl")
    run_summary = read_json(log_dir / "run_summary.json")
    sampled_train_points = downsample_train_points(train_points, args.max_train_points)

    validation_svg = render_svg(
        "SFT Validation Loss",
        "Validation loss",
        [
            (
                "validation loss",
                [(point.step, point.loss) for point in eval_points],
                "#1f77b4",
            )
        ],
    )
    (output_dir / "sft_validation_loss.svg").write_text(validation_svg, encoding="utf-8")

    train_svg = render_svg(
        "SFT Training Loss",
        "Training loss",
        [
            (
                "train loss",
                [(point.step, point.loss) for point in sampled_train_points],
                "#d62728",
            )
        ],
    )
    (output_dir / "sft_train_loss.svg").write_text(train_svg, encoding="utf-8")

    write_eval_tables(output_dir, eval_points)
    write_summary(output_dir, run_summary, eval_points)
    print(f"Wrote SFT plots and tables to {output_dir}")


if __name__ == "__main__":
    main()
