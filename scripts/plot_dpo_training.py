#!/usr/bin/env python3
"""Create writeup-ready plots and tables for supplement DPO training."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = "runs/logs/ch5/dpo"
DEFAULT_OUTPUT_DIR = "artifacts/experiments/supplement/ch5/dpo"


@dataclass(frozen=True)
class EvalPoint:
    step: int
    loss: float
    classification_accuracy: float
    mean_dpo_margin: float
    num_examples: int
    final: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_eval_points(metrics_path: Path) -> list[EvalPoint]:
    points: list[EvalPoint] = []
    for row in read_jsonl(metrics_path):
        if row.get("type") != "eval":
            continue
        points.append(
            EvalPoint(
                step=int(row["step"]),
                loss=float(row["loss"]),
                classification_accuracy=float(row["classification_accuracy"]),
                mean_dpo_margin=float(row["mean_dpo_margin"]),
                num_examples=int(row["num_examples"]),
                final=bool(row.get("final", False)),
            )
        )
    if not points:
        raise ValueError(f"No eval rows found in {metrics_path}")
    return points


def nice_range(values: list[float]) -> tuple[float, float]:
    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        pad = 0.05 if min_value == 0 else abs(min_value) * 0.05
        return min_value - pad, max_value + pad
    pad = (max_value - min_value) * 0.08
    return min_value - pad, max_value + pad


def render_svg(points: list[EvalPoint]) -> str:
    width, height = 900, 520
    left, right, top, bottom = 82, 32, 48, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [point.step for point in points]
    y_values = [point.classification_accuracy for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = nice_range(y_values)

    coords: list[str] = []
    for point in points:
        px = left if x_max == x_min else left + (point.step - x_min) / (x_max - x_min) * plot_width
        py = top if y_max == y_min else top + (y_max - point.classification_accuracy) / (y_max - y_min) * plot_height
        coords.append(f"{px:.1f},{py:.1f}")

    x_ticks = [x_min + (x_max - x_min) * i / 4 for i in range(5)]
    y_ticks = [y_min + (y_max - y_min) * i / 4 for i in range(5)]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="26" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="700">DPO Validation Accuracy</text>',
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
            f'<text transform="translate(22,{top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Validation classification accuracy</text>',
            f'<polyline points="{html.escape(" ".join(coords))}" fill="none" stroke="#2563eb" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def write_tables(output_dir: Path, points: list[EvalPoint]) -> None:
    rows = [
        {
            "step": point.step,
            "loss": point.loss,
            "classification_accuracy": point.classification_accuracy,
            "mean_dpo_margin": point.mean_dpo_margin,
            "num_examples": point.num_examples,
            "final": point.final,
        }
        for point in points
    ]
    write_json(output_dir / "dpo_eval_points.json", rows)
    with (output_dir / "dpo_eval_points.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(log_dir / "config.json")
    summary_path = log_dir / "run_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    points = load_eval_points(log_dir / "metrics.jsonl")

    (output_dir / "dpo_validation_accuracy.svg").write_text(
        render_svg(points),
        encoding="utf-8",
    )
    write_tables(output_dir, points)
    write_json(
        output_dir / "dpo_run_summary.json",
        {
            "config": config,
            "run_summary": summary,
            "num_eval_points": len(points),
            "best_eval_accuracy": max(point.classification_accuracy for point in points),
            "final_eval_accuracy": points[-1].classification_accuracy,
        },
    )


if __name__ == "__main__":
    main()
