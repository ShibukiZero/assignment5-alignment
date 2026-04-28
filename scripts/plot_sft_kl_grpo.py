#!/usr/bin/env python3
"""Create writeup-ready artifacts for the SFT warm-start + KL GRPO experiment."""

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

from plot_grpo_train_loop import (
    COLORS,
    nice_y_max,
    point_to_svg_xy,
    points_to_polyline,
    render_svg_line_plot,
)


DEFAULT_OUTPUT_DIR = "artifacts/experiments/ch7/sft_kl_grpo"
DEFAULT_SFT_RUN_DIR = ".agents/logs/reruns/sft_kl_grpo/sft_256_step100"
DEFAULT_GRPO_RUN_DIR = (
    ".agents/logs/reruns/sft_kl_grpo/grpo_sft256_step100_kl0p01_190"
)
DEFAULT_REFERENCE_RUN_DIR = "artifacts/experiments/ch7/grpo_length_normalization/runs/masked_mean"
DEFAULT_REFERENCE_LABEL = "direct_grpo_lr4e-5"
ARCHIVE_IGNORE = shutil.ignore_patterns("sample_rollouts.jsonl")
SFT_STAGE_PLOT_UNITS = 10.0


@dataclass(frozen=True)
class SftEvalPoint:
    step: int
    answer_accuracy: float
    format_accuracy: float
    eval_seconds: float
    num_examples: int


@dataclass
class SftRunData:
    label: str
    run_dir: Path
    config: dict[str, Any]
    eval_points: list[SftEvalPoint]
    train_losses: list[tuple[int, float]]

    @property
    def best_eval(self) -> SftEvalPoint:
        return max(self.eval_points, key=lambda point: point.answer_accuracy)

    @property
    def final_eval(self) -> SftEvalPoint:
        return max(self.eval_points, key=lambda point: point.step)


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
    reference_kl: float | None
    kl_penalty: float | None


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
class GrpoRunData:
    label: str
    role: str
    run_dir: Path
    config: dict[str, Any]
    run_summary: dict[str, Any]
    eval_points: list[EvalPoint]
    train_points: list[TrainPoint]
    rollout_points: list[RolloutPoint]
    step_summaries: list[StepSummary]

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

    @property
    def kl_coef(self) -> float:
        return float(self.config.get("kl_coef", 0.0))

    @property
    def n_grpo_steps(self) -> int:
        return int(self.config.get("n_grpo_steps", self.final_eval.step))

    @property
    def status(self) -> str:
        if self.final_eval.step < self.n_grpo_steps:
            return "stopped early"
        return "completed"

    def cumulative_step_seconds(self) -> dict[int, float]:
        elapsed = 0.0
        by_step: dict[int, float] = {0: 0.0}
        for point in sorted(self.step_summaries, key=lambda item: item.step):
            elapsed += point.step_seconds
            by_step[point.step] = elapsed
        return by_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-run-dir", default=DEFAULT_SFT_RUN_DIR)
    parser.add_argument("--grpo-run-dir", default=DEFAULT_GRPO_RUN_DIR)
    parser.add_argument("--reference-run-dir", default=DEFAULT_REFERENCE_RUN_DIR)
    parser.add_argument("--reference-label", default=DEFAULT_REFERENCE_LABEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-run-archive",
        action="store_true",
        help="Generate plots and summaries without copying raw run logs into runs/.",
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


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def load_sft_run(label: str, run_dir: Path) -> SftRunData:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing SFT metrics file: {metrics_path}")

    config = read_json(run_dir / "config.json")
    eval_points: list[SftEvalPoint] = []
    train_losses: list[tuple[int, float]] = []
    for row in read_jsonl(metrics_path):
        row_type = row.get("type")
        if row_type == "eval":
            eval_points.append(
                SftEvalPoint(
                    step=int(row["train_step"]),
                    answer_accuracy=float(row["answer_accuracy"]),
                    format_accuracy=float(row["format_accuracy"]),
                    eval_seconds=float(row["eval_seconds"]),
                    num_examples=int(row["num_examples"]),
                )
            )
        elif row_type == "train":
            train_losses.append((int(row["train_step"]), float(row["loss"])))

    if not eval_points:
        raise ValueError(f"No SFT eval points found in {metrics_path}")
    return SftRunData(
        label=label,
        run_dir=run_dir,
        config=config,
        eval_points=sorted(eval_points, key=lambda point: point.step),
        train_losses=train_losses,
    )


def load_grpo_run(label: str, role: str, run_dir: Path) -> GrpoRunData:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing GRPO metrics file: {metrics_path}")

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
                    token_entropy=float(row["token_entropy"]),
                    grad_norm=float(row["grad_norm"]),
                    loss=float(row["loss"]),
                    train_answer_accuracy=float(row["train_answer_accuracy"]),
                    train_format_accuracy=float(row["train_format_accuracy"]),
                    reference_kl=optional_float(row.get("reference_kl")),
                    kl_penalty=optional_float(row.get("kl_penalty")),
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
        elif row_type == "grpo_step_summary":
            step_summaries.append(
                StepSummary(
                    step=int(row["grpo_step"]),
                    step_seconds=float(row["step_seconds"]),
                )
            )

    if not eval_points:
        raise ValueError(f"No GRPO eval points found in {metrics_path}")
    if not train_points:
        raise ValueError(f"No GRPO train points found in {metrics_path}")
    if not rollout_points:
        raise ValueError(f"No GRPO rollout points found in {metrics_path}")

    return GrpoRunData(
        label=label,
        role=role,
        run_dir=run_dir,
        config=config,
        run_summary=run_summary,
        eval_points=sorted(eval_points, key=lambda point: point.step),
        train_points=sorted(train_points, key=lambda point: point.optimizer_step),
        rollout_points=sorted(rollout_points, key=lambda point: point.step),
        step_summaries=sorted(step_summaries, key=lambda point: point.step),
    )


def write_sft_eval_points(sft: SftRunData, output_dir: Path) -> None:
    write_csv(
        output_dir / "sft_kl_grpo_sft_eval_points.csv",
        [
            "run",
            "train_step",
            "answer_accuracy",
            "format_accuracy",
            "num_examples",
            "eval_seconds",
        ],
        [
            [
                sft.label,
                point.step,
                point.answer_accuracy,
                point.format_accuracy,
                point.num_examples,
                point.eval_seconds,
            ]
            for point in sft.eval_points
        ],
    )


def staged_step_boundary() -> float:
    return SFT_STAGE_PLOT_UNITS


def render_staged_validation_reward(
    *,
    sft: SftRunData,
    grpo: GrpoRunData,
    reference: GrpoRunData,
    output_path: Path,
) -> None:
    width = 1100
    height = 660
    left = 95
    right = 300
    top = 74
    bottom = 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    boundary = staged_step_boundary()
    sft_final_step = float(sft.final_eval.step)

    staged_points = [
        (
            boundary * float(point.step) / sft_final_step,
            point.answer_accuracy,
        )
        for point in sft.eval_points
    ] + [
        (boundary + point.step, point.answer_accuracy) for point in grpo.eval_points
    ]
    reference_points = [
        (boundary + point.step, point.answer_accuracy)
        for point in reference.eval_points
    ]
    series = [
        ("SFT warm start + KL GRPO", staged_points),
        ("Direct GRPO reference", reference_points),
    ]
    cleaned_series = [
        (label, [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)])
        for label, points in series
    ]
    all_points = [point for _, points in cleaned_series for point in points]
    x_min = 0.0
    x_max = max(point[0] for point in all_points)
    y_min = 0.0
    y_max = nice_y_max(max(point[1] for point in all_points))
    boundary_x, _ = point_to_svg_xy(
        x=boundary,
        y=y_min,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        left=left,
        top=top,
        plot_width=plot_width,
        plot_height=plot_height,
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" '
        'font-family="Arial" font-size="22" font-weight="700">'
        "SFT warm start + KL GRPO staged validation reward</text>",
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        'stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{boundary_x:.1f}" y1="{top}" x2="{boundary_x:.1f}" '
        f'y2="{top + plot_height}" stroke="#666" stroke-width="1.5" '
        'stroke-dasharray="6 5"/>',
        f'<text x="{boundary_x + 8:.1f}" y="{top + 20}" '
        'font-family="Arial" font-size="13" fill="#444">GRPO begins</text>',
        f'<text x="{(left + boundary_x) / 2:.1f}" y="{top + 20}" '
        'text-anchor="middle" font-family="Arial" font-size="13" fill="#666">'
        "SFT</text>",
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
            f"{100.0 * y_value:.0f}%</text>"
        )

    x_ticks = [
        (boundary, "0"),
        (boundary + 50, "50"),
        (boundary + 100, "100"),
        (boundary + 150, "150"),
        (x_max, str(int(x_max - boundary))),
    ]
    for x_value, tick_label in x_ticks:
        px, _ = point_to_svg_xy(
            x=x_value,
            y=y_min,
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
            f'<line x1="{px:.1f}" y1="{top + plot_height}" x2="{px:.1f}" '
            f'y2="{top + plot_height + 5}" stroke="#222" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px:.1f}" y="{top + plot_height + 24}" text-anchor="middle" '
            'font-family="Arial" font-size="12">'
            f"{html.escape(tick_label)}</text>"
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
        "GRPO step after compressed SFT warm start</text>"
    )
    parts.append(
        f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" '
        'font-family="Arial" font-size="15" '
        f'transform="rotate(-90 24 {top + plot_height / 2})">'
        "Answer reward</text>"
    )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_grpo_eval_points(runs: list[GrpoRunData], output_dir: Path) -> None:
    rows: list[list[Any]] = []
    for run in runs:
        elapsed_by_step = run.cumulative_step_seconds()
        for point in run.eval_points:
            elapsed_seconds = elapsed_by_step.get(point.step)
            rows.append(
                [
                    run.label,
                    run.role,
                    point.step,
                    point.answer_accuracy,
                    point.format_accuracy,
                    point.reward,
                    point.num_examples,
                    point.eval_seconds,
                    (
                        elapsed_seconds / 3600.0
                        if elapsed_seconds is not None
                        else None
                    ),
                ]
            )
    write_csv(
        output_dir / "sft_kl_grpo_eval_points.csv",
        [
            "run",
            "role",
            "grpo_step",
            "answer_accuracy",
            "format_accuracy",
            "reward",
            "num_examples",
            "eval_seconds",
            "wall_clock_hours",
        ],
        rows,
    )


def write_train_points(run: GrpoRunData, output_dir: Path) -> None:
    write_csv(
        output_dir / "sft_kl_grpo_train_points.csv",
        [
            "run",
            "grpo_step",
            "optimizer_step",
            "loss",
            "grad_norm",
            "token_entropy",
            "reference_kl",
            "kl_penalty",
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
                point.reference_kl,
                point.kl_penalty,
                point.train_answer_accuracy,
                point.train_format_accuracy,
            ]
            for point in run.train_points
        ],
    )


def write_rollout_points(run: GrpoRunData, output_dir: Path) -> None:
    write_csv(
        output_dir / "sft_kl_grpo_rollout_points.csv",
        [
            "run",
            "grpo_step",
            "answer_accuracy",
            "format_accuracy",
            "avg_response_token_length",
        ],
        [
            [
                run.label,
                point.step,
                point.answer_accuracy,
                point.format_accuracy,
                point.avg_response_token_length,
            ]
            for point in run.rollout_points
        ],
    )


def write_summary(sft: SftRunData, grpo: GrpoRunData, reference: GrpoRunData, output_dir: Path) -> None:
    rows = [
        [
            "SFT warm start",
            sft.label,
            "",
            sft.best_eval.step,
            sft.best_eval.answer_accuracy,
            sft.final_eval.step,
            sft.final_eval.answer_accuracy,
            sft.final_eval.format_accuracy,
            "",
            "",
            "",
        ],
        [
            "GRPO after SFT + KL",
            grpo.label,
            grpo.kl_coef,
            grpo.best_eval.step,
            grpo.best_eval.answer_accuracy,
            grpo.final_eval.step,
            grpo.final_eval.answer_accuracy,
            grpo.final_eval.format_accuracy,
            grpo.final_rollout.answer_accuracy,
            grpo.final_rollout.avg_response_token_length,
            grpo.final_train.reference_kl,
        ],
        [
            "Direct GRPO reference",
            reference.label,
            reference.kl_coef,
            reference.best_eval.step,
            reference.best_eval.answer_accuracy,
            reference.final_eval.step,
            reference.final_eval.answer_accuracy,
            reference.final_eval.format_accuracy,
            reference.final_rollout.answer_accuracy,
            reference.final_rollout.avg_response_token_length,
            reference.final_train.reference_kl,
        ],
    ]
    header = [
        "role",
        "run",
        "kl_coef",
        "best_step",
        "best_answer_accuracy",
        "final_step",
        "final_answer_accuracy",
        "final_format_accuracy",
        "final_rollout_answer_accuracy",
        "final_avg_response_token_length",
        "final_reference_kl",
    ]
    write_csv(output_dir / "sft_kl_grpo_summary.csv", header, rows)

    lines = [
        "# SFT warm-start + KL GRPO summary",
        "",
        "| role | run | KL coef | best answer | best step | final answer | final step | final format | final rollout answer | final avg length | final reference KL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        (
            role,
            run,
            kl_coef,
            best_step,
            best_answer,
            final_step,
            final_answer,
            final_format,
            final_rollout_answer,
            final_avg_length,
            final_reference_kl,
        ) = row
        lines.append(
            "| "
            + " | ".join(
                [
                    str(role),
                    f"`{run}`",
                    "" if kl_coef == "" else f"{float(kl_coef):.4g}",
                    percent(float(best_answer)),
                    str(best_step),
                    percent(float(final_answer)),
                    str(final_step),
                    percent(float(final_format)),
                    (
                        ""
                        if final_rollout_answer == ""
                        else percent(float(final_rollout_answer))
                    ),
                    (
                        ""
                        if final_avg_length == ""
                        else f"{float(final_avg_length):.1f}"
                    ),
                    (
                        ""
                        if final_reference_kl is None or final_reference_kl == ""
                        else f"{float(final_reference_kl):.4f}"
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Generated artifacts:",
            "",
            "- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_staged_validation_reward.svg`",
            "- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_validation_reward.svg`",
            "- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_validation_reward_wall_clock.svg`",
            "- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_format_accuracy.svg`",
            "- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_response_length.svg`",
            "- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_token_entropy.svg`",
            "- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_reference_kl.svg`",
            "- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_summary.csv`",
            "- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_eval_points.csv`",
            "- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_train_points.csv`",
            "- `artifacts/experiments/ch7/sft_kl_grpo/sft_kl_grpo_rollout_points.csv`",
        ]
    )
    (output_dir / "sft_kl_grpo_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def archive_runs(
    *,
    sft: SftRunData,
    grpo: GrpoRunData,
    reference: GrpoRunData,
    output_dir: Path,
    skip_run_archive: bool,
) -> None:
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if not skip_run_archive:
        destinations = [
            (sft.run_dir, runs_dir / sft.label),
            (grpo.run_dir, runs_dir / grpo.label),
            (reference.run_dir, runs_dir / reference.label),
        ]
        for source, destination in destinations:
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination, ignore=ARCHIVE_IGNORE)

    manifest = {
        "sft_run": str(sft.run_dir),
        "grpo_run": str(grpo.run_dir),
        "reference_run": str(reference.run_dir),
        "sample_rollouts_omitted": True,
    }
    (output_dir / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# SFT warm-start + KL GRPO run archive",
        "",
        "This archive summarizes the SFT warm-start + KL-regularized GRPO experiment.",
        "",
        "| role | run | source log | best answer | best step | final answer | final step |",
        "|---|---|---|---:|---:|---:|---:|",
        (
            f"| SFT warm start | `{sft.label}` | `{sft.run_dir}` | "
            f"{percent(sft.best_eval.answer_accuracy)} | {sft.best_eval.step} | "
            f"{percent(sft.final_eval.answer_accuracy)} | {sft.final_eval.step} |"
        ),
        (
            f"| GRPO after SFT + KL | `{grpo.label}` | `{grpo.run_dir}` | "
            f"{percent(grpo.best_eval.answer_accuracy)} | {grpo.best_eval.step} | "
            f"{percent(grpo.final_eval.answer_accuracy)} | {grpo.final_eval.step} |"
        ),
        (
            f"| Direct GRPO reference | `{reference.label}` | `{reference.run_dir}` | "
            f"{percent(reference.best_eval.answer_accuracy)} | {reference.best_eval.step} | "
            f"{percent(reference.final_eval.answer_accuracy)} | {reference.final_eval.step} |"
        ),
        "",
        "Raw run files are archived under `artifacts/experiments/ch7/sft_kl_grpo/runs/`.",
        "`sample_rollouts.jsonl` files are intentionally omitted because aggregate rollout summaries are sufficient for the writeup.",
    ]
    (output_dir / "run_summaries_archive.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def render_plots(sft: SftRunData, grpo: GrpoRunData, reference: GrpoRunData, output_dir: Path) -> None:
    render_staged_validation_reward(
        sft=sft,
        grpo=grpo,
        reference=reference,
        output_path=output_dir / "sft_kl_grpo_staged_validation_reward.svg",
    )

    render_svg_line_plot(
        series=[
            (
                "SFT + KL GRPO",
                [(point.step, point.answer_accuracy) for point in grpo.eval_points],
            ),
            (
                "Direct GRPO reference",
                [(point.step, point.answer_accuracy) for point in reference.eval_points],
            ),
        ],
        title="SFT + KL GRPO validation answer reward",
        x_label="GRPO step",
        y_label="Answer reward",
        output_path=output_dir / "sft_kl_grpo_validation_reward.svg",
        y_percent_axis=True,
    )

    wall_clock_series: list[tuple[str, list[tuple[float, float]]]] = []
    for run, label in [(grpo, "SFT + KL GRPO"), (reference, "Direct GRPO reference")]:
        elapsed_by_step = run.cumulative_step_seconds()
        points = [
            (elapsed_by_step[point.step] / 3600.0, point.answer_accuracy)
            for point in run.eval_points
            if point.step in elapsed_by_step
        ]
        if points:
            wall_clock_series.append((label, points))
    render_svg_line_plot(
        series=wall_clock_series,
        title="SFT + KL GRPO answer reward vs wall-clock time",
        x_label="Wall-clock hours",
        y_label="Answer reward",
        output_path=output_dir / "sft_kl_grpo_validation_reward_wall_clock.svg",
        x_tick_decimals=1,
        y_percent_axis=True,
    )

    render_svg_line_plot(
        series=[
            (
                "SFT + KL GRPO",
                [(point.step, point.format_accuracy) for point in grpo.eval_points],
            ),
            (
                "Direct GRPO reference",
                [(point.step, point.format_accuracy) for point in reference.eval_points],
            ),
        ],
        title="SFT + KL GRPO validation format accuracy",
        x_label="GRPO step",
        y_label="Format accuracy",
        output_path=output_dir / "sft_kl_grpo_format_accuracy.svg",
        y_percent_axis=True,
    )

    render_svg_line_plot(
        series=[
            (
                "SFT + KL GRPO",
                [
                    (point.step, point.avg_response_token_length)
                    for point in grpo.rollout_points
                ],
            ),
            (
                "Direct GRPO reference",
                [
                    (point.step, point.avg_response_token_length)
                    for point in reference.rollout_points
                ],
            ),
        ],
        title="SFT + KL GRPO rollout response length",
        x_label="GRPO step",
        y_label="Average response tokens",
        output_path=output_dir / "sft_kl_grpo_response_length.svg",
        y_percent_axis=False,
    )

    render_svg_line_plot(
        series=[
            (
                "SFT + KL GRPO",
                [(point.step, point.token_entropy) for point in grpo.train_points],
            ),
            (
                "Direct GRPO reference",
                [(point.step, point.token_entropy) for point in reference.train_points],
            ),
        ],
        title="SFT + KL GRPO token entropy",
        x_label="GRPO step",
        y_label="Mean token entropy",
        output_path=output_dir / "sft_kl_grpo_token_entropy.svg",
        y_percent_axis=False,
    )

    kl_points = [
        (point.step, point.reference_kl)
        for point in grpo.train_points
        if point.reference_kl is not None
    ]
    if kl_points:
        render_svg_line_plot(
            series=[("SFT + KL GRPO", kl_points)],
            title="SFT + KL GRPO reference KL",
            x_label="GRPO step",
            y_label="Mean sampled-token KL",
            output_path=output_dir / "sft_kl_grpo_reference_kl.svg",
            y_percent_axis=False,
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft = load_sft_run("sft_256_step100", Path(args.sft_run_dir))
    grpo = load_grpo_run(
        "grpo_sft256_step100_kl0p01_190",
        "sft_kl_grpo",
        Path(args.grpo_run_dir),
    )
    reference = load_grpo_run(
        args.reference_label,
        "direct_grpo_reference",
        Path(args.reference_run_dir),
    )

    write_sft_eval_points(sft, output_dir)
    write_grpo_eval_points([grpo, reference], output_dir)
    write_train_points(grpo, output_dir)
    write_rollout_points(grpo, output_dir)
    write_summary(sft, grpo, reference, output_dir)
    render_plots(sft, grpo, reference, output_dir)
    archive_runs(
        sft=sft,
        grpo=grpo,
        reference=reference,
        output_dir=output_dir,
        skip_run_archive=args.skip_run_archive,
    )


if __name__ == "__main__":
    main()
