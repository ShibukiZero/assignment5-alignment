#!/usr/bin/env python3
"""Create writeup-ready artifacts for the Ch7 GRPO group-std ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from plot_grpo_length_normalization import (
    RunData,
    archive_runs,
    load_run,
    percent,
    render_svg_line_plot,
    write_csv,
)


DEFAULT_OUTPUT_DIR = "artifacts/experiments/ch7/grpo_group_standard_deviation"

DEFAULT_RUNS = [
    (
        "std_normalization",
        "runs/logs/ch7/grpo_on_policy_ablations/length_normalization_rerun_staging_single_gpu/masked_mean_lr4e-5",
    ),
    (
        "no_std_normalization",
        "runs/logs/reruns/grpo_group_standard_deviation_retry_single_gpu/grpo_group_standard_deviation/no_std_normalization",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-run",
        action="append",
        default=None,
        help=(
            "Optional label=log_dir entry. "
            "Example: --include-run no_std_lr8e-5=runs/logs/..."
        ),
    )
    parser.add_argument(
        "--skip-run-archive",
        action="store_true",
        help="Generate plots and summaries without copying raw runs into runs/.",
    )
    return parser.parse_args()


def parse_extra_run(spec: str) -> tuple[str, str]:
    parts = spec.split("=", maxsplit=1)
    if len(parts) != 2:
        raise ValueError("--include-run must have the form label=log_dir")
    return parts[0], parts[1]


def load_runs(extra_specs: list[str] | None) -> list[RunData]:
    run_specs = list(DEFAULT_RUNS)
    if extra_specs:
        run_specs.extend(parse_extra_run(spec) for spec in extra_specs)
    return [load_run(label, Path(log_dir)) for label, log_dir in run_specs]


def use_std_normalization(run: RunData) -> bool:
    return bool(run.config["use_std_normalization"])


def std_label(run: RunData) -> str:
    return "std=True" if use_std_normalization(run) else "std=False"


def loss_normalization(run: RunData) -> str:
    return str(run.config.get("loss_normalization") or "masked_mean")


def write_eval_points(runs: list[RunData], output_dir: Path) -> None:
    rows: list[list[Any]] = []
    for run in runs:
        for point in run.eval_points:
            rows.append(
                [
                    run.label,
                    use_std_normalization(run),
                    loss_normalization(run),
                    run.learning_rate,
                    point.step,
                    point.answer_accuracy,
                    point.format_accuracy,
                    point.reward,
                    point.num_examples,
                ]
            )
    write_csv(
        output_dir / "grpo_group_standard_deviation_eval_points.csv",
        [
            "run",
            "use_std_normalization",
            "loss_normalization",
            "learning_rate",
            "grpo_step",
            "answer_accuracy",
            "format_accuracy",
            "reward",
            "num_examples",
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
                    use_std_normalization(run),
                    point.step,
                    point.answer_accuracy,
                    point.format_accuracy,
                    point.avg_response_token_length,
                ]
            )
    write_csv(
        output_dir / "grpo_group_standard_deviation_rollout_points.csv",
        [
            "run",
            "use_std_normalization",
            "grpo_step",
            "rollout_answer_accuracy",
            "rollout_format_accuracy",
            "avg_response_token_length",
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
                    use_std_normalization(run),
                    point.step,
                    point.loss,
                    point.grad_norm,
                    point.token_entropy,
                    point.train_answer_accuracy,
                    point.train_format_accuracy,
                ]
            )
    write_csv(
        output_dir / "grpo_group_standard_deviation_train_points.csv",
        [
            "run",
            "use_std_normalization",
            "grpo_step",
            "loss",
            "grad_norm",
            "token_entropy",
            "train_answer_accuracy",
            "train_format_accuracy",
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
                use_std_normalization(run),
                loss_normalization(run),
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
                "use_std_normalization": use_std_normalization(run),
                "loss_normalization": loss_normalization(run),
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
        output_dir / "grpo_group_standard_deviation_summary.csv",
        [
            "run",
            "use_std_normalization",
            "loss_normalization",
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
        "# GRPO Group Standard Deviation Summary",
        "",
        (
            "| run | use std normalization | status | best answer | best step | "
            "final answer | final format | final rollout answer | final avg length | "
            "final grad norm |"
        ),
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        markdown_lines.append(
            (
                f"| `{run.label}` | `{use_std_normalization(run)}` | "
                f"{run.status} | "
                f"{percent(run.best_eval.answer_accuracy)} | {run.best_eval.step} | "
                f"{percent(run.final_eval.answer_accuracy)} | "
                f"{percent(run.final_eval.format_accuracy)} | "
                f"{percent(run.final_rollout.answer_accuracy)} | "
                f"{run.final_rollout.avg_response_token_length:.1f} | "
                f"{run.final_train.grad_norm:.2f} |"
            )
        )
    markdown_lines.extend(
        [
            "",
            "Default source logs:",
            "",
            *[
                f"- `{label}`: `{log_dir}`"
                for label, log_dir in DEFAULT_RUNS
            ],
            "",
            (
                "Raw run files are archived under "
                "`artifacts/experiments/ch7/grpo_group_standard_deviation/runs/`. "
                "`sample_rollouts.jsonl` files are intentionally omitted because "
                "aggregate rollout summaries are sufficient for the writeup."
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
    runs = load_runs(args.include_run)

    if not args.skip_run_archive:
        archive_runs(runs, output_dir)

    write_eval_points(runs, output_dir)
    write_rollout_points(runs, output_dir)
    write_train_points(runs, output_dir)
    write_summary_files(runs, output_dir)

    answer_series = [
        (
            std_label(run),
            [(point.step, point.answer_accuracy) for point in run.eval_points],
        )
        for run in runs
    ]
    format_series = [
        (
            std_label(run),
            [(point.step, point.format_accuracy) for point in run.eval_points],
        )
        for run in runs
    ]
    length_series = [
        (
            std_label(run),
            [
                (point.step, point.avg_response_token_length)
                for point in run.rollout_points
            ],
        )
        for run in runs
    ]
    grad_norm_series = [
        (
            std_label(run),
            [(point.step, point.grad_norm) for point in run.train_points],
        )
        for run in runs
    ]
    entropy_series = [
        (
            std_label(run),
            [(point.step, point.token_entropy) for point in run.train_points],
        )
        for run in runs
    ]

    render_svg_line_plot(
        series=answer_series,
        title="GRPO validation answer reward by group std normalization",
        x_label="GRPO step",
        y_label="validation answer reward",
        output_path=output_dir / "grpo_group_standard_deviation_validation_reward.svg",
    )
    render_svg_line_plot(
        series=format_series,
        title="GRPO validation format accuracy by group std normalization",
        x_label="GRPO step",
        y_label="validation format accuracy",
        output_path=output_dir / "grpo_group_standard_deviation_format_accuracy.svg",
        y_max=1.0,
    )
    render_svg_line_plot(
        series=length_series,
        title="GRPO rollout response length by group std normalization",
        x_label="GRPO step",
        y_label="average response tokens",
        output_path=output_dir / "grpo_group_standard_deviation_response_length.svg",
    )
    render_svg_line_plot(
        series=grad_norm_series,
        title="GRPO gradient norm by group std normalization",
        x_label="GRPO step",
        y_label="pre-clip gradient norm",
        output_path=output_dir / "grpo_group_standard_deviation_grad_norm.svg",
    )
    render_svg_line_plot(
        series=entropy_series,
        title="GRPO token entropy by group std normalization",
        x_label="GRPO step",
        y_label="token entropy (nats)",
        output_path=output_dir / "grpo_group_standard_deviation_token_entropy.svg",
    )


if __name__ == "__main__":
    main()
