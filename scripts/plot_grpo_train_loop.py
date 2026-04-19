#!/usr/bin/env python3
"""Create writeup-ready artifacts for the Ch7 on-policy GRPO train-loop run."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = ".agents/logs/ch7/grpo_on_policy_lr1e-5"
DEFAULT_OUTPUT_DIR = "artifacts/experiments/ch7/grpo_train_loop"
DEFAULT_RUN_NAME = "grpo_on_policy_lr1e-5"
SELECTED_ROLLOUT_STEPS = (1, 50, 100, 150, 200)


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


def format_percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def linear_scale(
    domain_min: float,
    domain_max: float,
    range_min: float,
    range_max: float,
):
    if domain_max == domain_min:
        return lambda _: (range_min + range_max) / 2
    return lambda value: range_min + (
        (value - domain_min) / (domain_max - domain_min)
    ) * (range_max - range_min)


def write_svg_plot(
    *,
    path: Path,
    title: str,
    y_label: str,
    points: list[tuple[int, float]],
    y_max: float = 1.0,
) -> None:
    width = 820
    height = 430
    margin_top = 54
    margin_right = 36
    margin_bottom = 58
    margin_left = 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_max = max(step for step, _ in points)
    x = linear_scale(0, x_max, margin_left, margin_left + plot_width)
    y = linear_scale(0, y_max, margin_top + plot_height, margin_top)

    grid: list[str] = []
    for i in range(6):
        value = i / 5
        yy = y(value)
        grid.append(
            f'<line x1="{margin_left}" y1="{yy:.2f}" '
            f'x2="{margin_left + plot_width:.2f}" y2="{yy:.2f}" '
            'stroke="#d8dee9" stroke-width="1" />'
        )
        grid.append(
            f'<text x="{margin_left - 12}" y="{yy + 4:.2f}" '
            'text-anchor="end" font-size="12" fill="#4b5563">'
            f"{round(value * 100)}%</text>"
        )

    for i in range(5):
        value = round(x_max * i / 4)
        xx = x(value)
        grid.append(
            f'<line x1="{xx:.2f}" y1="{margin_top}" '
            f'x2="{xx:.2f}" y2="{margin_top + plot_height:.2f}" '
            'stroke="#edf2f7" stroke-width="1" />'
        )
        grid.append(
            f'<text x="{xx:.2f}" y="{height - 22}" text-anchor="middle" '
            f'font-size="12" fill="#4b5563">{value}</text>'
        )

    polyline_points = " ".join(
        f"{x(step):.2f},{y(value):.2f}" for step, value in points
    )
    circles = "\n  ".join(
        (
            f'<circle cx="{x(step):.2f}" cy="{y(value):.2f}" r="3" '
            f'fill="#2563eb"><title>step {step}: '
            f"{html.escape(format_percent(value))}</title></circle>"
        )
        for step, value in points
    )
    title_escaped = html.escape(title)
    y_label_escaped = html.escape(y_label)
    grid_svg = "\n  ".join(grid)

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title_escaped}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{margin_left}" y="30" font-size="20" font-weight="700" fill="#111827">{title_escaped}</text>
  {grid_svg}
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.4" />
  <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.4" />
  <polyline fill="none" stroke="#2563eb" stroke-width="3" points="{polyline_points}" />
  {circles}
  <text x="{margin_left + plot_width / 2}" y="{height - 6}" text-anchor="middle" font-size="13" fill="#111827">GRPO step</text>
  <text x="18" y="{margin_top + plot_height / 2}" text-anchor="middle" font-size="13" fill="#111827" transform="rotate(-90 18 {margin_top + plot_height / 2})">{y_label_escaped}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


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


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_run_dir = output_dir / "runs" / args.run_name

    if not args.skip_run_archive and run_dir.resolve() != archive_run_dir.resolve():
        shutil.copytree(run_dir, archive_run_dir, dirs_exist_ok=True)

    config = read_json(run_dir / "config.json")
    run_summary = read_json(run_dir / "run_summary.json")
    metrics = read_jsonl(run_dir / "metrics.jsonl")

    eval_rows = sorted(
        [row for row in metrics if row.get("type") == "eval"],
        key=lambda row: row["grpo_step"],
    )
    train_rows = sorted(
        [row for row in metrics if row.get("type") == "train"],
        key=lambda row: row["grpo_step"],
    )
    rollout_rows = sorted(
        [row for row in metrics if row.get("type") == "rollout"],
        key=lambda row: row["grpo_step"],
    )

    if not eval_rows or not train_rows or not rollout_rows:
        raise ValueError(f"Missing required eval/train/rollout metrics in {run_dir}")

    write_csv(
        output_dir / "grpo_train_loop_eval_points.csv",
        [
            "grpo_step",
            "answer_accuracy",
            "format_accuracy",
            "reward",
            "num_examples",
            "eval_seconds",
        ],
        [
            [
                row["grpo_step"],
                row["answer_accuracy"],
                row["format_accuracy"],
                row["reward"],
                row["num_examples"],
                row["eval_seconds"],
            ]
            for row in eval_rows
        ],
    )

    rollout_examples: list[dict[str, Any]] = []
    for step in SELECTED_ROLLOUT_STEPS:
        sample_path = (
            run_dir
            / f"grpo_step_{step:06d}"
            / "sample_rollouts.jsonl"
        )
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

    write_svg_plot(
        path=output_dir / "grpo_train_loop_validation_reward.svg",
        title="On-policy GRPO validation answer reward",
        y_label="validation answer reward",
        points=[(row["grpo_step"], row["answer_accuracy"]) for row in eval_rows],
    )
    write_svg_plot(
        path=output_dir / "grpo_train_loop_format_accuracy.svg",
        title="On-policy GRPO validation format accuracy",
        y_label="validation format accuracy",
        points=[(row["grpo_step"], row["format_accuracy"]) for row in eval_rows],
    )

    first_eval = eval_rows[0]
    best_eval = max(eval_rows, key=lambda row: row["answer_accuracy"])
    final_eval = eval_rows[-1]
    first_rollout = rollout_rows[0]
    final_rollout = rollout_rows[-1]
    final_train = train_rows[-1]

    run_summaries = [
        {
            "run_name": args.run_name,
            "archived_run_dir": str(archive_run_dir),
            "learning_rate": config["learning_rate"],
            "n_grpo_steps": config["n_grpo_steps"],
            "rollout_batch_size": config["rollout_batch_size"],
            "group_size": config["group_size"],
            "loss_type": config["loss_type"],
            "use_std_normalization": config["use_std_normalization"],
            "first_eval": {
                "grpo_step": first_eval["grpo_step"],
                "answer_accuracy": first_eval["answer_accuracy"],
                "format_accuracy": first_eval["format_accuracy"],
            },
            "best_eval": {
                "grpo_step": best_eval["grpo_step"],
                "answer_accuracy": best_eval["answer_accuracy"],
                "format_accuracy": best_eval["format_accuracy"],
            },
            "final_eval": {
                "grpo_step": final_eval["grpo_step"],
                "answer_accuracy": final_eval["answer_accuracy"],
                "format_accuracy": final_eval["format_accuracy"],
            },
            "first_rollout": {
                "grpo_step": first_rollout["grpo_step"],
                "answer_accuracy": first_rollout["answer_accuracy"],
                "format_accuracy": first_rollout["format_accuracy"],
                "avg_response_token_length": first_rollout[
                    "avg_response_token_length"
                ],
            },
            "final_rollout": {
                "grpo_step": final_rollout["grpo_step"],
                "answer_accuracy": final_rollout["answer_accuracy"],
                "format_accuracy": final_rollout["format_accuracy"],
                "avg_response_token_length": final_rollout[
                    "avg_response_token_length"
                ],
            },
            "final_train": {
                "grpo_step": final_train["grpo_step"],
                "token_entropy": final_train["token_entropy"],
                "grad_norm": final_train["grad_norm"],
                "loss": final_train["loss"],
            },
            "run_summary": run_summary,
        }
    ]
    (output_dir / "run_summaries.json").write_text(
        json.dumps(run_summaries, indent=2) + "\n",
        encoding="utf-8",
    )

    archive_md = f"""# GRPO Train Loop Run Archive

| run | learning rate | steps | initial answer | best answer | final answer | final format | final train answer | final avg length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `{args.run_name}` | `{config["learning_rate"]}` | {config["n_grpo_steps"]} | {format_percent(first_eval["answer_accuracy"])} | {format_percent(best_eval["answer_accuracy"])} @ {best_eval["grpo_step"]} | {format_percent(final_eval["answer_accuracy"])} | {format_percent(final_eval["format_accuracy"])} | {format_percent(final_rollout["answer_accuracy"])} | {final_rollout["avg_response_token_length"]:.1f} |

Raw run files are archived under `{archive_run_dir}/`.
"""
    (output_dir / "run_summaries_archive.md").write_text(
        archive_md,
        encoding="utf-8",
    )

    examples_lines = ["# GRPO Train Loop Rollout Examples", ""]
    for row in rollout_examples:
        examples_lines.extend(
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
        "\n".join(examples_lines),
        encoding="utf-8",
    )

    summary_md = f"""# GRPO Train Loop Summary

| metric | value |
|---|---:|
| initial validation answer reward | {format_percent(first_eval["answer_accuracy"])} |
| best validation answer reward | {format_percent(best_eval["answer_accuracy"])} at step {best_eval["grpo_step"]} |
| final validation answer reward | {format_percent(final_eval["answer_accuracy"])} |
| final validation format accuracy | {format_percent(final_eval["format_accuracy"])} |
| final rollout answer reward | {format_percent(final_rollout["answer_accuracy"])} |
| final rollout format accuracy | {format_percent(final_rollout["format_accuracy"])} |
| final average response token length | {final_rollout["avg_response_token_length"]:.1f} |
| final token entropy | {final_train["token_entropy"]:.4f} |
| final gradient norm | {final_train["grad_norm"]:.4f} |
"""
    (output_dir / "grpo_train_loop_summary.md").write_text(
        summary_md,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
