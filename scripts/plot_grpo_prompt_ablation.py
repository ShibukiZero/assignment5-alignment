#!/usr/bin/env python3
"""Create writeup-ready artifacts for the Ch7 GRPO prompt ablation."""

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


DEFAULT_OUTPUT_DIR = "artifacts/experiments/ch7/grpo_prompt_ablation"

DEFAULT_RUNS = [
    (
        "r1_zero_reference",
        ".agents/logs/ch7/grpo_on_policy_ablations/length_normalization_rerun_staging_single_gpu/masked_mean_lr4e-5",
    ),
    (
        "question_only",
        ".agents/logs/reruns/off_policy_clip_ablation_e2_tb256_single_gpu_std_norm/question_only_e1_tb256_std_norm",
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
            "Example: --include-run alt_question_only=.agents/logs/..."
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


def prompt_template(run: RunData) -> str:
    return str(run.config.get("prompt_template", "unknown"))


def reward_fn(run: RunData) -> str:
    return str(run.config.get("reward_fn", "r1_zero"))


def stop_contract(run: RunData) -> str:
    eval0 = run.eval_points[0]
    stop_value = run.config.get("stop")
    if stop_value is not None:
        return str(stop_value)
    if prompt_template(run).endswith("question_only.prompt"):
        return "none"
    if eval0.num_examples is not None:
        # The config does not persist stop directly for older runs, so use prompt contract.
        return "</answer>" if "r1_zero" in prompt_template(run) else "none"
    return "unknown"


def write_eval_points(runs: list[RunData], output_dir: Path) -> None:
    rows: list[list[Any]] = []
    for run in runs:
        for point in run.eval_points:
            rows.append(
                [
                    run.label,
                    prompt_template(run),
                    reward_fn(run),
                    run.config.get("loss_type", "unknown"),
                    run.learning_rate,
                    point.step,
                    point.answer_accuracy,
                    point.format_accuracy,
                    point.reward,
                    point.num_examples,
                ]
            )
    write_csv(
        output_dir / "grpo_prompt_ablation_eval_points.csv",
        [
            "run",
            "prompt_template",
            "reward_fn",
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


def write_rollout_points(runs: list[RunData], output_dir: Path) -> None:
    rows: list[list[Any]] = []
    for run in runs:
        for point in run.rollout_points:
            rows.append(
                [
                    run.label,
                    prompt_template(run),
                    reward_fn(run),
                    point.step,
                    point.answer_accuracy,
                    point.format_accuracy,
                    point.avg_response_token_length,
                ]
            )
    write_csv(
        output_dir / "grpo_prompt_ablation_rollout_points.csv",
        [
            "run",
            "prompt_template",
            "reward_fn",
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
                    prompt_template(run),
                    reward_fn(run),
                    point.step,
                    point.loss,
                    point.grad_norm,
                    point.token_entropy,
                    point.train_answer_accuracy,
                    point.train_format_accuracy,
                ]
            )
    write_csv(
        output_dir / "grpo_prompt_ablation_train_points.csv",
        [
            "run",
            "prompt_template",
            "reward_fn",
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
                prompt_template(run),
                reward_fn(run),
                stop_contract(run),
                run.config.get("loss_type", "unknown"),
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
                "prompt_template": prompt_template(run),
                "reward_fn": reward_fn(run),
                "stop_contract": stop_contract(run),
                "loss_type": run.config.get("loss_type", "unknown"),
                "learning_rate": run.learning_rate,
                "status": run.status,
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
                    "avg_response_token_length": run.final_rollout.avg_response_token_length,
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
        output_dir / "grpo_prompt_ablation_summary.csv",
        [
            "run",
            "prompt_template",
            "reward_fn",
            "stop_contract",
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
        "# GRPO Prompt Ablation Run Archive",
        "",
        "| run | prompt | reward | stop | best answer | best step | final answer | final format |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        markdown_lines.append(
            f"| `{run.label}` | `{prompt_template(run)}` | `{reward_fn(run)}` | "
            f"`{stop_contract(run)}` | {percent(run.best_eval.answer_accuracy)} | "
            f"{run.best_eval.step} | {percent(run.final_eval.answer_accuracy)} | "
            f"{percent(run.final_eval.format_accuracy)} |"
        )
    markdown_lines.extend(
        [
            "",
            (
                "Raw run files are archived under `artifacts/experiments/ch7/grpo_prompt_ablation/runs/`. "
                "`sample_rollouts.jsonl` files are intentionally omitted because aggregate "
                "rollout summaries are sufficient for the writeup."
            ),
            "",
        ]
    )
    (output_dir / "run_summaries_archive.md").write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )


def write_experiment_log(runs: list[RunData], output_dir: Path) -> None:
    r1_zero = next(run for run in runs if run.label == "r1_zero_reference")
    question_only = next(run for run in runs if run.label == "question_only")

    lines = [
        "# GRPO Prompt Ablation Experiment Log",
        "",
        "## Setup",
        "",
        "- We compare the R1-Zero on-policy reference line used in the off-policy sweep with the prefix-cache-repaired question-only rerun.",
        "- Both runs use a 200-step GRPO budget with `rollout_batch_size = 256`, `train_batch_size = 256`, `learning_rate = 4e-5`, `group_size = 8`, `epochs_per_rollout_batch = 1`, and `use_std_normalization = true`.",
        f"- The reference run uses `{prompt_template(r1_zero)}` with `r1_zero_reward_fn` and the standard `</answer>` stop condition.",
        f"- The ablation run uses `{prompt_template(question_only)}` with `question_only_reward_fn` and no `</answer>` stop condition.",
        "- The reference uses `reinforce_with_baseline`; the question-only rerun uses `grpo_clip`, so this remains a prompt-and-evaluation contract comparison rather than a pure prompt-wording ablation.",
        "",
        "## Findings",
        "",
        f"- The question-only contract starts much stronger at step 0, reaching {percent(question_only.eval_points[0].answer_accuracy)} before any RL updates.",
        f"- The question-only run peaks at {percent(question_only.best_eval.answer_accuracy)} on step {question_only.best_eval.step} and finishes at {percent(question_only.final_eval.answer_accuracy)}.",
        f"- The R1-Zero reference peaks at {percent(r1_zero.best_eval.answer_accuracy)} and finishes at {percent(r1_zero.final_eval.answer_accuracy)}.",
        f"- The question-only run also maintains high final format accuracy ({percent(question_only.final_eval.format_accuracy)}), while ending with a longer average rollout response length ({question_only.final_rollout.avg_response_token_length:.1f} vs. {r1_zero.final_rollout.avg_response_token_length:.1f}).",
        "- Since the prompt, reward function, and stop contract all change together, this should be interpreted as a comparison between two prompt-and-evaluation contracts rather than a pure prompt-wording ablation.",
        "",
    ]
    (output_dir / "grpo_prompt_ablation_experiment_log.md").write_text(
        "\n".join(lines),
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
    write_experiment_log(runs, output_dir)

    answer_series = [
        (run.label, [(point.step, point.answer_accuracy) for point in run.eval_points])
        for run in runs
    ]
    render_svg_line_plot(
        series=answer_series,
        title="GRPO prompt ablation: validation answer reward",
        x_label="GRPO step",
        y_label="Validation answer reward",
        output_path=output_dir / "grpo_prompt_ablation_validation_reward.svg",
        y_min=0.0,
        y_max=0.9,
    )

    format_series = [
        (run.label, [(point.step, point.format_accuracy) for point in run.eval_points])
        for run in runs
    ]
    render_svg_line_plot(
        series=format_series,
        title="GRPO prompt ablation: validation format accuracy",
        x_label="GRPO step",
        y_label="Validation format accuracy",
        output_path=output_dir / "grpo_prompt_ablation_format_accuracy.svg",
        y_min=0.0,
        y_max=1.0,
    )

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
        series=response_length_series,
        title="GRPO prompt ablation: rollout response length",
        x_label="GRPO step",
        y_label="Average rollout response tokens",
        output_path=output_dir / "grpo_prompt_ablation_response_length.svg",
        y_min=0.0,
        y_max=None,
    )

    entropy_series = [
        (run.label, [(point.step, point.token_entropy) for point in run.train_points])
        for run in runs
    ]
    render_svg_line_plot(
        series=entropy_series,
        title="GRPO prompt ablation: token entropy",
        x_label="GRPO step",
        y_label="Token entropy (nats)",
        output_path=output_dir / "grpo_prompt_ablation_token_entropy.svg",
        y_min=0.0,
        y_max=None,
    )

    grad_norm_series = [
        (run.label, [(point.step, point.grad_norm) for point in run.train_points])
        for run in runs
    ]
    render_svg_line_plot(
        series=grad_norm_series,
        title="GRPO prompt ablation: gradient norm",
        x_label="GRPO step",
        y_label="Gradient norm",
        output_path=output_dir / "grpo_prompt_ablation_grad_norm.svg",
        y_min=0.0,
        y_max=None,
    )

    print(f"Wrote GRPO prompt ablation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
