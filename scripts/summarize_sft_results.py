#!/usr/bin/env python3
"""Create writeup-ready result tables for the Ch4 SFT experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SIZE_SWEEP_LOG_ROOT = ".agents/logs/ch4/sft_noisy_size_sweep_bs16_lr5e-5"
DEFAULT_FILTERED_LOG_ROOT = ".agents/logs/ch4/sft_noisy_filtered_bs16_lr5e-5"
DEFAULT_OUTPUT_DIR = "artifacts/experiments/ch4/sft_experiment"
ARCHIVE_IGNORE = shutil.ignore_patterns("sample_rollouts.jsonl")


@dataclass(frozen=True)
class EvalPoint:
    step: int
    answer_accuracy: float
    format_accuracy: float
    reward: float
    num_examples: int


@dataclass(frozen=True)
class RunSummary:
    run_name: str
    label: str
    num_train_examples: int | None
    train_batch_size: int | None
    num_train_steps: int
    effective_epochs: float | None
    best_step: int
    best_answer_accuracy: float
    best_format_accuracy: float
    final_step: int
    final_answer_accuracy: float
    final_format_accuracy: float
    final_train_loss: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun-log-root",
        default=None,
        help=(
            "Root containing both size* and filtered_full* rerun directories. "
            "When set, this overrides --size-sweep-log-root and --filtered-log-root."
        ),
    )
    parser.add_argument("--size-sweep-log-root", default=DEFAULT_SIZE_SWEEP_LOG_ROOT)
    parser.add_argument("--filtered-log-root", default=DEFAULT_FILTERED_LOG_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--archive-runs",
        action="store_true",
        help="Copy the selected raw run directories into output-dir/runs.",
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


def label_for_run(run_name: str, num_train_examples: int | None) -> str:
    if run_name.startswith("filtered_full"):
        return "Filtered full"
    if run_name.startswith("sizefull"):
        return "Noisy full"
    match = re.match(r"size(\d+)_", run_name)
    if match:
        return match.group(1)
    if num_train_examples is not None:
        return str(num_train_examples)
    return run_name


def sort_key(summary: RunSummary) -> tuple[int, int, str]:
    if summary.run_name.startswith("filtered_full"):
        return (1, 0, summary.run_name)
    if summary.run_name.startswith("sizefull"):
        return (0, 10**9, summary.run_name)
    if summary.num_train_examples is not None:
        return (0, summary.num_train_examples, summary.run_name)
    return (0, 10**8, summary.run_name)


def load_eval_points(run_dir: Path) -> list[EvalPoint]:
    points: list[EvalPoint] = []
    for path in sorted(run_dir.glob("eval_summary_step_*.json")):
        summary = read_json(path)
        step = int(path.stem.removeprefix("eval_summary_step_"))
        points.append(
            EvalPoint(
                step=step,
                answer_accuracy=float(summary["answer_accuracy"]),
                format_accuracy=float(summary["format_accuracy"]),
                reward=float(summary.get("reward", summary["answer_accuracy"])),
                num_examples=int(summary["num_examples"]),
            )
        )
    if not points:
        raise ValueError(f"No eval summaries found in {run_dir}")
    return sorted(points, key=lambda point: point.step)


def summarize_run(run_dir: Path) -> RunSummary | None:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None

    eval_points = load_eval_points(run_dir)
    best = max(eval_points, key=lambda point: point.answer_accuracy)
    final = max(eval_points, key=lambda point: point.step)

    metrics = read_jsonl(metrics_path)
    train_rows = [row for row in metrics if row.get("type") == "train"]
    if train_rows:
        last_train = train_rows[-1]
        num_train_examples = int(last_train["num_train_examples"])
        train_batch_size = int(last_train["train_batch_size"])
        num_train_steps = len(train_rows)
        final_train_loss = float(last_train["loss"])
        effective_epochs = num_train_steps * train_batch_size / num_train_examples
    else:
        num_train_examples = None
        train_batch_size = None
        num_train_steps = 0
        final_train_loss = None
        effective_epochs = None

    return RunSummary(
        run_name=run_dir.name,
        label=label_for_run(run_dir.name, num_train_examples),
        num_train_examples=num_train_examples,
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        effective_epochs=effective_epochs,
        best_step=best.step,
        best_answer_accuracy=best.answer_accuracy,
        best_format_accuracy=best.format_accuracy,
        final_step=final.step,
        final_answer_accuracy=final.answer_accuracy,
        final_format_accuracy=final.format_accuracy,
        final_train_loss=final_train_loss,
    )


def load_summaries(log_root: Path) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for child in sorted(log_root.iterdir()):
        if child.is_dir():
            summary = summarize_run(child)
            if summary is not None:
                summaries.append(summary)
    return sorted(summaries, key=sort_key)


def select_log_roots(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    if args.rerun_log_root is None:
        return [Path(args.size_sweep_log_root)], [Path(args.filtered_log_root)]

    rerun_root = Path(args.rerun_log_root)
    size_dirs = sorted(
        child
        for child in rerun_root.iterdir()
        if child.is_dir() and child.name.startswith("size")
    )
    filtered_dirs = sorted(
        child
        for child in rerun_root.iterdir()
        if child.is_dir() and child.name.startswith("filtered_full")
    )
    if not size_dirs:
        raise ValueError(f"No size* run directories found under {rerun_root}")
    if not filtered_dirs:
        raise ValueError(f"No filtered_full* run directories found under {rerun_root}")
    return size_dirs, filtered_dirs


def load_summaries_from_dirs(run_dirs: list[Path]) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for run_dir in run_dirs:
        summary = summarize_run(run_dir)
        if summary is not None:
            summaries.append(summary)
    return sorted(summaries, key=sort_key)


def percent(value: float) -> str:
    return f"{100 * value:.2f}"


def maybe_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def summary_to_csv_row(summary: RunSummary) -> dict[str, str]:
    return {
        "run": summary.run_name,
        "label": summary.label,
        "train_examples": "" if summary.num_train_examples is None else str(summary.num_train_examples),
        "effective_epochs": maybe_float(summary.effective_epochs, 2),
        "best_answer_accuracy_percent": percent(summary.best_answer_accuracy),
        "best_step": str(summary.best_step),
        "best_format_accuracy_percent": percent(summary.best_format_accuracy),
        "final_answer_accuracy_percent": percent(summary.final_answer_accuracy),
        "final_step": str(summary.final_step),
        "final_format_accuracy_percent": percent(summary.final_format_accuracy),
        "final_train_loss": maybe_float(summary.final_train_loss, 4),
    }


def write_csv(path: Path, summaries: list[RunSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [summary_to_csv_row(summary) for summary in summaries]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, summaries: list[RunSummary], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "| setting | train examples | effective epochs | best answer acc | best step | final answer acc | final step |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| "
            f"{summary.label} | "
            f"{summary.num_train_examples or ''} | "
            f"{maybe_float(summary.effective_epochs, 1)} | "
            f"{percent(summary.best_answer_accuracy)}% | "
            f"{summary.best_step} | "
            f"{percent(summary.final_answer_accuracy)}% | "
            f"{summary.final_step} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def write_latex_table(path: Path, summaries: list[RunSummary], caption: str, label: str) -> None:
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Setting & Examples & Epochs & Best Acc. & Best Step & Final Acc. \\\\",
        "\\midrule",
    ]
    for summary in summaries:
        examples = "" if summary.num_train_examples is None else str(summary.num_train_examples)
        lines.append(
            f"{latex_escape(summary.label)} & "
            f"{examples} & "
            f"{maybe_float(summary.effective_epochs, 1)} & "
            f"{percent(summary.best_answer_accuracy)}\\% & "
            f"{summary.best_step} & "
            f"{percent(summary.final_answer_accuracy)}\\% \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{latex_escape(caption)}}}",
            f"\\label{{{latex_escape(label)}}}",
            "\\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_filtered_comparison(path: Path, noisy_full: RunSummary, filtered_full: RunSummary) -> None:
    best_delta = filtered_full.best_answer_accuracy - noisy_full.best_answer_accuracy
    final_delta = filtered_full.final_answer_accuracy - noisy_full.final_answer_accuracy
    lines = [
        "# Filtered SFT Comparison",
        "",
        "| setting | train examples | best answer acc | best step | final answer acc | final step |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in [noisy_full, filtered_full]:
        lines.append(
            "| "
            f"{summary.label} | "
            f"{summary.num_train_examples or ''} | "
            f"{percent(summary.best_answer_accuracy)}% | "
            f"{summary.best_step} | "
            f"{percent(summary.final_answer_accuracy)}% | "
            f"{summary.final_step} |"
        )
    lines.extend(
        [
            "",
            f"- Best-answer-accuracy gain from filtering: {100 * best_delta:.2f} percentage points.",
            f"- Final-answer-accuracy gain from filtering: {100 * final_delta:.2f} percentage points.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json_summary(path: Path, summaries: list[RunSummary]) -> None:
    payload = [summary_to_csv_row(summary) for summary in summaries]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_run_summaries_json(path: Path, summaries: list[RunSummary], run_dirs: dict[str, Path]) -> None:
    payload: list[dict[str, Any]] = []
    for summary in summaries:
        run_dir = run_dirs[summary.run_name]
        payload.append(
            {
                "experiment": (
                    "sft_filtered_full"
                    if summary.run_name.startswith("filtered_full")
                    else "sft_noisy_size_sweep"
                ),
                "run_name": summary.run_name,
                "label": summary.label,
                "status": "completed",
                "train_examples": summary.num_train_examples,
                "effective_epochs": summary.effective_epochs,
                "train_batch_size": summary.train_batch_size,
                "num_train_steps": summary.num_train_steps,
                "learning_rate": _last_train_value(run_dir, "learning_rate"),
                "gradient_accumulation_steps": _last_train_value(
                    run_dir,
                    "gradient_accumulation_steps",
                ),
                "best_answer_accuracy": summary.best_answer_accuracy,
                "best_step": summary.best_step,
                "best_format_accuracy": summary.best_format_accuracy,
                "final_answer_accuracy": summary.final_answer_accuracy,
                "final_step": summary.final_step,
                "final_format_accuracy": summary.final_format_accuracy,
                "final_train_loss": summary.final_train_loss,
                "output_dir": _config_value(run_dir, "output_dir"),
                "log_dir": str(run_dir),
                "run_summary_path": str(run_dir / "run_summary.json"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _config_value(run_dir: Path, key: str) -> Any:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    return read_json(config_path).get(key)


def _last_train_value(run_dir: Path, key: str) -> Any:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    train_rows = [row for row in metrics if row.get("type") == "train"]
    if not train_rows:
        return None
    return train_rows[-1].get(key)


def write_run_summaries_archive(path: Path, summaries: list[RunSummary], run_dirs: dict[str, Path]) -> None:
    lines = [
        "# Ch4 SFT Run Summaries",
        "",
        "This archive records the run summary data used by the Ch4 `sft_experiment` writeup section.",
        "",
        "The raw per-run data needed to reproduce the writeup tables and curves are archived under "
        "`artifacts/experiments/ch4/sft_experiment/runs/`.",
        "",
        "| run | experiment | train examples | effective epochs | best answer accuracy | best step | final answer accuracy | final step | final format accuracy | source run summary |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        experiment = (
            "filtered full"
            if summary.run_name.startswith("filtered_full")
            else "noisy size sweep"
        )
        lines.append(
            "| "
            f"`{summary.run_name}` | "
            f"{experiment} | "
            f"{summary.num_train_examples or ''} | "
            f"{maybe_float(summary.effective_epochs, 2)} | "
            f"{summary.best_answer_accuracy:.6f} | "
            f"{summary.best_step} | "
            f"{summary.final_answer_accuracy:.6f} | "
            f"{summary.final_step} | "
            f"{summary.final_format_accuracy:.6f} | "
            f"`{run_dirs[summary.run_name] / 'run_summary.json'}` |"
        )
    lines.extend(
        [
            "",
            "Additional archived artifacts:",
            "",
            "- `artifacts/experiments/ch4/sft_experiment/sft_results_summary.csv`",
            "- `artifacts/experiments/ch4/sft_experiment/sft_results_summary.md`",
            "- `artifacts/experiments/ch4/sft_experiment/sft_result_table.csv`",
            "- `artifacts/experiments/ch4/sft_experiment/sft_result_table.json`",
            "- `artifacts/experiments/ch4/sft_experiment/sft_size_sweep_table.md`",
            "- `artifacts/experiments/ch4/sft_experiment/sft_filtered_comparison_table.md`",
            "- `artifacts/experiments/ch4/sft_experiment/run_summaries.json`",
            "- `artifacts/experiments/ch4/sft_experiment/runs/`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def archive_run_dirs(output_dir: Path, run_dirs: dict[str, Path]) -> None:
    runs_output_dir = output_dir / "runs"
    runs_output_dir.mkdir(parents=True, exist_ok=True)
    selected_names = set(run_dirs)
    for child in runs_output_dir.iterdir():
        if child.is_dir() and child.name in selected_names:
            shutil.rmtree(child)
    for run_name, source_dir in run_dirs.items():
        shutil.copytree(source_dir, runs_output_dir / run_name, ignore=ARCHIVE_IGNORE)


def main() -> None:
    args = parse_args()
    size_dirs, filtered_dirs = select_log_roots(args)
    size_summaries = load_summaries_from_dirs(size_dirs)
    filtered_summaries = load_summaries_from_dirs(filtered_dirs)
    all_summaries = sorted(size_summaries + filtered_summaries, key=sort_key)
    run_dirs = {path.name: path for path in size_dirs + filtered_dirs}
    output_dir = Path(args.output_dir)

    if args.archive_runs:
        archive_run_dirs(output_dir, run_dirs)
        run_dirs = {run_name: output_dir / "runs" / run_name for run_name in run_dirs}

    write_csv(output_dir / "sft_result_table.csv", all_summaries)
    write_json_summary(output_dir / "sft_result_table.json", all_summaries)
    write_run_summaries_json(output_dir / "run_summaries.json", all_summaries, run_dirs)
    write_run_summaries_archive(
        output_dir / "run_summaries_archive.md",
        all_summaries,
        run_dirs,
    )
    write_markdown_table(
        output_dir / "sft_size_sweep_table.md",
        size_summaries,
        "SFT Dataset Size Sweep",
    )
    write_latex_table(
        output_dir / "sft_size_sweep_table.tex",
        size_summaries,
        "SFT validation accuracy across dataset sizes.",
        "tab:sft-size-sweep",
    )

    noisy_full = next(summary for summary in size_summaries if summary.run_name.startswith("sizefull"))
    filtered_full = filtered_summaries[0]
    write_filtered_comparison(
        output_dir / "sft_filtered_comparison_table.md",
        noisy_full,
        filtered_full,
    )
    write_latex_table(
        output_dir / "sft_filtered_comparison_table.tex",
        [noisy_full, filtered_full],
        "Effect of reward filtering on full-dataset SFT.",
        "tab:sft-filtered",
    )

    print(f"Wrote SFT result tables to {output_dir}")


if __name__ == "__main__":
    main()
