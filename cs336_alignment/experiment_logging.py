from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal


ExperimentKind = Literal["sft", "ei", "grpo"]


def read_jsonl(path: str, max_examples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if max_examples is not None and len(rows) >= max_examples:
                break
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def load_prompt_template(path: str) -> str:
    template = Path(path).read_text(encoding="utf-8")
    if "{question}" not in template:
        raise ValueError(f"Prompt template must contain '{{question}}': {path}")
    return template


def get_question(example: dict[str, Any]) -> str:
    question = example.get("question") or example.get("problem")
    if question is None:
        raise ValueError("Example must contain `question` or `problem`.")
    return question


def get_ground_truth(example: dict[str, Any]) -> str:
    ground_truth = example.get("ground_truth", example.get("answer"))
    if ground_truth is None:
        raise ValueError("Example must contain `ground_truth` or `answer`.")
    return str(ground_truth)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class ExperimentLogWriter:
    def __init__(self, experiment: ExperimentKind, output_dir: Path, log_dir: Path) -> None:
        self.experiment = experiment
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.metrics_path = log_dir / "metrics.jsonl"

    def write_config(self, config: dict[str, Any]) -> None:
        write_json(self.log_dir / "config.json", config)

    def append_metric(self, record: dict[str, Any]) -> None:
        append_jsonl(self.metrics_path, record)

    def append_eval_generation(self, step: int, record: dict[str, Any]) -> None:
        append_jsonl(self.eval_generations_path(step), record)

    def write_eval_summary(self, step: int, summary: dict[str, Any]) -> None:
        write_json(self.eval_summary_path(step), summary)

    def write_rollouts(self, step: int, records: list[dict[str, Any]]) -> None:
        write_jsonl(self.step_output_dir(step) / "rollouts.jsonl", records)

    def write_accepted_sft(self, step: int, records: list[dict[str, Any]]) -> None:
        if self.experiment != "ei":
            raise ValueError("accepted_sft.jsonl is only defined for EI experiments.")
        write_jsonl(self.step_output_dir(step) / "accepted_sft.jsonl", records)

    def write_sample_rollouts(self, step: int, records: list[dict[str, Any]]) -> None:
        if self.experiment != "grpo":
            raise ValueError("sample_rollouts.jsonl is only defined for GRPO experiments.")
        write_jsonl(self.step_log_dir(step) / "sample_rollouts.jsonl", records)

    def write_rollout_summary(self, step: int, summary: dict[str, Any]) -> None:
        write_json(self.step_log_dir(step) / "rollout_summary.json", summary)

    def write_run_summary(self, summary: dict[str, Any]) -> None:
        write_json(self.log_dir / "run_summary.json", summary)

    def eval_generations_path(self, step: int) -> Path:
        if self.experiment == "sft":
            return self.output_dir / f"eval_generations_step_{step:06d}.jsonl"
        if self.experiment == "ei":
            return self.output_dir / f"eval_generations_ei_step_{step:03d}.jsonl"
        return self.output_dir / f"eval_generations_grpo_step_{step:06d}.jsonl"

    def eval_summary_path(self, step: int) -> Path:
        if self.experiment == "sft":
            return self.log_dir / f"eval_summary_step_{step:06d}.json"
        if self.experiment == "ei":
            return self.log_dir / f"eval_summary_ei_step_{step:03d}.json"
        return self.log_dir / f"eval_summary_grpo_step_{step:06d}.json"

    def step_output_dir(self, step: int) -> Path:
        if self.experiment == "ei":
            return self.output_dir / f"ei_step_{step:03d}"
        if self.experiment == "grpo":
            return self.output_dir / f"grpo_step_{step:06d}"
        raise ValueError("Step output directories are only defined for EI and GRPO.")

    def step_log_dir(self, step: int) -> Path:
        if self.experiment == "ei":
            return self.log_dir / f"ei_step_{step:03d}"
        if self.experiment == "grpo":
            return self.log_dir / f"grpo_step_{step:06d}"
        raise ValueError("Step log directories are only defined for EI and GRPO.")
