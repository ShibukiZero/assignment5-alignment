#!/usr/bin/env python3
"""Probe backend sleep/wake behavior and GPU memory on a single GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from vllm import SamplingParams

from cs336_alignment.backend_probe import build_probe_summary
from cs336_alignment.backend_lifecycle import (
    BackendLifecycleManager,
    InferenceBackendConfig,
    TrainingBackendConfig,
)


DEFAULT_MODEL = "/root/autodl-tmp/a5-alignment/models/Qwen2.5-Math-1.5B"
DEFAULT_VAL_PATH = "/root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric/validation.jsonl"
DEFAULT_PROMPT_TEMPLATE = "cs336_alignment/prompts/r1_zero.prompt"
DEFAULT_LOG_DIR = ".agents/logs/backend_sleep_wake_probe"
BYTES_PER_GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe single-GPU backend sleep/wake behavior.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sleep-level", type=int, default=1, choices=[1, 2])
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--disable-prefix-caching", dest="enable_prefix_caching", action="store_false")
    parser.set_defaults(enable_prefix_caching=True)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    return parser.parse_args()


def read_jsonl(path: str, max_examples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if len(rows) >= max_examples:
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


def snapshot_cuda_memory(device: str, label: str) -> dict[str, Any]:
    cuda_device = torch.device(device)
    torch.cuda.synchronize(cuda_device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_device)
    return {
        "label": label,
        "timestamp": time.time(),
        "free_gib": free_bytes / BYTES_PER_GIB,
        "used_gib": (total_bytes - free_bytes) / BYTES_PER_GIB,
        "total_gib": total_bytes / BYTES_PER_GIB,
        "torch_allocated_gib": torch.cuda.memory_allocated(cuda_device) / BYTES_PER_GIB,
        "torch_reserved_gib": torch.cuda.memory_reserved(cuda_device) / BYTES_PER_GIB,
        "torch_max_allocated_gib": torch.cuda.max_memory_allocated(cuda_device) / BYTES_PER_GIB,
        "torch_max_reserved_gib": torch.cuda.max_memory_reserved(cuda_device) / BYTES_PER_GIB,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def probe_generate(
    manager: BackendLifecycleManager,
    prompts: list[str],
    max_new_tokens: int,
) -> dict[str, Any]:
    llm = manager.enter_inference_phase(sync_weights=True)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    return {
        "generate_seconds": elapsed,
        "num_prompts": len(prompts),
        "avg_output_chars": sum(len(output.outputs[0].text) for output in outputs) / len(outputs),
    }


def main() -> None:
    args = parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(log_dir / "config.json", vars(args))

    examples = read_jsonl(args.val_path, max_examples=args.num_prompts)
    prompt_template = load_prompt_template(args.prompt_template)
    prompts = [prompt_template.format(question=get_question(example)) for example in examples]

    manager = BackendLifecycleManager(
        training_config=TrainingBackendConfig(
            model_id=args.model,
            device=args.device,
        ),
        inference_config=InferenceBackendConfig(
            model_id=args.model,
            device=args.device,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=dtype,
            enable_prefix_caching=args.enable_prefix_caching,
            enable_sleep_mode=True,
            sleep_level=args.sleep_level,
            reset_prefix_cache_after_weight_sync=True,
        ),
    )

    snapshots: list[dict[str, Any]] = []

    snapshots.append(snapshot_cuda_memory(args.device, "baseline"))
    manager.enter_training_phase()
    snapshots.append(snapshot_cuda_memory(args.device, "after_training_phase"))

    manager.initialize_inference_backend()
    snapshots.append(snapshot_cuda_memory(args.device, "after_inference_backend_init_sleeping"))

    first_generate = probe_generate(manager, prompts, args.max_new_tokens)
    snapshots.append(snapshot_cuda_memory(args.device, "after_first_generate"))

    manager.after_inference()
    snapshots.append(snapshot_cuda_memory(args.device, "after_return_to_training"))

    second_generate = probe_generate(manager, prompts, args.max_new_tokens)
    snapshots.append(snapshot_cuda_memory(args.device, "after_second_generate"))

    manager.after_inference()
    snapshots.append(snapshot_cuda_memory(args.device, "after_final_sleep"))

    memory_path = log_dir / "memory_snapshots.jsonl"
    for snapshot in snapshots:
        append_jsonl(memory_path, snapshot)

    summary = build_probe_summary(snapshots, first_generate, second_generate)
    write_json(log_dir / "probe_summary.json", summary)


if __name__ == "__main__":
    main()
