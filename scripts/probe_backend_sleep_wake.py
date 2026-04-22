#!/usr/bin/env python3
"""Probe backend sleep/wake behavior and GPU memory on a single GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from vllm import SamplingParams

from cs336_alignment.backend_probe import build_probe_summary, summarize_training_runtime
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
    parser.add_argument("--enable-training-offload", action="store_true")
    parser.add_argument("--disable-training-offload", dest="enable_training_offload", action="store_false")
    parser.set_defaults(enable_training_offload=False)
    parser.add_argument("--offload-optimizer-state", action="store_true")
    parser.add_argument("--no-offload-optimizer-state", dest="offload_optimizer_state", action="store_false")
    parser.set_defaults(offload_optimizer_state=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--materialize-optimizer-state", action="store_true")
    parser.add_argument("--skip-materialize-optimizer-state", dest="materialize_optimizer_state", action="store_false")
    parser.set_defaults(materialize_optimizer_state=True)
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


def snapshot_cpu_memory(label: str) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    import psutil

    process = psutil.Process()
    vm = psutil.virtual_memory()
    return {
        "label": label,
        "timestamp": time.time(),
        "process_rss_gib": process.memory_info().rss / BYTES_PER_GIB,
        "system_used_gib": vm.used / BYTES_PER_GIB,
        "system_available_gib": vm.available / BYTES_PER_GIB,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def probe_generate(
    llm: Any,
    prompts: list[str],
    max_new_tokens: int,
) -> dict[str, Any]:
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


def materialize_training_state(
    manager: BackendLifecycleManager,
    prompts: list[str],
    *,
    learning_rate: float,
) -> AdamW:
    policy, tokenizer = manager.enter_training_phase()
    optimizer = AdamW(policy.parameters(), lr=learning_rate)
    manager.attach_training_optimizer(optimizer)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    tokenizer.padding_side = original_padding_side
    input_ids = tokenized["input_ids"].to(manager.training_config.device)
    attention_mask = tokenized["attention_mask"].to(manager.training_config.device)
    labels = input_ids.clone()

    optimizer.zero_grad(set_to_none=True)
    outputs = policy(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return optimizer


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
            enable_cpu_offload_during_inference=args.enable_training_offload,
            offload_optimizer_state=args.offload_optimizer_state,
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
    cpu_snapshots: list[dict[str, Any]] = []
    state_snapshots: list[dict[str, Any]] = []

    snapshots.append(snapshot_cuda_memory(args.device, "baseline"))
    cpu_snapshots.append(snapshot_cpu_memory("baseline"))
    manager.enter_training_phase()
    snapshots.append(snapshot_cuda_memory(args.device, "after_training_phase"))
    cpu_snapshots.append(snapshot_cpu_memory("after_training_phase"))

    optimizer = None
    if args.materialize_optimizer_state:
        optimizer = materialize_training_state(
            manager,
            prompts,
            learning_rate=args.learning_rate,
        )
        snapshots.append(snapshot_cuda_memory(args.device, "after_optimizer_step"))
        cpu_snapshots.append(snapshot_cpu_memory("after_optimizer_step"))
        state_snapshots.append(
            {
                "label": "after_optimizer_step",
                **manager.debug_state(),
                **summarize_training_runtime(manager.training_model(), optimizer),
            }
        )

    manager.initialize_inference_backend()
    snapshots.append(snapshot_cuda_memory(args.device, "after_inference_backend_init_sleeping"))
    cpu_snapshots.append(snapshot_cpu_memory("after_inference_backend_init_sleeping"))
    if optimizer is not None:
        state_snapshots.append(
            {
                "label": "after_inference_backend_init_sleeping",
                **manager.debug_state(),
                **summarize_training_runtime(manager.training_model(), optimizer),
            }
        )

    transition_start = time.time()
    llm = manager.enter_inference_phase(sync_weights=True)
    first_transition_seconds = time.time() - transition_start
    snapshots.append(snapshot_cuda_memory(args.device, "after_enter_inference_phase"))
    cpu_snapshots.append(snapshot_cpu_memory("after_enter_inference_phase"))
    if optimizer is not None:
        state_snapshots.append(
            {
                "label": "after_enter_inference_phase",
                **manager.debug_state(),
                **summarize_training_runtime(manager._policy, optimizer),  # noqa: SLF001
            }
        )

    first_generate = probe_generate(llm, prompts, args.max_new_tokens)
    first_generate["transition_seconds"] = first_transition_seconds
    snapshots.append(snapshot_cuda_memory(args.device, "after_first_generate"))
    cpu_snapshots.append(snapshot_cpu_memory("after_first_generate"))

    transition_start = time.time()
    manager.after_inference()
    first_return_seconds = time.time() - transition_start
    snapshots.append(snapshot_cuda_memory(args.device, "after_return_to_training"))
    cpu_snapshots.append(snapshot_cpu_memory("after_return_to_training"))
    if optimizer is not None:
        state_snapshots.append(
            {
                "label": "after_return_to_training",
                **manager.debug_state(),
                **summarize_training_runtime(manager.training_model(), optimizer),
            }
        )

    transition_start = time.time()
    llm = manager.enter_inference_phase(sync_weights=True)
    second_transition_seconds = time.time() - transition_start
    snapshots.append(snapshot_cuda_memory(args.device, "after_second_enter_inference_phase"))
    cpu_snapshots.append(snapshot_cpu_memory("after_second_enter_inference_phase"))
    if optimizer is not None:
        state_snapshots.append(
            {
                "label": "after_second_enter_inference_phase",
                **manager.debug_state(),
                **summarize_training_runtime(manager._policy, optimizer),  # noqa: SLF001
            }
        )

    second_generate = probe_generate(llm, prompts, args.max_new_tokens)
    second_generate["transition_seconds"] = second_transition_seconds
    snapshots.append(snapshot_cuda_memory(args.device, "after_second_generate"))
    cpu_snapshots.append(snapshot_cpu_memory("after_second_generate"))

    transition_start = time.time()
    manager.after_inference()
    second_return_seconds = time.time() - transition_start
    snapshots.append(snapshot_cuda_memory(args.device, "after_final_sleep"))
    cpu_snapshots.append(snapshot_cpu_memory("after_final_sleep"))
    if optimizer is not None:
        state_snapshots.append(
            {
                "label": "after_final_sleep",
                **manager.debug_state(),
                **summarize_training_runtime(manager.training_model(), optimizer),
                "return_to_training_seconds": second_return_seconds,
            }
        )

    memory_path = log_dir / "memory_snapshots.jsonl"
    for snapshot in snapshots:
        append_jsonl(memory_path, snapshot)
    cpu_memory_path = log_dir / "cpu_memory_snapshots.jsonl"
    for snapshot in cpu_snapshots:
        append_jsonl(cpu_memory_path, snapshot)
    state_path = log_dir / "state_snapshots.jsonl"
    for snapshot in state_snapshots:
        append_jsonl(state_path, snapshot)

    summary = build_probe_summary(snapshots, first_generate, second_generate)
    summary.update(
        {
            "cpu_rss_gib_by_label": {
                snapshot["label"]: snapshot["process_rss_gib"] for snapshot in cpu_snapshots
            },
            "training_state_by_label": {
                snapshot["label"]: {
                    "current_phase": snapshot["current_phase"],
                    "training_backend_offloaded": snapshot["training_backend_offloaded"],
                    "num_grad_tensors": snapshot["num_grad_tensors"],
                    "num_cuda_grad_tensors": snapshot["num_cuda_grad_tensors"],
                    "num_optimizer_tensors": snapshot["num_optimizer_tensors"],
                    "num_cuda_optimizer_tensors": snapshot["num_cuda_optimizer_tensors"],
                    "policy_parameter_devices": snapshot["policy_parameter_devices"],
                    "optimizer_tensor_devices": snapshot["optimizer_tensor_devices"],
                }
                for snapshot in state_snapshots
            },
            "transition_seconds": {
                "first_enter_inference": first_transition_seconds,
                "first_return_to_training": first_return_seconds,
                "second_enter_inference": second_transition_seconds,
                "second_return_to_training": second_return_seconds,
            },
            "training_offload_enabled": args.enable_training_offload,
            "optimizer_state_offload_enabled": args.offload_optimizer_state,
            "materialized_optimizer_state": args.materialize_optimizer_state,
        }
    )
    write_json(log_dir / "probe_summary.json", summary)


if __name__ == "__main__":
    main()
