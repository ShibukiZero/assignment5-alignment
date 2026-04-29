from __future__ import annotations

from typing import Any

import torch


def build_probe_summary(
    snapshots: list[dict[str, Any]],
    first_generate: dict[str, Any],
    second_generate: dict[str, Any],
) -> dict[str, Any]:
    used_gib_by_label = {snapshot["label"]: snapshot["used_gib"] for snapshot in snapshots}
    return {
        "first_generate": first_generate,
        "second_generate": second_generate,
        "memory_labels": [snapshot["label"] for snapshot in snapshots],
        "used_gib_by_label": used_gib_by_label,
        "sleep_release_after_first_generate_gib": (
            used_gib_by_label["after_first_generate"]
            - used_gib_by_label["after_return_to_training"]
        ),
        "sleep_release_after_second_generate_gib": (
            used_gib_by_label["after_second_generate"]
            - used_gib_by_label["after_final_sleep"]
        ),
    }


def summarize_training_runtime(policy: Any, optimizer: Any | None) -> dict[str, Any]:
    grad_tensors = []
    param_devices: dict[str, int] = {}
    for parameter in policy.parameters():
        device = str(parameter.device)
        param_devices[device] = param_devices.get(device, 0) + 1
        if parameter.grad is not None:
            grad_tensors.append(parameter.grad)

    optimizer_tensors = []
    if optimizer is not None:
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    optimizer_tensors.append(value)

    optimizer_devices: dict[str, int] = {}
    for tensor in optimizer_tensors:
        device = str(tensor.device)
        optimizer_devices[device] = optimizer_devices.get(device, 0) + 1

    return {
        "policy_parameter_devices": param_devices,
        "num_grad_tensors": len(grad_tensors),
        "num_cuda_grad_tensors": sum(1 for grad in grad_tensors if grad.device.type == "cuda"),
        "optimizer_tensor_devices": optimizer_devices,
        "num_optimizer_tensors": len(optimizer_tensors),
        "num_cuda_optimizer_tensors": sum(1 for tensor in optimizer_tensors if tensor.device.type == "cuda"),
    }
