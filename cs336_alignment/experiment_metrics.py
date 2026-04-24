from __future__ import annotations

from statistics import mean

import torch


BYTES_PER_GIB = 1024**3


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def cuda_memory_metrics(device: str) -> dict[str, float]:
    if not device.startswith("cuda"):
        return {}
    cuda_device = torch.device(device)
    return {
        "cuda_memory_allocated_gib": torch.cuda.memory_allocated(cuda_device) / BYTES_PER_GIB,
        "cuda_memory_reserved_gib": torch.cuda.memory_reserved(cuda_device) / BYTES_PER_GIB,
        "cuda_max_memory_allocated_gib": torch.cuda.max_memory_allocated(cuda_device) / BYTES_PER_GIB,
        "cuda_max_memory_reserved_gib": torch.cuda.max_memory_reserved(cuda_device) / BYTES_PER_GIB,
    }
