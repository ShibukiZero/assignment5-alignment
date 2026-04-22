from __future__ import annotations

from typing import Any


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
