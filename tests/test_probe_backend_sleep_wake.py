from __future__ import annotations

from cs336_alignment.backend_probe import build_probe_summary


def test_build_probe_summary_computes_sleep_release_deltas():
    snapshots = [
        {"label": "baseline", "used_gib": 10.0},
        {"label": "after_first_generate", "used_gib": 20.5},
        {"label": "after_return_to_training", "used_gib": 14.0},
        {"label": "after_second_generate", "used_gib": 21.0},
        {"label": "after_final_sleep", "used_gib": 13.5},
    ]

    summary = build_probe_summary(
        snapshots=snapshots,
        first_generate={"generate_seconds": 1.0},
        second_generate={"generate_seconds": 0.9},
    )

    assert summary["memory_labels"] == [
        "baseline",
        "after_first_generate",
        "after_return_to_training",
        "after_second_generate",
        "after_final_sleep",
    ]
    assert summary["used_gib_by_label"]["after_first_generate"] == 20.5
    assert summary["sleep_release_after_first_generate_gib"] == 6.5
    assert summary["sleep_release_after_second_generate_gib"] == 7.5
