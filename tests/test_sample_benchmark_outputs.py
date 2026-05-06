from scripts.sample_benchmark_outputs import (
    alpaca_model_preference,
    build_pair_index,
    classify_binary_pair,
    sample_records,
    truncate_text,
)


def test_build_pair_index_uses_key_function() -> None:
    records = [
        {"id": "a", "value": 1},
        {"id": "b", "value": 2},
    ]

    assert build_pair_index(records, lambda record: record["id"]) == {
        "a": {"id": "a", "value": 1},
        "b": {"id": "b", "value": 2},
    }


def test_sample_records_is_deterministic() -> None:
    records = [{"id": index} for index in range(20)]

    first = sample_records(records, count=5, seed=7)
    second = sample_records(records, count=5, seed=7)

    assert first == second
    assert len(first) == 5


def test_classify_binary_pair_names_improvements_and_regressions() -> None:
    assert classify_binary_pair(False, True, "baseline", "candidate") == (
        "baseline_wrong_candidate_correct"
    )
    assert classify_binary_pair(True, False, "baseline", "candidate") == (
        "baseline_correct_candidate_wrong"
    )
    assert classify_binary_pair(False, False, "baseline", "candidate") == "both_wrong"
    assert classify_binary_pair(True, True, "baseline", "candidate") == "both_correct"


def test_alpaca_model_preference_matches_leaderboard_convention() -> None:
    assert alpaca_model_preference({"preference": 2.0}) == "model_preferred"
    assert alpaca_model_preference({"preference": 1.0}) == "reference_preferred"
    assert alpaca_model_preference({"preference": 1.5}) == "draw"


def test_truncate_text_preserves_short_text_and_marks_long_text() -> None:
    assert truncate_text("short", 10) == "short"
    assert truncate_text("abcdefghijklmnopqrstuvwxyz", 10) == "abcdefg..."
