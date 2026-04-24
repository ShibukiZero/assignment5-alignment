from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import logging
import signal
from typing import Literal

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn


RewardFnName = Literal["r1_zero", "question_only"]
ZERO_REWARD = {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}


class RewardScoringTimeout(RuntimeError):
    pass


def resolve_reward_fn(name: RewardFnName):
    if name == "r1_zero":
        return r1_zero_reward_fn
    if name == "question_only":
        return question_only_reward_fn
    raise ValueError(f"Unknown reward function: {name}")


def handle_reward_timeout(signum, frame) -> None:
    raise RewardScoringTimeout("Reward scoring timed out.")


@contextmanager
def reward_timeout(seconds: int | None):
    if seconds is None or seconds <= 0:
        yield
        return

    previous_handler = signal.signal(signal.SIGALRM, handle_reward_timeout)
    previous_alarm = signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        if previous_alarm:
            signal.alarm(previous_alarm)
        signal.signal(signal.SIGALRM, previous_handler)


def score_response_with_reward_name(
    task: tuple[str, str, RewardFnName, int | None],
) -> dict[str, float]:
    response, ground_truth, reward_fn_name, timeout_seconds = task
    reward_fn = resolve_reward_fn(reward_fn_name)
    return score_response_with_reward_fn(
        response=response,
        ground_truth=ground_truth,
        reward_fn=reward_fn,
        timeout_seconds=timeout_seconds,
    )


def score_response_with_reward_fn(
    response: str,
    ground_truth: str,
    reward_fn,
    timeout_seconds: int | None = 5,
) -> dict[str, float]:
    try:
        with reward_timeout(timeout_seconds):
            return reward_fn(response, ground_truth)
    except RewardScoringTimeout:
        return dict(ZERO_REWARD)


def quiet_reward_parser_logs() -> None:
    for logger_name in (
        "__init__",
        "math_verify",
        "latex2sympy2_extended",
        "pylatexenc",
        "sympy",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def score_responses(
    responses: list[str],
    ground_truths: list[str],
    reward_fn_name: RewardFnName = "r1_zero",
    reward_workers: int = 1,
    quiet_parser_logs: bool = True,
    timeout_seconds: int | None = 5,
) -> list[dict[str, float]]:
    if len(responses) != len(ground_truths):
        raise ValueError("responses and ground_truths must have the same length.")

    tasks = [
        (response, ground_truth, reward_fn_name, timeout_seconds)
        for response, ground_truth in zip(responses, ground_truths)
    ]
    if reward_workers <= 1 or len(tasks) <= 1:
        if quiet_parser_logs:
            quiet_reward_parser_logs()
        return [score_response_with_reward_name(task) for task in tasks]

    # Workers receive only strings and a reward-function name, never GPU/vLLM objects.
    worker_initializer = quiet_reward_parser_logs if quiet_parser_logs else None
    with ProcessPoolExecutor(max_workers=reward_workers, initializer=worker_initializer) as executor:
        return list(executor.map(score_response_with_reward_name, tasks))
