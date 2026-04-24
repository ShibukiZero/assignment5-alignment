from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Literal

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn


RewardFnName = Literal["r1_zero", "question_only"]


def resolve_reward_fn(name: RewardFnName):
    if name == "r1_zero":
        return r1_zero_reward_fn
    if name == "question_only":
        return question_only_reward_fn
    raise ValueError(f"Unknown reward function: {name}")


def score_response_with_reward_name(task: tuple[str, str, RewardFnName]) -> dict[str, float]:
    response, ground_truth, reward_fn_name = task
    reward_fn = resolve_reward_fn(reward_fn_name)
    return reward_fn(response, ground_truth)


def score_responses(
    responses: list[str],
    ground_truths: list[str],
    reward_fn_name: RewardFnName = "r1_zero",
    reward_workers: int = 1,
) -> list[dict[str, float]]:
    if len(responses) != len(ground_truths):
        raise ValueError("responses and ground_truths must have the same length.")

    tasks = [(response, ground_truth, reward_fn_name) for response, ground_truth in zip(responses, ground_truths)]
    if reward_workers <= 1 or len(tasks) <= 1:
        return [score_response_with_reward_name(task) for task in tasks]

    # Workers receive only strings and a reward-function name, never GPU/vLLM objects.
    with ProcessPoolExecutor(max_workers=reward_workers) as executor:
        return list(executor.map(score_response_with_reward_name, tasks))
