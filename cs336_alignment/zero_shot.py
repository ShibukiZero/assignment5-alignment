from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams


SUPPLEMENT_SYSTEM_PROMPT = """Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some controversial topics.

# Query:

```{instruction}```

# Answer:
"""


def format_supplement_prompt(instruction: str) -> str:
    return SUPPLEMENT_SYSTEM_PROMPT.format(instruction=instruction)


def greedy_sampling_params(max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["# Query:"],
    )


def generate_texts(
    prompts: list[str],
    model: str,
    tensor_parallel_size: int,
    max_tokens: int,
    gpu_memory_utilization: float,
    trust_remote_code: bool,
) -> tuple[list[str], dict[str, Any]]:
    load_started_at = time.perf_counter()
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
    )
    model_load_seconds = time.perf_counter() - load_started_at
    generation_started_at = time.perf_counter()
    outputs = llm.generate(prompts, greedy_sampling_params(max_tokens=max_tokens))
    generation_elapsed_seconds = time.perf_counter() - generation_started_at
    generations = [output.outputs[0].text for output in outputs]
    return generations, {
        "model_load_seconds": model_load_seconds,
        "generation_elapsed_seconds": generation_elapsed_seconds,
        "examples_per_second": (
            len(prompts) / generation_elapsed_seconds
            if generation_elapsed_seconds > 0
            else None
        ),
        "num_examples": len(prompts),
    }


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
