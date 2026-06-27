#!/usr/bin/env python3
"""Train the supplement SFT checkpoint with DPO on Anthropic HH."""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.optim import RMSprop
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from cs336_alignment.dpo import per_instance_dpo_loss_with_metrics
from cs336_alignment.dpo_data import load_hh_preference_data
from cs336_alignment.experiment_logging import append_jsonl, write_json
from cs336_alignment.experiment_metrics import cuda_memory_metrics


DEFAULT_MODEL = "runs/supplement/ch3/sft/final_model"
DEFAULT_HH_PATH = "data/datasets/hh_rlhf"
DEFAULT_OUTPUT_DIR = "runs/supplement/ch5/dpo"
DEFAULT_LOG_DIR = "runs/logs/ch5/dpo"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--hh-path", default=DEFAULT_HH_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--rmsprop-alpha", type=float, default=0.99)
    parser.add_argument("--rmsprop-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--ref-device", default="cuda:1")
    parser.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Transformers attention implementation, e.g. flash_attention_2.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)
    parser.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args: argparse.Namespace) -> None:
    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be positive.")
    if args.validation_size < 0:
        raise ValueError("--validation-size must be non-negative.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")
    if args.beta <= 0:
        raise ValueError("--beta must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive.")
    if args.max_train_examples is not None and args.max_train_examples <= 0:
        raise ValueError("--max-train-examples must be positive when set.")
    if args.max_val_examples is not None and args.max_val_examples <= 0:
        raise ValueError("--max-val-examples must be positive when set.")


def split_records(
    records: list[dict[str, Any]],
    validation_size: int,
    seed: int,
    max_train_examples: int | None,
    max_val_examples: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    validation_size = min(validation_size, len(shuffled) - 1)
    val_records = shuffled[:validation_size]
    train_records = shuffled[validation_size:]
    if max_val_examples is not None:
        val_records = val_records[:max_val_examples]
    if max_train_examples is not None:
        train_records = train_records[:max_train_examples]
    if not train_records:
        raise ValueError("No HH training records after split/capping.")
    if not val_records:
        raise ValueError("No HH validation records after split/capping.")
    return train_records, val_records


def load_model(
    model_path: str,
    device: str,
    torch_dtype: torch.dtype,
    attn_implementation: str,
) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    model.config.use_cache = False
    model.to(device)
    return model


def save_model(model: PreTrainedModel, tokenizer: Any, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


def optimizer_step(
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
) -> float:
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm.detach().cpu().item())


def evaluate_dpo(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tokenizer: Any,
    records: list[dict[str, Any]],
    beta: float,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    ref_model.eval()
    losses: list[float] = []
    correct: list[float] = []
    margins: list[float] = []
    start_time = time.time()
    with torch.no_grad():
        for record in records:
            loss, metrics = per_instance_dpo_loss_with_metrics(
                lm=model,
                lm_ref=ref_model,
                tokenizer=tokenizer,
                beta=beta,
                prompt=str(record["instruction"]),
                response_chosen=str(record["chosen"]),
                response_rejected=str(record["rejected"]),
            )
            losses.append(float(loss.detach().cpu().item()))
            correct.append(float(metrics["classification_correct"].detach().cpu().item()))
            margins.append(float(metrics["dpo_margin"].detach().cpu().item()))

    if was_training:
        model.train()
    return {
        "loss": mean(losses),
        "classification_accuracy": mean(correct),
        "mean_dpo_margin": mean(margins),
        "num_examples": len(records),
        "eval_seconds": time.time() - start_time,
    }


def train_one_step(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tokenizer: Any,
    records: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    beta: float,
    max_grad_norm: float,
) -> dict[str, Any]:
    losses: list[float] = []
    correct: list[float] = []
    margins: list[float] = []
    step_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for record in records:
        loss, metrics = per_instance_dpo_loss_with_metrics(
            lm=model,
            lm_ref=ref_model,
            tokenizer=tokenizer,
            beta=beta,
            prompt=str(record["instruction"]),
            response_chosen=str(record["chosen"]),
            response_rejected=str(record["rejected"]),
        )
        (loss / len(records)).backward()
        losses.append(float(loss.detach().cpu().item()))
        correct.append(float(metrics["classification_correct"].detach().cpu().item()))
        margins.append(float(metrics["dpo_margin"].detach().cpu().item()))

    grad_norm = optimizer_step(
        model=model,
        optimizer=optimizer,
        max_grad_norm=max_grad_norm,
    )
    return {
        "loss": mean(losses),
        "classification_accuracy": mean(correct),
        "mean_dpo_margin": mean(margins),
        "num_examples": len(records),
        "grad_norm": grad_norm,
        "step_seconds": time.time() - step_start,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "metrics.jsonl"
    write_json(log_dir / "config.json", vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch_dtype_from_name(args.torch_dtype)
    model = load_model(
        model_path=args.model,
        device=args.policy_device,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    ref_model = load_model(
        model_path=args.model,
        device=args.ref_device,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)

    records = load_hh_preference_data(args.hh_path)
    train_records, val_records = split_records(
        records=records,
        validation_size=args.validation_size,
        seed=args.seed,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
    )
    optimizer = RMSprop(
        model.parameters(),
        lr=args.learning_rate,
        alpha=args.rmsprop_alpha,
        eps=args.rmsprop_eps,
        weight_decay=args.weight_decay,
    )

    logger.info("HH records: %d train, %d validation", len(train_records), len(val_records))
    logger.info("gradient accumulation steps: %d", args.gradient_accumulation_steps)
    logger.info("beta: %.4f learning_rate: %.3e", args.beta, args.learning_rate)

    best_val_accuracy = -math.inf
    best_val_loss = math.inf
    global_step = 0
    train_start = time.time()

    initial_eval = evaluate_dpo(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        records=val_records,
        beta=args.beta,
    )
    append_jsonl(
        metrics_path,
        {
            "type": "eval",
            "step": global_step,
            "initial": True,
            **initial_eval,
            "elapsed_seconds": time.time() - train_start,
        },
    )
    best_val_accuracy = float(initial_eval["classification_accuracy"])
    best_val_loss = float(initial_eval["loss"])
    if args.save_best:
        save_model(model, tokenizer, output_dir / "best_model")

    for epoch in range(args.num_epochs):
        epoch_records = list(train_records)
        random.Random(args.seed + epoch).shuffle(epoch_records)
        step_chunks = [
            epoch_records[i : i + args.gradient_accumulation_steps]
            for i in range(0, len(epoch_records), args.gradient_accumulation_steps)
        ]
        progress = tqdm(step_chunks, desc=f"DPO epoch {epoch + 1}")
        for chunk in progress:
            if args.policy_device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(torch.device(args.policy_device))

            train_metrics = train_one_step(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                records=chunk,
                optimizer=optimizer,
                beta=args.beta,
                max_grad_norm=args.max_grad_norm,
            )
            global_step += 1
            train_record = {
                "type": "train",
                "epoch": epoch + 1,
                "step": global_step,
                "learning_rate": args.learning_rate,
                **train_metrics,
                "elapsed_seconds": time.time() - train_start,
            }
            train_record.update(cuda_memory_metrics(args.policy_device))
            append_jsonl(metrics_path, train_record)

            if global_step % args.log_every == 0:
                logger.info(
                    "step=%d epoch=%d loss=%.4f acc=%.4f margin=%.4f grad_norm=%.4f",
                    global_step,
                    epoch + 1,
                    train_metrics["loss"],
                    train_metrics["classification_accuracy"],
                    train_metrics["mean_dpo_margin"],
                    train_metrics["grad_norm"],
                )

            if global_step % args.eval_every == 0:
                val_metrics = evaluate_dpo(
                    model=model,
                    ref_model=ref_model,
                    tokenizer=tokenizer,
                    records=val_records,
                    beta=args.beta,
                )
                eval_record = {
                    "type": "eval",
                    "epoch": epoch + 1,
                    "step": global_step,
                    **val_metrics,
                    "elapsed_seconds": time.time() - train_start,
                }
                append_jsonl(metrics_path, eval_record)
                val_accuracy = float(val_metrics["classification_accuracy"])
                val_loss = float(val_metrics["loss"])
                logger.info(
                    "eval step=%d loss=%.4f acc=%.4f margin=%.4f",
                    global_step,
                    val_loss,
                    val_accuracy,
                    val_metrics["mean_dpo_margin"],
                )
                if (val_accuracy > best_val_accuracy) or (
                    val_accuracy == best_val_accuracy and val_loss < best_val_loss
                ):
                    best_val_accuracy = val_accuracy
                    best_val_loss = val_loss
                    if args.save_best:
                        save_model(model, tokenizer, output_dir / "best_model")

    final_eval = evaluate_dpo(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        records=val_records,
        beta=args.beta,
    )
    append_jsonl(
        metrics_path,
        {
            "type": "eval",
            "step": global_step,
            "final": True,
            **final_eval,
            "elapsed_seconds": time.time() - train_start,
        },
    )
    final_accuracy = float(final_eval["classification_accuracy"])
    final_loss = float(final_eval["loss"])
    if (final_accuracy > best_val_accuracy) or (
        final_accuracy == best_val_accuracy and final_loss < best_val_loss
    ):
        best_val_accuracy = final_accuracy
        best_val_loss = final_loss
        if args.save_best:
            save_model(model, tokenizer, output_dir / "best_model")
    if args.save_final:
        save_model(model, tokenizer, output_dir / "final_model")

    write_json(
        log_dir / "run_summary.json",
        {
            "model": args.model,
            "hh_path": args.hh_path,
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
            "num_train_examples": len(train_records),
            "num_val_examples": len(val_records),
            "num_epochs": args.num_epochs,
            "optimizer_steps": global_step,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.gradient_accumulation_steps,
            "beta": args.beta,
            "learning_rate": args.learning_rate,
            "optimizer": "RMSprop",
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "final_val_accuracy": final_accuracy,
            "final_val_loss": final_loss,
            "train_seconds": time.time() - train_start,
            "save_final": args.save_final,
            "save_best": args.save_best,
        },
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
