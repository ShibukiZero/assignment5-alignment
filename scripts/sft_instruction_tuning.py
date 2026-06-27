#!/usr/bin/env python3
"""Instruction-tune Llama 3.1 8B on packed SFT data."""

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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    get_cosine_schedule_with_warmup,
)

from cs336_alignment.experiment_logging import append_jsonl, write_json
from cs336_alignment.experiment_metrics import cuda_memory_metrics
from cs336_alignment.sft_data import PackedSFTDataset, iterate_batches


DEFAULT_MODEL = "models/Llama-3.1-8B"
DEFAULT_TRAIN_PATH = "data/datasets/ch3/train.jsonl.gz"
DEFAULT_VAL_PATH = "data/datasets/ch3/test.jsonl.gz"
DEFAULT_OUTPUT_DIR = "runs/supplement/ch3/sft"
DEFAULT_LOG_DIR = "runs/logs/ch3/sft"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Optional optimizer-step cap for smoke tests.",
    )
    parser.add_argument(
        "--max-train-records",
        type=int,
        default=None,
        help="Optional cap on raw train records before packing.",
    )
    parser.add_argument(
        "--max-val-records",
        type=int,
        default=None,
        help="Optional cap on raw validation records before packing.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional cap on validation batches for smoke tests.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
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
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save final model and tokenizer under output-dir/final_model.",
    )
    parser.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the lowest-validation-loss model under output-dir/best_model.",
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
    if args.seq_length <= 0:
        raise ValueError("--seq-length must be positive.")
    if args.microbatch_size <= 0:
        raise ValueError("--microbatch-size must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")
    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be positive.")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("--max-train-steps must be positive when set.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup-ratio must be between 0 and 1.")


def compute_lm_loss(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids=input_ids).logits
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
    )


def evaluate_loss(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: str,
    max_batches: int | None,
) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss = compute_lm_loss(model, input_ids=input_ids, labels=labels)
            num_tokens = int(labels.numel())
            total_loss += float(loss.detach().cpu().item()) * num_tokens
            total_tokens += num_tokens
            total_batches += 1
            del input_ids, labels, loss
    if total_tokens == 0:
        raise ValueError("Validation dataloader produced no tokens.")
    model.train()
    avg_loss = total_loss / total_tokens
    return {
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss) if avg_loss < 100 else float("inf"),
        "num_batches": total_batches,
        "num_tokens": total_tokens,
        "eval_seconds": time.time() - start_time,
    }


def optimizer_step(
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    max_grad_norm: float,
) -> float:
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm.detach().cpu().item())


def save_model(model: PreTrainedModel, tokenizer: Any, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


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
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(args.device)
    model.train()

    train_dataset = PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path=args.train_path,
        seq_length=args.seq_length,
        shuffle=True,
        max_records=args.max_train_records,
    )
    val_dataset = PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path=args.val_path,
        seq_length=args.seq_length,
        shuffle=False,
        max_records=args.max_val_records,
    )
    train_loader = iterate_batches(
        train_dataset,
        batch_size=args.microbatch_size,
        shuffle=True,
    )
    val_loader = iterate_batches(
        val_dataset,
        batch_size=args.microbatch_size,
        shuffle=False,
    )
    if len(train_dataset) == 0:
        raise ValueError("Train dataset produced zero packed sequences.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset produced zero packed sequences.")

    microbatches_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(
        microbatches_per_epoch / args.gradient_accumulation_steps
    )
    planned_optimizer_steps = optimizer_steps_per_epoch * args.num_epochs
    if args.max_train_steps is not None:
        planned_optimizer_steps = min(planned_optimizer_steps, args.max_train_steps)
    warmup_steps = int(round(planned_optimizer_steps * args.warmup_ratio))

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_optimizer_steps,
    )
    optimizer.zero_grad(set_to_none=True)

    logger.info("train packed sequences: %d", len(train_dataset))
    logger.info("validation packed sequences: %d", len(val_dataset))
    logger.info("microbatches per epoch: %d", microbatches_per_epoch)
    logger.info("planned optimizer steps: %d", planned_optimizer_steps)
    logger.info("warmup steps: %d", warmup_steps)

    best_val_loss = float("inf")
    train_start = time.time()
    global_step = 0

    for epoch in range(args.num_epochs):
        if global_step >= planned_optimizer_steps:
            break
        train_iter = iter(train_loader)
        microbatch_index = 0
        progress = tqdm(
            total=optimizer_steps_per_epoch,
            desc=f"SFT epoch {epoch + 1}",
        )
        while microbatch_index < microbatches_per_epoch:
            if global_step >= planned_optimizer_steps:
                break
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(torch.device(args.device))

            remaining = microbatches_per_epoch - microbatch_index
            accumulation_target = min(args.gradient_accumulation_steps, remaining)
            micro_losses: list[float] = []
            step_tokens = 0
            step_start = time.time()

            for _ in range(accumulation_target):
                batch = next(train_iter)
                input_ids = batch["input_ids"].to(args.device)
                labels = batch["labels"].to(args.device)
                raw_loss = compute_lm_loss(model, input_ids=input_ids, labels=labels)
                loss = raw_loss / accumulation_target
                loss.backward()
                micro_losses.append(float(raw_loss.detach().cpu().item()))
                step_tokens += int(labels.numel())
                microbatch_index += 1
                del input_ids, labels, raw_loss, loss

            grad_norm = optimizer_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                max_grad_norm=args.max_grad_norm,
            )
            global_step += 1
            learning_rate = float(scheduler.get_last_lr()[0])
            train_record: dict[str, Any] = {
                "type": "train",
                "epoch": epoch + 1,
                "step": global_step,
                "loss": mean(micro_losses),
                "learning_rate": learning_rate,
                "grad_norm": grad_norm,
                "num_microbatches": accumulation_target,
                "num_tokens": step_tokens,
                "step_seconds": time.time() - step_start,
                "elapsed_seconds": time.time() - train_start,
            }
            if args.device.startswith("cuda"):
                train_record.update(cuda_memory_metrics(args.device))
            append_jsonl(metrics_path, train_record)

            if global_step % args.log_every == 0:
                logger.info(
                    "step=%d epoch=%d loss=%.4f lr=%.3e grad_norm=%.4f",
                    global_step,
                    epoch + 1,
                    train_record["loss"],
                    learning_rate,
                    grad_norm,
                )

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                val_metrics = evaluate_loss(
                    model=model,
                    dataloader=val_loader,
                    device=args.device,
                    max_batches=args.max_val_batches,
                )
                eval_record = {
                    "type": "eval",
                    "epoch": epoch + 1,
                    "step": global_step,
                    **val_metrics,
                    "elapsed_seconds": time.time() - train_start,
                }
                append_jsonl(metrics_path, eval_record)
                logger.info(
                    "eval step=%d loss=%.4f ppl=%.4f",
                    global_step,
                    val_metrics["loss"],
                    val_metrics["perplexity"],
                )
                if float(val_metrics["loss"]) < best_val_loss:
                    best_val_loss = float(val_metrics["loss"])
                    if args.save_best:
                        save_model(model, tokenizer, output_dir / "best_model")

            if args.save_every is not None and global_step % args.save_every == 0:
                save_model(model, tokenizer, output_dir / f"checkpoint_step_{global_step:06d}")

            progress.update(1)
        progress.close()

    final_val_metrics = evaluate_loss(
        model=model,
        dataloader=val_loader,
        device=args.device,
        max_batches=args.max_val_batches,
    )
    append_jsonl(
        metrics_path,
        {
            "type": "eval",
            "step": global_step,
            "final": True,
            **final_val_metrics,
            "elapsed_seconds": time.time() - train_start,
        },
    )
    if float(final_val_metrics["loss"]) < best_val_loss:
        best_val_loss = float(final_val_metrics["loss"])
        if args.save_best:
            save_model(model, tokenizer, output_dir / "best_model")

    if args.save_final:
        save_model(model, tokenizer, output_dir / "final_model")

    write_json(
        log_dir / "run_summary.json",
        {
            "model": args.model,
            "train_path": args.train_path,
            "val_path": args.val_path,
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
            "seq_length": args.seq_length,
            "microbatch_size": args.microbatch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.microbatch_size
            * args.gradient_accumulation_steps,
            "num_epochs": args.num_epochs,
            "optimizer_steps": global_step,
            "planned_optimizer_steps": planned_optimizer_steps,
            "learning_rate": args.learning_rate,
            "warmup_steps": warmup_steps,
            "best_val_loss": best_val_loss,
            "final_val_loss": final_val_metrics["loss"],
            "final_val_perplexity": final_val_metrics["perplexity"],
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
