# Off-Policy GRPO Sweep Commands

## 作业原文要求拆解

- `grpo_off_policy` 要交付的是训练 infra：每个 rollout batch 生成后，先用当前 policy 在 `torch.inference_mode()` 下缓存 response token 的 `old_log_probs`，然后对同一批 rollout 做多次 optimizer updates。
- 更新次数由 `rollout_batch_size`, `epochs_per_rollout_batch`, `train_batch_size` 决定：每个 rollout batch 的 optimizer updates 数量是 `epochs_per_rollout_batch * rollout_batch_size / train_batch_size`。
- off-policy 设置需要使用 `"GRPO-Clip"` loss type，也就是脚本里的 `--loss-type grpo_clip`。
- `grpo_off_policy_sweep` 固定真实 sweep 的 `rollout_batch_size=256`，扫描 `epochs_per_rollout_batch` 和 `train_batch_size`。先做 `<50` GRPO steps 的 broad sweep，再做 200 steps focused sweep。
- 对比对象是 on-policy：`epochs_per_rollout_batch=1`, `train_batch_size=256`。当前主线应继承前面消融中更好的设置：`learning_rate=4e-5`, `loss_normalization=masked_mean`, `use_std_normalization=False`。
- 作业提示要注意显存：改变 `train_batch_size` 时同步改变 `gradient_accumulation_steps`，这里保持 microbatch size 为 2。

## Smoke

这个 smoke 只检查 off-policy infra，不属于正式 sweep，因为它把 rollout batch 缩小到了 32。

直接运行训练脚本：

```bash
uv run python scripts/grpo_experiment.py \
  --model /root/autodl-tmp/a5-alignment/models/Qwen2.5-Math-1.5B \
  --train-path /root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric/train.jsonl \
  --val-path /root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric/validation.jsonl \
  --prompt-template cs336_alignment/prompts/r1_zero.prompt \
  --output-dir /root/autodl-tmp/a5-alignment/runs/grpo_off_policy_sweep/smoke_e2_tb16_rb32 \
  --log-dir .agents/logs/ch7/grpo_off_policy_sweep/smoke_e2_tb16_rb32 \
  --n-grpo-steps 2 \
  --learning-rate 4e-5 \
  --rollout-batch-size 32 \
  --group-size 8 \
  --epochs-per-rollout-batch 2 \
  --train-batch-size 16 \
  --gradient-accumulation-steps 8 \
  --loss-type grpo_clip \
  --cliprange 0.2 \
  --loss-normalization masked_mean \
  --loss-normalize-constant 1.0 \
  --no-use-std-normalization \
  --eval-every 1 \
  --eval-max-examples 64 \
  --max-train-questions 64 \
  --max-new-tokens 1024 \
  --seed 0 \
  --policy-device cuda:0 \
  --vllm-device cuda:1 \
  --vllm-gpu-memory-utilization 0.85
```

或者通过 runner 运行并自动做落盘检查：

```bash
uv run python scripts/run_grpo_off_policy_sweep.py \
  --grid .agents/logs/ch7/grpo_off_policy_sweep/grid.json \
  --phase smoke
```

如果只想看实际展开出来的命令：

```bash
uv run python scripts/run_grpo_off_policy_sweep.py \
  --grid .agents/logs/ch7/grpo_off_policy_sweep/grid.json \
  --phase smoke \
  --dry-run
```

## Broad Sweep

Broad sweep 使用 `n_grpo_steps=40`，满足题目 `<50` 的要求。每个 run 都固定 `rollout_batch_size=256`。

```bash
uv run python scripts/run_grpo_off_policy_sweep.py \
  --grid .agents/logs/ch7/grpo_off_policy_sweep/grid.json \
  --phase broad
```

## Focused Sweep

Focused sweep 预先放了三组 200-step 候选。严格来说，最好先看 broad sweep，再删改 `grid.json` 里的 focused runs；如果要无人值守，就直接用当前预注册 focused grid。

```bash
uv run python scripts/run_grpo_off_policy_sweep.py \
  --grid .agents/logs/ch7/grpo_off_policy_sweep/grid.json \
  --phase focused
```

## Check Only

检查某个 phase 是否已经完整落盘：

```bash
uv run python scripts/run_grpo_off_policy_sweep.py \
  --grid .agents/logs/ch7/grpo_off_policy_sweep/grid.json \
  --phase broad \
  --check-only
```

`check-only` 会确认：

- `config.json`
- `metrics.jsonl`
- `run_summary.json`
- output directory
- `final_policy` directory, but only for runs with `save_checkpoints=true`
- `final_optimizer_step` 是否等于 `n_grpo_steps * epochs_per_rollout_batch * rollout_batch_size / train_batch_size`
- `metrics.jsonl` 中 `type=train` 的行数是否等于 expected optimizer steps

## Unattended Run, Verify, Shutdown

先 dry run 检查控制流，不会关机：

```bash
DRY_RUN=1 bash .agents/logs/ch7/grpo_off_policy_sweep/run_all_and_shutdown.sh
```

真正无人值守跑 broad + focused，全部完成并通过落盘检查后 `sync` 并关机：

```bash
bash .agents/logs/ch7/grpo_off_policy_sweep/run_all_and_shutdown.sh
```

如果机器需要 sudo 才能关机：

```bash
USE_SUDO_SHUTDOWN=1 bash .agents/logs/ch7/grpo_off_policy_sweep/run_all_and_shutdown.sh
```

只跑 broad 并关机：

```bash
PHASE=broad bash .agents/logs/ch7/grpo_off_policy_sweep/run_all_and_shutdown.sh
```

只跑 focused 并关机：

```bash
PHASE=focused bash .agents/logs/ch7/grpo_off_policy_sweep/run_all_and_shutdown.sh
```
