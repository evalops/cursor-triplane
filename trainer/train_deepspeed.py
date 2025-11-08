from __future__ import annotations

import argparse
from pathlib import Path

import deepspeed
import torch

from .config import DeepSpeedMoEConfig, TrainConfig
from .data import batch_records, load_rollout_file, to_tensors
from .moe_deepspeed import PPOPolicyValue, build_engine_config


def iter_batches(path: Path, batch_size: int):
    records = load_rollout_file(path)
    for batch in batch_records(records, batch_size):
        yield to_tensors(batch)


def ppo_loss(cfg: TrainConfig, outputs, batch) -> torch.Tensor:
    logits, values = outputs
    log_probs = torch.log_softmax(logits, dim=-1)

    actions = batch["actions"]
    logprobs_old = batch["logprobs"]
    returns = batch["returns"]
    advantages = batch["advantages"]

    selected = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    logprobs = selected.mean(dim=-1)

    ratio = torch.exp(logprobs - logprobs_old)
    clipped = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)
    policy_loss = -(torch.min(ratio * advantages, clipped * advantages)).mean()

    value_loss = torch.nn.functional.mse_loss(values, returns)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

    loss = policy_loss + cfg.value_coef * value_loss - cfg.ent_coef * entropy
    return loss


def train(args) -> None:
    train_cfg = TrainConfig(
        lr=args.lr,
        clip_range=args.clip_range,
        value_coef=args.value_coef,
        ent_coef=args.ent_coef,
        micro_batch_size=args.micro_batch_size,
        total_steps=args.total_steps,
    )

    moe_cfg = DeepSpeedMoEConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.heads,
        num_layers=args.layers,
        num_experts=args.experts,
        expert_capacity=args.capacity_factor,
        sequence_length=args.sequence_length,
        vocab_size=args.vocab_size,
        tensor_parallel=args.tensor_parallel,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        dropout=args.dropout,
        fp8=args.fp8,
        dtype=args.dtype,
    )

    model = PPOPolicyValue(moe_cfg)
    engine_cfg = build_engine_config(
        moe_cfg,
        {"lr": train_cfg.lr, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": train_cfg.weight_decay},
    )

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=engine_cfg,
    )

    device = engine.device
    rollout_path = Path(args.rollouts)
    batches = iter_batches(rollout_path, train_cfg.micro_batch_size)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step, batch in enumerate(batches, start=1):
        device_batch = {k: v.to(device) for k, v in batch.items()}
        outputs = engine(device_batch["obs"])
        loss = ppo_loss(train_cfg, outputs, device_batch)
        engine.backward(loss)
        engine.step()

        if step % args.checkpoint_interval == 0:
            tag = f"step-{step:06d}"
            engine.save_checkpoint(str(checkpoint_dir), tag=tag)
            if engine.global_rank == 0:
                print(f"[ds-train] step={step} loss={loss.item():.4f} checkpoint={tag}")

        if step >= train_cfg.total_steps:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSpeed MoE PPO Trainer")
    parser.add_argument("--rollouts", required=True, help="Path to rollout jsonl")
    parser.add_argument("--checkpoint-dir", default="./checkpoints-ds", help="Checkpoint output directory")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--total-steps", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
