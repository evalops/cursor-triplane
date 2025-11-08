from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from torch.optim import AdamW

from .config import TrainConfig
from .data import batch_records, load_rollout_file, to_tensors
from .model import PolicyModel


def prepare_batch(tensors: dict, device: str) -> dict:
    return {key: value.to(device) for key, value in tensors.items()}


def ppo_step(cfg: TrainConfig, model: PolicyModel, value_head: torch.nn.Module, optimizer, batch: dict) -> float:
    obs = batch["obs"]
    actions = batch["actions"]
    logprobs_old = batch["logprobs"]
    returns = batch["returns"]
    advantages = batch["advantages"]

    logprobs = model.logprob(obs, actions)
    ratio = torch.exp(logprobs - logprobs_old)
    clipped = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)
    policy_loss = -(torch.min(ratio * advantages, clipped * advantages)).mean()

    hidden = model.hidden(obs)
    values = value_head(hidden).squeeze(-1)
    value_loss = torch.nn.functional.mse_loss(values, returns)

    entropy = -(torch.exp(logprobs) * logprobs).mean()
    loss = policy_loss + cfg.value_coef * value_loss - cfg.ent_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def train_loop(cfg: TrainConfig, checkpoint_dir: Path, rollouts: Iterable[dict]) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = PolicyModel(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        num_experts=cfg.moe_num_experts,
        dropout=cfg.dropout,
    ).to(device)
    value_head = torch.nn.Linear(cfg.d_model, 1).to(device)
    optimizer = AdamW(
        list(model.parameters()) + list(value_head.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    model.train()
    value_head.train()

    for step, batch in enumerate(rollouts, start=1):
        tensors = prepare_batch(batch, device)
        loss = ppo_step(cfg, model, value_head, optimizer, tensors)
        if step % 50 == 0:
            ckpt_path = checkpoint_dir / f"step-{step:06d}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "value_head": value_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }, ckpt_path)
            print(f"[train] step={step} loss={loss:.4f} checkpoint={ckpt_path}")
        if step >= cfg.total_steps:
            break


def iter_batches(dataset_path: Path, batch_size: int) -> Iterable[dict]:
    records = load_rollout_file(dataset_path)
    for batch in batch_records(records, batch_size):
        yield to_tensors(batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO MoE policy")
    parser.add_argument("--rollouts", type=Path, required=True, help="Path to rollout jsonl file")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--total-steps", type=int, default=TrainConfig.total_steps, help="Total PPO steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device override (cuda/cpu)")
    args = parser.parse_args()

    cfg = TrainConfig(total_steps=args.total_steps, device=args.device)
    batches = iter_batches(args.rollouts, cfg.micro_batch_size)
    train_loop(cfg, args.checkpoint_dir, batches)


if __name__ == "__main__":
    main()
