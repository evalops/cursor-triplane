from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from deepspeed.moe.layer import MoE

    HAS_DEEPSPEED = True
except ImportError:  # pragma: no cover - optional dependency
    MoE = None  # type: ignore[assignment]
    HAS_DEEPSPEED = False

try:
    from transformer_engine.pytorch import fp8_autocast as _fp8_autocast

    HAS_TRANSFORMER_ENGINE = True
except ImportError:  # pragma: no cover - optional dependency
    _fp8_autocast = None
    HAS_TRANSFORMER_ENGINE = False

from .config import DeepSpeedMoEConfig


class ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden)
        self.fc2 = nn.Linear(ffn_hidden, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def _fp8_context(enabled: bool):
    if enabled and HAS_TRANSFORMER_ENGINE and _fp8_autocast is not None:
        return _fp8_autocast(enabled=enabled)
    return nullcontext()


def build_moe(hidden_size: int, ffn_hidden: int, num_experts: int, capacity_factor: float) -> MoE:
    if not HAS_DEEPSPEED:
        raise RuntimeError("DeepSpeed MoE is not available; install deepspeed to enable this module")
    return MoE(
        hidden_size=hidden_size,
        expert=ExpertMLP,
        num_experts=num_experts,
        ep_size=1,
        min_capacity=4,
        k=2,
        capacity_factor=capacity_factor,
        hidden_size_per_expert=ffn_hidden,
    )


class TransformerMoELayer(nn.Module):
    def __init__(self, cfg: DeepSpeedMoEConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.hidden_size, cfg.num_attention_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        ffn_hidden = cfg.hidden_size * 4
        self.moe = build_moe(cfg.hidden_size, ffn_hidden, cfg.num_experts, cfg.expert_capacity)
        self.dropout = nn.Dropout(cfg.dropout if hasattr(cfg, "dropout") else 0.1)
        self.fp8 = cfg.fp8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _fp8_context(self.fp8):
            attn_out, _ = self.attn(x, x, x, need_weights=False)
            x = self.ln1(x + self.dropout(attn_out))
            bsz, seqlen, hidden = x.shape
            moe_in = x.view(bsz * seqlen, hidden)
            moe_out, _ = self.moe(moe_in)
            moe_out = moe_out.view(bsz, seqlen, hidden)
            x = self.ln2(x + self.dropout(moe_out))
        return x


class DeepSpeedMoETransformer(nn.Module):
    def __init__(self, cfg: DeepSpeedMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.sequence_length, cfg.hidden_size)
        self.layers = nn.ModuleList(TransformerMoELayer(cfg) for _ in range(cfg.num_layers))
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size)

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        encoded = self._encode(input_ids)
        logits = self.head(encoded)
        return logits

    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        encoded = self._encode(input_ids)
        return encoded.mean(dim=1)


class PPOPolicyValue(nn.Module):
    def __init__(self, cfg: DeepSpeedMoEConfig):
        super().__init__()
        self.policy = DeepSpeedMoETransformer(cfg)
        self.value_head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy(input_ids)
        hidden = self.policy.hidden(input_ids)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values

    def logprob(self, input_ids: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(input_ids)
        log_probs = torch.log_softmax(logits, dim=-1)
        selected = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        return selected.mean(dim=-1)


def build_engine_config(cfg: DeepSpeedMoEConfig, optimizer: Dict[str, float]) -> Dict:
    base = cfg.to_deepspeed_dict()
    base["optimizer"] = {"type": "AdamW", "params": optimizer}
    return base


__all__ = ["DeepSpeedMoETransformer", "PPOPolicyValue", "build_engine_config", "HAS_DEEPSPEED"]
