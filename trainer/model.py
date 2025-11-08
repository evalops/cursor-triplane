from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * expansion)
        self.fc2 = nn.Linear(d_model * expansion, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.fc2(self.act(self.fc1(x)))


class SimpleMoE(nn.Module):
    def __init__(self, d_model: int, num_experts: int, expansion: int = 4):
        super().__init__()
        self.experts = nn.ModuleList(ExpertFFN(d_model, expansion) for _ in range(num_experts))
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = F.softmax(self.router(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return (expert_outputs * scores.unsqueeze(-1)).sum(dim=-2)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, num_experts: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = SimpleMoE(d_model, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        moe_out = self.moe(x)
        x = self.norm2(x + self.dropout(moe_out))
        return x


class PolicyModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        num_experts: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(4096, d_model)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, n_heads, dropout, num_experts) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    @property
    def hidden_size(self) -> int:
        return self.head.in_features

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits

    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x.mean(dim=1)

    def logprob(self, input_ids: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids)
        log_probs = logits.log_softmax(dim=-1)
        selected = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        return selected.mean(dim=-1)


__all__ = ["PolicyModel"]
