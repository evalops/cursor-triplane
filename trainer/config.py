from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainConfig:
    lr: float = 1e-5
    clip_range: float = 0.2
    value_coef: float = 0.5
    ent_coef: float = 0.01
    weight_decay: float = 0.01
    seq_len: int = 512
    vocab_size: int = 8192
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    moe_num_experts: int = 8
    dropout: float = 0.1
    fp8: bool = True
    device: str = "cuda"
    micro_batch_size: int = 4
    total_steps: int = 1000


__all__ = ["TrainConfig"]
