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


@dataclass(slots=True)
class DeepSpeedMoEConfig:
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_layers: int = 16
    num_experts: int = 8
    expert_capacity: float = 1.25
    sequence_length: int = 1024
    vocab_size: int = 8192
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    micro_batch_size: int = 4
    global_batch_size: int = 32
    dropout: float = 0.1
    fp8: bool = True
    dtype: str = "bf16"

    def to_deepspeed_dict(self) -> dict:
        return {
            "train_batch_size": self.global_batch_size,
            "train_micro_batch_size_per_gpu": self.micro_batch_size,
            "steps_per_print": 10,
            "bf16": {"enabled": self.dtype == "bf16"},
            "fp16": {"enabled": self.dtype == "fp16"},
            "zero_optimization": {"stage": 1},
            "wall_clock_breakdown": False,
            "tensor_parallel": {
                "enabled": self.tensor_parallel > 1,
                "tp_size": self.tensor_parallel,
            },
            "moe": {
                "enabled": True,
                "expert_parallelism": self.num_experts,
                "moe_tp_size": self.tensor_parallel,
                "moe_experts": self.num_experts,
                "capacity_factor": self.expert_capacity,
            },
        }


__all__ = ["TrainConfig", "DeepSpeedMoEConfig"]
