from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import torch


def load_rollout_file(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def batch_records(records: Iterable[Dict], batch_size: int) -> Iterator[List[Dict]]:
    batch: List[Dict] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _scalar(sample: Dict, key: str, fallback: Optional[str] = None) -> float:
    if key in sample:
        return float(sample[key])
    if fallback and fallback in sample:
        return float(sample[fallback])
    raise KeyError(key)


def to_tensors(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    obs = torch.tensor([sample["obs"] for sample in batch], dtype=torch.long)
    actions = torch.tensor([sample["actions"] for sample in batch], dtype=torch.long)
    logprobs = torch.tensor([_scalar(sample, "logprob", "logprobs") for sample in batch], dtype=torch.float32)
    returns = torch.tensor([_scalar(sample, "return", "returns") for sample in batch], dtype=torch.float32)
    advantages = torch.tensor([_scalar(sample, "advantage", "advantages") for sample in batch], dtype=torch.float32)
    values = torch.tensor([_scalar(sample, "value", "values") for sample in batch], dtype=torch.float32)
    return {
        "obs": obs,
        "actions": actions,
        "logprobs": logprobs,
        "returns": returns,
        "advantages": advantages,
        "values": values,
    }


__all__ = ["load_rollout_file", "batch_records", "to_tensors"]
