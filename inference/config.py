from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class SamplerConfig:
    kind: str = os.environ.get("SAMPLER_KIND", "stub")
    prompt_template: str | None = os.environ.get("SAMPLER_PROMPT_TEMPLATE")
    temperature: float = float(os.environ.get("SAMPLER_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.environ.get("SAMPLER_MAX_TOKENS", "512"))
    top_p: float = float(os.environ.get("SAMPLER_TOP_P", "0.95"))
    top_k: int = int(os.environ.get("SAMPLER_TOP_K", "0"))
    vllm_model: str | None = os.environ.get("VLLM_MODEL")
    vllm_tensor_parallel: int = int(os.environ.get("VLLM_TP_SIZE", "1"))
    vllm_download_dir: str | None = os.environ.get("VLLM_DOWNLOAD_DIR")
    vllm_rpc_port: int = int(os.environ.get("VLLM_RPC_PORT", "8000"))
    vllm_rpc_host: str | None = os.environ.get("VLLM_RPC_HOST")


@dataclass(slots=True)
class StorageConfig:
    kind: str = os.environ.get("STORAGE_KIND", "jsonl")
    jsonl_path: str = os.environ.get("STORAGE_JSONL_PATH", "./rollouts.jsonl")
    s3_bucket: str | None = os.environ.get("STORAGE_S3_BUCKET")
    s3_prefix: str = os.environ.get("STORAGE_S3_PREFIX", "rollouts/")
    clickhouse_url: str | None = os.environ.get("CLICKHOUSE_URL")
    clickhouse_table: str = os.environ.get("CLICKHOUSE_TABLE", "rollouts")


__all__ = ["SamplerConfig", "StorageConfig"]
