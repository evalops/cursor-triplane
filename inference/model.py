from __future__ import annotations

import json
from typing import Any, Protocol

import httpx

from .config import SamplerConfig


class SamplerBackend(Protocol):
    async def sample(self, prompt: str) -> str:  # noqa: D401
        ...


class StubSampler:
    def __init__(self, template: str | None = None):
        self._template = template

    async def sample(self, prompt: str) -> str:
        del prompt
        plan = {
            "parallel": [
                {"tool": "SearchCode", "args": {"query": "failing test", "k": 5}},
            ],
            "then": [
                {"tool": "RunLint", "args": {}},
                {"tool": "Exec", "args": {"cmd": "pytest -q", "timeout_s": 120}},
            ],
        }
        return json.dumps(plan)


class OpenAIVLLMSampler:
    def __init__(self, cfg: SamplerConfig):
        if not cfg.vllm_rpc_host:
            raise ValueError("VLLM_RPC_HOST must be set for vLLM sampler")
        self._endpoint = f"http://{cfg.vllm_rpc_host}:{cfg.vllm_rpc_port}/v1/completions"
        self._model = cfg.vllm_model or "EvalOps/cursor-moe"
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._top_p = cfg.top_p
        self._top_k = cfg.top_k if cfg.top_k > 0 else None
        self._prompt_template = cfg.prompt_template or "{prompt}\nPlan:"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    async def sample(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "prompt": self._prompt_template.format(prompt=prompt),
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": self._top_p,
        }
        if self._top_k is not None:
            payload["top_k"] = self._top_k
        resp = await self._client.post(self._endpoint, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["text"]
        return normalize_plan_text(text)

    async def close(self) -> None:
        await self._client.aclose()


def normalize_plan_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.split("Plan:")[-1].strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    return json.dumps({"then": []})


async def ensure_json_plan(sample: str) -> str:
    try:
        json.loads(sample)
        return sample
    except json.JSONDecodeError:
        fallback = {"then": [{"tool": "RunLint", "args": {}}]}
        return json.dumps(fallback)


def create_sampler(cfg: SamplerConfig) -> SamplerBackend:
    if cfg.kind == "vllm-openai":
        return OpenAIVLLMSampler(cfg)
    return StubSampler(cfg.prompt_template)


__all__ = ["create_sampler", "ensure_json_plan"]
