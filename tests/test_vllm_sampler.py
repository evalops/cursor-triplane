import asyncio
import json

import httpx
import pytest

from inference.config import SamplerConfig
from inference.model import create_sampler


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_vllm_sampler_formats_plan(monkeypatch):
    cfg = SamplerConfig(kind="vllm-openai", vllm_rpc_host="localhost")
    sampler = create_sampler(cfg)

    async def fake_post(self, url, json):
        assert "Fix bug" in json["prompt"]
        return DummyResponse({"choices": [{"text": "{\"then\": []}"}]})

    async def fake_close(self):
        return None

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post, raising=False)
    monkeypatch.setattr(httpx.AsyncClient, "aclose", fake_close, raising=False)

    plan = await sampler.sample("Fix bug")
    assert json.loads(plan)["then"] == []

    await sampler.close()
