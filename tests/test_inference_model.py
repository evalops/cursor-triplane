import asyncio
import json

from inference.config import SamplerConfig
from inference.model import create_sampler, ensure_json_plan


def test_stub_sampler_emits_plan():
    sampler = create_sampler(SamplerConfig(kind="stub"))
    plan_json = asyncio.run(sampler.sample("Investigate flake"))
    plan = json.loads(plan_json)
    assert "parallel" in plan or "then" in plan


def test_ensure_json_plan_handles_invalid_json():
    repaired = asyncio.run(ensure_json_plan("not-json"))
    plan = json.loads(repaired)
    assert "then" in plan
