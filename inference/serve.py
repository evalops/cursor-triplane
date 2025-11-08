from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Tuple

import ray
import grpc
from google.protobuf.json_format import MessageToDict

from envd.generated import tools_pb2, tools_pb2_grpc
from .config import SamplerConfig, StorageConfig
from .model import create_sampler, ensure_json_plan
from .storage import create_storage, write_rollout_records


@dataclass(slots=True)
class ToolSpec:
    request_cls: Any
    attr: str


TOOL_SPECS: Dict[str, ToolSpec] = {
    "ReadFile": ToolSpec(tools_pb2.ReadFileReq, "ReadFile"),
    "EditFile": ToolSpec(tools_pb2.EditFileReq, "EditFile"),
    "SearchCode": ToolSpec(tools_pb2.SearchReq, "SearchCode"),
    "RunLint": ToolSpec(tools_pb2.Empty, "RunLint"),
    "Exec": ToolSpec(tools_pb2.ExecReq, "Exec"),
}


def build_request(tool: str, args: Dict[str, Any]) -> Any:
    spec = TOOL_SPECS.get(tool)
    if not spec:
        raise ValueError(f"unknown tool: {tool}")
    return spec.request_cls(**args)


def stub_method(stub: tools_pb2_grpc.ToolsStub, tool: str):
    spec = TOOL_SPECS.get(tool)
    if not spec:
        raise ValueError(f"unknown tool: {tool}")
    return getattr(stub, spec.attr)


@ray.remote(num_cpus=0.05)
class EnvClient:
    def __init__(self, host: str):
        self._host = host
        self._channel = grpc.aio.insecure_channel(host)
        self._stub = tools_pb2_grpc.ToolsStub(self._channel)

    async def call(self, tool: str, args: Dict[str, Any], *, timeout: float | None = None) -> Dict[str, Any]:
        request = build_request(tool, args)
        rpc = stub_method(self._stub, tool)
        try:
            response = await rpc(request, timeout=timeout)
            payload = MessageToDict(response, preserving_proto_field_name=True, including_default_value_fields=True)
            return {"tool": tool, "ok": True, "response": payload}
        except grpc.aio.AioRpcError as exc:  # noqa: D
            return {
                "tool": tool,
                "ok": False,
                "code": exc.code().name,
                "details": exc.details(),
            }

    async def close(self) -> None:
        await self._channel.close()


@ray.remote(num_cpus=0.1)
class ModelSampler:
    def __init__(self, cfg_dict: Dict[str, Any]):
        self._cfg = SamplerConfig(**cfg_dict)
        self._backend = create_sampler(self._cfg)

    async def sample(self, prompt: str) -> str:
        plan = await self._backend.sample(prompt)
        normalized = await ensure_json_plan(plan)
        return normalized


@ray.remote(num_cpus=0.2)
class Sampler:
    def __init__(self, env_hosts: Iterable[str], model_ref):
        self._envs = [EnvClient.remote(host) for host in env_hosts]
        self._model = model_ref

    async def _call_env(self, env, tool: str, args: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        try:
            return await env.call.remote(tool, args, timeout=timeout)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            return {"tool": tool, "ok": False, "details": str(exc)}

    async def rollout(self, prompt: str, *, budget_s: float = 45.0, group_timeout_s: float = 10.0) -> Dict[str, Any]:
        started = time.perf_counter()
        plan_json = await self._model.sample.remote(prompt)
        plan = json.loads(plan_json)
        if not isinstance(plan, dict):
            plan = {}
        trajectory: List[Dict[str, Any]] = []

        for step_name in ("parallel", "then"):
            if step_name not in plan:
                continue
            tasks = []
            for call in plan[step_name]:
                env = random.choice(self._envs)
                tasks.append(
                    asyncio.create_task(
                        self._call_env(env, call["tool"], call.get("args", {}), timeout=group_timeout_s)
                    )
                )

            done, pending = await asyncio.wait(tasks, timeout=group_timeout_s)
            results = [task.result() for task in done if not task.cancelled()]
            for task in pending:
                task.cancel()
            trajectory.append({"step": step_name, "results": results})

            if time.perf_counter() - started >= budget_s:
                break

        latency = time.perf_counter() - started
        return {"trajectory": trajectory, "latency_s": latency}


@ray.remote(num_cpus=0.05)
class Controller:
    def __init__(self, sampler_refs: Iterable[ray.actor.ActorHandle], storage_cfg: Dict[str, Any]):
        self._samplers = list(sampler_refs)
        self._storage = create_storage(StorageConfig(**storage_cfg))

    async def batch_rollouts(self, prompts: List[str], *, replicas: int = 3, timeout_s: float = 60.0) -> List[Dict[str, Any]]:
        object_refs: List[ray.ObjectRef] = []
        prompt_map: Dict[ray.ObjectRef, str] = {}
        for prompt in prompts:
            for _ in range(replicas):
                sampler = random.choice(self._samplers)
                ref = sampler.rollout.remote(prompt)
                object_refs.append(ref)
                prompt_map[ref] = prompt

        deadline = time.perf_counter() + timeout_s
        pairs: List[Tuple[str, Dict[str, Any]]] = []
        pending = list(object_refs)

        while pending and time.perf_counter() < deadline:
            wait_timeout = max(0.0, deadline - time.perf_counter())
            ready, pending = ray.wait(pending, timeout=wait_timeout, num_returns=1)
            for ref in ready:
                prompt = prompt_map.get(ref, "")
                try:
                    result = await ref
                except Exception as exc:  # noqa: BLE001
                    result = {"error": str(exc)}
                pairs.append((prompt, result))
                prompt_map.pop(ref, None)

        for ref in pending:
            ray.cancel(ref, force=True)

        records = await write_rollout_records(self._storage, pairs)
        return records


def bootstrap_ray(env_hosts: List[str], *, num_samplers: int = 2):
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    sampler_cfg = SamplerConfig()
    storage_cfg = StorageConfig()
    model = ModelSampler.remote(asdict(sampler_cfg))
    sampler_refs = [Sampler.remote(env_hosts, model) for _ in range(num_samplers)]
    controller = Controller.remote(sampler_refs, asdict(storage_cfg))
    return controller


async def demo(controller, prompt: str):
    results = await controller.batch_rollouts.remote([prompt])
    return results


__all__ = [
    "EnvClient",
    "ModelSampler",
    "Sampler",
    "Controller",
    "bootstrap_ray",
    "demo",
]
