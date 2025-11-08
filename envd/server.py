import asyncio
import json
import os
import subprocess
import time
from concurrent import futures
from pathlib import Path
from typing import List

import grpc

from .generated import tools_pb2, tools_pb2_grpc

WORKDIR = Path(os.environ.get("WORKDIR", "/work")).resolve()
SEMANTIC_URI = os.environ.get("SEMANTIC_ENDPOINT")


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.read()


def _git_diff(path: Path) -> str:
    proc = subprocess.run(
        ["git", "diff", "--", str(path.relative_to(WORKDIR))],
        cwd=WORKDIR,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout


def _run(cmd: List[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=WORKDIR,
        capture_output=True,
        input=input_bytes,
        check=False,
    )


class ToolsServicer(tools_pb2_grpc.ToolsServicer):
    def ReadFile(self, request, context):
        path = (WORKDIR / request.path).resolve()
        if not path.is_file() or WORKDIR not in path.parents and path != WORKDIR:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("file not found")
            return tools_pb2.ReadFileResp()
        return tools_pb2.ReadFileResp(content=_read_text(path))

    def EditFile(self, request, context):
        path = (WORKDIR / request.path).resolve()
        if WORKDIR not in path.parents and path != WORKDIR:
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details("path escapes workdir")
            return tools_pb2.EditFileResp()

        if request.dry_run:
            proc = _run(["git", "apply", "--check", "-p0", "-"] , input_bytes=request.patch_unified.encode())
            applied = proc.returncode == 0
            return tools_pb2.EditFileResp(applied=applied, diff=proc.stderr)

        proc = _run(["git", "apply", "-p0", "-"], input_bytes=request.patch_unified.encode())
        applied = proc.returncode == 0
        diff = _git_diff(path)
        return tools_pb2.EditFileResp(applied=applied, diff=diff, stderr=proc.stderr)

    def Exec(self, request, context):
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                request.cmd,
                shell=True,
                cwd=WORKDIR,
                capture_output=True,
                text=True,
                timeout=request.timeout_s or 30,
            )
        except subprocess.TimeoutExpired as exc:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details("command timeout")
            return tools_pb2.ExecResp(
                code=-1,
                stdout=(exc.stdout or "")[:200_000],
                stderr=(exc.stderr or "timeout")[:200_000],
                duration_s=time.perf_counter() - started,
            )

        return tools_pb2.ExecResp(
            code=proc.returncode,
            stdout=(proc.stdout if request.capture_output else "")[:200_000],
            stderr=proc.stderr[:200_000],
            duration_s=time.perf_counter() - started,
        )

    def RunLint(self, request, context):
        proc = subprocess.run(
            ["ruff", "check", "--output-format", "sarif", "--exit-zero", "."],
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            check=False,
        )
        sarif_json = proc.stdout if proc.stdout else json.dumps({"runs": []})
        return tools_pb2.LintResp(sarif_json=sarif_json[:200_000], stderr=proc.stderr[:200_000])

    def SearchCode(self, request, context):
        if request.semantic and SEMANTIC_URI:
            try:
                import httpx

                resp = httpx.post(
                    f"{SEMANTIC_URI}/search",
                    json={"query": request.query, "k": request.k or 20},
                    timeout=5.0,
                )
                resp.raise_for_status()
                payload = resp.json()
                return tools_pb2.SearchResp(
                    paths=payload.get("paths", []),
                    scores=[float(s) for s in payload.get("scores", [])],
                )
            except Exception as exc:  # noqa: BLE001
                context.set_code(grpc.StatusCode.ABORTED)
                context.set_details(f"semantic search failed: {exc}")

        proc = subprocess.run(
            ["rg", "-l", request.query],
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            check=False,
        )
        paths = [line for line in proc.stdout.splitlines() if line.strip()]
        limit = request.k or 20
        return tools_pb2.SearchResp(paths=paths[:limit])


async def serve_async(port: int = 50051) -> None:
    server = grpc.aio.server(options=[("grpc.max_send_message_length", 16 * 1024 * 1024)])
    tools_pb2_grpc.add_ToolsServicer_to_server(ToolsServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    await server.wait_for_termination()


def serve(port: int = 50051) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(serve_async(port))


if __name__ == "__main__":
    serve()
