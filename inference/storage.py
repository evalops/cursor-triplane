from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Tuple, Callable

from .config import StorageConfig


class StorageWriter:
    async def write(self, records: Iterable[Dict[str, Any]]) -> None:  # noqa: D401
        raise NotImplementedError


class JSONLWriter(StorageWriter):
    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def write(self, records: Iterable[Dict[str, Any]]) -> None:
        payload = "".join(json.dumps(record) + "\n" for record in records)
        await asyncio.to_thread(self._append, payload)

    def _append(self, data: str) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(data)


class S3Writer(StorageWriter):
    def __init__(self, bucket: str, prefix: str):
        import boto3

        self._client = boto3.client("s3")
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")

    async def write(self, records: Iterable[Dict[str, Any]]) -> None:
        key = f"{self._prefix}/{uuid.uuid4().hex}.jsonl"
        body = "".join(json.dumps(record) + "\n" for record in records)
        await asyncio.to_thread(self._client.put_object, Bucket=self._bucket, Key=key, Body=body.encode("utf-8"))


class ClickHouseWriter(StorageWriter):
    def __init__(self, url: str, table: str):
        from clickhouse_driver import Client

        self._table = table
        self._client = Client.from_url(url)

    async def write(self, records: Iterable[Dict[str, Any]]) -> None:
        rows = [(json.dumps(record),) for record in records]
        query = f"INSERT INTO {self._table} (payload) VALUES"
        await asyncio.to_thread(self._client.execute, query, rows)


class NoOpWriter(StorageWriter):
    async def write(self, records: Iterable[Dict[str, Any]]) -> None:
        return None


def build_rollout_records(entries: Iterable[Tuple[str, Dict[str, Any]]], now: Callable[[], float] | None = None) -> List[Dict[str, Any]]:
    timestamp = now or time.time
    return [
        {
            "prompt": prompt,
            "result": result,
            "timestamp_s": timestamp(),
        }
        for prompt, result in entries
    ]


async def write_rollout_records(storage: StorageWriter, entries: Iterable[Tuple[str, Dict[str, Any]]], now: Callable[[], float] | None = None) -> List[Dict[str, Any]]:
    records = build_rollout_records(entries, now=now)
    if records:
        await storage.write(records)
    return records


def create_storage(cfg: StorageConfig) -> StorageWriter:
    try:
        if cfg.kind == "s3" and cfg.s3_bucket:
            return S3Writer(cfg.s3_bucket, cfg.s3_prefix)
        if cfg.kind == "clickhouse" and cfg.clickhouse_url:
            return ClickHouseWriter(cfg.clickhouse_url, cfg.clickhouse_table)
        if cfg.kind == "jsonl":
            return JSONLWriter(cfg.jsonl_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[storage] falling back to noop writer: {exc}")
    return NoOpWriter()


__all__ = ["create_storage", "StorageWriter", "build_rollout_records", "write_rollout_records"]
