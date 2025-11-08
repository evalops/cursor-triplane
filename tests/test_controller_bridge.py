import asyncio

from inference.storage import build_rollout_records, write_rollout_records


class InMemoryStorage:
    def __init__(self):
        self.records = None

    async def write(self, records):
        self.records = list(records)


def test_build_rollout_records_structure():
    records = build_rollout_records([("prompt-1", {"ok": True})], now=lambda: 123.0)
    assert records[0]["prompt"] == "prompt-1"
    assert records[0]["result"] == {"ok": True}
    assert records[0]["timestamp_s"] == 123.0


def test_write_rollout_records_persists():
    storage = InMemoryStorage()
    records = asyncio.run(write_rollout_records(storage, [("prompt-2", {"latency": 1.0})], now=lambda: 456.0))
    assert storage.records == records
    assert records[0]["timestamp_s"] == 456.0
