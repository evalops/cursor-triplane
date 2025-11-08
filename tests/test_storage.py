import asyncio
import json

from inference.config import StorageConfig
from inference.storage import JSONLWriter, create_storage


def test_jsonl_writer_persists_records(tmp_path):
    target = tmp_path / "rollouts.jsonl"
    writer = JSONLWriter(str(target))
    asyncio.run(writer.write([{"prompt": "fix bug", "latency": 1.2}]))
    content = target.read_text().strip().splitlines()
    assert json.loads(content[0])["prompt"] == "fix bug"


def test_create_storage_noop_when_missing_bucket():
    writer = create_storage(StorageConfig(kind="s3", s3_bucket=None))
    assert writer.__class__.__name__ == "NoOpWriter"
