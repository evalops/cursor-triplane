#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python3 -m trainer.train --rollouts "$ROOT/experiments/sample_rollouts.jsonl" --checkpoint-dir "$ROOT/checkpoints"
