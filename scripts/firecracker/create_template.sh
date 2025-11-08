#!/usr/bin/env bash
set -euo pipefail

ignite run env-base \
  --name env-template-py \
  --cpus 2 --memory 4GB --size 10GB \
  --ssh --snapshot
