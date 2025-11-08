#!/usr/bin/env bash
set -euo pipefail

COUNT=${COUNT:-10}
TEMPLATE=${TEMPLATE:-env-template-py}
SEED_REPO=${SEED_REPO:-./seed_repo}
START_INDEX=${START_INDEX:-1}

for i in $(seq "${START_INDEX}" "$((START_INDEX + COUNT - 1))"); do
  ignite run "${TEMPLATE}" \
    --name "env-py-${i}" \
    --snapshot \
    --copy-files "${SEED_REPO}:/work" \
    --port 5005${i}:50051/tcp \
    --cmd "/usr/local/bin/envd --serve --workdir /work" \
    --cpus 2 --memory 4GB --size 10GB
done
