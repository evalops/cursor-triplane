#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-env-base:latest}
DOCKERFILE=${DOCKERFILE:-Dockerfile.env}

docker build -f "${DOCKERFILE}" -t "${IMAGE_NAME}" .
ignite image import "${IMAGE_NAME}" --name env-base --force
