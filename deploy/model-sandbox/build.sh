#!/bin/sh
set -eu

base_image="${MODEL_SANDBOX_BASE_IMAGE:?MODEL_SANDBOX_BASE_IMAGE is required}"
sandbox_image="${MODEL_SANDBOX_IMAGE:?MODEL_SANDBOX_IMAGE is required}"
sandbox_host="${MODEL_SANDBOX_DOCKER_HOST:?MODEL_SANDBOX_DOCKER_HOST is required}"

docker image inspect "$base_image" >/dev/null
docker save "$base_image" | docker --host "$sandbox_host" load >/dev/null
docker --host "$sandbox_host" build \
  --build-arg "MODEL_SANDBOX_BASE_IMAGE=$base_image" \
  --tag "$sandbox_image" \
  /context
