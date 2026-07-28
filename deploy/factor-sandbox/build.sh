#!/bin/sh
set -eu

base_image="${FACTOR_SANDBOX_BASE_IMAGE:?FACTOR_SANDBOX_BASE_IMAGE is required}"
sandbox_image="${FACTOR_SANDBOX_IMAGE:?FACTOR_SANDBOX_IMAGE is required}"
sandbox_host="${FACTOR_SANDBOX_DOCKER_HOST:?FACTOR_SANDBOX_DOCKER_HOST is required}"

docker image inspect "$base_image" >/dev/null
docker save "$base_image" | docker --host "$sandbox_host" load >/dev/null
docker --host "$sandbox_host" build \
  --build-arg "FACTOR_SANDBOX_BASE_IMAGE=$base_image" \
  --tag "$sandbox_image" \
  /context
