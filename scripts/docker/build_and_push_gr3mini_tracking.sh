#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

registry="${REGISTRY:-docker.fftaicorp.com}"
project="${PROJECT:-gr3mini_tracking}"
image_name="${IMAGE_NAME:-gr3mini-tracking}"
tag="${TAG:-py312-cuda12.8-v2}"
base_image="${BASE_IMAGE:-nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04}"

image="${registry}/${project}/${image_name}:${tag}"
latest_image="${registry}/${project}/${image_name}:latest"

echo "Building ${image}"
docker build \
  --build-arg "BASE_IMAGE=${base_image}" \
  --file Dockerfile \
  --tag "${image}" \
  --tag "${latest_image}" \
  .

echo "Pushing ${image}"
docker push "${image}"

echo "Pushing ${latest_image}"
docker push "${latest_image}"

echo
echo "Done:"
echo "  ${image}"
echo "  ${latest_image}"
