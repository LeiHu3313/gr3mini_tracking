#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${IMAGE:-docker.fftaicorp.com/gr3mini_tracking/gr3mini-tracking:py312-cuda12.8-v2}"
container_name="${CONTAINER_NAME:-gr3mini_tracking}"
gpu_devices="${GPU_DEVICES:-all}"
host_log_dir="${HOST_LOG_DIR:-${root_dir}/logs}"

mkdir -p "$host_log_dir"

docker_args=(
  run
  --interactive
  --tty
  --rm
  --name "$container_name"
  --gpus "$gpu_devices"
  --net=host
  --ipc=host
  --workdir /workspace/gr3mini_tracking
  --env NVIDIA_DRIVER_CAPABILITIES=all
  --volume "${root_dir}:/workspace/gr3mini_tracking"
  --volume "${host_log_dir}:/workspace/gr3mini_tracking/logs"
)

# The image keeps its virtual environment under /opt, outside this bind mount.
# Therefore source, rewards, motions, and scripts always use the host checkout.

# X11 is optional: training is headless, while replay/play can use this mount.
if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
  xhost +local:root >/dev/null
  trap 'xhost -local:root >/dev/null || true' EXIT
  docker_args+=(
    --env "DISPLAY=${DISPLAY}"
    --volume /tmp/.X11-unix:/tmp/.X11-unix
  )

  xauthority_file="${XAUTHORITY:-${HOME}/.Xauthority}"
  if [[ -r "$xauthority_file" ]]; then
    docker_args+=(--volume "${xauthority_file}:/root/.Xauthority:ro")
  fi
fi

# Hardware device access is unnecessary for simulation/training. Enable it
# only for robot-side tools: HARDWARE_MODE=1 ./scripts/docker/run_gr3mini_tracking.sh
if [[ "${HARDWARE_MODE:-0}" == "1" ]]; then
  docker_args+=(
    --privileged
    --cap-add=SYS_NICE
    --volume /dev:/dev
    --volume /run/ethercatd:/run/ethercatd
  )
fi

if [[ -n "${CPUSET_CPUS:-}" ]]; then
  docker_args+=(--cpuset-cpus "$CPUSET_CPUS")
fi

if [[ $# -eq 0 ]]; then
  set -- bash
fi

exec docker "${docker_args[@]}" "$image" "$@"
