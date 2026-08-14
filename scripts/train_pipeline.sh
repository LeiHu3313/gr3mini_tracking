#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 teacher|adapter [teacher_checkpoint] [extra gr3mini-train args...]" >&2
  exit 2
fi

stage=$1
shift

case "$stage" in
  teacher)
    exec uv run gr3mini-train Gr3Mini-Tracking-Teacher "$@"
    ;;
  adapter)
    if [[ $# -lt 1 ]]; then
      echo "adapter stage requires a teacher checkpoint" >&2
      exit 2
    fi
    teacher_checkpoint=$1
    shift
    exec uv run gr3mini-train Gr3Mini-Tracking-Adapter \
      --agent.teacher-checkpoint "$teacher_checkpoint" "$@"
    ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac
