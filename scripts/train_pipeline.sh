#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 teacher|adapter [teacher_checkpoint] [extra gr3mini-train args...]" >&2
  exit 2
fi

stage=$1
shift

uv_run=(uv run)
if [[ "${GR3MINI_USE_PREBUILT_ENV:-0}" == "1" ]]; then
  uv_run+=(--no-sync)
fi

case "$stage" in
  teacher)
    exec "${uv_run[@]}" gr3mini-train Gr3Mini-Tracking-Teacher "$@"
    ;;
  adapter)
    if [[ $# -lt 1 ]]; then
      echo "adapter stage requires a teacher checkpoint" >&2
      exit 2
    fi
    teacher_checkpoint=$1
    shift
    exec "${uv_run[@]}" gr3mini-train Gr3Mini-Tracking-Adapter \
      --agent.teacher-checkpoint "$teacher_checkpoint" "$@"
    ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac


# cd /workspace/gr3mini_tracking

# GR3MINI_USE_PREBUILT_ENV=1 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# ./scripts/train_pipeline.sh teacher \
#   --gpu-ids all \
#   --env.scene.num-envs 4096 \
#   --env.commands.motion.motion-file /workspace/gr3mini_tracking/motions/easy_standing_backflip_04_corrected.npz \
#   --agent.max-iterations 10000 \
#   --agent.save-interval 1000 \
#   --agent.experiment-name gr3mini_teacher_backflip \
#   --agent.run-name teacher_8gpu_4096env_backflip04
