# GR3Mini211 Tracking Reward / Termination Tuning

This experiment tightens the teacher/adapter tracking task after the motor
model alignment work, with source-aligned fall gates.

## What Changed

- Reward weights are aligned back to the source general-DR balance.
- `body_local_pos`: `2.0 -> 1.0`
- `body_local_rot`: `0.5 -> 1.0`
- `torso_world_angvel`: `5.0 -> 3.0`
- `torso_height_tracking`: `5.0 -> 3.0`
- `dof_pos_limit`: `-5.0 -> -10.0`
- Hard resets match the source HUL teacher:
  - `root_height = 0.3`
  - `shoulder_height = 0.45`
  - `body_position = 0.50`
  - `torso_ground_contact_too_long = 1.0s`
  - `invalid_state`

## Why

The local port now matches the source HUL teacher's fall gating instead of the
earlier relaxed version.

## Key Files

- `src/gr3mini_tracking/tasks/tracking/env_cfg.py`
- `src/gr3mini_tracking/tasks/tracking/rewards.py`
- `src/gr3mini_tracking/tasks/tracking/terminations.py`
- `tests/test_contracts.py`

## Train

Use the same teacher/adapter commands as the motor-alignment experiment, for
example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./scripts/train_pipeline.sh teacher \
  --gpu-ids all \
  --env.scene.num-envs 4096 \
  --env.commands.motion.motion-file "$PWD/motions/90_06_stageii.npz" \
  --agent.experiment-name gr3mini_teacher_reward_tuned \
  --agent.run-name teacher_8gpu_4096env_reward_tuned \
  --agent.max-iterations 10000 \
  --agent.save-interval 1000 \
  --agent.num-steps-per-env 20
```

Adapter:

```bash
TEACHER_CKPT="$PWD/logs/rsl_rl/gr3mini_teacher_reward_tuned/<run>/model_10000.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./scripts/train_pipeline.sh adapter "$TEACHER_CKPT" \
  --gpu-ids all \
  --env.scene.num-envs 4096 \
  --env.commands.motion.motion-file "$PWD/motions/90_06_stageii.npz" \
  --agent.experiment-name gr3mini_adapter_reward_tuned \
  --agent.run-name adapter_8gpu_4096env_reward_tuned \
  --agent.max-iterations 10000 \
  --agent.save-interval 1000 \
  --agent.num-steps-per-env 20 \
  --agent.world-model-num-mini-batches 32
```

## Watch

- termination metrics now line up with the source HUL fall gates
