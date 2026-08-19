# GR3Mini211 Motor Alignment Experiment

This experiment aligns the teacher and adapter training tasks with the source
motor behavior used in the original tracker.

## Scope

- Piecewise torque-speed envelope.
- Episode-level PD gain scaling.
- Episode-level RFI torque-noise scale.
- Training only. Play/eval stays deterministic.

## Code Path

- Motor model: `src/gr3mini_tracking/robots/motor_model.py`
- Robot wiring: `src/gr3mini_tracking/robots/gr3mini211.py`
- Reset-time randomization: `src/gr3mini_tracking/tasks/tracking/events.py`
- Task config: `src/gr3mini_tracking/tasks/tracking/env_cfg.py`

## Train

Teacher:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./scripts/train_pipeline.sh teacher \
  --gpu-ids all \
  --env.scene.num-envs 4096 \
  --env.commands.motion.motion-file "$PWD/motions/90_06_stageii.npz" \
  --agent.experiment-name gr3mini_teacher_motor_aligned \
  --agent.run-name teacher_8gpu_4096env_motor_aligned \
  --agent.max-iterations 10000 \
  --agent.save-interval 1000 \
  --agent.num-steps-per-env 20
```

Adapter:

```bash
TEACHER_CKPT="$PWD/logs/rsl_rl/gr3mini_teacher_motor_aligned/<run>/model_10000.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./scripts/train_pipeline.sh adapter "$TEACHER_CKPT" \
  --gpu-ids all \
  --env.scene.num-envs 4096 \
  --env.commands.motion.motion-file "$PWD/motions/90_06_stageii.npz" \
  --agent.experiment-name gr3mini_adapter_motor_aligned \
  --agent.run-name adapter_8gpu_4096env_motor_aligned \
  --agent.max-iterations 10000 \
  --agent.save-interval 1000 \
  --agent.num-steps-per-env 20 \
  --agent.world-model-num-mini-batches 32
```

## Checks

```bash
uv run pytest tests/test_motor_model.py
```

What to watch:

- torque clamps at high speed instead of staying flat
- `RFI` scales reset per episode in training
- play does not enable the motor-noise reset event

