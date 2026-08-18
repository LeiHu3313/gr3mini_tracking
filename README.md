# gr3mini_tracking

## 环境配置

```bash
cd /home/hul/workspace/hl/my_projects/gr3mini_tracking
uv sync
```

## Docker

```bash
# 首次拉取镜像
docker pull docker.fftaicorp.com/gr3mini_tracking/gr3mini-tracking:py312-cuda12.8-v2

# 进入容器
./scripts/docker/run_gr3mini_tracking.sh
```

## 训练

**Teacher（8卡）：**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./scripts/train_pipeline.sh teacher \
  --gpu-ids all \
  --env.scene.num-envs 4096 \
  --env.commands.motion.motion-file "$PWD/motions/90_06_stageii.npz" \
  --agent.experiment-name gr3mini_teacher_backflip \
  --agent.run-name teacher_8gpu_4096env_backflip04 \
  --agent.max-iterations 10000 \
  --agent.save-interval 1000 \
  --agent.num-steps-per-env 20
```

checkpoint 输出到 `logs/rsl_rl/<experiment-name>/<run-name>/model_<iter>.pt`。

**Teacher Resume：**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./scripts/train_pipeline.sh teacher \
  --gpu-ids all \
  --env.scene.num-envs 4096 \
  --env.commands.motion.motion-file "$PWD/motions/90_06_stageii.npz" \
  --agent.experiment-name gr3mini_teacher_backflip \
  --agent.resume True \
  --agent.load-run 'teacher_8gpu_4096env_backflip04' \
  --agent.load-checkpoint 'model_5000.pt' \
  --agent.max-iterations 10000
```

**Adapter（8卡）：**

```bash
TEACHER_CKPT="$PWD/logs/rsl_rl/gr3mini_teacher_backflip/<run>/model_10000.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./scripts/train_pipeline.sh adapter "$TEACHER_CKPT" \
  --gpu-ids all \
  --env.scene.num-envs 4096 \
  --env.commands.motion.motion-file "$PWD/motions/90_06_stageii.npz" \
  --agent.experiment-name gr3mini_adapter_backflip \
  --agent.run-name adapter_8gpu_4096env_backflip04 \
  --agent.max-iterations 10000 \
  --agent.save-interval 1000 \
  --agent.num-steps-per-env 20 \
  --agent.world-model-num-mini-batches 32
```

**Adapter Resume（仍需传 teacher checkpoint 用于构造模型）：**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./scripts/train_pipeline.sh adapter "$TEACHER_CKPT" \
  --gpu-ids all \
  --env.scene.num-envs 4096 \
  --env.commands.motion.motion-file "$PWD/motions/90_06_stageii.npz" \
  --agent.experiment-name gr3mini_adapter_backflip \
  --agent.resume True \
  --agent.load-run 'adapter_8gpu_4096env_backflip04' \
  --agent.load-checkpoint 'model_5000.pt' \
  --agent.max-iterations 10000
```

Docker 容器内加 `GR3MINI_USE_PREBUILT_ENV=1` 前缀跳过 uv sync：

```bash
GR3MINI_USE_PREBUILT_ENV=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./scripts/train_pipeline.sh teacher --gpu-ids all ...
```

## Play

```bash
uv run gr3mini-play Gr3Mini-Tracking-Adapter \
  --checkpoint-file logs/rsl_rl/gr3mini_adapter_backflip/<run>/model_4000.pt \
  --motion-file "$PWD/motions/90_06_stageii.npz"
```

## 导出部署

```bash
CKPT=logs/rsl_rl/gr3mini_adapter_backflip/<run>/model_4000.pt
MOTION=motions/90_06_stageii.npz
PROFILE=/path/to/grx_sdk/config/gr3miniv2_1_1/mini_deploy_any2track_task/config/profiles/backflip_90_06

# 导出权重
uv run gr3mini-export-adapter "$CKPT" "$PROFILE/exported/"

# 导出轨迹
uv run gr3mini-export-ref-motion "$MOTION" "$PROFILE/"
```

导出后在taskplugin的 `config.yaml` 的 `state_profiles` 里把 `UserController_N` 指向 `backflip_90_06` 即可上机测试。

## 运动数据转换

```bash
# pkl → npz
uv run gr3mini-convert-motion motions/source.pkl motions/output.npz

# 回放查看
uv run gr3mini-replay-motion --motion-file motions/output.npz
```
