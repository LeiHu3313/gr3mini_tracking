# gr3mini_tracking

这是一个独立的 mjlab 项目，只迁移 Any2Track 的
`GR3Mini211TrackingGeneralDRDiffCritic` teacher 和它对应的 adapter 两阶段训练。
项目不依赖 `track_mj`、JAX、Brax 或 `mujoco_playground`；`mjlab` 作为 editable
依赖使用，源码仓库无需修改。

详细的迁移范围、合同和验收标准见 [MIGRATION_PLAN.md](MIGRATION_PLAN.md)。

## 固定版本

- Python: 3.12
- mjlab: 1.6.0，本地提交 `0fb8a681136be94ffc636a3dd423cabb97d91f10`
- PyTorch: 2.9.0 + CUDA 12.8（显式固定，已在 RTX 4080 验证）
- mjlab 路径: `../mjlab`
- 其余依赖版本由 `uv.lock` 固定

安装：

```bash
cd /home/hul/workspace/hl/my_projects/gr3mini_tracking
uv sync
uv run gr3mini-list-envs --keyword Gr3Mini
```

应看到且只新增两个项目任务：

- `Gr3Mini-Tracking-Teacher`
- `Gr3Mini-Tracking-Adapter`

## Docker（服务器推荐）

Docker 镜像会固定 CUDA 12.8、Python 3.12、PyTorch 2.9.0，以及
`mjlab v1.6.0`（commit `0fb8a681136be94ffc636a3dd423cabb97d91f10`）。因此服务器不需要
准备 `../mjlab` 或项目的 `.venv`。

构建并推送镜像：

```bash
docker login docker.fftaicorp.com

PROJECT=gr3mini_tracking \
IMAGE_NAME=gr3mini-tracking \
TAG=py312-cuda12.8-v2 \
./scripts/docker/build_and_push_gr3mini_tracking.sh
```

服务器拉取并验证 GPU：

```bash
IMAGE=docker.fftaicorp.com/gr3mini_tracking/gr3mini-tracking:py312-cuda12.8-v2
docker pull "$IMAGE"
docker run --rm --gpus all --ipc=host "$IMAGE" Gr3Mini-Tracking-Teacher --help
```

单卡训练时只需挂载日志目录：

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  -v "$PWD/logs:/workspace/gr3mini_tracking/logs" \
  "$IMAGE" Gr3Mini-Tracking-Teacher \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 5000 \
  --agent.save-interval 500 \
  --agent.run-name teacher_docker
```

多卡训练可将 `--gpus '"device=0,1,2,3"'` 与 `--gpu-ids all` 一起使用；
`--env.scene.num-envs` 仍表示每张卡的环境数。

交互使用已发布的 digest 镜像时，可用：

```bash
./scripts/docker/run_gr3mini_tracking.sh
```

该脚本会将当前宿主项目完整挂到镜像的项目路径，因而代码、奖励、motions 和脚本改动都会
立即生效；镜像的 `.venv` 位于 `/opt/gr3mini-tracking-venv`，不会被挂载遮住。它还会在有
`DISPLAY` 时自动启用 X11。例如运行单卡训练：

```bash
GPU_DEVICES='device=0' ./scripts/docker/run_gr3mini_tracking.sh \
  Gr3Mini-Tracking-Teacher \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 5000 \
  --agent.run-name teacher_docker
```

只有机器人侧工具需要 `HARDWARE_MODE=1`，训练不需要 `--privileged`、`/dev` 或 EtherCAT
挂载。若项目的 Python 依赖或 `uv.lock` 改动，则重新构建镜像；一般的 Python/奖励/配置
改动不需要重建。容器会使用镜像中预构建的环境并跳过 `uv` 同步。

## 运动数据

当前默认 motion 是：
`motions/Extended_3_stageii_from_g1_gr3mini_v211_isaaclab_mjlab.npz`。它来自
Isaac Lab 处理过的 23-DoF 数据；转换器会固定缺少的 head yaw/pitch，并用当前 mjlab
模型生成 25-DoF、28-body、足端与速度参考，保证与训练仿真坐标系一致。若源数据更新：

```bash
uv run gr3mini-convert-motion \
  /home/hul/workspace/hl/my_projects/whole_body_tracking/data/gr3mini/custom/Extended_3_stageii_from_g1_gr3mini_v211.npz \
  motions/Extended_3_stageii_from_g1_gr3mini_v211_isaaclab_mjlab.npz
```

转换器也兼容原 Any2Track archive；两种输入都会强校验 joint/body/site、50 Hz 与
`wxyz` 四元数合同，不会静默接受 body order 不一致的数据。

直接查看参考运动（不跑 policy 或物理）：

```bash
uv run gr3mini-replay-motion
```

`Space` 暂停/继续，`R` 从起点重播。可先做无界面的检查：

```bash
uv run gr3mini-replay-motion --headless --max-frames 10
```

## 第一阶段：DiffCritic teacher

先用较小规模验证完整训练链路：

```bash
uv run gr3mini-train Gr3Mini-Tracking-Teacher \
  --env.scene.num-envs 64 \
  --agent.max-iterations 2 \
  --agent.save-interval 1
```

正式启动（默认每个训练进程 4096 env、24 rollout steps、50000 iterations）：

```bash
./scripts/train_pipeline.sh teacher
```

默认 PPO 映射为 lr `3e-4`、gamma `0.98`、lambda `0.95`、
clip `0.2`、entropy `0.01`、32 mini-batches、4 epochs，actor/critic MLP 都是
`(512, 512, 256, 256, 128)`。动作分布使用项目内的 Brax-compatible
tanh-normal：网络输出 `2 * 25` 个 location/scale 参数。

当前 tracking reward 使用 whole-body root-local pose/velocity（4 项）为主，root
orientation/velocity 与 torso/feet height 为整体动态辅助；joint tracking 为低权重辅助。
soft regularization 仅保留 torque（`-1e-5`）和 joint smoothness（`-2e-7`）；
reference-residual target 的 action-rate 暂不启用，避免把参考轨迹本身的快速变化当作
policy 抖动惩罚。关节/速度限位和 self-collision safety penalties 保持启用。

观测按 **状态历史 → 动作历史 → 未来参考** 拼接。Actor 输入为
`6 × 56 + 6 × 25 + 5 × 40 = 686` 维，严格只使用 encoder、IMU 与自己发出的 action：
状态帧为 `[q-default, qd×0.05, gyro×0.05, projected_gravity]`，动作帧为
`last_target-default`；未来命令帧为
`[q_ref-q, ref_linvel_local×0.05, (ref_angvel-current_gyro)×0.05,
current-to-ref_base_rot6d, torso_height_ref, feet_height_ref]`。Actor 不读取当前
root/world position、current root linear velocity、torso height、feet height 或 contact。

Critic 输入为 `686 + 398 + 405 = 1489` 维：先是无噪声语义副本的 Actor observation；
再接当前特权状态 `[root_linvel, torso/feet height, contact, 26 body root-local
pose/velocity]`；最后接 reward-aligned tracking error
`[26 body local pose/velocity error, root linear/angular velocity error,
current-to-ref root rot6d, torso/feet height error]`。这避免重复六帧特权状态。所有角速度、
线速度和关节速度均乘 `0.05`；torso 明确为 `torso_link`，不是 floating `base_link`。

这是 observation layout v3。所有旧 671-D/656-D teacher 的首层输入都与新 686-D Actor
不匹配，不能 resume，也不能作为 adapter 的 `--agent.teacher-checkpoint`；必须先训练
v3 teacher。

checkpoint 默认位于：

```text
logs/rsl_rl/gr3mini_teacher_normal/<timestamp>_teacher/model_<iteration>.pt
```

## 第二阶段：对应 adapter

adapter 训练必须显式传入本项目第一阶段产生的 v3 teacher checkpoint。启动时会核对
teacher actor 的 686 输入维度、隐藏层和 50 输出维度；不匹配直接报错。

Smoke：

```bash
TEACHER_CKPT=/absolute/path/to/model_1.pt
uv run gr3mini-train Gr3Mini-Tracking-Adapter \
  --agent.teacher-checkpoint "$TEACHER_CKPT" \
  --env.scene.num-envs 64 \
  --agent.max-iterations 2 \
  --agent.save-interval 1 \
  --agent.world-model-num-mini-batches 4
```

正式启动：

```bash
./scripts/train_pipeline.sh adapter /absolute/path/to/teacher/model.pt
```

adapter actor 会冻结完整 teacher MLP，并在每个 Linear（包括输出层）叠加一个
零初始化分支。PPO 只更新 adapter、action distribution 和新 critic；79 帧 history
encoder 与 world model 由独立的加权 autoregressive world loss 更新。当前 adapter
world-model runner 是单 GPU 实现。

## Resume 与播放

Teacher resume：

```bash
uv run gr3mini-train Gr3Mini-Tracking-Teacher \
  --agent.resume True \
  --agent.load-run '2026-.*' \
  --agent.load-checkpoint 'model_.*.pt' \
  --agent.max-iterations 1000
```

Adapter resume 仍需给出原 teacher 路径用于构造阶段，随后 adapter checkpoint 会覆盖
完整模型状态：

```bash
uv run gr3mini-train Gr3Mini-Tracking-Adapter \
  --agent.teacher-checkpoint /absolute/path/to/teacher/model.pt \
  --agent.resume True \
  --agent.max-iterations 1000
```

本地 teacher 播放：

```bash
uv run gr3mini-play Gr3Mini-Tracking-Teacher \
  --checkpoint-file /absolute/path/to/teacher/model.pt \
  --motion-file "$PWD/motions/Extended_3_stageii_from_g1_gr3mini_v211_isaaclab_mjlab.npz"
```

本地 adapter 播放同理，将 task 和 checkpoint 换成 Adapter；adapter checkpoint 已含
teacher base、history encoder 和 world model，不需要额外传 teacher checkpoint。

## 训练合同

| 项目 | Teacher | Adapter |
|---|---:|---:|
| Actor observation | 686 | 686 + dynamics embedding |
| Critic observation | 1489 | 1489 + dynamics embedding |
| Action | 25-D `reference q + residual` | 相同 |
| Actor state / action history | `6 x 56` / `6 x 25` | 相同 |
| Critic privileged / error | `398` / `405`（均为当前单帧） | 相同 |
| Future reference | Actor `5 x 40`，并作为 Critic 前缀的一部分 | 相同 |
| Adapter history | - | 79 x 81 |
| Dynamics embedding | - | 128 |
| World state | - | 57 |

源实现是 JAX/MJX + Brax，目标是 PyTorch + MuJoCo-Warp + RSL-RL，因此旧 JAX
checkpoint 不能直接加载；teacher 和 adapter 必须都由这个项目重新训练。源 torque-speed
包络和 RFI motor noise 没有在本次最小迁移里数值复刻，PD、effort limit、armature、
friction 和主要 DR 已保留。完整差异见迁移计划。

## 验证

```bash
uv run ruff check .
uv run pyright
uv run pytest -q
```

资产来源与许可证见 `assets/gr3mini211/README.md` 和
`assets/gr3mini211/UPSTREAM_LICENSE`。
