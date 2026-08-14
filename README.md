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

## 运动数据

仓库内已经转换了当前 smoke/default motion：
`motions/Extended_3_stageii_new3_mjlab.npz`。若源数据更新，重新转换：

```bash
uv run gr3mini-convert-motion \
  /home/hul/workspace/hl/code/any2track/storage/data/mocap/lafan1/FourierGR3Mini211/Extended_3_stageii_new3.npz \
  motions/Extended_3_stageii_new3_mjlab.npz
```

转换器会强校验 25 个 joint、28 个 body、site 名称、50 Hz 和 `wxyz` 四元数合同，
不会静默接受 body order 不一致的数据。

## 第一阶段：DiffCritic teacher

先用较小规模验证完整训练链路：

```bash
uv run gr3mini-train Gr3Mini-Tracking-Teacher \
  --env.scene.num-envs 64 \
  --agent.max-iterations 2 \
  --agent.save-interval 1
```

正式启动（默认 4096 env、20 rollout steps、36622 iterations）：

```bash
./scripts/train_pipeline.sh teacher
```

默认 PPO 映射保留源任务的 lr `3e-4`、gamma `0.97`、lambda `0.95`、
clip `0.2`、entropy `0.01`、32 mini-batches、4 epochs，actor/critic MLP 都是
`(512, 512, 256, 256, 128)`。动作分布使用项目内的 Brax-compatible
tanh-normal：网络输出 `2 * 25` 个 location/scale 参数。

critic 使用 `[t-5, ..., t]` 的六帧完整 privileged state，每帧 477 维且已包含上一动作
target，再接五帧 DiffCritic relative future reference。因此当前项目的 critic 输入为
`6 × 477 + 5 × 37 = 3047` 维；actor 仍是原来的 671 维。先前 662/1067 维版本的
checkpoint 不能 resume teacher/adapter 训练，但其中 teacher actor 权重仍可用作新
adapter 阶段的 `--agent.teacher-checkpoint`。

checkpoint 默认位于：

```text
logs/rsl_rl/gr3mini_diffcritic_teacher/<timestamp>_teacher/model_<iteration>.pt
```

## 第二阶段：对应 adapter

adapter 训练必须显式传入本项目第一阶段产生的 teacher checkpoint。启动时会核对
teacher actor 的 671 输入维度、隐藏层和 50 输出维度；不匹配直接报错。

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
  --motion-file "$PWD/motions/Extended_3_stageii_new3_mjlab.npz"
```

本地 adapter 播放同理，将 task 和 checkpoint 换成 Adapter；adapter checkpoint 已含
teacher base、history encoder 和 world model，不需要额外传 teacher checkpoint。

## 训练合同

| 项目 | Teacher | Adapter |
|---|---:|---:|
| Actor observation | 671 | 671 + dynamics embedding |
| Critic observation | 3047 | 3047 + dynamics embedding |
| Action | 25-D `reference q + residual` | 相同 |
| Actor state history | 6 x 81 | 6 x 81 |
| Critic privileged state/action history | 6 x 477 | 6 x 477 |
| Future reference | 5 x 37 raw | 相同 |
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
