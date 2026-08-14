# GR3Mini211 DiffCritic Teacher / Adapter 迁移计划

## 1. 目标与边界

在独立项目 `gr3mini_tracking` 中，以 `mjlab` 为依赖，干净重建两阶段训练：

1. `GR3Mini211TrackingGeneralDRDiffCritic` teacher 的 PPO 基础训练。
2. 与该 teacher actor 契约严格匹配的 residual adapter + world model PPO 训练。

本次只迁移平地 GR3Mini211 DiffCritic 链路，不迁移 RawCritic、Robust、
CommandEncoding、DAgger、rough terrain、评估报表或旧 JAX/Brax 训练框架。

## 2. 已确认的源契约

源代码：

- Teacher 配置：
  `/home/hul/workspace/hl/code/any2track/track_mj/envs/gr3mini211_tracking/train/gr3mini211_env_tracking_general_dr.py`
- Adapter 配置：
  `/home/hul/workspace/hl/code/any2track/track_mj/envs/gr3mini211_tracking_adapter/train/gr3mini211_env_tracking_general_dr.py`
- Adapter 网络/损失：
  `/home/hul/workspace/hl/code/any2track/track_mj/learning/policy/model_based_ppo/`
- 机器人资产：
  `/home/hul/workspace/hl/code/any2track/storage/assets/fourier_gr3mini_v211/`

必须保留的训练合同：

| 合同 | 值 |
|---|---|
| 控制/物理周期 | `0.02 s` / `0.002 s`，decimation `10` |
| 动作 | 25 维，`reference_joint_pos + residual_action` |
| Actor 当前帧 | gravity 3 + gyro 3 + joint pos 25 + joint vel 25 + previous motor target 25 = 81 |
| Actor 历史 | 5 个过去帧 + 当前帧，共 `6 * 81` |
| Actor future | `t+1..t+5`，每帧 37 维 raw reference |
| Actor 总维度 | `6 * 81 + 5 * 37 = 671` |
| Critic 当前状态 | 477 维 privileged state |
| Critic future | `future_raw - current_tracking_state`，`5 * 37` |
| 源 Critic 总维度 | `477 + 5 * 37 = 662` |
| 项目 Critic history 扩展 | `[t-5, ..., t]` 的 6 帧完整 privileged state/action，`6 * 477 = 2862` |
| 项目 Critic 总维度 | `6 * 477 + 5 * 37 = 3047` |
| Teacher MLP | actor/critic 均为 `(512, 512, 256, 256, 128)` |
| PPO 关键参数 | lr `3e-4`、gamma `0.97`、lambda `0.95`、clip `0.2`、entropy `0.01` |
| Adapter 历史 | 默认 79 帧，每帧 81 维 |
| History encoder | Conv1d `64@k9/s5 -> 64@k6/s3`，输出 128 维 |
| Residual adapter | teacher actor 每层均叠加零初始化 adapter 分支；teacher base 冻结 |
| World state | gyro 3 + gravity 3 + joint pos 25 + joint vel 25 + root height 1 = 57 |
| World model | `(512, 512, 256, 256, 256, 128)`，预测 gyro/joint-velocity/root-height 增量 |
| Adapter 优化 | adapter + critic 用 PPO；history encoder + world model 用加权 world-model loss |

DiffCritic 与 RawCritic 的 actor 完全相同；DiffCritic 只把 critic 的五帧 future goal
改为相对当前 tracking state 的差值。本项目保留该 DiffCritic goal，并按训练需求将
critic 扩展为六帧完整 privileged state/action history；这与源任务 662 维 critic 的唯一
观测差异。

## 3. 目标结构

```text
gr3mini_tracking/
├── pyproject.toml
├── README.md
├── MIGRATION_PLAN.md
├── assets/gr3mini211/
├── motions/
├── scripts/
│   └── train_pipeline.sh
├── src/gr3mini_tracking/
│   ├── robots/gr3mini211.py
│   ├── tasks/tracking/
│   │   ├── actions.py
│   │   ├── observations.py
│   │   ├── rewards.py
│   │   ├── terminations.py
│   │   ├── env_cfg.py
│   │   └── rl_cfg.py
│   ├── adapter/
│   │   ├── models.py
│   │   ├── algorithm.py
│   │   └── runner.py
│   └── cli/
└── tests/
```

## 4. 实施步骤

### 阶段 A：独立项目与依赖

- 用 `uv init --package` 初始化项目并创建独立 Git 仓库。
- 以 editable path dependency 使用
  `/home/hul/workspace/hl/my_projects/mjlab`；锁文件记录 mjlab `1.6.0`，同时在
  README 记录当前源码提交 `0fb8a681136be94ffc636a3dd423cabb97d91f10`。
- 入口命令先导入 `gr3mini_tracking.tasks` 完成外部任务注册，再调用 mjlab 的
  train/play/list-envs。

### 阶段 B：机器人与运动数据

- 复制 GR3Mini211 MJCF 和 mesh；保留 Apache-2.0 来源说明。
- 用 mjlab `EntityCfg` 重建 25 个 position-controlled actuator，保留源 joint order、
  默认姿态、PD gains、effort limits 和 soft limits。
- 增加转换工具，将 Any2Track NPZ 转成 mjlab MotionLoader 的
  `joint_pos/joint_vel/body_*` 格式，显式校验 joint/body 名称与四元数顺序。
- 转换 `Extended_3_stageii_new3.npz` 为项目内 smoke motion。

### 阶段 C：DiffCritic teacher

- 实现 reference-relative residual joint-position action。
- 实现 671 维 actor 与 3047 维 critic observation；critic 是 `[t-5, ..., t]` 的
  `6 x 477` 维完整 privileged state/action history，后接 5 帧 future relative goal，
  并以断言锁定维度和顺序。
- 移植 tracking rewards、termination、push、观测噪声和主要 dynamics DR。
- 注册 `Gr3Mini-Tracking-Teacher`，接入 mjlab/RSL-RL PPO。

### 阶段 D：对应 adapter

- 增加 79 帧 `history`、57 维 `world_state` 和 `ref_world_state` observation groups。
- 实现 PyTorch residual-adapter actor：复制 teacher actor/base normalization，冻结 base，
  零初始化 adapter 分支。
- 实现 history encoder 与 autoregressive world model。
- 在自定义 RSL-RL algorithm/runner 中分别更新 PPO 参数与 world-model 参数；保存
  teacher 来源、网络合同和完整 optimizer state。
- 注册 `Gr3Mini-Tracking-Adapter`；启动时必须显式给 teacher checkpoint，
  并在加载前核对 actor 输入/隐藏层/输出维度。

### 阶段 E：验证与交付

按成本从低到高执行：

1. Ruff/语法/import。
2. 运动数据转换合同测试。
3. observation 维度、history 顺序、future diff、adapter 零输出与冻结测试。
4. 任务注册与 config 构造。
5. 单环境 reset/step。
6. 小环境 teacher PPO smoke。
7. 用 smoke teacher checkpoint 启动 adapter，确认只有 adapter/critic/world-model/history
   参数更新。

如果 GPU/MuJoCo-Warp 运行时阻止较高层验证，交付时明确报告已验证到哪一级，不把
静态通过表述为训练已跑通。

## 5. 验收标准

- 项目不 import `track_mj`、JAX、Brax 或 mujoco_playground。
- mjlab 源码仓库保持零修改。
- 两个 task 可由项目自己的 CLI 列出并解析配置。
- Actor/Critic observation 分别严格为 671/3047；critic history 为 `6 * 477`，adapter
  history 为 `79 * 81`。
- Teacher checkpoint 与 adapter 不匹配时 fail fast。
- Adapter 初始化时 deterministic action 与 teacher 一致（容许浮点误差）。
- 冻结 teacher base 后一次 adapter update 不改变其权重。
- README 给出 motion 转换、teacher smoke/正式训练、adapter smoke/正式训练、resume 和
  play 命令。

## 6. 已知差异与风险

- 源实现是 JAX/MJX + Brax PPO，目标是 PyTorch + MuJoCo-Warp + RSL-RL；checkpoint
  数值不能跨框架直接复用，只能在新项目内完成 teacher → adapter 两阶段。
- 源 torque-speed envelope 和 RFI motor noise 没有 mjlab 内置的一对一实现；优先保留
  PD/effort/armature/friction合同，额外 motor model 若需逐点数值一致应作为后续独立里程碑。
- 30 亿步、32768 env 的 JAX 预算不能直接等价换算为 RSL-RL iteration；README 同时给
  出忠实超参数映射和可运行 smoke 配置，不声称两种优化器轨迹数值等价。

## 7. 实施结果（2026-08-14）

上述阶段 A-E 已完成。最终项目固定为 Python 3.12、mjlab 1.6.0、PyTorch 2.9.0
（CUDA 12.8），并生成了项目内默认 motion。验收记录如下：

- 项目 CLI 只列出 Teacher/Adapter 两个新增任务。
- Ruff、Pyright 和 5 个合同测试全部通过。
- RTX 4080 上完成 teacher 与 adapter 环境 reset/step；观测实测为
  `actor=671`、`critic=3047`、`critic_history=2862`、`history=6399`、`world=57`、
  `reference_world=57`，
  reward/observation 均为有限值。
- Teacher 已完成 1 次最小 PPO update，并成功生成 RSL-RL checkpoint 与 ONNX。
- Adapter 使用该 teacher checkpoint 完成 1 次 PPO + world-model update；checkpoint
  包含 history encoder、world model 和两个 optimizer，且检查确认 frozen teacher base
  的 12 个参数张量完全未变、12 个 adapter 参数张量均已发生非零更新。
- `mjlab` 与 `any2track` 源仓库均保持零修改；smoke 日志位于本地 `logs/smoke/`，已被
  `.gitignore` 排除。
