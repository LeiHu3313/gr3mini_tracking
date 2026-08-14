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
| Actor 状态帧 | default-relative joint pos 25 + scaled joint vel 25 + scaled gyro 3 + projected gravity 3 = 56 |
| Actor 动作帧 | default-relative previous joint target = 25 |
| Actor 历史 | 状态 `6 * 56`，后接动作 `6 * 25` |
| Actor future | `t+1..t+5`，每帧 `[q_ref-q, ref root linvel local, ref-current angvel, current-to-ref base rot6d, torso target, feet targets]` = 40 |
| Actor 总维度 | `6 * 56 + 6 * 25 + 5 * 40 = 686` |
| Critic 的 actor 部分 | 无噪声语义副本，686 维（状态历史、动作历史、未来参考） |
| Critic 当前特权状态 | root linvel、torso/feet 高度、接触及 26 body 的 root-local pose/velocity = 398 |
| Critic tracking error | 26 body 的 root-local pose/velocity error + root velocity/rotation + torso/feet height error = 405 |
| Critic 总维度 | `686 + 398 + 405 = 1489` |
| Teacher MLP | actor/critic 均为 `(512, 512, 256, 256, 128)` |
| PPO 关键参数 | lr `3e-4`、gamma `0.98`、lambda `0.95`、clip `0.2`、entropy `0.01` |
| Adapter 历史 | 默认 79 帧，每帧 81 维 |
| History encoder | Conv1d `64@k9/s5 -> 64@k6/s3`，输出 128 维 |
| Residual adapter | teacher actor 每层均叠加零初始化 adapter 分支；teacher base 冻结 |
| World state | gyro 3 + gravity 3 + joint pos 25 + joint vel 25 + root height 1 = 57 |
| World model | `(512, 512, 256, 256, 256, 128)`，预测 gyro/joint-velocity/root-height 增量 |
| Adapter 优化 | adapter + critic 用 PPO；history encoder + world model 用加权 world-model loss |

本项目 observation layout v3 以部署可观测性为边界：Actor 只能使用 encoder、IMU 和
自己发出的 action；它不读取当前 root/world position 或 current root linear velocity。
未来命令显式提供 canonicalized reference root linear velocity 与 current-to-reference
base rotation 6D。Critic 以 Actor observation 为前缀，只追加单帧 current privileged
state 与 reward-aligned tracking error，不再重复六帧特权状态。torso 高度固定为
`torso_link`，不是 floating `base_link`。

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
- 实现 686 维 actor 与 1489 维 asymmetric critic observation；Actor 按状态历史、动作
  历史、未来参考依次拼接（`6x56 + 6x25 + 5x40`）。Critic 依次拼接 clean Actor
  observation、398 维当前特权状态、405 维 reward-aligned error，并以断言锁定维度和顺序。
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
- Actor/Critic observation 分别严格为 686/1489；Critic 的布局严格为
  `Actor 686 + current privileged 398 + tracking error 405`，adapter history 为
  `79 * 81`。
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

以下是 observation layout v1/v2 的历史实施记录。当前 v3 已将 Actor 改为 686 维、Critic
改为 1489 维 asymmetric layout，因此 v1/v2 的 teacher / adapter 训练结果、reset/step 和
PPO smoke 不能视为 v3 的运行时验收；v3 必须从新 teacher 重新开始。

旧 layout 项目固定为 Python 3.12、mjlab 1.6.0、PyTorch 2.9.0（CUDA 12.8），并生成了
项目内默认 motion。验收记录如下：

- 项目 CLI 只列出 Teacher/Adapter 两个新增任务。
- Ruff、Pyright 和 5 个合同测试全部通过；v3 修改后需要重新执行 reset/step smoke。
- RTX 4080 上完成 teacher 与 adapter 环境 reset/step；此记录对应旧 layout。v2 的目标
  观测为 `actor=656`、`critic=3047`、`critic_state_history=2712`、
  `critic_action_history=150`、`history=6399`、`world=57`、
  `reference_world=57`，
  reward/observation 均为有限值。
- Teacher 已完成 1 次最小 PPO update，并成功生成 RSL-RL checkpoint 与 ONNX。
- Adapter 使用该 teacher checkpoint 完成 1 次 PPO + world-model update；checkpoint
  包含 history encoder、world model 和两个 optimizer，且检查确认 frozen teacher base
  的 12 个参数张量完全未变、12 个 adapter 参数张量均已发生非零更新。
- `mjlab` 与 `any2track` 源仓库均保持零修改；smoke 日志位于本地 `logs/smoke/`，已被
  `.gitignore` 排除。

### v3 重新验收（2026-08-14）

- Actor 合同已更新为 686 维，Critic 合同更新为 1489 维；ObservationManager 打印的各项
  子块严格为 Actor `336 + 150 + 200`、Critic `336 + 150 + 200 + 398 + 405`。
- Ruff、Pyright、6 个合同测试和 git diff whitespace 检查全部通过。
- RTX 4080 上以 64 environments、4 rollout steps、2 PPO iterations 重新完成 teacher
  smoke；使用产生的 686-D checkpoint 继续完成 adapter 的 2 PPO/world-model iterations。
  两阶段均没有 NaN 或 observation-contract 错误。
