# Unitree G1 Isaac Lab skill training

本仓库记录基于宇树 Unitree G1、Isaac Lab 和 RSL-RL PPO 的正式技能训练。按照当前任务要求，上一轮**未加入高跷结构**，先完成平面上的行走、跑步、跳跃和跳舞训练，并为每个技能保留训练日志、配置、里程碑 checkpoint 和演示视频。

## 新一轮：参考宇树 Unitree RL Lab 的四技能训练

当前按新请求接入官方 `isaaclab3.0` 分支的 G1 29 DOF 速度跟踪与动作模仿，实现行走、跑步、舞蹈和真实附加刚体高跷任务。本机使用明确记录的 Isaac Lab 2.3 / Sim 5.1 PhysX 兼容移植；不宣称运行了 Lab 3.0 / Sim 6.0。新任务、版本适配、复现命令和证据边界见 [Unitree reference training](docs/unitree_reference.md)。

新一轮输出在 `artifacts/unitree_reference/` 和 `logs/unitree_reference/`。下方的原生 G1 四技能说明及 `artifacts/formal/` 证据属于上一轮无高跷实验。

## 上一轮阶段边界

- 机器人：Isaac Lab 原生 Unitree G1 minimal asset。
- 场景：平面，不注册任何高跷 USD、附加刚体或约束。
- 行走/跑步：原生 G1 velocity-tracking MDP，分别使用不同的命令速度范围。
- 跳跃：基于身体高度/垂向速度周期参考和双脚离地奖励的 G1 技能任务。
- 跳舞：基于身体横向/偏航周期和上肢关节波形参考的 G1 技能任务。
- 跳跃和跳舞的周期参考是可复现的合成训练信号，不宣称已经完成 BeyondMimic/MimicKit 风格的人体动作捕捉复现；相关论文和代码只作为后续 imitation 接口的研究参考。

## 任务 ID

- `StiltBot-G1-Walk-v0`
- `StiltBot-G1-Run-v0` / `StiltBot-G1-Run-v1`
- `StiltBot-G1-Jump-v0` / `StiltBot-G1-Jump-v1`
- `StiltBot-G1-Dance-v0` / `StiltBot-G1-Dance-v1`

`StiltBot-G1-Warmup-v0` 仍保留为运行时基线。高跷任务没有注册。

## 形式化证据布局

正式训练完成后，每项技能的证据写入；当前已完成的训练与独立审计汇总见 [`docs/formal_evidence.md`](docs/formal_evidence.md)：

```text
logs/rsl_rl/<experiment>/<timestamp>_formal_seed42/
  model_0.pt, model_50.pt, ..., model_1499.pt
  params/env.yaml
  params/agent.yaml
  events.out.tfevents.*
  videos/play/rl-video-episode-0.mp4
artifacts/formal/<skill>.json
logs/formal_suite/<skill>.train.log
logs/formal_suite/<skill>.video.log
```

`artifacts/formal/<skill>.json` 是可复核清单，包含任务 ID、训练预算、seed、配置快照、TensorBoard event 文件、每个 checkpoint 的 SHA-256、最终 checkpoint、视频路径/哈希和 `stilt_geometry_registered: false` 的范围声明。`logs/` 和模型/视频文件默认不提交到 GitHub，但会保留在本机的实验目录中。

## 运行环境和命令

```bash
export ISAACLAB_PATH=/data/robot_motion_lib_isaaclab_eval/IsaacLab
export ISAACLAB_PYTHON=/home/ninesoo/anaconda3/envs/isaacSim/bin/python
export WARP_PATH=/home/ninesoo/wzh/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/extscache/omni.warp.core-1.8.2+lx64
export PYTHONPATH="$WARP_PATH:$PWD/src:${PYTHONPATH:-}"

python scripts/check_readiness.py \
  --isaaclab "$ISAACLAB_PATH" \
  --python "$ISAACLAB_PYTHON"

# 先做小规模任务级 smoke
TERM=xterm SKILL=walk NUM_ENVS=4 MAX_ITERATIONS=2 \
  scripts/train_skill.sh --headless

# 正式四技能串行训练、视频和 milestone manifest
TERM=xterm NUM_ENVS=4096 MAX_ITERATIONS=1500 SEED=42 \
  scripts/run_formal_suite.sh
```

单项训练或从指定 checkpoint 录制视频：

```bash
TERM=xterm SKILL=run NUM_ENVS=4096 MAX_ITERATIONS=1500 \
  scripts/train_skill.sh --headless

TERM=xterm SKILL=run RUN=<timestamp> CHECKPOINT=model_1499.pt VIDEO_LENGTH=500 \
  scripts/record_skill_video.sh --headless
```

视频录制脚本使用系统 `ffmpeg` 写入 RGB 帧，避免依赖 Gymnasium 的 MoviePy 可选插件；`scripts/play_task_impl.py` 保留 Isaac Lab play 逻辑并只替换视频编码层。

## 相关工作

研究记录和证据边界见 [`docs/research.md`](docs/research.md)，包括 BeyondMimic（arXiv:2508.08241）、MimicKit（arXiv:2510.13794）、BeamDojo（arXiv:2502.10363）及 Stilt-bot 方向。后续若引入高跷，必须另行完成 USD、质量/质心、摩擦、碰撞、接触传感器和约束验证，不能把本阶段平面 G1 技能结果称为高跷行走。

## 许可证与资产边界

Unitree 和 Isaac Lab 资产保持为外部依赖，不在仓库中重新分发。人体动作或 retargeted motion 数据必须保留原始许可证和来源。成功启动仿真、奖励曲线、checkpoint 或单段视频不等同于真实机器人成功；当前里程碑只证明本地 Isaac Lab G1 技能训练和可复核 rollout 证据。
