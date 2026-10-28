# G1 全身运动学习：技术路径调研与官方跟踪器复现

核验日期：2026-10-29（Asia/Shanghai）。范围：G1、Isaac Lab、本机单张 RTX 4090；优先可审计的实际训练，而不是用视频或论文宣传认定“全球最好”。本次已启动的是 **BeyondMimic 官方 motion-tracking 阶段的兼容移植训练**，不是其完整 guided-diffusion 框架，更不是 SONIC / SuperMimic 的完整复现。

## 1. 结论与选择标准

“最好”必须限定指标：单动作跟踪误差、多动作覆盖、算力效率、真机可部署性、开源可复现性和本机兼容性不是同一排名。论文之间的数据规模、终止条件、硬件和训练预算不同，不能直接用作者的单个数字做公平排名。

- **当前主线：BeyondMimic 官方 G1 跟踪器。** 官方有 Isaac Lab 的 tracking MDP、G1 执行器配置和 PPO 超参数；本机已经具备可用的动作、物理运行时与基线。此次直接使用官方代码，而非再命名一个手写奖励的任务为“论文复现”。
- **大规模通用控制对照：GEAR-SONIC。** 官方现已开放训练、部署、检查点；将多种参考输入编码为通用运动 token，再由统一策略控制全身。README 当前提供 G1 的默认、低延迟、v1.1 模型，运行于 50 Hz。官方训练说明建议微调也使用 64+ GPU。本机从零复现该大规模预算不现实；使用公开权重只是迁移/评估，不等于从零复现。
- **近期值得验证的效率路线：SuperMimic，2026-10-20 发布。** 用普通仿真的真实状态转移训练可微世界模型，然后在短时想象轨迹上通过 BPTT 优化策略，critic 补偿有限时域；世界模型和策略交替持续更新，不对接触仿真器本身反向传播。作者报告单张 H100 3 小时超过发布版 SONIC 的特定基准回报；不是 RTX 4090 的训练承诺。其 129,514 个动作的失败统计是训练动作覆盖，而非未见动作泛化。本次核验的官方主页未找到训练仓库链接；网上的独立复现须单独审计，不能冒称官方。
- **近期形态泛化路线：MimicAll，2026-10-24 发布。** 将机器人形态/动力学描述与对应参考动作一起输入策略，随机化本体并训练共享跟踪器。对改变腿长、质量与支撑形式有启发；论文数字不是 G1 穿 30 cm 高跷的成功证据。本次未运行此方法。
- **MimicKit：算法研究工具箱。** 提供 DeepMimic、AMP、AWR、ASE、LCP、ADD、SMP 等方法。明确目标姿态跟踪与判别器/运动先验匹配是不同学习目标；不是“换成 AMP 一定更好”。适合后续与相同机器人/数据/预算配对比较。
- **BeamDojo：稀疏落脚点路线。** 多边形足底采样落脚奖励、dense/sparse 双 critic，先在平地物理下学习目标地形感知，再到真实任务地形微调，降低稀疏支撑上的探索难度。可借鉴高跷课程，但高跷是机器人形态/接触变化，不等于在窄梁地形上走；本次未实现其 double-critic 算法。
- **Stilt-bot：不同硬件平台参考。** 通过 Crossref 核验到 ACC 2025 论文 DOI `10.23919/ACC63710.2025.11107631`。本次未获取可核验正文/训练代码；不能据标题推断其使用 PPO，或宣称已有可直接套用的 G1 高跷实现。

维护状态很重要：`HybridRobotics/whole_body_tracking` 于 2026-10-16 归档，README 指向 mjlab 中的跟踪任务。mjlab 使用 MuJoCo-Warp，不是 Isaac Lab / PhysX。此次为了遵守用户指定 Isaac Lab 而固定归档前官方代码；如后续选择 mjlab，应作为独立仿真后端对照，不能暗中替换。

## 2. BeyondMimic 怎样完成学习

### 2.1 从人类参考到机器人的监督目标

典型前处理链是：人体动作 → 针对机器人关节/比例的重定向 → 根位姿与关节角序列 → FK 得到所有身体的位姿和速度 → 物理 RL。

本次没有重新运行“人类到 G1”的重定向，使用宇树已重定向的 `dance_102` CSV。60 Hz CSV 的 29 个 SDK 关节按名称映射到 PhysX articulation 顺序，根四元数由 XYZW 转 WXYZ，位置/关节线性插值、姿态 SLERP 到 50 Hz。随后在 **BeyondMimic 官方 G1 cylinder 机器人**上重新做 FK：29 actuated joints、37 bodies；不能复用上一轮 35-body URDF 的按索引身体数组。

验证结果保存在 `artifacts/beyondmimic/motion_validation.json`：关节映射最大绝对误差约 2.07e-5 rad，四元数模长误差约 5.96e-7，身体线速度与位置差分 RMSE 约 0.0904 m/s，所有值有限。此处直接写状态仅用于 FK 制备数据，**不是策略控制的成功演示**。

### 2.2 策略输入、动作输出与执行器

本次保留官方 `Tracking-Flat-G1` 的带状态估计输入，而非删掉位置/速度的消融版本：

- actor 输入 160D：参考关节角/速度 58D、参考锚点相对位置 3D/朝向表示 6D、基座线速度 3D/角速度 3D、关节相对角 29D/速度 29D、上一步动作 29D。
- critic 输入 286D：除本体状态和参考外，还能访问多身体相对位置/朝向等训练特权信息。
- actor/critic 均为 512 → 256 → 128、ELU；actor 输出 29 个关节目标偏移。
- actor/critic 分别做 observation normalization，兼容本机 RSL-RL 新版接口。

动作不是直接改根位置：`q_target = q_default + scale * action`。隐式 PD 执行器把关节目标变为受扭矩/速度约束的物理作用，浮动基座只能靠接触力与关节控制保持平衡。官方按反射转子惯量设定增益：`kp = I * omega^2`，`kd = 2 * I * zeta * omega`，使用 10 Hz 自然频率、阻尼比 2；动作比例按扭矩限值和刚度设定。这些设计约束“策略输出”与真实执行器的关系，不只是网络大小。

### 2.3 跟踪奖励不是关节动作的简单回归

PPO 优化物理 rollout 的累计回报，而不是给每帧目标做行为克隆：

`r = sum_i w_i * exp(-error_i^2 / sigma_i^2) - smoothness - joint_limit - undesired_contact`。

跟踪项包含：全局 torso 锚点位置/朝向、多身体相对位置/朝向、身体世界线速度/角速度。相对项允许一定平面漂移而保持动作形态；全局锚点项避免无限漂离目标轨迹。具体公式以官方代码为准，不把上述示意公式当成各项精确实现。

这次沿用官方 tracking rewards：不额外加入宇树移植中的 joint acceleration / torque 惩罚，不通过不断调奖励凑一个“更像论文”的版本。

### 2.4 参考状态初始化与自适应采样

每次 reset 从参考片段的某一时刻设置物理初态，再加入位姿、速度和关节扰动。之后完全由策略在重力和接触动力学中执行。这样不用从默认站姿学会进入每个高动态片段。

将时间轴分桶，统计在哪个片段更容易失败，对失败多的片段增加训练概率，保留均匀采样成分，平滑采样概率。学会难段后概率可以回到较均衡分布。它改变的是训练时的练习分布，不是评估时挑简单起点：最终评估始终从第零帧跑完完整片段。

### 2.5 PPO、非对称 critic 与域随机化

本次：4096 environments × 24 steps = 每次迭代 98,304 个状态转移；PPO 5 epochs / 4 minibatches，初始学习率 1e-3、adaptive KL schedule、clip 0.2、gamma 0.99、GAE lambda 0.95、entropy 0.005。策略 50 Hz、PhysX 200 Hz，4 次物理子步。

critic 可以使用仿真特权身体状态估计长期回报，actor 只依赖设定的输入；GAE 给策略提供优势估计，PPO clip 控制单次更新幅度。噪声、材料、关节零位、躯干质心随机化与外力速度扰动降低策略对某个理想仿真参数的过拟合。实际真机还需状态估计、延迟/摩擦等核验、sim2sim、扭矩与急停安全验证；本次只在仿真训练，未向真实机器人发指令。

### 2.6 论文的第二部分：本次没有训练

完整 BeyondMimic 还将多种跟踪技能收集成物理轨迹，通过条件动作 latent 表示和序列扩散模型学习可组合的运动分布，再在推理去噪中以任务代价引导。它解决新任务组合/规划，而当前 motion tracker 解决按给定参考稳定执行。**训练一个 dance tracking policy，不能称已复现 guided diffusion、通用多技能大模型或自由文本控制。**

## 3. 本次实际复现边界

- 官方仓库 commit：`9db915838b17785c6fec90c91fe80b95b147e1bd`。
- 官方机器人描述包固定 GCS generation `1749613070556521`，归档与文件 SHA256 记录于 `configs/beyondmimic_lock.json`。
- 原始源码不改，兼容副本在 `vendor/whole_body_tracking`，完整差异在 `artifacts/upstream/beyondmimic_lab23.patch`，许可证保留。
- 官方原运行环境 Lab 2.1 / Sim 4.5；本机 Lab 2.3 / Sim 5.1 / PhysX。改动是噪声类型名、逆四元数旋转 API 名、避免递归注册不相关环境、RSL-RL observation routing/normalization 和 reset 缓存同步；不是严格原软件版本数值复现。
- 新 Gym task：`BeyondMimic-G1-29dof-Dance-v0`，独立检查点谱系；不能加载旧 154D actor 的权重当成继续训练。
- 新训练从零开始，seed 42，4096 envs，30,000 iterations，保留官方 default 预算；这不保证最终收敛。
- 训练 PID、启动时 git dirty/revision、源代码和参考哈希、完整 YAML、TensorBoard、所有检查点与里程碑可追溯。权重、参考数据和视频大文件留本地，不把 Git 仓库当大模型存储。

## 4. 修正了旧舞蹈评估器，而不是藏掉失败

旧 `UnitreeRef` dance 最终审计报告所有环境在第一个控制步终止，且 settled metrics 全为 null。诊断发现 Lab2.3 全量 `reset()` 不计算 MotionCommand 的相对身体参考缓存；物理姿态与新参考帧一致，但 `body_pos_relative_w` 仍来自之前状态，所以 ee-body 终止误触发。

修复在同一帧刷新参考缓存、重新计算观察；不推进仿真、不跳过第一步判定、不放宽门限。新 evaluator 在 termination.compute 边界复制物理状态/原因，避免读取自动 reset 后的姿态，再将其误记为失败前姿态。

配对结果使用同一旧 `model_1999.pt`：

- 未刷新缓存的负对照：64/64 在第 0 步触发 ee_body_pos；详见 `legacy_dance_reset_regression_unprimed/evaluation.json`。
- 刷新后完整 1455 步（29.1 s）：64/64 存活，关节 RMSE 0.14684 rad，身体位置平均误差 0.04316 m，通过原门限；不是新30k训练结果。
- 旧失败 JSON/轨迹保留，不原位覆盖。`scripts/unitree_reference/rollout.py` 现在委托版本化 V2 evaluator，之后新评估不会重复旧 bug。
- “任务终止”与“物理摔倒”分别解释：anchor 或末端跟踪偏差也可触发任务失败，因此 V2 使用 `task_termination_rate` 并记录逐项 termination causes，不把全部叫摔倒。
- 10 秒视频只供动作外观核验，`horizon_complete=false`，不能代替完整片段审计。小规模官方 model0 评估也只验证 pipeline，预期 gate=false。

## 5. 里程碑与验收

训练每 500 迭代存模型。监督脚本监视不可变检查点、确认文件大小稳定后再开始独立评估，同一时刻最多一个评估 worker。正式训练进程不为录像中断，也不反复重启 optimizer。

- 视频节点：500、1000、3000、10000、29999；1280×720 / 50 fps / 500 frames，真实闭环 policy rollout，非参考回放。
- 完整 64-env 审计节点：1000、3000、10000、29999。固定参考起点0、无 observation noise / pushes；保留 startup 随机化。
- 舞蹈门限：完整片段 survival ≥95%，per-joint RMSE ≤0.3 rad，body-position mean error ≤0.15 m。
- 成功/失败、视频ffprobe、检查点/视频SHA均写入每次 session 的 progress.json；预算完成与 gate 通过分开。
- 目前只是正式运行开始，不能用训练 reward 或模型文件存在认定学会。

## 6. 高跷为什么不能直接套普通身体策略

上一轮 30 cm 刚性高跷：64-env 30s 无终止，但平均前向速度约 -0.00175 m/s、前进仅约 2.4–2.8 cm；这是稳定站立的局部最优，不是行走。机器人确实有额外刚体、质量/惯量和接触碰撞，不是视觉拉长腿。

下一步应采用 **形态相容参考 + 真实接触 + 逐级课程**：先验证原地换重心与抬一脚，再低速交替踏步；在更短/更宽支撑高跷学会后逐渐过渡到目标几何，每个阶段重新验证动力学及目标高度。参考 FK、支撑足位置、根高度、碰撞几何和身体速度都必须一致，不能只给原始参考根高度加 0.3 m 就宣称处理了新的接触动力学。形态连续变化时惯量与初态也应物理一致。BeamDojo 的软约束到硬约束思路仅是方法借鉴，不是本次已实现的算法。

本次新正式运行当前只训练官方 dance tracker；普通走跑已有上一轮通过记录，高跷仍未通过且未被新训练覆盖。后续高跷课程应单独注册版本、保留当前失败控制，并与同预算/同seed直接训练做配对比较。

## 7. 可复现命令（不要在已有正式进程旁再启动第二份）

```bash
cd /data/踩高跷
source scripts/beyondmimic/runtime.sh
# 干净机器先下载；当前机器已具备以下源与资产。
git clone https://github.com/HybridRobotics/whole_body_tracking.git third_party/whole_body_tracking
git -C third_party/whole_body_tracking checkout 9db915838b17785c6fec90c91fe80b95b147e1bd
curl -L --fail -o third_party/unitree_description.tar.gz 'https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz?generation=1749613070556521'
python scripts/beyondmimic/prepare.py
# 需要既有锁定的 Unitree CSV 和 SDK 名称 manifest：参见 docs/unitree_reference.md
python scripts/beyondmimic/convert_motion.py --headless
python scripts/beyondmimic/validate_motion.py
python -m unittest discover -s tests -v
# 仅当当前训练不在运行时启动：
python scripts/beyondmimic/run_training.py --iterations 30000 --num-envs 4096 --seed 42
# 只读查询：
python scripts/beyondmimic/status.py
```

## 一手来源

- BeyondMimic paper：https://arxiv.org/abs/2508.08241
- 官方 tracking code：https://github.com/HybridRobotics/whole_body_tracking
- 官方维护继任（不同仿真后端）：https://github.com/mujocolab/mjlab
- MimicKit：https://github.com/xbpeng/MimicKit ，https://arxiv.org/abs/2510.13794
- BeamDojo：https://arxiv.org/abs/2502.10363 ，https://why618188.github.io/beamdojo/
- SONIC 官方代码/训练说明：https://github.com/NVlabs/GR00T-WholeBodyControl
- SuperMimic：https://arxiv.org/abs/2610.28480 ，https://supermimic.github.io/
- MimicAll：https://arxiv.org/abs/2610.33508
- Stilt-bot DOI：https://doi.org/10.23919/ACC63710.2025.11107631 （只核验元数据，未核验正文算法）

调研下载的原始网页/论文内容只留本地 `artifacts/research/*.source`，不发布整篇论文；已记录的抓取时间和 SHA 在 `artifacts/research/sources.json`。来源作者报告不等于本机独立验证。
