# TOP-ERL 适配代码审查报告

## 1. 仓库概览

本仓库当前是 UAV-BS 通信轨迹控制代码，核心训练框架为 DRQN/MADRQN。根据 `AGENTS.md`、`docs/adaptation_plan.md` 和 `docs/TOP_ERL_notes.md`，当前阶段只做代码审查和适配规划，不实现完整 TOP-ERL，不修改原始通信模型，也不改现有训练脚本。

后续 TOP-ERL 风格目标是新增长时域轨迹接口：

```text
step_trajectory(action_seq)
```

其中目标 `action_seq` 形状为 `[H, n_uavs, 2]`，表示未来 `H` 个时隙内所有 UAV 的二维水平位移或速度。注意：这不同于当前环境里的离散单时隙动作。

## 2. 主环境文件

主多 UAV 环境文件是：

- `envs/mubs_cov/mubs_cov.py`

核心类是：

- `MultiUbsCoverageEnv`，定义于 `envs/mubs_cov/mubs_cov.py:11`

该类负责多 UAV 位置、用户位置、动作空间、单步移动、通信服务、干扰、SINR、速率、吞吐量、公平性、碰撞惩罚和奖励。

相关地图/场景参数文件是：

- `envs/mubs_cov/maps.py`

该文件定义 UAV 数量、用户数量、区域大小、时隙长度、速度集合、通信半径、覆盖半径、episode 长度等地图参数。

## 3. UAV 和用户位置初始化

多 UAV 环境中，位置变量在 `MultiUbsCoverageEnv.__init__()` 中声明：

- UAV 位置数组：`self.pos_ubs`，见 `envs/mubs_cov/mubs_cov.py:42`
- 用户位置数组：`self.pos_gts`，见 `envs/mubs_cov/mubs_cov.py:43`

实际初始位置在 `reset()` 中从地图对象取得：

- 调用 `positions = self.map.set_positions()`，见 `envs/mubs_cov/mubs_cov.py:94`
- 赋值 `self.pos_ubs, self.pos_gts = positions['ubs'], positions['gt']`，见 `envs/mubs_cov/mubs_cov.py:95`

地图参数和位置生成位于 `envs/mubs_cov/maps.py`：

- 基类 `Map` 定义区域大小、episode 长度、UAV 数量、GT 数量、覆盖半径、RB 数量、速度集合等，见 `envs/mubs_cov/maps.py:7` 到 `envs/mubs_cov/maps.py:25`
- 基类随机位置生成：`Map.set_positions()`，见 `envs/mubs_cov/maps.py:31` 到 `envs/mubs_cov/maps.py:35`
- 调试地图固定位置：`Debug.set_positions()`，见 `envs/mubs_cov/maps.py:47` 到 `envs/mubs_cov/maps.py:50`
- 热点地图：`HotSpot.set_positions()`，见 `envs/mubs_cov/maps.py:64` 到 `envs/mubs_cov/maps.py:75`
- 密集热点地图：`DenseHotSpot.set_positions()`，见 `envs/mubs_cov/maps.py:97` 到 `envs/mubs_cov/maps.py:113`
- `DenseHotSpotV2.set_positions()`，见 `envs/mubs_cov/maps.py:125` 到 `envs/mubs_cov/maps.py:132`
- 地图注册表 `MAPS`，见 `envs/mubs_cov/maps.py:138`

## 4. 动作空间和 UAV 移动更新

当前动作是离散动作，不是连续速度、轨迹参数或 waypoint。

动作集合在 `MultiUbsCoverageEnv.__init__()` 中构造：

- 移动距离由 `dt * vels` 生成，见 `envs/mubs_cov/mubs_cov.py:61`
- 飞行方向由 `n_dirs` 均分角度生成，见 `envs/mubs_cov/mubs_cov.py:62` 到 `envs/mubs_cov/mubs_cov.py:63`
- `self.avail_moves` 包含悬停动作和所有速度/方向组合，见 `envs/mubs_cov/mubs_cov.py:64`
- 动作数量 `self.n_actions`，见 `envs/mubs_cov/mubs_cov.py:67`
- Gym 动作空间 `self.action_space = [Discrete(self.n_actions)]`，见 `envs/mubs_cov/mubs_cov.py:76`

单时隙移动更新在 `step(actions)` 中：

- `step()` 定义于 `envs/mubs_cov/mubs_cov.py:104`
- 将每个 UAV 的离散动作索引映射到位移：`moves = self.avail_moves[np.array(actions, dtype=int)]`，见 `envs/mubs_cov/mubs_cov.py:108`
- 更新并裁剪 UAV 位置：`self.pos_ubs = np.clip(self.pos_ubs + moves, 0, self.range_pos)`，见 `envs/mubs_cov/mubs_cov.py:109`

因此，TOP-ERL 第一阶段不能直接把 `[H, n_uavs, 2]` 连续轨迹动作塞进原始 `step()`；需要新增 wrapper/subclass 来解释二维位移或速度序列。

## 5. A2G 信道模型

A2G 信道模型定义在：

- `envs/common.py`

核心类和函数：

- `AirToGroundChannel`，见 `envs/common.py:31`
- 场景参数 `chan_params`，见 `envs/common.py:34`
- `estimate_chan_gain(d_level, h_ubs)`，见 `envs/common.py:49`

信道增益计算包括：

- LoS 概率 `p_los`，见 `envs/common.py:52`
- 三维距离和自由空间路径损耗，见 `envs/common.py:56`
- LoS/NLoS 加权路径损耗，见 `envs/common.py:58`
- 返回信道增益 `1 / pl`，见 `envs/common.py:59`

多 UAV 环境中使用该信道模型的位置：

- 创建信道模型：`self.chan = AirToGroundChannel(self.scene, self.fc)`，见 `envs/mubs_cov/mubs_cov.py:36`
- 计算最大链路速率前的最大信道增益，见 `envs/mubs_cov/mubs_cov.py:37` 到 `envs/mubs_cov/mubs_cov.py:39`
- 在 `_transmit_data()` 中计算 UAV 到用户的信道增益 `g`，见 `envs/mubs_cov/mubs_cov.py:174`

UAV 高度、发射功率、噪声、带宽、载频和场景参数定义在：

- `h_ubs`，见 `envs/mubs_cov/mubs_cov.py:14`
- `p_tx`，见 `envs/mubs_cov/mubs_cov.py:15`
- `n0`，见 `envs/mubs_cov/mubs_cov.py:16`
- `bw`，见 `envs/mubs_cov/mubs_cov.py:17`
- `fc`，见 `envs/mubs_cov/mubs_cov.py:18`
- `scene`，见 `envs/mubs_cov/mubs_cov.py:19`

## 6. 干扰、SINR、速率、吞吐量、公平性和碰撞惩罚

这些通信和指标逻辑主要集中在：

- `MultiUbsCoverageEnv._transmit_data()`，定义于 `envs/mubs_cov/mubs_cov.py:131`

空间关系：

- UAV-GT 距离矩阵 `self.d_u2g`，见 `envs/mubs_cov/mubs_cov.py:135` 和 `envs/mubs_cov/mubs_cov.py:139`
- UAV-UAV 距离矩阵 `self.d_u2u`，见 `envs/mubs_cov/mubs_cov.py:136` 和 `envs/mubs_cov/mubs_cov.py:141`
- 多智能体通信邻接 `self.adj`，见 `envs/mubs_cov/mubs_cov.py:143`

碰撞检测和碰撞计数：

- 碰撞安全距离 `safe_dist`，见 `envs/mubs_cov/mubs_cov.py:20`
- 碰撞惩罚强度 `penalty`，见 `envs/mubs_cov/mubs_cov.py:21`
- 碰撞 mask `self.mask_collision`，见 `envs/mubs_cov/mubs_cov.py:144`
- episode 内碰撞计数 `self.n_colls += ...`，见 `envs/mubs_cov/mubs_cov.py:145`
- `info["ProbCollision"]`，见 `envs/mubs_cov/mubs_cov.py:121`

用户调度、RB 分配和干扰：

- 调度矩阵 `self.sched`，见 `envs/mubs_cov/mubs_cov.py:172`
- 干扰张量 `p_itf`，见 `envs/mubs_cov/mubs_cov.py:173`
- 覆盖/干扰范围 mask `mask_itf`，见 `envs/mubs_cov/mubs_cov.py:175`
- 按用户优先级遍历 `self.prior_gts`，见 `envs/mubs_cov/mubs_cov.py:176`
- 为每个 GT 按距离排序候选 UAV，见 `envs/mubs_cov/mubs_cov.py:177`
- 找空闲 RB，见 `envs/mubs_cov/mubs_cov.py:181`
- 计算每个 RB 当前干扰，见 `envs/mubs_cov/mubs_cov.py:183`
- 选择最低干扰 RB 并建立调度关系，见 `envs/mubs_cov/mubs_cov.py:187`
- 将同 RB 上由当前 UAV 造成的干扰写入 `p_itf`，见 `envs/mubs_cov/mubs_cov.py:189`
- 清除目标 GT 自身信号项，见 `envs/mubs_cov/mubs_cov.py:190`

SINR 和速率：

- 初始化每用户瞬时速率 `self.rate_per_gt`，见 `envs/mubs_cov/mubs_cov.py:194`
- 找到服务该 GT 的 UAV 和 RB，见 `envs/mubs_cov/mubs_cov.py:197`
- SINR 计算，见 `envs/mubs_cov/mubs_cov.py:198`
- Mbps 速率计算，见 `envs/mubs_cov/mubs_cov.py:199`
- 每 UAV 服务速率聚合 `self.rate_per_ubs`，见 `envs/mubs_cov/mubs_cov.py:200`

吞吐量、公平性和全局效用：

- 长期平均每用户速率 `self.avg_rate_per_gt`，见 `envs/mubs_cov/mubs_cov.py:205`
- 总吞吐量 `self.total_throughput`，见 `envs/mubs_cov/mubs_cov.py:206`
- Jain 公平性 `self.fair_idx`，见 `envs/mubs_cov/mubs_cov.py:207`
- 全局效用 `self.global_util = self.fair_idx * self.rate_per_gt.mean()`，见 `envs/mubs_cov/mubs_cov.py:208`
- 平均全局效用，见 `envs/mubs_cov/mubs_cov.py:209`
- 下一时隙用户优先级 `self.prior_gts = np.argsort(self.avg_rate_per_gt)`，见 `envs/mubs_cov/mubs_cov.py:210`

Jain 公平性函数定义在：

- `compute_jain_fairness_index()`，见 `envs/common.py:19`

## 7. 奖励函数

奖励函数定义在：

- `MultiUbsCoverageEnv._get_reward()`，见 `envs/mubs_cov/mubs_cov.py:324`

奖励逻辑：

- 若启用公平服务，则每个 UAV 的基础奖励来自 `self.global_util`，见 `envs/mubs_cov/mubs_cov.py:328`
- 若不启用公平服务，则基础奖励来自平均瞬时速率，见 `envs/mubs_cov/mubs_cov.py:330`
- 奖励按 `reward_scale_rate / max_rate` 缩放，见 `envs/mubs_cov/mubs_cov.py:333`
- 未服务任何 GT 的 UAV 奖励清零，见 `envs/mubs_cov/mubs_cov.py:334` 到 `envs/mubs_cov/mubs_cov.py:335`
- 若启用避碰，则碰撞 UAV 获得负惩罚，见 `envs/mubs_cov/mubs_cov.py:338` 到 `envs/mubs_cov/mubs_cov.py:339`

`step()` 中调用奖励的位置：

- `reward = self._get_reward()`，见 `envs/mubs_cov/mubs_cov.py:114`
- episode return 累加 `reward.mean()`，见 `envs/mubs_cov/mubs_cov.py:115`

## 8. 状态和观测设计

局部观测：

- `get_obs()` 返回所有 UAV 的局部观测列表，见 `envs/mubs_cov/mubs_cov.py:212`
- `get_obs_agent(agent_id)` 返回单个 UAV 的观测字典，见 `envs/mubs_cov/mubs_cov.py:215`
- 自身位置特征 `own_feats[0:2]`，见 `envs/mubs_cov/mubs_cov.py:222`
- 其他 UAV 相对位置特征，见 `envs/mubs_cov/mubs_cov.py:227` 到 `envs/mubs_cov/mubs_cov.py:230`
- GT 相对位置、瞬时速率、平均速率特征，见 `envs/mubs_cov/mubs_cov.py:234` 到 `envs/mubs_cov/mubs_cov.py:240`

全局状态：

- `get_state()` 定义于 `envs/mubs_cov/mubs_cov.py:280`
- UAV 归一化位置，见 `envs/mubs_cov/mubs_cov.py:289`
- GT 归一化位置，见 `envs/mubs_cov/mubs_cov.py:292`
- GT 瞬时速率，见 `envs/mubs_cov/mubs_cov.py:293`
- GT 平均速率，见 `envs/mubs_cov/mubs_cov.py:295`

TOP-ERL 后续 Transformer critic 的状态序列可以优先复用 `get_state()` 的输出。

## 9. 训练和评估脚本

训练脚本：

- `run_exp1.py`：单 UAV DRQN 实验，环境入口 `SingleUbsCoverageEnv`，见 `run_exp1.py:20`
- `run_exp2.py`：多 UAV MADRQN 实验，环境入口 `MultiUbsCoverageEnv`，见 `run_exp2.py:20`
- `run_exp2.py` 的地图、fairness、collision 参数，见 `run_exp2.py:21` 到 `run_exp2.py:23`
- `run_exp3.py`：多 UAV GNN/通信实验，环境入口 `MultiUbsCoverageEnv`，见 `run_exp3.py:21`
- `run_exp3.py` 的地图、fairness、collision 参数，见 `run_exp3.py:22` 到 `run_exp3.py:24`

训练循环：

- 单 UAV DRQN 训练入口 `algos/drqn/run.py:22`
- 多 UAV MADRQN 训练入口 `algos/madrqn/run.py:22`
- 多 UAV 训练环境创建，见 `algos/madrqn/run.py:48`
- 多 UAV 测试环境创建，见 `algos/madrqn/run.py:49`
- 多 UAV 主训练循环中调用 `env.step(a)`，见 `algos/madrqn/run.py:85`
- 多 UAV 经验缓存，见 `algos/madrqn/run.py:87`
- 多 UAV 参数更新，见 `algos/madrqn/run.py:98`
- 多 UAV 日志指标，见 `algos/madrqn/run.py:115` 到 `algos/madrqn/run.py:126`

评估脚本：

- `test_policies.py` 统一加载 DRQN/MADRQN 评估函数，见 `test_policies.py:9` 到 `test_policies.py:14`
- `test_series()`，见 `test_policies.py:36`
- 从 `config.json` 恢复环境，见 `test_policies.py:47` 到 `test_policies.py:55`
- 调用具体算法的 `load_and_run_policy()`，见 `test_policies.py:67`
- 写出评估 CSV，见 `test_policies.py:82` 和 `test_policies.py:91`
- 训练曲线汇总脚本 `collect_curves.py`，入口函数见 `collect_curves.py:7`

Replay buffer：

- 多 UAV replay buffer：`algos/madrqn/buffer.py:7`
- 多 UAV learner 中创建 replay buffer：`algos/madrqn/learner.py:47`
- 多 UAV learner 缓存 transition：`algos/madrqn/learner.py:82`
- 单 UAV replay buffer：`algos/drqn/buffer.py:5`

这些 replay buffer 是面向 step-based recurrent Q-learning 的，不是 TOP-ERL 轨迹 replay buffer。

## 10. 应复用的 TOP-ERL 长时域轨迹接口部分

第一阶段应复用以下部分：

1. 复用 `MultiUbsCoverageEnv` 作为通信环境主体。
2. 复用 `envs/mubs_cov/maps.py` 中的地图参数和位置初始化。
3. 复用 `envs/common.py` 中的 `AirToGroundChannel`。
4. 复用 `MultiUbsCoverageEnv._transmit_data()` 中的用户调度、干扰、SINR、速率、吞吐量、公平性和碰撞统计。
5. 复用 `MultiUbsCoverageEnv._get_reward()` 中的奖励计算。
6. 复用 `MultiUbsCoverageEnv.get_state()` 和 `get_obs()` 生成状态/观测序列。
7. 复用 `replay()` 和 recorder/plotting 逻辑用于轨迹可视化。

建议新增一个 wrapper 或 subclass，而不是直接修改原始环境。例如后续可新增：

```text
envs/mubs_cov/mubs_cov_traj.py
```

并定义：

```text
MultiUbsCoverageTrajEnv
```

新增类只负责解释长时域动作序列：

```text
step_trajectory(action_seq)
```

其中 `action_seq.shape == [H, n_uavs, 2]`。每个时隙内部应更新 UAV 水平位移或速度，然后调用原始通信服务逻辑来计算速率、奖励和指标。

如果需要严格复用原始离散 `step()`，则必须先把 `[H, n_uavs, 2]` 投影或量化到原始离散动作索引。但根据当前说明文件，目标接口更偏向连续二维位移/速度序列，因此第一版 wrapper 更适合复用通信与奖励逻辑，而不是复用原始离散动作空间本身。

## 11. 第一阶段不应修改的部分

第一阶段不应修改：

1. 不修改 `envs/mubs_cov/mubs_cov.py` 中的原始 `step()`。
2. 不修改 A2G 信道模型 `envs/common.py`。
3. 不修改 `_transmit_data()` 中的调度、干扰、SINR、速率、吞吐量和公平性逻辑。
4. 不修改 `_get_reward()` 中的奖励和碰撞惩罚逻辑。
5. 不修改 `run_exp1.py`、`run_exp2.py`、`run_exp3.py`，保持原实验可运行。
6. 不修改 `algos/drqn` 和 `algos/madrqn` 的训练循环。
7. 不修改现有 replay buffer；TOP-ERL 轨迹 replay buffer 应后续单独新增。
8. 不在第一阶段实现 Transformer critic、ProDMP、B-spline、TRPL、全协方差 Gaussian policy、ensemble critic 或完整 TOP-ERL。
9. 不把资源分配作为第一阶段动作；RB 分配、用户调度和干扰处理应保持原环境内部逻辑。

## 12. 最小 `step_trajectory(action_seq)` 计划

后续最小改动建议：

1. 新增 `envs/mubs_cov/mubs_cov_traj.py`。
2. 定义 `MultiUbsCoverageTrajEnv(MultiUbsCoverageEnv)`。
3. 添加 `step_trajectory(action_seq)`，输入形状 `[H, n_uavs, 2]`。
4. 方法假设调用者已先执行 `reset()`。
5. 每个内部时隙按二维位移或速度更新 `self.pos_ubs`，并裁剪到 `[0, self.range_pos]`。
6. 每次位置更新后调用原始 `_transmit_data()`。
7. 每次调用原始 `_get_reward()`。
8. 记录序列：state、obs、action、reward、rate、fairness、throughput、done、info、UAV 位置、GT 位置。
9. 输出 final observation、累计奖励、done、info，以及后续 Transformer critic 所需的完整序列。
10. 添加短 smoke test，不运行长训练。

## 13. 风险和不明确点

1. 当前原始动作是离散动作索引，而目标 TOP-ERL 动作是 `[H, n_uavs, 2]` 连续二维序列。需要明确该二维量是“位移”还是“速度”。
2. 如果 action_seq 表示速度，则每步位移应为 `velocity * dt`；如果表示位移，则可直接加到 `pos_ubs`。
3. 当前 `total_throughput` 的注释有时写 Mb，但计算 `rate(Mbps) * dt / 1e3` 更接近 Gb，应在后续报告/绘图中统一单位。
4. `reset()` 中会调用 `_transmit_data()`，如果初始 UAV 位置碰撞，碰撞计数可能在第一次外部动作前已经增加。
5. 当前多 UAV wrapper 中定义了 reward normalization，但 `MultiAgentWrapper.step()` 返回的是原始 `rew`，没有调用 `self.reward(rew)`，见 `envs/multi_agent_env.py:65` 到 `envs/multi_agent_env.py:69`。后续若比较奖励尺度，需要确认是否使用原始奖励。

## 14. 推荐下一步任务

下一步建议只做一个小改动：

新增 `envs/mubs_cov/mubs_cov_traj.py` 和一个 smoke test 脚本 `scripts/smoke_test_traj_env.py`，实现最小 `step_trajectory(action_seq)`，验证它能在 `map_id="debug"` 上运行短 horizon，并输出累计奖励、平均速率、公平性、碰撞概率和 done 标记。

不要在下一步实现 Transformer critic 或完整 TOP-ERL 训练。
