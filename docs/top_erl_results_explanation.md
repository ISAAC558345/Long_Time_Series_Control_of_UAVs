# UAV-BS TOP-ERL-inspired 阶段性结果说明

本文档用于解释当前 UAV-BS TOP-ERL-inspired 最小系统的代码状态、实验配置、结果文件、图表含义和阶段性结论。它面向后续继续开发、复现实验、写论文或做阶段汇报。

注意：本文总结的是当前仓库中已经生成并归档的实验结果。部分结果是在最近的结构稳定化改造前生成的，因此它们应被视为已有实验记录，而不是对最新代码结构的最终性能宣称。后续如果要评估最新结构，需要重新运行对应训练和评估。

## 1. 当前代码整体状态

当前代码已经从原始离散单步 UAV 动作控制，扩展为一个 TOP-ERL-inspired 的长时域轨迹控制原型。整体框架仍然保持为：

```text
trajectory actor
  -> trajectory parameter w
  -> trajectory generator
  -> continuous_action_seq [H, n_uavs, 2]
  -> MultiUbsCoverageTrajEnv.step_trajectory(...)
  -> TrajectoryReplayBuffer
  -> TrajectoryTransformerCritic
  -> critic loss / actor loss
```

当前已经完成的主要模块包括：

1. `MultiUbsCoverageTrajEnv`
   - 新增的多 UAV 长时域轨迹环境封装。
   - 复用原始 `MultiUbsCoverageEnv` 的通信、调度、干扰、SINR、rate、throughput、fairness、collision 和 reward 逻辑。
   - 不改变原始 `envs/mubs_cov/mubs_cov.py` 的核心通信模型。

2. `step_trajectory(action_seq, action_type="displacement")`
   - 输入连续轨迹动作序列，shape 为 `[H, n_uavs, 2]`。
   - 每个内部 slot 更新 UAV 水平位置，然后调用原始通信逻辑。
   - 输出完整 trajectory rollout 信息，包括 state/action/reward/done/info 序列。

3. Trajectory generator
   - 早期实验主要基于 `LinearWaypointTrajectoryGenerator`。
   - 当前代码已经进一步支持更平滑的轨迹生成思路，例如 spline / polynomial 风格的生成器和可微 PyTorch 生成路径。
   - 轨迹生成器输出仍保持 `[H, n_uavs, 2]`，从而兼容 `step_trajectory(...)`。

4. `TrajectoryReplayBuffer`
   - 存储 `step_trajectory(...)` 返回的完整 rollout。
   - 支持采样固定长度 segment。
   - 已支持 `state_subseq`，shape 为 `[batch_size, segment_length + 1, state_dim]`，用于严格 N-step target。

5. `TrajectoryTransformerCritic`
   - 输入 `start_state` 和一段 `action_subseq`。
   - 输出 `state_value` 和 `prefix_q_values`。
   - 用于评估一段长时域 UAV 轨迹动作序列的 prefix value。

6. Critic loss / target modes
   - 已支持 N-step critic target。
   - 已新增 target mode：
     - `bootstrapped_n_step`
     - `no_bootstrap`
     - `weighted_bootstrap`
   - 当前阶段性结果中，`no_bootstrap` 表现更稳定，因此作为后续默认 target mode。

7. `TrajectoryActor`
   - 输入全局状态 `env.get_state()`。
   - 输出 trajectory parameter `w` 的 diagonal Gaussian policy。
   - 支持 deterministic mean evaluation 和 stochastic sampling evaluation。

8. `actor_loss`
   - 使用 critic 对 actor 生成轨迹的 `prefix_q_values` 进行反向传播。
   - actor update 不调用环境，因为环境不可微。

9. 训练、评估和诊断脚本
   - `scripts/train_top_erl_minimal.py`
   - `scripts/eval_top_erl_minimal.py`
   - `scripts/plot_top_erl_minimal_results.py`
   - `scripts/plot_top_erl_trajectories.py`
   - `scripts/diagnose_top_erl_trajectories.py`
   - 若干 multi-seed、calibration、diagnostics、report 脚本。

当前系统仍不是完整 TOP-ERL 算法。它是一个 TOP-ERL-inspired 的最小长时域 UAV trajectory actor-critic 原型。

## 2. 当前无人机通信网络是什么

底层通信环境仍然来自原 UAV-BS 覆盖控制代码。当前系统中的 UAV-BS 网络可以理解为：

1. 多个 UAV base stations 在二维区域内移动。
2. 地面用户 GT 分布在区域内。
3. UAV 之间、UAV 与 GT 之间的位置关系决定链路质量和服务关系。
4. 每个 slot 内环境会计算：
   - A2G channel gain；
   - UAV-GT 距离；
   - UAV-UAV 距离；
   - 干扰；
   - SINR；
   - per-GT rate；
   - per-UAV service rate；
   - TotalThroughput；
   - Jain fairness index；
   - collision probability；
   - reward。

当前 TOP-ERL-inspired 扩展只改变 UAV 轨迹动作的表达方式，不改变通信物理模型本身。

原始离散控制是：

```text
step(actions)
```

当前长时域控制是：

```text
step_trajectory(action_seq)
```

其中：

```text
action_seq.shape = [H, n_uavs, 2]
```

这表示未来 `H` 个 slot 中，每个 UAV 的二维水平位移序列。

## 3. 当前默认实验配置

当前最稳定的阶段性配置是 no-bootstrap target 配置。

核心设置：

```text
map_id = debug
target_mode = no_bootstrap
bootstrap_weight = 0.0
num_iterations = 500
seeds = 0, 1, 2
horizon = 4
segment_length = 4
critic_warmup_iterations = 20
critic_updates_per_iteration = 4
actor_update_interval = 2
min_buffer_rollouts_before_actor = 20
target_tau = 0.005
```

评估策略：

```text
main evaluation = deterministic actor
auxiliary evaluation = stochastic actor with std_scale = 0.25
baseline = random trajectory parameter
```

为什么主评估使用 deterministic actor：

1. 原始 stochastic actor 在 `std_scale=1.0` 时采样偏激，曾在部分实验中引入碰撞或性能波动。
2. `std_scale=0.25` 的 stochastic actor 更适合作为辅助稳定性诊断。
3. deterministic actor 更适合作为当前阶段的主评估策略，因为它不受采样噪声影响。

## 4. 当前主要结果总结

### 4.1 Debug map, no_bootstrap, 500 iterations

结果目录：

```text
outputs/top_erl_no_bootstrap_500_multiseed
```

汇总文件：

```text
outputs/top_erl_no_bootstrap_500_multiseed/multiseed_summary.csv
outputs/top_erl_no_bootstrap_500_multiseed/report/summary_report.md
```

三 seed 结果如下。

| seed | deterministic reward | stochastic std=0.25 reward | random reward | deterministic reward gap | stochastic reward gap | deterministic FairIdx gap | stochastic FairIdx gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.253737 | 0.214649 | 0.083097 | 0.170640 | 0.131553 | 0.319284 | 0.297884 |
| 1 | 0.432042 | 0.393645 | 0.085005 | 0.347037 | 0.308640 | 0.443347 | 0.364588 |
| 2 | 0.040352 | 0.147314 | 0.041266 | -0.000914 | 0.106048 | 0.172636 | 0.233791 |

平均结果：

| metric | value |
|---|---:|
| mean deterministic reward gap | 0.172255 |
| mean stochastic std=0.25 reward gap | 0.182080 |
| mean deterministic FairIdx gap | 0.311756 |
| mean stochastic std=0.25 FairIdx gap | 0.298755 |
| deterministic below random seeds | seed 2, gap very small |
| stochastic std=0.25 below random seeds | none |
| NaN/Inf | no |
| collision | no |

阶段性解读：

1. 在 debug map 上，no_bootstrap + 500 iterations 已经形成稳定阶段性结果。
2. deterministic actor 平均优于 random baseline。
3. stochastic actor with `std_scale=0.25` 在 3 个 seed 上全部优于 random baseline。
4. fairness 提升明显。
5. 没有 NaN/Inf，没有 collision。
6. seed 2 的 deterministic actor 略低于 random baseline，但差距极小。
7. 该结果仍然只是 debug map + 3 seeds 的 sanity result，不能作为最终算法结论。

### 4.2 与 no_bootstrap 200 iterations 对比

可读取的对比文件：

```text
outputs/top_erl_no_bootstrap_200_multiseed/multiseed_summary.csv
```

对比结果：

| metric | 200 iterations | 500 iterations |
|---|---:|---:|
| deterministic reward gap | 0.160370 | 0.172255 |
| stochastic reward gap | -0.008448 | 0.182080 |
| deterministic FairIdx gap | 0.058949 | 0.311756 |
| stochastic FairIdx gap | 0.047950 | 0.298755 |

解读：

1. 500 iterations 相比 200 iterations 更稳定。
2. stochastic std=0.25 的提升尤其明显。
3. fairness gap 的提升明显大于 reward gap。
4. 这说明更长训练对当前最小系统有帮助，但仍需更多地图和更多 seeds 验证。

### 4.3 r400 map, no_bootstrap, 100 iterations

结果目录：

```text
outputs/top_erl_r400_multiseed_100
```

主要结论：

1. 训练和评估流程稳定。
2. 没有 NaN/Inf。
3. eval 阶段没有 collision。
4. actor 的 reward 没有稳定优于 random baseline。
5. FairIdx 有轻微提升趋势，但 reward / throughput 优势不明显。
6. 100 iterations 在 r400 上明显偏短，更像欠训练 sanity run。

平均 gap：

| metric | value |
|---|---:|
| mean deterministic reward gap | -0.000466 |
| mean stochastic std=0.25 reward gap | -0.040870 |
| mean deterministic FairIdx gap | 0.024152 |
| mean stochastic std=0.25 FairIdx gap | 0.023818 |

### 4.4 r400 map, no_bootstrap, 500 iterations

结果目录：

```text
outputs/top_erl_r400_multiseed_500
```

汇总文件：

```text
outputs/top_erl_r400_multiseed_500/multiseed_summary.csv
```

三 seed 主要结果：

| seed | deterministic reward gap | stochastic std=0.25 reward gap | deterministic FairIdx gap | stochastic std=0.25 FairIdx gap | collision | finite |
|---:|---:|---:|---:|---:|---|---|
| 0 | 0.110866 | 0.063609 | -0.053722 | -0.128942 | no | true |
| 1 | 0.026120 | 0.074844 | -0.114871 | -0.230288 | no | true |
| 2 | -0.045513 | 0.020610 | 0.137456 | -0.027202 | no | true |

平均结果：

| metric | value |
|---|---:|
| mean deterministic reward gap | 0.030491 |
| mean stochastic std=0.25 reward gap | 0.053021 |
| mean deterministic FairIdx gap | -0.010379 |
| mean stochastic std=0.25 FairIdx gap | -0.128810 |
| NaN/Inf | no |
| collision | no |

阶段性解读：

1. r400 的 500-iteration run 比 100-iteration 更稳定，reward gap 转为正向。
2. stochastic std=0.25 在 reward 上 3 个 seed 全部高于 random baseline。
3. deterministic actor 在 seed 2 仍低于 random baseline。
4. FairIdx gap 在 r400 上不如 debug map 稳定，说明 r400 更像扩展压力测试。
5. r400 不应被视为当前阶段最终性能证明，但说明训练流程可以扩展到非 debug map。

## 5. 各类输出文件在哪里看

### 5.1 Debug 500-iteration 主要结果

```text
outputs/top_erl_no_bootstrap_500_multiseed/multiseed_summary.csv
outputs/top_erl_no_bootstrap_500_multiseed/report/summary_report.md
outputs/top_erl_no_bootstrap_500_multiseed/report/*.png
```

每个 seed 的训练日志：

```text
outputs/top_erl_no_bootstrap_500_multiseed/seed_0/train_log.csv
outputs/top_erl_no_bootstrap_500_multiseed/seed_1/train_log.csv
outputs/top_erl_no_bootstrap_500_multiseed/seed_2/train_log.csv
```

每个 seed 的 checkpoint：

```text
outputs/top_erl_no_bootstrap_500_multiseed/seed_0/checkpoints/actor.pt
outputs/top_erl_no_bootstrap_500_multiseed/seed_0/checkpoints/critic.pt
outputs/top_erl_no_bootstrap_500_multiseed/seed_0/checkpoints/target_critic.pt
```

seed 1 和 seed 2 目录结构相同。

每个 seed 的 eval 文件：

```text
outputs/top_erl_no_bootstrap_500_multiseed/seed_0/eval/eval_deterministic_actor.csv
outputs/top_erl_no_bootstrap_500_multiseed/seed_0/eval/eval_stochastic_actor_std_0.25.csv
outputs/top_erl_no_bootstrap_500_multiseed/seed_0/eval/eval_random_baseline.csv
outputs/top_erl_no_bootstrap_500_multiseed/seed_0/eval/eval_summary.csv
```

seed 1 和 seed 2 目录结构相同。

### 5.2 r400 500-iteration 主要结果

```text
outputs/top_erl_r400_multiseed_500/multiseed_summary.csv
outputs/top_erl_r400_multiseed_500/seed_0/train_log.csv
outputs/top_erl_r400_multiseed_500/seed_1/train_log.csv
outputs/top_erl_r400_multiseed_500/seed_2/train_log.csv
```

### 5.3 诊断文件

critic target diagnostics：

```text
outputs/critic_target_diagnostics/critic_target_batch_stats.csv
outputs/critic_target_diagnostics/critic_target_summary.csv
outputs/critic_target_diagnostics/critic_target_analysis.md
```

stochastic collision diagnostics：

```text
outputs/top_erl_no_bootstrap_200_multiseed/stochastic_collision_diagnostics/stochastic_collision_eval.csv
outputs/top_erl_no_bootstrap_200_multiseed/stochastic_collision_diagnostics/stochastic_collision_summary.csv
outputs/top_erl_no_bootstrap_200_multiseed/stochastic_collision_diagnostics/stochastic_collision_analysis.md
```

critic calibration diagnostics：

```text
outputs/top_erl_critic_warmup_multiseed/critic_calibration/critic_calibration_candidates.csv
outputs/top_erl_critic_warmup_multiseed/critic_calibration/critic_calibration_summary.csv
outputs/top_erl_critic_warmup_multiseed/critic_calibration/critic_calibration_analysis.md
```

## 6. 图片含义解释

当前所有结果图的标题、坐标轴和图例已改为中文。PNG 文件名仍保持英文，避免路径兼容问题。

### 6.1 训练奖励曲线

文件：

```text
outputs/top_erl_no_bootstrap_500_multiseed/report/training_reward_curve_seed0.png
outputs/top_erl_no_bootstrap_500_multiseed/report/training_reward_curve_seed1.png
outputs/top_erl_no_bootstrap_500_multiseed/report/training_reward_curve_seed2.png
```

含义：

1. 横轴是 training iteration。
2. 纵轴是每次 rollout 的 cumulative reward。
3. 用来观察训练过程是否稳定、是否出现 NaN/Inf、是否有明显 reward 崩溃。
4. 这些曲线不是严格收敛证明，只是 sanity training trace。

### 6.2 训练 loss 曲线

文件：

```text
outputs/top_erl_no_bootstrap_500_multiseed/report/training_loss_curve_seed0.png
outputs/top_erl_no_bootstrap_500_multiseed/report/training_loss_curve_seed1.png
outputs/top_erl_no_bootstrap_500_multiseed/report/training_loss_curve_seed2.png
```

含义：

1. 展示 critic loss 和 actor loss 的变化。
2. warmup 阶段 actor 可能不会更新，因此 actor loss 会缺失或为空。
3. 主要用于检查 loss 是否有限、是否爆炸。
4. 不应仅根据 loss 曲线判断最终策略好坏，因为 actor performance 还取决于 critic ranking。

### 6.3 评估奖励提升柱状图

文件：

```text
outputs/top_erl_no_bootstrap_500_multiseed/report/eval_reward_gap_bar.png
```

含义：

1. 展示 actor 相对 random baseline 的 reward gap。
2. gap 计算方式是：

```text
actor mean reward - random baseline mean reward
```

3. 正值表示 actor 高于 random baseline。
4. debug 500 中，deterministic actor 平均为正，stochastic std=0.25 平均也为正。

### 6.4 评估公平性提升柱状图

文件：

```text
outputs/top_erl_no_bootstrap_500_multiseed/report/eval_fairness_gap_bar.png
```

含义：

1. 展示 actor 相对 random baseline 的 FairIdx gap。
2. FairIdx 越高说明用户间长期服务公平性越好。
3. debug 500 中 fairness gap 明显为正，是当前最明显的收益之一。

### 6.5 评估奖励对比图

文件：

```text
outputs/top_erl_no_bootstrap_500_multiseed/report/eval_reward_comparison.png
```

含义：

1. 直接比较 deterministic actor、stochastic std=0.25 actor 和 random baseline 的 mean cumulative reward。
2. 适合直观看每个 seed 的绝对 reward 水平。

### 6.6 评估公平性对比图

文件：

```text
outputs/top_erl_no_bootstrap_500_multiseed/report/eval_fairness_comparison.png
```

含义：

1. 直接比较不同 policy 的 mean FairIdx。
2. 用于判断 actor 是否主要提升公平性。

### 6.7 轨迹图

在 500-iteration report 目录中未找到独立轨迹图。当前仓库中可找到的轨迹可视化来自早期 sanity 诊断：

```text
outputs/top_erl_sanity_50/figures/trajectory_deterministic_actor.png
outputs/top_erl_sanity_50/figures/trajectory_random_baseline.png
outputs/top_erl_sanity_50/figures/trajectory_comparison.png
```

含义：

1. `trajectory_deterministic_actor.png`
   - 展示 deterministic actor 的 UAV 轨迹、GT 位置、起点和终点。

2. `trajectory_random_baseline.png`
   - 展示 random baseline 的 UAV 轨迹。

3. `trajectory_comparison.png`
   - 同图比较 deterministic actor 与 random baseline 的空间移动行为。

这些轨迹图主要用于检查：

1. UAV 是否基本不动；
2. UAV 是否贴边；
3. UAV 是否重叠或过近；
4. 是否存在异常大跳变；
5. 轨迹是否朝 GT 分布区域移动。

## 7. 当前结果应如何解读

当前阶段可以得出的保守结论：

1. 长时域轨迹接口已经跑通。
2. actor -> trajectory generator -> environment -> replay buffer -> transformer critic -> actor/critic update 这条链路已经可以训练。
3. no_bootstrap target 比 full bootstrap 更稳定。
4. debug map 上 no_bootstrap + 500 iterations 已经产生稳定正向 sanity result。
5. deterministic actor 是当前主评估策略。
6. stochastic actor 在 `std_scale=0.25` 下适合作为辅助评估策略。
7. r400 map 可以跑通，但提升不如 debug 稳定，应作为扩展压力测试。

当前不能得出的结论：

1. 不能说该方法已经优于原始 MADRQN / DRQN。
2. 不能说 critic ranking 已经完全可靠。
3. 不能说该方法在所有地图和用户分布上都稳定。
4. 不能说当前 trajectory generator 已经是最优表达形式。
5. 不能把 debug map + 3 seeds 的 sanity result 当作最终算法性能。

## 8. 当前仍存在的问题

1. Critic Q-ranking 仍然不够可靠
   - 多次 calibration 诊断显示，critic 的 Q-reward correlation 可能较低或为负。
   - no_bootstrap 改善了稳定性，但没有彻底解决 candidate ranking 问题。

2. Deterministic actor 仍有 seed sensitivity
   - debug 500 中 seed 2 deterministic actor 略低于 random baseline。
   - r400 500 中 seed 2 deterministic actor 也低于 random baseline。

3. Stochastic actor 需要 eval-time std_scale
   - 原始 `std_scale=1.0` 采样过激。
   - `std_scale=0.25` 更稳定，但这说明 actor variance 仍需继续诊断。

4. 当前主要结果集中在 debug map
   - debug map 是小规模、轻量、固定或较简单的验证场景。
   - r400 已经开始扩展，但性能优势仍不强。

5. 还没有正式对比原始 MARL 方法
   - 当前只和 random trajectory parameter baseline 比较。
   - 后续需要和 MADRQN / DRQN 或原始离散动作控制方法做更正式对比。

6. 训练规模仍然较小
   - 当前结果主要来自 3 seeds。
   - 还没有更多 seeds、更长训练、更复杂地图的大规模实验。

7. 最新结构升级尚需重新跑正式实验
   - 当前已有报告主要来自 no_bootstrap minimal pipeline。
   - 最近的 trajectory smoothing、critic normalization、reward scaling 等结构稳定化改造需要重新验证。

## 9. 后续建议

优先级建议：

1. 用最新代码重新跑 debug map 的短 sanity test
   - 确认结构稳定化改造后没有破坏已有流程。

2. 重新跑 debug map 的 no_bootstrap 500 multi-seed
   - 判断最新结构是否保持或改善已有结果。

3. 对 r400 继续做更长训练或更多 seeds
   - r400 是当前合适的非 debug 压力测试。

4. 增加更多 map
   - 例如 `r800`、`4ubs`、`6ubs`、`8ubs` 等已注册 map。

5. 继续诊断 critic ranking
   - 尤其是 Q-reward correlation、top-1 regret、candidate ranking。

6. 增加与原始算法的对比
   - 当前只有 random baseline，后续应加入原始 MADRQN / DRQN 的统一评估。

7. 继续优化 trajectory generator
   - 当前应先保持接口稳定。
   - 更复杂的 B-spline、ProDMP 等可以后续再做，不应直接打乱当前可复现实验。

## 10. 复现实验命令

### 10.1 单 seed 训练模板

```powershell
py scripts\train_top_erl_minimal.py `
  --map_id debug `
  --num_iterations 500 `
  --critic_warmup_iterations 20 `
  --critic_updates_per_iteration 4 `
  --actor_update_interval 2 `
  --min_buffer_rollouts_before_actor 20 `
  --target_tau 0.005 `
  --target_mode no_bootstrap `
  --bootstrap_weight 0.0 `
  --output_dir outputs/top_erl_no_bootstrap_500_multiseed/seed_0 `
  --seed 0
```

### 10.2 Deterministic actor evaluation

```powershell
py scripts\eval_top_erl_minimal.py `
  --checkpoint outputs/top_erl_no_bootstrap_500_multiseed/seed_0/checkpoints/actor.pt `
  --map_id debug `
  --num_episodes 20 `
  --deterministic `
  --output_dir outputs/top_erl_no_bootstrap_500_multiseed/seed_0/eval `
  --seed 0
```

### 10.3 Stochastic actor evaluation with std_scale=0.25

```powershell
py scripts\eval_top_erl_minimal.py `
  --checkpoint outputs/top_erl_no_bootstrap_500_multiseed/seed_0/checkpoints/actor.pt `
  --map_id debug `
  --num_episodes 20 `
  --std_scale 0.25 `
  --output_dir outputs/top_erl_no_bootstrap_500_multiseed/seed_0/eval `
  --seed 0
```

### 10.4 Random baseline evaluation

```powershell
py scripts\eval_top_erl_minimal.py `
  --checkpoint outputs/top_erl_no_bootstrap_500_multiseed/seed_0/checkpoints/actor.pt `
  --map_id debug `
  --num_episodes 20 `
  --random_baseline `
  --output_dir outputs/top_erl_no_bootstrap_500_multiseed/seed_0/eval `
  --seed 0
```

### 10.5 生成 debug 500 阶段性报告

```powershell
py scripts\report_top_erl_no_bootstrap_500.py `
  --run_dir outputs/top_erl_no_bootstrap_500_multiseed `
  --compare_run_dir outputs/top_erl_no_bootstrap_200_multiseed
```

### 10.6 查看关键结果文件

```text
outputs/top_erl_no_bootstrap_500_multiseed/multiseed_summary.csv
outputs/top_erl_no_bootstrap_500_multiseed/report/summary_report.md
outputs/top_erl_no_bootstrap_500_multiseed/report/*.png
```

## 11. 输出文件字段含义

### 11.1 `train_log.csv`

常见字段：

| column | meaning |
|---|---|
| iteration | 训练迭代编号 |
| rollout_cumulative_reward | 当前 rollout 的累计 reward |
| TotalThroughput | 当前 rollout 或最终环境 info 中的总吞吐量 |
| FairIdx | Jain fairness index |
| ProbCollision | 碰撞概率 |
| critic_loss | critic update loss |
| actor_loss | actor update loss，warmup 或未更新时可能为空 |
| actor_updated | 当前 iteration 是否执行 actor update |
| critic_updates | 当前 iteration 执行的 critic update 次数 |
| actor_updates | 当前 iteration 执行的 actor update 次数 |
| replay_buffer_size | replay buffer 中已有 rollout 数 |
| target_mode | critic target mode |
| bootstrap_weight | bootstrap 权重 |

### 11.2 `eval_*.csv`

常见字段：

| column | meaning |
|---|---|
| episode | 评估 episode 编号 |
| policy_type | 评估策略类型 |
| std_scale | stochastic actor 的 eval-time std scale |
| cumulative_reward | 该 episode 的累计 reward |
| TotalThroughput | 总吞吐量 |
| FairIdx | 公平性指标 |
| ProbCollision | 碰撞概率 |
| done | trajectory rollout 后是否 done |

### 11.3 `eval_summary.csv`

每次评估会追加 summary 行，通常包括：

1. `policy_type`
2. `num_episodes`
3. mean / std cumulative reward
4. mean / std TotalThroughput
5. mean / std FairIdx
6. mean / std ProbCollision
7. `all_metrics_finite`

### 11.4 `multiseed_summary.csv`

用于跨 seed 汇总。

最重要字段：

| column | meaning |
|---|---|
| deterministic_mean_reward | deterministic actor 平均 reward |
| stochastic_std_0.25_mean_reward | stochastic actor with std_scale=0.25 平均 reward |
| random_mean_reward | random baseline 平均 reward |
| deterministic_reward_gap | deterministic actor reward - random reward |
| stochastic_std_0.25_reward_gap | stochastic std=0.25 reward - random reward |
| deterministic_FairIdx_gap | deterministic actor FairIdx - random FairIdx |
| stochastic_std_0.25_FairIdx_gap | stochastic std=0.25 FairIdx - random FairIdx |
| finite_check | 是否所有关键指标有限 |

## 12. 最终阶段性结论

当前项目已经完成从原 UAV-BS 离散单步控制到 TOP-ERL-inspired 长时域轨迹 actor-critic 原型的阶段性迁移。

当前最可靠的已有结果是：

```text
debug map
target_mode = no_bootstrap
bootstrap_weight = 0.0
num_iterations = 500
seeds = 0, 1, 2
main eval = deterministic actor
auxiliary eval = stochastic actor with std_scale = 0.25
baseline = random trajectory parameter
```

在该设置下：

1. deterministic actor 平均 reward gap 为 `0.172255`。
2. stochastic std=0.25 actor 平均 reward gap 为 `0.182080`。
3. deterministic actor 平均 FairIdx gap 为 `0.311756`。
4. stochastic std=0.25 actor 平均 FairIdx gap 为 `0.298755`。
5. 没有 NaN/Inf。
6. 没有 collision。
7. stochastic std=0.25 在 3 个 seed 上全部高于 random baseline。
8. deterministic actor 在 seed 2 略低于 random baseline，但差距极小。

因此，当前系统已经具备稳定的阶段性 sanity result，适合继续向更多地图、更长训练、更多 seeds 和正式 baseline 对比扩展。

但这仍然不是最终算法结论。后续必须继续验证：

1. 最新结构稳定化改造后的正式结果；
2. r400 / r800 / 多 UAV 地图上的稳定性；
3. critic Q-ranking 是否真正改善；
4. 与原始 MADRQN / DRQN 的正式对比；
5. 更多 seeds 下的统计稳定性。
