# TOP-ERL Minimal 阶段性项目状态

本文档记录当前 TOP-ERL-inspired minimal pipeline 的实现状态、默认实验配置、阶段性结果、已知问题和复现实验命令。当前内容只针对已经完成的最小长时域轨迹控制链路，不代表完整 TOP-ERL 算法。

## 一、当前已经完成的模块

1. `MultiUbsCoverageTrajEnv`

   文件：`envs/mubs_cov/mubs_cov_traj.py`

   该类继承原始 `MultiUbsCoverageEnv`，作为长时域轨迹接口的环境入口。它复用原始多 UAV 通信环境中的 A2G channel、用户调度、干扰、SINR、rate、throughput、fairness、collision penalty 和 reward 逻辑，没有重写通信模型。

2. `step_trajectory(action_seq)`

   文件：`envs/mubs_cov/mubs_cov_traj.py`

   接口接收连续二维动作序列，`action_seq` shape 为 `[H, n_uavs, 2]`。第一版支持 `action_type="displacement"` 和 `action_type="velocity"`。每个内部 slot 更新 UAV 水平位置并裁剪到 `[0, range_pos]`，然后调用原始 `_transmit_data()` 和 `_get_reward()`，并记录 state/action/reward/done/info 等序列。

3. `LinearWaypointTrajectoryGenerator`

   文件：`models/trajectory_generator.py`

   这是最小非可微轨迹生成器。actor 或随机 baseline 输出低维轨迹参数 `w`，shape 为 `[n_uavs, 2]`，生成 `continuous_action_seq`，shape 为 `[H, n_uavs, 2]`。该生成器用于环境交互阶段。

4. `TorchLinearWaypointTrajectoryGenerator`

   文件：`models/trajectory_generator.py`

   这是可微 PyTorch 版本轨迹生成器，用于 actor loss。输入 `current_pos` shape 为 `[batch_size, n_uavs, 2]`，`w` shape 为 `[batch_size, n_uavs, 2]`，输出 `continuous_action_seq` shape 为 `[batch_size, H, n_uavs, 2]`。生成过程中保持 PyTorch 计算图，并使用 norm-based scaling 做单步位移裁剪。

5. `TrajectoryReplayBuffer`

   文件：`buffers/trajectory_replay_buffer.py`

   用于存储 `step_trajectory(...)` 返回的完整 trajectory rollout。每条 rollout 至少包含 `state_seq`、`action_seq`、`reward_seq`、`done_seq`、`info_seq`，可选保存 `obs_seq`。当前已支持采样固定长度 segment，并返回 `state_subseq`，shape 为 `[batch_size, segment_length + 1, state_dim]`，用于严格 N-step target。

6. `TrajectoryTransformerCritic`

   文件：`models/transformer_critic.py`

   最小 Transformer critic。输入为 `start_state`，shape `[batch_size, state_dim]`，和 `action_subseq`，shape `[batch_size, segment_length, n_uavs, 2]`。每个 slot 的多 UAV 动作被展平为 action token，state 作为第一个 token，加入 positional encoding 并使用 causal mask。输出 `state_value` 和 `prefix_q_values`。

7. `critic_loss / target modes`

   文件：`models/critic_loss.py`

   当前实现 N-step critic loss，并支持 target mode：

   - `bootstrapped_n_step`：reward prefix + target critic bootstrap；
   - `no_bootstrap`：只使用 reward prefix return；
   - `weighted_bootstrap`：reward prefix + `bootstrap_weight` 加权 bootstrap。

   当前阶段性实验发现 `no_bootstrap` 更稳定，因此作为当前默认 target mode。

8. `TrajectoryActor`

   文件：`models/trajectory_actor.py`

   最小 trajectory actor。输入为全局状态 `state = env.get_state()`，支持 `[state_dim]` 和 `[batch_size, state_dim]`。输出 diagonal Gaussian policy over trajectory parameter `w`，其中 mean/log_std shape 为 `[batch_size, n_uavs * 2]`，采样后 reshape 为 `[batch_size, n_uavs, 2]`。

9. `actor_loss`

   文件：`models/actor_loss.py`

   最小 actor update loss。actor 从 replay buffer 采样的 `start_state` 生成 `w`，再通过可微轨迹生成器得到 action sequence，并输入 critic。第一版目标为 `actor_loss = -prefix_q_values.mean()`。actor update 时冻结 critic 参数，但不使用 `torch.no_grad()`，以便梯度从 critic 对 action sequence 的输出回传到 actor。

10. `train_top_erl_minimal.py`

    文件：`scripts/train_top_erl_minimal.py`

    最小可运行训练脚本。连接 env、actor、trajectory generator、replay buffer、critic、critic loss 和 actor loss。当前已支持 critic-first 调度、`target_mode`、`bootstrap_weight`、soft target update、训练日志和 checkpoint 保存。

11. `eval_top_erl_minimal.py`

    文件：`scripts/eval_top_erl_minimal.py`

    最小评估脚本。支持加载 actor checkpoint，执行 deterministic actor、stochastic actor、random baseline。当前 stochastic actor 支持 eval-time `--std_scale`，默认 `1.0`，推荐辅助诊断使用 `0.25`。

12. `plot_top_erl_minimal_results.py`

    文件：`scripts/plot_top_erl_minimal_results.py`

    用于单次 sanity run 的训练曲线和评估指标图。主要读取训练日志和评估 CSV，生成 reward、通信指标和 loss 曲线。

13. `plot_top_erl_trajectories.py`

    文件：`scripts/plot_top_erl_trajectories.py`

    用于加载 actor checkpoint，并在 debug map 上可视化 deterministic actor 与 random baseline 的 UAV 轨迹、GT 位置、UAV 起点和终点。

14. `diagnose_top_erl_trajectories.py`

    文件：`scripts/diagnose_top_erl_trajectories.py`

    用于轨迹数值诊断，包括 path length、final displacement、mean/max step length、UAV 间距离、UAV 到最近 GT 距离、boundary hit ratio、finite check 等指标。

15. no_bootstrap multiseed scripts and reports

    相关文件：

    - `scripts/run_top_erl_no_bootstrap_multiseed.py`
    - `scripts/summarize_top_erl_stdscale_multiseed.py`
    - `scripts/report_top_erl_no_bootstrap_500.py`
    - `outputs/top_erl_no_bootstrap_500_multiseed/multiseed_summary.csv`
    - `outputs/top_erl_no_bootstrap_500_multiseed/report/summary_report.md`
    - `outputs/top_erl_no_bootstrap_500_multiseed/report/*.png`

    当前已经完成 no_bootstrap 500-iteration multi-seed sanity experiment，并生成阶段性报告和图表。

## 二、当前默认实验配置

当前建议默认配置如下：

- `map_id = debug`
- `target_mode = no_bootstrap`
- `bootstrap_weight = 0.0`
- `num_iterations = 500`
- `seeds = 0, 1, 2`
- `horizon = 4`
- `segment_length = 4`
- `critic_warmup_iterations = 20`
- `critic_updates_per_iteration = 4`
- `actor_update_interval = 2`
- `target_tau = 0.005`
- main evaluation = deterministic actor
- auxiliary evaluation = stochastic actor with `std_scale=0.25`
- baseline = random trajectory parameter

## 三、当前阶段性结果

主要结果来自：

- `outputs/top_erl_no_bootstrap_500_multiseed/report/summary_report.md`
- `outputs/top_erl_no_bootstrap_500_multiseed/multiseed_summary.csv`

500-iteration no_bootstrap multi-seed sanity result：

- mean deterministic reward gap: `0.172255`
- mean stochastic std_scale=0.25 reward gap: `0.182080`
- mean deterministic FairIdx gap: `0.311756`
- mean stochastic std_scale=0.25 FairIdx gap: `0.298755`
- NaN/Inf: 无
- collision: 无
- deterministic actor 低于 random 的 seed: `seed 2`
- seed 2 deterministic reward gap: `-0.000914`，差距极小
- stochastic actor with `std_scale=0.25` 在 3 个 seed 上全部高于 random baseline

与 200-iteration no_bootstrap 结果相比：

| metric | 200 iterations | 500 iterations |
| --- | ---: | ---: |
| deterministic reward gap | 0.160370 | 0.172255 |
| stochastic reward gap | -0.008448 | 0.182080 |
| deterministic FairIdx gap | 0.058949 | 0.311756 |
| stochastic FairIdx gap | 0.047950 | 0.298755 |

当前结论限制：

该结果只是在 `debug` map 和 3 个 seeds 上的 sanity result，不能作为最终算法结论。它说明当前 minimal pipeline 已经能稳定跑通，并且 no_bootstrap target 在该小规模设置下比 full bootstrap 更适合作为下一阶段默认 target mode。

## 四、当前仍存在的问题

1. critic Q-ranking 仍然不够可靠。

   之前的 critic calibration / Q-ranking 诊断显示，critic predicted Q 与真实 reward 的相关性仍然较弱。当前 no_bootstrap 改善了稳定性，但没有彻底解决 Q-ranking 校准问题。

2. deterministic actor 虽然整体稳定，但 seed 2 略低于 random。

   在 500-iteration no_bootstrap 结果中，seed 2 deterministic actor 的 reward gap 为 `-0.000914`。差距很小，但仍说明 deterministic extraction 还需要更多 seed 和地图验证。

3. stochastic actor 需要 eval-time std_scale。

   原始 stochastic sampling `std_scale=1.0` 曾出现采样过激和碰撞风险。当前建议只把 stochastic actor with `std_scale=0.25` 作为辅助诊断策略。

4. 当前只在 debug map 验证。

   `debug` map 很轻量，不能代表 hotspot / dense hotspot 等更复杂用户分布。

5. 轨迹生成器仍是 linear waypoint。

   当前使用的是最小 linear waypoint trajectory generator，不是 B-spline、DMP 或 ProDMP。

6. 还没有和原始 MADRQN / DRQN 做正式对比。

   当前 baseline 主要是 random trajectory parameter，尚未形成与原仓库训练算法的正式对照实验。

7. 还没有复杂地图、多用户分布、多 seed 大规模实验。

   当前实验只是开发期 sanity check，不足以支撑论文级结论。

## 五、下一阶段建议

优先级建议：

1. 扩展到其他 map，例如 `hotspot`、`dense hotspot` 或仓库已有的更复杂地图配置。
2. 增加 seeds，从 3 个 seeds 扩展到更多随机种子。
3. 增加训练 iteration，例如继续做 1000-iteration 或更长训练的 sanity check。
4. 做和 random baseline、原始 MADRQN 的对比。
5. 继续诊断 critic Q-ranking，尤其是 candidate ranking、Q-reward correlation、target calibration。
6. 后续再考虑更复杂 trajectory generator，例如 B-spline 或 ProDMP；当前阶段不建议马上引入复杂轨迹参数化。

## 六、复现实验命令

### 1. 训练 no_bootstrap 500 multiseed

对每个 seed 分别运行：

```powershell
py scripts\train_top_erl_minimal.py --num_iterations 500 --critic_warmup_iterations 20 --critic_updates_per_iteration 4 --actor_update_interval 2 --min_buffer_rollouts_before_actor 20 --target_tau 0.005 --target_mode no_bootstrap --bootstrap_weight 0.0 --output_dir outputs/top_erl_no_bootstrap_500_multiseed/seed_<seed> --seed <seed>
```

其中 `<seed>` 取 `0`、`1`、`2`。

### 2. deterministic actor evaluation

```powershell
py scripts\eval_top_erl_minimal.py --checkpoint outputs/top_erl_no_bootstrap_500_multiseed/seed_<seed>/checkpoints/actor.pt --num_episodes 20 --deterministic --output_dir outputs/top_erl_no_bootstrap_500_multiseed/seed_<seed>/eval --seed <seed>
```

### 3. stochastic actor evaluation with std_scale=0.25

```powershell
py scripts\eval_top_erl_minimal.py --checkpoint outputs/top_erl_no_bootstrap_500_multiseed/seed_<seed>/checkpoints/actor.pt --num_episodes 20 --std_scale 0.25 --output_dir outputs/top_erl_no_bootstrap_500_multiseed/seed_<seed>/eval --seed <seed>
```

### 4. random baseline evaluation

```powershell
py scripts\eval_top_erl_minimal.py --checkpoint outputs/top_erl_no_bootstrap_500_multiseed/seed_<seed>/checkpoints/actor.pt --num_episodes 20 --random_baseline --output_dir outputs/top_erl_no_bootstrap_500_multiseed/seed_<seed>/eval --seed <seed>
```

### 5. 生成 multi-seed 汇总表

```powershell
py scripts\summarize_top_erl_stdscale_multiseed.py --output_root outputs/top_erl_no_bootstrap_500_multiseed --seeds 0 1 2 --stochastic_policy_type stochastic_actor_std_0.25
```

### 6. 生成阶段性报告和图表

```powershell
py scripts\report_top_erl_no_bootstrap_500.py --run_dir outputs/top_erl_no_bootstrap_500_multiseed --compare_run_dir outputs/top_erl_no_bootstrap_200_multiseed
```

### 7. 查看结果文件路径

主要查看：

```text
outputs/top_erl_no_bootstrap_500_multiseed/multiseed_summary.csv
outputs/top_erl_no_bootstrap_500_multiseed/report/summary_report.md
outputs/top_erl_no_bootstrap_500_multiseed/report/*.png
```

## 七、文件路径清单

### 核心汇总与报告

- `outputs/top_erl_no_bootstrap_500_multiseed/multiseed_summary.csv`
- `outputs/top_erl_no_bootstrap_500_multiseed/report/summary_report.md`
- `outputs/top_erl_no_bootstrap_500_multiseed/report/training_reward_curve_seed0.png`
- `outputs/top_erl_no_bootstrap_500_multiseed/report/training_reward_curve_seed1.png`
- `outputs/top_erl_no_bootstrap_500_multiseed/report/training_reward_curve_seed2.png`
- `outputs/top_erl_no_bootstrap_500_multiseed/report/training_loss_curve_seed0.png`
- `outputs/top_erl_no_bootstrap_500_multiseed/report/training_loss_curve_seed1.png`
- `outputs/top_erl_no_bootstrap_500_multiseed/report/training_loss_curve_seed2.png`
- `outputs/top_erl_no_bootstrap_500_multiseed/report/eval_reward_gap_bar.png`
- `outputs/top_erl_no_bootstrap_500_multiseed/report/eval_fairness_gap_bar.png`
- `outputs/top_erl_no_bootstrap_500_multiseed/report/eval_reward_comparison.png`
- `outputs/top_erl_no_bootstrap_500_multiseed/report/eval_fairness_comparison.png`

### 每个 seed 的训练日志

- `outputs/top_erl_no_bootstrap_500_multiseed/seed_0/train_log.csv`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_1/train_log.csv`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_2/train_log.csv`

### 每个 seed 的 checkpoint

- `outputs/top_erl_no_bootstrap_500_multiseed/seed_0/checkpoints/actor.pt`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_0/checkpoints/critic.pt`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_0/checkpoints/target_critic.pt`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_1/checkpoints/actor.pt`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_1/checkpoints/critic.pt`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_1/checkpoints/target_critic.pt`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_2/checkpoints/actor.pt`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_2/checkpoints/critic.pt`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_2/checkpoints/target_critic.pt`

### 每个 seed 的评估结果

- `outputs/top_erl_no_bootstrap_500_multiseed/seed_0/eval/eval_deterministic_actor.csv`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_0/eval/eval_stochastic_actor_std_0.25.csv`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_0/eval/eval_random_baseline.csv`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_0/eval/eval_summary.csv`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_1/eval/eval_deterministic_actor.csv`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_1/eval/eval_stochastic_actor_std_0.25.csv`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_1/eval/eval_random_baseline.csv`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_1/eval/eval_summary.csv`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_2/eval/eval_deterministic_actor.csv`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_2/eval/eval_stochastic_actor_std_0.25.csv`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_2/eval/eval_random_baseline.csv`
- `outputs/top_erl_no_bootstrap_500_multiseed/seed_2/eval/eval_summary.csv`

## 八、当前阶段建议默认配置

当前阶段建议：

- 训练 target 默认使用 `no_bootstrap`。
- 主评估使用 deterministic actor。
- 辅助评估使用 stochastic actor with `std_scale=0.25`。
- random trajectory parameter 继续作为轻量 baseline。

该默认配置适用于下一阶段扩展地图、增加 seed、增加训练时长和准备阶段性汇报；仍不应被视为最终算法设置。
