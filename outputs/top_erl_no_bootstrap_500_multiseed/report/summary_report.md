# no_bootstrap 500-Iteration Multi-Seed Sanity Report

## 一、实验设置

- target_mode = `no_bootstrap`
- bootstrap_weight = `0.0`
- num_iterations = `500`
- seeds = `0, 1, 2`
- horizon = `4`
- map_id = `debug`
- critic_warmup_iterations = `20`
- critic_updates_per_iteration = `4`
- actor_update_interval = `2`
- target_tau = `0.005`
- evaluation policies: deterministic actor, stochastic actor with `std_scale=0.25`, random baseline

## 二、每个 Seed 的结果

| seed | det reward | stoch0.25 reward | random reward | det reward gap | stoch0.25 reward gap | det FairIdx | stoch0.25 FairIdx | random FairIdx | det FairIdx gap | stoch0.25 FairIdx gap | max ProbCollision | finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.253737 | 0.214649 | 0.083097 | 0.170640 | 0.131553 | 0.660698 | 0.639298 | 0.341414 | 0.319284 | 0.297884 | 0.000000 | True |
| 1 | 0.432042 | 0.393645 | 0.085005 | 0.347037 | 0.308640 | 0.886327 | 0.807568 | 0.442980 | 0.443347 | 0.364588 | 0.000000 | True |
| 2 | 0.040352 | 0.147314 | 0.041266 | -0.000914 | 0.106048 | 0.499607 | 0.560763 | 0.326972 | 0.172636 | 0.233791 | 0.000000 | True |

## 三、平均结果

- mean deterministic reward gap: `0.172255`
- mean stochastic std_scale=0.25 reward gap: `0.182080`
- mean deterministic FairIdx gap: `0.311756`
- mean stochastic std_scale=0.25 FairIdx gap: `0.298755`
- deterministic actor 低于 random 的 seed: `2`
- stochastic std_scale=0.25 低于 random 的 seed: `无`
- 是否存在 NaN/Inf: `False`
- 是否存在 collision: `False`

## 四、与 200-Iteration no_bootstrap 对比

已读取对比文件：`outputs\top_erl_no_bootstrap_200_multiseed\multiseed_summary.csv`

| metric | 200 iterations | 500 iterations |
| --- | --- | --- |
| deterministic reward gap | 0.160370 | 0.172255 |
| stochastic reward gap | -0.008448 | 0.182080 |
| deterministic FairIdx gap | 0.058949 | 0.311756 |
| stochastic FairIdx gap | 0.047950 | 0.298755 |

## 五、阶段性判断

1. `no_bootstrap + 500 iterations` 在 `debug` map 上已经形成稳定阶段性结果。
2. deterministic actor 平均优于 random baseline，但 seed 2 略低于 random，差距极小。
3. stochastic actor with `std_scale=0.25` 在 3 个 seed 上全部优于 random baseline。
4. fairness 提升明显，尤其相对 200-iteration 结果更突出。
5. 当前结果没有碰撞，也没有 NaN/Inf。
6. 这仍然只是 `debug` map + 3 seeds 的 sanity result，不能作为最终算法结论。
7. 后续需要扩展到更多地图、更长训练、更多 seed，并继续检查 critic Q-ranking。

## 六、生成图表

当前所有结果图的标题、坐标轴和图例已改为中文。

- `outputs\top_erl_no_bootstrap_500_multiseed\report\training_reward_curve_seed0.png`
- `outputs\top_erl_no_bootstrap_500_multiseed\report\training_loss_curve_seed0.png`
- `outputs\top_erl_no_bootstrap_500_multiseed\report\training_reward_curve_seed1.png`
- `outputs\top_erl_no_bootstrap_500_multiseed\report\training_loss_curve_seed1.png`
- `outputs\top_erl_no_bootstrap_500_multiseed\report\training_reward_curve_seed2.png`
- `outputs\top_erl_no_bootstrap_500_multiseed\report\training_loss_curve_seed2.png`
- `outputs\top_erl_no_bootstrap_500_multiseed\report\eval_reward_gap_bar.png`
- `outputs\top_erl_no_bootstrap_500_multiseed\report\eval_fairness_gap_bar.png`
- `outputs\top_erl_no_bootstrap_500_multiseed\report\eval_reward_comparison.png`
- `outputs\top_erl_no_bootstrap_500_multiseed\report\eval_fairness_comparison.png`

## 七、运行命令

```powershell
py scripts\report_top_erl_no_bootstrap_500.py --run_dir outputs/top_erl_no_bootstrap_500_multiseed --compare_run_dir outputs/top_erl_no_bootstrap_200_multiseed
```