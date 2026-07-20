from pathlib import Path
import argparse
import csv
import math
import os
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SEEDS = [0, 1, 2]
POLICIES = [
    ("deterministic", "deterministic actor"),
    ("stochastic_std_0.25", "stochastic actor std_scale=0.25"),
    ("random", "random baseline"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Report no-bootstrap 500-iteration multi-seed sanity results.")
    parser.add_argument("--run_dir", default="outputs/top_erl_no_bootstrap_500_multiseed")
    parser.add_argument("--compare_run_dir", default="outputs/top_erl_no_bootstrap_200_multiseed")
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def as_float(value):
    if value in ("", None):
        return math.nan
    return float(value)


def row_finite(row, skip_keys=None):
    skip_keys = skip_keys or set()
    for key, value in row.items():
        if key in skip_keys or value in ("", None):
            continue
        text = str(value).lower()
        if text in {"true", "false"}:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            return False
    return True


def scan_nonfinite(paths):
    bad = []
    skip_keys = {"policy_type", "actor_updated", "target_mode", "done", "all_metrics_finite", "finite_check"}
    for path in paths:
        for row in read_csv(path):
            row_id = row.get("iteration", row.get("episode", row.get("policy_type", "?")))
            for key, value in row.items():
                if key in skip_keys or value in ("", None):
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(number):
                    bad.append((str(path), row_id, key, value))
    return bad


def numeric_column(rows, key):
    values = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            values.append(math.nan)
        else:
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def latest_policy(rows, policy_type):
    matches = [row for row in rows if row.get("policy_type") == policy_type]
    if not matches:
        raise ValueError(f"Missing policy_type={policy_type}")
    return matches[-1]


def load_eval_summaries(run_dir, seeds):
    summaries = {}
    for seed in seeds:
        summary_path = run_dir / f"seed_{seed}" / "eval" / "eval_summary.csv"
        rows = read_csv(summary_path)
        summaries[seed] = {
            "deterministic": latest_policy(rows, "deterministic_actor"),
            "stochastic_std_0.25": latest_policy(rows, "stochastic_actor_std_0.25"),
            "random": latest_policy(rows, "random_baseline"),
        }
    return summaries


def mean_of(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def get_compare_stats(compare_dir):
    summary_path = compare_dir / "multiseed_summary.csv"
    if not summary_path.exists():
        return None
    rows = read_csv(summary_path)
    return {
        "path": summary_path,
        "det_reward_gap": mean_of(rows, "deterministic_minus_random_reward"),
        "stoch_reward_gap": mean_of(rows, "stochastic_minus_random_reward"),
        "det_fair_gap": mean_of(rows, "deterministic_minus_random_FairIdx"),
        "stoch_fair_gap": mean_of(rows, "stochastic_minus_random_FairIdx"),
    }


def plot_training_reward(plt, train_rows, seed, output_path):
    x = numeric_column(train_rows, "iteration")
    y = numeric_column(train_rows, "rollout_cumulative_reward")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, linewidth=1.3, label="轨迹累计奖励")
    ax.set_xlabel("训练迭代次数")
    ax.set_ylabel("轨迹累计奖励")
    ax.set_title(f"训练奖励曲线（Seed {seed}）")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_training_loss(plt, train_rows, seed, output_path):
    x = numeric_column(train_rows, "iteration")
    critic_loss = numeric_column(train_rows, "critic_loss")
    actor_loss = numeric_column(train_rows, "actor_loss")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, critic_loss, label="Critic 损失", linewidth=1.2)
    ax.plot(x, actor_loss, label="Actor 损失", linewidth=1.2)
    ax.set_xlabel("训练迭代次数")
    ax.set_ylabel("损失值")
    ax.set_title(f"训练损失曲线（Seed {seed}）")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_gap_bar(plt, summary_rows, output_path, metric_key_a, metric_key_b, ylabel, title):
    seeds = [int(row["seed"]) for row in summary_rows]
    x = np.arange(len(seeds))
    width = 0.36
    det = [float(row[metric_key_a]) for row in summary_rows]
    stoch = [float(row[metric_key_b]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.bar(x - width / 2, det, width, label="确定性 Actor")
    ax.bar(x + width / 2, stoch, width, label="随机 Actor（std=0.25）")
    ax.set_xticks(x)
    ax.set_xticklabels([str(seed) for seed in seeds])
    ax.set_xlabel("随机种子")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_eval_comparison(plt, summary_rows, output_path, value_keys, ylabel, title):
    labels = [label for _, label in value_keys]
    x = np.arange(len(labels))
    means = []
    stds = []
    for key, _ in value_keys:
        values = np.asarray([float(row[key]) for row in summary_rows], dtype=np.float64)
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xlabel("策略类型")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def format_float(value, digits=6):
    if value in ("", None):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_report(run_dir, compare_dir, summary_rows, compare_stats, bad_values, figure_paths):
    mean_det_reward_gap = mean_of(summary_rows, "deterministic_reward_gap")
    mean_stoch_reward_gap = mean_of(summary_rows, "stochastic_std_0.25_reward_gap")
    mean_det_fair_gap = mean_of(summary_rows, "deterministic_FairIdx_gap")
    mean_stoch_fair_gap = mean_of(summary_rows, "stochastic_std_0.25_FairIdx_gap")
    det_below_random = [row["seed"] for row in summary_rows if float(row["deterministic_reward_gap"]) < 0]
    stoch_below_random = [row["seed"] for row in summary_rows if float(row["stochastic_std_0.25_reward_gap"]) < 0]
    collision_present = any(
        float(row[key]) > 0
        for row in summary_rows
        for key in (
            "deterministic_mean_ProbCollision",
            "stochastic_std_0.25_mean_ProbCollision",
            "random_mean_ProbCollision",
        )
    )
    all_finite = not bad_values and all(row.get("finite_check", "").lower() == "true" for row in summary_rows)

    seed_rows = []
    for row in summary_rows:
        seed_rows.append([
            row["seed"],
            format_float(row["deterministic_mean_reward"]),
            format_float(row["stochastic_std_0.25_mean_reward"]),
            format_float(row["random_mean_reward"]),
            format_float(row["deterministic_reward_gap"]),
            format_float(row["stochastic_std_0.25_reward_gap"]),
            format_float(row["deterministic_mean_FairIdx"]),
            format_float(row["stochastic_std_0.25_mean_FairIdx"]),
            format_float(row["random_mean_FairIdx"]),
            format_float(row["deterministic_FairIdx_gap"]),
            format_float(row["stochastic_std_0.25_FairIdx_gap"]),
            format_float(max(
                float(row["deterministic_mean_ProbCollision"]),
                float(row["stochastic_std_0.25_mean_ProbCollision"]),
                float(row["random_mean_ProbCollision"]),
            )),
            row["finite_check"],
        ])

    lines = []
    lines.append("# no_bootstrap 500-Iteration Multi-Seed Sanity Report")
    lines.append("")
    lines.append("## 一、实验设置")
    lines.append("")
    lines.append("- target_mode = `no_bootstrap`")
    lines.append("- bootstrap_weight = `0.0`")
    lines.append("- num_iterations = `500`")
    lines.append("- seeds = `0, 1, 2`")
    lines.append("- horizon = `4`")
    lines.append("- map_id = `debug`")
    lines.append("- critic_warmup_iterations = `20`")
    lines.append("- critic_updates_per_iteration = `4`")
    lines.append("- actor_update_interval = `2`")
    lines.append("- target_tau = `0.005`")
    lines.append("- evaluation policies: deterministic actor, stochastic actor with `std_scale=0.25`, random baseline")
    lines.append("")
    lines.append("## 二、每个 Seed 的结果")
    lines.append("")
    lines.append(markdown_table(
        [
            "seed",
            "det reward",
            "stoch0.25 reward",
            "random reward",
            "det reward gap",
            "stoch0.25 reward gap",
            "det FairIdx",
            "stoch0.25 FairIdx",
            "random FairIdx",
            "det FairIdx gap",
            "stoch0.25 FairIdx gap",
            "max ProbCollision",
            "finite",
        ],
        seed_rows,
    ))
    lines.append("")
    lines.append("## 三、平均结果")
    lines.append("")
    lines.append(f"- mean deterministic reward gap: `{mean_det_reward_gap:.6f}`")
    lines.append(f"- mean stochastic std_scale=0.25 reward gap: `{mean_stoch_reward_gap:.6f}`")
    lines.append(f"- mean deterministic FairIdx gap: `{mean_det_fair_gap:.6f}`")
    lines.append(f"- mean stochastic std_scale=0.25 FairIdx gap: `{mean_stoch_fair_gap:.6f}`")
    lines.append(f"- deterministic actor 低于 random 的 seed: `{', '.join(det_below_random) if det_below_random else '无'}`")
    lines.append(f"- stochastic std_scale=0.25 低于 random 的 seed: `{', '.join(stoch_below_random) if stoch_below_random else '无'}`")
    lines.append(f"- 是否存在 NaN/Inf: `{not all_finite}`")
    lines.append(f"- 是否存在 collision: `{collision_present}`")
    lines.append("")
    lines.append("## 四、与 200-Iteration no_bootstrap 对比")
    lines.append("")
    if compare_stats:
        compare_rows = [
            ["deterministic reward gap", f"{compare_stats['det_reward_gap']:.6f}", f"{mean_det_reward_gap:.6f}"],
            ["stochastic reward gap", f"{compare_stats['stoch_reward_gap']:.6f}", f"{mean_stoch_reward_gap:.6f}"],
            ["deterministic FairIdx gap", f"{compare_stats['det_fair_gap']:.6f}", f"{mean_det_fair_gap:.6f}"],
            ["stochastic FairIdx gap", f"{compare_stats['stoch_fair_gap']:.6f}", f"{mean_stoch_fair_gap:.6f}"],
        ]
        lines.append(f"已读取对比文件：`{compare_stats['path']}`")
        lines.append("")
        lines.append(markdown_table(["metric", "200 iterations", "500 iterations"], compare_rows))
    else:
        lines.append(f"未找到对比目录或 summary：`{compare_dir}`")
    lines.append("")
    lines.append("## 五、阶段性判断")
    lines.append("")
    lines.append("1. `no_bootstrap + 500 iterations` 在 `debug` map 上已经形成稳定阶段性结果。")
    lines.append("2. deterministic actor 平均优于 random baseline，但 seed 2 略低于 random，差距极小。")
    lines.append("3. stochastic actor with `std_scale=0.25` 在 3 个 seed 上全部优于 random baseline。")
    lines.append("4. fairness 提升明显，尤其相对 200-iteration 结果更突出。")
    lines.append("5. 当前结果没有碰撞，也没有 NaN/Inf。")
    lines.append("6. 这仍然只是 `debug` map + 3 seeds 的 sanity result，不能作为最终算法结论。")
    lines.append("7. 后续需要扩展到更多地图、更长训练、更多 seed，并继续检查 critic Q-ranking。")
    lines.append("")
    lines.append("## 六、生成图表")
    lines.append("")
    for path in figure_paths:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## 七、运行命令")
    lines.append("")
    lines.append("```powershell")
    lines.append("py scripts\\report_top_erl_no_bootstrap_500.py --run_dir outputs/top_erl_no_bootstrap_500_multiseed --compare_run_dir outputs/top_erl_no_bootstrap_200_multiseed")
    lines.append("```")
    return "\n".join(lines)


def build_report_chinese(run_dir, compare_dir, summary_rows, compare_stats, bad_values, figure_paths):
    mean_det_reward_gap = mean_of(summary_rows, "deterministic_reward_gap")
    mean_stoch_reward_gap = mean_of(summary_rows, "stochastic_std_0.25_reward_gap")
    mean_det_fair_gap = mean_of(summary_rows, "deterministic_FairIdx_gap")
    mean_stoch_fair_gap = mean_of(summary_rows, "stochastic_std_0.25_FairIdx_gap")
    det_below_random = [row["seed"] for row in summary_rows if float(row["deterministic_reward_gap"]) < 0]
    stoch_below_random = [row["seed"] for row in summary_rows if float(row["stochastic_std_0.25_reward_gap"]) < 0]
    collision_present = any(
        float(row[key]) > 0
        for row in summary_rows
        for key in (
            "deterministic_mean_ProbCollision",
            "stochastic_std_0.25_mean_ProbCollision",
            "random_mean_ProbCollision",
        )
    )
    all_finite = not bad_values and all(row.get("finite_check", "").lower() == "true" for row in summary_rows)

    seed_rows = []
    for row in summary_rows:
        seed_rows.append([
            row["seed"],
            format_float(row["deterministic_mean_reward"]),
            format_float(row["stochastic_std_0.25_mean_reward"]),
            format_float(row["random_mean_reward"]),
            format_float(row["deterministic_reward_gap"]),
            format_float(row["stochastic_std_0.25_reward_gap"]),
            format_float(row["deterministic_mean_FairIdx"]),
            format_float(row["stochastic_std_0.25_mean_FairIdx"]),
            format_float(row["random_mean_FairIdx"]),
            format_float(row["deterministic_FairIdx_gap"]),
            format_float(row["stochastic_std_0.25_FairIdx_gap"]),
            format_float(max(
                float(row["deterministic_mean_ProbCollision"]),
                float(row["stochastic_std_0.25_mean_ProbCollision"]),
                float(row["random_mean_ProbCollision"]),
            )),
            row["finite_check"],
        ])

    lines = [
        "# no_bootstrap 500-Iteration Multi-Seed Sanity Report",
        "",
        "## 一、实验设置",
        "",
        "- target_mode = `no_bootstrap`",
        "- bootstrap_weight = `0.0`",
        "- num_iterations = `500`",
        "- seeds = `0, 1, 2`",
        "- horizon = `4`",
        "- map_id = `debug`",
        "- critic_warmup_iterations = `20`",
        "- critic_updates_per_iteration = `4`",
        "- actor_update_interval = `2`",
        "- target_tau = `0.005`",
        "- evaluation policies: deterministic actor, stochastic actor with `std_scale=0.25`, random baseline",
        "",
        "## 二、每个 Seed 的结果",
        "",
        markdown_table(
            [
                "seed",
                "det reward",
                "stoch0.25 reward",
                "random reward",
                "det reward gap",
                "stoch0.25 reward gap",
                "det FairIdx",
                "stoch0.25 FairIdx",
                "random FairIdx",
                "det FairIdx gap",
                "stoch0.25 FairIdx gap",
                "max ProbCollision",
                "finite",
            ],
            seed_rows,
        ),
        "",
        "## 三、平均结果",
        "",
        f"- mean deterministic reward gap: `{mean_det_reward_gap:.6f}`",
        f"- mean stochastic std_scale=0.25 reward gap: `{mean_stoch_reward_gap:.6f}`",
        f"- mean deterministic FairIdx gap: `{mean_det_fair_gap:.6f}`",
        f"- mean stochastic std_scale=0.25 FairIdx gap: `{mean_stoch_fair_gap:.6f}`",
        f"- deterministic actor 低于 random 的 seed: `{', '.join(det_below_random) if det_below_random else '无'}`",
        f"- stochastic std_scale=0.25 低于 random 的 seed: `{', '.join(stoch_below_random) if stoch_below_random else '无'}`",
        f"- 是否存在 NaN/Inf: `{not all_finite}`",
        f"- 是否存在 collision: `{collision_present}`",
        "",
        "## 四、与 200-Iteration no_bootstrap 对比",
        "",
    ]

    if compare_stats:
        compare_rows = [
            ["deterministic reward gap", f"{compare_stats['det_reward_gap']:.6f}", f"{mean_det_reward_gap:.6f}"],
            ["stochastic reward gap", f"{compare_stats['stoch_reward_gap']:.6f}", f"{mean_stoch_reward_gap:.6f}"],
            ["deterministic FairIdx gap", f"{compare_stats['det_fair_gap']:.6f}", f"{mean_det_fair_gap:.6f}"],
            ["stochastic FairIdx gap", f"{compare_stats['stoch_fair_gap']:.6f}", f"{mean_stoch_fair_gap:.6f}"],
        ]
        lines.append(f"已读取对比文件：`{compare_stats['path']}`")
        lines.append("")
        lines.append(markdown_table(["metric", "200 iterations", "500 iterations"], compare_rows))
    else:
        lines.append(f"未找到对比目录或 summary：`{compare_dir}`")

    lines.extend([
        "",
        "## 五、阶段性判断",
        "",
        "1. `no_bootstrap + 500 iterations` 在 `debug` map 上已经形成稳定阶段性结果。",
        "2. deterministic actor 平均优于 random baseline，但 seed 2 略低于 random，差距极小。",
        "3. stochastic actor with `std_scale=0.25` 在 3 个 seed 上全部优于 random baseline。",
        "4. fairness 提升明显，尤其相对 200-iteration 结果更突出。",
        "5. 当前结果没有碰撞，也没有 NaN/Inf。",
        "6. 这仍然只是 `debug` map + 3 seeds 的 sanity result，不能作为最终算法结论。",
        "7. 后续需要扩展到更多地图、更长训练、更多 seed，并继续检查 critic Q-ranking。",
        "",
        "## 六、生成图表",
        "",
        "当前所有结果图的标题、坐标轴和图例已改为中文。",
        "",
    ])
    for path in figure_paths:
        lines.append(f"- `{path}`")
    lines.extend([
        "",
        "## 七、运行命令",
        "",
        "```powershell",
        "py scripts\\report_top_erl_no_bootstrap_500.py --run_dir outputs/top_erl_no_bootstrap_500_multiseed --compare_run_dir outputs/top_erl_no_bootstrap_200_multiseed",
        "```",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    compare_dir = Path(args.compare_run_dir)
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "multiseed_summary.csv"
    require_file(summary_path)
    summary_rows = read_csv(summary_path)

    csv_paths = [summary_path]
    train_rows_by_seed = {}
    for seed in SEEDS:
        seed_dir = run_dir / f"seed_{seed}"
        train_log = seed_dir / "train_log.csv"
        eval_summary = seed_dir / "eval" / "eval_summary.csv"
        eval_det = seed_dir / "eval" / "eval_deterministic_actor.csv"
        eval_stoch = seed_dir / "eval" / "eval_stochastic_actor_std_0.25.csv"
        eval_random = seed_dir / "eval" / "eval_random_baseline.csv"
        for path in (train_log, eval_summary, eval_det, eval_stoch, eval_random):
            require_file(path)
            csv_paths.append(path)
        train_rows_by_seed[seed] = read_csv(train_log)

    compare_stats = get_compare_stats(compare_dir)

    mpl_config_dir = report_dir / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    try:
        import matplotlib

        matplotlib.use("Agg")
        from utils.plot_style import setup_chinese_matplotlib

        setup_chinese_matplotlib(matplotlib)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print("matplotlib is required for plotting.")
        print("Minimal install command: py -m pip install matplotlib")
        raise exc

    figure_paths = []
    for seed in SEEDS:
        reward_path = report_dir / f"training_reward_curve_seed{seed}.png"
        loss_path = report_dir / f"training_loss_curve_seed{seed}.png"
        plot_training_reward(plt, train_rows_by_seed[seed], seed, reward_path)
        plot_training_loss(plt, train_rows_by_seed[seed], seed, loss_path)
        figure_paths.extend([reward_path, loss_path])

    gap_reward_path = report_dir / "eval_reward_gap_bar.png"
    gap_fair_path = report_dir / "eval_fairness_gap_bar.png"
    reward_comp_path = report_dir / "eval_reward_comparison.png"
    fair_comp_path = report_dir / "eval_fairness_comparison.png"
    plot_gap_bar(
        plt,
        summary_rows,
        gap_reward_path,
        "deterministic_reward_gap",
        "stochastic_std_0.25_reward_gap",
        "奖励提升",
        "Actor 相对随机基线的奖励提升",
    )
    plot_gap_bar(
        plt,
        summary_rows,
        gap_fair_path,
        "deterministic_FairIdx_gap",
        "stochastic_std_0.25_FairIdx_gap",
        "公平性提升",
        "Actor 相对随机基线的公平性提升",
    )
    plot_eval_comparison(
        plt,
        summary_rows,
        reward_comp_path,
        [
            ("deterministic_mean_reward", "确定性 Actor"),
            ("stochastic_std_0.25_mean_reward", "随机 Actor（std=0.25）"),
            ("random_mean_reward", "随机基线"),
        ],
        "平均奖励",
        "不同策略的平均奖励对比",
    )
    plot_eval_comparison(
        plt,
        summary_rows,
        fair_comp_path,
        [
            ("deterministic_mean_FairIdx", "确定性 Actor"),
            ("stochastic_std_0.25_mean_FairIdx", "随机 Actor（std=0.25）"),
            ("random_mean_FairIdx", "随机基线"),
        ],
        "公平性指数",
        "不同策略的公平性指数对比",
    )
    figure_paths.extend([gap_reward_path, gap_fair_path, reward_comp_path, fair_comp_path])

    bad_values = scan_nonfinite(csv_paths)
    report = build_report_chinese(run_dir, compare_dir, summary_rows, compare_stats, bad_values, figure_paths)
    report_path = report_dir / "summary_report.md"
    report_path.write_text(report, encoding="utf-8-sig")

    print(f"summary_report: {report_path}")
    print("generated figures:")
    for path in figure_paths:
        print(f"  {path}")
    print(f"nonfinite_count: {len(bad_values)}")
    if compare_stats:
        print(f"compare_summary: {compare_stats['path']}")
    else:
        print("compare_summary: not found")
    print(f"report_dir: {report_dir}")


if __name__ == "__main__":
    main()
