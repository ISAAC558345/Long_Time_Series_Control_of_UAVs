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


def parse_args():
    parser = argparse.ArgumentParser(description="Plot minimal TOP-ERL sanity run results.")
    parser.add_argument("--run_dir", default="outputs/top_erl_sanity_50")
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def as_float(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def latest_summary(rows, policy_type):
    matches = [row for row in rows if row["policy_type"] == policy_type]
    if not matches:
        raise ValueError(f"No eval summary row found for policy_type={policy_type}.")
    return matches[-1]


def scan_nonfinite(csv_paths):
    skip_keys = {"policy_type", "done", "all_metrics_finite"}
    bad = []
    for path in csv_paths:
        for row in read_csv(path):
            row_id = row.get("iteration", row.get("episode", row.get("policy_type", "?")))
            for key, value in row.items():
                if key in skip_keys or value in ("", None):
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    bad.append((str(path), row_id, key, value))
    return bad


def plot_training_reward(plt, rows, output_path):
    iterations = as_float(rows, "iteration")
    reward = as_float(rows, "rollout_cumulative_reward")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iterations, reward, marker="o", linewidth=1.5, markersize=3, label="轨迹累计奖励")
    ax.set_xlabel("训练迭代次数")
    ax.set_ylabel("轨迹累计奖励")
    ax.set_title("训练奖励曲线")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_training_metrics(plt, rows, output_path):
    iterations = as_float(rows, "iteration")
    metrics = [
        ("TotalThroughput", "总吞吐量"),
        ("FairIdx", "公平性指数"),
        ("ProbCollision", "碰撞概率"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(7, 8), sharex=True)
    for ax, (key, label) in zip(axes, metrics):
        ax.plot(iterations, as_float(rows, key), marker="o", linewidth=1.5, markersize=3)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("训练迭代次数")
    fig.suptitle("训练通信指标曲线")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_training_loss(plt, rows, output_path):
    iterations = as_float(rows, "iteration")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iterations, as_float(rows, "critic_loss"), label="Critic 损失", linewidth=1.5)
    ax.plot(iterations, as_float(rows, "actor_loss"), label="Actor 损失", linewidth=1.5)
    ax.set_xlabel("训练迭代次数")
    ax.set_ylabel("损失值")
    ax.set_title("训练损失曲线")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_eval_reward(plt, summaries, output_path):
    labels = ["确定性 Actor", "随机基线"]
    policy_keys = ["deterministic_actor", "random_baseline"]
    means = [float(summaries[key]["mean_cumulative_reward"]) for key in policy_keys]
    stds = [float(summaries[key]["std_cumulative_reward"]) for key in policy_keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=stds, capsize=5)
    ax.set_xlabel("策略类型")
    ax.set_ylabel("平均奖励")
    ax.set_title("评估奖励对比")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_eval_metrics(plt, summaries, output_path):
    policies = [("deterministic_actor", "确定性 Actor"), ("random_baseline", "随机基线")]
    metrics = [
        ("总吞吐量", "mean_TotalThroughput", "std_TotalThroughput"),
        ("公平性指数", "mean_FairIdx", "std_FairIdx"),
        ("碰撞概率", "mean_ProbCollision", "std_ProbCollision"),
    ]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (policy, label) in enumerate(policies):
        means = [float(summaries[policy][mean_key]) for _, mean_key, _ in metrics]
        stds = [float(summaries[policy][std_key]) for _, _, std_key in metrics]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=4, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _, _ in metrics])
    ax.set_title("评估通信指标对比")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_summary(final_train_row, summaries, bad_values, figure_paths):
    print("final training iteration:")
    print(f"  reward: {final_train_row['rollout_cumulative_reward']}")
    print(f"  TotalThroughput: {final_train_row['TotalThroughput']}")
    print(f"  FairIdx: {final_train_row['FairIdx']}")
    print(f"  ProbCollision: {final_train_row['ProbCollision']}")
    print(f"  critic_loss: {final_train_row['critic_loss']}")
    print(f"  actor_loss: {final_train_row['actor_loss']}")

    for policy in ("deterministic_actor", "random_baseline"):
        summary = summaries[policy]
        print(f"{policy} eval summary:")
        print(f"  mean_cumulative_reward: {summary['mean_cumulative_reward']}")
        print(f"  std_cumulative_reward: {summary['std_cumulative_reward']}")
        print(f"  mean_TotalThroughput: {summary['mean_TotalThroughput']}")
        print(f"  std_TotalThroughput: {summary['std_TotalThroughput']}")
        print(f"  mean_FairIdx: {summary['mean_FairIdx']}")
        print(f"  std_FairIdx: {summary['std_FairIdx']}")
        print(f"  mean_ProbCollision: {summary['mean_ProbCollision']}")
        print(f"  std_ProbCollision: {summary['std_ProbCollision']}")

    print(f"NaN or Inf present: {bool(bad_values)}")
    if bad_values:
        print(f"non-finite values: {bad_values}")

    print("generated figures:")
    for path in figure_paths:
        print(f"  {path}")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    train_log = run_dir / "train_log.csv"
    eval_det = run_dir / "eval" / "eval_deterministic_actor.csv"
    eval_random = run_dir / "eval" / "eval_random_baseline.csv"
    eval_summary = run_dir / "eval" / "eval_summary.csv"
    figures_dir = run_dir / "figures"

    for path in (train_log, eval_det, eval_random, eval_summary):
        require_file(path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = figures_dir / ".mplconfig"
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

    train_rows = read_csv(train_log)
    summary_rows = read_csv(eval_summary)
    if not train_rows:
        raise ValueError(f"No training rows found in {train_log}")

    summaries = {
        "deterministic_actor": latest_summary(summary_rows, "deterministic_actor"),
        "random_baseline": latest_summary(summary_rows, "random_baseline"),
    }

    figure_paths = [
        figures_dir / "training_reward_curve.png",
        figures_dir / "training_metrics_curve.png",
        figures_dir / "training_loss_curve.png",
        figures_dir / "eval_reward_comparison.png",
        figures_dir / "eval_metrics_comparison.png",
    ]

    plot_training_reward(plt, train_rows, figure_paths[0])
    plot_training_metrics(plt, train_rows, figure_paths[1])
    plot_training_loss(plt, train_rows, figure_paths[2])
    plot_eval_reward(plt, summaries, figure_paths[3])
    plot_eval_metrics(plt, summaries, figure_paths[4])

    bad_values = scan_nonfinite([train_log, eval_det, eval_random, eval_summary])
    print_summary(train_rows[-1], summaries, bad_values, figure_paths)


if __name__ == "__main__":
    main()
