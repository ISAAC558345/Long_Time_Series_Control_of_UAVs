from pathlib import Path
import argparse
import csv
import math
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RUN_CONFIGS = [
    dict(
        label="full_bootstrap",
        run_dir="outputs/top_erl_target_ablation/full_bootstrap/seed_0",
        calibration_dir="outputs/top_erl_target_ablation/full_bootstrap/critic_calibration",
    ),
    dict(
        label="no_bootstrap",
        run_dir="outputs/top_erl_target_ablation/no_bootstrap/seed_0",
        calibration_dir="outputs/top_erl_target_ablation/no_bootstrap/critic_calibration",
    ),
    dict(
        label="weak_bootstrap_025",
        run_dir="outputs/top_erl_target_ablation/weak_bootstrap_025/seed_0",
        calibration_dir="outputs/top_erl_target_ablation/weak_bootstrap_025/critic_calibration",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize target ablation outputs from existing CSV files.")
    parser.add_argument("--output_dir", default="outputs/top_erl_target_ablation")
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value):
    if value in ("", None):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def finite_value(value):
    if value in ("", None):
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return str(value).lower() in {"true", "false"}


def policy_row(eval_rows, policy_type):
    for row in eval_rows:
        if row["policy_type"] == policy_type:
            return row
    raise ValueError(f"Missing policy_type={policy_type} in eval summary.")


def target_row(target_rows, label):
    for row in target_rows:
        if str(row["run_name"]).startswith(f"{label}__"):
            return row
    raise ValueError(f"Missing target diagnostics row for {label}.")


def load_run_summary(config, target_rows):
    run_dir = Path(config["run_dir"])
    train_rows = read_csv(run_dir / "train_log.csv")
    final_train = train_rows[-1]
    eval_rows = read_csv(run_dir / "eval" / "eval_summary.csv")
    calibration_rows = read_csv(Path(config["calibration_dir"]) / "critic_calibration_summary.csv")
    calibration = calibration_rows[0]
    target = target_row(target_rows, config["label"])
    deterministic = policy_row(eval_rows, "deterministic_actor")
    stochastic = policy_row(eval_rows, "stochastic_actor")
    random = policy_row(eval_rows, "random_baseline")

    row = dict(
        target_label=config["label"],
        target_mode=final_train.get("target_mode", target.get("target_mode", "")),
        bootstrap_weight=safe_float(final_train.get("bootstrap_weight", target.get("bootstrap_weight", ""))),
        final_training_reward=safe_float(final_train["rollout_cumulative_reward"]),
        final_TotalThroughput=safe_float(final_train["TotalThroughput"]),
        final_FairIdx=safe_float(final_train["FairIdx"]),
        final_ProbCollision=safe_float(final_train["ProbCollision"]),
        final_critic_loss=safe_float(final_train["critic_loss"]),
        final_actor_loss=safe_float(final_train.get("actor_loss", "")),
        last_nonempty_actor_loss=safe_float(target.get("last_nonempty_actor_loss", "")),
        deterministic_mean_reward=safe_float(deterministic["mean_cumulative_reward"]),
        stochastic_mean_reward=safe_float(stochastic["mean_cumulative_reward"]),
        random_mean_reward=safe_float(random["mean_cumulative_reward"]),
        deterministic_mean_TotalThroughput=safe_float(deterministic["mean_TotalThroughput"]),
        stochastic_mean_TotalThroughput=safe_float(stochastic["mean_TotalThroughput"]),
        random_mean_TotalThroughput=safe_float(random["mean_TotalThroughput"]),
        deterministic_mean_FairIdx=safe_float(deterministic["mean_FairIdx"]),
        stochastic_mean_FairIdx=safe_float(stochastic["mean_FairIdx"]),
        random_mean_FairIdx=safe_float(random["mean_FairIdx"]),
        deterministic_mean_ProbCollision=safe_float(deterministic["mean_ProbCollision"]),
        stochastic_mean_ProbCollision=safe_float(stochastic["mean_ProbCollision"]),
        random_mean_ProbCollision=safe_float(random["mean_ProbCollision"]),
        q_reward_corr=safe_float(calibration["q_reward_corr_mean"]),
        q_throughput_corr=safe_float(calibration["q_throughput_corr"]),
        q_fairness_corr=safe_float(calibration["q_fairness_corr"]),
        top1_regret=safe_float(calibration["mean_top1_regret"]),
        q_target_corr=safe_float(target["q_target_corr_mean"]),
        q_target_mse=safe_float(target["q_target_mse_mean"]),
        bootstrap_reward_ratio=safe_float(target["bootstrap_reward_abs_ratio_mean"]),
    )
    finite_fields = [
        value for key, value in row.items()
        if key not in {"target_label", "target_mode", "final_actor_loss"}
    ]
    row["finite_check"] = all(finite_value(value) for value in finite_fields)
    return row


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def best_label(rows, key, larger_is_better=True):
    valid = [row for row in rows if isinstance(row[key], float) and math.isfinite(row[key])]
    if not valid:
        return "NA"
    selected = max(valid, key=lambda row: row[key]) if larger_is_better else min(valid, key=lambda row: row[key])
    return selected["target_label"]


def write_report(path, rows):
    best_q_corr = best_label(rows, "q_reward_corr", larger_is_better=True)
    best_regret = best_label(rows, "top1_regret", larger_is_better=False)
    best_det_reward = best_label(rows, "deterministic_mean_reward", larger_is_better=True)
    best_det_fair = best_label(rows, "deterministic_mean_FairIdx", larger_is_better=True)
    all_finite = all(row["finite_check"] for row in rows)

    lines = [
        "# Target Ablation Summary",
        "",
        "This report only reads existing CSV outputs. It does not retrain or modify checkpoints.",
        "",
        "| target | mode | weight | det reward | stoch reward | random reward | det FairIdx | Q-reward corr | top-1 regret | Q-target corr | Q-target MSE | bootstrap/reward | finite |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target_label']} | {row['target_mode']} | {row['bootstrap_weight']} | "
            f"{row['deterministic_mean_reward']:.6f} | {row['stochastic_mean_reward']:.6f} | "
            f"{row['random_mean_reward']:.6f} | {row['deterministic_mean_FairIdx']:.6f} | "
            f"{row['q_reward_corr']:.6f} | {row['top1_regret']:.6f} | "
            f"{row['q_target_corr']:.6f} | {row['q_target_mse']:.6f} | "
            f"{row['bootstrap_reward_ratio']:.6f} | {row['finite_check']} |"
        )

    lines.extend([
        "",
        "## Readout",
        "",
        f"- Best Q-reward correlation: {best_q_corr}.",
        f"- Lowest top-1 regret: {best_regret}.",
        f"- Best deterministic actor reward: {best_det_reward}.",
        f"- Best deterministic FairIdx: {best_det_fair}.",
        f"- All summarized metrics finite: {all_finite}.",
        "",
        "## Diagnostic interpretation",
        "",
        "- This is a seed-0 ablation only; it is not a final algorithm conclusion.",
        "- Prefer advancing a target mode to multi-seed only if it improves Q-ranking or actor evaluation without numerical instability.",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    target_rows = read_csv(output_dir / "critic_target_diagnostics" / "critic_target_summary.csv")
    rows = [load_run_summary(config, target_rows) for config in RUN_CONFIGS]

    summary_path = output_dir / "target_ablation_summary.csv"
    report_path = output_dir / "target_ablation_analysis.md"
    write_csv(summary_path, rows)
    write_report(report_path, rows)

    print(f"target_ablation_summary_csv: {summary_path}")
    print(f"target_ablation_report: {report_path}")
    for row in rows:
        print(
            f"{row['target_label']}: det_reward={row['deterministic_mean_reward']:.6f}, "
            f"stoch_reward={row['stochastic_mean_reward']:.6f}, "
            f"q_reward_corr={row['q_reward_corr']:.6f}, "
            f"top1_regret={row['top1_regret']:.6f}, finite={row['finite_check']}"
        )


if __name__ == "__main__":
    main()
