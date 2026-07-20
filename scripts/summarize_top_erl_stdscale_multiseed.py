from pathlib import Path
import argparse
import csv
import math


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize completed TOP-ERL minimal multi-seed runs with a chosen stochastic eval policy."
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--stochastic_policy_type", default="stochastic_actor_std_0.25")
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value):
    if value in ("", None):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def latest_policy(rows, policy_type):
    matches = [row for row in rows if row.get("policy_type") == policy_type]
    if not matches:
        raise ValueError(f"Missing policy_type={policy_type}")
    return matches[-1]


def numeric_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def last_nonempty_numeric(rows, key):
    values = numeric_values(rows, key)
    return values[-1] if values else ""


def row_is_finite(row, skip_keys=None):
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


def collect_seed(output_root, seed, stochastic_policy_type):
    seed_dir = output_root / f"seed_{seed}"
    train_rows = read_csv(seed_dir / "train_log.csv")
    eval_rows = read_csv(seed_dir / "eval" / "eval_summary.csv")

    final_train = train_rows[-1]
    deterministic = latest_policy(eval_rows, "deterministic_actor")
    stochastic = latest_policy(eval_rows, stochastic_policy_type)
    random = latest_policy(eval_rows, "random_baseline")

    det_reward = float(deterministic["mean_cumulative_reward"])
    stoch_reward = float(stochastic["mean_cumulative_reward"])
    random_reward = float(random["mean_cumulative_reward"])
    det_throughput = float(deterministic["mean_TotalThroughput"])
    stoch_throughput = float(stochastic["mean_TotalThroughput"])
    random_throughput = float(random["mean_TotalThroughput"])
    det_fair = float(deterministic["mean_FairIdx"])
    stoch_fair = float(stochastic["mean_FairIdx"])
    random_fair = float(random["mean_FairIdx"])

    row = {
        "seed": seed,
        "target_mode": final_train.get("target_mode", ""),
        "bootstrap_weight": to_float(final_train.get("bootstrap_weight", "")),
        "final_training_reward": to_float(final_train["rollout_cumulative_reward"]),
        "final_TotalThroughput": to_float(final_train["TotalThroughput"]),
        "final_FairIdx": to_float(final_train["FairIdx"]),
        "final_ProbCollision": to_float(final_train["ProbCollision"]),
        "final_critic_loss": to_float(final_train["critic_loss"]),
        "final_actor_loss": to_float(final_train.get("actor_loss", "")),
        "last_nonempty_actor_loss": last_nonempty_numeric(train_rows, "actor_loss"),
        "deterministic_mean_reward": det_reward,
        "stochastic_std_0.25_mean_reward": stoch_reward,
        "random_mean_reward": random_reward,
        "deterministic_reward_gap": det_reward - random_reward,
        "stochastic_std_0.25_reward_gap": stoch_reward - random_reward,
        "deterministic_mean_TotalThroughput": det_throughput,
        "stochastic_std_0.25_mean_TotalThroughput": stoch_throughput,
        "random_mean_TotalThroughput": random_throughput,
        "deterministic_mean_FairIdx": det_fair,
        "stochastic_std_0.25_mean_FairIdx": stoch_fair,
        "random_mean_FairIdx": random_fair,
        "deterministic_FairIdx_gap": det_fair - random_fair,
        "stochastic_std_0.25_FairIdx_gap": stoch_fair - random_fair,
        "deterministic_mean_ProbCollision": to_float(deterministic["mean_ProbCollision"]),
        "stochastic_std_0.25_mean_ProbCollision": to_float(stochastic["mean_ProbCollision"]),
        "random_mean_ProbCollision": to_float(random["mean_ProbCollision"]),
    }
    row["finite_check"] = (
        all(row_is_finite(train_row, skip_keys={"actor_updated", "target_mode"}) for train_row in train_rows)
        and all(row_is_finite(eval_row, skip_keys={"policy_type"}) for eval_row in eval_rows)
    )
    return row


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    rows = [collect_seed(output_root, seed, args.stochastic_policy_type) for seed in args.seeds]
    summary_path = output_root / "multiseed_summary.csv"
    write_summary(summary_path, rows)

    print(f"multiseed_summary: {summary_path}")
    for row in rows:
        print(
            f"seed {row['seed']}: "
            f"det_reward={row['deterministic_mean_reward']:.6f}, "
            f"stoch025_reward={row['stochastic_std_0.25_mean_reward']:.6f}, "
            f"random_reward={row['random_mean_reward']:.6f}, "
            f"det_gap={row['deterministic_reward_gap']:.6f}, "
            f"stoch025_gap={row['stochastic_std_0.25_reward_gap']:.6f}, "
            f"finite={row['finite_check']}"
        )

    mean_det_gap = sum(row["deterministic_reward_gap"] for row in rows) / len(rows)
    mean_stoch_gap = sum(row["stochastic_std_0.25_reward_gap"] for row in rows) / len(rows)
    mean_det_fair_gap = sum(row["deterministic_FairIdx_gap"] for row in rows) / len(rows)
    mean_stoch_fair_gap = sum(row["stochastic_std_0.25_FairIdx_gap"] for row in rows) / len(rows)
    print(f"mean deterministic reward gap: {mean_det_gap}")
    print(f"mean stochastic std 0.25 reward gap: {mean_stoch_gap}")
    print(f"mean deterministic FairIdx gap: {mean_det_fair_gap}")
    print(f"mean stochastic std 0.25 FairIdx gap: {mean_stoch_fair_gap}")
    print(f"all finite: {all(row['finite_check'] for row in rows)}")


if __name__ == "__main__":
    main()
