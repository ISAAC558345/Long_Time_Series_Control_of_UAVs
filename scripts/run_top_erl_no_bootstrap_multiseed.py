from pathlib import Path
import argparse
import csv
import math
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run no-bootstrap TOP-ERL minimal multi-seed sanity experiment.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--map_id", default="debug")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--segment_length", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--critic_warmup_iterations", type=int, default=20)
    parser.add_argument("--critic_updates_per_iteration", type=int, default=4)
    parser.add_argument("--actor_update_interval", type=int, default=2)
    parser.add_argument("--min_buffer_rollouts_before_actor", type=int, default=20)
    parser.add_argument("--target_tau", type=float, default=0.005)
    parser.add_argument("--num_eval_episodes", type=int, default=20)
    parser.add_argument("--num_calibration_states", type=int, default=20)
    parser.add_argument("--num_calibration_candidates", type=int, default=32)
    parser.add_argument("--output_root", default="outputs/top_erl_no_bootstrap_multiseed")
    return parser.parse_args()


def run_command(command):
    print("running:", " ".join(command))
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = result.stdout.strip()
    if result.returncode != 0:
        print(output)
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
    for line in output.splitlines()[-10:]:
        print(line)


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def latest_policy(rows, policy_type):
    matches = [row for row in rows if row["policy_type"] == policy_type]
    if not matches:
        raise ValueError(f"Missing eval summary for policy_type={policy_type}.")
    return matches[-1]


def numeric_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def last_nonempty_numeric(rows, key):
    values = numeric_values(rows, key)
    return values[-1] if values else ""


def safe_float(value):
    if value in ("", None):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def finite_mapping(row, skip_keys=None):
    skip_keys = skip_keys or set()
    for key, value in row.items():
        if key in skip_keys or value in ("", None):
            continue
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            text = str(value).lower()
            if text in {"true", "false"}:
                continue
            continue
        if not math.isfinite(number):
            return False
    return True


def collect_seed_summary(seed, output_root):
    seed_dir = output_root / f"seed_{seed}"
    train_rows = read_csv(seed_dir / "train_log.csv")
    eval_rows = read_csv(seed_dir / "eval" / "eval_summary.csv")
    calibration_rows = read_csv(output_root / "critic_calibration" / "critic_calibration_summary.csv")
    calibration = next(row for row in calibration_rows if int(row["seed"]) == seed)

    final_train = train_rows[-1]
    deterministic = latest_policy(eval_rows, "deterministic_actor")
    stochastic = latest_policy(eval_rows, "stochastic_actor")
    random = latest_policy(eval_rows, "random_baseline")

    det_reward = float(deterministic["mean_cumulative_reward"])
    stoch_reward = float(stochastic["mean_cumulative_reward"])
    rand_reward = float(random["mean_cumulative_reward"])
    det_fair = float(deterministic["mean_FairIdx"])
    stoch_fair = float(stochastic["mean_FairIdx"])
    rand_fair = float(random["mean_FairIdx"])
    det_throughput = float(deterministic["mean_TotalThroughput"])
    stoch_throughput = float(stochastic["mean_TotalThroughput"])
    rand_throughput = float(random["mean_TotalThroughput"])

    row = dict(
        seed=seed,
        target_mode=final_train.get("target_mode", ""),
        bootstrap_weight=safe_float(final_train.get("bootstrap_weight", "")),
        final_training_reward=safe_float(final_train["rollout_cumulative_reward"]),
        final_TotalThroughput=safe_float(final_train["TotalThroughput"]),
        final_FairIdx=safe_float(final_train["FairIdx"]),
        final_ProbCollision=safe_float(final_train["ProbCollision"]),
        final_critic_loss=safe_float(final_train["critic_loss"]),
        final_actor_loss=safe_float(final_train.get("actor_loss", "")),
        last_nonempty_actor_loss=last_nonempty_numeric(train_rows, "actor_loss"),
        deterministic_mean_reward=det_reward,
        stochastic_mean_reward=stoch_reward,
        random_mean_reward=rand_reward,
        deterministic_mean_TotalThroughput=det_throughput,
        stochastic_mean_TotalThroughput=stoch_throughput,
        random_mean_TotalThroughput=rand_throughput,
        deterministic_mean_FairIdx=det_fair,
        stochastic_mean_FairIdx=stoch_fair,
        random_mean_FairIdx=rand_fair,
        deterministic_mean_ProbCollision=safe_float(deterministic["mean_ProbCollision"]),
        stochastic_mean_ProbCollision=safe_float(stochastic["mean_ProbCollision"]),
        random_mean_ProbCollision=safe_float(random["mean_ProbCollision"]),
        deterministic_minus_random_reward=det_reward - rand_reward,
        stochastic_minus_random_reward=stoch_reward - rand_reward,
        deterministic_minus_random_TotalThroughput=det_throughput - rand_throughput,
        stochastic_minus_random_TotalThroughput=stoch_throughput - rand_throughput,
        deterministic_minus_random_FairIdx=det_fair - rand_fair,
        stochastic_minus_random_FairIdx=stoch_fair - rand_fair,
        q_reward_corr=safe_float(calibration["q_reward_corr_mean"]),
        q_throughput_corr=safe_float(calibration["q_throughput_corr"]),
        q_fairness_corr=safe_float(calibration["q_fairness_corr"]),
        top1_regret=safe_float(calibration["mean_top1_regret"]),
    )
    row["finite_check"] = (
        all(finite_mapping(train_row, skip_keys={"actor_updated", "target_mode"}) for train_row in train_rows)
        and all(finite_mapping(eval_row, skip_keys={"policy_type"}) for eval_row in eval_rows)
        and finite_mapping(calibration)
    )
    return row


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, output_root):
    print(f"multiseed_summary_csv: {output_root / 'multiseed_summary.csv'}")
    print("per-seed no-bootstrap comparison:")
    for row in rows:
        print(
            "seed {seed}: det_reward={deterministic_mean_reward:.6f}, "
            "stoch_reward={stochastic_mean_reward:.6f}, random_reward={random_mean_reward:.6f}, "
            "det_gap={deterministic_minus_random_reward:.6f}, "
            "stoch_gap={stochastic_minus_random_reward:.6f}, "
            "fair_gap={deterministic_minus_random_FairIdx:.6f}, "
            "q_reward_corr={q_reward_corr:.6f}, top1_regret={top1_regret:.6f}, "
            "finite={finite_check}".format(**row)
        )
    mean_det_gap = sum(row["deterministic_minus_random_reward"] for row in rows) / len(rows)
    mean_stoch_gap = sum(row["stochastic_minus_random_reward"] for row in rows) / len(rows)
    mean_fair_gap = sum(row["deterministic_minus_random_FairIdx"] for row in rows) / len(rows)
    all_finite = all(row["finite_check"] for row in rows)
    bad_det_seeds = [row["seed"] for row in rows if row["deterministic_minus_random_reward"] < 0]
    print(f"mean deterministic reward gap: {mean_det_gap}")
    print(f"mean stochastic reward gap: {mean_stoch_gap}")
    print(f"mean deterministic fairness gap: {mean_fair_gap}")
    print(f"all finite: {all_finite}")
    print(f"deterministic below random seeds: {bad_det_seeds}")


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        seed_dir = output_root / f"seed_{seed}"
        checkpoint = seed_dir / "checkpoints" / "actor.pt"
        common_train_args = [
            sys.executable,
            "scripts/train_top_erl_minimal.py",
            "--map_id", args.map_id,
            "--horizon", str(args.horizon),
            "--segment_length", str(args.segment_length),
            "--num_iterations", str(args.num_iterations),
            "--critic_warmup_iterations", str(args.critic_warmup_iterations),
            "--critic_updates_per_iteration", str(args.critic_updates_per_iteration),
            "--actor_update_interval", str(args.actor_update_interval),
            "--min_buffer_rollouts_before_actor", str(args.min_buffer_rollouts_before_actor),
            "--target_tau", str(args.target_tau),
            "--target_mode", "no_bootstrap",
            "--bootstrap_weight", "0.0",
            "--output_dir", str(seed_dir),
            "--seed", str(seed),
        ]
        run_command(common_train_args)

        for extra_args in (
            ["--deterministic"],
            [],
            ["--random_baseline"],
        ):
            run_command([
                sys.executable,
                "scripts/eval_top_erl_minimal.py",
                "--checkpoint", str(checkpoint),
                "--map_id", args.map_id,
                "--horizon", str(args.horizon),
                "--num_episodes", str(args.num_eval_episodes),
                "--output_dir", str(seed_dir / "eval"),
                "--seed", str(seed),
                *extra_args,
            ])

    run_command([
        sys.executable,
        "scripts/analyze_top_erl_critic_calibration.py",
        "--base_dir", str(output_root),
        "--seeds", *[str(seed) for seed in args.seeds],
        "--map_id", args.map_id,
        "--horizon", str(args.horizon),
        "--num_states", str(args.num_calibration_states),
        "--num_candidates", str(args.num_calibration_candidates),
        "--output_dir", str(output_root / "critic_calibration"),
    ])

    rows = [collect_seed_summary(seed, output_root) for seed in args.seeds]
    write_summary(output_root / "multiseed_summary.csv", rows)
    print_summary(rows, output_root)


if __name__ == "__main__":
    main()
