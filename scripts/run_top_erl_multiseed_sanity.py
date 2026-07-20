from pathlib import Path
import argparse
import csv
import math
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run minimal TOP-ERL multi-seed sanity experiments.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--num_eval_episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--segment_length", type=int, default=4)
    parser.add_argument("--map_id", default="debug")
    parser.add_argument("--output_root", default="outputs/top_erl_multiseed")
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
    lines = output.splitlines()
    for line in lines[-8:]:
        print(line)


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def latest_summary(rows, policy_type):
    matches = [row for row in rows if row["policy_type"] == policy_type]
    if not matches:
        raise ValueError(f"No eval summary for policy_type={policy_type}.")
    return matches[-1]


def is_finite_mapping(row, skip_keys):
    for key, value in row.items():
        if key in skip_keys or value in ("", None):
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if not math.isfinite(number):
            return False
    return True


def collect_seed_summary(seed, seed_dir):
    train_rows = read_csv(seed_dir / "train_log.csv")
    eval_summary_rows = read_csv(seed_dir / "eval" / "eval_summary.csv")
    diagnostics_rows = read_csv(seed_dir / "diagnostics" / "trajectory_diagnostics.csv")

    final_train = train_rows[-1]
    det = latest_summary(eval_summary_rows, "deterministic_actor")
    rand = latest_summary(eval_summary_rows, "random_baseline")

    det_reward = float(det["mean_cumulative_reward"])
    rand_reward = float(rand["mean_cumulative_reward"])
    det_throughput = float(det["mean_TotalThroughput"])
    rand_throughput = float(rand["mean_TotalThroughput"])
    det_fairness = float(det["mean_FairIdx"])
    rand_fairness = float(rand["mean_FairIdx"])

    finite = (
        is_finite_mapping(final_train, skip_keys=set())
        and is_finite_mapping(det, skip_keys={"policy_type", "all_metrics_finite"})
        and is_finite_mapping(rand, skip_keys={"policy_type", "all_metrics_finite"})
        and all(row.get("trajectory_finite") == "True" for row in diagnostics_rows)
    )

    return dict(
        seed=seed,
        final_training_reward=float(final_train["rollout_cumulative_reward"]),
        final_TotalThroughput=float(final_train["TotalThroughput"]),
        final_FairIdx=float(final_train["FairIdx"]),
        final_ProbCollision=float(final_train["ProbCollision"]),
        final_critic_loss=float(final_train["critic_loss"]),
        final_actor_loss=float(final_train["actor_loss"]),
        deterministic_mean_cumulative_reward=det_reward,
        deterministic_mean_TotalThroughput=det_throughput,
        deterministic_mean_FairIdx=det_fairness,
        deterministic_mean_ProbCollision=float(det["mean_ProbCollision"]),
        random_mean_cumulative_reward=rand_reward,
        random_mean_TotalThroughput=rand_throughput,
        random_mean_FairIdx=rand_fairness,
        random_mean_ProbCollision=float(rand["mean_ProbCollision"]),
        deterministic_minus_random_reward=det_reward - rand_reward,
        deterministic_minus_random_TotalThroughput=det_throughput - rand_throughput,
        deterministic_minus_random_FairIdx=det_fairness - rand_fairness,
        finite_check=finite,
    )


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("per-seed comparison:")
    for row in rows:
        print(
            "seed {seed}: reward_gap={deterministic_minus_random_reward:.6f}, "
            "throughput_gap={deterministic_minus_random_TotalThroughput:.6f}, "
            "fairness_gap={deterministic_minus_random_FairIdx:.6f}, finite={finite_check}".format(**row)
        )

    mean_reward_gap = sum(row["deterministic_minus_random_reward"] for row in rows) / len(rows)
    mean_throughput_gap = sum(row["deterministic_minus_random_TotalThroughput"] for row in rows) / len(rows)
    mean_fairness_gap = sum(row["deterministic_minus_random_FairIdx"] for row in rows) / len(rows)
    all_finite = all(bool(row["finite_check"]) for row in rows)
    worse_seeds = [row["seed"] for row in rows if row["deterministic_minus_random_reward"] < 0]

    print(f"mean reward gap: {mean_reward_gap}")
    print(f"mean throughput gap: {mean_throughput_gap}")
    print(f"mean fairness gap: {mean_fairness_gap}")
    print(f"all results finite: {all_finite}")
    print(f"seeds with deterministic reward below random: {worse_seeds}")


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    rows = []

    for seed in args.seeds:
        seed_dir = output_root / f"seed_{seed}"
        checkpoint = seed_dir / "checkpoints" / "actor.pt"

        run_command([
            sys.executable,
            "scripts/train_top_erl_minimal.py",
            "--map_id", args.map_id,
            "--horizon", str(args.horizon),
            "--segment_length", str(args.segment_length),
            "--num_iterations", str(args.num_iterations),
            "--output_dir", str(seed_dir),
            "--seed", str(seed),
        ])
        run_command([
            sys.executable,
            "scripts/eval_top_erl_minimal.py",
            "--checkpoint", str(checkpoint),
            "--map_id", args.map_id,
            "--horizon", str(args.horizon),
            "--num_episodes", str(args.num_eval_episodes),
            "--deterministic",
            "--output_dir", str(seed_dir / "eval"),
            "--seed", str(seed),
        ])
        run_command([
            sys.executable,
            "scripts/eval_top_erl_minimal.py",
            "--checkpoint", str(checkpoint),
            "--map_id", args.map_id,
            "--horizon", str(args.horizon),
            "--num_episodes", str(args.num_eval_episodes),
            "--random_baseline",
            "--output_dir", str(seed_dir / "eval"),
            "--seed", str(seed),
        ])
        run_command([
            sys.executable,
            "scripts/diagnose_top_erl_trajectories.py",
            "--checkpoint", str(checkpoint),
            "--map_id", args.map_id,
            "--horizon", str(args.horizon),
            "--num_segments", "5",
            "--output_dir", str(seed_dir / "diagnostics"),
            "--seed", str(seed),
        ])

        rows.append(collect_seed_summary(seed, seed_dir))

    summary_path = output_root / "multiseed_summary.csv"
    write_summary(summary_path, rows)
    print(f"multiseed_summary_csv: {summary_path}")
    print_summary(rows)


if __name__ == "__main__":
    main()
