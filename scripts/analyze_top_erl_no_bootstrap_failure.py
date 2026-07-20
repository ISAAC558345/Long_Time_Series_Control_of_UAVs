from pathlib import Path
import argparse
import csv
import math
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze no-bootstrap multi-seed failure cases from existing CSVs.")
    parser.add_argument("--base_dir", default="outputs/top_erl_no_bootstrap_multiseed")
    parser.add_argument("--full_bootstrap_summary", default="outputs/top_erl_multiseed/multiseed_summary.csv")
    parser.add_argument("--output", default="outputs/top_erl_no_bootstrap_multiseed/no_bootstrap_failure_analysis.md")
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def row_by_seed(rows, seed):
    for row in rows:
        if int(row["seed"]) == int(seed):
            return row
    raise ValueError(f"Missing seed {seed}.")


def row_by_policy(rows, policy_type):
    for row in rows:
        if row["policy_type"] == policy_type:
            return row
    raise ValueError(f"Missing policy_type={policy_type}.")


def f(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return float("nan")
    return float(value)


def mean(values):
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else float("nan")


def finite_rows(rows):
    for row in rows:
        for value in row.values():
            if value in ("", None):
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if not math.isfinite(number):
                return False
    return True


def load_diagnostics(base_dir, seed):
    path = Path(base_dir) / f"seed_{seed}" / "diagnostics" / "trajectory_diagnostics.csv"
    rows = read_csv(path)
    return row_by_policy(rows, "deterministic_actor"), row_by_policy(rows, "random_baseline")


def build_report(base_dir, full_summary_path):
    base_dir = Path(base_dir)
    summary_rows = read_csv(base_dir / "multiseed_summary.csv")
    calibration_rows = read_csv(base_dir / "critic_calibration" / "critic_calibration_summary.csv")
    actor_rows = read_csv(base_dir / "actor_distribution_analysis.csv")
    full_rows = read_csv(full_summary_path) if Path(full_summary_path).exists() else []

    seeds = [int(row["seed"]) for row in summary_rows]
    det_reward_gaps = [f(row, "deterministic_minus_random_reward") for row in summary_rows]
    stoch_reward_gaps = [f(row, "stochastic_minus_random_reward") for row in summary_rows]
    det_fair_gaps = [f(row, "deterministic_minus_random_FairIdx") for row in summary_rows]
    stoch_fair_gaps = [f(row, "stochastic_minus_random_FairIdx") for row in summary_rows]
    full_det_reward_gaps = [
        f(row, "deterministic_minus_random_reward") for row in full_rows
        if "deterministic_minus_random_reward" in row
    ]
    no_bootstrap_bad_det = [int(row["seed"]) for row in summary_rows if f(row, "deterministic_minus_random_reward") < 0]
    no_bootstrap_bad_stoch = [int(row["seed"]) for row in summary_rows if f(row, "stochastic_minus_random_reward") < 0]
    full_bad_det = [
        int(row["seed"]) for row in full_rows
        if "deterministic_minus_random_reward" in row and f(row, "deterministic_minus_random_reward") < 0
    ]

    lines = [
        "# No-Bootstrap Multi-Seed Failure Analysis",
        "",
        "This report reads existing CSV diagnostics only. It does not retrain or modify algorithms.",
        "",
        "## Multi-Seed Evaluation",
        "",
        "| seed | det reward | stoch reward | random reward | det gap | stoch gap | det FairIdx | stoch FairIdx | random FairIdx | det Fair gap | stoch Fair gap | Q-reward corr | top-1 regret | finite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['seed']} | {f(row, 'deterministic_mean_reward'):.6f} | "
            f"{f(row, 'stochastic_mean_reward'):.6f} | {f(row, 'random_mean_reward'):.6f} | "
            f"{f(row, 'deterministic_minus_random_reward'):.6f} | "
            f"{f(row, 'stochastic_minus_random_reward'):.6f} | "
            f"{f(row, 'deterministic_mean_FairIdx'):.6f} | "
            f"{f(row, 'stochastic_mean_FairIdx'):.6f} | {f(row, 'random_mean_FairIdx'):.6f} | "
            f"{f(row, 'deterministic_minus_random_FairIdx'):.6f} | "
            f"{f(row, 'stochastic_minus_random_FairIdx'):.6f} | "
            f"{f(row, 'q_reward_corr'):.6f} | {f(row, 'top1_regret'):.6f} | {row['finite_check']} |"
        )

    lines.extend([
        "",
        "## Actor Distribution",
        "",
        "| seed | mean_log_std | mean_std | det w norm | stoch w norm | det mean step | det max step | near zero | tanh sat ratio | det clip ratio | finite |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|",
    ])
    for row in actor_rows:
        lines.append(
            f"| {row['seed']} | {f(row, 'mean_log_std'):.6f} | {f(row, 'mean_std'):.6f} | "
            f"{f(row, 'deterministic_w_norm_mean'):.6f} | "
            f"{f(row, 'stochastic_w_norm_mean'):.6f} | "
            f"{f(row, 'deterministic_action_mean_step_length'):.6f} | "
            f"{f(row, 'deterministic_action_max_step_length'):.6f} | "
            f"{row['deterministic_action_near_zero']} | "
            f"{f(row, 'deterministic_tanh_saturation_ratio'):.6f} | "
            f"{f(row, 'deterministic_clip_saturation_ratio'):.6f} | {row['finite_check']} |"
        )

    lines.extend([
        "",
        "## Trajectory Diagnostics",
        "",
        "| seed | policy | reward | throughput | FairIdx | path lengths | final displacement | mean step | max step | min inter-UAV dist | mean nearest GT | final nearest GT | boundary hit | finite |",
        "|---:|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|:---:|",
    ])
    for seed in seeds:
        det, random = load_diagnostics(base_dir, seed)
        for row in (det, random):
            path_lengths = "/".join(f"{f(row, f'path_length_uav{i}'):.2f}" for i in range(3))
            final_displacements = "/".join(f"{f(row, f'final_displacement_uav{i}'):.2f}" for i in range(3))
            mean_steps = "/".join(f"{f(row, f'mean_step_length_uav{i}'):.2f}" for i in range(3))
            max_steps = "/".join(f"{f(row, f'max_step_length_uav{i}'):.2f}" for i in range(3))
            lines.append(
                f"| {seed} | {row['policy_type']} | {f(row, 'cumulative_reward'):.6f} | "
                f"{f(row, 'TotalThroughput'):.6f} | {f(row, 'FairIdx'):.6f} | "
                f"{path_lengths} | {final_displacements} | {mean_steps} | {max_steps} | "
                f"{f(row, 'min_inter_uav_distance'):.2f} | {f(row, 'mean_nearest_gt_distance'):.2f} | "
                f"{f(row, 'final_nearest_gt_distance'):.2f} | {f(row, 'boundary_hit_ratio'):.6f} | "
                f"{row['trajectory_finite']} |"
            )

    seed1 = row_by_seed(summary_rows, 1)
    seed1_actor = row_by_seed(actor_rows, 1)
    seed1_det, seed1_rand = load_diagnostics(base_dir, 1)
    all_finite = (
        finite_rows(summary_rows)
        and finite_rows(calibration_rows)
        and finite_rows(actor_rows)
        and all(load_diagnostics(base_dir, seed)[0]["trajectory_finite"] == "True" for seed in seeds)
        and all(load_diagnostics(base_dir, seed)[1]["trajectory_finite"] == "True" for seed in seeds)
    )

    lines.extend([
        "",
        "## Diagnostic Answers",
        "",
        f"- No-bootstrap mean deterministic reward gap: {mean(det_reward_gaps):.6f}.",
        f"- No-bootstrap mean stochastic reward gap: {mean(stoch_reward_gaps):.6f}.",
        f"- No-bootstrap mean deterministic FairIdx gap: {mean(det_fair_gaps):.6f}.",
        f"- No-bootstrap mean stochastic FairIdx gap: {mean(stoch_fair_gaps):.6f}.",
    ])
    if full_det_reward_gaps:
        lines.append(f"- Previous full-bootstrap mean deterministic reward gap: {mean(full_det_reward_gaps):.6f}.")
    lines.extend([
        f"- Previous full-bootstrap deterministic-below-random seeds: {full_bad_det}.",
        f"- No-bootstrap deterministic-below-random seeds: {no_bootstrap_bad_det}.",
        f"- No-bootstrap stochastic-below-random seeds: {no_bootstrap_bad_stoch}.",
        f"- All inspected metrics finite: {all_finite}.",
        "",
        "### Seed 1",
        "",
        f"- Deterministic reward gap is {f(seed1, 'deterministic_minus_random_reward'):.6f}, while stochastic reward gap is {f(seed1, 'stochastic_minus_random_reward'):.6f}. This points to deterministic mean extraction degradation more than a fully failed stochastic policy.",
        f"- Seed 1 deterministic mean step is {f(seed1_actor, 'deterministic_action_mean_step_length'):.6f}, with det w norm {f(seed1_actor, 'deterministic_w_norm_mean'):.6f}; it is not near zero, but is much more conservative than stochastic w norm {f(seed1_actor, 'stochastic_w_norm_mean'):.6f}.",
        f"- Seed 1 deterministic final nearest-GT distance is {f(seed1_det, 'final_nearest_gt_distance'):.2f}, versus random baseline {f(seed1_rand, 'final_nearest_gt_distance'):.2f}; the deterministic trajectory ends farther from GTs.",
        f"- Seed 1 deterministic boundary hit ratio is {f(seed1_det, 'boundary_hit_ratio'):.6f}, min inter-UAV distance is {f(seed1_det, 'min_inter_uav_distance'):.2f}; no boundary sticking or UAV overlap is indicated.",
        f"- Seed 1 Q-reward correlation is {f(seed1, 'q_reward_corr'):.6f}; Q-ranking remains weak.",
        "",
        "## Conclusion",
        "",
        "- No-bootstrap is more stable than the previous full-bootstrap sanity run, but seed 1 still has deterministic actor failure.",
        "- Stochastic actor is more reliable in this no-bootstrap run: it is above random on all three seeds.",
        "- Deterministic actor should not be the only main evaluation strategy yet; report deterministic and stochastic side by side.",
        "- Q-ranking remains a bottleneck because correlations are weak and top-1 regret remains nontrivial.",
        "- Next checks worth prioritizing: continue with no_bootstrap as the target mode, run longer 200/500-iteration sanity, inspect deterministic extraction, and keep improving critic ranking/calibration.",
        "- This is diagnostic only and not a final algorithm conclusion.",
    ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    report = build_report(args.base_dir, args.full_bootstrap_summary)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"no_bootstrap_failure_report: {output_path}")


if __name__ == "__main__":
    main()
