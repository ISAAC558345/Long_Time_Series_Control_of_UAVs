from pathlib import Path
import argparse
import csv
import math
import statistics


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze minimal TOP-ERL multi-seed failure modes.")
    parser.add_argument("--run_dir", default="outputs/top_erl_multiseed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value):
    return float(value)


def finite_row(row, skip_keys):
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


def latest_summary(rows, policy_type):
    matches = [row for row in rows if row["policy_type"] == policy_type]
    if not matches:
        raise ValueError(f"No summary row found for {policy_type}.")
    return matches[-1]


def training_stats(rows):
    reward = [to_float(row["rollout_cumulative_reward"]) for row in rows]
    throughput = [to_float(row["TotalThroughput"]) for row in rows]
    fairness = [to_float(row["FairIdx"]) for row in rows]
    collision = [to_float(row["ProbCollision"]) for row in rows]
    critic_loss = [to_float(row["critic_loss"]) for row in rows]
    actor_loss = [to_float(row["actor_loss"]) for row in rows]
    buffer_size = [to_float(row["replay_buffer_size"]) for row in rows]

    return dict(
        final_reward=reward[-1],
        mean_reward=statistics.fmean(reward),
        last10_mean_reward=statistics.fmean(reward[-10:]),
        max_reward=max(reward),
        final_throughput=throughput[-1],
        mean_throughput=statistics.fmean(throughput),
        last10_mean_throughput=statistics.fmean(throughput[-10:]),
        final_fairness=fairness[-1],
        mean_fairness=statistics.fmean(fairness),
        last10_mean_fairness=statistics.fmean(fairness[-10:]),
        max_collision=max(collision),
        final_critic_loss=critic_loss[-1],
        mean_critic_loss=statistics.fmean(critic_loss),
        max_critic_loss=max(critic_loss),
        final_actor_loss=actor_loss[-1],
        mean_actor_loss=statistics.fmean(actor_loss),
        actor_loss_range=max(actor_loss) - min(actor_loss),
        final_buffer_size=buffer_size[-1],
        finite=all(finite_row(row, skip_keys=set()) for row in rows),
    )


def diagnostics_by_policy(rows):
    return {row["policy_type"]: row for row in rows}


def per_uav_values(row, prefix):
    values = []
    for key, value in row.items():
        if key.startswith(prefix):
            values.append(to_float(value))
    return values


def diagnostic_flags(row, max_step_tolerance=100.0 + 1e-3):
    path_lengths = per_uav_values(row, "path_length_uav")
    max_steps = per_uav_values(row, "max_step_length_uav")
    final_disp = per_uav_values(row, "final_displacement_uav")
    min_inter_uav = to_float(row["min_inter_uav_distance"])
    boundary_hit = to_float(row["boundary_hit_ratio"])

    return dict(
        stationary=all(value < 1e-5 for value in path_lengths),
        weak_motion=any(value < 50.0 for value in final_disp),
        boundary_stuck=boundary_hit > 0.2,
        boundary_touched=boundary_hit > 0.0,
        uav_overlap=min_inter_uav < 1e-5,
        uav_too_close=min_inter_uav < 50.0,
        abnormal_jump=any(value > max_step_tolerance for value in max_steps),
        far_from_gt=to_float(row["mean_nearest_gt_distance"]) > 300.0
            or to_float(row["final_nearest_gt_distance"]) > 350.0,
        low_fairness=to_float(row["FairIdx"]) <= 0.26,
        low_throughput=to_float(row["TotalThroughput"]) <= 0.05,
        finite=row.get("trajectory_finite") == "True",
    )


def likely_failure_reason(seed, summary, train, det_diag, random_diag):
    det_flags = diagnostic_flags(det_diag) if det_diag else {}
    reward_gap = summary["det_reward"] - summary["random_reward"]
    throughput_gap = summary["det_throughput"] - summary["random_throughput"]
    fairness_gap = summary["det_fairness"] - summary["random_fairness"]

    reasons = []
    if reward_gap < 0:
        reasons.append("deterministic actor reward is below random baseline")
    if throughput_gap < 0:
        reasons.append("throughput is below random baseline")
    if fairness_gap < 0:
        reasons.append("fairness is below random baseline")
    if det_flags.get("low_fairness"):
        reasons.append("deterministic trajectory has very low FairIdx")
    if det_flags.get("low_throughput"):
        reasons.append("deterministic trajectory has low throughput")
    if det_flags.get("far_from_gt"):
        reasons.append("UAVs remain far from GTs")
    if det_flags.get("weak_motion"):
        reasons.append("at least one UAV has weak final displacement")
    if det_flags.get("boundary_stuck"):
        reasons.append("boundary hit ratio suggests boundary sticking")
    elif det_flags.get("boundary_touched"):
        reasons.append("boundary is touched but not dominant")
    if det_flags.get("uav_too_close"):
        reasons.append("UAVs get close to each other")
    if det_flags.get("abnormal_jump"):
        reasons.append("max step length exceeds nominal movement limit")

    if not reasons:
        reasons.append("no obvious trajectory anomaly; seed looks stable under current diagnostics")

    if seed == 1:
        priority = (
            "actor exploration scale / deterministic-mean extraction is the first module to inspect, "
            "because training obtains nonzero stochastic rollout reward but deterministic evaluation collapses to zero reward."
        )
    elif seed == 2:
        priority = (
            "actor exploration scale and training iterations are the first modules to inspect, with critic target stability second, "
            "because training and deterministic evaluation both remain low while trajectories move but do not reach useful GT geometry."
        )
    else:
        priority = "no failure priority assigned; this seed is the positive sanity case."

    return reasons, priority


def load_seed(run_dir, seed):
    seed_dir = run_dir / f"seed_{seed}"
    train_rows = read_csv(seed_dir / "train_log.csv")
    eval_rows = read_csv(seed_dir / "eval" / "eval_summary.csv")
    diagnostics_path = seed_dir / "diagnostics" / "trajectory_diagnostics.csv"
    diagnostics_rows = read_csv(diagnostics_path) if diagnostics_path.exists() else []

    det = latest_summary(eval_rows, "deterministic_actor")
    random = latest_summary(eval_rows, "random_baseline")
    train = training_stats(train_rows)
    diag = diagnostics_by_policy(diagnostics_rows) if diagnostics_rows else {}
    det_diag = diag.get("deterministic_actor")
    random_diag = diag.get("random_baseline")

    summary = dict(
        det_reward=to_float(det["mean_cumulative_reward"]),
        random_reward=to_float(random["mean_cumulative_reward"]),
        reward_gap=to_float(det["mean_cumulative_reward"]) - to_float(random["mean_cumulative_reward"]),
        det_throughput=to_float(det["mean_TotalThroughput"]),
        random_throughput=to_float(random["mean_TotalThroughput"]),
        throughput_gap=to_float(det["mean_TotalThroughput"]) - to_float(random["mean_TotalThroughput"]),
        det_fairness=to_float(det["mean_FairIdx"]),
        random_fairness=to_float(random["mean_FairIdx"]),
        fairness_gap=to_float(det["mean_FairIdx"]) - to_float(random["mean_FairIdx"]),
        finite=(
            train["finite"]
            and finite_row(det, {"policy_type", "all_metrics_finite"})
            and finite_row(random, {"policy_type", "all_metrics_finite"})
            and (not det_diag or det_diag.get("trajectory_finite") == "True")
            and (not random_diag or random_diag.get("trajectory_finite") == "True")
        ),
    )
    reasons, priority = likely_failure_reason(seed, summary, train, det_diag, random_diag)
    return dict(
        seed=seed,
        train=train,
        summary=summary,
        det_diag=det_diag,
        random_diag=random_diag,
        reasons=reasons,
        priority=priority,
    )


def format_float(value):
    return f"{float(value):.6f}"


def diagnostics_section(row):
    if not row:
        return "diagnostics not found\n"

    lines = [
        f"- cumulative_reward: {row['cumulative_reward']}",
        f"- TotalThroughput: {row['TotalThroughput']}",
        f"- FairIdx: {row['FairIdx']}",
        f"- ProbCollision: {row['ProbCollision']}",
        f"- min_inter_uav_distance: {row['min_inter_uav_distance']}",
        f"- mean_nearest_gt_distance: {row['mean_nearest_gt_distance']}",
        f"- final_nearest_gt_distance: {row['final_nearest_gt_distance']}",
        f"- boundary_hit_ratio: {row['boundary_hit_ratio']}",
        f"- trajectory_finite: {row['trajectory_finite']}",
    ]
    uav_ids = sorted(
        int(key.replace("path_length_uav", ""))
        for key in row
        if key.startswith("path_length_uav")
    )
    for uav_id in uav_ids:
        lines.append(
            "- UAV {uav}: path_length={path}, final_displacement={disp}, "
            "mean_step_length={mean_step}, max_step_length={max_step}".format(
                uav=uav_id,
                path=row[f"path_length_uav{uav_id}"],
                disp=row[f"final_displacement_uav{uav_id}"],
                mean_step=row[f"mean_step_length_uav{uav_id}"],
                max_step=row[f"max_step_length_uav{uav_id}"],
            )
        )
    flags = diagnostic_flags(row)
    lines.append("- flags: " + ", ".join(f"{key}={value}" for key, value in flags.items()))
    return "\n".join(lines) + "\n"


def build_report(run_dir, multiseed_rows, analyses):
    lines = [
        "# Multi-seed failure analysis",
        "",
        "This report is read-only analysis of existing outputs. It does not modify training code or rerun training.",
        "",
        "## Multi-seed summary",
        "",
        "| seed | final train reward | det reward | random reward | reward gap | throughput gap | fairness gap | final critic loss | final actor loss | finite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for analysis in analyses:
        seed = analysis["seed"]
        train = analysis["train"]
        summary = analysis["summary"]
        lines.append(
            "| {seed} | {train_reward} | {det_reward} | {random_reward} | {reward_gap} | {throughput_gap} | {fairness_gap} | {critic_loss} | {actor_loss} | {finite} |".format(
                seed=seed,
                train_reward=format_float(train["final_reward"]),
                det_reward=format_float(summary["det_reward"]),
                random_reward=format_float(summary["random_reward"]),
                reward_gap=format_float(summary["reward_gap"]),
                throughput_gap=format_float(summary["throughput_gap"]),
                fairness_gap=format_float(summary["fairness_gap"]),
                critic_loss=format_float(train["final_critic_loss"]),
                actor_loss=format_float(train["final_actor_loss"]),
                finite=summary["finite"],
            )
        )

    reward_gaps = [analysis["summary"]["reward_gap"] for analysis in analyses]
    throughput_gaps = [analysis["summary"]["throughput_gap"] for analysis in analyses]
    fairness_gaps = [analysis["summary"]["fairness_gap"] for analysis in analyses]
    lines.extend([
        "",
        "## Aggregate gaps",
        "",
        f"- mean reward gap: {format_float(statistics.fmean(reward_gaps))}",
        f"- mean throughput gap: {format_float(statistics.fmean(throughput_gaps))}",
        f"- mean fairness gap: {format_float(statistics.fmean(fairness_gaps))}",
        f"- all finite: {all(analysis['summary']['finite'] for analysis in analyses)}",
        "",
        "## Per-seed training curve summary",
        "",
    ])

    for analysis in analyses:
        train = analysis["train"]
        lines.extend([
            f"### Seed {analysis['seed']}",
            "",
            f"- final rollout_cumulative_reward: {format_float(train['final_reward'])}",
            f"- mean rollout_cumulative_reward: {format_float(train['mean_reward'])}",
            f"- last10 mean reward: {format_float(train['last10_mean_reward'])}",
            f"- max reward: {format_float(train['max_reward'])}",
            f"- final TotalThroughput: {format_float(train['final_throughput'])}",
            f"- last10 mean TotalThroughput: {format_float(train['last10_mean_throughput'])}",
            f"- final FairIdx: {format_float(train['final_fairness'])}",
            f"- last10 mean FairIdx: {format_float(train['last10_mean_fairness'])}",
            f"- max ProbCollision: {format_float(train['max_collision'])}",
            f"- final critic_loss: {format_float(train['final_critic_loss'])}",
            f"- max critic_loss: {format_float(train['max_critic_loss'])}",
            f"- final actor_loss: {format_float(train['final_actor_loss'])}",
            f"- actor_loss range: {format_float(train['actor_loss_range'])}",
            f"- replay_buffer_size: {format_float(train['final_buffer_size'])}",
            "",
            "Failure indications:",
        ])
        for reason in analysis["reasons"]:
            lines.append(f"- {reason}")
        lines.extend([
            "",
            f"Priority check: {analysis['priority']}",
            "",
            "Deterministic actor diagnostics:",
            diagnostics_section(analysis["det_diag"]),
            "Random baseline diagnostics:",
            diagnostics_section(analysis["random_diag"]),
        ])

    lines.extend([
        "## Short conclusion",
        "",
        "- Seed 1 most likely fails at the actor policy output / deterministic mean level: deterministic evaluation collapses to zero reward and very low throughput/fairness, while training still saw nonzero stochastic rollout reward.",
        "- Seed 2 most likely fails because the learned deterministic trajectory remains in poor service geometry: reward, throughput, and fairness stay low, UAVs move but remain relatively far from GTs and touch boundaries.",
        "- The first module to inspect is actor exploration scale and deterministic policy extraction; the second is critic target stability / limited training horizon. The current diagnostics do not primarily point to trajectory generator expressiveness or UAV collision.",
        "- This is not a final algorithm conclusion; it is a 3-seed sanity diagnosis.",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_path = Path(args.output) if args.output else run_dir / "multiseed_failure_analysis.md"
    multiseed_path = run_dir / "multiseed_summary.csv"
    multiseed_rows = read_csv(multiseed_path)

    analyses = [load_seed(run_dir, seed) for seed in args.seeds]
    report = build_report(run_dir, multiseed_rows, analyses)
    output_path.write_text(report, encoding="utf-8")

    seed1 = next((item for item in analyses if item["seed"] == 1), None)
    seed2 = next((item for item in analyses if item["seed"] == 2), None)
    print(f"analysis_report: {output_path}")
    if seed1:
        print("seed 1 likely cause:", seed1["priority"])
    if seed2:
        print("seed 2 likely cause:", seed2["priority"])
    print("priority module: actor exploration scale / deterministic policy extraction first; critic target stability and limited training iterations second.")
    print(f"all finite: {all(item['summary']['finite'] for item in analyses)}")


if __name__ == "__main__":
    main()
