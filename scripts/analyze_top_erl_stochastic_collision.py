from pathlib import Path
import argparse
import csv
import math
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.dont_write_bytecode = True
sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))
sys.modules.setdefault("matplotlib.gridspec", types.ModuleType("matplotlib.gridspec"))


POLICY_TYPES = [
    "deterministic_actor",
    "stochastic_std_1.0",
    "stochastic_std_0.5",
    "stochastic_std_0.25",
    "random_baseline",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose stochastic actor collisions with eval-time std scaling.")
    parser.add_argument("--base_dir", default="outputs/top_erl_no_bootstrap_200_multiseed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--map_id", default="debug")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument(
        "--output_dir",
        default="outputs/top_erl_no_bootstrap_200_multiseed/stochastic_collision_diagnostics",
    )
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value):
    if value in ("", None):
        return float("nan")
    return float(value)


def finite_values(values):
    return all(math.isfinite(float(value)) for value in values)


def build_actor(env, horizon, checkpoint):
    import torch

    from models.trajectory_actor import TrajectoryActor

    state_dim = env.get_state().shape[-1]
    max_move_per_slot = float(np.max(np.asarray(env.vels, dtype=np.float32)) * env.dt)
    actor = TrajectoryActor(
        state_dim=state_dim,
        n_uavs=env.n_ubs,
        hidden_dim=64,
        w_scale=horizon * max_move_per_slot,
    )
    actor.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    actor.eval()
    return actor, max_move_per_slot


def actor_waypoint(actor, state, n_uavs, policy_type):
    import torch

    state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mean, log_std = actor(state_tensor)
        if policy_type == "deterministic_actor":
            raw_w = mean
        else:
            std_scale = float(policy_type.replace("stochastic_std_", ""))
            raw_w = mean + std_scale * log_std.exp() * torch.randn_like(mean)
        w = torch.tanh(raw_w).reshape(1, n_uavs, 2) * actor.w_scale
    return w[0].cpu().numpy()


def random_waypoint(rng, n_uavs, horizon, max_move_per_slot):
    limit = horizon * max_move_per_slot
    return rng.uniform(-limit, limit, size=(n_uavs, 2)).astype(np.float32)


def pairwise_inter_uav_distances(positions):
    n_uavs = positions.shape[1]
    distances = []
    for step in range(positions.shape[0]):
        for i in range(n_uavs):
            for j in range(i + 1, n_uavs):
                distances.append(float(np.linalg.norm(positions[step, i] - positions[step, j])))
    return np.asarray(distances, dtype=np.float64)


def nearest_gt_distances(positions, gt_positions):
    diff = positions[:, :, None, :] - gt_positions[None, None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    return distances.min(axis=-1)


def boundary_hit_ratio(positions, range_pos, tol=1e-5):
    lower_hits = positions <= tol
    upper_hits = positions >= (float(range_pos) - tol)
    return float(np.mean(lower_hits | upper_hits))


def trajectory_metrics(trajectory_info, gt_positions, range_pos, max_move_per_slot):
    positions = np.asarray(trajectory_info["pos_ubs_seq"], dtype=np.float64)
    gt_positions = np.asarray(gt_positions, dtype=np.float64)
    steps = np.diff(positions, axis=0)
    step_lengths = np.linalg.norm(steps, axis=-1)
    inter_uav = pairwise_inter_uav_distances(positions)
    nearest_gt = nearest_gt_distances(positions, gt_positions)
    boundary_ratio = boundary_hit_ratio(positions, range_pos)
    mean_step = float(np.mean(step_lengths)) if step_lengths.size else 0.0
    max_step = float(np.max(step_lengths)) if step_lengths.size else 0.0
    min_inter = float(np.min(inter_uav)) if inter_uav.size else float("nan")
    mean_inter = float(np.mean(inter_uav)) if inter_uav.size else float("nan")

    numeric = [
        mean_step,
        max_step,
        min_inter,
        mean_inter,
        float(np.mean(nearest_gt)),
        float(np.mean(nearest_gt[-1])),
        boundary_ratio,
    ]
    numeric.extend(positions.reshape(-1).tolist())

    return dict(
        trajectory_finite=finite_values(numeric),
        mean_step_length=mean_step,
        max_step_length=max_step,
        min_inter_uav_distance=min_inter,
        mean_inter_uav_distance=mean_inter,
        mean_nearest_gt_distance=float(np.mean(nearest_gt)),
        final_nearest_gt_distance=float(np.mean(nearest_gt[-1])),
        boundary_hit_ratio=boundary_ratio,
        whether_boundary_hit=boundary_ratio > 0.0,
        whether_abnormal_jump=max_step > max_move_per_slot + 1e-5,
    )


def evaluate_policy(seed, policy_type, args):
    import torch

    from envs.mubs_cov.mubs_cov_traj import MultiUbsCoverageTrajEnv
    from models.trajectory_generator import LinearWaypointTrajectoryGenerator

    seed_dir = Path(args.base_dir) / f"seed_{seed}"
    checkpoint = seed_dir / "checkpoints" / "actor.pt"
    env = MultiUbsCoverageTrajEnv(map_id=args.map_id, record=False)
    env.reset()
    actor, max_move_per_slot = build_actor(env, args.horizon, checkpoint)
    generator = LinearWaypointTrajectoryGenerator(
        H=args.horizon,
        n_uavs=env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=env.range_pos,
    )
    rng = np.random.default_rng(seed)
    rows = []

    for episode in range(args.num_episodes):
        env.reset()
        gt_positions = env.pos_gts.copy()
        state = env.get_state()
        if policy_type == "random_baseline":
            w = random_waypoint(rng, env.n_ubs, args.horizon, max_move_per_slot)
        else:
            w = actor_waypoint(actor, state, env.n_ubs, policy_type)

        action_seq = generator.generate(current_pos=env.pos_ubs.copy(), w=w)
        trajectory_info = env.step_trajectory(action_seq, action_type="displacement")
        info = trajectory_info["info"]
        metrics = trajectory_metrics(trajectory_info, gt_positions, env.range_pos, max_move_per_slot)
        row = dict(
            seed=seed,
            policy_type=policy_type,
            episode=episode,
            cumulative_reward=float(trajectory_info["cumulative_reward"]),
            TotalThroughput=float(info.get("TotalThroughput", float("nan"))),
            FairIdx=float(info.get("FairIdx", float("nan"))),
            ProbCollision=float(info.get("ProbCollision", float("nan"))),
            **metrics,
        )
        row["whether_collision"] = row["ProbCollision"] > 0.0
        numeric = [
            row["cumulative_reward"],
            row["TotalThroughput"],
            row["FairIdx"],
            row["ProbCollision"],
            row["mean_step_length"],
            row["max_step_length"],
            row["min_inter_uav_distance"],
            row["mean_inter_uav_distance"],
            row["mean_nearest_gt_distance"],
            row["final_nearest_gt_distance"],
            row["boundary_hit_ratio"],
        ]
        row["episode_finite"] = row["trajectory_finite"] and finite_values(numeric)
        rows.append(row)

    return rows


def mean(values):
    return float(np.mean(values)) if values else float("nan")


def std(values):
    return float(np.std(values)) if values else float("nan")


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["seed"], row["policy_type"]), []).append(row)

    summary_rows = []
    for (seed, policy_type), items in sorted(grouped.items()):
        rewards = [row["cumulative_reward"] for row in items]
        throughputs = [row["TotalThroughput"] for row in items]
        fairness = [row["FairIdx"] for row in items]
        collisions = [row["ProbCollision"] for row in items]
        collision_flags = [float(row["whether_collision"]) for row in items]
        boundary_flags = [float(row["whether_boundary_hit"]) for row in items]
        abnormal_flags = [float(row["whether_abnormal_jump"]) for row in items]
        summary_rows.append(dict(
            seed=seed,
            policy_type=policy_type,
            num_episodes=len(items),
            mean_cumulative_reward=mean(rewards),
            std_cumulative_reward=std(rewards),
            mean_TotalThroughput=mean(throughputs),
            mean_FairIdx=mean(fairness),
            mean_ProbCollision=mean(collisions),
            collision_episode_ratio=mean(collision_flags),
            mean_step_length=mean([row["mean_step_length"] for row in items]),
            max_step_length=max(row["max_step_length"] for row in items),
            min_inter_uav_distance=min(row["min_inter_uav_distance"] for row in items),
            mean_inter_uav_distance=mean([row["mean_inter_uav_distance"] for row in items]),
            mean_nearest_gt_distance=mean([row["mean_nearest_gt_distance"] for row in items]),
            final_nearest_gt_distance=mean([row["final_nearest_gt_distance"] for row in items]),
            boundary_hit_ratio=mean([row["boundary_hit_ratio"] for row in items]),
            boundary_episode_ratio=mean(boundary_flags),
            abnormal_jump_ratio=mean(abnormal_flags),
            all_metrics_finite=all(row["episode_finite"] for row in items),
        ))
    return summary_rows


def row_for(summary_rows, seed, policy_type):
    for row in summary_rows:
        if int(row["seed"]) == int(seed) and row["policy_type"] == policy_type:
            return row
    raise ValueError(f"Missing summary for seed={seed}, policy_type={policy_type}.")


def load_multiseed_summary(base_dir):
    path = Path(base_dir) / "multiseed_summary.csv"
    return read_csv(path) if path.exists() else []


def build_report(args, summary_rows):
    multiseed_rows = load_multiseed_summary(args.base_dir)
    lines = [
        "# Stochastic Collision Diagnostics",
        "",
        "This report evaluates saved checkpoints only. Eval-time std scaling is diagnostic and does not modify training or checkpoints.",
        "",
        "## Existing 200-Iteration Summary",
        "",
        "| seed | det reward | stoch reward | random reward | det gap | stoch gap | det FairIdx | stoch FairIdx | random FairIdx | det collision | stoch collision | random collision | finite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in multiseed_rows:
        lines.append(
            f"| {row['seed']} | {safe_float(row['deterministic_mean_reward']):.6f} | "
            f"{safe_float(row['stochastic_mean_reward']):.6f} | {safe_float(row['random_mean_reward']):.6f} | "
            f"{safe_float(row['deterministic_minus_random_reward']):.6f} | "
            f"{safe_float(row['stochastic_minus_random_reward']):.6f} | "
            f"{safe_float(row['deterministic_mean_FairIdx']):.6f} | "
            f"{safe_float(row['stochastic_mean_FairIdx']):.6f} | {safe_float(row['random_mean_FairIdx']):.6f} | "
            f"{safe_float(row['deterministic_mean_ProbCollision']):.6f} | "
            f"{safe_float(row['stochastic_mean_ProbCollision']):.6f} | "
            f"{safe_float(row['random_mean_ProbCollision']):.6f} | {row['finite_check']} |"
        )

    lines.extend([
        "",
        "## Std-Scale Evaluation",
        "",
        "| seed | policy | reward | FairIdx | ProbCollision | collision episode ratio | mean step | max step | min inter-UAV | boundary ratio | abnormal jump ratio | finite |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for row in summary_rows:
        lines.append(
            f"| {row['seed']} | {row['policy_type']} | {row['mean_cumulative_reward']:.6f} | "
            f"{row['mean_FairIdx']:.6f} | {row['mean_ProbCollision']:.6f} | "
            f"{row['collision_episode_ratio']:.6f} | {row['mean_step_length']:.6f} | "
            f"{row['max_step_length']:.6f} | {row['min_inter_uav_distance']:.6f} | "
            f"{row['boundary_hit_ratio']:.6f} | {row['abnormal_jump_ratio']:.6f} | "
            f"{row['all_metrics_finite']} |"
        )

    seed0_random = row_for(summary_rows, 0, "random_baseline")
    seed0_std1 = row_for(summary_rows, 0, "stochastic_std_1.0")
    seed0_std05 = row_for(summary_rows, 0, "stochastic_std_0.5")
    seed0_std025 = row_for(summary_rows, 0, "stochastic_std_0.25")
    seed0_det = row_for(summary_rows, 0, "deterministic_actor")
    all_finite = all(row["all_metrics_finite"] for row in summary_rows)
    hidden_collision = [
        (row["seed"], row["policy_type"])
        for row in summary_rows
        if row["collision_episode_ratio"] > 0.0
    ]

    lines.extend([
        "",
        "## Diagnostic Answers",
        "",
        f"- Seed 0 stochastic_std_1.0 reward is {seed0_std1['mean_cumulative_reward']:.6f}, random reward is {seed0_random['mean_cumulative_reward']:.6f}, deterministic reward is {seed0_det['mean_cumulative_reward']:.6f}.",
        f"- Seed 0 stochastic_std_1.0 collision episode ratio is {seed0_std1['collision_episode_ratio']:.6f}; std_0.5 is {seed0_std05['collision_episode_ratio']:.6f}; std_0.25 is {seed0_std025['collision_episode_ratio']:.6f}.",
        f"- Seed 0 stochastic_std_0.5 reward is {seed0_std05['mean_cumulative_reward']:.6f}; stochastic_std_0.25 reward is {seed0_std025['mean_cumulative_reward']:.6f}.",
        f"- Hidden collision cases across all summaries: {hidden_collision}.",
        f"- All metrics finite: {all_finite}.",
        "",
        "## Interpretation",
        "",
        "- If std scaling removes collisions and restores reward above random, the issue is likely eval-time exploration scale.",
        "- If collisions vanish but reward remains weak, the stochastic samples are not only unsafe but also lower-quality than the deterministic mean.",
        "- Deterministic actor should be the current main evaluation policy for this checkpoint family; stochastic actor remains useful as auxiliary robustness diagnostics.",
        "- This is diagnostic only, not a final algorithm conclusion.",
    ])
    return "\n".join(lines) + "\n"


def main():
    import torch

    args = parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        for policy_type in POLICY_TYPES:
            episode_rows.extend(evaluate_policy(seed, policy_type, args))

    summary_rows = summarize(episode_rows)
    episode_path = output_dir / "stochastic_collision_eval.csv"
    summary_path = output_dir / "stochastic_collision_summary.csv"
    report_path = output_dir / "stochastic_collision_analysis.md"
    write_csv(episode_path, episode_rows)
    write_csv(summary_path, summary_rows)
    report_path.write_text(build_report(args, summary_rows), encoding="utf-8")

    print(f"stochastic_collision_eval_csv: {episode_path}")
    print(f"stochastic_collision_summary_csv: {summary_path}")
    print(f"stochastic_collision_report: {report_path}")
    for row in summary_rows:
        print(
            "seed {seed} {policy_type}: reward={mean_cumulative_reward:.6f}, "
            "fair={mean_FairIdx:.6f}, collision={mean_ProbCollision:.6f}, "
            "collision_ratio={collision_episode_ratio:.6f}, finite={all_metrics_finite}".format(**row)
        )


if __name__ == "__main__":
    main()
