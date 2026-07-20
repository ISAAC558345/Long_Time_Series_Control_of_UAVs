from pathlib import Path
import argparse
import csv
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a minimal TOP-ERL-inspired actor checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/top_erl_minimal/checkpoints/actor.pt")
    parser.add_argument("--map_id", default="debug")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs/top_erl_minimal/eval")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--random_baseline", action="store_true")
    parser.add_argument("--std_scale", type=float, default=1.0)
    return parser.parse_args()


def validate_args(args):
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")
    if args.num_episodes <= 0:
        raise ValueError("--num_episodes must be positive.")
    if args.std_scale < 0.0:
        raise ValueError("--std_scale must be non-negative.")
    if args.random_baseline and args.deterministic:
        raise ValueError("--random_baseline and --deterministic are mutually exclusive.")


def get_policy_type(args):
    if args.random_baseline:
        return "random_baseline"
    if args.deterministic:
        return "deterministic_actor"
    return f"stochastic_actor_std_{args.std_scale:g}"


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
    state_dict = torch.load(checkpoint, map_location="cpu")
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor, max_move_per_slot


def actor_waypoint(actor, state, n_uavs, deterministic, std_scale=1.0):
    import torch

    state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        if deterministic:
            w, _, _ = actor.deterministic(state_tensor)
        else:
            mean, log_std = actor(state_tensor)
            raw_w = mean + float(std_scale) * log_std.exp() * torch.randn_like(mean)
            w = torch.tanh(raw_w).reshape(1, n_uavs, 2) * actor.w_scale
    return w[0].cpu().numpy()


def random_waypoint(rng, n_uavs, horizon, max_move_per_slot):
    limit = horizon * max_move_per_slot
    return rng.uniform(-limit, limit, size=(n_uavs, 2)).astype(np.float32)


def write_eval_log(log_path, rows):
    fieldnames = [
        "policy_type",
        "episode",
        "std_scale",
        "cumulative_reward",
        "TotalThroughput",
        "FairIdx",
        "ProbCollision",
        "trajectory_smoothness",
        "path_curvature",
        "oscillation_metric",
        "boundary_violation_ratio",
        "coverage_radius_per_uav",
        "coverage_consistency",
        "done",
    ]
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    rewards = [row["cumulative_reward"] for row in rows]
    throughputs = [row["TotalThroughput"] for row in rows]
    fairness = [row["FairIdx"] for row in rows]
    collisions = [row["ProbCollision"] for row in rows]
    smoothness = [row["trajectory_smoothness"] for row in rows]
    curvature = [row["path_curvature"] for row in rows]
    oscillation = [row["oscillation_metric"] for row in rows]
    boundary = [row["boundary_violation_ratio"] for row in rows]
    coverage_radius = [row["coverage_radius_per_uav"] for row in rows]
    coverage_consistency = [row["coverage_consistency"] for row in rows]
    return dict(
        mean_cumulative_reward=float(np.mean(rewards)),
        std_cumulative_reward=float(np.std(rewards)),
        mean_TotalThroughput=float(np.mean(throughputs)),
        std_TotalThroughput=float(np.std(throughputs)),
        mean_FairIdx=float(np.mean(fairness)),
        std_FairIdx=float(np.std(fairness)),
        mean_ProbCollision=float(np.mean(collisions)),
        std_ProbCollision=float(np.std(collisions)),
        mean_trajectory_smoothness=float(np.mean(smoothness)),
        std_trajectory_smoothness=float(np.std(smoothness)),
        mean_path_curvature=float(np.mean(curvature)),
        std_path_curvature=float(np.std(curvature)),
        mean_oscillation_metric=float(np.mean(oscillation)),
        std_oscillation_metric=float(np.std(oscillation)),
        mean_boundary_violation_ratio=float(np.mean(boundary)),
        std_boundary_violation_ratio=float(np.std(boundary)),
        mean_coverage_radius_per_uav=float(np.mean(coverage_radius)),
        mean_coverage_consistency=float(np.mean(coverage_consistency)),
        std_coverage_consistency=float(np.std(coverage_consistency)),
        reward_variance=float(np.var(rewards)),
        curvature_variance=float(np.var(curvature)),
    )


def append_eval_summary(summary_path, row):
    fieldnames = [
        "policy_type",
        "std_scale",
        "num_episodes",
        "mean_cumulative_reward",
        "std_cumulative_reward",
        "mean_TotalThroughput",
        "std_TotalThroughput",
        "mean_FairIdx",
        "std_FairIdx",
        "mean_ProbCollision",
        "std_ProbCollision",
        "mean_trajectory_smoothness",
        "std_trajectory_smoothness",
        "mean_path_curvature",
        "std_path_curvature",
        "mean_oscillation_metric",
        "std_oscillation_metric",
        "mean_boundary_violation_ratio",
        "std_boundary_violation_ratio",
        "mean_coverage_radius_per_uav",
        "mean_coverage_consistency",
        "std_coverage_consistency",
        "reward_variance",
        "curvature_variance",
        "all_metrics_finite",
    ]
    existing_rows = []
    rewrite_existing = False
    write_header = not summary_path.exists()
    if summary_path.exists():
        with summary_path.open(newline="") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            if existing_fieldnames != fieldnames:
                rewrite_existing = True
                existing_rows = list(reader)
                write_header = True
    if rewrite_existing:
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for old_row in existing_rows:
                writer.writerow({key: old_row.get(key, "") for key in fieldnames})
            writer.writerow(row)
        return
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def compute_trajectory_metrics(action_seq, pos_seq, range_pos, coverage_radius):
    action_seq = np.asarray(action_seq, dtype=np.float32)
    pos_seq = np.asarray(pos_seq, dtype=np.float32)

    if action_seq.ndim != 3 or action_seq.shape[-1] != 2:
        raise ValueError("action_seq must have shape [H, n_uavs, 2].")
    if pos_seq.ndim != 3 or pos_seq.shape[-1] != 2:
        raise ValueError("pos_seq must have shape [H+1, n_uavs, 2].")

    if action_seq.shape[0] >= 3:
        acceleration = action_seq[2:] - 2.0 * action_seq[1:-1] + action_seq[:-2]
        trajectory_smoothness = float(np.mean(np.square(acceleration)))
    else:
        trajectory_smoothness = 0.0

    if action_seq.shape[0] >= 2:
        prev = action_seq[:-1]
        nxt = action_seq[1:]
        prev_norm = np.linalg.norm(prev, axis=-1)
        nxt_norm = np.linalg.norm(nxt, axis=-1)
        valid = (prev_norm > 1e-8) & (nxt_norm > 1e-8)
        if np.any(valid):
            cosine = np.sum(prev * nxt, axis=-1) / (prev_norm * nxt_norm + 1e-8)
            cosine = np.clip(cosine[valid], -1.0, 1.0)
            path_curvature = float(np.mean(np.arccos(cosine)))
            oscillation_metric = float(np.mean(np.maximum(0.0, -cosine)))
        else:
            path_curvature = 0.0
            oscillation_metric = 0.0
    else:
        path_curvature = 0.0
        oscillation_metric = 0.0

    outside = (pos_seq < -1e-6) | (pos_seq > float(range_pos) + 1e-6)
    boundary_violation_ratio = float(np.mean(outside))

    coverage_consistency = 0.0
    return dict(
        trajectory_smoothness=trajectory_smoothness,
        path_curvature=path_curvature,
        oscillation_metric=oscillation_metric,
        boundary_violation_ratio=boundary_violation_ratio,
        coverage_radius_per_uav=float(coverage_radius),
        coverage_consistency=coverage_consistency,
    )


def compute_coverage_consistency(pos_seq, pos_gts_seq, coverage_radius):
    pos_seq = np.asarray(pos_seq, dtype=np.float32)
    pos_gts_seq = np.asarray(pos_gts_seq, dtype=np.float32)
    if pos_seq.ndim != 3 or pos_gts_seq.ndim != 3:
        return 0.0
    if pos_seq.shape[0] != pos_gts_seq.shape[0]:
        return 0.0
    distances = np.linalg.norm(pos_seq[:, :, None, :] - pos_gts_seq[:, None, :, :], axis=-1)
    covered_by_uav = (distances <= float(coverage_radius)).any(axis=-1).mean(axis=0)
    return float(1.0 / (1.0 + np.std(covered_by_uav)))


def main():
    import torch

    from envs.mubs_cov.mubs_cov_traj import MultiUbsCoverageTrajEnv
    from models.trajectory_generator import LinearWaypointTrajectoryGenerator

    args = parse_args()
    validate_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_type = get_policy_type(args)
    log_path = output_dir / f"eval_{policy_type}.csv"
    summary_path = output_dir / "eval_summary.csv"

    env = MultiUbsCoverageTrajEnv(map_id=args.map_id, record=False)
    env.reset()
    actor, max_move_per_slot = build_actor(env, args.horizon, args.checkpoint)
    generator = LinearWaypointTrajectoryGenerator(
        H=args.horizon,
        n_uavs=env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=env.range_pos,
    )

    rows = []
    for episode in range(args.num_episodes):
        env.reset()
        state = env.get_state()
        if args.random_baseline:
            w = random_waypoint(rng, env.n_ubs, args.horizon, max_move_per_slot)
        else:
            w = actor_waypoint(
                actor,
                state,
                env.n_ubs,
                deterministic=args.deterministic,
                std_scale=args.std_scale,
            )

        action_seq = generator.generate(current_pos=env.pos_ubs.copy(), w=w)
        trajectory_info = env.step_trajectory(action_seq, action_type="displacement")
        info = trajectory_info["info"]
        traj_metrics = compute_trajectory_metrics(
            action_seq=trajectory_info["displacement_seq"],
            pos_seq=trajectory_info["pos_ubs_seq"],
            range_pos=env.range_pos,
            coverage_radius=env.r_cov,
        )
        traj_metrics["coverage_consistency"] = compute_coverage_consistency(
            pos_seq=trajectory_info["pos_ubs_seq"],
            pos_gts_seq=trajectory_info["pos_gts_seq"],
            coverage_radius=env.r_cov,
        )
        row = dict(
            policy_type=policy_type,
            episode=episode,
            std_scale=float(args.std_scale),
            cumulative_reward=float(trajectory_info["cumulative_reward"]),
            TotalThroughput=float(info.get("TotalThroughput", float("nan"))),
            FairIdx=float(info.get("FairIdx", float("nan"))),
            ProbCollision=float(info.get("ProbCollision", float("nan"))),
            trajectory_smoothness=traj_metrics["trajectory_smoothness"],
            path_curvature=traj_metrics["path_curvature"],
            oscillation_metric=traj_metrics["oscillation_metric"],
            boundary_violation_ratio=traj_metrics["boundary_violation_ratio"],
            coverage_radius_per_uav=traj_metrics["coverage_radius_per_uav"],
            coverage_consistency=traj_metrics["coverage_consistency"],
            done=bool(trajectory_info["done"]),
        )
        rows.append(row)
        print(
            "episode {episode}: reward={cumulative_reward:.6f}, "
            "throughput={TotalThroughput:.6f}, fair={FairIdx:.6f}, "
            "collision={ProbCollision:.6f}, done={done}".format(**row)
        )

    write_eval_log(log_path, rows)
    summary = summarize(rows)
    finite_values = [
        summary["mean_cumulative_reward"],
        summary["std_cumulative_reward"],
        summary["mean_TotalThroughput"],
        summary["std_TotalThroughput"],
        summary["mean_FairIdx"],
        summary["std_FairIdx"],
        summary["mean_ProbCollision"],
        summary["std_ProbCollision"],
        summary["mean_trajectory_smoothness"],
        summary["std_trajectory_smoothness"],
        summary["mean_path_curvature"],
        summary["std_path_curvature"],
        summary["mean_oscillation_metric"],
        summary["std_oscillation_metric"],
        summary["mean_boundary_violation_ratio"],
        summary["std_boundary_violation_ratio"],
        summary["mean_coverage_radius_per_uav"],
        summary["mean_coverage_consistency"],
        summary["std_coverage_consistency"],
        summary["reward_variance"],
        summary["curvature_variance"],
    ]
    all_finite = bool(np.isfinite(finite_values).all())
    summary_row = dict(
        policy_type=policy_type,
        std_scale=float(args.std_scale),
        num_episodes=args.num_episodes,
        mean_cumulative_reward=summary["mean_cumulative_reward"],
        std_cumulative_reward=summary["std_cumulative_reward"],
        mean_TotalThroughput=summary["mean_TotalThroughput"],
        std_TotalThroughput=summary["std_TotalThroughput"],
        mean_FairIdx=summary["mean_FairIdx"],
        std_FairIdx=summary["std_FairIdx"],
        mean_ProbCollision=summary["mean_ProbCollision"],
        std_ProbCollision=summary["std_ProbCollision"],
        mean_trajectory_smoothness=summary["mean_trajectory_smoothness"],
        std_trajectory_smoothness=summary["std_trajectory_smoothness"],
        mean_path_curvature=summary["mean_path_curvature"],
        std_path_curvature=summary["std_path_curvature"],
        mean_oscillation_metric=summary["mean_oscillation_metric"],
        std_oscillation_metric=summary["std_oscillation_metric"],
        mean_boundary_violation_ratio=summary["mean_boundary_violation_ratio"],
        std_boundary_violation_ratio=summary["std_boundary_violation_ratio"],
        mean_coverage_radius_per_uav=summary["mean_coverage_radius_per_uav"],
        mean_coverage_consistency=summary["mean_coverage_consistency"],
        std_coverage_consistency=summary["std_coverage_consistency"],
        reward_variance=summary["reward_variance"],
        curvature_variance=summary["curvature_variance"],
        all_metrics_finite=all_finite,
    )
    append_eval_summary(summary_path, summary_row)

    print(f"eval_log: {log_path}")
    print(f"eval_summary: {summary_path}")
    print(f"policy_type: {policy_type}")
    print(f"mean cumulative reward: {summary['mean_cumulative_reward']}")
    print(f"std cumulative reward: {summary['std_cumulative_reward']}")
    print(f"mean TotalThroughput: {summary['mean_TotalThroughput']}")
    print(f"std TotalThroughput: {summary['std_TotalThroughput']}")
    print(f"mean FairIdx: {summary['mean_FairIdx']}")
    print(f"std FairIdx: {summary['std_FairIdx']}")
    print(f"mean ProbCollision: {summary['mean_ProbCollision']}")
    print(f"std ProbCollision: {summary['std_ProbCollision']}")
    print(f"mean trajectory smoothness: {summary['mean_trajectory_smoothness']}")
    print(f"mean path curvature: {summary['mean_path_curvature']}")
    print(f"mean oscillation metric: {summary['mean_oscillation_metric']}")
    print(f"mean boundary violation ratio: {summary['mean_boundary_violation_ratio']}")
    print(f"mean coverage consistency: {summary['mean_coverage_consistency']}")
    print(f"all summary metrics finite: {all_finite}")

    if not all_finite:
        raise RuntimeError("Evaluation summary contains NaN or Inf.")


if __name__ == "__main__":
    main()
