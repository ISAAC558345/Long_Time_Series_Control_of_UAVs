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


def parse_args():
    parser = argparse.ArgumentParser(description="Numerically diagnose minimal TOP-ERL trajectories.")
    parser.add_argument("--checkpoint", default="outputs/top_erl_sanity_50/checkpoints/actor.pt")
    parser.add_argument("--map_id", default="debug")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--num_segments", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs/top_erl_sanity_50/diagnostics")
    return parser.parse_args()


def validate_args(args):
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")
    if args.num_segments <= 0:
        raise ValueError("--num_segments must be positive.")


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


def deterministic_waypoint(actor, state, n_uavs):
    import torch

    state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mean, _ = actor(state_tensor)
        w = torch.tanh(mean).reshape(1, n_uavs, 2) * actor.w_scale
    return w[0].cpu().numpy()


def random_waypoint(rng, n_uavs, horizon, max_move_per_slot):
    limit = horizon * max_move_per_slot
    return rng.uniform(-limit, limit, size=(n_uavs, 2)).astype(np.float32)


def positions_from_trajectory_info(trajectory_info, n_uavs, range_pos):
    if "pos_ubs_seq" in trajectory_info:
        return np.asarray(trajectory_info["pos_ubs_seq"], dtype=np.float32)

    state_seq = np.asarray(trajectory_info["state_seq"], dtype=np.float32)
    pos_dim = n_uavs * 2
    pos_norm = state_seq[:, :pos_dim].reshape(state_seq.shape[0], n_uavs, 2)
    return pos_norm * float(range_pos)


def concat_segment_positions(position_segments):
    merged = []
    for idx, segment in enumerate(position_segments):
        merged.append(segment if idx == 0 else segment[1:])
    return np.concatenate(merged, axis=0)


def run_policy(env, generator, num_segments, policy_fn):
    env.reset()
    gt_positions = env.pos_gts.copy()
    position_segments = []
    cumulative_reward = 0.0
    final_info = {}
    done = False

    for _ in range(num_segments):
        state = env.get_state()
        w = policy_fn(state)
        action_seq = generator.generate(current_pos=env.pos_ubs.copy(), w=w)
        trajectory_info = env.step_trajectory(action_seq, action_type="displacement")
        position_segments.append(positions_from_trajectory_info(trajectory_info, env.n_ubs, env.range_pos))
        cumulative_reward += float(trajectory_info["cumulative_reward"])
        final_info = trajectory_info["info"]
        done = bool(trajectory_info["done"])
        if done:
            break

    return dict(
        positions=concat_segment_positions(position_segments),
        gt_positions=gt_positions,
        cumulative_reward=cumulative_reward,
        TotalThroughput=float(final_info.get("TotalThroughput", float("nan"))),
        FairIdx=float(final_info.get("FairIdx", float("nan"))),
        ProbCollision=float(final_info.get("ProbCollision", float("nan"))),
        done=done,
    )


def pairwise_inter_uav_distances(positions):
    n_uavs = positions.shape[1]
    distances = []
    for t in range(positions.shape[0]):
        for i in range(n_uavs):
            for j in range(i + 1, n_uavs):
                distances.append(float(np.linalg.norm(positions[t, i] - positions[t, j])))
    return np.asarray(distances, dtype=np.float64)


def nearest_gt_distances(positions, gt_positions):
    diff = positions[:, :, None, :] - gt_positions[None, None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    return distances.min(axis=-1)


def boundary_hit_ratio(positions, range_pos, tol=1e-5):
    lower_hits = positions <= tol
    upper_hits = positions >= (float(range_pos) - tol)
    return float(np.mean(lower_hits | upper_hits))


def finite_check(values):
    return all(math.isfinite(float(value)) for value in values)


def diagnose_policy(policy_type, result, range_pos):
    positions = np.asarray(result["positions"], dtype=np.float64)
    gt_positions = np.asarray(result["gt_positions"], dtype=np.float64)
    steps = np.diff(positions, axis=0)
    step_lengths = np.linalg.norm(steps, axis=-1)
    final_displacement = np.linalg.norm(positions[-1] - positions[0], axis=-1)
    path_lengths = step_lengths.sum(axis=0)
    mean_step_lengths = step_lengths.mean(axis=0)
    max_step_lengths = step_lengths.max(axis=0)
    inter_uav_distances = pairwise_inter_uav_distances(positions)
    nearest_gt = nearest_gt_distances(positions, gt_positions)
    final_nearest_gt = nearest_gt[-1]

    numeric_values = [
        result["cumulative_reward"],
        result["TotalThroughput"],
        result["FairIdx"],
        result["ProbCollision"],
        boundary_hit_ratio(positions, range_pos),
        inter_uav_distances.min(),
        inter_uav_distances.mean(),
        nearest_gt.mean(),
        final_nearest_gt.mean(),
    ]
    numeric_values.extend(path_lengths.tolist())
    numeric_values.extend(final_displacement.tolist())
    numeric_values.extend(mean_step_lengths.tolist())
    numeric_values.extend(max_step_lengths.tolist())
    numeric_values.extend(positions.reshape(-1).tolist())

    diagnostics = dict(
        policy_type=policy_type,
        cumulative_reward=float(result["cumulative_reward"]),
        TotalThroughput=float(result["TotalThroughput"]),
        FairIdx=float(result["FairIdx"]),
        ProbCollision=float(result["ProbCollision"]),
        min_inter_uav_distance=float(inter_uav_distances.min()),
        mean_inter_uav_distance=float(inter_uav_distances.mean()),
        mean_nearest_gt_distance=float(nearest_gt.mean()),
        final_nearest_gt_distance=float(final_nearest_gt.mean()),
        boundary_hit_ratio=boundary_hit_ratio(positions, range_pos),
        trajectory_finite=finite_check(numeric_values),
    )

    for uav_id in range(positions.shape[1]):
        diagnostics[f"path_length_uav{uav_id}"] = float(path_lengths[uav_id])
        diagnostics[f"final_displacement_uav{uav_id}"] = float(final_displacement[uav_id])
        diagnostics[f"mean_step_length_uav{uav_id}"] = float(mean_step_lengths[uav_id])
        diagnostics[f"max_step_length_uav{uav_id}"] = float(max_step_lengths[uav_id])
        diagnostics[f"final_nearest_gt_distance_uav{uav_id}"] = float(final_nearest_gt[uav_id])

    return diagnostics


def write_diagnostics(path, rows):
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_policy_summary(row):
    print(f"{row['policy_type']}:")
    keys = [
        "cumulative_reward",
        "TotalThroughput",
        "FairIdx",
        "ProbCollision",
        "min_inter_uav_distance",
        "mean_inter_uav_distance",
        "mean_nearest_gt_distance",
        "final_nearest_gt_distance",
        "boundary_hit_ratio",
        "trajectory_finite",
    ]
    for key in keys:
        print(f"  {key}: {row[key]}")

    uav_ids = sorted(
        int(key.replace("path_length_uav", ""))
        for key in row
        if key.startswith("path_length_uav")
    )
    for uav_id in uav_ids:
        print(
            f"  UAV {uav_id}: "
            f"path_length={row[f'path_length_uav{uav_id}']}, "
            f"final_displacement={row[f'final_displacement_uav{uav_id}']}, "
            f"mean_step_length={row[f'mean_step_length_uav{uav_id}']}, "
            f"max_step_length={row[f'max_step_length_uav{uav_id}']}, "
            f"final_nearest_gt_distance={row[f'final_nearest_gt_distance_uav{uav_id}']}"
        )


def behavior_flags(row, max_move_per_slot):
    path_lengths = [
        float(value)
        for key, value in row.items()
        if key.startswith("path_length_uav")
    ]
    max_steps = [
        float(value)
        for key, value in row.items()
        if key.startswith("max_step_length_uav")
    ]
    return dict(
        appears_stationary=all(length < 1e-5 for length in path_lengths),
        appears_boundary_stuck=float(row["boundary_hit_ratio"]) > 0.2,
        has_uav_overlap=float(row["min_inter_uav_distance"]) < 1e-5,
        has_abnormal_jump=any(step > max_move_per_slot + 1e-5 for step in max_steps),
    )


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

    env = MultiUbsCoverageTrajEnv(map_id=args.map_id, record=False)
    env.reset()
    actor, max_move_per_slot = build_actor(env, args.horizon, args.checkpoint)
    generator = LinearWaypointTrajectoryGenerator(
        H=args.horizon,
        n_uavs=env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=env.range_pos,
    )

    deterministic_result = run_policy(
        env=env,
        generator=generator,
        num_segments=args.num_segments,
        policy_fn=lambda state: deterministic_waypoint(actor, state, env.n_ubs),
    )
    random_result = run_policy(
        env=env,
        generator=generator,
        num_segments=args.num_segments,
        policy_fn=lambda _: random_waypoint(rng, env.n_ubs, args.horizon, max_move_per_slot),
    )

    rows = [
        diagnose_policy("deterministic_actor", deterministic_result, env.range_pos),
        diagnose_policy("random_baseline", random_result, env.range_pos),
    ]
    output_path = output_dir / "trajectory_diagnostics.csv"
    write_diagnostics(output_path, rows)

    print(f"diagnostics_csv: {output_path}")
    for row in rows:
        print_policy_summary(row)
        flags = behavior_flags(row, max_move_per_slot)
        print("  behavior_flags:")
        for key, value in flags.items():
            print(f"    {key}: {value}")

    any_nonfinite = not all(bool(row["trajectory_finite"]) for row in rows)
    print(f"NaN or Inf present: {any_nonfinite}")

    if any_nonfinite:
        raise RuntimeError("Trajectory diagnostics contain NaN or Inf.")


if __name__ == "__main__":
    main()
