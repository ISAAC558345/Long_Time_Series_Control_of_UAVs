from pathlib import Path
import argparse
import math
import os
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.dont_write_bytecode = True


def parse_args():
    parser = argparse.ArgumentParser(description="Plot minimal TOP-ERL trajectory diagnostics.")
    parser.add_argument("--checkpoint", default="outputs/top_erl_sanity_50/checkpoints/actor.pt")
    parser.add_argument("--map_id", default="debug")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--num_segments", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs/top_erl_sanity_50/figures")
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
        if idx == 0:
            merged.append(segment)
        else:
            merged.append(segment[1:])
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


def plot_single_trajectory(plt, result, title, output_path, range_pos):
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_ground_users(ax, result["gt_positions"])
    plot_uav_paths(ax, result["positions"], label_prefix="无人机", linestyle="-", show_endpoint_labels=True)
    finish_axes(ax, title, range_pos)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_comparison(plt, det_result, random_result, output_path, range_pos):
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_ground_users(ax, det_result["gt_positions"])
    plot_uav_paths(ax, det_result["positions"], label_prefix="Actor 无人机", linestyle="-", show_endpoint_labels=False)
    plot_uav_paths(ax, random_result["positions"], label_prefix="随机基线无人机", linestyle="--", show_endpoint_labels=False)
    finish_axes(ax, "Actor 与随机基线的无人机轨迹对比", range_pos)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_ground_users(ax, gt_positions):
    ax.scatter(
        gt_positions[:, 0],
        gt_positions[:, 1],
        c="black",
        marker="x",
        s=28,
        label="地面用户",
        alpha=0.85,
    )


def plot_uav_paths(ax, positions, label_prefix, linestyle, show_endpoint_labels):
    n_uavs = positions.shape[1]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for uav_id in range(n_uavs):
        path = positions[:, uav_id, :]
        color = colors[uav_id % len(colors)]
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
            label=f"{label_prefix} {uav_id}",
        )
        start_label = "起点" if show_endpoint_labels and uav_id == 0 else None
        end_label = "终点" if show_endpoint_labels and uav_id == 0 else None
        ax.scatter(path[0, 0], path[0, 1], color=color, marker="o", s=42, label=start_label)
        ax.scatter(path[-1, 0], path[-1, 1], color=color, marker="s", s=42, label=end_label)


def finish_axes(ax, title, range_pos):
    ax.set_title(title)
    ax.set_xlabel("X 坐标")
    ax.set_ylabel("Y 坐标")
    ax.set_xlim(0, range_pos)
    ax.set_ylim(0, range_pos)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def metrics_are_finite(results):
    values = []
    for result in results:
        values.extend([
            result["cumulative_reward"],
            result["TotalThroughput"],
            result["FairIdx"],
            result["ProbCollision"],
        ])
        values.extend(result["positions"].reshape(-1).tolist())
    return all(math.isfinite(float(value)) for value in values)


def main():
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = output_dir / ".mplconfig"
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

    import torch

    from envs.mubs_cov.mubs_cov_traj import MultiUbsCoverageTrajEnv
    from models.trajectory_generator import LinearWaypointTrajectoryGenerator

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = MultiUbsCoverageTrajEnv(map_id=args.map_id, record=False)
    env.reset()
    actor, max_move_per_slot = build_actor(env, args.horizon, args.checkpoint)
    generator = LinearWaypointTrajectoryGenerator(
        H=args.horizon,
        n_uavs=env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=env.range_pos,
    )

    det_result = run_policy(
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

    det_path = output_dir / "trajectory_deterministic_actor.png"
    random_path = output_dir / "trajectory_random_baseline.png"
    comparison_path = output_dir / "trajectory_comparison.png"

    plot_single_trajectory(plt, det_result, "确定性 Actor 控制下的无人机轨迹", det_path, env.range_pos)
    plot_single_trajectory(plt, random_result, "随机基线下的无人机轨迹", random_path, env.range_pos)
    plot_comparison(plt, det_result, random_result, comparison_path, env.range_pos)

    all_finite = metrics_are_finite([det_result, random_result])
    print("deterministic actor:")
    print(f"  cumulative_reward: {det_result['cumulative_reward']}")
    print(f"  TotalThroughput: {det_result['TotalThroughput']}")
    print(f"  FairIdx: {det_result['FairIdx']}")
    print(f"  ProbCollision: {det_result['ProbCollision']}")
    print("random baseline:")
    print(f"  cumulative_reward: {random_result['cumulative_reward']}")
    print(f"  TotalThroughput: {random_result['TotalThroughput']}")
    print(f"  FairIdx: {random_result['FairIdx']}")
    print(f"  ProbCollision: {random_result['ProbCollision']}")
    print("saved figures:")
    print(f"  {det_path}")
    print(f"  {random_path}")
    print(f"  {comparison_path}")
    print(f"NaN or Inf present: {not all_finite}")

    if not all_finite:
        raise RuntimeError("Trajectory diagnostics contain NaN or Inf.")


if __name__ == "__main__":
    main()
