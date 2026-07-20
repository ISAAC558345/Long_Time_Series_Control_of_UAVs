from copy import deepcopy
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
    parser = argparse.ArgumentParser(description="Minimal TOP-ERL-inspired training script.")
    parser.add_argument("--map_id", default="debug")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--segment_length", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=20)
    parser.add_argument("--rollouts_per_iteration", type=int, default=1)
    parser.add_argument("--critic_warmup_iterations", type=int, default=20)
    parser.add_argument("--critic_updates_per_iteration", type=int, default=4)
    parser.add_argument("--actor_updates_per_iteration", type=int, default=1)
    parser.add_argument("--actor_update_interval", type=int, default=2)
    parser.add_argument("--min_buffer_rollouts_before_actor", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--target_tau", type=float, default=0.01)
    parser.add_argument(
        "--target_mode",
        default="bootstrapped_n_step",
        choices=["bootstrapped_n_step", "no_bootstrap", "weighted_bootstrap"],
    )
    parser.add_argument("--bootstrap_weight", type=float, default=1.0)
    parser.add_argument("--reward_normalization", default="running_mean_abs", choices=["none", "mean_abs", "running_mean_abs"])
    parser.add_argument("--reward_clip", type=float, default=5.0)
    parser.add_argument("--log_std_start", type=float, default=1.0)
    parser.add_argument("--log_std_end", type=float, default=-1.0)
    parser.add_argument("--replay_capacity", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs/top_erl_minimal")
    return parser.parse_args()


def validate_args(args):
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")
    if args.segment_length <= 0:
        raise ValueError("--segment_length must be positive.")
    if args.segment_length > args.horizon:
        raise ValueError("--segment_length must be <= --horizon for this minimal script.")
    if args.num_iterations <= 0:
        raise ValueError("--num_iterations must be positive.")
    if args.rollouts_per_iteration <= 0:
        raise ValueError("--rollouts_per_iteration must be positive.")
    if args.critic_updates_per_iteration < 0:
        raise ValueError("--critic_updates_per_iteration must be non-negative.")
    if args.critic_warmup_iterations < 0:
        raise ValueError("--critic_warmup_iterations must be non-negative.")
    if args.actor_updates_per_iteration < 0:
        raise ValueError("--actor_updates_per_iteration must be non-negative.")
    if args.actor_update_interval <= 0:
        raise ValueError("--actor_update_interval must be positive.")
    if args.min_buffer_rollouts_before_actor < 0:
        raise ValueError("--min_buffer_rollouts_before_actor must be non-negative.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.bootstrap_weight < 0.0:
        raise ValueError("--bootstrap_weight must be non-negative.")
    if args.reward_clip <= 0.0:
        raise ValueError("--reward_clip must be positive.")
    if args.replay_capacity <= 0:
        raise ValueError("--replay_capacity must be positive.")
    if not 0.0 <= args.target_tau <= 1.0:
        raise ValueError("--target_tau must be in [0, 1].")


def collect_actor_rollout(env, actor, generator):
    import torch

    env.reset()
    state = env.get_state()
    state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        w, _, _, _ = actor.sample(state_tensor)

    # This tensor-to-numpy conversion is only for environment interaction.
    # Actor gradients are computed later through the differentiable generator.
    w_np = w[0].cpu().numpy()
    action_seq = generator.generate(current_pos=env.pos_ubs.copy(), w=w_np)
    return env.step_trajectory(action_seq, action_type="displacement")


def to_tensor_batch(batch):
    import torch

    return dict(
        start_state=torch.as_tensor(batch["start_state"], dtype=torch.float32),
        action_subseq=torch.as_tensor(batch["action_subseq"], dtype=torch.float32),
        reward_subseq=torch.as_tensor(batch["reward_subseq"], dtype=torch.float32),
        bootstrap_state=torch.as_tensor(batch["bootstrap_state"], dtype=torch.float32),
        done_subseq=torch.as_tensor(batch["done_subseq"], dtype=torch.bool),
        state_subseq=torch.as_tensor(batch["state_subseq"], dtype=torch.float32),
    )


def soft_update(target, source, tau):
    with __import__("torch").no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def mean_metric(values):
    return float(np.mean(values)) if values else float("nan")


def write_log(log_path, rows):
    fieldnames = [
        "iteration",
        "rollout_cumulative_reward",
        "TotalThroughput",
        "FairIdx",
        "ProbCollision",
        "critic_loss",
        "actor_loss",
        "actor_updated",
        "critic_updates",
        "actor_updates",
        "target_mode",
        "bootstrap_weight",
        "reward_normalization",
        "reward_clip",
        "log_std_schedule",
        "replay_buffer_size",
    ]
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoints(checkpoint_dir, actor, critic, target_critic):
    import torch

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), checkpoint_dir / "actor.pt")
    torch.save(critic.state_dict(), checkpoint_dir / "critic.pt")
    torch.save(target_critic.state_dict(), checkpoint_dir / "target_critic.pt")


def format_metric(value):
    return "NA" if value == "" else f"{float(value):.6f}"


def main():
    import torch

    from buffers.trajectory_replay_buffer import TrajectoryReplayBuffer
    from envs.mubs_cov.mubs_cov_traj import MultiUbsCoverageTrajEnv
    from models.actor_loss import compute_actor_loss
    from models.critic_loss import RunningRewardNormalizer, compute_critic_loss
    from models.trajectory_actor import TrajectoryActor
    from models.trajectory_generator import (
        LinearWaypointTrajectoryGenerator,
        TorchLinearWaypointTrajectoryGenerator,
    )
    from models.transformer_critic import TrajectoryTransformerCritic

    args = parse_args()
    validate_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.csv"
    checkpoint_dir = output_dir / "checkpoints"

    env = MultiUbsCoverageTrajEnv(map_id=args.map_id, record=False)
    env.reset()

    state_dim = env.get_state().shape[-1]
    max_move_per_slot = float(np.max(np.asarray(env.vels, dtype=np.float32)) * env.dt)
    actor = TrajectoryActor(
        state_dim=state_dim,
        n_uavs=env.n_ubs,
        hidden_dim=64,
        w_scale=args.horizon * max_move_per_slot,
    )
    critic = TrajectoryTransformerCritic(
        state_dim=state_dim,
        n_uavs=env.n_ubs,
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_segment_length=max(args.horizon, args.segment_length),
    )
    target_critic = deepcopy(critic)
    target_critic.eval()

    rollout_generator = LinearWaypointTrajectoryGenerator(
        H=args.horizon,
        n_uavs=env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=env.range_pos,
    )
    torch_generator = TorchLinearWaypointTrajectoryGenerator(
        H=args.horizon,
        n_uavs=env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=env.range_pos,
    )
    replay_buffer = TrajectoryReplayBuffer(capacity=args.replay_capacity, seed=args.seed)
    reward_normalizer = RunningRewardNormalizer(clip_value=args.reward_clip)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    log_rows = []
    last_critic_loss = float("nan")
    last_actor_loss = ""

    for iteration in range(args.num_iterations):
        progress = iteration / max(args.num_iterations - 1, 1)
        log_std_schedule = args.log_std_start + (args.log_std_end - args.log_std_start) * progress
        actor.set_log_std_schedule(log_std_schedule)

        rollout_rewards = []
        throughputs = []
        fairness_values = []
        collision_values = []
        critic_updates_done = 0
        actor_updates_done = 0
        actor_updated = False
        iteration_actor_loss = ""

        for _ in range(args.rollouts_per_iteration):
            trajectory_info = collect_actor_rollout(env, actor, rollout_generator)
            replay_buffer.add_rollout(trajectory_info)
            info = trajectory_info["info"]
            rollout_rewards.append(float(trajectory_info["cumulative_reward"]))
            throughputs.append(float(info.get("TotalThroughput", float("nan"))))
            fairness_values.append(float(info.get("FairIdx", float("nan"))))
            collision_values.append(float(info.get("ProbCollision", float("nan"))))

        for _ in range(args.critic_updates_per_iteration):
            batch = replay_buffer.sample_segments(
                batch_size=args.batch_size,
                segment_length=args.segment_length,
            )
            tensors = to_tensor_batch(batch)
            critic_loss, _ = compute_critic_loss(
                critic=critic,
                target_critic=target_critic,
                start_state=tensors["start_state"],
                action_subseq=tensors["action_subseq"],
                reward_subseq=tensors["reward_subseq"],
                bootstrap_state=tensors["bootstrap_state"],
                done_subseq=tensors["done_subseq"],
                state_subseq=tensors["state_subseq"],
                gamma=args.gamma,
                target_mode=args.target_mode,
                bootstrap_weight=args.bootstrap_weight,
                reward_normalization=args.reward_normalization,
                reward_normalizer=reward_normalizer,
                reward_clip=args.reward_clip,
            )
            if not torch.isfinite(critic_loss):
                raise RuntimeError("critic_loss became NaN or Inf.")

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            last_critic_loss = float(critic_loss.item())
            critic_updates_done += 1

            soft_update(target_critic, critic, args.target_tau)
            target_critic.eval()

        actor_can_update = (
            iteration >= args.critic_warmup_iterations
            and len(replay_buffer) >= args.min_buffer_rollouts_before_actor
            and iteration % args.actor_update_interval == 0
        )
        if actor_can_update:
            for _ in range(args.actor_updates_per_iteration):
                batch = replay_buffer.sample_segments(
                    batch_size=args.batch_size,
                    segment_length=args.segment_length,
                )
                start_state = torch.as_tensor(batch["start_state"], dtype=torch.float32)
                actor_loss, actor_info = compute_actor_loss(
                    actor=actor,
                    critic=critic,
                    start_state=start_state,
                    n_uavs=env.n_ubs,
                    H=torch_generator.H,
                    range_pos=env.range_pos,
                    max_move_per_slot=max_move_per_slot,
                )
                if not torch.isfinite(actor_loss):
                    raise RuntimeError("actor_loss became NaN or Inf.")

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                last_actor_loss = float(actor_loss.item())
                iteration_actor_loss = last_actor_loss
                actor_updated = True
                actor_updates_done += 1

                if actor_info["continuous_action_seq"].shape[1] != args.horizon:
                    raise RuntimeError("actor action horizon shape mismatch.")

        row = dict(
            iteration=iteration,
            rollout_cumulative_reward=mean_metric(rollout_rewards),
            TotalThroughput=mean_metric(throughputs),
            FairIdx=mean_metric(fairness_values),
            ProbCollision=mean_metric(collision_values),
            critic_loss=last_critic_loss,
            actor_loss=iteration_actor_loss,
            actor_updated=actor_updated,
            critic_updates=critic_updates_done,
            actor_updates=actor_updates_done,
            target_mode=args.target_mode,
            bootstrap_weight=args.bootstrap_weight,
            reward_normalization=args.reward_normalization,
            reward_clip=args.reward_clip,
            log_std_schedule=log_std_schedule,
            replay_buffer_size=len(replay_buffer),
        )
        log_rows.append(row)
        print(
            "iteration {iteration}: reward={rollout_cumulative_reward:.6f}, "
            "throughput={TotalThroughput:.6f}, fair={FairIdx:.6f}, "
            "collision={ProbCollision:.6f}, critic_loss={critic_loss}, "
            "actor_loss={actor_loss}, actor_updated={actor_updated}, "
            "critic_updates={critic_updates}, actor_updates={actor_updates}, "
            "buffer={replay_buffer_size}".format(
                iteration=row["iteration"],
                rollout_cumulative_reward=row["rollout_cumulative_reward"],
                TotalThroughput=row["TotalThroughput"],
                FairIdx=row["FairIdx"],
                ProbCollision=row["ProbCollision"],
                critic_loss=format_metric(row["critic_loss"]),
                actor_loss=format_metric(row["actor_loss"]),
                actor_updated=row["actor_updated"],
                critic_updates=row["critic_updates"],
                actor_updates=row["actor_updates"],
                replay_buffer_size=row["replay_buffer_size"],
            )
        )

    write_log(log_path, log_rows)
    save_checkpoints(checkpoint_dir, actor, critic, target_critic)

    final_row = log_rows[-1]
    finite_values = [
        final_row["rollout_cumulative_reward"],
        final_row["TotalThroughput"],
        final_row["FairIdx"],
        final_row["ProbCollision"],
        final_row["critic_loss"],
    ]
    if final_row["actor_loss"] != "":
        finite_values.append(final_row["actor_loss"])
    all_finite = bool(np.isfinite(finite_values).all())

    print(f"train_log: {log_path}")
    print(f"actor checkpoint: {checkpoint_dir / 'actor.pt'}")
    print(f"critic checkpoint: {checkpoint_dir / 'critic.pt'}")
    print(f"target critic checkpoint: {checkpoint_dir / 'target_critic.pt'}")
    print(f"final iteration: {final_row}")
    print(f"all final metrics finite: {all_finite}")

    if not all_finite:
        raise RuntimeError("Final logged metrics contain NaN or Inf.")


if __name__ == "__main__":
    main()
