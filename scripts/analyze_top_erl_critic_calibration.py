from copy import deepcopy
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
    parser = argparse.ArgumentParser(description="Critic calibration and Q-ranking diagnostics.")
    parser.add_argument("--base_dir", default="outputs/top_erl_multiseed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--map_id", default="debug")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--num_states", type=int, default=20)
    parser.add_argument("--num_candidates", type=int, default=32)
    parser.add_argument("--output_dir", default="outputs/top_erl_multiseed/critic_calibration")
    return parser.parse_args()


def validate_args(args):
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")
    if args.num_states <= 0:
        raise ValueError("--num_states must be positive.")
    if args.num_candidates < 3:
        raise ValueError("--num_candidates must be at least 3.")


def build_actor_and_critic(env, horizon, checkpoint_dir):
    import torch

    from models.trajectory_actor import TrajectoryActor
    from models.transformer_critic import TrajectoryTransformerCritic

    state_dim = env.get_state().shape[-1]
    max_move_per_slot = float(np.max(np.asarray(env.vels, dtype=np.float32)) * env.dt)

    actor = TrajectoryActor(
        state_dim=state_dim,
        n_uavs=env.n_ubs,
        hidden_dim=64,
        w_scale=horizon * max_move_per_slot,
    )
    actor.load_state_dict(torch.load(checkpoint_dir / "actor.pt", map_location="cpu"))
    actor.eval()

    critic = TrajectoryTransformerCritic(
        state_dim=state_dim,
        n_uavs=env.n_ubs,
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_segment_length=horizon,
    )
    critic.load_state_dict(torch.load(checkpoint_dir / "critic.pt", map_location="cpu"))
    critic.eval()
    return actor, critic, max_move_per_slot


def random_waypoint(rng, n_uavs, horizon, max_move_per_slot):
    limit = horizon * max_move_per_slot
    return rng.uniform(-limit, limit, size=(n_uavs, 2)).astype(np.float32)


def score_action_batch(critic, start_state, action_batch):
    import torch

    state_batch = np.repeat(start_state[None, :], action_batch.shape[0], axis=0)
    with torch.no_grad():
        start_state_t = torch.as_tensor(state_batch, dtype=torch.float32)
        action_t = torch.as_tensor(action_batch, dtype=torch.float32)
        _, prefix_q_values = critic(start_state_t, action_t)
        q_mean = prefix_q_values.mean(dim=1)
        q_last = prefix_q_values[:, -1]
    return q_mean.cpu().numpy(), q_last.cpu().numpy()


def execute_from_snapshot(env_snapshot, action_seq):
    env = deepcopy(env_snapshot)
    trajectory_info = env.step_trajectory(action_seq, action_type="displacement")
    info = trajectory_info["info"]
    return dict(
        cumulative_reward=float(trajectory_info["cumulative_reward"]),
        TotalThroughput=float(info.get("TotalThroughput", float("nan"))),
        FairIdx=float(info.get("FairIdx", float("nan"))),
        ProbCollision=float(info.get("ProbCollision", float("nan"))),
    )


def finite_values(values):
    return all(math.isfinite(float(value)) for value in values)


def make_candidate_row(seed, state_id, candidate_id, candidate_type, q_mean, q_last, metrics):
    values = [
        q_mean,
        q_last,
        metrics["cumulative_reward"],
        metrics["TotalThroughput"],
        metrics["FairIdx"],
        metrics["ProbCollision"],
    ]
    return dict(
        seed=seed,
        state_id=state_id,
        candidate_id=candidate_id,
        candidate_type=candidate_type,
        predicted_q_mean=float(q_mean),
        predicted_q_last=float(q_last),
        cumulative_reward=metrics["cumulative_reward"],
        TotalThroughput=metrics["TotalThroughput"],
        FairIdx=metrics["FairIdx"],
        ProbCollision=metrics["ProbCollision"],
        finite_check=finite_values(values),
    )


def generate_candidates(actor, generator, rng, start_state, current_pos, n_uavs, horizon, max_move_per_slot, num_candidates):
    import torch

    n_stochastic = (num_candidates - 1) // 2
    n_random = num_candidates - 1 - n_stochastic
    state_tensor = torch.as_tensor(start_state, dtype=torch.float32).unsqueeze(0)
    candidates = []

    with torch.no_grad():
        mean, _ = actor(state_tensor)
        det_w = torch.tanh(mean).reshape(1, n_uavs, 2) * actor.w_scale
    det_action = generator.generate(current_pos=current_pos, w=det_w[0].cpu().numpy())
    candidates.append(("deterministic_actor", det_action))

    if n_stochastic > 0:
        state_batch = state_tensor.repeat(n_stochastic, 1)
        with torch.no_grad():
            stochastic_w, _, _, _ = actor.sample(state_batch)
        for idx in range(n_stochastic):
            action = generator.generate(current_pos=current_pos, w=stochastic_w[idx].cpu().numpy())
            candidates.append(("actor_stochastic", action))

    for _ in range(n_random):
        w = random_waypoint(rng, n_uavs, horizon, max_move_per_slot)
        action = generator.generate(current_pos=current_pos, w=w)
        candidates.append(("random_baseline", action))

    candidate_types = [item[0] for item in candidates]
    action_batch = np.stack([item[1] for item in candidates]).astype(np.float32)
    return candidate_types, action_batch


def advance_sampler_env(env, generator, rng, horizon, max_move_per_slot):
    w = random_waypoint(rng, env.n_ubs, horizon, max_move_per_slot)
    action_seq = generator.generate(current_pos=env.pos_ubs.copy(), w=w)
    trajectory_info = env.step_trajectory(action_seq, action_type="displacement")
    if trajectory_info["done"]:
        env.reset()


def evaluate_seed(seed, args):
    from envs.mubs_cov.mubs_cov_traj import MultiUbsCoverageTrajEnv
    from models.trajectory_generator import LinearWaypointTrajectoryGenerator

    seed_dir = Path(args.base_dir) / f"seed_{seed}"
    checkpoint_dir = seed_dir / "checkpoints"
    sampler_env = MultiUbsCoverageTrajEnv(map_id=args.map_id, record=False)
    sampler_env.reset()
    actor, critic, max_move_per_slot = build_actor_and_critic(sampler_env, args.horizon, checkpoint_dir)
    generator = LinearWaypointTrajectoryGenerator(
        H=args.horizon,
        n_uavs=sampler_env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=sampler_env.range_pos,
    )
    rng = np.random.default_rng(seed)
    rows = []
    top1_regrets = []

    for state_id in range(args.num_states):
        env_snapshot = deepcopy(sampler_env)
        start_state = env_snapshot.get_state().copy()
        current_pos = env_snapshot.pos_ubs.copy()
        candidate_types, action_batch = generate_candidates(
            actor=actor,
            generator=generator,
            rng=rng,
            start_state=start_state,
            current_pos=current_pos,
            n_uavs=env_snapshot.n_ubs,
            horizon=args.horizon,
            max_move_per_slot=max_move_per_slot,
            num_candidates=args.num_candidates,
        )
        q_mean, q_last = score_action_batch(critic, start_state, action_batch)

        state_rewards = []
        state_rows = []
        for candidate_id, (candidate_type, action_seq) in enumerate(zip(candidate_types, action_batch)):
            metrics = execute_from_snapshot(env_snapshot, action_seq)
            row = make_candidate_row(
                seed=seed,
                state_id=state_id,
                candidate_id=candidate_id,
                candidate_type=candidate_type,
                q_mean=q_mean[candidate_id],
                q_last=q_last[candidate_id],
                metrics=metrics,
            )
            state_rows.append(row)
            state_rewards.append(metrics["cumulative_reward"])

        predicted_top_idx = int(np.argmax(q_mean))
        actual_top_reward = float(np.max(state_rewards))
        predicted_top_reward = float(state_rewards[predicted_top_idx])
        top1_regrets.append(actual_top_reward - predicted_top_reward)
        rows.extend(state_rows)

        advance_sampler_env(sampler_env, generator, rng, args.horizon, max_move_per_slot)

    return rows, top1_regrets


def pearson_corr(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 2 or ys.size < 2:
        return float("nan")
    if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def summarize_seed(seed, rows, regrets):
    seed_rows = [row for row in rows if int(row["seed"]) == seed]
    q_mean = np.asarray([float(row["predicted_q_mean"]) for row in seed_rows], dtype=np.float64)
    q_last = np.asarray([float(row["predicted_q_last"]) for row in seed_rows], dtype=np.float64)
    reward = np.asarray([float(row["cumulative_reward"]) for row in seed_rows], dtype=np.float64)
    throughput = np.asarray([float(row["TotalThroughput"]) for row in seed_rows], dtype=np.float64)
    fairness = np.asarray([float(row["FairIdx"]) for row in seed_rows], dtype=np.float64)

    return dict(
        seed=seed,
        q_reward_corr_mean=pearson_corr(q_mean, reward),
        q_last_reward_corr=pearson_corr(q_last, reward),
        q_throughput_corr=pearson_corr(q_mean, throughput),
        q_fairness_corr=pearson_corr(q_mean, fairness),
        mean_top1_regret=float(np.mean(regrets)),
        predicted_q_mean=float(np.mean(q_mean)),
        predicted_q_std=float(np.std(q_mean)),
        predicted_q_min=float(np.min(q_mean)),
        predicted_q_max=float(np.max(q_mean)),
        actual_reward_mean=float(np.mean(reward)),
        actual_reward_std=float(np.std(reward)),
        actual_reward_min=float(np.min(reward)),
        actual_reward_max=float(np.max(reward)),
        all_finite=all(row["finite_check"] for row in seed_rows),
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def describe_corr(value):
    if not math.isfinite(float(value)):
        return "unreliable"
    if value > 0.2:
        return "positive"
    if value < -0.05:
        return "negative"
    return "weak"


def build_report(summary_rows):
    lines = [
        "# Critic calibration / Q-ranking analysis",
        "",
        "This report evaluates saved critic checkpoints only. It does not retrain or modify the algorithm.",
        "",
        "## Summary",
        "",
        "| seed | q-reward corr | q-last reward corr | q-throughput corr | q-fairness corr | mean top1 regret | q std | reward std | finite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {seed} | {q_reward:.6f} | {q_last:.6f} | {q_thr:.6f} | {q_fair:.6f} | {regret:.6f} | {q_std:.6f} | {reward_std:.6f} | {finite} |".format(
                seed=row["seed"],
                q_reward=row["q_reward_corr_mean"],
                q_last=row["q_last_reward_corr"],
                q_thr=row["q_throughput_corr"],
                q_fair=row["q_fairness_corr"],
                regret=row["mean_top1_regret"],
                q_std=row["predicted_q_std"],
                reward_std=row["actual_reward_std"],
                finite=row["all_finite"],
            )
        )

    lines.extend(["", "## Per-seed diagnosis", ""])
    for row in summary_rows:
        corr_type = describe_corr(row["q_reward_corr_mean"])
        constant_like = row["predicted_q_std"] < 1e-6
        lines.extend([
            f"### Seed {row['seed']}",
            f"- Q-reward correlation is {corr_type}: {row['q_reward_corr_mean']}",
            f"- Q output is {'near-constant' if constant_like else 'variable'}; predicted_q_std={row['predicted_q_std']}",
            f"- mean top-1 regret: {row['mean_top1_regret']}",
            f"- Q-throughput correlation: {row['q_throughput_corr']}",
            f"- Q-fairness correlation: {row['q_fairness_corr']}",
        ])
        if row["q_reward_corr_mean"] < 0:
            lines.append("- Ranking direction is likely wrong or noisy for reward.")
        elif row["q_reward_corr_mean"] < 0.1:
            lines.append("- Ranking is weak; top candidate should not be trusted yet.")
        else:
            lines.append("- Ranking has some positive signal, but should still be validated.")
        lines.append("")

    lines.extend([
        "## Diagnostic answers",
        "",
        "- The critic Q-ranking is not consistently trustworthy when Q-reward correlation is weak or negative.",
        "- If Q outputs are variable but correlations are negative, the problem is more direction/ranking than constant-output collapse.",
        "- Low or negative Q-throughput and Q-fairness correlations indicate the critic is not reliably predicting either communication objective.",
        "- Next priority should be N-step target / critic loss and critic update schedule, including critic warmup or delayed actor update. Reward scaling is also worth checking before changing trajectory generator expressiveness.",
        "- Deterministic action extraction remains relevant, but this diagnostic directly points to critic target / Q ranking as a bottleneck.",
        "- This is not a final algorithm conclusion; it only diagnoses the current minimal implementation.",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    regrets_by_seed = {}
    for seed in args.seeds:
        rows, regrets = evaluate_seed(seed, args)
        all_rows.extend(rows)
        regrets_by_seed[seed] = regrets

    candidate_path = output_dir / "critic_calibration_candidates.csv"
    summary_path = output_dir / "critic_calibration_summary.csv"
    report_path = output_dir / "critic_calibration_analysis.md"
    write_csv(candidate_path, all_rows)
    summary_rows = [summarize_seed(seed, all_rows, regrets_by_seed[seed]) for seed in args.seeds]
    write_csv(summary_path, summary_rows)
    report_path.write_text(build_report(summary_rows), encoding="utf-8")

    print(f"critic_calibration_candidates_csv: {candidate_path}")
    print(f"critic_calibration_summary_csv: {summary_path}")
    print(f"critic_calibration_report: {report_path}")
    for row in summary_rows:
        print(
            "seed {seed}: q_reward_corr={q_reward:.6f}, q_last_reward_corr={q_last:.6f}, "
            "q_throughput_corr={q_thr:.6f}, q_fairness_corr={q_fair:.6f}, "
            "top1_regret={regret:.6f}, q_std={q_std:.6f}, reward_std={reward_std:.6f}, finite={finite}".format(
                seed=row["seed"],
                q_reward=row["q_reward_corr_mean"],
                q_last=row["q_last_reward_corr"],
                q_thr=row["q_throughput_corr"],
                q_fair=row["q_fairness_corr"],
                regret=row["mean_top1_regret"],
                q_std=row["predicted_q_std"],
                reward_std=row["actual_reward_std"],
                finite=row["all_finite"],
            )
        )


if __name__ == "__main__":
    main()
