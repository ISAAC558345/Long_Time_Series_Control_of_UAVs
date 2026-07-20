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
    parser = argparse.ArgumentParser(description="Critic-guided candidate selection diagnostics.")
    parser.add_argument("--base_dir", default="outputs/top_erl_multiseed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--map_id", default="debug")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--num_candidates", type=int, default=16)
    parser.add_argument("--output_dir", default="outputs/top_erl_multiseed/candidate_selection")
    return parser.parse_args()


def validate_args(args):
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")
    if args.num_episodes <= 0:
        raise ValueError("--num_episodes must be positive.")
    if args.num_candidates <= 0:
        raise ValueError("--num_candidates must be positive.")


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

    if action_batch.ndim == 3:
        action_batch = action_batch[None, ...]
    state_batch = np.repeat(start_state[None, :], action_batch.shape[0], axis=0)
    with torch.no_grad():
        start_state_t = torch.as_tensor(state_batch, dtype=torch.float32)
        action_t = torch.as_tensor(action_batch, dtype=torch.float32)
        _, prefix_q_values = critic(start_state_t, action_t)
        scores = prefix_q_values.mean(dim=1)
    return scores.cpu().numpy()


def execute_action(map_id, action_seq):
    from envs.mubs_cov.mubs_cov_traj import MultiUbsCoverageTrajEnv

    env = MultiUbsCoverageTrajEnv(map_id=map_id, record=False)
    env.reset()
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


def make_row(seed, episode, policy_type, metrics, predicted_q_score, selected_candidate_rank):
    values = [
        metrics["cumulative_reward"],
        metrics["TotalThroughput"],
        metrics["FairIdx"],
        metrics["ProbCollision"],
    ]
    if predicted_q_score is not None:
        values.append(predicted_q_score)

    return dict(
        seed=seed,
        episode=episode,
        policy_type=policy_type,
        cumulative_reward=metrics["cumulative_reward"],
        TotalThroughput=metrics["TotalThroughput"],
        FairIdx=metrics["FairIdx"],
        ProbCollision=metrics["ProbCollision"],
        predicted_q_score="" if predicted_q_score is None else float(predicted_q_score),
        selected_candidate_rank="" if selected_candidate_rank is None else int(selected_candidate_rank),
        finite_check=finite_values(values),
    )


def evaluate_seed(seed, args):
    import torch

    from envs.mubs_cov.mubs_cov_traj import MultiUbsCoverageTrajEnv
    from models.trajectory_generator import LinearWaypointTrajectoryGenerator, TorchLinearWaypointTrajectoryGenerator

    base_dir = Path(args.base_dir)
    checkpoint_dir = base_dir / f"seed_{seed}" / "checkpoints"
    probe_env = MultiUbsCoverageTrajEnv(map_id=args.map_id, record=False)
    probe_env.reset()
    actor, critic, max_move_per_slot = build_actor_and_critic(probe_env, args.horizon, checkpoint_dir)
    generator = LinearWaypointTrajectoryGenerator(
        H=args.horizon,
        n_uavs=probe_env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=probe_env.range_pos,
    )
    torch_generator = TorchLinearWaypointTrajectoryGenerator(
        H=args.horizon,
        n_uavs=probe_env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=probe_env.range_pos,
    )
    rng = np.random.default_rng(seed)

    rows = []
    candidate_scores_for_corr = []
    candidate_rewards_for_corr = []

    for episode in range(args.num_episodes):
        base_env = MultiUbsCoverageTrajEnv(map_id=args.map_id, record=False)
        base_env.reset()
        start_state = base_env.get_state().copy()
        current_pos = base_env.pos_ubs.copy()
        state_tensor = torch.as_tensor(start_state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            mean, _ = actor(state_tensor)
            det_w = torch.tanh(mean).reshape(1, probe_env.n_ubs, 2) * actor.w_scale
            stoch_w, _, _, _ = actor.sample(state_tensor)

        det_action = generator.generate(current_pos=current_pos, w=det_w[0].cpu().numpy())
        stoch_action = generator.generate(current_pos=current_pos, w=stoch_w[0].cpu().numpy())
        random_w = random_waypoint(rng, probe_env.n_ubs, args.horizon, max_move_per_slot)
        random_action = generator.generate(current_pos=current_pos, w=random_w)

        det_score = score_action_batch(critic, start_state, det_action)[0]
        stoch_score = score_action_batch(critic, start_state, stoch_action)[0]
        random_score = score_action_batch(critic, start_state, random_action)[0]

        rows.append(make_row(seed, episode, "deterministic_actor", execute_action(args.map_id, det_action), det_score, None))
        rows.append(make_row(seed, episode, "stochastic_actor_single", execute_action(args.map_id, stoch_action), stoch_score, None))
        rows.append(make_row(seed, episode, "random_baseline", execute_action(args.map_id, random_action), random_score, None))

        state_batch = state_tensor.repeat(args.num_candidates, 1)
        with torch.no_grad():
            candidate_w, _, _, _ = actor.sample(state_batch)
            current_pos_t = torch.as_tensor(current_pos, dtype=torch.float32).unsqueeze(0).repeat(args.num_candidates, 1, 1)
            candidate_actions_t = torch_generator.generate(current_pos=current_pos_t, w=candidate_w)
            candidate_actions = candidate_actions_t.cpu().numpy()
        candidate_scores = score_action_batch(critic, start_state, candidate_actions)
        best_idx = int(np.argmax(candidate_scores))
        best_action = candidate_actions[best_idx]
        best_metrics = execute_action(args.map_id, best_action)
        rows.append(make_row(seed, episode, "critic_selected_actor_topk", best_metrics, candidate_scores[best_idx], 1))

        for candidate_idx in range(args.num_candidates):
            candidate_metrics = execute_action(args.map_id, candidate_actions[candidate_idx])
            candidate_scores_for_corr.append(float(candidate_scores[candidate_idx]))
            candidate_rewards_for_corr.append(float(candidate_metrics["cumulative_reward"]))

    return rows, pearson_corr(candidate_scores_for_corr, candidate_rewards_for_corr)


def pearson_corr(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 2 or ys.size < 2:
        return float("nan")
    if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, correlations):
    summary_rows = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        for policy_type in sorted({row["policy_type"] for row in seed_rows}):
            policy_rows = [row for row in seed_rows if row["policy_type"] == policy_type]
            rewards = np.asarray([float(row["cumulative_reward"]) for row in policy_rows])
            throughputs = np.asarray([float(row["TotalThroughput"]) for row in policy_rows])
            fairness = np.asarray([float(row["FairIdx"]) for row in policy_rows])
            collisions = np.asarray([float(row["ProbCollision"]) for row in policy_rows])
            summary_rows.append(dict(
                seed=seed,
                policy_type=policy_type,
                mean_cumulative_reward=float(np.mean(rewards)),
                std_cumulative_reward=float(np.std(rewards)),
                mean_TotalThroughput=float(np.mean(throughputs)),
                std_TotalThroughput=float(np.std(throughputs)),
                mean_FairIdx=float(np.mean(fairness)),
                std_FairIdx=float(np.std(fairness)),
                mean_ProbCollision=float(np.mean(collisions)),
                all_metrics_finite=all(row["finite_check"] for row in policy_rows),
                candidate_score_reward_corr=correlations.get(seed, float("nan")),
            ))
    return summary_rows


def by_seed_policy(summary_rows):
    grouped = {}
    for row in summary_rows:
        grouped[(int(row["seed"]), row["policy_type"])] = row
    return grouped


def build_report(summary_rows, correlations, seeds):
    grouped = by_seed_policy(summary_rows)
    policies = [
        "deterministic_actor",
        "stochastic_actor_single",
        "critic_selected_actor_topk",
        "random_baseline",
    ]
    lines = [
        "# Critic-guided candidate selection analysis",
        "",
        "This report evaluates saved checkpoints only. It does not retrain or modify the algorithm.",
        "",
        "## Summary table",
        "",
        "| seed | policy | mean reward | mean throughput | mean FairIdx | mean collision | finite |",
        "|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for seed in seeds:
        for policy in policies:
            row = grouped[(seed, policy)]
            lines.append(
                "| {seed} | {policy} | {reward:.6f} | {throughput:.6f} | {fair:.6f} | {collision:.6f} | {finite} |".format(
                    seed=seed,
                    policy=policy,
                    reward=float(row["mean_cumulative_reward"]),
                    throughput=float(row["mean_TotalThroughput"]),
                    fair=float(row["mean_FairIdx"]),
                    collision=float(row["mean_ProbCollision"]),
                    finite=row["all_metrics_finite"],
                )
            )

    lines.extend(["", "## Candidate score-reward correlation", ""])
    for seed in seeds:
        corr = correlations.get(seed, float("nan"))
        lines.append(f"- seed {seed}: {corr}")

    lines.extend(["", "## Per-seed comparisons", ""])
    for seed in seeds:
        det = grouped[(seed, "deterministic_actor")]
        stoch = grouped[(seed, "stochastic_actor_single")]
        topk = grouped[(seed, "critic_selected_actor_topk")]
        rand = grouped[(seed, "random_baseline")]
        topk_reward = float(topk["mean_cumulative_reward"])
        lines.extend([
            f"### Seed {seed}",
            f"- top-k minus deterministic reward: {topk_reward - float(det['mean_cumulative_reward']):.6f}",
            f"- top-k minus stochastic reward: {topk_reward - float(stoch['mean_cumulative_reward']):.6f}",
            f"- top-k minus random reward: {topk_reward - float(rand['mean_cumulative_reward']):.6f}",
            f"- critic score / reward correlation: {correlations.get(seed, float('nan'))}",
            "",
        ])

    lines.extend([
        "## Diagnostic answers",
        "",
        "- If top-k improves over deterministic but not over stochastic/random, deterministic extraction is suspect while critic ranking remains limited.",
        "- If top-k underperforms all baselines or has weak score-reward correlation, critic target / Q ranking should be inspected.",
        "- Positive correlation is useful but not sufficient: the top candidate can still be poor if the actor candidate pool is weak.",
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
    correlations = {}
    for seed in args.seeds:
        rows, corr = evaluate_seed(seed, args)
        all_rows.extend(rows)
        correlations[seed] = corr

    eval_path = output_dir / "candidate_selection_eval.csv"
    summary_path = output_dir / "candidate_selection_summary.csv"
    report_path = output_dir / "candidate_selection_analysis.md"
    write_csv(eval_path, all_rows)
    summary_rows = summarize(all_rows, correlations)
    write_csv(summary_path, summary_rows)
    report_path.write_text(build_report(summary_rows, correlations, args.seeds), encoding="utf-8")

    print(f"candidate_selection_eval_csv: {eval_path}")
    print(f"candidate_selection_summary_csv: {summary_path}")
    print(f"candidate_selection_report: {report_path}")
    grouped = by_seed_policy(summary_rows)
    for seed in args.seeds:
        print(f"seed {seed}:")
        for policy in ("deterministic_actor", "stochastic_actor_single", "critic_selected_actor_topk", "random_baseline"):
            row = grouped[(seed, policy)]
            print(
                "  {policy}: reward={reward:.6f}, throughput={throughput:.6f}, fair={fair:.6f}".format(
                    policy=policy,
                    reward=float(row["mean_cumulative_reward"]),
                    throughput=float(row["mean_TotalThroughput"]),
                    fair=float(row["mean_FairIdx"]),
                )
            )
        print(f"  score_reward_corr: {correlations[seed]}")


if __name__ == "__main__":
    main()
