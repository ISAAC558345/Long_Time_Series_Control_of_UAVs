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


DEFAULT_RUN_DIRS = [
    "outputs/top_erl_critic_warmup_multiseed/seed_0",
    "outputs/top_erl_multiseed/seed_0",
    "outputs/top_erl_multiseed/seed_1",
    "outputs/top_erl_multiseed/seed_2",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose critic N-step targets and Q fit.")
    parser.add_argument("--run_dirs", nargs="+", default=DEFAULT_RUN_DIRS)
    parser.add_argument("--output_dir", default="outputs/critic_target_diagnostics")
    parser.add_argument("--map_id", default="debug")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--segment_length", type=int, default=4)
    parser.add_argument("--num_rollouts", type=int, default=20)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_args(args):
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")
    if args.segment_length <= 0:
        raise ValueError("--segment_length must be positive.")
    if args.segment_length > args.horizon:
        raise ValueError("--segment_length must be <= --horizon.")
    if args.num_rollouts <= 0:
        raise ValueError("--num_rollouts must be positive.")
    if args.num_batches <= 0:
        raise ValueError("--num_batches must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value):
    if value in ("", None):
        return None
    return float(value)


def numeric_values(rows, key):
    values = []
    for row in rows:
        value = safe_float(row.get(key))
        if value is not None:
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return dict(mean="", std="", min="", max="")
    return dict(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
    )


def finite_mapping(row, skip_keys):
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


def train_log_stats(run_dir):
    rows = read_csv(run_dir / "train_log.csv")
    final = rows[-1]
    critic_loss = stats(numeric_values(rows, "critic_loss"))
    actor_loss = stats(numeric_values(rows, "actor_loss"))
    reward = stats(numeric_values(rows, "rollout_cumulative_reward"))

    last_actor_values = numeric_values(rows, "actor_loss")
    last_nonempty_actor_loss = float(last_actor_values[-1]) if last_actor_values.size else ""
    return dict(
        final_rollout_cumulative_reward=safe_float(final["rollout_cumulative_reward"]),
        final_TotalThroughput=safe_float(final["TotalThroughput"]),
        final_FairIdx=safe_float(final["FairIdx"]),
        final_ProbCollision=safe_float(final["ProbCollision"]),
        final_critic_loss=safe_float(final["critic_loss"]),
        final_actor_loss=final.get("actor_loss", ""),
        last_nonempty_actor_loss=last_nonempty_actor_loss,
        critic_loss_mean=critic_loss["mean"],
        critic_loss_std=critic_loss["std"],
        critic_loss_min=critic_loss["min"],
        critic_loss_max=critic_loss["max"],
        actor_loss_mean=actor_loss["mean"],
        actor_loss_std=actor_loss["std"],
        actor_loss_min=actor_loss["min"],
        actor_loss_max=actor_loss["max"],
        reward_mean=reward["mean"],
        reward_std=reward["std"],
        reward_min=reward["min"],
        reward_max=reward["max"],
        target_mode=final.get("target_mode", "bootstrapped_n_step") or "bootstrapped_n_step",
        bootstrap_weight=safe_float(final.get("bootstrap_weight", "1.0")),
        train_log_finite=all(finite_mapping(row, skip_keys={"actor_updated"}) for row in rows),
    )


def run_label(run_dir):
    parts = run_dir.as_posix().split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}__{parts[-1]}"
    return run_dir.name


def seed_from_run_dir(run_dir):
    name = run_dir.name
    if name.startswith("seed_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return None
    return None


def build_models(env, run_dir, horizon):
    import torch

    from models.trajectory_actor import TrajectoryActor
    from models.transformer_critic import TrajectoryTransformerCritic

    checkpoint_dir = run_dir / "checkpoints"
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

    target_critic = TrajectoryTransformerCritic(
        state_dim=state_dim,
        n_uavs=env.n_ubs,
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_segment_length=horizon,
    )
    target_critic.load_state_dict(torch.load(checkpoint_dir / "target_critic.pt", map_location="cpu"))
    target_critic.eval()
    return actor, critic, target_critic, max_move_per_slot


def collect_actor_rollout(env, actor, generator):
    import torch

    env.reset()
    state = env.get_state()
    state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        w, _, _, _ = actor.sample(state_tensor)
    action_seq = generator.generate(current_pos=env.pos_ubs.copy(), w=w[0].cpu().numpy())
    return env.step_trajectory(action_seq, action_type="displacement")


def build_replay_buffer(env, actor, generator, num_rollouts, seed):
    from buffers.trajectory_replay_buffer import TrajectoryReplayBuffer

    buffer = TrajectoryReplayBuffer(capacity=max(num_rollouts, 1), seed=seed)
    for _ in range(num_rollouts):
        buffer.add_rollout(collect_actor_rollout(env, actor, generator))
    return buffer


def tensor_batch(batch):
    import torch

    return dict(
        start_state=torch.as_tensor(batch["start_state"], dtype=torch.float32),
        action_subseq=torch.as_tensor(batch["action_subseq"], dtype=torch.float32),
        reward_subseq=torch.as_tensor(batch["reward_subseq"], dtype=torch.float32),
        bootstrap_state=torch.as_tensor(batch["bootstrap_state"], dtype=torch.float32),
        done_subseq=torch.as_tensor(batch["done_subseq"], dtype=torch.bool),
        state_subseq=torch.as_tensor(batch["state_subseq"], dtype=torch.float32),
    )


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def array_stats(prefix, values):
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def batch_diagnostics(run_name, batch_id, critic, target_critic, batch, gamma, target_mode, bootstrap_weight):
    import torch
    import torch.nn.functional as F

    from models.critic_loss import compute_n_step_targets

    tensors = tensor_batch(batch)
    with torch.no_grad():
        _, prefix_q_values = critic(tensors["start_state"], tensors["action_subseq"])
        bootstrap_values = target_critic.value(tensors["state_subseq"][:, 1:, :])
        n_step_targets, scalar_reward_seq = compute_n_step_targets(
            reward_subseq=tensors["reward_subseq"],
            done_subseq=tensors["done_subseq"],
            bootstrap_values=bootstrap_values,
            gamma=gamma,
            target_mode=target_mode,
            bootstrap_weight=bootstrap_weight,
        )
        mse = F.mse_loss(prefix_q_values, n_step_targets).item()

    segment_length = scalar_reward_seq.shape[1]
    discounts = gamma ** torch.arange(segment_length, dtype=scalar_reward_seq.dtype)
    discounted_rewards = scalar_reward_seq * discounts.unsqueeze(0)
    discounted_prefix_returns = torch.cumsum(discounted_rewards, dim=1)
    done_prefix = torch.cumsum(tensors["done_subseq"].to(scalar_reward_seq.dtype), dim=1).clamp(max=1.0)
    bootstrap_discounts = gamma ** torch.arange(1, segment_length + 1, dtype=scalar_reward_seq.dtype)
    if target_mode == "bootstrapped_n_step":
        effective_bootstrap_weight = 1.0
    elif target_mode == "no_bootstrap":
        effective_bootstrap_weight = 0.0
    else:
        effective_bootstrap_weight = float(bootstrap_weight)
    bootstrap_term = effective_bootstrap_weight * (1.0 - done_prefix) * bootstrap_discounts.unsqueeze(0) * bootstrap_values

    scalar_reward_np = scalar_reward_seq.cpu().numpy()
    q_np = prefix_q_values.cpu().numpy()
    target_np = n_step_targets.cpu().numpy()
    bootstrap_np = bootstrap_values.cpu().numpy()
    discounted_reward_np = discounted_prefix_returns.cpu().numpy()
    bootstrap_term_np = bootstrap_term.cpu().numpy()

    mean_abs_bootstrap_term = float(np.mean(np.abs(bootstrap_term_np)))
    mean_abs_target = float(np.mean(np.abs(target_np)))
    mean_abs_discounted_reward = float(np.mean(np.abs(discounted_reward_np)))
    bootstrap_target_abs_ratio = mean_abs_bootstrap_term / (mean_abs_target + 1e-8)
    bootstrap_reward_abs_ratio = mean_abs_bootstrap_term / (mean_abs_discounted_reward + 1e-8)

    numeric = []
    for array in (scalar_reward_np, q_np, target_np, bootstrap_np, bootstrap_term_np):
        numeric.extend(array.reshape(-1).tolist())

    row = dict(
        run_name=run_name,
        batch_id=batch_id,
        target_mode=target_mode,
        bootstrap_weight=bootstrap_weight,
        q_target_mse=float(mse),
        q_target_corr=pearson_corr(q_np, target_np),
        bootstrap_target_abs_ratio=bootstrap_target_abs_ratio,
        bootstrap_reward_abs_ratio=bootstrap_reward_abs_ratio,
        n_step_targets_nearly_constant=float(np.std(target_np)) < 1e-6,
        prefix_q_values_nearly_constant=float(np.std(q_np)) < 1e-6,
        batch_finite=all(math.isfinite(float(value)) for value in numeric),
    )
    row.update(array_stats("scalar_reward_seq", scalar_reward_np))
    row.update(array_stats("prefix_q_values", q_np))
    row.update(array_stats("n_step_targets", target_np))
    row.update(array_stats("bootstrap_values", bootstrap_np))
    row.update(array_stats("discounted_reward_prefix", discounted_reward_np))
    row.update(array_stats("bootstrap_term", bootstrap_term_np))
    return row


def candidate_corr_for_run(run_dir):
    seed = seed_from_run_dir(run_dir)
    if seed is None:
        return ""

    base_dir = run_dir.parent
    candidates = [
        base_dir / "critic_calibration" / "critic_calibration_summary.csv",
        base_dir / "candidate_selection" / "candidate_selection_summary.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        rows = read_csv(path)
        for row in rows:
            if int(row["seed"]) != seed:
                continue
            if "q_reward_corr_mean" in row:
                return safe_float(row["q_reward_corr_mean"])
            if "candidate_score_reward_corr" in row:
                return safe_float(row["candidate_score_reward_corr"])
    return ""


def summarize_run(run_name, train_stats, batch_rows, candidate_q_reward_corr):
    q_target_mse = numeric_values(batch_rows, "q_target_mse")
    q_target_corr = numeric_values(batch_rows, "q_target_corr")
    bootstrap_target_ratio = numeric_values(batch_rows, "bootstrap_target_abs_ratio")
    bootstrap_reward_ratio = numeric_values(batch_rows, "bootstrap_reward_abs_ratio")

    summary = dict(
        run_name=run_name,
        target_mode=train_stats["target_mode"],
        bootstrap_weight=train_stats["bootstrap_weight"],
        final_rollout_cumulative_reward=train_stats["final_rollout_cumulative_reward"],
        final_TotalThroughput=train_stats["final_TotalThroughput"],
        final_FairIdx=train_stats["final_FairIdx"],
        final_ProbCollision=train_stats["final_ProbCollision"],
        final_critic_loss=train_stats["final_critic_loss"],
        final_actor_loss=train_stats["final_actor_loss"],
        last_nonempty_actor_loss=train_stats["last_nonempty_actor_loss"],
        critic_loss_mean=train_stats["critic_loss_mean"],
        critic_loss_std=train_stats["critic_loss_std"],
        critic_loss_min=train_stats["critic_loss_min"],
        critic_loss_max=train_stats["critic_loss_max"],
        actor_loss_mean=train_stats["actor_loss_mean"],
        actor_loss_std=train_stats["actor_loss_std"],
        actor_loss_min=train_stats["actor_loss_min"],
        actor_loss_max=train_stats["actor_loss_max"],
        reward_mean=train_stats["reward_mean"],
        reward_std=train_stats["reward_std"],
        reward_min=train_stats["reward_min"],
        reward_max=train_stats["reward_max"],
        q_target_mse_mean=float(np.mean(q_target_mse)),
        q_target_mse_std=float(np.std(q_target_mse)),
        q_target_corr_mean=float(np.mean(q_target_corr)),
        q_target_corr_std=float(np.std(q_target_corr)),
        bootstrap_target_abs_ratio_mean=float(np.mean(bootstrap_target_ratio)),
        bootstrap_reward_abs_ratio_mean=float(np.mean(bootstrap_reward_ratio)),
        n_step_targets_nearly_constant_rate=float(np.mean([row["n_step_targets_nearly_constant"] == "True" or row["n_step_targets_nearly_constant"] is True for row in batch_rows])),
        prefix_q_values_nearly_constant_rate=float(np.mean([row["prefix_q_values_nearly_constant"] == "True" or row["prefix_q_values_nearly_constant"] is True for row in batch_rows])),
        candidate_q_reward_corr=candidate_q_reward_corr,
        all_finite=train_stats["train_log_finite"] and all(row["batch_finite"] for row in batch_rows),
    )

    for key in (
        "scalar_reward_seq",
        "prefix_q_values",
        "n_step_targets",
        "bootstrap_values",
        "discounted_reward_prefix",
        "bootstrap_term",
    ):
        for suffix in ("mean", "std", "min", "max"):
            values = numeric_values(batch_rows, f"{key}_{suffix}")
            summary[f"{key}_{suffix}_mean"] = float(np.mean(values))
    return summary


def analyze_run(run_dir, args):
    from envs.mubs_cov.mubs_cov_traj import MultiUbsCoverageTrajEnv
    from models.trajectory_generator import LinearWaypointTrajectoryGenerator

    run_dir = Path(run_dir)
    run_name = run_label(run_dir)
    env = MultiUbsCoverageTrajEnv(map_id=args.map_id, record=False)
    env.reset()
    actor, critic, target_critic, max_move_per_slot = build_models(env, run_dir, args.horizon)
    generator = LinearWaypointTrajectoryGenerator(
        H=args.horizon,
        n_uavs=env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=env.range_pos,
    )
    buffer = build_replay_buffer(env, actor, generator, args.num_rollouts, args.seed)
    train_stats = train_log_stats(run_dir)
    target_mode = train_stats["target_mode"]
    bootstrap_weight = train_stats["bootstrap_weight"]

    batch_rows = []
    for batch_id in range(args.num_batches):
        batch = buffer.sample_segments(args.batch_size, args.segment_length)
        batch_rows.append(
            batch_diagnostics(
                run_name,
                batch_id,
                critic,
                target_critic,
                batch,
                args.gamma,
                target_mode,
                bootstrap_weight,
            )
        )

    summary = summarize_run(run_name, train_stats, batch_rows, candidate_corr_for_run(run_dir))
    return batch_rows, summary


def report_judgement(row):
    q_corr = safe_float(row["q_target_corr_mean"])
    candidate_corr = safe_float(row["candidate_q_reward_corr"])
    target_std = safe_float(row["n_step_targets_std_mean"])
    q_std = safe_float(row["prefix_q_values_std_mean"])
    bootstrap_reward_ratio = safe_float(row["bootstrap_reward_abs_ratio_mean"])
    bootstrap_target_ratio = safe_float(row["bootstrap_target_abs_ratio_mean"])

    notes = []
    if q_corr is not None and q_corr < 0.2:
        notes.append("Q-target correlation is weak; in-buffer critic fit is suspect.")
    else:
        notes.append("Q-target correlation is not obviously weak.")
    if target_std is not None and target_std < 1e-6:
        notes.append("N-step targets are nearly constant.")
    if q_std is not None and q_std < 1e-6:
        notes.append("Prefix Q values are nearly constant.")
    if bootstrap_reward_ratio is not None and bootstrap_reward_ratio > 5.0:
        notes.append("Bootstrap term is much larger than discounted reward prefix.")
    if bootstrap_target_ratio is not None and bootstrap_target_ratio > 0.7:
        notes.append("Bootstrap term dominates the target magnitude.")
    if candidate_corr is not None and q_corr is not None and q_corr >= 0.2 and candidate_corr < 0.1:
        notes.append("In-buffer fit is acceptable but candidate ranking looks OOD.")
    elif candidate_corr is not None and candidate_corr < 0.1:
        notes.append("Candidate Q-reward correlation is weak or negative.")
    return notes


def build_report(summary_rows):
    lines = [
        "# Critic target / loss diagnostics",
        "",
        "This report reloads saved checkpoints and analyzes critic targets on fresh short replay buffers. It does not retrain or modify checkpoints.",
        "",
        "## Summary",
        "",
        "| run | final reward | final critic loss | final actor loss | Q-target corr | Q-target MSE | reward mean | target mean | Q mean | bootstrap/target | bootstrap/reward | candidate Q-reward corr | finite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {run} | {reward} | {critic_loss} | {actor_loss} | {corr:.6f} | {mse:.6f} | {reward_mean:.6f} | {target_mean:.6f} | {q_mean:.6f} | {bt:.6f} | {br:.6f} | {candidate_corr} | {finite} |".format(
                run=row["run_name"],
                reward=row["final_rollout_cumulative_reward"],
                critic_loss=row["final_critic_loss"],
                actor_loss=row["final_actor_loss"] if row["final_actor_loss"] != "" else "NA",
                corr=row["q_target_corr_mean"],
                mse=row["q_target_mse_mean"],
                reward_mean=row["scalar_reward_seq_mean_mean"],
                target_mean=row["n_step_targets_mean_mean"],
                q_mean=row["prefix_q_values_mean_mean"],
                bt=row["bootstrap_target_abs_ratio_mean"],
                br=row["bootstrap_reward_abs_ratio_mean"],
                candidate_corr=row["candidate_q_reward_corr"],
                finite=row["all_finite"],
            )
        )

    lines.extend(["", "## Per-run interpretation", ""])
    for row in summary_rows:
        lines.append(f"### {row['run_name']}")
        lines.append(f"- scalar_reward_seq mean/std: {row['scalar_reward_seq_mean_mean']} / {row['scalar_reward_seq_std_mean']}")
        lines.append(f"- n_step_targets mean/std: {row['n_step_targets_mean_mean']} / {row['n_step_targets_std_mean']}")
        lines.append(f"- prefix_q_values mean/std: {row['prefix_q_values_mean_mean']} / {row['prefix_q_values_std_mean']}")
        lines.append(f"- bootstrap_values mean/std: {row['bootstrap_values_mean_mean']} / {row['bootstrap_values_std_mean']}")
        lines.append(f"- Q-target MSE mean: {row['q_target_mse_mean']}")
        lines.append(f"- Q-target correlation mean: {row['q_target_corr_mean']}")
        lines.append(f"- bootstrap/target abs ratio mean: {row['bootstrap_target_abs_ratio_mean']}")
        lines.append(f"- bootstrap/reward abs ratio mean: {row['bootstrap_reward_abs_ratio_mean']}")
        for note in report_judgement(row):
            lines.append(f"- {note}")
        lines.append("")

    lines.extend([
        "## Diagnostic answers",
        "",
        "- If Q-target correlation is weak, the critic is not reliably fitting its own N-step targets on fresh replay-buffer data.",
        "- If bootstrap/reward ratio is large, the bootstrap term dominates target scale and may hurt calibration.",
        "- If Q-target fit is weak and candidate Q-reward correlation is also weak, the issue is more likely target/loss/calibration than pure OOD candidate ranking.",
        "- First next checks: reward / target normalization, critic-only supervised warmup, more critic updates, and Monte Carlo target smoke test without bootstrap.",
        "- Delayed actor update remains useful, but this diagnostic focuses on whether critic targets themselves are learnable and calibrated.",
        "- This is diagnostic only; it is not a final algorithm conclusion.",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_batch_rows = []
    summary_rows = []
    for run_dir in args.run_dirs:
        batch_rows, summary = analyze_run(Path(run_dir), args)
        all_batch_rows.extend(batch_rows)
        summary_rows.append(summary)

    batch_path = output_dir / "critic_target_batch_stats.csv"
    summary_path = output_dir / "critic_target_summary.csv"
    report_path = output_dir / "critic_target_analysis.md"
    write_csv(batch_path, all_batch_rows)
    write_csv(summary_path, summary_rows)
    report_path.write_text(build_report(summary_rows), encoding="utf-8")

    print(f"critic_target_batch_stats_csv: {batch_path}")
    print(f"critic_target_summary_csv: {summary_path}")
    print(f"critic_target_report: {report_path}")
    for row in summary_rows:
        print(
            "{run}: q_target_corr={corr:.6f}, q_target_mse={mse:.6f}, "
            "reward_mean={reward:.6f}, target_mean={target:.6f}, q_mean={q:.6f}, "
            "bootstrap_target_ratio={bt:.6f}, bootstrap_reward_ratio={br:.6f}, finite={finite}".format(
                run=row["run_name"],
                corr=row["q_target_corr_mean"],
                mse=row["q_target_mse_mean"],
                reward=row["scalar_reward_seq_mean_mean"],
                target=row["n_step_targets_mean_mean"],
                q=row["prefix_q_values_mean_mean"],
                bt=row["bootstrap_target_abs_ratio_mean"],
                br=row["bootstrap_reward_abs_ratio_mean"],
                finite=row["all_finite"],
            )
        )


if __name__ == "__main__":
    main()
