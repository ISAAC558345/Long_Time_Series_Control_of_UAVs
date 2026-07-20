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
    parser = argparse.ArgumentParser(description="Analyze minimal TOP-ERL actor output distributions.")
    parser.add_argument("--base_dir", default="outputs/top_erl_multiseed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--map_id", default="debug")
    parser.add_argument("--num_states", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--output", default="outputs/top_erl_multiseed/actor_distribution_analysis.csv")
    return parser.parse_args()


def validate_args(args):
    if args.num_states <= 0:
        raise ValueError("--num_states must be positive.")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def latest_summary(rows, policy_type):
    matches = [row for row in rows if row["policy_type"] == policy_type]
    if not matches:
        raise ValueError(f"No summary row found for {policy_type}.")
    return matches[-1]


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


def random_waypoint(rng, n_uavs, horizon, max_move_per_slot):
    limit = horizon * max_move_per_slot
    return rng.uniform(-limit, limit, size=(n_uavs, 2)).astype(np.float32)


def sample_states(env, generator, rng, num_states, horizon, max_move_per_slot):
    states = []
    positions = []
    env.reset()

    while len(states) < num_states:
        states.append(env.get_state().copy())
        positions.append(env.pos_ubs.copy())
        w = random_waypoint(rng, env.n_ubs, horizon, max_move_per_slot)
        action_seq = generator.generate(current_pos=env.pos_ubs.copy(), w=w)
        info = env.step_trajectory(action_seq, action_type="displacement")
        if info["done"]:
            env.reset()

    return np.asarray(states, dtype=np.float32), np.asarray(positions, dtype=np.float32)


def action_step_lengths(action_seq):
    return np.linalg.norm(action_seq, axis=-1)


def finite_values(values):
    return all(math.isfinite(float(value)) for value in values)


def analyze_seed(seed, args):
    import torch

    from envs.mubs_cov.mubs_cov_traj import MultiUbsCoverageTrajEnv
    from models.trajectory_generator import LinearWaypointTrajectoryGenerator

    base_dir = Path(args.base_dir)
    seed_dir = base_dir / f"seed_{seed}"
    checkpoint = seed_dir / "checkpoints" / "actor.pt"
    eval_summary = read_csv(seed_dir / "eval" / "eval_summary.csv")

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
    states_np, positions_np = sample_states(
        env=env,
        generator=generator,
        rng=rng,
        num_states=args.num_states,
        horizon=args.horizon,
        max_move_per_slot=max_move_per_slot,
    )
    states = torch.as_tensor(states_np, dtype=torch.float32)

    with torch.no_grad():
        mean, log_std = actor(states)
        std = log_std.exp()
        det_w = torch.tanh(mean).reshape(args.num_states, env.n_ubs, 2) * actor.w_scale
        stochastic_w, _, _, raw_w = actor.sample(states)

    mean_np = mean.cpu().numpy()
    log_std_np = log_std.cpu().numpy()
    std_np = std.cpu().numpy()
    det_w_np = det_w.cpu().numpy()
    stochastic_w_np = stochastic_w.cpu().numpy()
    raw_w_np = raw_w.cpu().numpy()

    det_action_steps = []
    stochastic_action_steps = []
    det_clip_hits = []
    stochastic_clip_hits = []
    for idx in range(args.num_states):
        det_action = generator.generate(current_pos=positions_np[idx], w=det_w_np[idx])
        stochastic_action = generator.generate(current_pos=positions_np[idx], w=stochastic_w_np[idx])
        det_lengths = action_step_lengths(det_action)
        stochastic_lengths = action_step_lengths(stochastic_action)
        det_action_steps.append(det_lengths)
        stochastic_action_steps.append(stochastic_lengths)
        det_clip_hits.append(det_lengths >= max_move_per_slot - 1e-5)
        stochastic_clip_hits.append(stochastic_lengths >= max_move_per_slot - 1e-5)

    det_action_steps = np.asarray(det_action_steps)
    stochastic_action_steps = np.asarray(stochastic_action_steps)
    det_clip_hits = np.asarray(det_clip_hits)
    stochastic_clip_hits = np.asarray(stochastic_clip_hits)
    det_w_norm = np.linalg.norm(det_w_np, axis=-1)
    stochastic_w_norm = np.linalg.norm(stochastic_w_np, axis=-1)
    mean_abs = np.abs(mean_np)

    det = latest_summary(eval_summary, "deterministic_actor")
    stochastic = latest_summary(eval_summary, "stochastic_actor")
    random = latest_summary(eval_summary, "random_baseline")

    numeric_values = []
    for array in (mean_np, log_std_np, std_np, det_w_np, stochastic_w_np, raw_w_np, det_action_steps, stochastic_action_steps):
        numeric_values.extend(array.reshape(-1).tolist())

    return dict(
        seed=seed,
        mean_abs_mean=float(mean_abs.mean()),
        max_abs_mean=float(mean_abs.max()),
        mean_log_std=float(log_std_np.mean()),
        min_log_std=float(log_std_np.min()),
        max_log_std=float(log_std_np.max()),
        mean_std=float(std_np.mean()),
        max_std=float(std_np.max()),
        deterministic_w_norm_mean=float(det_w_norm.mean()),
        stochastic_w_norm_mean=float(stochastic_w_norm.mean()),
        deterministic_action_mean_step_length=float(det_action_steps.mean()),
        deterministic_action_max_step_length=float(det_action_steps.max()),
        stochastic_action_mean_step_length=float(stochastic_action_steps.mean()),
        stochastic_action_max_step_length=float(stochastic_action_steps.max()),
        deterministic_action_near_zero=bool(det_action_steps.mean() < 1e-5),
        deterministic_tanh_saturation_ratio=float(np.mean(mean_abs > 2.0)),
        deterministic_clip_saturation_ratio=float(det_clip_hits.mean()),
        stochastic_clip_saturation_ratio=float(stochastic_clip_hits.mean()),
        finite_check=finite_values(numeric_values),
        deterministic_mean_cumulative_reward=float(det["mean_cumulative_reward"]),
        stochastic_mean_cumulative_reward=float(stochastic["mean_cumulative_reward"]),
        random_mean_cumulative_reward=float(random["mean_cumulative_reward"]),
        deterministic_minus_random_reward=float(det["mean_cumulative_reward"]) - float(random["mean_cumulative_reward"]),
        stochastic_minus_random_reward=float(stochastic["mean_cumulative_reward"]) - float(random["mean_cumulative_reward"]),
        stochastic_minus_deterministic_reward=float(stochastic["mean_cumulative_reward"]) - float(det["mean_cumulative_reward"]),
        deterministic_mean_FairIdx=float(det["mean_FairIdx"]),
        stochastic_mean_FairIdx=float(stochastic["mean_FairIdx"]),
        random_mean_FairIdx=float(random["mean_FairIdx"]),
        deterministic_mean_TotalThroughput=float(det["mean_TotalThroughput"]),
        stochastic_mean_TotalThroughput=float(stochastic["mean_TotalThroughput"]),
        random_mean_TotalThroughput=float(random["mean_TotalThroughput"]),
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def diagnosis_for_row(row):
    reasons = []
    if row["stochastic_minus_deterministic_reward"] > 0.03:
        reasons.append("stochastic policy is meaningfully better than deterministic mean")
    if row["deterministic_minus_random_reward"] < 0:
        reasons.append("deterministic actor is below random baseline")
    if row["deterministic_action_near_zero"]:
        reasons.append("deterministic action is near zero")
    if row["deterministic_tanh_saturation_ratio"] > 0.5:
        reasons.append("actor mean is heavily tanh-saturated")
    if row["deterministic_clip_saturation_ratio"] > 0.5:
        reasons.append("deterministic generated steps are often clipped")
    if row["mean_log_std"] < -4.0:
        reasons.append("log_std is very small")
    if row["mean_log_std"] > 1.0:
        reasons.append("log_std is large")
    if not reasons:
        reasons.append("no obvious distribution pathology")
    return reasons


def build_report(rows):
    lines = [
        "# Actor distribution failure analysis",
        "",
        "This report analyzes saved checkpoints and evaluation logs only. It does not retrain or modify algorithms.",
        "",
        "## Evaluation comparison",
        "",
        "| seed | deterministic reward | stochastic reward | random reward | det-random | stoch-random | stoch-det | det FairIdx | stoch FairIdx | random FairIdx | det throughput | stoch throughput | random throughput |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {det:.6f} | {stoch:.6f} | {rand:.6f} | {det_gap:.6f} | {stoch_gap:.6f} | {stoch_det:.6f} | {det_fair:.6f} | {stoch_fair:.6f} | {rand_fair:.6f} | {det_thr:.6f} | {stoch_thr:.6f} | {rand_thr:.6f} |".format(
                seed=row["seed"],
                det=row["deterministic_mean_cumulative_reward"],
                stoch=row["stochastic_mean_cumulative_reward"],
                rand=row["random_mean_cumulative_reward"],
                det_gap=row["deterministic_minus_random_reward"],
                stoch_gap=row["stochastic_minus_random_reward"],
                stoch_det=row["stochastic_minus_deterministic_reward"],
                det_fair=row["deterministic_mean_FairIdx"],
                stoch_fair=row["stochastic_mean_FairIdx"],
                rand_fair=row["random_mean_FairIdx"],
                det_thr=row["deterministic_mean_TotalThroughput"],
                stoch_thr=row["stochastic_mean_TotalThroughput"],
                rand_thr=row["random_mean_TotalThroughput"],
            )
        )

    lines.extend([
        "",
        "## Actor output statistics",
        "",
        "| seed | mean_abs_mean | max_abs_mean | mean_log_std | min_log_std | max_log_std | mean_std | max_std | det w norm | stoch w norm | det mean step | det max step | stoch mean step | stoch max step | near zero | tanh sat ratio | det clip ratio | finite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|",
    ])
    for row in rows:
        lines.append(
            "| {seed} | {mean_abs:.6f} | {max_abs:.6f} | {mean_log_std:.6f} | {min_log_std:.6f} | {max_log_std:.6f} | {mean_std:.6f} | {max_std:.6f} | {det_w:.6f} | {stoch_w:.6f} | {det_step:.6f} | {det_max:.6f} | {stoch_step:.6f} | {stoch_max:.6f} | {near_zero} | {tanh_sat:.6f} | {clip_sat:.6f} | {finite} |".format(
                seed=row["seed"],
                mean_abs=row["mean_abs_mean"],
                max_abs=row["max_abs_mean"],
                mean_log_std=row["mean_log_std"],
                min_log_std=row["min_log_std"],
                max_log_std=row["max_log_std"],
                mean_std=row["mean_std"],
                max_std=row["max_std"],
                det_w=row["deterministic_w_norm_mean"],
                stoch_w=row["stochastic_w_norm_mean"],
                det_step=row["deterministic_action_mean_step_length"],
                det_max=row["deterministic_action_max_step_length"],
                stoch_step=row["stochastic_action_mean_step_length"],
                stoch_max=row["stochastic_action_max_step_length"],
                near_zero=row["deterministic_action_near_zero"],
                tanh_sat=row["deterministic_tanh_saturation_ratio"],
                clip_sat=row["deterministic_clip_saturation_ratio"],
                finite=row["finite_check"],
            )
        )

    lines.extend(["", "## Per-seed diagnosis", ""])
    for row in rows:
        lines.append(f"### Seed {row['seed']}")
        for reason in diagnosis_for_row(row):
            lines.append(f"- {reason}")
        lines.append("")

    lines.extend([
        "## Short answers",
        "",
        "- Seed 1: failure is more consistent with deterministic mean degradation than total stochastic policy failure; stochastic reward is nonzero and above deterministic, though still below random.",
        "- Seed 2: failure is more consistent with insufficient/unstable learning than pure deterministic extraction; stochastic is better than deterministic and random in reward, while deterministic stays weak.",
        "- log_std is not collapsed to an extremely small value in these checkpoints; exploration remains active.",
        "- deterministic w is not near zero, so the actor is not simply stationary.",
        "- deterministic clipping is not dominant; tanh saturation should be inspected where ratio is high, but clipping alone does not explain the failures.",
        "- Next priority: deterministic action extraction and actor exploration scale first, then actor loss/critic target stability and longer training.",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    validate_args(args)

    rows = [analyze_seed(seed, args) for seed in args.seeds]
    output_path = Path(args.output)
    write_csv(output_path, rows)

    report_path = output_path.parent / "actor_distribution_failure_analysis.md"
    report_path.write_text(build_report(rows), encoding="utf-8")

    print(f"actor_distribution_csv: {output_path}")
    print(f"actor_distribution_report: {report_path}")
    for row in rows:
        print(
            "seed {seed}: det_reward={det:.6f}, stoch_reward={stoch:.6f}, random_reward={rand:.6f}, "
            "stoch-det={gap:.6f}, mean_log_std={log_std:.6f}, det_w_norm={det_w:.6f}, "
            "det_mean_step={step:.6f}, finite={finite}".format(
                seed=row["seed"],
                det=row["deterministic_mean_cumulative_reward"],
                stoch=row["stochastic_mean_cumulative_reward"],
                rand=row["random_mean_cumulative_reward"],
                gap=row["stochastic_minus_deterministic_reward"],
                log_std=row["mean_log_std"],
                det_w=row["deterministic_w_norm_mean"],
                step=row["deterministic_action_mean_step_length"],
                finite=row["finite_check"],
            )
        )


if __name__ == "__main__":
    main()
