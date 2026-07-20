from copy import deepcopy
from pathlib import Path
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


def main():
    import torch

    from buffers.trajectory_replay_buffer import TrajectoryReplayBuffer
    from envs.mubs_cov.mubs_cov_traj import MultiUbsCoverageTrajEnv
    from models.critic_loss import compute_critic_loss
    from models.trajectory_generator import LinearWaypointTrajectoryGenerator
    from models.transformer_critic import TrajectoryTransformerCritic

    np.random.seed(11)
    torch.manual_seed(11)

    env = MultiUbsCoverageTrajEnv(map_id="debug", record=False)
    horizon = 6
    segment_length = 4
    batch_size = 3
    num_rollouts = 5
    max_move_per_slot = float(np.max(np.asarray(env.vels, dtype=np.float32)) * env.dt)
    generator = LinearWaypointTrajectoryGenerator(
        H=horizon,
        n_uavs=env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=env.range_pos,
    )
    buffer = TrajectoryReplayBuffer(capacity=10, seed=110)

    for _ in range(num_rollouts):
        env.reset()
        w = np.random.uniform(-200.0, 200.0, size=(env.n_ubs, 2)).astype(np.float32)
        action_seq = generator.generate(current_pos=env.pos_ubs.copy(), w=w)
        trajectory_info = env.step_trajectory(action_seq, action_type="displacement")
        buffer.add_rollout(trajectory_info)

    batch = buffer.sample_segments(batch_size=batch_size, segment_length=segment_length)
    start_state = torch.as_tensor(batch["start_state"], dtype=torch.float32)
    state_subseq = torch.as_tensor(batch["state_subseq"], dtype=torch.float32)
    action_subseq = torch.as_tensor(batch["action_subseq"], dtype=torch.float32)
    reward_subseq = torch.as_tensor(batch["reward_subseq"], dtype=torch.float32)
    bootstrap_state = torch.as_tensor(batch["bootstrap_state"], dtype=torch.float32)
    done_subseq = torch.as_tensor(batch["done_subseq"], dtype=torch.bool)

    critic = TrajectoryTransformerCritic(
        state_dim=start_state.shape[-1],
        n_uavs=action_subseq.shape[2],
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_segment_length=segment_length,
    )
    target_critic = deepcopy(critic)
    target_critic.eval()

    target_configs = [
        ("bootstrapped_n_step", 1.0),
        ("no_bootstrap", 0.0),
        ("weighted_bootstrap", 0.25),
    ]

    for target_mode, bootstrap_weight in target_configs:
        critic_loss, info = compute_critic_loss(
            critic=critic,
            target_critic=target_critic,
            start_state=start_state,
            action_subseq=action_subseq,
            reward_subseq=reward_subseq,
            bootstrap_state=bootstrap_state,
            done_subseq=done_subseq,
            state_subseq=state_subseq,
            gamma=0.99,
            target_mode=target_mode,
            bootstrap_weight=bootstrap_weight,
        )
        n_step_targets = info["n_step_targets"]
        print(f"target_mode: {target_mode}")
        print(f"bootstrap_weight: {bootstrap_weight}")
        print(f"scalar_reward_seq shape: {tuple(info['scalar_reward_seq'].shape)}")
        print(f"prefix_q_values shape: {tuple(info['prefix_q_values'].shape)}")
        print(f"n_step_targets shape: {tuple(n_step_targets.shape)}")
        print(
            "n_step_targets mean/std/min/max: "
            f"{n_step_targets.mean().item()} / "
            f"{n_step_targets.std(unbiased=False).item()} / "
            f"{n_step_targets.min().item()} / "
            f"{n_step_targets.max().item()}"
        )
        print(f"critic_loss: {critic_loss.item()}")
        print(f"critic_loss finite: {torch.isfinite(critic_loss).item()}")
        print("")

        assert info["scalar_reward_seq"].shape == (batch_size, segment_length)
        assert info["prefix_q_values"].shape == (batch_size, segment_length)
        assert n_step_targets.shape == (batch_size, segment_length)
        assert torch.isfinite(critic_loss)
        assert torch.isfinite(n_step_targets).all()


if __name__ == "__main__":
    main()
