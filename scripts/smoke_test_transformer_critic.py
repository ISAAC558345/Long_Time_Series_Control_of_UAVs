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
    from models.trajectory_generator import LinearWaypointTrajectoryGenerator
    from models.transformer_critic import TrajectoryTransformerCritic

    np.random.seed(3)
    torch.manual_seed(3)

    env = MultiUbsCoverageTrajEnv(map_id="debug", record=False)
    H = 6
    segment_length = 4
    batch_size = 3
    num_rollouts = 5
    max_move_per_slot = float(np.max(np.asarray(env.vels, dtype=np.float32)) * env.dt)
    generator = LinearWaypointTrajectoryGenerator(
        H=H,
        n_uavs=env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=env.range_pos,
    )
    buffer = TrajectoryReplayBuffer(capacity=10, seed=456)

    for _ in range(num_rollouts):
        env.reset()
        w = np.random.uniform(-200.0, 200.0, size=(env.n_ubs, 2)).astype(np.float32)
        action_seq = generator.generate(current_pos=env.pos_ubs.copy(), w=w)
        trajectory_info = env.step_trajectory(action_seq, action_type="displacement")
        buffer.add_rollout(trajectory_info)

    batch = buffer.sample_segments(batch_size=batch_size, segment_length=segment_length)
    start_state = torch.as_tensor(batch["start_state"], dtype=torch.float32)
    action_subseq = torch.as_tensor(batch["action_subseq"], dtype=torch.float32)

    critic = TrajectoryTransformerCritic(
        state_dim=start_state.shape[-1],
        n_uavs=action_subseq.shape[2],
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_segment_length=segment_length,
    )
    state_value, prefix_q_values = critic(start_state, action_subseq)

    print(f"start_state shape: {tuple(start_state.shape)}")
    print(f"action_subseq shape: {tuple(action_subseq.shape)}")
    print(f"state_value shape: {tuple(state_value.shape)}")
    print(f"prefix_q_values shape: {tuple(prefix_q_values.shape)}")
    print(f"prefix_q_values finite: {torch.isfinite(prefix_q_values).all().item()}")

    assert state_value.shape == (batch_size,)
    assert prefix_q_values.shape == (batch_size, segment_length)
    assert torch.isfinite(prefix_q_values).all()


if __name__ == "__main__":
    main()
