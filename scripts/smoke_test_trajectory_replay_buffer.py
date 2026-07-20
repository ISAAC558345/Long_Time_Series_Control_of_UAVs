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
    from buffers.trajectory_replay_buffer import TrajectoryReplayBuffer
    from envs.mubs_cov.mubs_cov_traj import MultiUbsCoverageTrajEnv
    from models.trajectory_generator import LinearWaypointTrajectoryGenerator

    np.random.seed(2)

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
    buffer = TrajectoryReplayBuffer(capacity=10, seed=123)

    for _ in range(num_rollouts):
        env.reset()
        w = np.random.uniform(-200.0, 200.0, size=(env.n_ubs, 2)).astype(np.float32)
        action_seq = generator.generate(current_pos=env.pos_ubs.copy(), w=w)
        trajectory_info = env.step_trajectory(action_seq, action_type="displacement")
        buffer.add_rollout(trajectory_info)

    batch = buffer.sample_segments(batch_size=batch_size, segment_length=segment_length)

    print(f"buffer size: {len(buffer)}")
    print(f"start_state shape: {batch['start_state'].shape}")
    print(f"action_subseq shape: {batch['action_subseq'].shape}")
    print(f"reward_subseq shape: {batch['reward_subseq'].shape}")
    print(f"bootstrap_state shape: {batch['bootstrap_state'].shape}")
    print(f"done_subseq shape: {batch['done_subseq'].shape}")

    assert len(buffer) == num_rollouts
    assert batch["start_state"].shape[0] == batch_size
    assert batch["action_subseq"].shape == (batch_size, segment_length, env.n_ubs, 2)
    assert batch["reward_subseq"].shape == (batch_size, segment_length, env.n_ubs)
    assert batch["bootstrap_state"].shape == batch["start_state"].shape
    assert batch["done_subseq"].shape == (batch_size, segment_length)


if __name__ == "__main__":
    main()
