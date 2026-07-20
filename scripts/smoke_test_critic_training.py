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

    np.random.seed(5)
    torch.manual_seed(5)

    env = MultiUbsCoverageTrajEnv(map_id="debug", record=False)
    H = 6
    segment_length = 4
    batch_size = 4
    num_rollouts = 8
    num_iters = 10
    max_move_per_slot = float(np.max(np.asarray(env.vels, dtype=np.float32)) * env.dt)
    generator = LinearWaypointTrajectoryGenerator(
        H=H,
        n_uavs=env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=env.range_pos,
    )
    buffer = TrajectoryReplayBuffer(capacity=20, seed=321)

    for _ in range(num_rollouts):
        env.reset()
        w = np.random.uniform(-200.0, 200.0, size=(env.n_ubs, 2)).astype(np.float32)
        action_seq = generator.generate(current_pos=env.pos_ubs.copy(), w=w)
        trajectory_info = env.step_trajectory(action_seq, action_type="displacement")
        buffer.add_rollout(trajectory_info)

    probe_batch = buffer.sample_segments(batch_size=batch_size, segment_length=segment_length)
    critic = TrajectoryTransformerCritic(
        state_dim=probe_batch["start_state"].shape[-1],
        n_uavs=probe_batch["action_subseq"].shape[2],
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_segment_length=segment_length,
    )
    target_critic = deepcopy(critic)
    target_critic.eval()
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    losses = []
    optimizer_step_completed = False
    last_batch = None
    for itr in range(num_iters):
        batch = buffer.sample_segments(batch_size=batch_size, segment_length=segment_length)
        last_batch = batch
        start_state = torch.as_tensor(batch["start_state"], dtype=torch.float32)
        state_subseq = torch.as_tensor(batch["state_subseq"], dtype=torch.float32)
        action_subseq = torch.as_tensor(batch["action_subseq"], dtype=torch.float32)
        reward_subseq = torch.as_tensor(batch["reward_subseq"], dtype=torch.float32)
        bootstrap_state = torch.as_tensor(batch["bootstrap_state"], dtype=torch.float32)
        done_subseq = torch.as_tensor(batch["done_subseq"], dtype=torch.bool)

        critic_loss, _ = compute_critic_loss(
            critic=critic,
            target_critic=target_critic,
            start_state=start_state,
            action_subseq=action_subseq,
            reward_subseq=reward_subseq,
            bootstrap_state=bootstrap_state,
            done_subseq=done_subseq,
            state_subseq=state_subseq,
            gamma=0.99,
        )
        assert torch.isfinite(critic_loss)

        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        optimizer_step_completed = True

        target_critic.load_state_dict(critic.state_dict())
        target_critic.eval()

        loss_value = float(critic_loss.item())
        losses.append(loss_value)
        print(f"iter {itr} critic loss: {loss_value}")

    print(f"buffer size: {len(buffer)}")
    print(f"start_state shape: {last_batch['start_state'].shape}")
    print(f"action_subseq shape: {last_batch['action_subseq'].shape}")
    print(f"reward_subseq shape: {last_batch['reward_subseq'].shape}")
    print(f"state_subseq shape: {last_batch['state_subseq'].shape}")
    print(f"initial critic loss: {losses[0]}")
    print(f"final critic loss: {losses[-1]}")
    print(f"loss finite: {np.isfinite(losses).all()}")
    print(f"optimizer step completed: {optimizer_step_completed}")

    assert len(buffer) == num_rollouts
    assert last_batch["start_state"].shape == (batch_size, probe_batch["start_state"].shape[-1])
    assert last_batch["action_subseq"].shape == (batch_size, segment_length, env.n_ubs, 2)
    assert last_batch["reward_subseq"].shape == (batch_size, segment_length, env.n_ubs)
    assert last_batch["state_subseq"].shape == (batch_size, segment_length + 1, probe_batch["start_state"].shape[-1])
    assert np.isfinite(losses).all()
    assert optimizer_step_completed


if __name__ == "__main__":
    main()
