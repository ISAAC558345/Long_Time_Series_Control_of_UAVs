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
    from models.actor_loss import compute_actor_loss
    from models.trajectory_actor import TrajectoryActor
    from models.trajectory_generator import LinearWaypointTrajectoryGenerator
    from models.transformer_critic import TrajectoryTransformerCritic

    np.random.seed(7)
    torch.manual_seed(7)

    env = MultiUbsCoverageTrajEnv(map_id="debug", record=False)
    H = 4
    batch_size = 4
    num_rollouts = 6
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

    batch = buffer.sample_segments(batch_size=batch_size, segment_length=H)
    start_state = torch.as_tensor(batch["start_state"], dtype=torch.float32)

    actor = TrajectoryActor(
        state_dim=start_state.shape[-1],
        n_uavs=env.n_ubs,
        hidden_dim=64,
        w_scale=H * max_move_per_slot,
    )
    critic = TrajectoryTransformerCritic(
        state_dim=start_state.shape[-1],
        n_uavs=env.n_ubs,
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_segment_length=H,
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    critic_params_before = [param.detach().clone() for param in critic.parameters()]
    actor_loss, info = compute_actor_loss(
        actor=actor,
        critic=critic,
        start_state=start_state,
        n_uavs=env.n_ubs,
        H=H,
        range_pos=env.range_pos,
        max_move_per_slot=max_move_per_slot,
    )
    actor_loss_finite = bool(torch.isfinite(actor_loss).item())

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_grad_nonzero = any(
        param.grad is not None and torch.any(param.grad.detach() != 0)
        for param in actor.parameters()
    )
    actor_optimizer.step()
    actor_optimizer_step_completed = True

    critic_params_unchanged = all(
        torch.allclose(before, after.detach())
        for before, after in zip(critic_params_before, critic.parameters())
    )

    print(f"start_state shape: {tuple(start_state.shape)}")
    print(f"current_pos shape: {tuple(info['current_pos'].shape)}")
    print(f"raw_w shape: {tuple(info['raw_w'].shape)}")
    print(f"log_std shape: {tuple(info['log_std'].shape)}")
    print(f"sampled w shape: {tuple(info['w'].shape)}")
    print(f"continuous_action_seq shape: {tuple(info['continuous_action_seq'].shape)}")
    print(f"prefix_q_values shape: {tuple(info['prefix_q_values'].shape)}")
    print(f"actor_loss: {float(actor_loss.item())}")
    print(f"actor_loss finite: {actor_loss_finite}")
    print(f"actor grad nonzero: {actor_grad_nonzero}")
    print(f"actor optimizer step completed: {actor_optimizer_step_completed}")
    print(f"critic params unchanged: {critic_params_unchanged}")

    assert start_state.shape == (batch_size, start_state.shape[-1])
    assert info["current_pos"].shape == (batch_size, env.n_ubs, 2)
    assert info["raw_w"].shape == (batch_size, env.n_ubs * 2)
    assert info["log_std"].shape == (batch_size, env.n_ubs * 2)
    assert info["w"].shape == (batch_size, env.n_ubs, 2)
    assert info["continuous_action_seq"].shape == (batch_size, H, env.n_ubs, 2)
    assert info["prefix_q_values"].shape == (batch_size, H)
    assert actor_loss_finite
    assert actor_grad_nonzero
    assert actor_optimizer_step_completed
    assert critic_params_unchanged


if __name__ == "__main__":
    main()
