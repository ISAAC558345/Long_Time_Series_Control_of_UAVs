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


def collect_actor_rollout(env, actor, generator):
    import torch

    env.reset()
    state = env.get_state()
    state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        w, _, _, _ = actor.sample(state_tensor)

    # Conversion to numpy is only for non-differentiable environment interaction.
    w_np = w[0].cpu().numpy()
    action_seq = generator.generate(current_pos=env.pos_ubs.copy(), w=w_np)
    trajectory_info = env.step_trajectory(action_seq, action_type="displacement")
    return trajectory_info


def main():
    import torch

    from buffers.trajectory_replay_buffer import TrajectoryReplayBuffer
    from envs.mubs_cov.mubs_cov_traj import MultiUbsCoverageTrajEnv
    from models.actor_loss import compute_actor_loss
    from models.critic_loss import compute_critic_loss
    from models.trajectory_actor import TrajectoryActor
    from models.trajectory_generator import (
        LinearWaypointTrajectoryGenerator,
        TorchLinearWaypointTrajectoryGenerator,
    )
    from models.transformer_critic import TrajectoryTransformerCritic

    np.random.seed(8)
    torch.manual_seed(8)

    env = MultiUbsCoverageTrajEnv(map_id="debug", record=False)
    env.reset()

    H = 4
    batch_size = 4
    num_initial_rollouts = 6
    num_iters = 5
    state_dim = env.get_state().shape[-1]
    max_move_per_slot = float(np.max(np.asarray(env.vels, dtype=np.float32)) * env.dt)

    actor = TrajectoryActor(
        state_dim=state_dim,
        n_uavs=env.n_ubs,
        hidden_dim=64,
        w_scale=H * max_move_per_slot,
    )
    critic = TrajectoryTransformerCritic(
        state_dim=state_dim,
        n_uavs=env.n_ubs,
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_segment_length=H,
    )
    target_critic = deepcopy(critic)
    target_critic.eval()

    rollout_generator = LinearWaypointTrajectoryGenerator(
        H=H,
        n_uavs=env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=env.range_pos,
    )
    torch_generator = TorchLinearWaypointTrajectoryGenerator(
        H=H,
        n_uavs=env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=env.range_pos,
    )
    buffer = TrajectoryReplayBuffer(capacity=30, seed=789)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    last_rollout_info = None
    for _ in range(num_initial_rollouts):
        last_rollout_info = collect_actor_rollout(env, actor, rollout_generator)
        buffer.add_rollout(last_rollout_info)

    critic_losses = []
    actor_losses = []
    critic_optimizer_step_completed = False
    actor_optimizer_step_completed = False
    target_critic_update_completed = False

    for _ in range(num_iters):
        last_rollout_info = collect_actor_rollout(env, actor, rollout_generator)
        buffer.add_rollout(last_rollout_info)

        batch = buffer.sample_segments(batch_size=batch_size, segment_length=H)
        start_state = torch.as_tensor(batch["start_state"], dtype=torch.float32)
        action_subseq = torch.as_tensor(batch["action_subseq"], dtype=torch.float32)
        reward_subseq = torch.as_tensor(batch["reward_subseq"], dtype=torch.float32)
        bootstrap_state = torch.as_tensor(batch["bootstrap_state"], dtype=torch.float32)
        done_subseq = torch.as_tensor(batch["done_subseq"], dtype=torch.bool)
        state_subseq = torch.as_tensor(batch["state_subseq"], dtype=torch.float32)

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

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        critic_optimizer_step_completed = True

        target_critic.load_state_dict(critic.state_dict())
        target_critic.eval()
        target_critic_update_completed = True

        actor_loss, actor_info = compute_actor_loss(
            actor=actor,
            critic=critic,
            start_state=start_state,
            n_uavs=env.n_ubs,
            H=torch_generator.H,
            range_pos=env.range_pos,
            max_move_per_slot=max_move_per_slot,
        )
        assert torch.isfinite(actor_loss)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_nonzero = any(
            param.grad is not None and torch.any(param.grad.detach() != 0)
            for param in actor.parameters()
        )
        assert actor_grad_nonzero
        actor_optimizer.step()
        actor_optimizer_step_completed = True

        critic_losses.append(float(critic_loss.item()))
        actor_losses.append(float(actor_loss.item()))

    info = last_rollout_info["info"]
    critic_loss_finite = bool(np.isfinite(critic_losses).all())
    actor_loss_finite = bool(np.isfinite(actor_losses).all())

    print(f"buffer size: {len(buffer)}")
    print(f"critic_loss list: {critic_losses}")
    print(f"actor_loss list: {actor_losses}")
    print(f"initial critic loss: {critic_losses[0]}")
    print(f"final critic loss: {critic_losses[-1]}")
    print(f"initial actor loss: {actor_losses[0]}")
    print(f"final actor loss: {actor_losses[-1]}")
    print(f"critic loss finite: {critic_loss_finite}")
    print(f"actor loss finite: {actor_loss_finite}")
    print(f"critic optimizer step completed: {critic_optimizer_step_completed}")
    print(f"actor optimizer step completed: {actor_optimizer_step_completed}")
    print(f"target critic update completed: {target_critic_update_completed}")
    print(f"last TotalThroughput: {info.get('TotalThroughput')}")
    print(f"last FairIdx: {info.get('FairIdx')}")
    print(f"last ProbCollision: {info.get('ProbCollision')}")

    assert len(buffer) == num_initial_rollouts + num_iters
    assert len(critic_losses) == num_iters
    assert len(actor_losses) == num_iters
    assert actor_info["continuous_action_seq"].shape == (batch_size, H, env.n_ubs, 2)
    assert critic_loss_finite
    assert actor_loss_finite
    assert critic_optimizer_step_completed
    assert actor_optimizer_step_completed
    assert target_critic_update_completed
    for key in ("TotalThroughput", "FairIdx", "ProbCollision"):
        assert key in info


if __name__ == "__main__":
    main()
