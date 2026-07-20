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

    from envs.mubs_cov.mubs_cov_traj import MultiUbsCoverageTrajEnv
    from models.trajectory_actor import TrajectoryActor
    from models.trajectory_generator import LinearWaypointTrajectoryGenerator

    np.random.seed(6)
    torch.manual_seed(6)

    env = MultiUbsCoverageTrajEnv(map_id="debug", record=False)
    env.reset()

    H = 4
    state = env.get_state()
    max_move_per_slot = float(np.max(np.asarray(env.vels, dtype=np.float32)) * env.dt)
    w_scale = H * max_move_per_slot

    actor = TrajectoryActor(
        state_dim=state.shape[-1],
        n_uavs=env.n_ubs,
        hidden_dim=64,
        w_scale=w_scale,
    )
    generator = LinearWaypointTrajectoryGenerator(
        H=H,
        n_uavs=env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=env.range_pos,
    )

    w, mean, log_std, raw_w = actor.sample(state)
    w_np = w[0].detach().cpu().numpy()
    continuous_action_seq = generator.generate(current_pos=env.pos_ubs.copy(), w=w_np)
    result = env.step_trajectory(continuous_action_seq, action_type="displacement")
    info = result["info"]

    print(f"state shape: {state.shape}")
    print(f"mean shape: {tuple(mean.shape)}")
    print(f"log_std shape: {tuple(log_std.shape)}")
    print(f"sampled w shape: {tuple(w.shape)}")
    print(f"continuous_action_seq shape: {continuous_action_seq.shape}")
    print(f"cumulative_reward: {result['cumulative_reward']}")
    print(f"TotalThroughput: {info.get('TotalThroughput')}")
    print(f"FairIdx: {info.get('FairIdx')}")
    print(f"ProbCollision: {info.get('ProbCollision')}")
    print(f"done: {result['done']}")

    assert raw_w.shape == (1, env.n_ubs * 2)
    assert mean.shape == (1, env.n_ubs * 2)
    assert log_std.shape == (1, env.n_ubs * 2)
    assert w.shape == (1, env.n_ubs, 2)
    assert continuous_action_seq.shape == (H, env.n_ubs, 2)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_std).all()
    assert torch.isfinite(w).all()
    assert np.isfinite(result["cumulative_reward"])
    for key in ("TotalThroughput", "FairIdx", "ProbCollision"):
        assert key in info


if __name__ == "__main__":
    main()
