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
    from envs.mubs_cov.mubs_cov_traj import MultiUbsCoverageTrajEnv
    from models.trajectory_generator import LinearWaypointTrajectoryGenerator

    np.random.seed(1)

    env = MultiUbsCoverageTrajEnv(map_id="debug", record=False)
    env.reset()

    H = 4
    max_move_per_slot = float(np.max(np.asarray(env.vels, dtype=np.float32)) * env.dt)
    generator = LinearWaypointTrajectoryGenerator(
        H=H,
        n_uavs=env.n_ubs,
        max_move_per_slot=max_move_per_slot,
        range_pos=env.range_pos,
    )

    current_pos = env.pos_ubs.copy()
    w = np.random.uniform(-200.0, 200.0, size=(env.n_ubs, 2)).astype(np.float32)
    continuous_action_seq = generator.generate(current_pos=current_pos, w=w)

    displacement_norms = np.linalg.norm(continuous_action_seq, axis=2)
    simulated_pos = current_pos[None, :, :] + np.cumsum(continuous_action_seq, axis=0)
    assert continuous_action_seq.shape == (H, env.n_ubs, 2)
    assert np.all(displacement_norms <= max_move_per_slot + 1e-5)
    assert np.all(simulated_pos >= -1e-5)
    assert np.all(simulated_pos <= env.range_pos + 1e-5)

    result = env.step_trajectory(continuous_action_seq, action_type="displacement")
    info = result["info"]

    print(f"w shape: {w.shape}")
    print(f"continuous_action_seq shape: {continuous_action_seq.shape}")
    print(f"state_seq shape: {result['state_seq'].shape}")
    print(f"reward_seq shape: {result['reward_seq'].shape}")
    print(f"cumulative_reward: {result['cumulative_reward']}")
    print(f"TotalThroughput: {info.get('TotalThroughput')}")
    print(f"FairIdx: {info.get('FairIdx')}")
    print(f"ProbCollision: {info.get('ProbCollision')}")
    print(f"done: {result['done']}")

    assert np.isfinite(result["cumulative_reward"])
    for key in ("TotalThroughput", "FairIdx", "ProbCollision"):
        assert key in info


if __name__ == "__main__":
    main()
