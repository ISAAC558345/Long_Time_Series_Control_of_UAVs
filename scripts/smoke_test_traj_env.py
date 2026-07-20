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

    np.random.seed(0)

    env = MultiUbsCoverageTrajEnv(map_id="debug", record=False)
    env.reset()

    horizon = 3
    action_seq = np.random.uniform(-10.0, 10.0, size=(horizon, env.n_ubs, 2)).astype(np.float32)
    result = env.step_trajectory(action_seq, action_type="displacement")
    info = result["info"]

    print(f"cumulative_reward: {result['cumulative_reward']}")
    print(f"final done: {result['done']}")
    print(f"EpRet: {info.get('EpRet')}")
    print(f"TotalThroughput: {info.get('TotalThroughput')}")
    print(f"FairIdx: {info.get('FairIdx')}")
    print(f"ProbCollision: {info.get('ProbCollision')}")
    print(f"action_seq shape: {result['action_seq'].shape}")
    print(f"state_seq shape: {result['state_seq'].shape}")
    print(f"reward_seq shape: {result['reward_seq'].shape}")

    assert result["action_seq"].shape == (horizon, env.n_ubs, 2)
    assert result["state_seq"].shape[0] == horizon + 1
    assert result["reward_seq"].shape == (horizon, env.n_agents)
    assert np.isfinite(result["cumulative_reward"])
    for key in ("EpRet", "TotalThroughput", "FairIdx", "ProbCollision"):
        assert key in info


if __name__ == "__main__":
    main()
