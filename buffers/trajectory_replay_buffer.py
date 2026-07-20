import copy

import numpy as np


class TrajectoryReplayBuffer:
    """Replay buffer for complete trajectory rollouts and fixed-length segments."""

    def __init__(self, capacity=None, seed=None):
        if capacity is not None and int(capacity) <= 0:
            raise ValueError("capacity must be positive or None.")

        self.capacity = None if capacity is None else int(capacity)
        self.rollouts = []
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rollouts)

    def add_rollout(self, trajectory_info):
        state_seq = np.asarray(trajectory_info["state_seq"], dtype=np.float32).copy()
        action_seq = np.asarray(trajectory_info["action_seq"], dtype=np.float32).copy()
        reward_seq = np.asarray(trajectory_info["reward_seq"], dtype=np.float32).copy()
        done_seq = np.asarray(trajectory_info["done_seq"], dtype=bool).copy()

        self._validate_rollout_shapes(state_seq, action_seq, reward_seq, done_seq)

        rollout = dict(
            state_seq=state_seq,
            action_seq=action_seq,
            reward_seq=reward_seq,
            done_seq=done_seq,
            info_seq=copy.deepcopy(trajectory_info.get("info_seq", [])),
        )
        if "obs_seq" in trajectory_info:
            rollout["obs_seq"] = self._copy_nested_arrays(trajectory_info["obs_seq"])

        self.rollouts.append(rollout)
        if self.capacity is not None and len(self.rollouts) > self.capacity:
            self.rollouts.pop(0)

    def sample_segments(self, batch_size, segment_length):
        batch_size = int(batch_size)
        segment_length = int(segment_length)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if segment_length <= 0:
            raise ValueError("segment_length must be positive.")

        eligible = [
            idx for idx, rollout in enumerate(self.rollouts)
            if rollout["action_seq"].shape[0] >= segment_length
        ]
        if not eligible:
            raise ValueError("No rollout is long enough for the requested segment_length.")

        start_states = []
        state_subseqs = []
        action_subseqs = []
        reward_subseqs = []
        bootstrap_states = []
        done_subseqs = []
        rollout_indices = []
        start_indices = []

        for _ in range(batch_size):
            rollout_idx = int(self.rng.choice(eligible))
            rollout = self.rollouts[rollout_idx]
            horizon = rollout["action_seq"].shape[0]
            start = int(self.rng.integers(0, horizon - segment_length + 1))
            end = start + segment_length

            start_states.append(rollout["state_seq"][start])
            state_subseqs.append(rollout["state_seq"][start:end + 1])
            action_subseqs.append(rollout["action_seq"][start:end])
            reward_subseqs.append(rollout["reward_seq"][start:end])
            bootstrap_states.append(rollout["state_seq"][end])
            done_subseqs.append(rollout["done_seq"][start:end])
            rollout_indices.append(rollout_idx)
            start_indices.append(start)

        return dict(
            start_state=np.stack(start_states),
            state_subseq=np.stack(state_subseqs),
            action_subseq=np.stack(action_subseqs),
            reward_subseq=np.stack(reward_subseqs),
            bootstrap_state=np.stack(bootstrap_states),
            done_subseq=np.stack(done_subseqs),
            rollout_indices=np.asarray(rollout_indices, dtype=np.int64),
            start_indices=np.asarray(start_indices, dtype=np.int64),
        )

    @staticmethod
    def _validate_rollout_shapes(state_seq, action_seq, reward_seq, done_seq):
        if state_seq.ndim != 2:
            raise ValueError(f"state_seq must have shape [H+1, state_dim], got {state_seq.shape}.")
        if action_seq.ndim != 3 or action_seq.shape[2] != 2:
            raise ValueError(f"action_seq must have shape [H, n_uavs, 2], got {action_seq.shape}.")
        if reward_seq.ndim != 2:
            raise ValueError(f"reward_seq must have shape [H, n_uavs], got {reward_seq.shape}.")
        if done_seq.ndim != 1:
            raise ValueError(f"done_seq must have shape [H], got {done_seq.shape}.")

        horizon = action_seq.shape[0]
        n_uavs = action_seq.shape[1]
        if state_seq.shape[0] != horizon + 1:
            raise ValueError("state_seq length must equal action_seq length + 1.")
        if reward_seq.shape != (horizon, n_uavs):
            raise ValueError("reward_seq must have shape [H, n_uavs] matching action_seq.")
        if done_seq.shape[0] != horizon:
            raise ValueError("done_seq length must match action_seq length.")

    @classmethod
    def _copy_nested_arrays(cls, value):
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, list):
            return [cls._copy_nested_arrays(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._copy_nested_arrays(v) for v in value)
        if isinstance(value, dict):
            return {k: cls._copy_nested_arrays(v) for k, v in value.items()}
        return copy.deepcopy(value)
