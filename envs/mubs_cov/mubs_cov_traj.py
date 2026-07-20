import numpy as np

from envs.mubs_cov.mubs_cov import MultiUbsCoverageEnv


class MultiUbsCoverageTrajEnv(MultiUbsCoverageEnv):
    """Trajectory-action wrapper for the multi-UAV coverage environment."""

    def step_trajectory(self, action_seq, action_type="displacement"):
        """
        Execute a continuous long-horizon UAV trajectory.

        Args:
            action_seq: ndarray-like with shape [H, n_uavs, 2].
            action_type: "displacement" or "velocity". Velocity is multiplied by dt.

        Returns:
            A dict containing final outputs and trajectory sequences.
        """
        if self.t is None:
            raise RuntimeError("Call reset() before step_trajectory().")
        if action_type not in ("displacement", "velocity"):
            raise ValueError("action_type must be 'displacement' or 'velocity'.")

        action_seq = np.asarray(action_seq, dtype=np.float32)
        if action_seq.ndim != 3 or action_seq.shape[1:] != (self.n_ubs, 2):
            raise ValueError(
                f"action_seq must have shape [H, {self.n_ubs}, 2], got {action_seq.shape}."
            )

        obs_seq = [self.get_obs()]
        state_seq = [self.get_state().copy()]
        pos_ubs_seq = [self.pos_ubs.copy()]
        pos_gts_seq = [self.pos_gts.copy()]
        rate_seq = [self.rate_per_gt.copy()]
        avg_rate_seq = [self.avg_rate_per_gt.copy()]
        fair_idx_seq = [self.fair_idx]
        throughput_seq = [self.total_throughput]
        global_util_seq = [self.global_util]

        executed_actions = []
        displacement_seq = []
        reward_seq = []
        done_seq = []
        info_seq = []
        cumulative_reward = 0.0
        done = False
        info = {}

        for action in action_seq:
            if self._get_terminate():
                done = True
                break

            self.t += 1
            displacement = action * self.dt if action_type == "velocity" else action
            self.pos_ubs = np.clip(self.pos_ubs + displacement, 0, self.range_pos)

            self._transmit_data()
            reward = self._get_reward()
            mean_reward = float(np.mean(reward))
            cumulative_reward += mean_reward
            self.ep_ret += mean_reward

            done = self._get_terminate()
            info = dict(
                EpRet=self.ep_ret,
                EpLen=self.t,
                AvgGlobalUtility=self.avg_global_util,
                FairIdx=self.fair_idx,
                TotalThroughput=self.total_throughput,
                ProbCollision=self.n_colls / self.t,
            )
            info["BadMask"] = True if self.t == self.episode_limit else False

            if self.recorder is not None:
                self.recorder.click(pos_ubs=self.pos_ubs.copy(), fair_idx=self.fair_idx, reward=mean_reward)

            executed_actions.append(action.copy())
            displacement_seq.append(displacement.copy())
            reward_seq.append(reward.copy())
            done_seq.append(done)
            info_seq.append(info.copy())
            obs_seq.append(self.get_obs())
            state_seq.append(self.get_state().copy())
            pos_ubs_seq.append(self.pos_ubs.copy())
            pos_gts_seq.append(self.pos_gts.copy())
            rate_seq.append(self.rate_per_gt.copy())
            avg_rate_seq.append(self.avg_rate_per_gt.copy())
            fair_idx_seq.append(self.fair_idx)
            throughput_seq.append(self.total_throughput)
            global_util_seq.append(self.global_util)

            if done:
                break

        return dict(
            final_obs=obs_seq[-1],
            final_state=state_seq[-1],
            cumulative_reward=cumulative_reward,
            done=done,
            info=info,
            obs_seq=obs_seq,
            state_seq=np.stack(state_seq),
            action_seq=np.asarray(executed_actions, dtype=np.float32).reshape((-1, self.n_ubs, 2)),
            displacement_seq=np.asarray(displacement_seq, dtype=np.float32).reshape((-1, self.n_ubs, 2)),
            reward_seq=np.asarray(reward_seq, dtype=np.float32).reshape((-1, self.n_agents)),
            done_seq=np.asarray(done_seq, dtype=bool),
            info_seq=info_seq,
            pos_ubs_seq=np.stack(pos_ubs_seq),
            pos_gts_seq=np.stack(pos_gts_seq),
            rate_seq=np.stack(rate_seq),
            avg_rate_seq=np.stack(avg_rate_seq),
            fair_idx_seq=np.asarray(fair_idx_seq, dtype=np.float32),
            throughput_seq=np.asarray(throughput_seq, dtype=np.float32),
            global_util_seq=np.asarray(global_util_seq, dtype=np.float32),
        )
