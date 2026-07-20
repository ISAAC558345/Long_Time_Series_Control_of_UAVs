# docs/adaptation_plan.md

## 1. Research Goal

The goal is to adapt an existing UAV-BS communication trajectory-control environment into a TOP-ERL-inspired long-horizon trajectory-control environment.

The base UAV code should provide the communication model:

* UAV-BS positions;
* ground user positions;
* A2G channel;
* interference;
* SINR;
* rate;
* throughput;
* fairness;
* collision or safety penalties.

The TOP-ERL-inspired part should later replace per-slot UAV actions with long-horizon trajectory actions.

## 2. Current Task

Current task: only inspect the codebase and write an adaptation plan.

Do not modify algorithms yet.

The expected output is:

`CODE_REVIEW_TOP_ERL_ADAPTATION.md`

## 3. Files to Inspect First

Please inspect these files or folders first:

* `README.md`
* `envs/`
* `algos/`
* `run_exp1.py`
* `run_exp2.py`
* `run_exp3.py`
* any configuration files
* any replay buffer or policy files

If the repository has a specific UAV-BS environment file, inspect it carefully. In the known version of the codebase, important files may include:

* `envs/mubs_cov/mubs_cov.py`
* `envs/common.py`

## 4. What to Identify

The code review should identify:

1. Where the UAV number, ground terminal number, map size, altitude, velocity, and episode length are defined.
2. Where UAV positions and ground user positions are initialized.
3. Where the A2G channel model is computed.
4. Where interference is computed.
5. Where SINR is computed.
6. Where rate, throughput, and fairness are computed.
7. Where the reward function is constructed.
8. Where the UAV action space is defined.
9. Where `step()` updates UAV positions.
10. Whether the current action is discrete movement, continuous velocity, waypoint, or another form.
11. Whether collision penalty, boundary penalty, or coverage radius constraints exist.
12. Where training loops and evaluation loops are located.
13. Which functions can be reused for long-horizon trajectory control.
14. Which functions need wrappers or refactoring.

## 5. Adaptation Direction

The intended adaptation should be done gradually.

### Stage 1: Code Review Only

Create `CODE_REVIEW_TOP_ERL_ADAPTATION.md`.

No source-code modification.

### Stage 2: Add Long-Horizon Environment Wrapper

Add a new environment wrapper or subclass, for example:

`envs/mubs_cov/mubs_cov_traj.py`

Target class name:

`MultiUbsCoverageTrajEnv`

This class should reuse the original communication model and add:

`step_trajectory(action_seq)`

where `action_seq` has shape:

`[H, n_uavs, 2]`

The method should internally execute `H` consecutive per-slot movements and record:

* state sequence;
* action sequence;
* reward sequence;
* rate sequence;
* SINR sequence if available;
* cumulative reward.

### Stage 3: Add Trajectory Generator

Add a simple trajectory generator later.

First version should use simple polynomial or B-spline basis.

Do not implement ProDMP at the beginning.

Possible file:

`models/trajectory_generator.py`

Input:

* current UAV positions;
* trajectory parameters;
* planning horizon `H`.

Output:

* future displacement sequence or waypoint sequence.

### Stage 4: Add TOP-ERL-Inspired Replay Buffer

Add a replay buffer that stores full trajectory rollouts and supports random segment sampling.

Possible file:

`buffers/trajectory_replay_buffer.py`

It should store:

* state sequence;
* action sequence;
* reward sequence;
* done sequence;
* optional trajectory parameters.

It should support sampling:

* segment start state;
* action subsequence;
* reward subsequence;
* bootstrap state.

### Stage 5: Add Transformer Critic

Add a simplified Transformer critic later.

Possible file:

`models/transformer_critic.py`

Input tokens:

* first token: segment start state;
* following tokens: action sequence.

Output:

* state value;
* prefix Q-values for action subsequences.

Do not implement full TOP-ERL at first. Start with a minimal working critic.

### Stage 6: Add Simplified Training Script

Possible file:

`algos/top_erl_simplified.py`

Possible script:

`scripts/train_top_erl_uav.py`

The training loop should be added only after the environment wrapper, trajectory generator, replay buffer, and critic interface are individually smoke-tested.

## 6. Testing Requirements

Every stage should include a smoke test.

For the trajectory environment wrapper, add:

`scripts/smoke_test_traj_env.py`

The smoke test should:

1. create the environment;
2. reset the environment;
3. generate random trajectory actions with shape `[H, n_uavs, 2]`;
4. call `step_trajectory(action_seq)`;
5. print cumulative reward, average rate, fairness, collision count, and done flag;
6. save a UAV trajectory figure if plotting utilities are available.

Do not run long training jobs.

## 7. Reporting Format

When creating `CODE_REVIEW_TOP_ERL_ADAPTATION.md`, use the following structure:

1. Repository overview
2. Main environment files
3. Communication model
4. State and observation design
5. Action space and UAV movement update
6. Reward and metrics
7. Training and evaluation scripts
8. Reusable modules
9. Modules requiring refactoring
10. Minimal plan for adding `step_trajectory(action_seq)`
11. Risks and unclear points
12. Recommended next Codex task
