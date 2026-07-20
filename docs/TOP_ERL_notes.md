# docs/TOP_ERL_notes.md

## 1. Purpose

This note summarizes how TOP-ERL should inspire the UAV trajectory-control adaptation.

The goal is not to fully reproduce TOP-ERL at the beginning. The goal is to borrow its key idea: the policy should output a long-horizon action trajectory or trajectory parameters, and the critic should evaluate action sequences rather than only one-step actions.

## 2. Key Idea of TOP-ERL

TOP-ERL uses an episodic reinforcement learning formulation.

Instead of choosing one action at every environment step, the policy predicts trajectory parameters:

`w ~ pi(w | s)`

A trajectory generator maps `w` into a full action trajectory:

`a_0, a_1, ..., a_{T-1}`

The environment then executes this action trajectory and records:

* states;
* actions;
* rewards;
* done flags.

The trajectory is later divided into shorter segments. A Transformer critic evaluates the value of each action sequence segment.

## 3. Difference from Step-Based RL

In ordinary step-based RL, the policy outputs one action:

`a_t ~ pi(a_t | s_t)`

In TOP-ERL-style control, the policy outputs a trajectory parameter or a long action sequence:

`w ~ pi(w | s_0)`

or

`a_{0:H-1} ~ pi(a_{0:H-1} | s_0)`

This is useful for UAV trajectory planning because UAV motion should be smooth and long-horizon. It is often unnatural to control only an instantaneous velocity without considering the future path.

## 4. How to Map TOP-ERL to UAV Communication

In the UAV communication environment:

State may include:

* UAV positions;
* ground user positions;
* distance matrix;
* channel gains;
* SINR or rate information;
* remaining episode length;
* previous UAV motion if needed.

Action sequence may be:

* future UAV velocity sequence;
* future UAV displacement sequence;
* future UAV waypoint sequence;
* trajectory parameters that are decoded into UAV positions.

A simple first version should use:

`action_seq.shape = [H, n_uavs, 2]`

where each `[n_uavs, 2]` slice represents the horizontal displacement or velocity of all UAVs at one future slot.

## 5. Receding-Horizon Use

For UAV communication, it is better to use a receding-horizon style.

At decision time `t`, the actor generates a future trajectory for the next `H` slots.

The environment executes this sequence, or executes part of it, then replans again.

This is more flexible than generating one fixed trajectory for the whole episode, because wireless channels, user coverage, interference, and UAV geometry change over time.

## 6. Transformer Critic Concept

The Transformer critic should evaluate a segment:

`(s_k, a_k, a_{k+1}, ..., a_{k+L-1})`

Input tokens:

1. state token for `s_k`;
2. action token for `a_k`;
3. action token for `a_{k+1}`;
4. ...
5. action token for `a_{k+L-1}`.

The critic outputs:

* `V(s_k)`;
* `Q(s_k, a_k)`;
* `Q(s_k, a_k, a_{k+1})`;
* ...
* `Q(s_k, a_k, ..., a_{k+L-1})`.

A causal mask should be used so each action prefix does not attend to future actions.

## 7. N-Step Return Target

For a sequence prefix of length `N`, the target can be written as:

`G_N = r_k + gamma r_{k+1} + ... + gamma^{N-1} r_{k+N-1} + gamma^N V_target(s_{k+N})`

This allows the critic to learn the value of action subsequences.

## 8. Simplifications for This Project

Do not implement all TOP-ERL details at the beginning.

First version should avoid:

* ProDMP;
* full covariance Gaussian policy;
* TRPL;
* ensemble critics;
* clipped double Q;
* random segment length;
* complex off-policy corrections.

First version should implement:

1. long-horizon trajectory action interface;
2. rollout storage with state/action/reward sequences;
3. random segment sampling;
4. simple Transformer critic;
5. fixed segment length;
6. short smoke-test training.

## 9. First Milestone

The first technical milestone is not the Transformer.

The first milestone is:

`step_trajectory(action_seq)`

This method must execute a future UAV trajectory sequence inside the original UAV communication environment and return all sequences needed for later Transformer critic training.

Without this interface, TOP-ERL adaptation cannot be implemented cleanly.
