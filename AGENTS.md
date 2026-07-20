# AGENTS.md

## Project Goal

This repository adapts a UAV-BS communication trajectory-control codebase into a TOP-ERL-inspired long-horizon trajectory-control framework.

The base UAV system should be treated as the communication environment. The later learning framework should allow an actor to output a long-horizon UAV trajectory or trajectory parameters, instead of selecting one instantaneous per-slot movement action.

## Core Rules

1. Read the repository structure before editing any source code.
2. Do not rewrite the whole project.
3. Preserve the original UAV communication model whenever possible, including A2G channel, interference, SINR, rate, throughput, fairness metric, collision penalty, and plotting utilities.
4. Do not implement the full TOP-ERL algorithm in the first step.
5. Do not add Transformer critic, ProDMP, B-spline, TRPL, full covariance Gaussian policy, or ensemble critics unless explicitly requested later.
6. Prefer small, testable changes over large refactors.
7. Keep original experiment scripts runnable unless the user explicitly asks to change them.
8. For every code change, report:

   * modified files;
   * new files;
   * key interface changes;
   * commands to run;
   * whether tests or smoke tests passed.
9. Do not run long training jobs. Use smoke tests or short runs only.
10. Write explanations and review reports in Chinese unless the user explicitly asks for English.

## Current Stage

The current stage is code inspection and adaptation planning.

Do not modify source code yet. First create a code review document named:

`CODE_REVIEW_TOP_ERL_ADAPTATION.md`

The review should identify where the existing code defines:

* UAV number, user number, area size, altitude, speed, and episode length;
* A2G channel model;
* UAV action space;
* UAV position update in `step()`;
* user scheduling;
* interference, SINR, rate, throughput;
* reward function;
* fairness and collision penalty;
* replay buffer and training loop.

## Target Interface for Later Work

Later, the environment should support a long-horizon trajectory action interface:

`step_trajectory(action_seq)`

where `action_seq` has shape:

`[H, n_uavs, 2]`

and represents future UAV horizontal displacements or velocities for the next `H` slots.

The later interface should return:

* final observation;
* cumulative reward;
* done flag;
* info dictionary;
* state sequence;
* action sequence;
* reward sequence;
* rate sequence;
* SINR sequence if available.

## Important Distinction

The original per-slot action selects one movement/action for the current time slot.

The target TOP-ERL-style action should represent a long-horizon trajectory or trajectory parameter, allowing the policy to plan a sequence of future UAV motions.

Do not confuse long-horizon trajectory control with resource allocation. The first adaptation should focus on UAV trajectory control only. Resource allocation should remain fixed, greedy, or unchanged from the original code unless requested later.
