# Stochastic Collision Diagnostics

This report evaluates saved checkpoints only. Eval-time std scaling is diagnostic and does not modify training or checkpoints.

## Existing 200-Iteration Summary

| seed | det reward | stoch reward | random reward | det gap | stoch gap | det FairIdx | stoch FairIdx | random FairIdx | det collision | stoch collision | random collision | finite |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.504320 | -0.059742 | 0.083097 | 0.421224 | -0.142838 | 0.640018 | 0.447901 | 0.341414 | 0.000000 | 0.012500 | 0.000000 | True |
| 1 | 0.104320 | 0.106346 | 0.085005 | 0.019315 | 0.021342 | 0.398194 | 0.402282 | 0.442980 | 0.000000 | 0.000000 | 0.000000 | True |
| 2 | 0.081839 | 0.137420 | 0.041266 | 0.040573 | 0.096154 | 0.250001 | 0.405033 | 0.326972 | 0.000000 | 0.000000 | 0.000000 | True |

## Std-Scale Evaluation

| seed | policy | reward | FairIdx | ProbCollision | collision episode ratio | mean step | max step | min inter-UAV | boundary ratio | abnormal jump ratio | finite |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | deterministic_actor | 0.504320 | 0.640018 | 0.000000 | 0.000000 | 64.231173 | 64.835772 | 294.677341 | 0.000000 | 0.000000 | True |
| 0 | random_baseline | 0.071742 | 0.355781 | 0.000000 | 0.000000 | 67.714473 | 100.000030 | 70.100825 | 0.042000 | 0.260000 | True |
| 0 | stochastic_std_0.25 | 0.250446 | 0.567043 | 0.000000 | 0.000000 | 63.344467 | 100.000025 | 31.005675 | 0.014000 | 0.040000 | True |
| 0 | stochastic_std_0.5 | 0.161763 | 0.480811 | 0.000000 | 0.000000 | 72.334986 | 100.000022 | 56.846704 | 0.023333 | 0.060000 | True |
| 0 | stochastic_std_1.0 | 0.093311 | 0.436277 | 0.000000 | 0.000000 | 80.890757 | 100.000033 | 68.232812 | 0.037333 | 0.280000 | True |
| 1 | deterministic_actor | 0.104320 | 0.398194 | 0.000000 | 0.000000 | 35.878295 | 62.516237 | 335.276223 | 0.000000 | 0.000000 | True |
| 1 | random_baseline | 0.067969 | 0.384001 | 0.000000 | 0.000000 | 68.216669 | 100.000038 | 65.873202 | 0.036667 | 0.160000 | True |
| 1 | stochastic_std_0.25 | 0.075815 | 0.379778 | 0.000000 | 0.000000 | 48.624321 | 100.000009 | 145.033524 | 0.013333 | 0.000000 | True |
| 1 | stochastic_std_0.5 | 0.061767 | 0.382938 | 0.000000 | 0.000000 | 58.046651 | 100.000036 | 103.146543 | 0.038667 | 0.100000 | True |
| 1 | stochastic_std_1.0 | 0.070654 | 0.378473 | 0.000000 | 0.000000 | 69.927930 | 100.000034 | 59.149503 | 0.051333 | 0.360000 | True |
| 2 | deterministic_actor | 0.081839 | 0.250001 | 0.000000 | 0.000000 | 23.835312 | 33.456277 | 502.101384 | 0.000000 | 0.000000 | True |
| 2 | random_baseline | 0.051366 | 0.360012 | 0.000000 | 0.000000 | 66.476899 | 100.000024 | 53.422343 | 0.046000 | 0.100000 | True |
| 2 | stochastic_std_0.25 | 0.089726 | 0.285097 | 0.000000 | 0.000000 | 36.682391 | 76.340040 | 289.656013 | 0.008000 | 0.000000 | True |
| 2 | stochastic_std_0.5 | 0.105782 | 0.351059 | 0.000000 | 0.000000 | 54.655358 | 100.000012 | 37.530499 | 0.022000 | 0.020000 | True |
| 2 | stochastic_std_1.0 | 0.066541 | 0.349502 | 0.000000 | 0.000000 | 69.378966 | 100.000035 | 43.258966 | 0.038667 | 0.300000 | True |

## Diagnostic Answers

- Seed 0 stochastic_std_1.0 reward is 0.093311, random reward is 0.071742, deterministic reward is 0.504320.
- Seed 0 stochastic_std_1.0 collision episode ratio is 0.000000; std_0.5 is 0.000000; std_0.25 is 0.000000.
- Seed 0 stochastic_std_0.5 reward is 0.161763; stochastic_std_0.25 reward is 0.250446.
- Hidden collision cases across all summaries: [].
- All metrics finite: True.

## Interpretation

- If std scaling removes collisions and restores reward above random, the issue is likely eval-time exploration scale.
- If collisions vanish but reward remains weak, the stochastic samples are not only unsafe but also lower-quality than the deterministic mean.
- Deterministic actor should be the current main evaluation policy for this checkpoint family; stochastic actor remains useful as auxiliary robustness diagnostics.
- This is diagnostic only, not a final algorithm conclusion.
