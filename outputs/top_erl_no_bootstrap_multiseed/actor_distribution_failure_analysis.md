# Actor distribution failure analysis

This report analyzes saved checkpoints and evaluation logs only. It does not retrain or modify algorithms.

## Evaluation comparison

| seed | deterministic reward | stochastic reward | random reward | det-random | stoch-random | stoch-det | det FairIdx | stoch FairIdx | random FairIdx | det throughput | stoch throughput | random throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.253602 | 0.126018 | 0.083097 | 0.170505 | 0.042922 | -0.127583 | 0.558028 | 0.442755 | 0.341414 | 0.102093 | 0.066309 | 0.062968 |
| 1 | 0.038322 | 0.103868 | 0.085005 | -0.046682 | 0.018863 | 0.065545 | 0.500001 | 0.417442 | 0.442980 | 0.039525 | 0.066782 | 0.061619 |
| 2 | 0.080657 | 0.105352 | 0.041266 | 0.039391 | 0.064086 | 0.024695 | 0.250001 | 0.387143 | 0.326972 | 0.103074 | 0.074032 | 0.048535 |

## Actor output statistics

| seed | mean_abs_mean | max_abs_mean | mean_log_std | min_log_std | max_log_std | mean_std | max_std | det w norm | stoch w norm | det mean step | det max step | stoch mean step | stoch max step | near zero | tanh sat ratio | det clip ratio | finite |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|
| 0 | 0.214559 | 0.468348 | 0.086060 | -0.198403 | 0.445169 | 1.116198 | 1.560754 | 129.337448 | 347.120941 | 32.137840 | 48.453346 | 72.942741 | 100.000038 | False | 0.000000 | 0.000000 | True |
| 1 | 0.104189 | 0.370870 | 0.042667 | -0.209594 | 0.345898 | 1.059523 | 1.413259 | 74.039154 | 351.096466 | 17.268253 | 35.490799 | 64.381927 | 100.000031 | False | 0.000000 | 0.000000 | True |
| 2 | 0.077884 | 0.202596 | -0.011136 | -0.147773 | 0.169121 | 0.992637 | 1.184263 | 49.357922 | 326.256073 | 12.315550 | 20.189777 | 65.525200 | 100.000023 | False | 0.000000 | 0.000000 | True |

## Per-seed diagnosis

### Seed 0
- no obvious distribution pathology

### Seed 1
- stochastic policy is meaningfully better than deterministic mean
- deterministic actor is below random baseline

### Seed 2
- no obvious distribution pathology

## Short answers

- Seed 1: failure is more consistent with deterministic mean degradation than total stochastic policy failure; stochastic reward is nonzero and above deterministic, though still below random.
- Seed 2: failure is more consistent with insufficient/unstable learning than pure deterministic extraction; stochastic is better than deterministic and random in reward, while deterministic stays weak.
- log_std is not collapsed to an extremely small value in these checkpoints; exploration remains active.
- deterministic w is not near zero, so the actor is not simply stationary.
- deterministic clipping is not dominant; tanh saturation should be inspected where ratio is high, but clipping alone does not explain the failures.
- Next priority: deterministic action extraction and actor exploration scale first, then actor loss/critic target stability and longer training.
