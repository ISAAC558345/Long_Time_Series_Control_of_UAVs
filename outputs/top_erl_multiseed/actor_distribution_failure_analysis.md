# Actor distribution failure analysis

This report analyzes saved checkpoints and evaluation logs only. It does not retrain or modify algorithms.

## Evaluation comparison

| seed | deterministic reward | stochastic reward | random reward | det-random | stoch-random | stoch-det | det FairIdx | stoch FairIdx | random FairIdx | det throughput | stoch throughput | random throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.306512 | 0.110492 | 0.083097 | 0.223416 | 0.027395 | -0.196020 | 0.689130 | 0.425157 | 0.341414 | 0.101755 | 0.064150 | 0.062968 |
| 1 | 0.000000 | 0.065299 | 0.085005 | -0.085005 | -0.019706 | 0.065299 | 0.250004 | 0.429317 | 0.442980 | 0.019724 | 0.054587 | 0.061619 |
| 2 | 0.019645 | 0.047300 | 0.041266 | -0.021621 | 0.006034 | 0.027655 | 0.250002 | 0.324077 | 0.326972 | 0.040025 | 0.048284 | 0.048535 |

## Actor output statistics

| seed | mean_abs_mean | max_abs_mean | mean_log_std | min_log_std | max_log_std | mean_std | max_std | det w norm | stoch w norm | det mean step | det max step | stoch mean step | stoch max step | near zero | tanh sat ratio | det clip ratio | finite |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|
| 0 | 0.196185 | 0.502906 | 0.061327 | -0.469632 | 0.420013 | 1.104934 | 1.521981 | 138.120834 | 352.700043 | 33.620693 | 46.475739 | 69.287804 | 100.000031 | False | 0.000000 | 0.000000 | True |
| 1 | 0.270800 | 0.607164 | 0.036019 | -0.210381 | 0.487790 | 1.060583 | 1.628713 | 148.888504 | 342.101166 | 34.772011 | 68.799614 | 65.377342 | 100.000031 | False | 0.000000 | 0.000000 | True |
| 2 | 0.395933 | 0.881265 | -0.098604 | -0.595652 | 0.507440 | 0.950569 | 1.661034 | 204.509903 | 365.093872 | 47.893295 | 95.120094 | 72.275017 | 100.000038 | False | 0.000000 | 0.000000 | True |

## Per-seed diagnosis

### Seed 0
- no obvious distribution pathology

### Seed 1
- stochastic policy is meaningfully better than deterministic mean
- deterministic actor is below random baseline

### Seed 2
- deterministic actor is below random baseline

## Short answers

- Seed 1: failure is more consistent with deterministic mean degradation than total stochastic policy failure; stochastic reward is nonzero and above deterministic, though still below random.
- Seed 2: failure is more consistent with insufficient/unstable learning than pure deterministic extraction; stochastic is better than deterministic and random in reward, while deterministic stays weak.
- log_std is not collapsed to an extremely small value in these checkpoints; exploration remains active.
- deterministic w is not near zero, so the actor is not simply stationary.
- deterministic clipping is not dominant; tanh saturation should be inspected where ratio is high, but clipping alone does not explain the failures.
- Next priority: deterministic action extraction and actor exploration scale first, then actor loss/critic target stability and longer training.
