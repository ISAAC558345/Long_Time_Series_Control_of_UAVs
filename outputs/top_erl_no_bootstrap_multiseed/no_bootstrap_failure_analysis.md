# No-Bootstrap Multi-Seed Failure Analysis

This report reads existing CSV diagnostics only. It does not retrain or modify algorithms.

## Multi-Seed Evaluation

| seed | det reward | stoch reward | random reward | det gap | stoch gap | det FairIdx | stoch FairIdx | random FairIdx | det Fair gap | stoch Fair gap | Q-reward corr | top-1 regret | finite |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.253602 | 0.126018 | 0.083097 | 0.170505 | 0.042922 | 0.558028 | 0.442755 | 0.341414 | 0.216614 | 0.101340 | -0.033061 | 0.221863 | True |
| 1 | 0.038322 | 0.103868 | 0.085005 | -0.046682 | 0.018863 | 0.500001 | 0.417442 | 0.442980 | 0.057021 | -0.025538 | -0.077254 | 0.231683 | True |
| 2 | 0.080657 | 0.105352 | 0.041266 | 0.039391 | 0.064086 | 0.250001 | 0.387143 | 0.326972 | -0.076971 | 0.060172 | 0.076742 | 0.277678 | True |

## Actor Distribution

| seed | mean_log_std | mean_std | det w norm | stoch w norm | det mean step | det max step | near zero | tanh sat ratio | det clip ratio | finite |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|
| 0 | 0.086060 | 1.116198 | 129.337448 | 347.120941 | 32.137840 | 48.453346 | False | 0.000000 | 0.000000 | True |
| 1 | 0.042667 | 1.059523 | 74.039154 | 351.096466 | 17.268253 | 35.490799 | False | 0.000000 | 0.000000 | True |
| 2 | -0.011136 | 0.992637 | 49.357922 | 326.256073 | 12.315550 | 20.189777 | False | 0.000000 | 0.000000 | True |

## Trajectory Diagnostics

| seed | policy | reward | throughput | FairIdx | path lengths | final displacement | mean step | max step | min inter-UAV dist | mean nearest GT | final nearest GT | boundary hit | finite |
|---:|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|:---:|
| 0 | deterministic_actor | 1.078043 | 0.310510 | 0.564923 | 280.49/441.73/246.66 | 280.48/441.41/246.27 | 28.05/44.17/24.67 | 30.82/48.45/29.20 | 260.76 | 178.98 | 166.93 | 0.000000 | True |
| 0 | random_baseline | 0.214166 | 0.163759 | 0.317212 | 616.75/908.77/638.00 | 363.29/226.14/345.09 | 61.67/90.88/63.80 | 100.00/100.00/99.45 | 16.25 | 227.24 | 301.29 | 0.121212 | True |
| 1 | deterministic_actor | 0.038322 | 0.039525 | 0.500004 | 297.46/59.56/184.05 | 297.30/57.62/183.94 | 29.75/5.96/18.40 | 35.49/7.94/24.81 | 270.78 | 286.50 | 345.05 | 0.000000 | True |
| 1 | random_baseline | 0.108216 | 0.101492 | 0.478673 | 766.51/860.92/519.07 | 451.56/299.39/117.04 | 76.65/86.09/51.91 | 90.12/100.00/75.71 | 239.10 | 204.88 | 161.93 | 0.000000 | True |
| 2 | deterministic_actor | 0.181895 | 0.207691 | 0.250001 | 199.09/130.79/58.82 | 199.09/130.79/58.80 | 19.91/13.08/5.88 | 20.00/13.15/5.96 | 482.78 | 191.45 | 178.20 | 0.000000 | True |
| 2 | random_baseline | 0.000000 | 0.019724 | 0.250008 | 494.09/564.13/493.64 | 379.41/23.47/196.46 | 49.41/56.41/49.36 | 62.43/70.71/71.09 | 509.90 | 336.59 | 357.20 | 0.106061 | True |

## Diagnostic Answers

- No-bootstrap mean deterministic reward gap: 0.054405.
- No-bootstrap mean stochastic reward gap: 0.041957.
- No-bootstrap mean deterministic FairIdx gap: 0.065554.
- No-bootstrap mean stochastic FairIdx gap: 0.045325.
- Previous full-bootstrap mean deterministic reward gap: 0.038930.
- Previous full-bootstrap deterministic-below-random seeds: [1, 2].
- No-bootstrap deterministic-below-random seeds: [1].
- No-bootstrap stochastic-below-random seeds: [].
- All inspected metrics finite: True.

### Seed 1

- Deterministic reward gap is -0.046682, while stochastic reward gap is 0.018863. This points to deterministic mean extraction degradation more than a fully failed stochastic policy.
- Seed 1 deterministic mean step is 17.268253, with det w norm 74.039154; it is not near zero, but is much more conservative than stochastic w norm 351.096466.
- Seed 1 deterministic final nearest-GT distance is 345.05, versus random baseline 161.93; the deterministic trajectory ends farther from GTs.
- Seed 1 deterministic boundary hit ratio is 0.000000, min inter-UAV distance is 270.78; no boundary sticking or UAV overlap is indicated.
- Seed 1 Q-reward correlation is -0.077254; Q-ranking remains weak.

## Conclusion

- No-bootstrap is more stable than the previous full-bootstrap sanity run, but seed 1 still has deterministic actor failure.
- Stochastic actor is more reliable in this no-bootstrap run: it is above random on all three seeds.
- Deterministic actor should not be the only main evaluation strategy yet; report deterministic and stochastic side by side.
- Q-ranking remains a bottleneck because correlations are weak and top-1 regret remains nontrivial.
- Next checks worth prioritizing: continue with no_bootstrap as the target mode, run longer 200/500-iteration sanity, inspect deterministic extraction, and keep improving critic ranking/calibration.
- This is diagnostic only and not a final algorithm conclusion.
