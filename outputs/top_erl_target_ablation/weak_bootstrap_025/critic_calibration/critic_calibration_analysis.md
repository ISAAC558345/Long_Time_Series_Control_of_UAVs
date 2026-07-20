# Critic calibration / Q-ranking analysis

This report evaluates saved critic checkpoints only. It does not retrain or modify the algorithm.

## Summary

| seed | q-reward corr | q-last reward corr | q-throughput corr | q-fairness corr | mean top1 regret | q std | reward std | finite |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.001029 | -0.007008 | 0.037279 | 0.009963 | 0.215552 | 0.215877 | 0.161683 | True |

## Per-seed diagnosis

### Seed 0
- Q-reward correlation is weak: 0.0010293951225895477
- Q output is variable; predicted_q_std=0.21587723112972695
- mean top-1 regret: 0.2155522396252607
- Q-throughput correlation: 0.037279141581508704
- Q-fairness correlation: 0.009962632992689421
- Ranking is weak; top candidate should not be trusted yet.

## Diagnostic answers

- The critic Q-ranking is not consistently trustworthy when Q-reward correlation is weak or negative.
- If Q outputs are variable but correlations are negative, the problem is more direction/ranking than constant-output collapse.
- Low or negative Q-throughput and Q-fairness correlations indicate the critic is not reliably predicting either communication objective.
- Next priority should be N-step target / critic loss and critic update schedule, including critic warmup or delayed actor update. Reward scaling is also worth checking before changing trajectory generator expressiveness.
- Deterministic action extraction remains relevant, but this diagnostic directly points to critic target / Q ranking as a bottleneck.
- This is not a final algorithm conclusion; it only diagnoses the current minimal implementation.
