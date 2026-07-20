# Critic calibration / Q-ranking analysis

This report evaluates saved critic checkpoints only. It does not retrain or modify the algorithm.

## Summary

| seed | q-reward corr | q-last reward corr | q-throughput corr | q-fairness corr | mean top1 regret | q std | reward std | finite |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | -0.025077 | -0.025009 | 0.006851 | -0.020416 | 0.232555 | 0.228633 | 0.086405 | True |

## Per-seed diagnosis

### Seed 0
- Q-reward correlation is weak: -0.025077164808357647
- Q output is variable; predicted_q_std=0.22863258730655495
- mean top-1 regret: 0.23255482672317168
- Q-throughput correlation: 0.006850863502446713
- Q-fairness correlation: -0.02041617374582691
- Ranking direction is likely wrong or noisy for reward.

## Diagnostic answers

- The critic Q-ranking is not consistently trustworthy when Q-reward correlation is weak or negative.
- If Q outputs are variable but correlations are negative, the problem is more direction/ranking than constant-output collapse.
- Low or negative Q-throughput and Q-fairness correlations indicate the critic is not reliably predicting either communication objective.
- Next priority should be N-step target / critic loss and critic update schedule, including critic warmup or delayed actor update. Reward scaling is also worth checking before changing trajectory generator expressiveness.
- Deterministic action extraction remains relevant, but this diagnostic directly points to critic target / Q ranking as a bottleneck.
- This is not a final algorithm conclusion; it only diagnoses the current minimal implementation.
