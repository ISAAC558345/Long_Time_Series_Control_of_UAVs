# Critic calibration / Q-ranking analysis

This report evaluates saved critic checkpoints only. It does not retrain or modify the algorithm.

## Summary

| seed | q-reward corr | q-last reward corr | q-throughput corr | q-fairness corr | mean top1 regret | q std | reward std | finite |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | -0.033061 | -0.031276 | 0.001182 | -0.018004 | 0.221863 | 0.219456 | 0.089481 | True |
| 1 | -0.077254 | -0.076029 | -0.059799 | 0.018622 | 0.231683 | 0.343403 | 0.091426 | True |
| 2 | 0.076742 | 0.089647 | 0.046291 | 0.022033 | 0.277678 | 0.375097 | 0.249502 | True |

## Per-seed diagnosis

### Seed 0
- Q-reward correlation is weak: -0.03306080222286047
- Q output is variable; predicted_q_std=0.21945573867382437
- mean top-1 regret: 0.22186306656272864
- Q-throughput correlation: 0.001181997126953319
- Q-fairness correlation: -0.018003806834682414
- Ranking direction is likely wrong or noisy for reward.

### Seed 1
- Q-reward correlation is negative: -0.07725384330197717
- Q output is variable; predicted_q_std=0.34340257352058806
- mean top-1 regret: 0.23168289219605773
- Q-throughput correlation: -0.05979886250972285
- Q-fairness correlation: 0.01862244419649866
- Ranking direction is likely wrong or noisy for reward.

### Seed 2
- Q-reward correlation is weak: 0.07674204194854399
- Q output is variable; predicted_q_std=0.37509695136755994
- mean top-1 regret: 0.2776776605954356
- Q-throughput correlation: 0.046291482871549695
- Q-fairness correlation: 0.022033134201707653
- Ranking is weak; top candidate should not be trusted yet.

## Diagnostic answers

- The critic Q-ranking is not consistently trustworthy when Q-reward correlation is weak or negative.
- If Q outputs are variable but correlations are negative, the problem is more direction/ranking than constant-output collapse.
- Low or negative Q-throughput and Q-fairness correlations indicate the critic is not reliably predicting either communication objective.
- Next priority should be N-step target / critic loss and critic update schedule, including critic warmup or delayed actor update. Reward scaling is also worth checking before changing trajectory generator expressiveness.
- Deterministic action extraction remains relevant, but this diagnostic directly points to critic target / Q ranking as a bottleneck.
- This is not a final algorithm conclusion; it only diagnoses the current minimal implementation.
