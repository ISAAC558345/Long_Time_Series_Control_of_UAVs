# Critic calibration / Q-ranking analysis

This report evaluates saved critic checkpoints only. It does not retrain or modify the algorithm.

## Summary

| seed | q-reward corr | q-last reward corr | q-throughput corr | q-fairness corr | mean top1 regret | q std | reward std | finite |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.213076 | 0.221257 | 0.129527 | 0.153478 | 0.189424 | 0.161274 | 0.097435 | True |
| 1 | -0.024712 | -0.007291 | 0.022837 | -0.003272 | 0.237486 | 0.138211 | 0.093490 | True |
| 2 | 0.116083 | 0.129947 | 0.029289 | 0.027191 | 0.294066 | 0.245476 | 0.211068 | True |

## Per-seed diagnosis

### Seed 0
- Q-reward correlation is positive: 0.21307644351431587
- Q output is variable; predicted_q_std=0.1612742836877904
- mean top-1 regret: 0.18942399470525076
- Q-throughput correlation: 0.12952701849056777
- Q-fairness correlation: 0.15347830126855086
- Ranking has some positive signal, but should still be validated.

### Seed 1
- Q-reward correlation is weak: -0.024712457253913633
- Q output is variable; predicted_q_std=0.13821068019617008
- mean top-1 regret: 0.23748551628449222
- Q-throughput correlation: 0.022836759549768118
- Q-fairness correlation: -0.003271637032998886
- Ranking direction is likely wrong or noisy for reward.

### Seed 2
- Q-reward correlation is weak: 0.11608338138271379
- Q output is variable; predicted_q_std=0.2454764990770954
- mean top-1 regret: 0.29406613262681675
- Q-throughput correlation: 0.029289189099377975
- Q-fairness correlation: 0.02719135146896224
- Ranking has some positive signal, but should still be validated.

## Diagnostic answers

- The critic Q-ranking is not consistently trustworthy when Q-reward correlation is weak or negative.
- If Q outputs are variable but correlations are negative, the problem is more direction/ranking than constant-output collapse.
- Low or negative Q-throughput and Q-fairness correlations indicate the critic is not reliably predicting either communication objective.
- Next priority should be N-step target / critic loss and critic update schedule, including critic warmup or delayed actor update. Reward scaling is also worth checking before changing trajectory generator expressiveness.
- Deterministic action extraction remains relevant, but this diagnostic directly points to critic target / Q ranking as a bottleneck.
- This is not a final algorithm conclusion; it only diagnoses the current minimal implementation.
