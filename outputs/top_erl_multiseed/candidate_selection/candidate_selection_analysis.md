# Critic-guided candidate selection analysis

This report evaluates saved checkpoints only. It does not retrain or modify the algorithm.

## Summary table

| seed | policy | mean reward | mean throughput | mean FairIdx | mean collision | finite |
|---:|---|---:|---:|---:|---:|:---:|
| 0 | deterministic_actor | 0.306512 | 0.101755 | 0.689130 | 0.000000 | True |
| 0 | stochastic_actor_single | 0.056843 | 0.049584 | 0.370986 | 0.000000 | True |
| 0 | critic_selected_actor_topk | 0.090602 | 0.058912 | 0.430897 | 0.000000 | True |
| 0 | random_baseline | 0.083097 | 0.062968 | 0.341414 | 0.000000 | True |
| 1 | deterministic_actor | 0.000000 | 0.019724 | 0.250004 | 0.000000 | True |
| 1 | stochastic_actor_single | 0.088553 | 0.056770 | 0.397769 | 0.000000 | True |
| 1 | critic_selected_actor_topk | 0.024004 | 0.038132 | 0.294701 | 0.000000 | True |
| 1 | random_baseline | 0.085005 | 0.061619 | 0.442980 | 0.000000 | True |
| 2 | deterministic_actor | 0.019645 | 0.040025 | 0.250002 | 0.000000 | True |
| 2 | stochastic_actor_single | 0.032322 | 0.044399 | 0.296192 | 0.000000 | True |
| 2 | critic_selected_actor_topk | 0.049327 | 0.054522 | 0.347290 | 0.000000 | True |
| 2 | random_baseline | 0.041266 | 0.048535 | 0.326972 | 0.000000 | True |

## Candidate score-reward correlation

- seed 0: 0.04889934608822303
- seed 1: -0.1193668047376683
- seed 2: -0.044987231729478605

## Per-seed comparisons

### Seed 0
- top-k minus deterministic reward: -0.215910
- top-k minus stochastic reward: 0.033759
- top-k minus random reward: 0.007506
- critic score / reward correlation: 0.04889934608822303

### Seed 1
- top-k minus deterministic reward: 0.024004
- top-k minus stochastic reward: -0.064550
- top-k minus random reward: -0.061001
- critic score / reward correlation: -0.1193668047376683

### Seed 2
- top-k minus deterministic reward: 0.029682
- top-k minus stochastic reward: 0.017004
- top-k minus random reward: 0.008060
- critic score / reward correlation: -0.044987231729478605

## Diagnostic answers

- If top-k improves over deterministic but not over stochastic/random, deterministic extraction is suspect while critic ranking remains limited.
- If top-k underperforms all baselines or has weak score-reward correlation, critic target / Q ranking should be inspected.
- Positive correlation is useful but not sufficient: the top candidate can still be poor if the actor candidate pool is weak.
- This is not a final algorithm conclusion; it only diagnoses the current minimal implementation.
