# Target Ablation Summary

This report only reads existing CSV outputs. It does not retrain or modify checkpoints.

| target | mode | weight | det reward | stoch reward | random reward | det FairIdx | Q-reward corr | top-1 regret | Q-target corr | Q-target MSE | bootstrap/reward | finite |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| full_bootstrap | bootstrapped_n_step | 1.0 | 0.251771 | 0.134620 | 0.083097 | 0.557196 | -0.051892 | 0.216135 | -0.078522 | 0.048155 | 3.753961 | True |
| no_bootstrap | no_bootstrap | 0.0 | 0.253602 | 0.126018 | 0.083097 | 0.558028 | -0.029194 | 0.200228 | 0.154809 | 0.025252 | 0.000000 | True |
| weak_bootstrap_025 | weighted_bootstrap | 0.25 | 0.253203 | 0.126014 | 0.083097 | 0.557910 | 0.001029 | 0.215552 | 0.086162 | 0.050760 | 0.614834 | True |

## Readout

- Best Q-reward correlation: weak_bootstrap_025.
- Lowest top-1 regret: no_bootstrap.
- Best deterministic actor reward: no_bootstrap.
- Best deterministic FairIdx: no_bootstrap.
- All summarized metrics finite: True.

## Diagnostic interpretation

- This is a seed-0 ablation only; it is not a final algorithm conclusion.
- Prefer advancing a target mode to multi-seed only if it improves Q-ranking or actor evaluation without numerical instability.
