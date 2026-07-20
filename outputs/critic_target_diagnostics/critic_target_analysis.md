# Critic target / loss diagnostics

This report reloads saved checkpoints and analyzes critic targets on fresh short replay buffers. It does not retrain or modify checkpoints.

## Summary

| run | final reward | final critic loss | final actor loss | Q-target corr | Q-target MSE | reward mean | target mean | Q mean | bootstrap/target | bootstrap/reward | candidate Q-reward corr | finite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| top_erl_critic_warmup_multiseed__seed_0 | 0.07530579887153765 | 0.018063554540276527 | NA | 0.234548 | 0.022418 | 0.033415 | -0.104345 | -0.140672 | 1.397412 | 2.224208 | -0.025077164808357647 | True |
| top_erl_multiseed__seed_0 | 0.5107350553621475 | 0.919681966304779 | -0.6689963936805725 | 0.107176 | 1.792593 | 0.021741 | -0.075034 | 0.691806 | 1.332952 | 2.339815 | -0.044795939230736256 | True |
| top_erl_multiseed__seed_1 | 0.14926655354470508 | 0.4954429864883423 | 0.2067033350467682 | -0.316373 | 1.619153 | 0.024561 | 0.159634 | -0.629789 | 0.566389 | 1.351201 | -0.1193668047376683 | True |
| top_erl_multiseed__seed_2 | 0.019269835057636794 | 1.374298095703125 | 0.5715623497962952 | 0.043730 | 0.714993 | 0.008709 | 0.219519 | -0.096936 | 0.874225 | 7.221265 | -0.044987231729478605 | True |

## Per-run interpretation

### top_erl_critic_warmup_multiseed__seed_0
- scalar_reward_seq mean/std: 0.03341496137552895 / 0.05825733697099409
- n_step_targets mean/std: -0.1043450113129802 / 0.11171922107763727
- prefix_q_values mean/std: -0.14067215144168585 / 0.09718185440428993
- bootstrap_values mean/std: -0.20585724564269184 / 0.02670423928514281
- Q-target MSE mean: 0.02241789722815156
- Q-target correlation mean: 0.23454770806813222
- bootstrap/target abs ratio mean: 1.3974122640089288
- bootstrap/reward abs ratio mean: 2.2242080390352243
- Q-target correlation is not obviously weak.
- Bootstrap term dominates the target magnitude.
- In-buffer fit is acceptable but candidate ranking looks OOD.

### top_erl_multiseed__seed_0
- scalar_reward_seq mean/std: 0.021741386885405518 / 0.042422297623607295
- n_step_targets mean/std: -0.07503410429460927 / 0.082534783226931
- prefix_q_values mean/std: 0.6918064135499299 / 1.038611198911021
- bootstrap_values mean/std: -0.14268008599290624 / 0.033266232136822116
- Q-target MSE mean: 1.7925926733016968
- Q-target correlation mean: 0.10717604670593758
- bootstrap/target abs ratio mean: 1.3329522841685844
- bootstrap/reward abs ratio mean: 2.339814870990065
- Q-target correlation is weak; in-buffer critic fit is suspect.
- Bootstrap term dominates the target magnitude.
- Candidate Q-reward correlation is weak or negative.

### top_erl_multiseed__seed_1
- scalar_reward_seq mean/std: 0.02456096928741317 / 0.048880709479625126
- n_step_targets mean/std: 0.15963397174025887 / 0.10136939336702816
- prefix_q_values mean/std: -0.6297888567997143 / 0.9169803640330575
- bootstrap_values mean/std: 0.09196451750583946 / 0.039466363173811636
- Q-target MSE mean: 1.619153039455414
- Q-target correlation mean: -0.31637341131049224
- bootstrap/target abs ratio mean: 0.5663886864240696
- bootstrap/reward abs ratio mean: 1.3512007711044598
- Q-target correlation is weak; in-buffer critic fit is suspect.
- Candidate Q-reward correlation is weak or negative.

### top_erl_multiseed__seed_2
- scalar_reward_seq mean/std: 0.008709035339998082 / 0.013123345958291599
- n_step_targets mean/std: 0.21951857171021402 / 0.037816531201811145
- prefix_q_values mean/std: -0.09693629995570519 / 0.7484532706903237
- bootstrap_values mean/std: 0.19670728982891889 / 0.0400755667536304
- Q-target MSE mean: 0.7149934804439545
- Q-target correlation mean: 0.04372980224348771
- bootstrap/target abs ratio mean: 0.8742249681482054
- bootstrap/reward abs ratio mean: 7.221265300986271
- Q-target correlation is weak; in-buffer critic fit is suspect.
- Bootstrap term is much larger than discounted reward prefix.
- Bootstrap term dominates the target magnitude.
- Candidate Q-reward correlation is weak or negative.

## Diagnostic answers

- If Q-target correlation is weak, the critic is not reliably fitting its own N-step targets on fresh replay-buffer data.
- If bootstrap/reward ratio is large, the bootstrap term dominates target scale and may hurt calibration.
- If Q-target fit is weak and candidate Q-reward correlation is also weak, the issue is more likely target/loss/calibration than pure OOD candidate ranking.
- First next checks: reward / target normalization, critic-only supervised warmup, more critic updates, and Monte Carlo target smoke test without bootstrap.
- Delayed actor update remains useful, but this diagnostic focuses on whether critic targets themselves are learnable and calibrated.
- This is diagnostic only; it is not a final algorithm conclusion.
