# Critic target / loss diagnostics

This report reloads saved checkpoints and analyzes critic targets on fresh short replay buffers. It does not retrain or modify checkpoints.

## Summary

| run | final reward | final critic loss | final actor loss | Q-target corr | Q-target MSE | reward mean | target mean | Q mean | bootstrap/target | bootstrap/reward | candidate Q-reward corr | finite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| full_bootstrap__seed_0 | 0.07530579887153765 | 0.018063554540276527 | NA | -0.078522 | 0.048155 | 0.019442 | -0.143582 | -0.212448 | 1.321433 | 3.753961 | -0.05189171743373554 | True |
| no_bootstrap__seed_0 | 0.07531254734882045 | 0.016270099207758904 | NA | 0.154809 | 0.025252 | 0.017707 | 0.049688 | -0.003641 | 0.000000 | 0.000000 | -0.029193521855979582 | True |
| weak_bootstrap_025__seed_0 | 0.07530627184174038 | 0.01660638488829136 | NA | 0.086162 | 0.050760 | 0.029691 | 0.036802 | -0.003857 | 0.801022 | 0.614834 | 0.0010293951225895477 | True |

## Per-run interpretation

### full_bootstrap__seed_0
- scalar_reward_seq mean/std: 0.019441814235178753 / 0.029699461391776306
- n_step_targets mean/std: -0.1435817122948356 / 0.05947888258855983
- prefix_q_values mean/std: -0.21244836745550855 / 0.18361366841367344
- bootstrap_values mean/std: -0.2035899747116491 / 0.028543709834477224
- Q-target MSE mean: 0.048154539186507465
- Q-target correlation mean: -0.07852224349135693
- bootstrap/target abs ratio mean: 1.3214330916169927
- bootstrap/reward abs ratio mean: 3.7539614279078957
- Q-target correlation is weak; in-buffer critic fit is suspect.
- Bootstrap term dominates the target magnitude.
- Candidate Q-reward correlation is weak or negative.

### no_bootstrap__seed_0
- scalar_reward_seq mean/std: 0.01770657401997596 / 0.03016852593331739
- n_step_targets mean/std: 0.04968768500431906 / 0.05903955348598519
- prefix_q_values mean/std: -0.003641481667291373 / 0.1406883969493368
- bootstrap_values mean/std: -0.20264839415438474 / 0.02730566471075514
- Q-target MSE mean: 0.025252325385808946
- Q-target correlation mean: 0.15480923090895987
- bootstrap/target abs ratio mean: 0.0
- bootstrap/reward abs ratio mean: 0.0
- Q-target correlation is weak; in-buffer critic fit is suspect.
- Candidate Q-reward correlation is weak or negative.

### weak_bootstrap_025__seed_0
- scalar_reward_seq mean/std: 0.0296907173411455 / 0.060219754923692935
- n_step_targets mean/std: 0.03680150228785351 / 0.11408489284028077
- prefix_q_values mean/std: -0.003856799118220806 / 0.18080487052076297
- bootstrap_values mean/std: -0.2003559391433373 / 0.03254327650855627
- Q-target MSE mean: 0.05076016709208488
- Q-target correlation mean: 0.0861624644651585
- bootstrap/target abs ratio mean: 0.8010217292033601
- bootstrap/reward abs ratio mean: 0.6148338898631773
- Q-target correlation is weak; in-buffer critic fit is suspect.
- Bootstrap term dominates the target magnitude.
- Candidate Q-reward correlation is weak or negative.

## Diagnostic answers

- If Q-target correlation is weak, the critic is not reliably fitting its own N-step targets on fresh replay-buffer data.
- If bootstrap/reward ratio is large, the bootstrap term dominates target scale and may hurt calibration.
- If Q-target fit is weak and candidate Q-reward correlation is also weak, the issue is more likely target/loss/calibration than pure OOD candidate ranking.
- First next checks: reward / target normalization, critic-only supervised warmup, more critic updates, and Monte Carlo target smoke test without bootstrap.
- Delayed actor update remains useful, but this diagnostic focuses on whether critic targets themselves are learnable and calibrated.
- This is diagnostic only; it is not a final algorithm conclusion.
