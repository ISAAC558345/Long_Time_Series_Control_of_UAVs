# Multi-seed failure analysis

This report is read-only analysis of existing outputs. It does not modify training code or rerun training.

## Multi-seed summary

| seed | final train reward | det reward | random reward | reward gap | throughput gap | fairness gap | final critic loss | final actor loss | finite |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.510735 | 0.306512 | 0.083097 | 0.223416 | 0.038788 | 0.347715 | 0.919682 | -0.668996 | True |
| 1 | 0.149267 | 0.000000 | 0.085005 | -0.085005 | -0.041894 | -0.192976 | 0.495443 | 0.206703 | True |
| 2 | 0.019270 | 0.019645 | 0.041266 | -0.021621 | -0.008511 | -0.076970 | 1.374298 | 0.571562 | True |

## Aggregate gaps

- mean reward gap: 0.038930
- mean throughput gap: -0.003872
- mean fairness gap: 0.025923
- all finite: True

## Per-seed training curve summary

### Seed 0

- final rollout_cumulative_reward: 0.510735
- mean rollout_cumulative_reward: 0.066807
- last10 mean reward: 0.104622
- max reward: 0.510735
- final TotalThroughput: 0.122821
- last10 mean TotalThroughput: 0.060837
- final FairIdx: 0.635901
- last10 mean FairIdx: 0.446436
- max ProbCollision: 0.000000
- final critic_loss: 0.919682
- max critic_loss: 279.570862
- final actor_loss: -0.668996
- actor_loss range: 12.081292
- replay_buffer_size: 50.000000

Failure indications:
- no obvious trajectory anomaly; seed looks stable under current diagnostics

Priority check: no failure priority assigned; this seed is the positive sanity case.

Deterministic actor diagnostics:
- cumulative_reward: 0.4453250274685042
- TotalThroughput: 0.1641453504562378
- FairIdx: 0.5217428803443909
- ProbCollision: 0.0
- min_inter_uav_distance: 258.19234882365777
- mean_nearest_gt_distance: 222.89585837289215
- final_nearest_gt_distance: 296.4390731948659
- boundary_hit_ratio: 0.0
- trajectory_finite: True
- UAV 0: path_length=372.62609569709474, final_displacement=372.32973254504816, mean_step_length=37.262609569709475, max_step_length=43.48164750675995
- UAV 1: path_length=188.2908327495826, final_displacement=187.94428072096386, mean_step_length=18.829083274958258, max_step_length=28.19440516629301
- UAV 2: path_length=393.9668381634396, final_displacement=393.81051555375666, mean_step_length=39.39668381634396, max_step_length=46.475740755429335
- flags: stationary=False, weak_motion=False, boundary_stuck=False, boundary_touched=False, uav_overlap=False, uav_too_close=False, abnormal_jump=False, far_from_gt=False, low_fairness=False, low_throughput=False, finite=True

Random baseline diagnostics:
- cumulative_reward: 0.21416580361263682
- TotalThroughput: 0.1637585163116455
- FairIdx: 0.31721174716949463
- ProbCollision: 0.0
- min_inter_uav_distance: 16.254029802639163
- mean_nearest_gt_distance: 227.23551185976453
- final_nearest_gt_distance: 301.28691530516903
- boundary_hit_ratio: 0.12121212121212122
- trajectory_finite: True
- UAV 0: path_length=616.7482568905898, final_displacement=363.28717544947375, mean_step_length=61.67482568905898, max_step_length=100.00001148995412
- UAV 1: path_length=908.7703852121108, final_displacement=226.13557988041333, mean_step_length=90.87703852121108, max_step_length=99.99999929487305
- UAV 2: path_length=638.0004745077915, final_displacement=345.0903498744758, mean_step_length=63.800047450779154, max_step_length=99.45233154296875
- flags: stationary=False, weak_motion=False, boundary_stuck=False, boundary_touched=True, uav_overlap=False, uav_too_close=True, abnormal_jump=False, far_from_gt=False, low_fairness=False, low_throughput=False, finite=True

### Seed 1

- final rollout_cumulative_reward: 0.149267
- mean rollout_cumulative_reward: 0.076428
- last10 mean reward: 0.143252
- max reward: 0.429369
- final TotalThroughput: 0.080965
- last10 mean TotalThroughput: 0.072796
- final FairIdx: 0.408208
- last10 mean FairIdx: 0.490647
- max ProbCollision: 0.000000
- final critic_loss: 0.495443
- max critic_loss: 109.285095
- final actor_loss: 0.206703
- actor_loss range: 15.612020
- replay_buffer_size: 50.000000

Failure indications:
- deterministic actor reward is below random baseline
- throughput is below random baseline
- fairness is below random baseline
- deterministic trajectory has very low FairIdx
- deterministic trajectory has low throughput
- at least one UAV has weak final displacement
- boundary is touched but not dominant

Priority check: actor exploration scale / deterministic-mean extraction is the first module to inspect, because training obtains nonzero stochastic rollout reward but deterministic evaluation collapses to zero reward.

Deterministic actor diagnostics:
- cumulative_reward: 0.0
- TotalThroughput: 0.019724423065781593
- FairIdx: 0.25000834465026855
- ProbCollision: 0.0
- min_inter_uav_distance: 509.9019513592785
- mean_nearest_gt_distance: 267.1812036953077
- final_nearest_gt_distance: 274.52510851275724
- boundary_hit_ratio: 0.045454545454545456
- trajectory_finite: True
- UAV 0: path_length=375.28669297172553, final_displacement=356.8728257459385, mean_step_length=37.528669297172556, max_step_length=50.26284110278763
- UAV 1: path_length=34.62394626567395, final_displacement=29.414303591017685, mean_step_length=3.462394626567395, max_step_length=3.7985886815812715
- UAV 2: path_length=592.3937501426378, final_displacement=592.3286732733535, mean_step_length=59.239375014263786, max_step_length=68.79961818539242
- flags: stationary=False, weak_motion=True, boundary_stuck=False, boundary_touched=True, uav_overlap=False, uav_too_close=False, abnormal_jump=False, far_from_gt=False, low_fairness=True, low_throughput=True, finite=True

Random baseline diagnostics:
- cumulative_reward: 0.10821605304669576
- TotalThroughput: 0.10149166733026505
- FairIdx: 0.47867336869239807
- ProbCollision: 0.0
- min_inter_uav_distance: 239.0981961758541
- mean_nearest_gt_distance: 204.88071585298616
- final_nearest_gt_distance: 161.92816965684207
- boundary_hit_ratio: 0.0
- trajectory_finite: True
- UAV 0: path_length=766.5093368040222, final_displacement=451.5619404193891, mean_step_length=76.65093368040222, max_step_length=90.12379249444393
- UAV 1: path_length=860.9191278166629, final_displacement=299.39242439730725, mean_step_length=86.0919127816663, max_step_length=100.0000115666951
- UAV 2: path_length=519.0719518200017, final_displacement=117.0363007088577, mean_step_length=51.907195182000166, max_step_length=75.71329422470092
- flags: stationary=False, weak_motion=False, boundary_stuck=False, boundary_touched=False, uav_overlap=False, uav_too_close=False, abnormal_jump=False, far_from_gt=False, low_fairness=False, low_throughput=False, finite=True

### Seed 2

- final rollout_cumulative_reward: 0.019270
- mean rollout_cumulative_reward: 0.096238
- last10 mean reward: 0.029476
- max reward: 0.641223
- final TotalThroughput: 0.039637
- last10 mean TotalThroughput: 0.046478
- final FairIdx: 0.250002
- last10 mean FairIdx: 0.269282
- max ProbCollision: 0.000000
- final critic_loss: 1.374298
- max critic_loss: 607.097290
- final actor_loss: 0.571562
- actor_loss range: 27.990459
- replay_buffer_size: 50.000000

Failure indications:
- deterministic actor reward is below random baseline
- throughput is below random baseline
- fairness is below random baseline
- deterministic trajectory has very low FairIdx
- deterministic trajectory has low throughput
- UAVs remain far from GTs
- boundary is touched but not dominant

Priority check: actor exploration scale and training iterations are the first modules to inspect, with critic target stability second, because training and deterministic evaluation both remain low while trajectories move but do not reach useful GT geometry.

Deterministic actor diagnostics:
- cumulative_reward: 0.019644702011630388
- TotalThroughput: 0.04002481326460838
- FairIdx: 0.2500041723251343
- ProbCollision: 0.0
- min_inter_uav_distance: 111.43257572337413
- mean_nearest_gt_distance: 351.8693079310403
- final_nearest_gt_distance: 405.85095272317443
- boundary_hit_ratio: 0.10606060606060606
- trajectory_finite: True
- UAV 0: path_length=848.5212440071523, final_displacement=847.1502628718571, mean_step_length=84.85212440071523, max_step_length=90.90734211590993
- UAV 1: path_length=418.91348055392365, final_displacement=366.7512037659595, mean_step_length=41.89134805539236, max_step_length=58.78208909927208
- UAV 2: path_length=169.56098160654685, final_displacement=164.7498474446223, mean_step_length=16.956098160654683, max_step_length=17.71488519030753
- flags: stationary=False, weak_motion=False, boundary_stuck=False, boundary_touched=True, uav_overlap=False, uav_too_close=False, abnormal_jump=False, far_from_gt=True, low_fairness=True, low_throughput=True, finite=True

Random baseline diagnostics:
- cumulative_reward: 0.0
- TotalThroughput: 0.019724423065781593
- FairIdx: 0.25000834465026855
- ProbCollision: 0.0
- min_inter_uav_distance: 509.9019513592785
- mean_nearest_gt_distance: 336.5930127851857
- final_nearest_gt_distance: 357.20237564405267
- boundary_hit_ratio: 0.10606060606060606
- trajectory_finite: True
- UAV 0: path_length=494.09208638424286, final_displacement=379.41336445401805, mean_step_length=49.40920863842429, max_step_length=62.429047307074335
- UAV 1: path_length=564.1296732561989, final_displacement=23.47005619242207, mean_step_length=56.41296732561989, max_step_length=70.71067811865476
- UAV 2: path_length=493.6437507978885, final_displacement=196.46221712991135, mean_step_length=49.36437507978885, max_step_length=71.0868293936001
- flags: stationary=False, weak_motion=True, boundary_stuck=False, boundary_touched=True, uav_overlap=False, uav_too_close=False, abnormal_jump=False, far_from_gt=True, low_fairness=True, low_throughput=True, finite=True

## Short conclusion

- Seed 1 most likely fails at the actor policy output / deterministic mean level: deterministic evaluation collapses to zero reward and very low throughput/fairness, while training still saw nonzero stochastic rollout reward.
- Seed 2 most likely fails because the learned deterministic trajectory remains in poor service geometry: reward, throughput, and fairness stay low, UAVs move but remain relatively far from GTs and touch boundaries.
- The first module to inspect is actor exploration scale and deterministic policy extraction; the second is critic target stability / limited training horizon. The current diagnostics do not primarily point to trajectory generator expressiveness or UAV collision.
- This is not a final algorithm conclusion; it is a 3-seed sanity diagnosis.
