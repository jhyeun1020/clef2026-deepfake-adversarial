# Phase 1: Pilot Report

DINOv2 layers: [4, 9, 14, 19, 23], feat_dim: 1024
Covariance: LedoitWolf (tied)

## AUROC

          layer           task   auroc
0             4    real_vs_adv  1.0000
1             4  real_vs_clean  0.9988
2             4    adv_vs_jpeg  0.4632
3             4   adv_vs_clean  1.0000
4             9    real_vs_adv  1.0000
5             9  real_vs_clean  1.0000
6             9    adv_vs_jpeg  0.9999
7             9   adv_vs_clean  1.0000
8            14    real_vs_adv  1.0000
9            14  real_vs_clean  1.0000
10           14    adv_vs_jpeg  0.1929
11           14   adv_vs_clean  1.0000
12           19    real_vs_adv  1.0000
13           19  real_vs_clean  1.0000
14           19    adv_vs_jpeg  1.0000
15           19   adv_vs_clean  1.0000
16           23    real_vs_adv  1.0000
17           23  real_vs_clean  0.9996
18           23    adv_vs_jpeg  0.9814
19           23   adv_vs_clean  0.9233
20  ensemble_LR    real_vs_adv  1.0000
21  ensemble_LR    adv_vs_jpeg  1.0000
22  ensemble_LR   adv_vs_clean  1.0000

## 판정: **✓ Mahalanobis 작동, Phase 2 진행**
- adv_vs_jpeg ensemble: 1.0000
- real_vs_adv ensemble: 1.0000
