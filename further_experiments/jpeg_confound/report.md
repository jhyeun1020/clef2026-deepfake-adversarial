# JPEG Confound Resolution Experiments

## Experiment 1: Layer Ratio (Median 3x3)
Best metrics for Adv vs JPEG Q80:
- ratio_L19_L14: AUROC=0.7269
- ratio_L23_L19: AUROC=0.5900
- ratio_L19_L9: AUROC=0.8150
- cross_layer_var: AUROC=0.6719
- delta_L14: AUROC=0.5631
- delta_L19: AUROC=0.7104
- delta_L23: AUROC=0.7274

## Experiment 2: Multi-Purification (L19, Adv vs JPEG Q80)
- median_3: AUROC=0.7104
- median_5: AUROC=0.7598
- gaussian_1: AUROC=0.8745
- gaussian_2: AUROC=0.6513
- bit_depth_4: AUROC=0.9289
- bilateral: AUROC=0.7909

## Experiment 3: Cosine (Median 3x3, Adv vs JPEG Q80)
- cos_L4: AUROC=0.7647
- cos_L9: AUROC=0.7074
- cos_L14: AUROC=0.5287
- cos_L19: AUROC=0.7275
- cos_L23: AUROC=0.7357

## Experiment 4: Combined LR
- Adv vs JPEG Q80: **0.9980 ± 0.0040**
- Adv vs All: **0.9322 ± 0.0635**

## Feature Importance
| feature               |        coef |
|:----------------------|------------:|
| delta_L19_median_5    | -2.00989    |
| delta_L19_bit_depth_4 | -1.95634    |
| delta_L19_gaussian_1  |  1.41055    |
| delta_logit           |  1.20987    |
| ratio_L19_L9          |  0.445307   |
| delta_L19_bilateral   | -0.435233   |
| delta_L19             |  0.300491   |
| ratio_L19_L14         |  0.26318    |
| delta_L23             |  0.0891394  |
| cos_L23               |  0.0294646  |
| delta_L14             | -0.017769   |
| cos_L19               |  0.00808379 |
| cos_L14               |  0.00352968 |
| cross_layer_var       | -0.00205433 |
