# Mahalanobis Distribution Shift — Diagnosis Report

## 1. Distance 분포

      group     n    mean  median     std    p10     p25     p75     p90    min      max
 pilot_real   100  255.92  257.02   39.68 205.93  225.99  280.64  309.32 161.79   348.40
  pilot_adv   100  185.50  184.18   20.48 157.74  173.76  196.85  211.06 142.76   266.26
pilot_clean   100  214.61  214.40   23.46 184.66  198.60  229.53  245.37 162.60   279.74
 pilot_jpeg   100  240.77  237.30   39.42 189.41  215.13  268.00  290.32 155.44   335.44
   test_all 24404 1513.94 1298.38 1016.98 819.91 1048.95 1633.09 2193.86 359.26 12284.58
  test_ours  1002  686.05  663.02  155.86 516.49  572.54  769.05  883.83 397.80  1577.16
test_others 23402 1549.38 1318.31 1023.18 878.57 1079.71 1653.10 2226.50 359.26 12284.58

![distance](./distance_distribution_comparison.png)

## 2. Class별 distance

Our 1002 avg: {'avg_dist_real': np.float64(1693.6), 'avg_dist_adv': np.float64(1036.7), 'avg_dist_clean_fake': np.float64(1701.5), 'avg_dist_jpeg': np.float64(1316.5)}
Others avg: {'avg_dist_real': np.float64(2993.5), 'avg_dist_adv': np.float64(3446.9), 'avg_dist_clean_fake': np.float64(3126.5), 'avg_dist_jpeg': np.float64(3294.2)}
Our closest: {'adv': 1002}

## 3. t-SNE

![tsne](./feature_space_visualization.png)

## 4. RF probability

![rf](./rf_probability_distributions.png)

## 5. Covariance

source  layer   sinv_trace  sinv_cond  sinv_eig_min  sinv_eig_max  sinv_eig_median
 pilot      4 5.796108e+08   13361.71     53.922363     720495.19      720320.0000
 pilot      9 1.767452e+08    3354.32     67.326546     225835.11      225800.2500
 pilot     14 8.399051e+06    2269.15      4.842839      10989.15       10988.1758
 pilot     19 2.127221e+05    1630.90      0.174491        284.58         284.5600
 pilot     23 3.969650e+03     410.31      0.012871          5.28           5.2812
  full      4 4.346373e+09  261741.81     47.553104   12446636.00     3136840.5000
  full      9 1.127778e+09   50927.43     68.942192    3511048.50      713987.6250
  full     14 3.699256e+07   29664.74      5.190377     153971.20       21397.4648
  full     19 7.125282e+05   28673.34      0.148071       4245.68         373.1995
  full     23 1.056357e+04    8063.28      0.010319         83.20           5.3039

## 6. Fix attempts

| Fix | Det rate | Our recall | Notes |
|---|---|---|---|
| A (re-fit with Phase2 Gaussian) | N/A | N/A |  |
| B (z-score normalized) | 0.0% | 0.0% |  |
| C (distance threshold) | 0.0% | 0.0% |  |
| D (relative distance RF) | 0.2% | 4.4% | 1.0000 |
| E (simple inverted dist, t=0.5) | 100.0% | 100.0% |  |

## 7. 진단 결론

- Pilot mean distance: 224.2
- Test mean distance: 1513.9
- Ratio (test/pilot): **6.75x**

⚠ Test set distances가 pilot보다 훨씬 큼 → **covariance scale mismatch** 확정
Phase 2 full Gaussian이 더 spread-out한 covariance → distance 값 전체적으로 커짐
Pilot에서 학습한 RF의 decision boundary가 test에서 의미 없음

