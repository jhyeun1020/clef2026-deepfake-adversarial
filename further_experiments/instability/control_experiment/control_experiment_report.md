# Augmented Real — Control Experiment Report

> 9 augmentations × 20 real images + 15 baselines (suspect/adv/real)
> 8 detectors × 4 variants, 24-dim instability features

---

## 1. Cosine Similarity (mean across pairs)

source                 |   →suspect |       →adv |      →real
------------------------------------------------------------
original               |      0.699 |      0.636 |      0.462
gaussian_noise_002     |      0.657 |      0.648 |      0.502
gaussian_noise_005     |      0.449 |      0.388 |      0.300
gaussian_blur_3        |      0.470 |      0.430 |      0.427
gaussian_blur_5        |      0.446 |      0.395 |      0.451
jpeg_q50               |      0.684 |      0.653 |      0.398
jpeg_q70               |      0.754 |      0.716 |      0.419
rotation_15            |      0.524 |      0.486 |      0.491
poisson_10             |      0.503 |      0.552 |      0.448
rotation_20            |      0.543 |      0.506 |      0.503
suspect                |      0.848 |      0.820 |      0.489
adv_baseline           |      0.820 |      0.848 |      0.476
real_baseline          |      0.489 |      0.476 |      0.593

## 2. Augmentation별 판정

| aug_type | cos(→suspect) | cos(→adv) | verdict |
|---|---|---|---|
| original | 0.901 | 0.829 | original is SIMILAR to suspect/adversarial (cos=0.901) |
| gaussian_noise_002 | 0.890 | 0.873 | gaussian_noise_002 is PARTIALLY similar (cos=0.890) — grey area |
| gaussian_noise_005 | 0.682 | 0.614 | gaussian_noise_005 is DISTINCT from suspect (cos=0.682) — confirms adversarial-specific |
| gaussian_blur_3 | 0.765 | 0.724 | gaussian_blur_3 is PARTIALLY similar (cos=0.765) — grey area |
| gaussian_blur_5 | 0.674 | 0.616 | gaussian_blur_5 is DISTINCT from suspect (cos=0.674) — confirms adversarial-specific |
| jpeg_q50 | 0.921 | 0.891 | jpeg_q50 is SIMILAR to suspect/adversarial (cos=0.921) |
| jpeg_q70 | 0.964 | 0.914 | jpeg_q70 is SIMILAR to suspect/adversarial (cos=0.964) |
| rotation_15 | 0.817 | 0.769 | rotation_15 is PARTIALLY similar (cos=0.817) — grey area |
| poisson_10 | 0.642 | 0.703 | poisson_10 is DISTINCT from suspect (cos=0.642) — confirms adversarial-specific |
| rotation_20 | 0.826 | 0.772 | rotation_20 is PARTIALLY similar (cos=0.826) — grey area |

## 3. 종합 판정: **Case B**

Detector also responds to: ['jpeg_q50', 'jpeg_q70']. Not purely adversarial-specific.

### suspect와 가장 유사한 augmentation (상위 3)

- **jpeg_q70**: cos=0.964
- **jpeg_q50**: cos=0.921
- **gaussian_noise_002**: cos=0.890

### suspect와 가장 다른 augmentation (상위 3)

- **gaussian_noise_005**: cos=0.682
- **gaussian_blur_5**: cos=0.674
- **poisson_10**: cos=0.642

## 4. t-SNE 해석

![t-SNE](./tsne_instability.png)

위 t-SNE plot에서:
- ★ suspect (빨강)와 ★ adv_baseline (주황)이 같은 cluster에 위치하는지
- ○ augmented real이 별도 cluster인지
- ★ real_baseline (초록)이 어디에 위치하는지
를 확인하면 instability 신호의 구별 능력을 시각적으로 판단 가능.

## 5. 출력 파일

- [augmented_features.parquet](./augmented_features.parquet)
- [similarity_matrix.csv](./similarity_matrix.csv)
- [similarity_heatmap.png](./similarity_heatmap.png)
- [extended_instability_heatmap.png](./extended_instability_heatmap.png)
- [tsne_instability.png](./tsne_instability.png)
- [verdict.md](./verdict.md)
- [sample_augmented_images.png](./sample_augmented_images.png)
