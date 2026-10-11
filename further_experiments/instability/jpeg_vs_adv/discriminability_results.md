# Adversarial vs JPEG-only Real Discriminability

> 400 JPEG-only real (100 imgs × Q50/Q60/Q70/Q80) vs 400 adversarial fake
> 24-dim instability features, 5-fold stratified CV

## Classification Results

| Model | AUROC | Accuracy |
|---|---|---|
| logistic_regression | 0.9155 | 0.8325 |
| random_forest | 0.9522 | 0.8912 |

## 판정: **✓ DISCRIMINABLE — JPEG-balanced training will work**

## Feature Importance (RF, JPEG-only vs Adv)

| rank | feature | importance |
|---|---|---|
| 1 | xception__random_crop_192_a | 0.1232 |
| 2 | xception__center_crop_192 | 0.1228 |
| 3 | siglip_dino__face_crop_variable | 0.0786 |
| 4 | siglip_dino__random_crop_192_a | 0.0706 |
| 5 | xception__face_crop_variable | 0.0681 |

### 비교: real(원본) vs adv RF top-5

| rank | feature | importance |
|---|---|---|
| 1 | aide_genimage__center_crop_192 | 0.1234 |
| 2 | xception__random_crop_192_a | 0.0953 |
| 3 | aide_genimage__random_crop_192_a | 0.0889 |
| 4 | xception__center_crop_192 | 0.0774 |
| 5 | distildire__face_crop_variable | 0.0765 |

Top-5 겹침: **2/5**

## 해석

Adversarial fake와 JPEG-only real은 instability feature space에서 **구분 가능**.
JPEG-balanced training으로 false positive를 줄일 수 있음.

## 출력 파일

- [jpeg_only_real_features.parquet](./jpeg_only_real_features.parquet)
- [adv_features_sample.parquet](./adv_features_sample.parquet)
- [tsne_jpeg_vs_adv.png](./tsne_jpeg_vs_adv.png)
