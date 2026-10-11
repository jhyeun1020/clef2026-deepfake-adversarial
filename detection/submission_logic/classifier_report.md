# Classifier-based Submission Report

## Training Data
- Real holdout: 485
- Fake holdout: 700
- Adversarial: 100
- Total: 1285

## CV Results

| Model | AUROC | Accuracy |
|---|---|---|
| RF | 0.9249±0.0926 | 0.8918±0.1171 |
| GB | 0.9256±0.0917 | 0.8872±0.1198 |
| LR | 0.9192±0.1012 | 0.8802±0.1354 |

## Best Model: **GB**
- Optimal threshold: **0.236**
- Per-fold thresholds: [0.2, 0.2, 0.35000000000000014, 0.23000000000000004, 0.2]

## Feature Importance
- maha_str_siglip_dinov2: 0.7704
- maha_str_forada: 0.0421
- prob_forada: 0.0385
- prob_siglip: 0.0373
- dis_siglip_dinov2: 0.0275
- dis_combined: 0.0170
- d_fake_siglip_dinov2: 0.0166
- d_real_siglip_dinov2: 0.0127
- dis_forada: 0.0109
- d_fake_forada: 0.0107
- d_real_forada: 0.0083
- dis_max: 0.0082

## Test Set Results

| Method | FAKE | REAL | Ratio |
|---|---|---|---|
| Rule Conservative | 17627 | - | - |
| Rule Balanced | 17627 | 6777 | 0.722 |
| **Classifier (t=0.236)** | **16062** | **8342** | **0.658** |

## Agreement
- Classifier vs Rule-based: **83.6%**
- Rule=FAKE, Clf=REAL: 2786
- Rule=REAL, Clf=FAKE: 1221

## Classifier Probability Distribution
- Mean: 0.5955
- Median: 0.9184
- >0.8: 13559
- <0.2: 8399
