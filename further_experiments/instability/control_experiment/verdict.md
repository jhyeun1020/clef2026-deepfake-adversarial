# Augmented Real Control Experiment — Verdict

## 종합: **Case B**

Detector also responds to: ['jpeg_q50', 'jpeg_q70']. Not purely adversarial-specific.

## Augmentation별 cosine similarity to suspect

| aug_type | →suspect | →adv | verdict |
|---|---|---|---|
| original | 0.901 | 0.829 | original is SIMILAR to suspect/adversarial |
| gaussian_noise_002 | 0.890 | 0.873 | gaussian_noise_002 is PARTIALLY similar |
| gaussian_noise_005 | 0.682 | 0.614 | gaussian_noise_005 is DISTINCT from suspect |
| gaussian_blur_3 | 0.765 | 0.724 | gaussian_blur_3 is PARTIALLY similar |
| gaussian_blur_5 | 0.674 | 0.616 | gaussian_blur_5 is DISTINCT from suspect |
| jpeg_q50 | 0.921 | 0.891 | jpeg_q50 is SIMILAR to suspect/adversarial |
| jpeg_q70 | 0.964 | 0.914 | jpeg_q70 is SIMILAR to suspect/adversarial |
| rotation_15 | 0.817 | 0.769 | rotation_15 is PARTIALLY similar |
| poisson_10 | 0.642 | 0.703 | poisson_10 is DISTINCT from suspect |
| rotation_20 | 0.826 | 0.772 | rotation_20 is PARTIALLY similar |

max similarity to suspect (excluding original): **0.964**
Augmentations with cos > 0.85: 3
Augmentations with cos < 0.70: 3
