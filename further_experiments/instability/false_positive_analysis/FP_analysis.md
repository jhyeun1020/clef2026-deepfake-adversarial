# False Positive Analysis — 5 suspect images

## 대상 이미지

| # | image_id | adv_prob (RF) | top feature | top value | %ile in adv | %ile in real |
|---|---|---|---|---|---|---|
| 1 | 6488e0306ef3ca2f3c90 | 0.820 | aide_genimage__center_crop_192 | 0.5168 | 30% | 71% |
| 2 | e819e3aad4a3226de7d0 | 0.895 | aide_genimage__center_crop_192 | 0.8168 | 90% | 93% |
| 3 | ec634859158b8bcc121a | 0.915 | aide_genimage__center_crop_192 | 0.6230 | 48% | 78% |
| 4 | 1b91da9e460c6a7d0cdc | 0.895 | aide_genimage__center_crop_192 | 0.7009 | 65% | 84% |
| 5 | c3957f266d7078d9e959 | 0.825 | aide_genimage__center_crop_192 | 0.6549 | 56% | 80% |

## 대조군 — adv baseline (우리 1002 high-prob)

| # | image_id | adv_prob | top feature | top value |
|---|---|---|---|---|
| 1 | 60feecd82812b3ca8519 | 1.000 | aide_genimage__center_crop_192 | 0.7020 |
| 2 | 6229ffe049db851dd7f5 | 1.000 | aide_genimage__center_crop_192 | 0.8087 |
| 3 | 5a647e1d6b497a8f4d0a | 1.000 | aide_genimage__center_crop_192 | 0.6203 |
| 4 | 6b94443c0a878d394159 | 1.000 | distildire__face_crop_variable | 0.9818 |
| 5 | 6c112186b0c66912301d | 1.000 | aide_genimage__center_crop_192 | 0.7263 |

## 대조군 — real baseline (low-prob)

| # | image_id | adv_prob | top feature | top value |
|---|---|---|---|---|
| 1 | b5b10f28ca5ae7776c14 | 0.000 | siglip_dino__face_crop_variable | 0.8551 |
| 2 | 4cbba497dc2a7fe4987a | 0.000 | distildire__face_crop_variable | 0.7731 |
| 3 | 82b8053a47a4c2f0d74d | 0.000 | distildire__face_crop_variable | 0.8185 |
| 4 | 986236cec8e28dffb100 | 0.000 | distildire__face_crop_variable | 0.8301 |
| 5 | 6f6f606dba8d02273619 | 0.000 | haywood__random_crop_192_a | 0.8647 |

## 패턴 유사도 (평균 instability vector cosine similarity)

- suspect ↔ adv_baseline : **0.972**
- suspect ↔ real_baseline: **0.730**
- adv ↔ real             : 0.709

### 판정: **Case A — 의심 이미지들의 instability 패턴이 adv baseline과 유사**

시각적 인상과 달리, 이 이미지들은 adversarial 공격을 받았을 가능성이 높습니다.
모델이 visually subtle한 adversarial perturbation을 정확하게 포착한 것으로 보입니다.

## 개별 이미지 해석

### 1. `6488e0306ef3ca2f3c90`

- adv_prob: 0.820
- cos(vs adv): 0.831 | cos(vs real): 0.608
- **판정: adversarial-like**
- Top features:

| rank | feature | value | %ile(adv) | %ile(real) |
|---|---|---|---|---|
| 1 | aide_genimage__center_crop_192 | 0.5168 | 30% | 71% |
| 2 | distildire__face_crop_variable | 0.6794 | 12% | 47% |
| 3 | aide_genimage__random_crop_192_a | 0.5635 | 37% | 73% |
| 4 | xception__face_crop_variable | 0.2957 | 89% | 55% |
| 5 | aide_genimage__face_crop_variable | 0.5041 | 40% | 69% |

### 2. `e819e3aad4a3226de7d0`

- adv_prob: 0.895
- cos(vs adv): 0.965 | cos(vs real): 0.687
- **판정: adversarial-like**
- Top features:

| rank | feature | value | %ile(adv) | %ile(real) |
|---|---|---|---|---|
| 1 | aide_genimage__center_crop_192 | 0.8168 | 90% | 93% |
| 2 | distildire__face_crop_variable | 0.9431 | 33% | 73% |
| 3 | aide_genimage__random_crop_192_a | 0.8034 | 85% | 90% |
| 4 | npr__random_crop_192_a | 1.0000 | 90% | 80% |
| 5 | aide_genimage__face_crop_variable | 0.4730 | 39% | 67% |

### 3. `ec634859158b8bcc121a`

- adv_prob: 0.915
- cos(vs adv): 0.924 | cos(vs real): 0.696
- **판정: adversarial-like**
- Top features:

| rank | feature | value | %ile(adv) | %ile(real) |
|---|---|---|---|---|
| 1 | aide_genimage__center_crop_192 | 0.6230 | 48% | 78% |
| 2 | distildire__face_crop_variable | 0.9759 | 52% | 85% |
| 3 | aide_genimage__random_crop_192_a | 0.5944 | 40% | 76% |
| 4 | npr__random_crop_192_a | 1.0000 | 90% | 80% |
| 5 | npr__face_crop_variable | 1.0000 | 68% | 84% |

### 4. `1b91da9e460c6a7d0cdc`

- adv_prob: 0.895
- cos(vs adv): 0.883 | cos(vs real): 0.724
- **판정: adversarial-like**
- Top features:

| rank | feature | value | %ile(adv) | %ile(real) |
|---|---|---|---|---|
| 1 | aide_genimage__center_crop_192 | 0.7009 | 65% | 84% |
| 2 | distildire__face_crop_variable | 0.8824 | 22% | 62% |
| 3 | aide_genimage__random_crop_192_a | 0.6835 | 56% | 81% |
| 4 | aide_sd14__face_crop_variable | 0.4912 | 96% | 85% |
| 5 | aide_sd14__center_crop_192 | 0.6260 | 97% | 96% |

### 5. `c3957f266d7078d9e959`

- adv_prob: 0.825
- cos(vs adv): 0.866 | cos(vs real): 0.646
- **판정: adversarial-like**
- Top features:

| rank | feature | value | %ile(adv) | %ile(real) |
|---|---|---|---|---|
| 1 | aide_genimage__center_crop_192 | 0.6549 | 56% | 80% |
| 2 | aide_genimage__random_crop_192_a | 0.6923 | 57% | 82% |
| 3 | distildire__face_crop_variable | 0.7840 | 16% | 52% |
| 4 | npr__random_crop_192_a | 1.0000 | 90% | 80% |
| 5 | aide_genimage__face_crop_variable | 0.8717 | 89% | 97% |

## 출력 파일

- [raw_scores_table.csv](./raw_scores_table.csv)
- [rf_decision_breakdown.csv](./rf_decision_breakdown.csv)
- [comparison_patterns.csv](./comparison_patterns.csv)
- [instability_heatmap.png](./instability_heatmap.png)
- [comparison_grid.png](./comparison_grid.png)
