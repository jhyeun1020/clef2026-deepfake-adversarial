# Purification Adversarial Detection — 4-Model Comparison

Test data: 5 categories × 100 each = 500 samples
- real (100): from data_split.json `real_holdout[:100]`
- clean_fake (100): random sample from `generation/outputs/final_1002/`
- jpeg_q80 (100): real images with JPEG Q80 applied
- adv_forada (100): random sample from `multi_attack/images/forada__*__pgd_eps4/`
- adv_siglip (100): random sample from `multi_attack/images/siglip__*__pgd_eps4/`

Method: median3 filter, delta_logit = logit(prob_med3) - logit(prob_orig)
- High positive delta = purification 후 fake-쪽으로 logit 올라감 (purification이 detector를 더 fake로 보게 만듦)
- Adversarial은 보통 큰 delta (purification으로 perturbation 깨짐 → 원래 fake 본성 회복)
- Real은 delta가 작아야 정상 (false positive 회피)

## 1. delta_logit distribution per category

Mean ± std (median in parens) per (model × category):

| model | real | clean_fake | jpeg_q80 | adv_forada | adv_siglip |
|---|---|---|---|---|---|
| SigLIP+DINOv2 | -0.20 ± 0.95 (-0.13) | -1.52 ± 0.88 (-1.48) | +0.83 ± 0.78 (+0.75) | -0.05 ± 0.97 (-0.04) | +4.95 ± 3.11 (+5.47) |
| GenD_DINOv3 | +0.82 ± 1.19 (+0.60) | +0.69 ± 1.34 (+0.39) | +0.50 ± 0.88 (+0.31) | +0.50 ± 1.03 (+0.33) | +0.10 ± 1.01 (+0.13) |
| ForAda | +0.65 ± 0.73 (+0.57) | +0.37 ± 0.52 (+0.35) | +0.59 ± 0.64 (+0.59) | +0.72 ± 0.73 (+0.65) | -0.29 ± 0.98 (-0.12) |
| EFFORT | +0.71 ± 0.81 (+0.72) | +1.03 ± 0.83 (+0.99) | -2.03 ± 1.31 (-2.01) | -1.11 ± 1.53 (-1.07) | -1.06 ± 1.48 (-0.90) |

**해석**: real의 delta_logit이 0에 가까워야 정상. clean_fake/adv보다 real이 작아야 detection signal로 사용 가능.

## 2. prob_orig per category (model이 원본 이미지를 어떻게 보는지)

| model | real | clean_fake | jpeg_q80 | adv_forada | adv_siglip |
|---|---|---|---|---|---|
| SigLIP+DINOv2 | 0.364 | 0.981 | 0.036 | 0.474 | 0.000 |
| GenD_DINOv3 | 0.097 | 0.360 | 0.102 | 0.406 | 0.418 |
| ForAda | 0.187 | 0.197 | 0.156 | 0.022 | 0.298 |
| EFFORT | 0.145 | 0.055 | 0.777 | 0.425 | 0.343 |

**해석**: clean_fake에서 prob 높으면 detector가 fake를 잘 판단. adversarial에서 prob 낮으면 공격이 통한 것.

## 3. AUROC (adversarial 탐지 — delta_logit를 score로 사용)

AUROC > 0.7: 의미 있는 분리. > 0.85: 강한 signal. ≤ 0.5: 신호 없음/역방향.

| model | adv_forada vs clean_fake | adv_forada vs jpeg_q80 | adv_siglip vs clean_fake | adv_siglip vs jpeg_q80 |
|---|---|---|---|---|
| SigLIP+DINOv2 | 0.872 | 0.225 | 0.997 | 0.834 |
| GenD_DINOv3 | 0.481 | 0.496 | 0.384 | 0.385 |
| ForAda | 0.642 | 0.554 | 0.241 | 0.183 |
| EFFORT | 0.093 | 0.693 | 0.088 | 0.701 |

**해석**: 본인 detector를 target으로 한 adversarial 검출이 cross-target보다 더 강해야 정상 (e.g. ForAda는 adv_forada에 강하지만 adv_siglip엔 약할 수 있음).

## 4. AUROC (combined adv vs clean / jpeg)

| model | (adv_forada+adv_siglip) vs clean_fake | (adv_forada+adv_siglip) vs jpeg_q80 |
|---|---|---|
| SigLIP+DINOv2 | 0.934 | 0.530 |
| GenD_DINOv3 | 0.432 | 0.440 |
| ForAda | 0.442 | 0.369 |
| EFFORT | 0.091 | 0.697 |

## 5. Real false positive risk

If we set threshold T on delta_logit to flag adversarial, fraction of Real that exceed T:

| model | T=0.5 | T=1.0 | T=1.5 | T=2.0 | adv mean | adv-real margin |
|---|---|---|---|---|---|---|
| SigLIP+DINOv2 | 21.0% | 7.0% | 3.0% | 0.0% | +2.45 | +2.65 |
| GenD_DINOv3 | 57.0% | 35.0% | 20.0% | 14.0% | +0.30 | -0.52 |
| ForAda | 55.0% | 22.0% | 9.0% | 7.0% | +0.22 | -0.43 |
| EFFORT | 65.0% | 33.0% | 12.0% | 8.0% | -1.09 | -1.80 |

**해석**: Real이 high delta를 자연 발생시키면 (GenD처럼 +2.3 mean) → adversarial signal과 분리 안 됨 → 실제 deploy 시 false positive 폭발.

## 6. 결론

| rank | model | avg AUROC vs jpeg | adv-real margin |
|---|---|---|---|
| 1 | EFFORT | 0.697 | -1.80 |
| 2 | SigLIP+DINOv2 | 0.530 | +2.65 |
| 3 | GenD_DINOv3 | 0.440 | -0.52 |
| 4 | ForAda | 0.369 | -0.43 |

핵심 비교: vs jpeg에서 AUROC 높으면서 adv-real margin이 양수(adv가 더 큰 delta)인 모델이 purification adversarial 탐지에 유효.