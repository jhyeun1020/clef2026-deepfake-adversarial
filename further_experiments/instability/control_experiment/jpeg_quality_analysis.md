# JPEG Quality Analysis

## Task 1: JPEG Q80 Instability Pattern

| target | cosine sim |
|---|---|
| Q80 → suspect | **0.952** |
| Q80 → adv_baseline | 0.888 |
| Q80 → real_baseline | 0.661 |

**판정**: ⚠ JPEG Q80 matches adversarial pattern

비교:
| aug | →suspect |
|---|---|
| jpeg_q50 | 0.921 |
| jpeg_q70 | 0.964 |
| **jpeg_q80** | **0.952** |

→ Q50~Q80 모두 suspect와 cos > 0.9. **JPEG 재압축 자체가 adversarial과 동일한 instability signature를 만듦.**

## Task 2: Test Set JPEG Quality Distribution

24404장 BAR (Block Artifact Ratio) 기반 추정:

| Quality | n | % |
|---|---|---|
| Q~45 | 853 | 3.5% |
| Q~55 | 1480 | 6.1% |
| Q~65 | 1775 | 7.3% |
| Q~75 | 3541 | 14.5% |
| Q~85 | 5234 | 21.4% |
| Q~95 | 11521 | 47.2% |

BAR 평균: 1.0294, 중앙값: 1.0145

## Task 3: Quality vs Adversarial Probability

- Spearman(BAR, adv_prob) = **0.5030** (p=0.00e+00)
- Spearman(quality, adv_prob) = **-0.5192** (p=0.00e+00)

**판정**: ⚠ Positive correlation (ρ=0.503) — higher BAR (lower quality) → higher adv prob. JPEG artifacts confound detection.

| Quality bin | n | adv_rate | mean_prob |
|---|---|---|---|
| <50 | 556 | 0.872 | 0.809 |
| 50-60 | 618 | 0.717 | 0.643 |
| 60-70 | 469 | 0.431 | 0.434 |
| 70-80 | 823 | 0.230 | 0.291 |
| 80-90 | 1091 | 0.105 | 0.208 |
| 90+ | 2445 | 0.067 | 0.172 |

## 종합 해석

1. JPEG Q80 instability는 adversarial과 cos=0.952로 매우 유사
2. Test set의 JPEG quality 분포와 adversarial 판정률의 상관관계로 confounding 정도 판단
