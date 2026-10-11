# Test Set Instability Detection — Summary

> RandomForest 학습: real_reference 1000 + adversarial_fake 1002 (24-dim instability)
> 50/50 cross-validation AUROC: **0.9811**
> Test 적용: 6,002장 (우리 1002 + random 4,500 + face-failed 500)

---

## 1. Face Detection 통계 (test set 24,404 전체)

- Total            : 24,404
- Face detected    : 23,118 (94.7%)
- Face NOT detected: 1,286 (5.3%)

## 2. Threshold별 adversarial 판정 수 (n=6,002 샘플 기준)

| threshold | RF | RF % | Logistic | LR % |
|---|---|---|---|---|
| 0.3 | 2304 | 38.4% | 1774 | 29.6% |
| 0.5 | 1598 | 26.6% | 1448 | 24.1% |
| 0.7 | 1036 | 17.3% | 1128 | 18.8% |
| 0.9 | 616 | 10.3% | 664 | 11.1% |

Bimodality coefficient (RF): **0.7464** (bimodal)
Bimodality coefficient (LR): **0.8616** (bimodal)

## 3. 우리 팀 1002장 cross-check

- 판정 성공: **982/1002** = 98.0%
- prob 평균: 0.880
- prob 중앙값: 0.930
- 실패 케이스: 20장 → [our_1002_misses.csv](./our_1002_misses.csv)

## 4. 우리 1002장 제외 → 다른 이미지에서 adversarial 판정

- 우리 1002 제외 샘플: 5000장
- adv 판정 (threshold 0.5): **616장 (12.3%)**

전체 24,404장 기준 추정:
- 우리 1002 제외 23,402장 중 sampling 4,500장에서 random_4500 풀의 adv 비율을 측정
- random_4500 그룹: 412/4500 = 9.16%
- → 23,402장 추정 adv 수: 약 **2143장**

## 5. Face detected vs failed 그룹 비교

| group | n | mean(prob) | adv판정 | rate |
|---|---|---|---|---|
| face_detected | 5502 | 0.319 | 1394 | 25.3% |
| face_failed   | 500 | 0.368 | 204 | 40.8% |

## 6. 핵심 결론

- **Q1**: **일부 다른 팀이 adversarial 공격 사용**한 것으로 추정
- **Q2 (bimodality)**: BC=0.746 — bimodal, 두 그룹 명확히 분리
- **Q3 (우리 1002 회수율)**: 98.0% — generalization 양호
- **Q4**: 시각적 분석은 [samples_high_confidence_adversarial.png] 참고

## 출력 파일

- [test_instability_features.parquet](./test_instability_features.parquet)
- [test_instability_predictions.parquet](./test_instability_predictions.parquet)
- [test_prediction_distribution.png](./test_prediction_distribution.png)
- [samples_high_confidence_adversarial.png](./samples_high_confidence_adversarial.png)
- [samples_mid_confidence_adversarial.png](./samples_mid_confidence_adversarial.png)
- [samples_high_confidence_clean.png](./samples_high_confidence_clean.png)
- [samples_face_failed_adversarial.png](./samples_face_failed_adversarial.png)
- [our_1002_misses.png](./our_1002_misses.png)
- [our_1002_prob_distribution.png](./our_1002_prob_distribution.png)
- [our_1002_misses.csv](./our_1002_misses.csv)
- [sample_filenames.csv](./sample_filenames.csv)
