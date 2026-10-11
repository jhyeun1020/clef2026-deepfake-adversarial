# Hypothesis Analysis

## Test others subgroups
- high (>0.7): 0 (0.0%)
- mid (0.4-0.7): 592 (29.6%)
- low (<0.4): 1408

## 가설 검증

### H1: Others 대부분 real → adversarial 없음
**지지됨**: high-prob others가 0.0%로 매우 소수

### H2: 별도 cluster → 다른 스타일 adversarial
### H3: Our 1002만 outlier → overfitting
our ↔ pilot_adv 거리: 1.2
our tightness: 19.1
→ pilot_adv와 가까움 → **overfitting 아님, 진짜 adversarial signal**

