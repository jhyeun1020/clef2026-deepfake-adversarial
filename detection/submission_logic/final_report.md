# Final Submission Report

## Purification Classifier CV AUROC
- SigLIP purification: **0.7865 ± 0.1799**
- GenD purification: **0.6562 ± 0.3044**
- Combined: **0.7213 ± 0.3362**

## Final Prediction

| Metric | Value |
|---|---|
| FAKE | 21201 |
| REAL | 3203 |
| Ratio | 0.869 |
| Our 1,002 all FAKE | True |

## Breakdown

| Rule | Count | % |
|---|---|---|
| ensemble | 17527 | 71.8% |
| real | 3203 | 13.1% |
| adv_siglip | 2350 | 9.6% |
| phash | 1002 | 4.1% |
| adv_gend | 322 | 1.3% |

## Comparison with Previous Submissions

| method            |   fake |   real |    ratio |
|:------------------|-------:|-------:|---------:|
| Rule Conservative |  14103 |  10301 | 0.577897 |
| Rule Balanced     |  17627 |   6777 | 0.7223   |
| Ensemble+Disagree |  19399 |   5005 | 0.794911 |
| Classifier        |  16062 |   8342 | 0.658171 |
| ** Final **       |  21201 |   3203 | 0.868751 |

## Pipeline
1. **pHash**: 우리 1,002장 확정 FAKE
2. **2-Detector Ensemble**: max(SigLIP, GenD_DINOv3) > 0.5 → FAKE
3. **SigLIP Purification Classifier**: adversarial 탐지 (SigLIP target 공격)
4. **GenD Purification Classifier**: adversarial 탐지 (ForAda target 공격)
5. **Default**: REAL
