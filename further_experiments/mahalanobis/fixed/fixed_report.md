# Mahalanobis RF Fix — Phase 2 Scale

## RF CV AUROC: **0.9998**

## Our 1002 Recall: **99.6%** (mean prob=0.952)

## Test Det Rate (>0.5): **17.7%**

## Bimodality: BC=0.5495 (unimodal)

## Correlation with Instability: r=0.696

## Three-way Comparison

| Method | CV AUROC | Test det | Our recall |
|---|---|---|---|
| Instability 24-dim | 0.9797 | ~26% | ~98% |
| Mahalanobis 20-dim (fixed) | 0.9998 | 17.7% | 99.6% |
| Ensemble (avg) | N/A | 22.3% | 100.0% |

## 판정

**✓ Fix 성공, Mahalanobis approach 유효**

## 출력 파일

- [pilot_distances_phase2scale.parquet](./pilot_distances_phase2scale.parquet)
- [rf_model_phase2.pkl](./rf_model_phase2.pkl)
- [test_predictions_fixed.parquet](./test_predictions_fixed.parquet)
- [test_probability_distributions.png](./test_probability_distributions.png)
- [correlation_inst_vs_mah.png](./correlation_inst_vs_mah.png)
