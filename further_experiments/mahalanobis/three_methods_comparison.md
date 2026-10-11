# Three Methods Comparison

| Method | CV AUROC | Test det rate | Our 1002 recall | mean prob |
|---|---|---|---|---|
| Instability (24-dim RF) | 0.9797 | N/A (6k sample) | ~98% (prev) | N/A |
| **Mahalanobis (20-dim RF)** | **1.0000** | **0.0%** | **0.0%** | 0.185 |

Correlation (instability vs mahalanobis): **r=-0.011**

→ 약한 상관: **독립적 signal** — ensemble 효과 기대
