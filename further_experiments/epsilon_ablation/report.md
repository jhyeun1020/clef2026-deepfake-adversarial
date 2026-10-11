# ε Ablation: ε=2/255 vs ε=8/255

> 100 paired images, same PGD pipeline (attack_1002_all_nonface.py)
> ε=2: α=0.5/255, ε=8: α=2.0/255, 30 iterations, 13-detector ensemble

---

## 1. Perturbation Statistics

| metric | ε=2 | ε=8 |
|---|---|---|
| linf | 23.3900 | 39.3500 |
| l1_mean | 2.2373 | 6.1153 |
| face_nonface_ratio | 1.1409 | 1.0080 |
| patch_std | 0.3776 | 0.2758 |

## 2. Spatial Distribution

- ε=2 face/non-face ratio: **1.141**
- ε=8 face/non-face ratio: **1.008**
- **✗ EoT가 ε에 무관하게 균일화 유지**

## 3. RF Model Generalization

Model trained on real vs ε=2 adversarial:

| | det rate (>0.5) | mean prob |
|---|---|---|
| ε=2 (100장) | 100.0% | 0.925 |
| ε=8 (100장) | 86.0% | 0.678 |

**~ 부분적 generalization**

## 4. Instability Feature Comparison

ε=2 mean instability: 0.2796
ε=8 mean instability: 0.2295
Ratio (ε=8/ε=2): 0.82x

## 출력 파일

- [perturbation_comparison.csv](./perturbation_comparison.csv)
- [ratio_by_epsilon.png](./ratio_by_epsilon.png)
- [perturbation_heatmap_comparison.png](./perturbation_heatmap_comparison.png)
- [rf_probability_by_epsilon.png](./rf_probability_by_epsilon.png)
- [epsilon_comparison_scatter.png](./epsilon_comparison_scatter.png)
