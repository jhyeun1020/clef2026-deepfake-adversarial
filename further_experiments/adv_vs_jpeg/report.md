# Pure Adversarial vs JPEG Decomposition

> A: Pure PGD (100), B: Pure JPEG Q80 (100), C: PGD+JPEG (100), D: Clean (200)

---

## 1. Cosine Similarity Matrix (mean vector)

| | A(pure adv) | B(pure jpeg) | C(adv+jpeg) | clean_fake | real |
|---|---|---|---|---|---|
| A_pure_adv | 1.000 | 0.949 | 0.942 | 0.539 | 0.924 |
| B_pure_jpeg | 0.949 | 1.000 | 0.960 | 0.525 | 0.940 |
| C_adv_jpeg | 0.942 | 0.960 | 1.000 | 0.511 | 0.931 |
| D_clean_fake | 0.539 | 0.525 | 0.511 | 1.000 | 0.553 |
| D_real | 0.924 | 0.940 | 0.931 | 0.553 | 1.000 |

**Key**: A↔B = **0.949**, A↔C = **0.942**, B↔C = **0.960**

## 2. Q1: 순수 adversarial과 순수 JPEG 유사도

**⚠ 순수 adversarial과 순수 JPEG 매우 유사 — 구분 불가능**

## 3. Q2: 기존 모델이 잡는 것은?

| version | det rate | mean prob |
|---|---|---|
| A (pure adv) | **67%** | 0.592 |
| B (pure jpeg) | 52% | 0.500 |
| C (adv+jpeg) | 99% | 0.869 |

**~ 모델이 adversarial을 부분적으로 포착**

## 4. Q3: 3-class 분리

- 3-class accuracy: **0.9100**
- **Adv vs JPEG AUROC: 0.9975**
- Confusion matrix (0=clean, 1=adv, 2=jpeg):

```
[[188   4   8]
 [ 11  87   2]
 [ 11   0  89]]
```

→ 순수 adversarial과 JPEG이 **잘 구분됨**. 분리 학습 시 pure adversarial signature 포착 가능.

## 5. Q4: cos=0.95의 원인 분해

- A↔B (순수 adv ↔ 순수 jpeg): **0.949**
- A↔C (순수 adv ↔ adv+jpeg): **0.942**
- B↔C (순수 jpeg ↔ adv+jpeg): **0.960**

B↔C > A↔C → C의 signature는 **JPEG에 더 가까움** (JPEG contribution dominant)

## 출력 파일

- [features.parquet](./features.parquet)
- [similarity_matrix.csv](./similarity_matrix.csv)
- [tsne_decomposition.png](./tsne_decomposition.png)
- [feature_decomposition.csv](./feature_decomposition.csv)
