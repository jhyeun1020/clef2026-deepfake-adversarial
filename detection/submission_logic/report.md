# Test Set Submission Report

## Pipeline
1. SigLIP+DINOv2 probability (pre-existing scores)
2. ForAda probability (newly extracted)
3. Mahalanobis distances (SigLIP + ForAda, Clean Gaussian)
4. Disagreement scores
5. pHash matching (1,002 adversarial)
6. Rule-based prediction

## Data
- Test set: 24404 images
- pHash matched (our adversarial): 1002
- Feature NaN: 0

## Distribution

| Signal | Mean | Median | >threshold |
|---|---|---|---|
| SigLIP prob | 0.6152 | 0.7655 | >0.8: 11636 |
| ForAda prob | 0.3338 | 0.1779 | >0.5: 6690 |
| Maha SigLIP FAKE | - | - | 10918 |
| Maha ForAda FAKE | - | - | 10825 |
| Disagreement | 0.000241 | 0.000007 | >0.0003: 6335 |

## Our Adversarial (pHash matched: 1002)
- SigLIP prob mean: 0.1168
- ForAda prob mean: 0.0864
- Maha FAKE (sig): 571/1002
- Disagree mean: 0.000433

## Submissions

| Variant | FAKE | REAL | Ratio |
|---|---|---|---|
| Conservative | 14103 | 10301 | 0.578 |
| **Balanced** | **17627** | **6777** | **0.722** |
| Aggressive | 20151 | 4253 | 0.826 |

## Balanced Prediction Breakdown

| Rule | Count | % |
|---|---|---|
| siglip_high | 11632 | 47.7% |
| default_real | 6777 | 27.8% |
| forada_high | 1994 | 8.2% |
| disagreement | 1368 | 5.6% |
| both_maha_fake | 1294 | 5.3% |
| phash | 1002 | 4.1% |
| sig_maha_prob | 337 | 1.4% |

## Sanity Checks
- Our 1,002 adv all FAKE (balanced): True
- FAKE ratio 0.3-0.7: False
