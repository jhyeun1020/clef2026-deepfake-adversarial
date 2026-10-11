# Fine-tuning Ablation — Final 6-Model Comparison

**Hypothesis**: Backbone weights가 inference 시 그대로 통과되는 fine-tuning만 purification (median3) 신호가 살아남는다.
- Backbone 미수정 (frozen / LoRA = 병렬 경로): 작동 ✅
- Backbone forward 자체 수정 (LayerNorm / Adapter / SVD-residual): 깨짐 ❌

## 1. delta_logit per category (mean ± std)

| model | real | clean_fake | jpeg_q80 | adv_lp | adv_lora | adv_forada | adv_siglip |
|---|---|---|---|---|---|---|---|
| Linear Probe | +0.40±0.47 | +0.49±0.45 | +0.10±0.51 | +3.04±2.27 | — | -0.21±1.05 | -0.09±0.88 |
| LoRA (new) | +1.24±0.93 | +0.28±0.88 | +0.69±1.05 | +2.10±2.34 | +2.70±2.20 | +0.57±1.32 | -0.31±0.91 |
| SigLIP+DINOv2 | -0.20±0.95 | -1.52±0.88 | +0.83±0.78 | — | — | -0.05±0.97 | +4.95±3.11 |
| GenD_DINOv3 | +0.82±1.19 | +0.69±1.34 | +0.50±0.88 | — | — | +0.50±1.03 | +0.10±1.01 |
| ForAda | +0.65±0.73 | +0.37±0.52 | +0.59±0.64 | — | — | +0.72±0.73 | -0.29±0.98 |
| EFFORT | +0.71±0.81 | +1.03±0.83 | -2.03±1.31 | — | — | -1.11±1.53 | -1.06±1.48 |

`adv_lp` = Linear Probe target adv (Step 1). `adv_lora` = LoRA target adv (Step 2).

## 2. prob_orig per category

| model | real | clean_fake | jpeg_q80 | adv_lp | adv_lora | adv_forada | adv_siglip |
|---|---|---|---|---|---|---|---|
| Linear Probe | 0.306 | 0.246 | 0.362 | 0.000 | — | 0.690 | 0.580 |
| LoRA (new) | 0.186 | 0.252 | 0.238 | 0.000 | 0.000 | 0.584 | 0.698 |
| SigLIP+DINOv2 | 0.364 | 0.981 | 0.036 | — | — | 0.474 | 0.000 |
| GenD_DINOv3 | 0.097 | 0.360 | 0.102 | — | — | 0.406 | 0.418 |
| ForAda | 0.187 | 0.197 | 0.156 | — | — | 0.022 | 0.298 |
| EFFORT | 0.145 | 0.055 | 0.777 | — | — | 0.425 | 0.343 |

## 3. AUROC (delta_logit를 score로 사용)

| model | adv_lp vs jpeg_q80 | adv_lora vs jpeg_q80 | adv_forada vs jpeg_q80 | adv_siglip vs jpeg_q80 |
|---|---|---|---|---|
| Linear Probe | 0.954 | — | 0.403 | 0.462 |
| LoRA (new) | 0.675 | 0.804 | 0.472 | 0.234 |
| SigLIP+DINOv2 | — | — | 0.225 | 0.834 |
| GenD_DINOv3 | — | — | 0.496 | 0.385 |
| ForAda | — | — | 0.554 | 0.183 |
| EFFORT | — | — | 0.693 | 0.701 |

**해석**: 각 모델의 own-target (대각선)이 가장 강한 신호. SigLIP→adv_siglip, ForAda→adv_forada, Linear Probe→adv_lp, LoRA→adv_lora.

## 4. Real false positive risk (Real에 median3 적용 시 high delta 비율)

| model | T=0.5 | T=1.0 | T=1.5 | T=2.0 | own-adv mean | adv-real margin |
|---|---|---|---|---|---|---|
| Linear Probe | 33% | 9% | 4% | 0% | +3.04 (adv_lp) | +2.64 |
| LoRA (new) | 75% | 56% | 33% | 21% | +2.70 (adv_lora) | +1.46 |
| SigLIP+DINOv2 | 21% | 7% | 3% | 0% | +4.95 (adv_siglip) | +5.15 |
| GenD_DINOv3 | 57% | 35% | 20% | 14% | +0.30 ((combined adv)) | -0.52 |
| ForAda | 55% | 22% | 9% | 7% | +0.72 (adv_forada) | +0.07 |
| EFFORT | 65% | 33% | 12% | 8% | -1.09 ((combined adv)) | -1.80 |

## 5. 최종 결론 — 가설 검증

| 모델 | Backbone | Fine-tuning | 변경률 | Real Δ | Adv Δ | margin | 작동 |
|---|---|---|---|---|---|---|---|
| Linear Probe | CLIP ViT-L/14 | None (frozen) | 0% | +0.40 | +3.04 | +2.64 | ✅ |
| LoRA (new) | CLIP ViT-L/14 | LoRA r=32 | ~1% | +1.24 | +2.70 | +1.46 | ✅ |
| SigLIP+DINOv2 | SigLIP+DINOv2 | LoRA r=32 (Bombek1) | ~1% | -0.20 | +4.95 | +5.15 | ✅ |
| GenD_DINOv3 | DINOv3 | LayerNorm | 0.03% | +0.82 | +0.30 | -0.52 | ❌ |
| ForAda | CLIP ViT-L/14 | Adapter | ~5% | +0.65 | +0.72 | +0.07 | ⚠ |
| EFFORT | CLIP ViT-L/14 | SVD residual | ~5% | +0.71 | -1.09 | -1.80 | ❌ |

## 6. 가설 결과

- Linear Probe (frozen, 0% backbone change): ✅
- LoRA (~1% via parallel paths, backbone untouched): ✅

**가설 검증 성공**: 같은 CLIP backbone에서 frozen과 LoRA 모두 purification이 작동.
LayerNorm/Adapter/SVD-residual은 backbone forward를 직접 수정하여 신호 깨짐.

**결론**: SigLIP+DINOv2가 작동하는 이유는 dual-encoder나 큰 데이터가 아니라, **LoRA가 backbone을 수정하지 않기 때문**.