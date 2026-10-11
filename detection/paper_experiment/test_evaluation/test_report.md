# Test Set Final Evaluation

| Model | margin | AUROC vs clean | AUROC vs jpeg | AUROC vs all | TNR@TPR95 | Wilcoxon p |
|---|---|---|---|---|---|---|
| linear_probe | -0.81 | 0.540 | 0.671 | 0.469 | 0.390 | 5.15e-05 |
| lora | -2.84 | 0.519 | 0.515 | 0.409 | 0.029 | 5.99e-02 |
| full_finetune | -4.12 | 0.464 | 0.335 | 0.332 | 0.170 | 2.31e-04 |
| gend_clip | +1.96 | 0.998 | 0.994 | 0.987 | 0.966 | 0.00e+00 |
| forada | +3.35 | 0.999 | 1.000 | 0.999 | 0.998 | 0.00e+00 |
| effort | +6.09 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00e+00 |