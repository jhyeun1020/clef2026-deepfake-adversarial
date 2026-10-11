"""
Generate 4-model comparison report from siglip/gend/forada/effort _results.csv.

Output: comparison_report.md
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

OUT = Path('/home/s20204116/GEN_AI_GTG/detection/purification_comparison')

MODELS = [
    ('SigLIP+DINOv2', 'siglip_results.csv'),
    ('GenD_DINOv3',   'gend_results.csv'),
    ('ForAda',        'forada_results.csv'),
    ('EFFORT',        'effort_results.csv'),
]

results = {}
for name, fn in MODELS:
    p = OUT / fn
    if not p.exists():
        print(f"MISSING: {fn}")
        continue
    df = pd.read_csv(p)
    results[name] = df
    print(f"Loaded {name}: {len(df)} rows")

# 1. delta_logit distribution per category per model
def stats(df, col, cat):
    s = df[df['category'] == cat][col]
    return {'n': len(s), 'mean': s.mean(), 'std': s.std(),
            'p10': s.quantile(0.1), 'p50': s.median(), 'p90': s.quantile(0.9)}


# 2. AUROC for adv vs (clean / jpeg) using delta_logit
def compute_auroc(df, pos_cat, neg_cat):
    pos = df[df['category'] == pos_cat]['delta_logit'].values
    neg = df[df['category'] == neg_cat]['delta_logit'].values
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    return roc_auc_score(y, s)


# 3. Real FP analysis: % of Real with |delta_logit| > some threshold
def real_fp_rate(df, threshold):
    """Real이 delta_logit > threshold인 비율 (FP가 될 가능성)."""
    real = df[df['category'] == 'real']['delta_logit']
    return (real > threshold).mean()


lines = []
lines.append("# Purification Adversarial Detection — 4-Model Comparison\n")
lines.append("Test data: 5 categories × 100 each = 500 samples")
lines.append("- real (100): from data_split.json `real_holdout[:100]`")
lines.append("- clean_fake (100): random sample from `generation/outputs/final_1002/`")
lines.append("- jpeg_q80 (100): real images with JPEG Q80 applied")
lines.append("- adv_forada (100): random sample from `multi_attack/images/forada__*__pgd_eps4/`")
lines.append("- adv_siglip (100): random sample from `multi_attack/images/siglip__*__pgd_eps4/`")
lines.append("\nMethod: median3 filter, delta_logit = logit(prob_med3) - logit(prob_orig)")
lines.append("- High positive delta = purification 후 fake-쪽으로 logit 올라감 (purification이 detector를 더 fake로 보게 만듦)")
lines.append("- Adversarial은 보통 큰 delta (purification으로 perturbation 깨짐 → 원래 fake 본성 회복)")
lines.append("- Real은 delta가 작아야 정상 (false positive 회피)\n")

# Section 1: distribution table
lines.append("## 1. delta_logit distribution per category\n")
lines.append("Mean ± std (median in parens) per (model × category):\n")
header = ['model'] + ['real', 'clean_fake', 'jpeg_q80', 'adv_forada', 'adv_siglip']
lines.append('| ' + ' | '.join(header) + ' |')
lines.append('|' + '|'.join(['---']*len(header)) + '|')

for name, df in results.items():
    row = [name]
    for cat in ['real', 'clean_fake', 'jpeg_q80', 'adv_forada', 'adv_siglip']:
        s = stats(df, 'delta_logit', cat)
        row.append(f"{s['mean']:+.2f} ± {s['std']:.2f} ({s['p50']:+.2f})")
    lines.append('| ' + ' | '.join(row) + ' |')

lines.append("\n**해석**: real의 delta_logit이 0에 가까워야 정상. clean_fake/adv보다 real이 작아야 detection signal로 사용 가능.\n")

# Section 2: prob_orig per category (sanity)
lines.append("## 2. prob_orig per category (model이 원본 이미지를 어떻게 보는지)\n")
lines.append('| ' + ' | '.join(header) + ' |')
lines.append('|' + '|'.join(['---']*len(header)) + '|')
for name, df in results.items():
    row = [name]
    for cat in ['real', 'clean_fake', 'jpeg_q80', 'adv_forada', 'adv_siglip']:
        s = stats(df, 'prob_orig', cat)
        row.append(f"{s['mean']:.3f}")
    lines.append('| ' + ' | '.join(row) + ' |')

lines.append("\n**해석**: clean_fake에서 prob 높으면 detector가 fake를 잘 판단. adversarial에서 prob 낮으면 공격이 통한 것.\n")

# Section 3: AUROC (adversarial vs clean / jpeg)
lines.append("## 3. AUROC (adversarial 탐지 — delta_logit를 score로 사용)\n")
lines.append("AUROC > 0.7: 의미 있는 분리. > 0.85: 강한 signal. ≤ 0.5: 신호 없음/역방향.\n")
lines.append('| model | adv_forada vs clean_fake | adv_forada vs jpeg_q80 | adv_siglip vs clean_fake | adv_siglip vs jpeg_q80 |')
lines.append('|---|---|---|---|---|')

for name, df in results.items():
    a = compute_auroc(df, 'adv_forada', 'clean_fake')
    b = compute_auroc(df, 'adv_forada', 'jpeg_q80')
    c = compute_auroc(df, 'adv_siglip', 'clean_fake')
    d = compute_auroc(df, 'adv_siglip', 'jpeg_q80')
    lines.append(f"| {name} | {a:.3f} | {b:.3f} | {c:.3f} | {d:.3f} |")

lines.append("\n**해석**: 본인 detector를 target으로 한 adversarial 검출이 cross-target보다 더 강해야 정상 (e.g. ForAda는 adv_forada에 강하지만 adv_siglip엔 약할 수 있음).\n")

# Section 4: combined adv vs each
lines.append("## 4. AUROC (combined adv vs clean / jpeg)\n")
lines.append('| model | (adv_forada+adv_siglip) vs clean_fake | (adv_forada+adv_siglip) vs jpeg_q80 |')
lines.append('|---|---|---|')

for name, df in results.items():
    pos = df[df['category'].isin(['adv_forada', 'adv_siglip'])]['delta_logit'].values
    neg_clean = df[df['category'] == 'clean_fake']['delta_logit'].values
    neg_jpeg = df[df['category'] == 'jpeg_q80']['delta_logit'].values
    y1 = np.concatenate([np.ones(len(pos)), np.zeros(len(neg_clean))])
    s1 = np.concatenate([pos, neg_clean])
    y2 = np.concatenate([np.ones(len(pos)), np.zeros(len(neg_jpeg))])
    s2 = np.concatenate([pos, neg_jpeg])
    lines.append(f"| {name} | {roc_auc_score(y1, s1):.3f} | {roc_auc_score(y2, s2):.3f} |")

# Section 5: Real FP risk
lines.append("\n## 5. Real false positive risk\n")
lines.append("If we set threshold T on delta_logit to flag adversarial, fraction of Real that exceed T:\n")
lines.append('| model | T=0.5 | T=1.0 | T=1.5 | T=2.0 | adv mean | adv-real margin |')
lines.append('|---|---|---|---|---|---|---|')

for name, df in results.items():
    real_d = df[df['category'] == 'real']['delta_logit']
    adv_d  = df[df['category'].isin(['adv_forada','adv_siglip'])]['delta_logit']
    fps = [(real_d > t).mean() for t in [0.5, 1.0, 1.5, 2.0]]
    margin = adv_d.mean() - real_d.mean()
    lines.append(f"| {name} | {fps[0]:.1%} | {fps[1]:.1%} | {fps[2]:.1%} | {fps[3]:.1%} | {adv_d.mean():+.2f} | {margin:+.2f} |")

lines.append("\n**해석**: Real이 high delta를 자연 발생시키면 (GenD처럼 +2.3 mean) → adversarial signal과 분리 안 됨 → 실제 deploy 시 false positive 폭발.\n")

# Section 6: 결론
lines.append("## 6. 결론\n")
best_models = []
for name, df in results.items():
    a = compute_auroc(df, 'adv_forada', 'jpeg_q80')
    c = compute_auroc(df, 'adv_siglip', 'jpeg_q80')
    real_d = df[df['category'] == 'real']['delta_logit']
    adv_d  = df[df['category'].isin(['adv_forada','adv_siglip'])]['delta_logit']
    margin = adv_d.mean() - real_d.mean()
    avg_jpeg_auroc = (a + c) / 2
    best_models.append((name, avg_jpeg_auroc, margin))

best_models.sort(key=lambda x: -x[1])
lines.append('| rank | model | avg AUROC vs jpeg | adv-real margin |')
lines.append('|---|---|---|---|')
for i, (name, auroc, margin) in enumerate(best_models, 1):
    lines.append(f"| {i} | {name} | {auroc:.3f} | {margin:+.2f} |")

lines.append("\n핵심 비교: vs jpeg에서 AUROC 높으면서 adv-real margin이 양수(adv가 더 큰 delta)인 모델이 purification adversarial 탐지에 유효.")

(OUT / 'comparison_report.md').write_text('\n'.join(lines))
print(f"\nGenerated: {OUT / 'comparison_report.md'}")
