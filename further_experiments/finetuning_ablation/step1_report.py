"""
Step 1e: Generate Linear Probe vs prior 4 models comparison report.

Reads: linear_probe_purification.csv + previous purification_comparison/*_results.csv
Output: ablation_report.md
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

OUT = Path('/home/s20204116/GEN_AI_GTG/detection/finetuning_ablation')
PRIOR = Path('/home/s20204116/GEN_AI_GTG/detection/purification_comparison')

# Load Linear Probe
lp = pd.read_csv(OUT / 'linear_probe_purification.csv')
print(f"Linear Probe: {len(lp)} rows, categories: {lp['category'].value_counts().to_dict()}")

# Load prior 4 models
priors = {
    'SigLIP+DINOv2': PRIOR / 'siglip_results.csv',
    'GenD_DINOv3':   PRIOR / 'gend_results.csv',
    'ForAda':        PRIOR / 'forada_results.csv',
    'EFFORT':        PRIOR / 'effort_results.csv',
}
all_results = {'Linear Probe': lp}
for name, fp in priors.items():
    if fp.exists():
        all_results[name] = pd.read_csv(fp)


def stats(df, cat):
    s = df[df['category'] == cat]['delta_logit']
    if len(s) == 0:
        return None
    return {'n': len(s), 'mean': s.mean(), 'std': s.std(), 'p50': s.median()}


def auc(df, pos, neg):
    p = df[df['category'] == pos]['delta_logit'].values
    n = df[df['category'] == neg]['delta_logit'].values
    if len(p) == 0 or len(n) == 0: return None
    y = np.concatenate([np.ones(len(p)), np.zeros(len(n))])
    s = np.concatenate([p, n])
    return roc_auc_score(y, s)


# Build comparison
out = []
out.append("# Linear Probe vs Prior 4 Models — Purification Comparison")
out.append("")
out.append("**가설 검증**: backbone fine-tuning 변경률이 낮을수록 purification (median3) 신호가 살아남는다.")
out.append("- Linear Probe (frozen, 0% backbone 변경) → 작동해야 가설 지지")
out.append("- LayerNorm/Adapter/SVD (backbone 직접 변경) → 신호 깨짐")
out.append("")
out.append("## 1. delta_logit per category (mean ± std)")
out.append("")
header_cats = ['real', 'clean_fake', 'jpeg_q80', 'adv_lp', 'adv_forada', 'adv_siglip']
out.append('| model | ' + ' | '.join(header_cats) + ' |')
out.append('|' + '|'.join(['---']*(len(header_cats)+1)) + '|')

for name, df in all_results.items():
    row = [name]
    for cat in header_cats:
        s = stats(df, cat)
        if s is None:
            row.append('—')
        else:
            row.append(f"{s['mean']:+.2f}±{s['std']:.2f}")
    out.append('| ' + ' | '.join(row) + ' |')
out.append("")
out.append("**adv_lp** = Linear Probe target adversarial (이번 step에서 새로 생성)")
out.append("")

out.append("## 2. prob_orig per category (model 원본 응답)")
out.append("")
out.append('| model | ' + ' | '.join(header_cats) + ' |')
out.append('|' + '|'.join(['---']*(len(header_cats)+1)) + '|')
for name, df in all_results.items():
    row = [name]
    for cat in header_cats:
        s = df[df['category'] == cat]['prob_orig']
        if len(s) == 0:
            row.append('—')
        else:
            row.append(f"{s.mean():.3f}")
    out.append('| ' + ' | '.join(row) + ' |')
out.append("")

out.append("## 3. AUROC (delta_logit를 score로 사용 — adversarial 분리)")
out.append("")
tasks = [
    ('adv_lp',     'clean_fake'),
    ('adv_lp',     'jpeg_q80'),
    ('adv_forada', 'jpeg_q80'),
    ('adv_siglip', 'jpeg_q80'),
]
out.append('| model | ' + ' | '.join([f"{p} vs {n}" for p, n in tasks]) + ' |')
out.append('|' + '|'.join(['---']*(len(tasks)+1)) + '|')
for name, df in all_results.items():
    row = [name]
    for pos, neg in tasks:
        a = auc(df, pos, neg)
        row.append(f"{a:.3f}" if a is not None else '—')
    out.append('| ' + ' | '.join(row) + ' |')
out.append("")
out.append("**해석**: adv_lp 컬럼은 Linear Probe 자체 target adversarial이므로 LP에서 의미 있고, 다른 모델은 cross-target (LP target adv를 다른 모델이 어떻게 보는지). 모델별 own-target 비교는 표 진입선 굵게 표시.")
out.append("")

out.append("## 4. Real false positive risk")
out.append("")
out.append("Real 이미지에 median3 적용 시 delta_logit이 큰 비율 (false positive 위험)")
out.append('| model | T=0.5 | T=1.0 | T=1.5 | T=2.0 | adv_lp mean | adv_lp - real margin |')
out.append('|---|---|---|---|---|---|---|')
for name, df in all_results.items():
    real_d = df[df['category'] == 'real']['delta_logit']
    if 'adv_lp' in df['category'].values:
        adv_d = df[df['category'] == 'adv_lp']['delta_logit']
    else:
        # Use combined adv as proxy for prior models that don't have adv_lp
        adv_d = df[df['category'].isin(['adv_forada','adv_siglip'])]['delta_logit']
    fps = [(real_d > t).mean() for t in [0.5, 1.0, 1.5, 2.0]]
    margin = adv_d.mean() - real_d.mean()
    out.append(f"| {name} | {fps[0]:.1%} | {fps[1]:.1%} | {fps[2]:.1%} | {fps[3]:.1%} | {adv_d.mean():+.2f} | {margin:+.2f} |")
out.append("")

out.append("## 5. 결론 — backbone 변경률 가설 검증")
out.append("")
out.append("| 모델 | Backbone | Fine-tuning | 변경률 | Real Δlogit | Adv Δlogit | 작동? |")
out.append("|---|---|---|---|---|---|---|")
mods = [
    ('Linear Probe', 'CLIP ViT-L/14', 'None (frozen)', '0%'),
    ('SigLIP+DINOv2', 'SigLIP+DINOv2', 'LoRA r=32', '~1%'),
    ('GenD_DINOv3', 'DINOv3', 'LayerNorm', '0.03%'),
    ('ForAda', 'CLIP ViT-L/14', 'Adapter', '~5%'),
    ('EFFORT', 'CLIP ViT-L/14', 'SVD residual', '~5%'),
]
for n, b, ft, pct in mods:
    if n not in all_results: continue
    df = all_results[n]
    real_m = stats(df, 'real')['mean']
    if 'adv_lp' in df['category'].values:
        adv_m = stats(df, 'adv_lp')['mean']
    else:
        adv_m = df[df['category'].isin(['adv_forada','adv_siglip'])]['delta_logit'].mean()
    margin = adv_m - real_m
    status = '✅' if margin > 1.0 else ('⚠' if margin > 0 else '❌')
    out.append(f"| {n} | {b} | {ft} | {pct} | {real_m:+.2f} | {adv_m:+.2f} | {status} |")

(OUT / 'ablation_report.md').write_text('\n'.join(out))
print(f"Saved {OUT / 'ablation_report.md'}")
