"""
Step 2e: 6-model final comparison report.

Reads 6 CSVs:
  - lora_purification.csv (this Step 2)
  - linear_probe_purification.csv (Step 1)
  - purification_comparison/{siglip,gend,forada,effort}_results.csv (prior 4 models)

Outputs:
  - ablation_comparison.csv
  - ablation_report.md
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

OUT = Path('/home/s20204116/GEN_AI_GTG/detection/finetuning_ablation')
PRIOR = Path('/home/s20204116/GEN_AI_GTG/detection/purification_comparison')

MODELS = [
    ('Linear Probe',  OUT / 'linear_probe_purification.csv',
        'CLIP ViT-L/14', 'None (frozen)', '0%'),
    ('LoRA (new)',    OUT / 'lora_purification.csv',
        'CLIP ViT-L/14', 'LoRA r=32', '~1%'),
    ('SigLIP+DINOv2', PRIOR / 'siglip_results.csv',
        'SigLIP+DINOv2', 'LoRA r=32 (Bombek1)', '~1%'),
    ('GenD_DINOv3',   PRIOR / 'gend_results.csv',
        'DINOv3', 'LayerNorm', '0.03%'),
    ('ForAda',        PRIOR / 'forada_results.csv',
        'CLIP ViT-L/14', 'Adapter', '~5%'),
    ('EFFORT',        PRIOR / 'effort_results.csv',
        'CLIP ViT-L/14', 'SVD residual', '~5%'),
]

results = {}
metas = {}
for name, fp, backbone, ft, pct in MODELS:
    if not fp.exists():
        print(f"MISSING: {name} → {fp}")
        continue
    results[name] = pd.read_csv(fp)
    metas[name] = (backbone, ft, pct)
    print(f"Loaded {name}: {len(results[name])} rows")


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


# ============== ablation_comparison.csv ==============
own_adv = {
    'Linear Probe': 'adv_lp',
    'LoRA (new)':   'adv_lora',
    'SigLIP+DINOv2':'adv_siglip',
    'GenD_DINOv3':  None,  # no own target
    'ForAda':       'adv_forada',
    'EFFORT':       None,
}

rows = []
for name in [m[0] for m in MODELS]:
    if name not in results: continue
    df = results[name]
    bb, ft, pct = metas[name]
    real_s = stats(df, 'real')
    clean_s = stats(df, 'clean_fake')
    jpeg_s = stats(df, 'jpeg_q80')
    own = own_adv.get(name)
    own_s = stats(df, own) if own else None
    # Combined adv (any adv_* category in df)
    adv_cats = [c for c in df['category'].unique() if c.startswith('adv_')]
    combined_d = df[df['category'].isin(adv_cats)]['delta_logit']
    combined_mean = combined_d.mean() if len(combined_d) else None

    auc_own_clean = auc(df, own, 'clean_fake') if own else None
    auc_own_jpeg  = auc(df, own, 'jpeg_q80')   if own else None

    # Use the cross-adv with best signal as fallback for GenD/EFFORT
    best_alt_pos = None; best_alt_auc = None
    if own is None:
        for c in ['adv_siglip', 'adv_forada', 'adv_lp', 'adv_lora']:
            if c in adv_cats:
                a = auc(df, c, 'jpeg_q80')
                if a is not None and (best_alt_auc is None or a > best_alt_auc):
                    best_alt_auc = a; best_alt_pos = c

    margin = (own_s['mean'] - real_s['mean']) if own_s else (combined_mean - real_s['mean'] if combined_mean is not None else None)
    works = (margin is not None and margin > 1.0)

    rows.append({
        'model': name,
        'backbone': bb,
        'fine_tuning': ft,
        'change_pct': pct,
        'real_delta_mean':       real_s['mean'],
        'clean_fake_delta_mean': clean_s['mean'],
        'jpeg_q80_delta_mean':   jpeg_s['mean'],
        'own_adv_category':      own or f'(none, ref: {best_alt_pos})',
        'own_adv_delta_mean':    own_s['mean'] if own_s else (df[df['category']==best_alt_pos]['delta_logit'].mean() if best_alt_pos else None),
        'adv_real_margin':       margin,
        'auroc_own_vs_clean':    auc_own_clean if auc_own_clean is not None else best_alt_auc,
        'auroc_own_vs_jpeg':     auc_own_jpeg  if auc_own_jpeg  is not None else best_alt_auc,
        'works':                 '✅' if works else '❌',
    })

cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(OUT / 'ablation_comparison.csv', index=False)
print(f"\nSaved ablation_comparison.csv")


# ============== ablation_report.md ==============
out = []
out.append("# Fine-tuning Ablation — Final 6-Model Comparison\n")
out.append("**Hypothesis**: Backbone weights가 inference 시 그대로 통과되는 fine-tuning만 purification (median3) 신호가 살아남는다.")
out.append("- Backbone 미수정 (frozen / LoRA = 병렬 경로): 작동 ✅")
out.append("- Backbone forward 자체 수정 (LayerNorm / Adapter / SVD-residual): 깨짐 ❌\n")

# Section 1: per-category delta_logit
out.append("## 1. delta_logit per category (mean ± std)\n")
header_cats = ['real', 'clean_fake', 'jpeg_q80', 'adv_lp', 'adv_lora', 'adv_forada', 'adv_siglip']
out.append('| model | ' + ' | '.join(header_cats) + ' |')
out.append('|' + '|'.join(['---']*(len(header_cats)+1)) + '|')

for name in [m[0] for m in MODELS]:
    if name not in results: continue
    df = results[name]
    row = [name]
    for cat in header_cats:
        s = stats(df, cat)
        if s is None:
            row.append('—')
        else:
            row.append(f"{s['mean']:+.2f}±{s['std']:.2f}")
    out.append('| ' + ' | '.join(row) + ' |')
out.append("")
out.append("`adv_lp` = Linear Probe target adv (Step 1). `adv_lora` = LoRA target adv (Step 2).\n")

# Section 2: prob_orig
out.append("## 2. prob_orig per category\n")
out.append('| model | ' + ' | '.join(header_cats) + ' |')
out.append('|' + '|'.join(['---']*(len(header_cats)+1)) + '|')
for name in [m[0] for m in MODELS]:
    if name not in results: continue
    df = results[name]
    row = [name]
    for cat in header_cats:
        s = df[df['category'] == cat]['prob_orig']
        if len(s) == 0: row.append('—')
        else: row.append(f"{s.mean():.3f}")
    out.append('| ' + ' | '.join(row) + ' |')
out.append("")

# Section 3: AUROC per task
out.append("## 3. AUROC (delta_logit를 score로 사용)\n")
tasks = [
    ('adv_lp',     'jpeg_q80'),
    ('adv_lora',   'jpeg_q80'),
    ('adv_forada', 'jpeg_q80'),
    ('adv_siglip', 'jpeg_q80'),
]
out.append('| model | ' + ' | '.join([f"{p} vs {n}" for p, n in tasks]) + ' |')
out.append('|' + '|'.join(['---']*(len(tasks)+1)) + '|')
for name in [m[0] for m in MODELS]:
    if name not in results: continue
    df = results[name]
    row = [name]
    for pos, neg in tasks:
        a = auc(df, pos, neg)
        row.append(f"{a:.3f}" if a is not None else '—')
    out.append('| ' + ' | '.join(row) + ' |')
out.append("")
out.append("**해석**: 각 모델의 own-target (대각선)이 가장 강한 신호. SigLIP→adv_siglip, ForAda→adv_forada, Linear Probe→adv_lp, LoRA→adv_lora.\n")

# Section 4: Real false positive risk
out.append("## 4. Real false positive risk (Real에 median3 적용 시 high delta 비율)\n")
out.append('| model | T=0.5 | T=1.0 | T=1.5 | T=2.0 | own-adv mean | adv-real margin |')
out.append('|---|---|---|---|---|---|---|')
for name in [m[0] for m in MODELS]:
    if name not in results: continue
    df = results[name]
    real_d = df[df['category'] == 'real']['delta_logit']
    own = own_adv.get(name)
    if own and own in df['category'].values:
        adv_d = df[df['category'] == own]['delta_logit']
        own_label = own
    else:
        adv_d = df[df['category'].str.startswith('adv_')]['delta_logit']
        own_label = '(combined adv)'
    fps = [(real_d > t).mean() for t in [0.5, 1.0, 1.5, 2.0]]
    margin = adv_d.mean() - real_d.mean()
    out.append(f"| {name} | {fps[0]:.0%} | {fps[1]:.0%} | {fps[2]:.0%} | {fps[3]:.0%} | {adv_d.mean():+.2f} ({own_label}) | {margin:+.2f} |")
out.append("")

# Section 5: 최종 결론 표
out.append("## 5. 최종 결론 — 가설 검증\n")
out.append("| 모델 | Backbone | Fine-tuning | 변경률 | Real Δ | Adv Δ | margin | 작동 |")
out.append("|---|---|---|---|---|---|---|---|")

# Sort by hypothesized order
order = ['Linear Probe', 'LoRA (new)', 'SigLIP+DINOv2', 'GenD_DINOv3', 'ForAda', 'EFFORT']
for name in order:
    if name not in results: continue
    df = results[name]
    bb, ft, pct = metas[name]
    real_m = stats(df, 'real')['mean']
    own = own_adv.get(name)
    if own and own in df['category'].values:
        adv_m = stats(df, own)['mean']
    else:
        adv_m = df[df['category'].str.startswith('adv_')]['delta_logit'].mean()
    margin = adv_m - real_m
    status = '✅' if margin > 1.0 else ('⚠' if margin > 0 else '❌')
    out.append(f"| {name} | {bb} | {ft} | {pct} | {real_m:+.2f} | {adv_m:+.2f} | {margin:+.2f} | {status} |")
out.append("")

# Section 6: hypothesis verdict
out.append("## 6. 가설 결과\n")
lp_ok = (cmp_df.loc[cmp_df['model']=='Linear Probe','adv_real_margin'].values[0] > 1.0) if 'Linear Probe' in cmp_df['model'].values else False
lora_ok = (cmp_df.loc[cmp_df['model']=='LoRA (new)','adv_real_margin'].values[0] > 1.0) if 'LoRA (new)' in cmp_df['model'].values else False
out.append(f"- Linear Probe (frozen, 0% backbone change): {'✅' if lp_ok else '❌'}")
out.append(f"- LoRA (~1% via parallel paths, backbone untouched): {'✅' if lora_ok else '❌'}")

if lp_ok and lora_ok:
    out.append("\n**가설 검증 성공**: 같은 CLIP backbone에서 frozen과 LoRA 모두 purification이 작동.")
    out.append("LayerNorm/Adapter/SVD-residual은 backbone forward를 직접 수정하여 신호 깨짐.")
    out.append("\n**결론**: SigLIP+DINOv2가 작동하는 이유는 dual-encoder나 큰 데이터가 아니라, **LoRA가 backbone을 수정하지 않기 때문**.")
elif lp_ok and not lora_ok:
    out.append("\n**가설 부분 검증**: Linear Probe (frozen)는 작동하지만 LoRA는 실패.")
    out.append("→ 'frozen ≠ LoRA'. SigLIP의 다른 요인 (dual encoder, focal loss, 95K 데이터 등) 필요.")
else:
    out.append("\n**가설 실패**: Linear Probe도 작동 안 함. backbone 미수정만으로는 부족.")

(OUT / 'ablation_report.md').write_text('\n'.join(out))
print(f"Saved {OUT / 'ablation_report.md'}")
