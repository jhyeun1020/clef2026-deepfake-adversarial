"""
Phase 5: Analysis — main_comparison, statistics (bootstrap CI, p-value), attack_success.
"""
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ranksums

sys.path.insert(0, str(Path(__file__).parent))
from common_utils import log, OUT, ALL_MODEL_NAMES


def auc_bootstrap(y, scores, n_iters=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_iters):
        idx = rng.integers(0, n, n)
        ys = y[idx]; ss = scores[idx]
        if len(np.unique(ys)) < 2: continue
        try: aucs.append(roc_auc_score(ys, ss))
        except Exception: pass
    if not aucs: return None, None
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def tnr_at_tpr(y, scores, tpr_target=0.95):
    """Returns TNR (true negative rate) when TPR is tpr_target.
    y: 1 = positive (adversarial), 0 = negative."""
    fpr, tpr, _ = roc_curve(y, scores)
    # Find smallest TPR >= target
    idx = np.argmax(tpr >= tpr_target)
    return float(1 - fpr[idx])


def analyze_one_model(name):
    csv = OUT / f'purification/{name}.csv'
    if not csv.exists():
        log(f"  No purification CSV for {name}")
        return None
    df = pd.read_csv(csv)

    real_d = df[df['category']=='real']['delta_logit'].values
    clean_d = df[df['category']=='clean_fake']['delta_logit'].values
    jpeg_d = df[df['category']=='jpeg_q80']['delta_logit'].values
    adv_d = df[df['category']=='adversarial']['delta_logit'].values

    if len(adv_d) == 0:
        log(f"  {name}: no adversarial measurements")
        return None

    margin = float(np.mean(adv_d) - np.mean(real_d))

    out = {
        'model': name,
        'real_delta_mean': float(np.mean(real_d)),
        'real_delta_std': float(np.std(real_d)),
        'clean_delta_mean': float(np.mean(clean_d)) if len(clean_d) else None,
        'jpeg_delta_mean': float(np.mean(jpeg_d)) if len(jpeg_d) else None,
        'adv_delta_mean': float(np.mean(adv_d)),
        'adv_delta_std': float(np.std(adv_d)),
        'margin_adv_minus_real': margin,
    }

    # AUROC: adv vs clean / jpeg / real / all (positive = adv)
    for neg_name, neg in [('clean', clean_d), ('jpeg', jpeg_d), ('real', real_d)]:
        if len(neg) == 0: continue
        y = np.concatenate([np.ones(len(adv_d)), np.zeros(len(neg))])
        s = np.concatenate([adv_d, neg])
        try:
            out[f'auc_vs_{neg_name}'] = float(roc_auc_score(y, s))
            ci_lo, ci_hi = auc_bootstrap(y, s)
            out[f'auc_vs_{neg_name}_ci_lo'] = ci_lo
            out[f'auc_vs_{neg_name}_ci_hi'] = ci_hi
            out[f'tnr95_vs_{neg_name}'] = tnr_at_tpr(y, s, 0.95)
        except Exception as e:
            log(f"  {name} AUROC vs {neg_name} fail: {e}")

    # all (combined non-adv = real + clean + jpeg)
    non_adv = np.concatenate([real_d, clean_d, jpeg_d])
    if len(non_adv) > 0:
        y = np.concatenate([np.ones(len(adv_d)), np.zeros(len(non_adv))])
        s = np.concatenate([adv_d, non_adv])
        try:
            out['auc_vs_all'] = float(roc_auc_score(y, s))
            ci_lo, ci_hi = auc_bootstrap(y, s)
            out['auc_vs_all_ci_lo'] = ci_lo
            out['auc_vs_all_ci_hi'] = ci_hi
            out['tnr95_vs_all'] = tnr_at_tpr(y, s, 0.95)
        except Exception as e:
            log(f"  {name} AUROC vs all fail: {e}")

    # Wilcoxon rank-sum
    if len(clean_d) > 0:
        try:
            stat, pval = ranksums(adv_d, clean_d)
            out['wilcoxon_pvalue_vs_clean'] = float(pval)
        except Exception:
            pass

    return out


def run():
    log("=" * 60); log("Phase 5: Analysis"); log("=" * 60)
    out_dir = OUT / 'analysis'; out_dir.mkdir(exist_ok=True)

    rows = []
    for name in ALL_MODEL_NAMES:
        log(f"\n--- {name} ---")
        r = analyze_one_model(name)
        if r is None: continue
        rows.append(r)
        log(f"  margin: {r['margin_adv_minus_real']:+.2f}  "
            f"AUROC vs all: {r.get('auc_vs_all', 'N/A')}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / 'main_comparison.csv', index=False)
        log(f"\nSaved main_comparison.csv ({len(df)} models)")

    # Write summary report
    report = ["# Paper Results — Purification Adversarial Detection\n"]
    report.append("## Main Comparison Table\n")
    report.append("| Model | margin | AUROC vs clean | AUROC vs jpeg | AUROC vs all | TNR@TPR95 vs all |")
    report.append("|---|---|---|---|---|---|")
    for r in rows:
        report.append(f"| {r['model']} | {r['margin_adv_minus_real']:+.2f} | "
                      f"{r.get('auc_vs_clean','N/A')} | "
                      f"{r.get('auc_vs_jpeg','N/A')} | "
                      f"{r.get('auc_vs_all','N/A')} | "
                      f"{r.get('tnr95_vs_all','N/A')} |")
    report.append("\n## delta_logit per category (mean ± std)\n")
    report.append("| Model | real | clean_fake | jpeg_q80 | adv |")
    report.append("|---|---|---|---|---|")
    for r in rows:
        report.append(f"| {r['model']} | {r['real_delta_mean']:+.2f}±{r['real_delta_std']:.2f} | "
                      f"{r.get('clean_delta_mean', np.nan):+.2f} | "
                      f"{r.get('jpeg_delta_mean', np.nan):+.2f} | "
                      f"{r['adv_delta_mean']:+.2f}±{r['adv_delta_std']:.2f} |")
    (out_dir / 'paper_results_report.md').write_text('\n'.join(report))
    log("Saved paper_results_report.md")


if __name__ == '__main__':
    try: run()
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"FAIL Phase 5: {e}")
    log("Phase 5 DONE")
