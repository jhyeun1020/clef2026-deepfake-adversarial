"""
Phase 6: ε별 분석 (eps=2/255, 4/255, 8/255).
Best backbone-preserving model (Linear Probe or LoRA — auto select by AUROC).
"""
import os, sys, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from common_utils import (
    log, OUT, DEVICE,
    prob_fake, forward_model,
    load_by_name, unload,
)


def select_best_model():
    """Phase 5의 main_comparison에서 Linear Probe vs LoRA 비교 → 더 좋은 AUROC 선택."""
    main = pd.read_csv(OUT / 'analysis/main_comparison.csv')
    candidates = main[main['model'].isin(['linear_probe', 'lora'])]
    if len(candidates) == 0: return 'linear_probe'
    best = candidates.loc[candidates['auc_vs_all'].idxmax()]
    return best['model']


def pgd_eps(model_pack, x_clean, eps_val, steps=40):
    alpha = eps_val / 10
    x = x_clean.clone().detach()
    delta = torch.empty_like(x).uniform_(-eps_val, eps_val)
    delta = (x + delta).clamp(0, 1) - x
    delta = delta.detach().requires_grad_(True)
    for _ in range(steps):
        x_adv = x + delta
        logits = forward_model(model_pack, x_adv)
        loss = logits.sum()
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta - alpha * grad.sign()
            delta = delta.clamp(-eps_val, eps_val)
            delta = (x + delta).clamp(0, 1) - x
        delta = delta.detach().requires_grad_(True)
    return (x + delta).clamp(0, 1).detach()


SAFE = 1e-6
def logit_of(p):
    p = np.clip(p, SAFE, 1-SAFE)
    return np.log(p/(1-p))


def measure_with_model(model_pack, paths, category):
    rows = []
    BATCH = 32
    for bi in range(0, len(paths), BATCH):
        chunk = paths[bi:bi+BATCH]
        pils = [Image.open(p).convert('RGB').resize((256,256), Image.LANCZOS) for p in chunk]
        pils_m3 = [pi.filter(ImageFilter.MedianFilter(size=3)) for pi in pils]
        x = torch.stack([T.ToTensor()(pi) for pi in pils]).to(DEVICE)
        x_m3 = torch.stack([T.ToTensor()(pi) for pi in pils_m3]).to(DEVICE)
        with torch.no_grad():
            p_orig = prob_fake(model_pack, x).cpu().numpy()
            p_m3 = prob_fake(model_pack, x_m3).cpu().numpy()
        for i, p in enumerate(chunk):
            rows.append({'image_id': Path(p).stem, 'category': category,
                         'prob_original': float(p_orig[i]), 'prob_purified': float(p_m3[i]),
                         'delta_logit': float(logit_of(p_m3[i]) - logit_of(p_orig[i]))})
    return rows


def run():
    log("=" * 60); log("Phase 6: ε별 분석"); log("=" * 60)
    out_dir = OUT / 'epsilon_analysis'; out_dir.mkdir(exist_ok=True)

    name = select_best_model()
    log(f"Selected model: {name}")

    common_paths = [Path(p) for p in pd.read_csv(OUT / 'common_fakes.txt', header=None)[0].tolist()]
    real_paths = sorted((Path('/home/s20204116/GEN_AI_GTG/ff_frames/val/original')).glob('*.png'))
    jpeg_paths = sorted((OUT / 'jpeg_q80').glob('*.png'))

    EPS_LIST = [(2, 2/255), (4, 4/255), (8, 8/255)]

    # eps=2 already done in Phase 3 — reuse
    for eps_label, eps_val in EPS_LIST:
        adv_dir = out_dir / f'adversarial_eps{eps_label}'
        if eps_label == 2 and (OUT / f'adversarial/{name}').exists():
            # symlink/use existing eps=2 adv
            existing = sorted((OUT / f'adversarial/{name}').glob('*.png'))
            if len(existing) > 0:
                log(f"  eps=2: reusing Phase 3 adv ({len(existing)} images)")
                continue
        adv_dir.mkdir(parents=True, exist_ok=True)
        if len(list(adv_dir.glob('*.png'))) >= len(common_paths) - 5:
            log(f"  eps={eps_label}: already generated, skip")
            continue
        log(f"  Generating adv for eps={eps_label}/255 ...")
        mp = load_by_name(name)
        BATCH = 8
        for bi in range(0, len(common_paths), BATCH):
            chunk = common_paths[bi:bi+BATCH]
            pils = [Image.open(p).convert('RGB').resize((256,256), Image.LANCZOS) for p in chunk]
            x = torch.stack([T.ToTensor()(pi) for pi in pils]).to(DEVICE)
            try:
                x_adv = pgd_eps(mp, x, eps_val, 40)
            except Exception as e:
                log(f"    fail batch {bi}: {e}"); continue
            for j, p in enumerate(chunk):
                T.ToPILImage()(x_adv[j].cpu()).save(adv_dir / f"{Path(p).stem}.png")
        unload(mp)
        torch.cuda.empty_cache(); gc.collect()

    # Measure purification per eps
    rows = []
    for eps_label, _ in EPS_LIST:
        if eps_label == 2:
            adv_dir = OUT / f'adversarial/{name}'
        else:
            adv_dir = out_dir / f'adversarial_eps{eps_label}'
        adv_paths = sorted(adv_dir.glob('*.png'))
        if len(adv_paths) == 0: continue
        log(f"\n--- Measuring eps={eps_label} ({len(adv_paths)} adv) ---")
        mp = load_by_name(name)
        rs = measure_with_model(mp, real_paths, 'real')
        rs += measure_with_model(mp, common_paths, 'clean_fake')
        rs += measure_with_model(mp, jpeg_paths, 'jpeg_q80')
        rs += measure_with_model(mp, adv_paths, 'adversarial')
        for r in rs: r['eps'] = eps_label
        rows.extend(rs)
        unload(mp)
        torch.cuda.empty_cache(); gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'epsilon_purification.csv', index=False)
    log(f"Saved epsilon_purification.csv ({len(df)} rows)")

    # Report
    report = [f"# ε Analysis ({name})\n"]
    report.append("| ε | n_adv | Real Δ | Adv Δ | margin | AUROC vs all | TNR@TPR95 |")
    report.append("|---|---|---|---|---|---|---|")
    for eps_label, _ in EPS_LIST:
        sub = df[df['eps']==eps_label]
        if len(sub)==0: continue
        real_d = sub[sub['category']=='real']['delta_logit'].values
        adv_d = sub[sub['category']=='adversarial']['delta_logit'].values
        clean_d = sub[sub['category']=='clean_fake']['delta_logit'].values
        jpeg_d = sub[sub['category']=='jpeg_q80']['delta_logit'].values
        if len(adv_d) == 0: continue
        margin = float(np.mean(adv_d) - np.mean(real_d))
        non_adv = np.concatenate([real_d, clean_d, jpeg_d])
        y = np.concatenate([np.ones(len(adv_d)), np.zeros(len(non_adv))])
        s = np.concatenate([adv_d, non_adv])
        try:
            auc = roc_auc_score(y, s)
            fpr, tpr, _ = roc_curve(y, s)
            idx = np.argmax(tpr >= 0.95)
            tnr95 = float(1 - fpr[idx])
        except Exception:
            auc = np.nan; tnr95 = np.nan
        report.append(f"| {eps_label}/255 | {len(adv_d)} | "
                      f"{np.mean(real_d):+.2f} | {np.mean(adv_d):+.2f} | "
                      f"{margin:+.2f} | {auc:.3f} | {tnr95:.3f} |")

    (out_dir / 'epsilon_report.md').write_text('\n'.join(report))
    log("Saved epsilon_report.md")


if __name__ == '__main__':
    try: run()
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"FAIL Phase 6: {e}")
    log("Phase 6 DONE")
