"""
Phase 7: Test set evaluation. Repeat Phases 2-5 on ff_frames/test/.
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
from scipy.stats import ranksums

sys.path.insert(0, str(Path(__file__).parent))
from common_utils import (
    log, BASE, OUT, FF_TEST, DEVICE, FAKE_TYPES,
    gather_paths, batch_prob_fake, apply_jpeg_q80,
    forward_model, prob_fake,
    ALL_MODEL_NAMES, load_by_name, unload,
)


EPS = 2 / 255; ALPHA = 0.5 / 255; STEPS = 40
SAFE = 1e-6
def logit_of(p):
    p = np.clip(p, SAFE, 1-SAFE)
    return np.log(p/(1-p))


def step1_test_perf():
    log("--- Test Step 1: 6-model test performance ---")
    test_paths, test_labels = gather_paths(FF_TEST)
    df = pd.DataFrame({'image_path': [str(p) for p in test_paths],
                        'image_id': [p.stem for p in test_paths],
                        'source': [p.parent.name for p in test_paths],
                        'label': test_labels})
    out_parquet = OUT / 'test_evaluation/test_probs.parquet'
    if out_parquet.exists():
        log("test_probs.parquet exists, skip"); return
    for name in ALL_MODEL_NAMES:
        log(f"  inferring {name}...")
        try:
            mp = load_by_name(name)
            probs = batch_prob_fake(mp, test_paths)
            df[f'prob_{name}'] = probs
            log(f"    mean prob = {probs.mean():.3f}")
            unload(mp)
        except Exception as e:
            log(f"    FAIL {name}: {e}")
            df[f'prob_{name}'] = np.nan
        torch.cuda.empty_cache(); gc.collect()
    df.to_parquet(out_parquet, index=False)
    log(f"Saved {out_parquet.name}")


def step2_common_fakes():
    log("--- Test Step 2: Common fakes selection ---")
    out_csv = OUT / 'test_evaluation/test_common_fakes.txt'
    if out_csv.exists():
        return pd.read_csv(out_csv, header=None)[0].tolist()
    df = pd.read_parquet(OUT / 'test_evaluation/test_probs.parquet')
    fake_df = df[df['label']==1].copy()
    cols = [f'prob_{m}' for m in ALL_MODEL_NAMES if f'prob_{m}' in df.columns]
    def count(thr):
        m = (fake_df[cols] > thr).all(axis=1)
        return m.sum(), fake_df[m]['image_path'].tolist()
    n5, ps = count(0.5)
    log(f"  All > 0.5: {n5}")
    if n5 < 500:
        n4, ps = count(0.4)
        log(f"  All > 0.4: {n4}")
        if n4 < 300:
            agree = (fake_df[cols] > 0.5).sum(axis=1)
            ps = fake_df[agree >= 5]['image_path'].tolist()
            log(f"  ≥5 of 6 > 0.5: {len(ps)}")
    pd.Series(ps).to_csv(out_csv, header=False, index=False)
    log(f"Selected: {len(ps)}")
    return ps


def step3_make_jpeg(common_paths):
    out_dir = OUT / 'test_evaluation/jpeg_q80'; out_dir.mkdir(parents=True, exist_ok=True)
    for p in common_paths:
        p = Path(p)
        out_p = out_dir / f"{p.stem}.png"
        if out_p.exists(): continue
        try:
            img = Image.open(p).convert('RGB').resize((256,256), Image.LANCZOS)
            apply_jpeg_q80(img).save(out_p)
        except Exception as e:
            log(f"  jpeg fail {p.stem}: {e}")


def pgd(model_pack, x_clean):
    x = x_clean.clone().detach()
    delta = torch.empty_like(x).uniform_(-EPS, EPS)
    delta = (x + delta).clamp(0, 1) - x
    delta = delta.detach().requires_grad_(True)
    for _ in range(STEPS):
        x_adv = x + delta
        logits = forward_model(model_pack, x_adv)
        loss = logits.sum()
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta - ALPHA * grad.sign()
            delta = delta.clamp(-EPS, EPS)
            delta = (x + delta).clamp(0, 1) - x
        delta = delta.detach().requires_grad_(True)
    return (x + delta).clamp(0, 1).detach()


def step4_generate_adv(common_paths):
    log("--- Test Step 4: Generate adv per model ---")
    for name in ALL_MODEL_NAMES:
        adv_dir = OUT / f'test_evaluation/adversarial/{name}'
        adv_dir.mkdir(parents=True, exist_ok=True)
        existing = set(p.stem for p in adv_dir.glob('*.png'))
        todo = [p for p in common_paths if Path(p).stem not in existing]
        if not todo:
            log(f"  [{name}] all done"); continue
        log(f"  [{name}] {len(todo)} adv...")
        try:
            mp = load_by_name(name)
            BATCH = 8
            for bi in range(0, len(todo), BATCH):
                chunk = todo[bi:bi+BATCH]
                pils = [Image.open(p).convert('RGB').resize((256,256), Image.LANCZOS) for p in chunk]
                x = torch.stack([T.ToTensor()(pi) for pi in pils]).to(DEVICE)
                try:
                    x_adv = pgd(mp, x)
                except Exception as e:
                    log(f"    pgd fail batch {bi}: {e}"); continue
                for j, p in enumerate(chunk):
                    T.ToPILImage()(x_adv[j].cpu()).save(adv_dir / f"{Path(p).stem}.png")
            unload(mp)
        except Exception as e:
            log(f"  FAIL {name}: {e}")
        torch.cuda.empty_cache(); gc.collect()


def measure(model_pack, paths, category):
    rows = []; BATCH = 32
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


def step5_purification_and_analysis():
    log("--- Test Step 5: Purification + analysis ---")
    real_paths = sorted((FF_TEST / 'original').glob('*.png'))
    common_paths = [Path(p) for p in pd.read_csv(OUT / 'test_evaluation/test_common_fakes.txt', header=None)[0].tolist()]
    jpeg_paths = sorted((OUT / 'test_evaluation/jpeg_q80').glob('*.png'))

    purif_dir = OUT / 'test_evaluation/purification'; purif_dir.mkdir(exist_ok=True)
    summary_rows = []
    for name in ALL_MODEL_NAMES:
        out_csv = purif_dir / f'{name}.csv'
        if out_csv.exists():
            log(f"  [{name}] purification CSV exists")
        else:
            log(f"  [{name}] measuring...")
            try:
                mp = load_by_name(name)
                adv_paths = sorted((OUT / f'test_evaluation/adversarial/{name}').glob('*.png'))
                if len(adv_paths) == 0:
                    log(f"    no adv for {name}, skip"); unload(mp); continue
                rs = measure(mp, real_paths, 'real')
                rs += measure(mp, common_paths, 'clean_fake')
                rs += measure(mp, jpeg_paths, 'jpeg_q80')
                rs += measure(mp, adv_paths, 'adversarial')
                pd.DataFrame(rs).to_csv(out_csv, index=False)
                unload(mp)
            except Exception as e:
                import traceback; traceback.print_exc()
                log(f"  FAIL {name}: {e}"); continue
            torch.cuda.empty_cache(); gc.collect()

        # Analysis
        try:
            df = pd.read_csv(out_csv)
            real_d = df[df['category']=='real']['delta_logit'].values
            clean_d = df[df['category']=='clean_fake']['delta_logit'].values
            jpeg_d = df[df['category']=='jpeg_q80']['delta_logit'].values
            adv_d = df[df['category']=='adversarial']['delta_logit'].values
            if len(adv_d) == 0: continue
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
            row = {'model': name,
                   'real_delta_mean': float(np.mean(real_d)),
                   'adv_delta_mean': float(np.mean(adv_d)),
                   'margin': float(np.mean(adv_d) - np.mean(real_d)),
                   'auc_vs_all': auc, 'tnr95_vs_all': tnr95}
            if len(clean_d) > 0:
                y = np.concatenate([np.ones(len(adv_d)), np.zeros(len(clean_d))])
                s = np.concatenate([adv_d, clean_d])
                try: row['auc_vs_clean'] = roc_auc_score(y, s)
                except: pass
                try:
                    _, p = ranksums(adv_d, clean_d)
                    row['wilcoxon_p'] = float(p)
                except: pass
            if len(jpeg_d) > 0:
                y = np.concatenate([np.ones(len(adv_d)), np.zeros(len(jpeg_d))])
                s = np.concatenate([adv_d, jpeg_d])
                try: row['auc_vs_jpeg'] = roc_auc_score(y, s)
                except: pass
            summary_rows.append(row)
        except Exception as e:
            log(f"  Analysis fail {name}: {e}")

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(OUT / 'test_evaluation/test_comparison.csv', index=False)
        log("Saved test_comparison.csv")

        # Test report
        report = ["# Test Set Final Evaluation\n"]
        report.append("| Model | margin | AUROC vs clean | AUROC vs jpeg | AUROC vs all | TNR@TPR95 | Wilcoxon p |")
        report.append("|---|---|---|---|---|---|---|")
        for r in summary_rows:
            report.append(f"| {r['model']} | {r.get('margin', np.nan):+.2f} | "
                          f"{r.get('auc_vs_clean', np.nan):.3f} | "
                          f"{r.get('auc_vs_jpeg', np.nan):.3f} | "
                          f"{r.get('auc_vs_all', np.nan):.3f} | "
                          f"{r.get('tnr95_vs_all', np.nan):.3f} | "
                          f"{r.get('wilcoxon_p', np.nan):.2e} |")
        (OUT / 'test_evaluation/test_report.md').write_text('\n'.join(report))
        log("Saved test_report.md")


def run():
    log("=" * 60); log("Phase 7: Test set evaluation"); log("=" * 60)
    (OUT / 'test_evaluation').mkdir(exist_ok=True)
    try: step1_test_perf()
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"FAIL test step 1: {e}")
    try:
        common = step2_common_fakes()
        step3_make_jpeg(common)
        step4_generate_adv(common)
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"FAIL test step 2-4: {e}")
    try: step5_purification_and_analysis()
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"FAIL test step 5: {e}")


if __name__ == '__main__':
    try: run()
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"FAIL Phase 7: {e}")
    log("Phase 7 DONE")
