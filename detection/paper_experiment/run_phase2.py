"""
Phase 2: 6-model val performance check.
Saves val_performance.csv (val probs for each model).
"""
import os, sys, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common_utils import (
    log, BASE, OUT, FF_VAL, DEVICE, FAKE_TYPES,
    gather_paths, batch_prob_fake,
    ALL_MODEL_NAMES, load_by_name, unload,
)


def run():
    log("=" * 60); log("Phase 2: 6-model val performance"); log("=" * 60)
    out_csv = OUT / 'val_performance.csv'
    out_probs = OUT / 'val_probs.parquet'
    if out_probs.exists():
        log("Already computed. Skipping.")
        return

    val_paths, val_labels = gather_paths(FF_VAL)
    log(f"Val: {len(val_paths)} ({sum(1 for l in val_labels if l == 0)} real, "
        f"{sum(1 for l in val_labels if l == 1)} fake)")

    # Source labels (real / Deepfakes / ...)
    sources = []
    for p in val_paths:
        sources.append(p.parent.name)

    df = pd.DataFrame({'image_path': [str(p) for p in val_paths],
                        'image_id': [p.stem for p in val_paths],
                        'source': sources, 'label': val_labels})

    summary_rows = []
    for name in ALL_MODEL_NAMES:
        log(f"\n--- {name} ---")
        try:
            t0 = time.time()
            mp = load_by_name(name)
            probs = batch_prob_fake(mp, val_paths, batch_size=32)
            df[f'prob_{name}'] = probs
            log(f"  Inference: {time.time()-t0:.0f}s, mean prob_fake={probs.mean():.3f}")

            # Metrics
            y = np.array(val_labels)
            try:
                overall_auc = roc_auc_score(y, probs)
            except Exception:
                overall_auc = np.nan
            real_idx = (y == 0); fake_idx = (y == 1)
            real_fp_rate = (probs[real_idx] > 0.5).mean() if real_idx.sum() > 0 else 0
            det_rate = (probs[fake_idx] > 0.5).mean() if fake_idx.sum() > 0 else 0
            row = {'model': name, 'overall_auc': overall_auc,
                   'real_fp_rate': real_fp_rate, 'fake_detect_rate': det_rate,
                   'mean_prob_real': probs[real_idx].mean() if real_idx.sum() > 0 else np.nan,
                   'mean_prob_fake': probs[fake_idx].mean() if fake_idx.sum() > 0 else np.nan}
            # Per-manipulation
            for ft in FAKE_TYPES:
                ft_idx = np.array([s == ft for s in sources])
                if ft_idx.sum() == 0:
                    continue
                y_pair = np.concatenate([y[real_idx], y[ft_idx]])
                p_pair = np.concatenate([probs[real_idx], probs[ft_idx]])
                try:
                    row[f'auc_vs_{ft}'] = roc_auc_score(y_pair, p_pair)
                except Exception:
                    row[f'auc_vs_{ft}'] = np.nan
            summary_rows.append(row)
            log(f"  Overall AUROC: {overall_auc:.4f}  Real FP: {real_fp_rate:.3f}  Fake Det: {det_rate:.3f}")

            unload(mp)
        except Exception as e:
            import traceback; traceback.print_exc()
            log(f"FAIL {name}: {e}")
            summary_rows.append({'model': name, 'overall_auc': np.nan, 'error': str(e)})
            df[f'prob_{name}'] = np.nan

    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    df.to_parquet(out_probs, index=False)
    log(f"Saved {out_csv.name} and {out_probs.name}")


if __name__ == '__main__':
    try: run()
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"FAIL Phase 2: {e}")
    log("Phase 2 DONE")
