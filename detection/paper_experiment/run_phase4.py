"""
Phase 4: Purification (median3) measurement for each model.
"""
import os, sys, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from common_utils import (
    log, BASE, OUT, FF_VAL, DEVICE, FAKE_TYPES,
    prob_fake, forward_model,
    ALL_MODEL_NAMES, load_by_name, unload,
)


SAFE_EPS = 1e-6
def logit_of(p):
    p = np.clip(p, SAFE_EPS, 1 - SAFE_EPS)
    return np.log(p / (1 - p))


def measure_category(model_pack, paths, category, batch_size=32):
    """For each path, compute prob_orig + prob_med3 + delta_logit."""
    BATCH = batch_size
    rows = []
    for bi in range(0, len(paths), BATCH):
        chunk = paths[bi:bi+BATCH]
        pils = [Image.open(p).convert('RGB').resize((256,256), Image.LANCZOS) for p in chunk]
        pils_m3 = [pi.filter(ImageFilter.MedianFilter(size=3)) for pi in pils]
        x = torch.stack([T.ToTensor()(pi) for pi in pils]).to(DEVICE)
        x_m3 = torch.stack([T.ToTensor()(pi) for pi in pils_m3]).to(DEVICE)
        with torch.no_grad():
            p_orig = prob_fake(model_pack, x).cpu().numpy()
            p_med3 = prob_fake(model_pack, x_m3).cpu().numpy()
            log_orig = forward_model(model_pack, x).cpu().numpy()
            log_med3 = forward_model(model_pack, x_m3).cpu().numpy()
        for i, p in enumerate(chunk):
            rows.append({
                'image_id': Path(p).stem,
                'category': category,
                'logit_original': float(log_orig[i]),
                'logit_purified': float(log_med3[i]),
                'delta_logit':    float(logit_of(p_med3[i]) - logit_of(p_orig[i])),
                'prob_original':  float(p_orig[i]),
                'prob_purified':  float(p_med3[i]),
            })
    return rows


def run():
    log("=" * 60); log("Phase 4: Purification (median3)"); log("=" * 60)
    purif_dir = OUT / 'purification'
    purif_dir.mkdir(exist_ok=True)

    # Real (val/original)
    real_paths = sorted((FF_VAL / 'original').glob('*.png'))
    log(f"Real: {len(real_paths)}")

    # Common fakes (clean)
    common_paths = [Path(p) for p in pd.read_csv(OUT / 'common_fakes.txt', header=None)[0].tolist()]
    log(f"Clean fakes: {len(common_paths)}")

    # JPEG Q80
    jpeg_paths = sorted((OUT / 'jpeg_q80').glob('*.png'))
    log(f"JPEG Q80: {len(jpeg_paths)}")

    for name in ALL_MODEL_NAMES:
        out_csv = purif_dir / f'{name}.csv'
        if out_csv.exists():
            log(f"[{name}] Already done, skip.")
            continue
        log(f"\n--- {name} ---")
        try:
            mp = load_by_name(name)
            adv_paths = sorted((OUT / f'adversarial/{name}').glob('*.png'))
            log(f"  Adv: {len(adv_paths)}")
            if len(adv_paths) == 0:
                log(f"  No adv for {name}, skip purification.")
                unload(mp); continue

            rows = []
            t0 = time.time()
            rows.extend(measure_category(mp, real_paths, 'real'))
            log(f"  real done: {time.time()-t0:.0f}s")
            t0 = time.time()
            rows.extend(measure_category(mp, common_paths, 'clean_fake'))
            log(f"  clean_fake done: {time.time()-t0:.0f}s")
            t0 = time.time()
            rows.extend(measure_category(mp, jpeg_paths, 'jpeg_q80'))
            log(f"  jpeg done: {time.time()-t0:.0f}s")
            t0 = time.time()
            rows.extend(measure_category(mp, adv_paths, 'adversarial'))
            log(f"  adversarial done: {time.time()-t0:.0f}s")
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            log(f"  Saved {out_csv.name} ({len(rows)} rows)")
            unload(mp)
        except Exception as e:
            import traceback; traceback.print_exc()
            log(f"FAIL {name}: {e}")
        torch.cuda.empty_cache(); gc.collect()


if __name__ == '__main__':
    try: run()
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"FAIL Phase 4: {e}")
    log("Phase 4 DONE")
