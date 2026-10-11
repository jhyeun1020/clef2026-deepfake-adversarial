"""
Phase 3: Select common fakes + generate own-target adversarial for each model.
"""
import os, sys, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from common_utils import (
    log, BASE, OUT, FF_VAL, DEVICE, FAKE_TYPES,
    gather_paths, batch_prob_fake, apply_jpeg_q80,
    forward_model, prob_fake,
    ALL_MODEL_NAMES, load_by_name, unload,
)

EPS = 2 / 255
ALPHA = 0.5 / 255
STEPS = 40


def select_common_fakes():
    """6개 모델 모두 prob > 0.5인 fake 이미지 선택. fallback 룰 적용."""
    log("=" * 60); log("Phase 3.1: Select common fakes"); log("=" * 60)
    out_csv = OUT / 'common_fakes.txt'
    if out_csv.exists():
        log("Already selected.")
        return pd.read_csv(out_csv, header=None)[0].tolist()

    df = pd.read_parquet(OUT / 'val_probs.parquet')
    fake_df = df[df['label'] == 1].copy()
    cols = [f'prob_{m}' for m in ALL_MODEL_NAMES if f'prob_{m}' in df.columns]
    log(f"Available model columns: {cols}")
    log(f"Fake total: {len(fake_df)}")

    def count_above(thr):
        mask = (fake_df[cols] > thr).all(axis=1)
        return mask.sum(), fake_df[mask]['image_path'].tolist()

    n_all_at_05, paths = count_above(0.5)
    log(f"All models > 0.5: {n_all_at_05}")
    if n_all_at_05 >= 500:
        log("→ ≥500, OK.")
    else:
        n_all_at_04, paths = count_above(0.4)
        log(f"All models > 0.4: {n_all_at_04}")
        if n_all_at_04 >= 300:
            log("→ ≥300 at threshold 0.4, OK.")
        else:
            # 5 of 6 agreement at 0.5
            high = (fake_df[cols] > 0.5).sum(axis=1)
            mask = high >= 5
            paths = fake_df[mask]['image_path'].tolist()
            log(f"→ Fallback: ≥5 of 6 > 0.5 = {len(paths)}")

    log(f"Selected common fakes: {len(paths)}")
    pd.Series(paths).to_csv(out_csv, header=False, index=False)
    return paths


def make_jpeg_q80(common_paths):
    """common_fakes에 JPEG Q80 적용해서 디스크에 저장."""
    log("=" * 60); log("Phase 3.1.5: JPEG Q80 versions"); log("=" * 60)
    jpeg_dir = OUT / 'jpeg_q80'
    jpeg_dir.mkdir(exist_ok=True)
    n_done = 0
    for p in common_paths:
        p = Path(p)
        out_p = jpeg_dir / f"{p.stem}.png"
        if out_p.exists(): continue
        try:
            img = Image.open(p).convert('RGB').resize((256,256), Image.LANCZOS)
            j = apply_jpeg_q80(img)
            j.save(out_p)
            n_done += 1
        except Exception as e:
            log(f"  JPEG fail {p.stem}: {e}")
    log(f"JPEG Q80 saved: {n_done}")


def pgd_attack(model_pack, x_clean):
    """Push prob_fake → 0."""
    x = x_clean.clone().detach()
    delta = torch.empty_like(x).uniform_(-EPS, EPS)
    delta = (x + delta).clamp(0, 1) - x
    delta = delta.detach().requires_grad_(True)
    for _ in range(STEPS):
        x_adv = x + delta
        logits = forward_model(model_pack, x_adv)
        loss = logits.sum()  # minimize fake logit
        grad = torch.autograd.grad(loss, delta, allow_unused=False)[0]
        with torch.no_grad():
            delta = delta - ALPHA * grad.sign()
            delta = delta.clamp(-EPS, EPS)
            delta = (x + delta).clamp(0, 1) - x
        delta = delta.detach().requires_grad_(True)
    return (x + delta).clamp(0, 1).detach()


def generate_adv_for_model(name, common_paths):
    adv_dir = OUT / f'adversarial/{name}'
    adv_dir.mkdir(parents=True, exist_ok=True)
    existing = set(p.stem for p in adv_dir.glob('*.png'))
    todo = [p for p in common_paths if Path(p).stem not in existing]
    if not todo:
        log(f"[{name}] All {len(common_paths)} adv already generated, skip.")
        return None

    log(f"[{name}] Loading model...")
    try:
        mp = load_by_name(name)
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"FAIL load {name}: {e}")
        return None

    log(f"[{name}] Generating {len(todo)} adv (eps={EPS:.4f}, steps={STEPS})...")
    t0 = time.time()
    BATCH = 8
    success = 0
    probs_before, probs_after = [], []
    for bi in range(0, len(todo), BATCH):
        chunk = todo[bi:bi+BATCH]
        pils = [Image.open(p).convert('RGB').resize((256,256), Image.LANCZOS) for p in chunk]
        x = torch.stack([T.ToTensor()(p) for p in pils]).to(DEVICE)
        with torch.no_grad():
            p_orig = prob_fake(mp, x).cpu().numpy()
        try:
            x_adv = pgd_attack(mp, x)
        except Exception as e:
            log(f"  PGD fail batch {bi}: {e}")
            continue
        with torch.no_grad():
            p_adv = prob_fake(mp, x_adv).cpu().numpy()
        for j, p in enumerate(chunk):
            adv_pil = T.ToPILImage()(x_adv[j].cpu())
            adv_pil.save(adv_dir / f"{Path(p).stem}.png")
            probs_before.append(float(p_orig[j]))
            probs_after.append(float(p_adv[j]))
            if p_adv[j] < 0.5: success += 1
        if (bi // BATCH) % 50 == 0:
            log(f"  [{name}] {bi}/{len(todo)}  prob {np.mean(p_orig):.3f}→{np.mean(p_adv):.3f}")
    log(f"[{name}] Done {time.time()-t0:.0f}s. Success: {success}/{len(todo)} = {success/max(len(todo),1):.0%}")
    unload(mp)
    return {'model': name, 'n_images': len(todo),
            'success_count': success,
            'success_rate': success/max(len(todo),1),
            'mean_prob_before': float(np.mean(probs_before)) if probs_before else None,
            'mean_prob_after': float(np.mean(probs_after)) if probs_after else None}


def run():
    common = select_common_fakes()
    make_jpeg_q80(common)

    log("=" * 60); log("Phase 3.2: Generate adversarial per model"); log("=" * 60)
    summary = []
    for name in ALL_MODEL_NAMES:
        try:
            r = generate_adv_for_model(name, common)
            if r is not None: summary.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            log(f"FAIL {name}: {e}")
            summary.append({'model': name, 'error': str(e)})
        torch.cuda.empty_cache(); gc.collect()

    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(OUT / 'attack_success.csv', index=False)
        log(f"Saved attack_success.csv")


if __name__ == '__main__':
    try: run()
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"FAIL Phase 3: {e}")
    log("Phase 3 DONE")
