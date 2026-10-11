"""
import os
Mahalanobis Distance Full Pipeline (Overnight, unattended)
Phases 1-5 자동 연속 실행. 각 phase는 checkpoint 파일로 완료 확인.
"""
import os, sys, gc, io, time, pickle, json, traceback
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3')
os.environ.setdefault('HF_TOKEN', 'REMOVED_TOKEN_USE_ENV_VAR')

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from tqdm import tqdm
from PIL import Image

BASE = Path('/home/student/GEN_AI_GTG')
OUT = BASE / 'detection/mahalanobis'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'pilot').mkdir(exist_ok=True)

CLEAN_DIR    = BASE / 'generation/outputs/final_1002'
ADV_DIR      = BASE / 'generation/submit/third_sub/256'
REAL_DIR     = BASE / 'detection/real_reference/center'
TEST_DIR     = BASE / 'detection/ImageCLEF2026-DeepFakeDetection-Tes/Data/Images_Detection'
CROP_SCORES  = BASE / 'detection/crop_ablation/scores'
INST_PRED    = BASE / 'detection/instability/test_application/test_instability_predictions.parquet'

LOG_FILE = OUT / 'overnight_log.txt'
DEVICE = torch.device('cuda:0')

def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def checkpoint_exists(name):
    return (OUT / f'.checkpoint_{name}').exists()

def save_checkpoint(name):
    (OUT / f'.checkpoint_{name}').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))

t_start = time.time()

# ══════════════════════════════════════════════════════════════════════
# MODEL LOADING (shared across phases)
# ══════════════════════════════════════════════════════════════════════
log("SigLIP+DINOv2 모델 로드...")
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor
import importlib.util

HF_TOKEN = os.environ['HF_TOKEN']
REPO_ID = 'Bombek1/ai-image-detector-siglip-dinov2'
model_py = hf_hub_download(repo_id=REPO_ID, filename='model.py')
spec = importlib.util.spec_from_file_location('ai_det', model_py)
ai_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(ai_mod)
wt = hf_hub_download(repo_id=REPO_ID, filename='pytorch_model.pt')
ckpt = torch.load(wt, map_location='cpu', weights_only=False)
cfg = ckpt.get('config', {})
ens_model = ai_mod.create_model_with_lora(
    siglip_model_name=cfg.get('siglip_model', 'google/siglip2-so400m-patch14-384'),
    dinov2_model_name=cfg.get('dinov2_model', 'vit_large_patch14_dinov2.lvd142m'),
    image_size=cfg.get('image_size', 392))
ens_model.load_state_dict(ckpt['model_state_dict'])
ens_model.to(DEVICE).eval()
siglip_proc = AutoProcessor.from_pretrained('google/siglip2-so400m-patch14-384')

# Identify backbone blocks
# DINOv2 backbone: ens_model.dinov2 (VisionTransformer, .blocks = 24)
dinov2_backbone = ens_model.dinov2
dinov2_blocks = list(dinov2_backbone.blocks)
n_dino = len(dinov2_blocks)
n_sig = 0  # SigLIP은 PeftModel이라 block 접근 복잡 → DINOv2만 사용
log(f"  DINOv2 blocks: {n_dino}, SigLIP blocks: {n_sig}")

# Select 5 layers from DINOv2 (primary backbone for feature extraction)
if n_dino >= 5:
    layer_indices = [int(0.2*n_dino), int(0.4*n_dino), int(0.6*n_dino), int(0.8*n_dino), n_dino-1]
else:
    layer_indices = list(range(n_dino))
log(f"  Selected DINOv2 layers: {layer_indices}")

# Preprocessing
sig_mean = torch.tensor(siglip_proc.image_processor.image_mean).view(1,3,1,1).to(DEVICE)
sig_std  = torch.tensor(siglip_proc.image_processor.image_std).view(1,3,1,1).to(DEVICE)
sig_size = (siglip_proc.image_processor.size.get('height', 384),
            siglip_proc.image_processor.size.get('width', 384))
dino_mean = torch.tensor([.485,.456,.406]).view(1,3,1,1).to(DEVICE)
dino_std  = torch.tensor([.229,.224,.225]).view(1,3,1,1).to(DEVICE)

def extract_features_batch(pils, batch_size=32):
    """Extract DINOv2 intermediate CLS features for a list of PIL images.
    Returns dict[layer_idx] -> np.array (N, D)."""
    all_feats = {li: [] for li in layer_indices}
    hooks = []
    hook_outputs = {}

    def make_hook(li):
        def hook_fn(module, inp, out):
            # DINOv2 block output: Tensor (B, 785, 1024) — CLS at index 0
            t = out[0] if isinstance(out, tuple) else out
            hook_outputs[li] = t[:, 0, :].detach().cpu()  # CLS token
        return hook_fn

    for li in layer_indices:
        h = dinov2_blocks[li].register_forward_hook(make_hook(li))
        hooks.append(h)

    try:
        for i in range(0, len(pils), batch_size):
            chunk = pils[i:i+batch_size]
            imgs = torch.stack([T.ToTensor()(p.convert('RGB')) for p in chunk]).to(DEVICE)
            dino_inp = (F.interpolate(imgs, (392,392), mode='bilinear', align_corners=False) - dino_mean) / dino_std
            with torch.no_grad():
                _ = dinov2_backbone.forward_features(dino_inp)
            for li in layer_indices:
                all_feats[li].append(hook_outputs[li])
            hook_outputs.clear()
    finally:
        for h in hooks:
            h.remove()

    return {li: torch.cat(all_feats[li], dim=0).numpy() for li in layer_indices}

def apply_jpeg(img_bgr, quality):
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

# ══════════════════════════════════════════════════════════════════════
# PHASE 1: PILOT
# ══════════════════════════════════════════════════════════════════════
if not checkpoint_exists('phase1'):
    log("\n" + "="*70)
    log("PHASE 1: Pilot verification (400 images)")
    log("="*70)
    try:
        rng = np.random.RandomState(42)

        # Sample 100 images per class
        all_clean = sorted(CLEAN_DIR.glob('*.png'))
        sel = rng.choice(len(all_clean), 100, replace=False)
        clean_keys = [all_clean[i].stem for i in sorted(sel)]

        real_all = sorted(REAL_DIR.glob('*.png'))
        real_sel = rng.choice(len(real_all), 200, replace=False)
        real_paths_100 = [real_all[i] for i in real_sel[:100]]
        jpeg_paths_100 = [real_all[i] for i in real_sel[100:200]]

        # Load images
        def load_pils(paths):
            pils = []
            for p in paths:
                im = Image.open(p).convert('RGB')
                if im.size != (256,256): im = im.resize((256,256), Image.LANCZOS)
                pils.append(im)
            return pils

        pils_real = load_pils(real_paths_100)
        pils_adv = load_pils([ADV_DIR / f'{k}.png' for k in clean_keys])
        pils_clean = load_pils([CLEAN_DIR / f'{k}.png' for k in clean_keys])

        # JPEG Q70 real
        pils_jpeg = []
        for p in jpeg_paths_100:
            img = cv2.imread(str(p))
            if img.shape[:2] != (256,256): img = cv2.resize(img, (256,256))
            jimg = apply_jpeg(img, 70)
            pils_jpeg.append(Image.fromarray(cv2.cvtColor(jimg, cv2.COLOR_BGR2RGB)))

        log("  Feature extraction...")
        feats_real  = extract_features_batch(pils_real)
        feats_adv   = extract_features_batch(pils_adv)
        feats_clean = extract_features_batch(pils_clean)
        feats_jpeg  = extract_features_batch(pils_jpeg)

        feat_dim = feats_real[layer_indices[0]].shape[1]
        log(f"  Feature dim: {feat_dim}, n_layers: {len(layer_indices)}")

        # Gaussian fitting (4-class, tied LedoitWolf covariance)
        log("  Gaussian fitting (LedoitWolf)...")
        class_data = {'real': feats_real, 'adv': feats_adv, 'clean_fake': feats_clean, 'jpeg': feats_jpeg}
        class_names = list(class_data.keys())
        mu = {}; sigma_inv = {}

        for li in layer_indices:
            all_centered = []
            for cn in class_names:
                f = class_data[cn][li]
                mu[(cn, li)] = f.mean(axis=0)
                all_centered.append(f - mu[(cn, li)])
            centered = np.vstack(all_centered)
            try:
                lw = LedoitWolf().fit(centered)
                sigma_inv[li] = np.linalg.inv(lw.covariance_)
            except Exception:
                log(f"    ⚠ Layer {li}: LedoitWolf failed, using diagonal")
                var = centered.var(axis=0) + 1e-6
                sigma_inv[li] = np.diag(1.0 / var)

        # Mahalanobis distance
        def mahal_dist(x, mu_vec, sinv):
            d = x - mu_vec
            return float(d @ sinv @ d)

        def get_mahal_features(feats_dict, n):
            """Per image: distance to each class mean at each layer → n_layers × n_classes dim."""
            out = np.zeros((n, len(layer_indices) * len(class_names)))
            for li_idx, li in enumerate(layer_indices):
                for ci, cn in enumerate(class_names):
                    for i in range(n):
                        out[i, li_idx * len(class_names) + ci] = mahal_dist(
                            feats_dict[li][i], mu[(cn, li)], sigma_inv[li])
            return out

        M_real  = get_mahal_features(feats_real, 100)
        M_adv   = get_mahal_features(feats_adv, 100)
        M_clean = get_mahal_features(feats_clean, 100)
        M_jpeg  = get_mahal_features(feats_jpeg, 100)

        n_mahal_dim = M_real.shape[1]
        log(f"  Mahalanobis feature dim: {n_mahal_dim}")

        # Save pilot features
        rows = []
        for label, M, n in [('real', M_real, 100), ('adv', M_adv, 100),
                             ('clean_fake', M_clean, 100), ('jpeg', M_jpeg, 100)]:
            for i in range(n):
                row = {'label': label, 'idx': i}
                for j in range(n_mahal_dim):
                    row[f'mah_{j}'] = float(M[i, j])
                rows.append(row)
        pd.DataFrame(rows).to_parquet(OUT / 'pilot/pilot_features.parquet', index=False)

        # AUROC evaluation
        def auroc(neg, pos):
            if len(neg)==0 or len(pos)==0: return float('nan')
            n0, n1 = len(neg), len(pos)
            ranks = sp_stats.rankdata(np.concatenate([neg, pos]))
            return float((ranks[n0:].sum() - n1*(n1+1)/2) / (n0*n1))

        # Per-layer AUROC (using distance to real mean as score)
        auc_rows = []
        for li_idx, li in enumerate(layer_indices):
            ci_real = class_names.index('real')
            col = li_idx * len(class_names) + ci_real
            for task, neg_M, pos_M, neg_n, pos_n in [
                ('real_vs_adv', M_real, M_adv, 100, 100),
                ('real_vs_clean', M_real, M_clean, 100, 100),
                ('adv_vs_jpeg', M_jpeg, M_adv, 100, 100),
                ('adv_vs_clean', M_clean, M_adv, 100, 100),
            ]:
                neg_scores = neg_M[:neg_n, col]
                pos_scores = pos_M[:pos_n, col]
                auc_rows.append({'layer': li, 'task': task, 'auroc': round(auroc(neg_scores, pos_scores), 4)})

        # Multi-layer ensemble (LR)
        for task, label0, M0, n0, label1, M1, n1 in [
            ('real_vs_adv', 'real', M_real, 100, 'adv', M_adv, 100),
            ('adv_vs_jpeg', 'jpeg', M_jpeg, 100, 'adv', M_adv, 100),
            ('adv_vs_clean', 'clean', M_clean, 100, 'adv', M_adv, 100),
        ]:
            X = np.vstack([M0[:n0], M1[:n1]])
            y = np.array([0]*n0 + [1]*n1)
            try:
                proba = cross_val_predict(LogisticRegression(max_iter=2000), X, y,
                                          cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                          method='predict_proba')[:, 1]
                auc_val = roc_auc_score(y, proba)
            except Exception:
                auc_val = float('nan')
            auc_rows.append({'layer': 'ensemble_LR', 'task': task, 'auroc': round(auc_val, 4)})

        auc_df = pd.DataFrame(auc_rows)
        auc_df.to_csv(OUT / 'pilot/pilot_auroc.csv', index=False)

        # Key metrics
        auc_adv_vs_jpeg_ens = auc_df[(auc_df['layer']=='ensemble_LR') & (auc_df['task']=='adv_vs_jpeg')]['auroc'].values
        auc_adv_vs_jpeg = float(auc_adv_vs_jpeg_ens[0]) if len(auc_adv_vs_jpeg_ens) else 0
        auc_real_vs_adv_ens = auc_df[(auc_df['layer']=='ensemble_LR') & (auc_df['task']=='real_vs_adv')]['auroc'].values
        auc_real_vs_adv = float(auc_real_vs_adv_ens[0]) if len(auc_real_vs_adv_ens) else 0

        log(f"\n  Pilot AUROC:")
        for _, r in auc_df.iterrows():
            log(f"    layer={str(r['layer']):>12s}  task={r['task']:<16s}  auroc={r['auroc']:.4f}")

        # Pilot report
        if auc_adv_vs_jpeg > 0.85:
            verdict = '✓ Mahalanobis 작동, Phase 2 진행'
            proceed = True
        elif auc_adv_vs_jpeg > 0.75:
            verdict = '~ 부분 작동, Phase 2 신중히 진행'
            proceed = True
        else:
            verdict = '✗ Mahalanobis 실패'
            proceed = False
        log(f"\n  판정: {verdict} (adv_vs_jpeg ensemble AUROC={auc_adv_vs_jpeg:.4f})")

        with open(OUT / 'pilot/pilot_report.md', 'w') as f:
            f.write(f'# Phase 1: Pilot Report\n\n')
            f.write(f'DINOv2 layers: {layer_indices}, feat_dim: {feat_dim}\n')
            f.write(f'Covariance: LedoitWolf (tied)\n\n')
            f.write(f'## AUROC\n\n')
            try:
                f.write(auc_df.to_markdown(index=False))
            except Exception:
                f.write(auc_df.to_string())
            f.write(f'\n\n## 판정: **{verdict}**\n')
            f.write(f'- adv_vs_jpeg ensemble: {auc_adv_vs_jpeg:.4f}\n')
            f.write(f'- real_vs_adv ensemble: {auc_real_vs_adv:.4f}\n')

        # Save Gaussian params for reuse
        gaussian_params = {'mu': mu, 'sigma_inv': sigma_inv, 'layer_indices': layer_indices,
                           'class_names': class_names, 'feat_dim': feat_dim}
        with open(OUT / 'pilot/gaussian_params_pilot.pkl', 'wb') as f:
            pickle.dump(gaussian_params, f)

        save_checkpoint('phase1')
        log(f"  Phase 1 완료 ({time.time()-t_start:.0f}s)")

        if not proceed:
            log("⚠ Phase 1 실패 — Phase 2-4 건너뛰고 Phase 5로")
            save_checkpoint('phase2')
            save_checkpoint('phase3')
            save_checkpoint('phase4')
    except Exception as e:
        log(f"❌ Phase 1 실패: {e}")
        traceback.print_exc()
        with open(OUT / 'pilot/pilot_report.md', 'w') as f:
            f.write(f'# Phase 1: FAILED\n\n```\n{traceback.format_exc()}\n```\n')
        save_checkpoint('phase1')
        save_checkpoint('phase2')
        save_checkpoint('phase3')
        save_checkpoint('phase4')
else:
    log("Phase 1 이미 완료, skip")
    # Load gaussian params
    with open(OUT / 'pilot/gaussian_params_pilot.pkl', 'rb') as f:
        gaussian_params = pickle.load(f)
    mu = gaussian_params['mu']; sigma_inv = gaussian_params['sigma_inv']
    layer_indices = gaussian_params['layer_indices']
    class_names = gaussian_params['class_names']; feat_dim = gaussian_params['feat_dim']

# ══════════════════════════════════════════════════════════════════════
# PHASE 2: Full Gaussian Fitting
# ══════════════════════════════════════════════════════════════════════
if not checkpoint_exists('phase2'):
    log("\n" + "="*70)
    log("PHASE 2: Full Gaussian Fitting (~8,500 images)")
    log("="*70)
    try:
        rng2 = np.random.RandomState(2026)

        # Real: 3,350장 전체
        real_all_paths = sorted(REAL_DIR.glob('*.png'))
        log(f"  Real: {len(real_all_paths)}")
        pils_real_full = []
        for p in tqdm(real_all_paths, desc='load real', unit='img'):
            im = Image.open(p).convert('RGB')
            if im.size != (256,256): im = im.resize((256,256), Image.LANCZOS)
            pils_real_full.append(im)

        # Adversarial: 1002장 전체
        adv_stems = sorted(p.stem for p in ADV_DIR.glob('*.png'))
        pils_adv_full = []
        for s in tqdm(adv_stems, desc='load adv', unit='img'):
            im = Image.open(ADV_DIR / f'{s}.png').convert('RGB')
            pils_adv_full.append(im)
        log(f"  Adv: {len(pils_adv_full)}")

        # Clean fake: 1002장 전체
        pils_clean_full = []
        for s in tqdm(adv_stems, desc='load clean', unit='img'):
            p = CLEAN_DIR / f'{s}.png'
            if p.exists():
                im = Image.open(p).convert('RGB')
                if im.size != (256,256): im = im.resize((256,256), Image.LANCZOS)
                pils_clean_full.append(im)
        log(f"  Clean fake: {len(pils_clean_full)}")

        # JPEG real: 1000장 × Q50, Q70, Q80
        jpeg_sel = rng2.choice(len(real_all_paths), 1000, replace=False)
        pils_jpeg_full = []
        for q in [50, 70, 80]:
            for idx in tqdm(jpeg_sel, desc=f'jpeg Q{q}', unit='img'):
                img = cv2.imread(str(real_all_paths[idx]))
                if img.shape[:2] != (256,256): img = cv2.resize(img, (256,256))
                jimg = apply_jpeg(img, q)
                pils_jpeg_full.append(Image.fromarray(cv2.cvtColor(jimg, cv2.COLOR_BGR2RGB)))
        log(f"  JPEG real: {len(pils_jpeg_full)}")

        # Feature extraction (batch)
        log("  Feature extraction (this takes a while)...")
        feats_real_f  = extract_features_batch(pils_real_full, batch_size=64)
        feats_adv_f   = extract_features_batch(pils_adv_full, batch_size=64)
        feats_clean_f = extract_features_batch(pils_clean_full, batch_size=64)
        feats_jpeg_f  = extract_features_batch(pils_jpeg_full, batch_size=64)
        log("  Features extracted")

        # Free memory
        del pils_real_full, pils_adv_full, pils_clean_full, pils_jpeg_full
        gc.collect()

        # Gaussian fitting (LedoitWolf, tied)
        log("  Gaussian fitting (full)...")
        class_data_full = {'real': feats_real_f, 'adv': feats_adv_f,
                           'clean_fake': feats_clean_f, 'jpeg': feats_jpeg_f}
        mu_full = {}; sinv_full = {}
        for li in layer_indices:
            all_centered = []
            for cn in class_names:
                f = class_data_full[cn][li]
                mu_full[(cn, li)] = f.mean(axis=0)
                all_centered.append(f - mu_full[(cn, li)])
            centered = np.vstack(all_centered)
            try:
                lw = LedoitWolf().fit(centered)
                sinv_full[li] = np.linalg.inv(lw.covariance_)
                log(f"    Layer {li}: LedoitWolf OK (cond={np.linalg.cond(lw.covariance_):.1f})")
            except Exception as e:
                log(f"    Layer {li}: LedoitWolf failed ({e}), diagonal fallback")
                var = centered.var(axis=0) + 1e-6
                sinv_full[li] = np.diag(1.0 / var)

        gaussian_full = {'mu': mu_full, 'sigma_inv': sinv_full, 'layer_indices': layer_indices,
                         'class_names': class_names, 'feat_dim': feat_dim}
        with open(OUT / 'gaussian_params_full.pkl', 'wb') as f:
            pickle.dump(gaussian_full, f)

        # Update active params
        mu = mu_full; sigma_inv = sinv_full
        del feats_real_f, feats_adv_f, feats_clean_f, feats_jpeg_f, class_data_full
        gc.collect()

        save_checkpoint('phase2')
        log(f"  Phase 2 완료 ({time.time()-t_start:.0f}s)")
    except Exception as e:
        log(f"❌ Phase 2 실패: {e}")
        traceback.print_exc()
        save_checkpoint('phase2')
        # Fall back to pilot params
        if not mu:
            gp = pickle.load(open(OUT / 'pilot/gaussian_params_pilot.pkl', 'rb'))
            mu = gp['mu']; sigma_inv = gp['sigma_inv']
else:
    log("Phase 2 이미 완료, skip")
    if (OUT / 'gaussian_params_full.pkl').exists():
        gp = pickle.load(open(OUT / 'gaussian_params_full.pkl', 'rb'))
        mu = gp['mu']; sigma_inv = gp['sigma_inv']

# ══════════════════════════════════════════════════════════════════════
# PHASE 3: Test Set Full Inference
# ══════════════════════════════════════════════════════════════════════
if not checkpoint_exists('phase3'):
    log("\n" + "="*70)
    log("PHASE 3: Test set 24,404장 full inference")
    log("="*70)
    try:
        test_paths = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() == '.png')
        N_TEST = len(test_paths)
        log(f"  Test images: {N_TEST}")

        CHUNK = 500  # process in chunks to manage memory
        all_mahal_rows = []
        n_mahal_dim = len(layer_indices) * len(class_names)

        for chunk_start in range(0, N_TEST, CHUNK):
            chunk_end = min(chunk_start + CHUNK, N_TEST)
            chunk_paths = test_paths[chunk_start:chunk_end]
            pils = []
            for p in chunk_paths:
                im = Image.open(p).convert('RGB')
                if im.size != (256,256): im = im.resize((256,256), Image.LANCZOS)
                pils.append(im)

            feats = extract_features_batch(pils, batch_size=64)
            n = len(pils)

            for i in range(n):
                row = {'filename': chunk_paths[i].name}
                min_dist = float('inf'); closest = ''
                for li_idx, li in enumerate(layer_indices):
                    for ci, cn in enumerate(class_names):
                        d = float((feats[li][i] - mu[(cn, li)]) @ sigma_inv[li] @ (feats[li][i] - mu[(cn, li)]))
                        col_name = f'mah_L{li}_{cn}'
                        row[col_name] = round(d, 4)
                        if d < min_dist:
                            min_dist = d; closest = cn
                row['closest_class'] = closest
                row['min_distance'] = round(min_dist, 4)
                all_mahal_rows.append(row)

            del pils, feats; gc.collect()
            if (chunk_end) % 5000 == 0 or chunk_end == N_TEST:
                log(f"  [{chunk_end}/{N_TEST}] ({time.time()-t_start:.0f}s)")

        test_df = pd.DataFrame(all_mahal_rows)
        # Add softmax probabilities
        for cn in class_names:
            cols = [f'mah_L{li}_{cn}' for li in layer_indices]
            test_df[f'avg_dist_{cn}'] = test_df[cols].mean(axis=1)
        dist_cols = [f'avg_dist_{cn}' for cn in class_names]
        neg_dists = -test_df[dist_cols].values
        exp_d = np.exp(neg_dists - neg_dists.max(axis=1, keepdims=True))
        probs = exp_d / exp_d.sum(axis=1, keepdims=True)
        for ci, cn in enumerate(class_names):
            test_df[f'prob_{cn}'] = probs[:, ci].round(4)

        test_df.to_parquet(OUT / 'test_features.parquet', index=False)
        log(f"  test_features.parquet 저장 ({len(test_df)} rows)")

        save_checkpoint('phase3')
        log(f"  Phase 3 완료 ({time.time()-t_start:.0f}s)")
    except Exception as e:
        log(f"❌ Phase 3 실패: {e}")
        traceback.print_exc()
        save_checkpoint('phase3')
else:
    log("Phase 3 이미 완료, skip")

# ══════════════════════════════════════════════════════════════════════
# PHASE 4: Ensemble with Instability
# ══════════════════════════════════════════════════════════════════════
if not checkpoint_exists('phase4'):
    log("\n" + "="*70)
    log("PHASE 4: Ensemble with instability method")
    log("="*70)
    try:
        test_df = pd.read_parquet(OUT / 'test_features.parquet')

        # Mahalanobis features for classification (avg_dist per class)
        mah_cols = [f'avg_dist_{cn}' for cn in class_names]

        # Load instability predictions (6,002장만 존재)
        inst_df = pd.read_parquet(INST_PRED)
        inst_df['stem'] = inst_df['filename'].apply(lambda x: Path(x).stem)
        test_df['stem'] = test_df['filename'].apply(lambda x: Path(x).stem)

        merged = test_df.merge(inst_df[['stem', 'adv_prob_rf']], on='stem', how='inner')
        log(f"  Merged (instability ∩ mahalanobis): {len(merged)} rows")

        # Training data: use pilot's labeled data concept
        # Re-extract mahalanobis features for training data
        # Load pilot features
        pilot_df = pd.read_parquet(OUT / 'pilot/pilot_features.parquet')
        mah_pilot_cols = [c for c in pilot_df.columns if c.startswith('mah_')]

        X_pilot = pilot_df[mah_pilot_cols].values
        y_pilot = (pilot_df['label'] == 'adv').astype(int).values

        # Train RF on pilot mahalanobis features
        clf_mah = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        proba_mah_cv = cross_val_predict(clf_mah, X_pilot, y_pilot, cv=skf, method='predict_proba')[:, 1]
        auc_mah_cv = roc_auc_score(y_pilot, proba_mah_cv)
        log(f"  Mahalanobis-only RF CV AUROC: {auc_mah_cv:.4f}")

        # Load instability training data
        DETECTORS = ['npr', 'distildire', 'aide_genimage', 'xception',
                     'siglip_dino', 'aide_progan', 'haywood', 'aide_sd14']
        CROP_VARIANTS = ['face_crop_variable', 'random_crop_192_a', 'center_crop_192']
        def build_inst_X(ds):
            feats = []
            for det in DETECTORS:
                df_full = pd.read_parquet(CROP_SCORES / f'{det}__{ds}__full.parquet').sort_values('filename')
                for v in CROP_VARIANTS:
                    df_v = pd.read_parquet(CROP_SCORES / f'{det}__{ds}__{v}.parquet').sort_values('filename')
                    feats.append(np.abs(df_full['score'].to_numpy() - df_v['score'].to_numpy()))
            return np.stack(feats, axis=1)
        X_inst_real = build_inst_X('real_reference')
        X_inst_adv  = build_inst_X('adversarial_fake')
        X_inst = np.vstack([X_inst_real, X_inst_adv])
        y_inst = np.array([0]*len(X_inst_real) + [1]*len(X_inst_adv))
        proba_inst_cv = cross_val_predict(
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            X_inst, y_inst, cv=StratifiedKFold(5, shuffle=True, random_state=42),
            method='predict_proba')[:, 1]
        auc_inst_cv = roc_auc_score(y_inst, proba_inst_cv)
        log(f"  Instability-only RF CV AUROC: {auc_inst_cv:.4f}")

        # Test set prediction (Mahalanobis only)
        clf_mah_full = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf_mah_full.fit(X_pilot, y_pilot)
        # For test set, use avg_dist columns
        test_mah_features = test_df[mah_cols].values
        test_mah_prob = clf_mah_full.predict_proba(test_mah_features)[:, 1]
        test_df['adv_prob_mah'] = test_mah_prob.round(4)
        test_df.to_parquet(OUT / 'predictions_mahalanobis_only.parquet', index=False)

        # Combined prediction (where both exist)
        if len(merged) > 100:
            merged_mah = merged[mah_cols].values
            merged_inst = merged['adv_prob_rf'].values.reshape(-1, 1)
            combined = np.hstack([merged_mah, merged_inst])
            log(f"  Combined features dim: {combined.shape[1]}")

        # Comparison summary
        mah_det_rate = (test_mah_prob >= 0.5).mean()
        log(f"\n  Test set prediction (Mahalanobis only):")
        log(f"    adv detection rate (>0.5): {mah_det_rate*100:.1f}%")
        log(f"    mean prob: {test_mah_prob.mean():.3f}")

        # 3-method comparison
        results = {
            'instability_cv_auroc': round(auc_inst_cv, 4),
            'mahalanobis_cv_auroc': round(auc_mah_cv, 4),
            'mah_test_det_rate': round(float(mah_det_rate), 4),
        }
        with open(OUT / 'phase4_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        save_checkpoint('phase4')
        log(f"  Phase 4 완료 ({time.time()-t_start:.0f}s)")
    except Exception as e:
        log(f"❌ Phase 4 실패: {e}")
        traceback.print_exc()
        save_checkpoint('phase4')
else:
    log("Phase 4 이미 완료, skip")

# ══════════════════════════════════════════════════════════════════════
# PHASE 5: Analysis & Visualization
# ══════════════════════════════════════════════════════════════════════
if not checkpoint_exists('phase5'):
    log("\n" + "="*70)
    log("PHASE 5: Analysis, visualization, final report")
    log("="*70)
    try:
        test_df = pd.read_parquet(OUT / 'test_features.parquet') if (OUT / 'test_features.parquet').exists() else None

        # 5a. Layer ablation
        log("  5a: Layer ablation")
        if (OUT / 'pilot/pilot_auroc.csv').exists():
            auc_df = pd.read_csv(OUT / 'pilot/pilot_auroc.csv')
            auc_df.to_csv(OUT / 'layer_ablation.csv', index=False)

        # 5b. Mahalanobis distance distribution (test set)
        if test_df is not None:
            log("  5b: Distance distribution")
            fig, axes = plt.subplots(1, len(class_names), figsize=(4*len(class_names), 4))
            for ci, cn in enumerate(class_names):
                col = f'avg_dist_{cn}'
                if col in test_df.columns:
                    axes[ci].hist(test_df[col], bins=50, color='#444', edgecolor='none')
                    axes[ci].set_title(f'Distance to {cn}')
                    axes[ci].set_xlabel('Mahalanobis distance')
            plt.suptitle('Test set Mahalanobis distance distributions')
            plt.tight_layout()
            plt.savefig(OUT / 'mahalanobis_distribution.png', dpi=100, bbox_inches='tight')
            plt.close()

            # 5c. Closest class distribution
            log("  5c: Class assignment")
            class_counts = test_df['closest_class'].value_counts()
            log(f"    Closest class distribution:")
            for cn in class_names:
                n = class_counts.get(cn, 0)
                log(f"      {cn}: {n} ({n/len(test_df)*100:.1f}%)")

            with open(OUT / 'class_assignment_analysis.md', 'w') as f:
                f.write('# Class Assignment Analysis\n\n')
                f.write('| class | n | % |\n|---|---|---|\n')
                for cn in class_names:
                    n = class_counts.get(cn, 0)
                    f.write(f'| {cn} | {n} | {n/len(test_df)*100:.1f}% |\n')

        # 5d. Correlation with instability (if available)
        if INST_PRED.exists() and test_df is not None:
            log("  5d: Correlation with instability")
            inst_df = pd.read_parquet(INST_PRED)
            inst_df['stem'] = inst_df['filename'].apply(lambda x: Path(x).stem)
            test_df['stem'] = test_df['filename'].apply(lambda x: Path(x).stem)
            merged = test_df.merge(inst_df[['stem', 'adv_prob_rf']], on='stem', how='inner')
            if 'adv_prob_mah' in merged.columns and len(merged) > 10:
                rho, p = sp_stats.pearsonr(merged['adv_prob_rf'], merged['adv_prob_mah'])
                log(f"    Pearson(instability, mahalanobis): r={rho:.3f} p={p:.2e}")
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(merged['adv_prob_rf'], merged['adv_prob_mah'], s=3, alpha=0.2)
                ax.set_xlabel('Instability adv_prob'); ax.set_ylabel('Mahalanobis adv_prob')
                ax.set_title(f'Instability vs Mahalanobis (r={rho:.3f})')
                plt.tight_layout()
                plt.savefig(OUT / 'correlation_with_instability.png', dpi=130, bbox_inches='tight')
                plt.close()

        # 5e. Final report
        log("  5e: Final report")
        with open(OUT / 'final_report.md', 'w') as f:
            f.write('# Mahalanobis Distance Full Pipeline — Final Report\n\n')
            f.write(f'Total runtime: {(time.time()-t_start)/3600:.1f} hours\n\n')

            # Phase 1
            f.write('## Phase 1: Pilot\n\n')
            if (OUT / 'pilot/pilot_report.md').exists():
                f.write(open(OUT / 'pilot/pilot_report.md').read())
            f.write('\n\n')

            # Phase 2
            f.write('## Phase 2: Full Gaussian Fitting\n\n')
            if (OUT / 'gaussian_params_full.pkl').exists():
                f.write('✓ Completed\n\n')
            else:
                f.write('Used pilot params (full fitting skipped/failed)\n\n')

            # Phase 3
            f.write('## Phase 3: Test Set Inference\n\n')
            if test_df is not None:
                f.write(f'- {len(test_df)} images processed\n')
                class_counts = test_df['closest_class'].value_counts()
                f.write(f'- Closest class distribution:\n\n')
                for cn in class_names:
                    n = class_counts.get(cn, 0)
                    f.write(f'  - {cn}: {n} ({n/len(test_df)*100:.1f}%)\n')
            f.write('\n')

            # Phase 4
            f.write('## Phase 4: Ensemble\n\n')
            if (OUT / 'phase4_results.json').exists():
                r4 = json.load(open(OUT / 'phase4_results.json'))
                f.write(f'- Instability CV AUROC: {r4.get("instability_cv_auroc", "N/A")}\n')
                f.write(f'- Mahalanobis CV AUROC: {r4.get("mahalanobis_cv_auroc", "N/A")}\n')
                f.write(f'- Mahalanobis test det rate: {r4.get("mah_test_det_rate", "N/A")}\n')
            f.write('\n')

            # Conclusion
            f.write('## Conclusion\n\n')
            r4 = json.load(open(OUT / 'phase4_results.json')) if (OUT / 'phase4_results.json').exists() else {}
            inst_auc = r4.get('instability_cv_auroc', 0)
            mah_auc = r4.get('mahalanobis_cv_auroc', 0)
            best = max(inst_auc, mah_auc)
            if mah_auc > inst_auc + 0.02:
                f.write(f'✓ Mahalanobis ({mah_auc:.4f}) > Instability ({inst_auc:.4f})\n')
            elif inst_auc > mah_auc + 0.02:
                f.write(f'~ Instability ({inst_auc:.4f}) > Mahalanobis ({mah_auc:.4f})\n')
            else:
                f.write(f'~ 두 방법 유사 (inst={inst_auc:.4f}, mah={mah_auc:.4f})\n')

        save_checkpoint('phase5')
        log(f"  Phase 5 완료 ({time.time()-t_start:.0f}s)")
    except Exception as e:
        log(f"❌ Phase 5 실패: {e}")
        traceback.print_exc()
        save_checkpoint('phase5')
else:
    log("Phase 5 이미 완료, skip")

# ══════════════════════════════════════════════════════════════════════
total_time = time.time() - t_start
log(f"\n{'='*70}")
log(f"전체 파이프라인 완료: {total_time/3600:.1f}시간 ({total_time:.0f}초)")
log(f"{'='*70}")
