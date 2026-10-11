"""
import os
Test Set 24,404 Full Pipeline + Submission Generation
- Phase 1: Load existing assets (SigLIP scores, pHash)
- Phase 2: ForAda probability extraction (24,404)
- Phase 3: Mahalanobis + Disagreement extraction (24,404)
- Phase 4: Final prediction + submission
"""
import os, sys, json, pickle, time, gc
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('HF_TOKEN', 'REMOVED_TOKEN_USE_ENV_VAR')

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

BASE = Path('/home/student/GEN_AI_GTG')
OUT = BASE / 'detection/submission'
OUT.mkdir(parents=True, exist_ok=True)
TEST_DIR = BASE / 'detection/ImageCLEF2026-DeepFakeDetection-Tes/Data/Images_Detection'
DEVICE = torch.device('cuda:0')
LOG_FILE = OUT / 'run.log'
CHECKPOINT = OUT / 'checkpoint.json'

def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def ckpt_exists(name):
    return (OUT / f'.ckpt_{name}').exists()

def save_ckpt(name):
    (OUT / f'.ckpt_{name}').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))

# ══════════════════════════════════════════════════════════
# PHASE 1: LOAD EXISTING ASSETS
# ══════════════════════════════════════════════════════════
log("="*60)
log("PHASE 1: Loading existing assets")
log("="*60)

# 1.1: SigLIP scores (already extracted for 24,404)
sig_scores = pd.read_parquet(BASE / 'detection/scores/siglip_dino__test.parquet')
sig_scores['image_id'] = sig_scores['filename'].apply(lambda f: Path(f).stem)
sig_map = dict(zip(sig_scores['image_id'], sig_scores['score']))
log(f"SigLIP scores loaded: {len(sig_map)} images")

# 1.2: pHash matching
phash = pd.read_csv(BASE / 'detection/phash_matching/fake_gt_matches.csv')
phash_ids = set(Path(p).stem for p in phash['test_image_path'])
log(f"pHash matches: {len(phash_ids)} images")

# 1.3: Submission template
sub_template = pd.read_csv(BASE / 'detection/ImageCLEF2026-DeepFakeDetection-Tes/Data/Images_Detection_submission.csv')
all_image_names = sub_template['full_secret_name'].tolist()
all_image_ids = [n.replace('.png', '') for n in all_image_names]
log(f"Submission template: {len(all_image_names)} images")

# 1.4: Test image paths
test_paths = {Path(p).stem: p for p in sorted(TEST_DIR.glob('*.png'))}
log(f"Test images found: {len(test_paths)}")

# 1.5: Clean Gaussian
with open(BASE / 'detection/gaussian_fitting/clean_gaussian_params.pkl', 'rb') as f:
    clean_params = pickle.load(f)
log("Clean Gaussian loaded.")

# ══════════════════════════════════════════════════════════
# PHASE 2: ForAda PROBABILITY (24,404)
# ══════════════════════════════════════════════════════════
fa_scores_file = OUT / 'forada_test_scores.parquet'

if not ckpt_exists('phase2'):
    log("\n" + "="*60)
    log("PHASE 2: ForAda probability extraction")
    log("="*60)

    FA_ROOT = BASE / 'detection/ForensicsAdapter'
    sys.path.insert(0, str(FA_ROOT))
    from model.ds import DS as ForAda_DS

    fa_model = ForAda_DS(clip_name='ViT-L/14', adapter_vit_name='vit_tiny_patch16_224',
        num_quires=128, fusion_map={0:0,1:1,2:8,3:15},
        mlp_dim=256, mlp_out_dim=128, head_num=16, device=str(DEVICE))
    fa_ckpt = torch.load(str(FA_ROOT / 'weights/ckpt_best.pth'), map_location='cpu')
    fa_model.load_state_dict(fa_ckpt, strict=False)
    fa_model.to(DEVICE).eval()

    fa_mean = torch.tensor([0.48145466,0.4578275,0.40821073]).view(1,3,1,1).to(DEVICE)
    fa_std = torch.tensor([0.26862954,0.26130258,0.27577711]).view(1,3,1,1).to(DEVICE)

    fa_results = {}
    batch_size = 16
    ordered_ids = list(test_paths.keys())

    for bi in range(0, len(ordered_ids), batch_size):
        chunk_ids = ordered_ids[bi:bi+batch_size]
        pils = [Image.open(test_paths[iid]).convert('RGB').resize((256,256), Image.LANCZOS) for iid in chunk_ids]
        imgs = torch.stack([T.ToTensor()(p) for p in pils]).to(DEVICE)
        imgs_norm = (imgs - fa_mean) / fa_std
        B = imgs_norm.shape[0]

        with torch.no_grad():
            dd = {'image': imgs_norm, 'label': torch.zeros(B, dtype=torch.long, device=DEVICE),
                  'mask': None, 'landmark': None, 'if_boundary': torch.ones(B, 256, device=DEVICE)}
            pred = fa_model(dd, inference=True)
            probs = pred['prob'].cpu().numpy()
            fa_model.prob=[]; fa_model.label=[]; fa_model.correct=0; fa_model.total=0

        for j, iid in enumerate(chunk_ids):
            fa_results[iid] = float(probs[j])

        if (bi // batch_size) % 100 == 0:
            log(f"  ForAda: {bi+len(chunk_ids)}/{len(ordered_ids)}")
            torch.cuda.empty_cache()

    df_fa = pd.DataFrame([{'image_id': k, 'forada_prob': v} for k, v in fa_results.items()])
    df_fa.to_parquet(fa_scores_file, index=False)

    del fa_model; gc.collect(); torch.cuda.empty_cache()
    save_ckpt('phase2')
    log(f"ForAda done: {len(df_fa)} images")
else:
    log("Phase 2: checkpoint found, loading...")
    df_fa = pd.read_parquet(fa_scores_file)

fa_map = dict(zip(df_fa['image_id'], df_fa['forada_prob']))

# ══════════════════════════════════════════════════════════
# PHASE 3: MAHALANOBIS + DISAGREEMENT (24,404)
# ══════════════════════════════════════════════════════════
maha_file = OUT / 'test_mahalanobis.parquet'

if not ckpt_exists('phase3'):
    log("\n" + "="*60)
    log("PHASE 3: Mahalanobis + Disagreement extraction")
    log("="*60)

    # Load SigLIP+DINOv2
    from huggingface_hub import hf_hub_download
    import importlib.util

    REPO_ID = 'Bombek1/ai-image-detector-siglip-dinov2'
    model_py = hf_hub_download(repo_id=REPO_ID, filename='model.py')
    spec = importlib.util.spec_from_file_location('ai_det', model_py)
    ai_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(ai_mod)
    wt = hf_hub_download(repo_id=REPO_ID, filename='pytorch_model.pt')
    ckpt = torch.load(wt, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    sig_model = ai_mod.create_model_with_lora(
        siglip_model_name=cfg.get('siglip_model', 'google/siglip2-so400m-patch14-384'),
        dinov2_model_name=cfg.get('dinov2_model', 'vit_large_patch14_dinov2.lvd142m'),
        image_size=cfg.get('image_size', 392))
    sig_model.load_state_dict(ckpt['model_state_dict'])
    sig_model.to(DEVICE).eval()

    dinov2_backbone = sig_model.dinov2
    dinov2_blocks = list(dinov2_backbone.blocks)
    n_dino = len(dinov2_blocks)
    sig_li = [int(0.2*n_dino), int(0.4*n_dino), int(0.6*n_dino), int(0.8*n_dino), n_dino-1]
    dino_mean = torch.tensor([.485,.456,.406]).view(1,3,1,1).to(DEVICE)
    dino_std = torch.tensor([.229,.224,.225]).view(1,3,1,1).to(DEVICE)

    sig_hook_out = {}
    sig_hooks = []
    def sig_mk(li):
        def fn(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            sig_hook_out[li] = t[:, 0, :].detach().cpu()
        return fn
    for li in sig_li:
        sig_hooks.append(dinov2_blocks[li].register_forward_hook(sig_mk(li)))

    # Load ForAda for hooks
    FA_ROOT = BASE / 'detection/ForensicsAdapter'
    if str(FA_ROOT) not in sys.path:
        sys.path.insert(0, str(FA_ROOT))
    from model.ds import DS as ForAda_DS

    fa_model = ForAda_DS(clip_name='ViT-L/14', adapter_vit_name='vit_tiny_patch16_224',
        num_quires=128, fusion_map={0:0,1:1,2:8,3:15},
        mlp_dim=256, mlp_out_dim=128, head_num=16, device=str(DEVICE))
    fa_ckpt_data = torch.load(str(FA_ROOT / 'weights/ckpt_best.pth'), map_location='cpu')
    fa_model.load_state_dict(fa_ckpt_data, strict=False)
    fa_model.to(DEVICE).eval()
    fa_mean = torch.tensor([0.48145466,0.4578275,0.40821073]).view(1,3,1,1).to(DEVICE)
    fa_std = torch.tensor([0.26862954,0.26130258,0.27577711]).view(1,3,1,1).to(DEVICE)

    clip_resblocks = list(fa_model.clip_model.visual.transformer.resblocks)
    fa_hook_out = {}
    fa_hooks = []
    fa_li = [4, 9, 14, 19, 23]
    def fa_mk(li):
        def fn(m, i, o):
            fa_hook_out[li] = o[0].detach().cpu()
        return fn
    for li in fa_li:
        fa_hooks.append(clip_resblocks[li-1].register_forward_hook(fa_mk(li)))

    log("Both models loaded for Mahalanobis.")

    # Process all 24,404
    maha_results = []
    ordered_ids = list(test_paths.keys())
    batch_size = 16

    for bi in range(0, len(ordered_ids), batch_size):
        chunk_ids = ordered_ids[bi:bi+batch_size]
        pils = [Image.open(test_paths[iid]).convert('RGB').resize((256,256), Image.LANCZOS) for iid in chunk_ids]
        imgs = torch.stack([T.ToTensor()(p) for p in pils]).to(DEVICE)
        B = imgs.shape[0]

        with torch.no_grad():
            # DINOv2 hooks
            sig_hook_out.clear()
            dino_inp = (F.interpolate(imgs, (392,392), mode='bilinear', align_corners=False) - dino_mean) / dino_std
            _ = dinov2_backbone.forward_features(dino_inp)

            # ForAda hooks
            fa_hook_out.clear()
            fa_inp = (imgs - fa_mean) / fa_std
            dd = {'image': fa_inp, 'label': torch.zeros(B, dtype=torch.long, device=DEVICE),
                  'mask': None, 'landmark': None, 'if_boundary': torch.ones(B, 256, device=DEVICE)}
            _ = fa_model(dd, inference=True)
            fa_model.prob=[]; fa_model.label=[]; fa_model.correct=0; fa_model.total=0

        for j, iid in enumerate(chunk_ids):
            rec = {'image_id': iid}
            for det, layers, hook_out in [('siglip_dinov2', sig_li, sig_hook_out),
                                           ('forada', fa_li, fa_hook_out)]:
                d_real = 0.0; d_fake = 0.0
                for li in layers:
                    feat = hook_out[li][j].numpy().astype(np.float32)
                    p = clean_params[det][li]
                    dr = feat - p['mu_real']; df_ = feat - p['mu_fake']
                    d_real += float(dr @ p['Sigma_inv'] @ dr)
                    d_fake += float(df_ @ p['Sigma_inv'] @ df_)
                rec[f'd_real_{det}'] = d_real
                rec[f'd_fake_{det}'] = d_fake
            maha_results.append(rec)

        if (bi // batch_size) % 100 == 0:
            log(f"  Mahalanobis: {bi+len(chunk_ids)}/{len(ordered_ids)}")
            torch.cuda.empty_cache()

    df_maha = pd.DataFrame(maha_results)
    df_maha.to_parquet(maha_file, index=False)

    for h in sig_hooks: h.remove()
    for h in fa_hooks: h.remove()
    del sig_model, fa_model, dinov2_backbone; gc.collect(); torch.cuda.empty_cache()
    save_ckpt('phase3')
    log(f"Mahalanobis done: {len(df_maha)} images")
else:
    log("Phase 3: checkpoint found, loading...")
    df_maha = pd.read_parquet(maha_file)

# ══════════════════════════════════════════════════════════
# PHASE 4: BUILD FEATURE MATRIX + PREDICTIONS
# ══════════════════════════════════════════════════════════
log("\n" + "="*60)
log("PHASE 4: Final prediction + submission")
log("="*60)

# Build master DataFrame
df = pd.DataFrame({'image_id': all_image_ids, 'filename': all_image_names})

# Add SigLIP probability
df['prob_siglip'] = df['image_id'].map(sig_map)

# Add ForAda probability
df['prob_forada'] = df['image_id'].map(fa_map)

# Add Mahalanobis
maha_map = {r['image_id']: r for _, r in df_maha.iterrows()}
for col in ['d_real_siglip_dinov2', 'd_fake_siglip_dinov2', 'd_real_forada', 'd_fake_forada']:
    df[col] = df['image_id'].apply(lambda x: maha_map.get(x, {}).get(col, np.nan))

# Add pHash
df['is_our_adv'] = df['image_id'].isin(phash_ids)

# Compute derived features
for det in ['siglip_dinov2', 'forada']:
    dr = f'd_real_{det}'; df_ = f'd_fake_{det}'
    df[f'maha_str_{det}'] = (df[dr] - df[df_]) / (df[dr] + df[df_] + 1e-8)
    pc = 'prob_siglip' if det == 'siglip_dinov2' else 'prob_forada'
    df[f'dis_{det}'] = (1 - df[pc]) * df[f'maha_str_{det}'].clip(lower=0)
df['dis_combined'] = (df['dis_siglip_dinov2'] + df['dis_forada']) / 2

# Mahalanobis verdict
df['maha_verdict_sig'] = np.where(df['d_fake_siglip_dinov2'] < df['d_real_siglip_dinov2'], 'FAKE', 'REAL')
df['maha_verdict_fa'] = np.where(df['d_fake_forada'] < df['d_real_forada'], 'FAKE', 'REAL')

log(f"\nFeature matrix built: {len(df)} images, {len(df.columns)} columns")
log(f"NaN check: {df.isna().sum().sum()} total NaN values")

# Save feature matrix
df.to_parquet(OUT / 'test_features.parquet', index=False)

# ── Distribution summary ──
log("\n=== Distribution Summary ===")
log(f"SigLIP prob: mean={df['prob_siglip'].mean():.4f}, median={df['prob_siglip'].median():.4f}")
log(f"  >0.8: {(df['prob_siglip']>0.8).sum()}, <0.2: {(df['prob_siglip']<0.2).sum()}, 0.2-0.8: {((df['prob_siglip']>=0.2)&(df['prob_siglip']<=0.8)).sum()}")
log(f"ForAda prob: mean={df['prob_forada'].mean():.4f}, median={df['prob_forada'].median():.4f}")
log(f"  >0.5: {(df['prob_forada']>0.5).sum()}, <0.2: {(df['prob_forada']<0.2).sum()}")
log(f"Maha verdict SigLIP: FAKE={( df['maha_verdict_sig']=='FAKE').sum()}, REAL={(df['maha_verdict_sig']=='REAL').sum()}")
log(f"Maha verdict ForAda: FAKE={(df['maha_verdict_fa']=='FAKE').sum()}, REAL={(df['maha_verdict_fa']=='REAL').sum()}")
log(f"Disagreement combined: mean={df['dis_combined'].mean():.6f}, >0.0003: {(df['dis_combined']>0.0003).sum()}")
log(f"pHash matched: {df['is_our_adv'].sum()}")

# Our adversarial stats
our = df[df['is_our_adv']]
log(f"\nOur adversarial ({len(our)}):")
log(f"  SigLIP prob mean={our['prob_siglip'].mean():.4f}, ForAda prob mean={our['prob_forada'].mean():.4f}")
log(f"  Maha FAKE (sig): {(our['maha_verdict_sig']=='FAKE').sum()}/{len(our)}")
log(f"  Disagree combined mean={our['dis_combined'].mean():.6f}")

# ── Rule-based prediction ──
def predict_rules(row):
    # Rule 0: pHash → FAKE
    if row['is_our_adv']:
        return 1, 'phash'

    # Rule 1: SigLIP prob very high → FAKE (clean AI-gen)
    if row['prob_siglip'] > 0.8:
        return 1, 'siglip_high'

    # Rule 2: Both Mahalanobis say FAKE + SigLIP prob moderate
    if row['maha_verdict_sig'] == 'FAKE' and row['maha_verdict_fa'] == 'FAKE':
        if row['prob_siglip'] > 0.3:
            return 1, 'both_maha_fake'

    # Rule 3: SigLIP Mahalanobis FAKE + SigLIP prob > 0.5
    if row['maha_verdict_sig'] == 'FAKE' and row['prob_siglip'] > 0.5:
        return 1, 'sig_maha_prob'

    # Rule 4: Disagreement high → FAKE (adversarial)
    if row['dis_combined'] > 0.0003:
        return 1, 'disagreement'

    # Rule 5: ForAda prob high → FAKE (face manipulation)
    if row['prob_forada'] > 0.5:
        return 1, 'forada_high'

    # Rule 6: SigLIP prob moderate + Maha FAKE
    if row['prob_siglip'] > 0.5 and row['maha_verdict_sig'] == 'FAKE':
        return 1, 'sig_moderate_maha'

    # Default: REAL
    return 0, 'default_real'

df['pred_balanced'], df['reason_balanced'] = zip(*df.apply(predict_rules, axis=1))

# Conservative
def predict_conservative(row):
    if row['is_our_adv']: return 1
    if row['prob_siglip'] > 0.9: return 1
    if row['maha_verdict_sig'] == 'FAKE' and row['maha_verdict_fa'] == 'FAKE' and row['prob_siglip'] > 0.5:
        return 1
    if row['prob_forada'] > 0.7: return 1
    return 0

df['pred_conservative'] = df.apply(predict_conservative, axis=1)

# Aggressive
def predict_aggressive(row):
    if row['is_our_adv']: return 1
    if row['prob_siglip'] > 0.5: return 1
    if row['maha_verdict_sig'] == 'FAKE': return 1
    if row['prob_forada'] > 0.3: return 1
    if row['dis_combined'] > 0.0002: return 1
    return 0

df['pred_aggressive'] = df.apply(predict_aggressive, axis=1)

# ── Stats ──
for variant in ['conservative', 'balanced', 'aggressive']:
    col = f'pred_{variant}'
    fake_n = df[col].sum()
    log(f"\n{variant.upper()}: FAKE={fake_n}, REAL={len(df)-fake_n}, ratio={fake_n/len(df):.3f}")

log(f"\nBalanced reasons:")
log(df['reason_balanced'].value_counts().to_string())

# Sanity: our adversarial all FAKE?
for variant in ['conservative', 'balanced', 'aggressive']:
    our_pred = df[df['is_our_adv']][f'pred_{variant}'].sum()
    log(f"  {variant}: our adv FAKE = {our_pred}/{len(our)}")

# ══════════════════════════════════════════════════════════
# PHASE 5: SUBMISSION FILES
# ══════════════════════════════════════════════════════════
log("\n" + "="*60)
log("PHASE 5: Generating submission files")
log("="*60)

for variant in ['conservative', 'balanced', 'aggressive']:
    sub = pd.DataFrame({
        'full_secret_name': df['filename'],
        'prediction': df[f'pred_{variant}'],
    })
    out_path = OUT / f'submission_{variant}.csv'
    sub.to_csv(out_path, index=False)
    log(f"  {variant}: {out_path} ({sub['prediction'].sum()} FAKE / {(sub['prediction']==0).sum()} REAL)")

# Default submission = balanced
import shutil
shutil.copy(OUT / 'submission_balanced.csv', OUT / 'submission.csv')

# ══════════════════════════════════════════════════════════
# PHASE 6: REPORT
# ══════════════════════════════════════════════════════════
log("\n" + "="*60)
log("PHASE 6: Report")
log("="*60)

report = f"""# Test Set Submission Report

## Pipeline
1. SigLIP+DINOv2 probability (pre-existing scores)
2. ForAda probability (newly extracted)
3. Mahalanobis distances (SigLIP + ForAda, Clean Gaussian)
4. Disagreement scores
5. pHash matching (1,002 adversarial)
6. Rule-based prediction

## Data
- Test set: {len(df)} images
- pHash matched (our adversarial): {df['is_our_adv'].sum()}
- Feature NaN: {df[['prob_siglip','prob_forada','d_real_siglip_dinov2']].isna().sum().sum()}

## Distribution

| Signal | Mean | Median | >threshold |
|---|---|---|---|
| SigLIP prob | {df['prob_siglip'].mean():.4f} | {df['prob_siglip'].median():.4f} | >0.8: {(df['prob_siglip']>0.8).sum()} |
| ForAda prob | {df['prob_forada'].mean():.4f} | {df['prob_forada'].median():.4f} | >0.5: {(df['prob_forada']>0.5).sum()} |
| Maha SigLIP FAKE | - | - | {(df['maha_verdict_sig']=='FAKE').sum()} |
| Maha ForAda FAKE | - | - | {(df['maha_verdict_fa']=='FAKE').sum()} |
| Disagreement | {df['dis_combined'].mean():.6f} | {df['dis_combined'].median():.6f} | >0.0003: {(df['dis_combined']>0.0003).sum()} |

## Our Adversarial (pHash matched: {df['is_our_adv'].sum()})
- SigLIP prob mean: {our['prob_siglip'].mean():.4f}
- ForAda prob mean: {our['prob_forada'].mean():.4f}
- Maha FAKE (sig): {(our['maha_verdict_sig']=='FAKE').sum()}/{len(our)}
- Disagree mean: {our['dis_combined'].mean():.6f}

## Submissions

| Variant | FAKE | REAL | Ratio |
|---|---|---|---|
| Conservative | {df['pred_conservative'].sum()} | {(df['pred_conservative']==0).sum()} | {df['pred_conservative'].mean():.3f} |
| **Balanced** | **{df['pred_balanced'].sum()}** | **{(df['pred_balanced']==0).sum()}** | **{df['pred_balanced'].mean():.3f}** |
| Aggressive | {df['pred_aggressive'].sum()} | {(df['pred_aggressive']==0).sum()} | {df['pred_aggressive'].mean():.3f} |

## Balanced Prediction Breakdown

| Rule | Count | % |
|---|---|---|
"""
for reason, count in df['reason_balanced'].value_counts().items():
    report += f"| {reason} | {count} | {count/len(df):.1%} |\n"

report += f"""
## Sanity Checks
- Our 1,002 adv all FAKE (balanced): {df[df['is_our_adv']]['pred_balanced'].sum() == len(our)}
- FAKE ratio 0.3-0.7: {0.3 <= df['pred_balanced'].mean() <= 0.7}
"""

with open(OUT / 'report.md', 'w') as f:
    f.write(report)

log("\nAll done!")
log(f"Primary submission: {OUT / 'submission.csv'}")
