"""
import os
Phase 2 Gaussian 기반 RF 재학습.
1. Pilot 400장 raw features 재추출 (DINOv2 CLS)
2. Phase 2 params로 distance 재계산
3. RF 재학습 + CV
4. 우리 1002장 + test set 전체 prediction
5. Instability ensemble
6. Three-way 비교
"""
import os, sys, gc, csv, pickle, json, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('HF_TOKEN', 'REMOVED_TOKEN_USE_ENV_VAR')

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from PIL import Image
from tqdm import tqdm

BASE = Path('/home/student/GEN_AI_GTG')
OUT = BASE / 'detection/mahalanobis/fixed'
OUT.mkdir(parents=True, exist_ok=True)
MAH_DIR = BASE / 'detection/mahalanobis'
CLEAN_DIR = BASE / 'generation/outputs/final_1002'
ADV_DIR   = BASE / 'generation/submit/third_sub/256'
REAL_DIR  = BASE / 'detection/real_reference/center'
CROP_SCORES = BASE / 'detection/crop_ablation/scores'
INST_PRED = BASE / 'detection/instability/test_application/test_instability_predictions.parquet'

DEVICE = torch.device('cuda:0')
t0 = time.time()

# ── 1. Load model + Gaussian params ──────────────────────────────────
print("="*70)
print("[1/7] Model + Gaussian params 로드")
print("="*70)

from huggingface_hub import hf_hub_download
import importlib.util
HF_TOKEN = os.environ['HF_TOKEN']
REPO = 'Bombek1/ai-image-detector-siglip-dinov2'
model_py = hf_hub_download(repo_id=REPO, filename='model.py')
spec = importlib.util.spec_from_file_location('ai_det', model_py)
ai_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(ai_mod)
wt = hf_hub_download(repo_id=REPO, filename='pytorch_model.pt')
ckpt = torch.load(wt, map_location='cpu', weights_only=False)
cfg = ckpt.get('config', {})
ens_model = ai_mod.create_model_with_lora(
    siglip_model_name=cfg.get('siglip_model','google/siglip2-so400m-patch14-384'),
    dinov2_model_name=cfg.get('dinov2_model','vit_large_patch14_dinov2.lvd142m'),
    image_size=cfg.get('image_size',392))
ens_model.load_state_dict(ckpt['model_state_dict'])
ens_model.to(DEVICE).eval()

dinov2 = ens_model.dinov2
dinov2_blocks = list(dinov2.blocks)
print(f"  DINOv2: {len(dinov2_blocks)} blocks")

gp = pickle.load(open(MAH_DIR / 'gaussian_params_full.pkl', 'rb'))
mu = gp['mu']; sigma_inv = gp['sigma_inv']
layer_indices = gp['layer_indices']; class_names = gp['class_names']
feat_dim = gp['feat_dim']
print(f"  Layers: {layer_indices}, classes: {class_names}, dim: {feat_dim}")

dino_mean = torch.tensor([.485,.456,.406]).view(1,3,1,1).to(DEVICE)
dino_std  = torch.tensor([.229,.224,.225]).view(1,3,1,1).to(DEVICE)

def extract_features_batch(pils, batch_size=64):
    all_feats = {li: [] for li in layer_indices}
    hooks = []; hook_outputs = {}
    def make_hook(li):
        def hook_fn(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            hook_outputs[li] = t[:, 0, :].detach().cpu()
        return hook_fn
    for li in layer_indices:
        hooks.append(dinov2_blocks[li].register_forward_hook(make_hook(li)))
    try:
        for i in range(0, len(pils), batch_size):
            chunk = pils[i:i+batch_size]
            imgs = torch.stack([T.ToTensor()(p.convert('RGB')) for p in chunk]).to(DEVICE)
            dino_inp = (F.interpolate(imgs, (392,392), mode='bilinear', align_corners=False) - dino_mean) / dino_std
            with torch.no_grad():
                _ = dinov2.forward_features(dino_inp)
            for li in layer_indices:
                all_feats[li].append(hook_outputs[li])
            hook_outputs.clear()
    finally:
        for h in hooks: h.remove()
    return {li: torch.cat(all_feats[li], dim=0).numpy() for li in layer_indices}

def compute_mahalanobis(feats_dict, n):
    """feats_dict[layer] -> (n, D). Returns (n, n_layers*n_classes) distance matrix."""
    n_dim = len(layer_indices) * len(class_names)
    M = np.zeros((n, n_dim))
    for li_idx, li in enumerate(layer_indices):
        for ci, cn in enumerate(class_names):
            for i in range(n):
                d = feats_dict[li][i] - mu[(cn, li)]
                M[i, li_idx * len(class_names) + ci] = float(d @ sigma_inv[li] @ d)
    return M

# ── 2. Pilot 400장 feature 재추출 + Phase 2 distance ──────────────────
print(f"\n{'='*70}")
print("[2/7] Pilot 400장 feature 재추출 (Phase 2 Gaussian)")
print("="*70)

rng = np.random.RandomState(42)
all_clean = sorted(CLEAN_DIR.glob('*.png'))
sel = rng.choice(len(all_clean), 100, replace=False)
clean_keys = [all_clean[i].stem for i in sorted(sel)]

real_all = sorted(REAL_DIR.glob('*.png'))
real_sel = rng.choice(len(real_all), 200, replace=False)
real_100 = [real_all[i] for i in real_sel[:100]]
jpeg_100 = [real_all[i] for i in real_sel[100:200]]

def load_pils(paths):
    pils = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        if im.size != (256,256): im = im.resize((256,256), Image.LANCZOS)
        pils.append(im)
    return pils

pils_real = load_pils(real_100)
pils_adv = load_pils([ADV_DIR / f'{k}.png' for k in clean_keys])
pils_clean = load_pils([CLEAN_DIR / f'{k}.png' for k in clean_keys])
pils_jpeg = []
for p in jpeg_100:
    img = cv2.imread(str(p))
    if img.shape[:2] != (256,256): img = cv2.resize(img, (256,256))
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    pils_jpeg.append(Image.fromarray(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)))

print("  Feature extraction...")
feats_real  = extract_features_batch(pils_real)
feats_adv   = extract_features_batch(pils_adv)
feats_clean = extract_features_batch(pils_clean)
feats_jpeg  = extract_features_batch(pils_jpeg)
del pils_real, pils_adv, pils_clean, pils_jpeg; gc.collect()

print("  Mahalanobis distance 계산 (Phase 2 scale)...")
M_real  = compute_mahalanobis(feats_real, 100)
M_adv   = compute_mahalanobis(feats_adv, 100)
M_clean = compute_mahalanobis(feats_clean, 100)
M_jpeg  = compute_mahalanobis(feats_jpeg, 100)

print(f"  Distance ranges:")
print(f"    real:  [{M_real.min():.0f}, {M_real.max():.0f}] mean={M_real.mean():.0f}")
print(f"    adv:   [{M_adv.min():.0f}, {M_adv.max():.0f}] mean={M_adv.mean():.0f}")
print(f"    clean: [{M_clean.min():.0f}, {M_clean.max():.0f}] mean={M_clean.mean():.0f}")
print(f"    jpeg:  [{M_jpeg.min():.0f}, {M_jpeg.max():.0f}] mean={M_jpeg.mean():.0f}")

# Save
mah_cols = [f'mah_{i}' for i in range(M_real.shape[1])]
rows = []
for label, M in [('real', M_real), ('adv', M_adv), ('clean_fake', M_clean), ('jpeg', M_jpeg)]:
    for i in range(len(M)):
        row = {'label': label, 'idx': i}
        for j, cn in enumerate(mah_cols):
            row[cn] = float(M[i, j])
        rows.append(row)
pd.DataFrame(rows).to_parquet(OUT / 'pilot_distances_phase2scale.parquet', index=False)

# ── 3. RF 재학습 + CV ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print("[3/7] RF 재학습 (Phase 2 scale)")
print("="*70)

X_pilot = np.vstack([M_real, M_adv, M_clean, M_jpeg])
y_pilot = np.array([0]*100 + [1]*100 + [0]*100 + [0]*100)
print(f"  X: {X_pilot.shape}, y: adv={y_pilot.sum()}, others={(y_pilot==0).sum()}")

skf = StratifiedKFold(5, shuffle=True, random_state=42)
clf_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
proba_cv = cross_val_predict(clf_rf, X_pilot, y_pilot, cv=skf, method='predict_proba')[:, 1]
auc_cv = roc_auc_score(y_pilot, proba_cv)
print(f"  RF 5-fold CV AUROC: {auc_cv:.4f}")

# Per-class CV predictions
for label, start, n in [('real',0,100), ('adv',100,100), ('clean',200,100), ('jpeg',300,100)]:
    p = proba_cv[start:start+n]
    print(f"    {label:6s}: mean_prob={p.mean():.3f} det_rate={(p>=0.5).mean()*100:.0f}%")

clf_rf_full = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf_rf_full.fit(X_pilot, y_pilot)
with open(OUT / 'rf_model_phase2.pkl', 'wb') as f:
    pickle.dump(clf_rf_full, f)

# ── 4. 우리 1002장 평가 ──────────────────────────────────────────────
print(f"\n{'='*70}")
print("[4/7] 우리 1002장 평가")
print("="*70)

# test_features.parquet는 이미 Phase 2 scale
test_df = pd.read_parquet(MAH_DIR / 'test_features.parquet')
mah_cols_test = [c for c in test_df.columns if c.startswith('mah_L')]

ours_set = set()
for r in csv.DictReader(open(BASE / 'detection/phash_matching/fake_gt_matches.csv')):
    ours_set.add(Path(r['test_image_path']).name)
test_df['is_ours'] = test_df['filename'].isin(ours_set)

X_test = test_df[mah_cols_test].values
prob_test = clf_rf_full.predict_proba(X_test)[:, 1]
test_df['adv_prob_mah_fixed'] = prob_test.round(4)

ours_prob = prob_test[test_df['is_ours'].values]
ours_recall = (ours_prob >= 0.5).mean()
print(f"  Our 1002 recall (>0.5): {(ours_prob>=0.5).sum()}/{len(ours_prob)} = {ours_recall*100:.1f}%")
print(f"  Our 1002 mean prob: {ours_prob.mean():.3f}")
print(f"  Our 1002 median prob: {np.median(ours_prob):.3f}")

# ── 5. Test set 전체 ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print("[5/7] Test set 24,404장 prediction")
print("="*70)

for t in [0.3, 0.5, 0.7, 0.9]:
    n = int((prob_test >= t).sum())
    print(f"  threshold={t}: {n}/{len(prob_test)} = {n/len(prob_test)*100:.1f}%")

# Bimodality
from scipy.stats import skew, kurtosis
bc_num = skew(prob_test)**2 + 1
n = len(prob_test)
bc_den = kurtosis(prob_test, fisher=True) + 3*((n-1)**2)/((n-2)*(n-3))
bc = bc_num / bc_den
print(f"  Bimodality coefficient: {bc:.4f} ({'bimodal' if bc > 5/9 else 'unimodal'})")

# Save
test_df.to_parquet(OUT / 'test_predictions_fixed.parquet', index=False)

# Histogram
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
bins = np.linspace(0, 1, 51)
axes[0].hist(prob_test, bins=bins, color='#444', edgecolor='none')
axes[0].axvline(0.5, color='red', linestyle='--')
axes[0].set_title(f'Test all (24k)\ndet={int((prob_test>=0.5).sum())} ({(prob_test>=0.5).mean()*100:.1f}%)')
axes[1].hist(ours_prob, bins=bins, color='#d62728', edgecolor='none')
axes[1].axvline(0.5, color='red', linestyle='--')
axes[1].set_title(f'Our 1002\nrecall={int((ours_prob>=0.5).sum())} ({ours_recall*100:.1f}%)')
others_prob = prob_test[~test_df['is_ours'].values]
axes[2].hist(others_prob, bins=bins, color='#888', edgecolor='none')
axes[2].axvline(0.5, color='red', linestyle='--')
axes[2].set_title(f'Others (23k)\ndet={int((others_prob>=0.5).sum())} ({(others_prob>=0.5).mean()*100:.1f}%)')
for ax in axes: ax.set_xlabel('P(adversarial)'); ax.set_ylabel('count')
plt.suptitle('Mahalanobis RF (Phase 2 scale, fixed)')
plt.tight_layout()
plt.savefig(OUT / 'test_probability_distributions.png', dpi=130, bbox_inches='tight')
plt.close()

# ── 6. Ensemble with instability ─────────────────────────────────────
print(f"\n{'='*70}")
print("[6/7] Ensemble with instability")
print("="*70)

# Instability: load
if INST_PRED.exists():
    inst_df = pd.read_parquet(INST_PRED)
    inst_df['stem'] = inst_df['filename'].apply(lambda x: Path(x).stem)
    test_df['stem'] = test_df['filename'].apply(lambda x: Path(x).stem)
    merged = test_df.merge(inst_df[['stem', 'adv_prob_rf']], on='stem', how='inner')
    print(f"  Merged: {len(merged)} rows")

    rho, p = sp_stats.pearsonr(merged['adv_prob_rf'], merged['adv_prob_mah_fixed'])
    print(f"  Pearson(instability, mahalanobis_fixed): r={rho:.3f} p={p:.2e}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(merged['adv_prob_rf'], merged['adv_prob_mah_fixed'], s=3, alpha=0.15, c='#444')
    ax.set_xlabel('Instability adv_prob'); ax.set_ylabel('Mahalanobis adv_prob (fixed)')
    ax.set_title(f'Instability vs Mahalanobis fixed (r={rho:.3f})')
    ax.plot([0,1],[0,1],'r--',alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / 'correlation_inst_vs_mah.png', dpi=130, bbox_inches='tight')
    plt.close()

    # Simple ensemble: average of two probs
    merged['adv_prob_ensemble'] = (merged['adv_prob_rf'] + merged['adv_prob_mah_fixed']) / 2
    # Evaluate on our 1002 within merged
    ours_merged = merged[merged['is_ours']]
    if len(ours_merged) > 0:
        for method, col in [('instability', 'adv_prob_rf'), ('mahalanobis', 'adv_prob_mah_fixed'),
                            ('ensemble_avg', 'adv_prob_ensemble')]:
            recall = (ours_merged[col] >= 0.5).mean()
            print(f"  {method:20s}: recall={recall*100:.1f}% mean_prob={ours_merged[col].mean():.3f}")
else:
    print("  ⚠ Instability predictions not found")
    rho = float('nan')

# Instability CV AUROC (재사용)
DETECTORS = ['npr','distildire','aide_genimage','xception','siglip_dino','aide_progan','haywood','aide_sd14']
CROP_VARIANTS = ['face_crop_variable','random_crop_192_a','center_crop_192']
def build_inst_X(ds):
    feats = []
    for det in DETECTORS:
        df_full = pd.read_parquet(CROP_SCORES / f'{det}__{ds}__full.parquet').sort_values('filename')
        for v in CROP_VARIANTS:
            df_v = pd.read_parquet(CROP_SCORES / f'{det}__{ds}__{v}.parquet').sort_values('filename')
            feats.append(np.abs(df_full['score'].to_numpy() - df_v['score'].to_numpy()))
    return np.stack(feats, axis=1)
X_ir = build_inst_X('real_reference'); X_ia = build_inst_X('adversarial_fake')
X_inst = np.vstack([X_ir, X_ia]); y_inst = np.array([0]*len(X_ir) + [1]*len(X_ia))
p_inst_cv = cross_val_predict(RandomForestClassifier(200, random_state=42, n_jobs=-1),
                               X_inst, y_inst, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                               method='predict_proba')[:, 1]
auc_inst = roc_auc_score(y_inst, p_inst_cv)

# ── 7. Three-way comparison ──────────────────────────────────────────
print(f"\n{'='*70}")
print("[7/7] Three-way comparison")
print("="*70)

mah_det = (prob_test >= 0.5).mean()
mah_recall = ours_recall

print(f"\n  {'Method':<25} {'CV AUROC':>10} {'Test det':>10} {'Our recall':>11}")
print("  " + "-" * 58)
print(f"  {'Instability 24-dim':<25} {auc_inst:>10.4f} {'~26%':>10} {'~98%':>11}")
print(f"  {'Mahalanobis 20-dim':<25} {auc_cv:>10.4f} {mah_det*100:>9.1f}% {mah_recall*100:>10.1f}%")
if INST_PRED.exists() and len(ours_merged) > 0:
    ens_recall = (ours_merged['adv_prob_ensemble'] >= 0.5).mean()
    ens_det = (merged['adv_prob_ensemble'] >= 0.5).mean()
    print(f"  {'Ensemble (avg)':<25} {'N/A':>10} {ens_det*100:>9.1f}% {ens_recall*100:>10.1f}%")

# Report
with open(OUT / 'fixed_report.md', 'w') as f:
    f.write('# Mahalanobis RF Fix — Phase 2 Scale\n\n')
    f.write(f'## RF CV AUROC: **{auc_cv:.4f}**\n\n')
    f.write(f'## Our 1002 Recall: **{mah_recall*100:.1f}%** (mean prob={ours_prob.mean():.3f})\n\n')
    f.write(f'## Test Det Rate (>0.5): **{mah_det*100:.1f}%**\n\n')
    f.write(f'## Bimodality: BC={bc:.4f} ({"bimodal" if bc>5/9 else "unimodal"})\n\n')
    f.write(f'## Correlation with Instability: r={rho:.3f}\n\n')
    f.write('## Three-way Comparison\n\n')
    f.write('| Method | CV AUROC | Test det | Our recall |\n|---|---|---|---|\n')
    f.write(f'| Instability 24-dim | {auc_inst:.4f} | ~26% | ~98% |\n')
    f.write(f'| Mahalanobis 20-dim (fixed) | {auc_cv:.4f} | {mah_det*100:.1f}% | {mah_recall*100:.1f}% |\n')
    if INST_PRED.exists() and len(ours_merged) > 0:
        f.write(f'| Ensemble (avg) | N/A | {ens_det*100:.1f}% | {ens_recall*100:.1f}% |\n')

    f.write('\n## 판정\n\n')
    if mah_recall > 0.90:
        f.write('**✓ Fix 성공, Mahalanobis approach 유효**\n')
    elif mah_recall > 0.70:
        f.write('**~ 부분 성공, 추가 tuning 가능**\n')
    else:
        f.write('**✗ Fix 실패, 근본 문제 있음**\n')

    f.write('\n## 출력 파일\n\n')
    for fn in ['pilot_distances_phase2scale.parquet', 'rf_model_phase2.pkl',
               'test_predictions_fixed.parquet', 'test_probability_distributions.png',
               'correlation_inst_vs_mah.png']:
        f.write(f'- [{fn}](./{fn})\n')

print(f"\n경과: {time.time()-t0:.1f}s")
print("완료 ✓")
