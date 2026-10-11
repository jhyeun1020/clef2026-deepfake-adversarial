"""
import os
ε=8 vs ε=2 비교 분석.
- Perturbation spatial distribution (face ratio, patch_std)
- Instability features (crop variants) + RF prediction
- 3-way comparison (clean / eps2 / eps8)
- Visualizations + report
"""
import os, sys, gc, io, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('HF_TOKEN', 'REMOVED_TOKEN_USE_ENV_VAR')

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestClassifier

BASE = Path('/home/student/GEN_AI_GTG')
CLEAN_DIR   = BASE / 'generation/outputs/final_1002'
EPS2_DIR    = BASE / 'generation/submit/third_sub/256'
EPS8_DIR    = BASE / 'detection/epsilon_ablation/adv_eps8/jpeg_q80'
REAL_DIR    = BASE / 'detection/real_reference/center'
CROP_SCORES = BASE / 'detection/crop_ablation/scores'
OUT = BASE / 'detection/epsilon_ablation'

# 100 keys
with open(OUT / 'adv_eps8/selected_keys.txt') as f:
    keys_100 = [l.strip() for l in f if l.strip()]
print(f"keys: {len(keys_100)}")

GRID = 4; PATCH = 64
t0 = time.time()

# ── 1. Perturbation 공간 분포 (ε=2 vs ε=8) ──────────────────────────
print("=" * 70)
print("[1/5] Perturbation 공간 분포")
print("=" * 70)

fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def apply_jpeg_q80(img):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def get_face_mask(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    mask = np.zeros((256, 256), dtype=np.uint8)
    if not res.multi_face_landmarks: return None
    pts = np.array([(int(l.x*256), int(l.y*256)) for l in res.multi_face_landmarks[0].landmark], dtype=np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
    return mask

def patch_std(mag):
    vals = []
    for r in range(GRID):
        for c in range(GRID):
            vals.append(mag[r*PATCH:(r+1)*PATCH, c*PATCH:(c+1)*PATCH].mean())
    return float(np.std(vals))

pert_rows = []
for key in keys_100:
    clean = cv2.imread(str(CLEAN_DIR / f'{key}.png'))
    eps2  = cv2.imread(str(EPS2_DIR / f'{key}.png'))
    eps8  = cv2.imread(str(EPS8_DIR / f'{key}.png'))
    if clean is None or eps2 is None or eps8 is None: continue
    clean_q80 = apply_jpeg_q80(clean)
    face_mask = get_face_mask(clean)
    if face_mask is None: continue

    for label, adv_img in [('eps2', eps2), ('eps8', eps8)]:
        pert = adv_img.astype(np.float32) - clean_q80.astype(np.float32)
        mag = np.abs(pert).mean(axis=2)
        fp = mag[face_mask == 1]; nfp = mag[face_mask == 0]
        ratio = float(fp.mean()) / (float(nfp.mean()) + 1e-8)
        pert_rows.append({
            'key': key, 'epsilon': label,
            'linf': float(np.abs(pert).max()),
            'l1_mean': float(np.abs(pert).mean()),
            'face_mean': float(fp.mean()), 'nonface_mean': float(nfp.mean()),
            'face_nonface_ratio': round(ratio, 4),
            'patch_std': round(patch_std(mag), 4),
        })
fm.close()

pert_df = pd.DataFrame(pert_rows)
pert_df.to_csv(OUT / 'perturbation_comparison.csv', index=False)

print(f"\n  {'metric':<20} {'ε=2':>12} {'ε=8':>12}")
print("  " + "-" * 46)
for metric in ['linf', 'l1_mean', 'face_nonface_ratio', 'patch_std']:
    e2 = pert_df[pert_df['epsilon']=='eps2'][metric].mean()
    e8 = pert_df[pert_df['epsilon']=='eps8'][metric].mean()
    print(f"  {metric:<20} {e2:>12.4f} {e8:>12.4f}")

# ── 2. Crop variants 생성 + inference (ε=8) ──────────────────────────
print(f"\n{'='*70}")
print("[2/5] ε=8 crop variants 생성 + 8 detectors inference")
print("=" * 70)

VAR_DIR = OUT / 'eps8_variants'
VARIANTS = ['full', 'face_crop_variable', 'center_crop_192', 'random_crop_192_a']
for v in VARIANTS:
    (VAR_DIR / v).mkdir(parents=True, exist_ok=True)

fm2 = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
for idx, key in enumerate(keys_100):
    img = cv2.imread(str(EPS8_DIR / f'{key}.png'))
    if img is None: continue
    # full
    cv2.imwrite(str(VAR_DIR / 'full' / f'{key}.png'), img)
    # face crop variable (center fallback)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = fm2.process(rgb)
    if res.multi_face_landmarks:
        lm = [(l.x, l.y) for l in res.multi_face_landmarks[0].landmark]
        h, w = 256, 256
        xs = [l[0]*w for l in lm]; ys = [l[1]*h for l in lm]
        cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
        size = max(max(xs)-min(xs), max(ys)-min(ys)) * 1.3
        x1, y1 = max(0, int(cx-size/2)), max(0, int(cy-size/2))
        x2, y2 = min(w, int(cx+size/2)), min(h, int(cy+size/2))
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(str(VAR_DIR / 'face_crop_variable' / f'{key}.png'),
                    crop if crop.size > 0 else img[32:224, 32:224])
    else:
        cv2.imwrite(str(VAR_DIR / 'face_crop_variable' / f'{key}.png'), img[32:224, 32:224])
    # center 192
    o = (256-192)//2
    cv2.imwrite(str(VAR_DIR / 'center_crop_192' / f'{key}.png'), img[o:o+192, o:o+192])
    # random crop
    r = np.random.RandomState(42 + idx); mx = 256-192
    ox, oy = int(r.randint(0, mx+1)), int(r.randint(0, mx+1))
    cv2.imwrite(str(VAR_DIR / 'random_crop_192_a' / f'{key}.png'), img[oy:oy+192, ox:ox+192])
fm2.close()
print(f"  variants 생성 완료")

# Inference
_orig = sys.argv; sys.argv = ['x', '--detectors', 'dummy']
sys.path.insert(0, '/home/student/GEN_AI_GTG/detection/scores')
import run_inference as ri
sys.argv = _orig
import torch
from PIL import Image as PILImage

DETECTORS = ['npr', 'distildire', 'aide_genimage', 'xception',
             'siglip_dino', 'aide_progan', 'haywood', 'aide_sd14']
CROP_VARIANTS = ['face_crop_variable', 'random_crop_192_a', 'center_crop_192']
FEAT_NAMES = [f'{d}__{v}' for d in DETECTORS for v in CROP_VARIANTS]

SCORES_DIR = OUT / 'scores'
SCORES_DIR.mkdir(parents=True, exist_ok=True)

for det_name in DETECTORS:
    all_done = all((SCORES_DIR / f'{det_name}__{v}.parquet').exists() for v in VARIANTS)
    if all_done: continue
    spec = ri.DETECTOR_SPECS[det_name]
    print(f"  ▶ {det_name}...", end=' ', flush=True)
    model = spec['build']()
    for v in VARIANTS:
        out_p = SCORES_DIR / f'{det_name}__{v}.parquet'
        if out_p.exists(): continue
        paths = sorted((VAR_DIR / v).glob('*.png'))
        pils = [PILImage.open(p).convert('RGB') for p in paths]
        batch = 1 if v == 'face_crop_variable' else spec.get('batch', 32)
        with torch.no_grad():
            kind = spec['kind']
            if kind == 'wrapper':     scores = ri.score_simple_wrapper(model, pils, spec['ai_idx'], batch)
            elif kind == 'recce_ucf': scores = ri.score_recce_ucf(model, pils, batch)
            elif kind == 'aide':      scores = ri.score_aide(model, pils)
            elif kind == 'distildire':scores = ri.score_distildire(model, pils)
        pd.DataFrame({'filename': [p.name for p in paths], 'score': scores.astype(np.float32)}).to_parquet(out_p, index=False)
    if spec['kind'] == 'distildire':
        try: model[0].cpu(); model[1].cpu()
        except: pass
    else:
        try: model.cpu()
        except: pass
    del model; torch.cuda.empty_cache(); gc.collect()
    print("OK")

# ── 3. Instability features ──────────────────────────────────────────
print(f"\n{'='*70}")
print("[3/5] Instability features (ε=8) + RF prediction")
print("=" * 70)

# ε=8 features
feats_eps8 = []
for det in DETECTORS:
    df_full = pd.read_parquet(SCORES_DIR / f'{det}__full.parquet').sort_values('filename')
    for v in CROP_VARIANTS:
        df_v = pd.read_parquet(SCORES_DIR / f'{det}__{v}.parquet').sort_values('filename')
        feats_eps8.append(np.abs(df_full['score'].to_numpy() - df_v['score'].to_numpy()))
X_eps8 = np.stack(feats_eps8, axis=1)
print(f"  X_eps8: {X_eps8.shape}")

# ε=2 features (from crop_ablation — 전체 1002장에서 같은 100장 추출)
def load_adv_features_for_keys(keys):
    feats = []
    for det in DETECTORS:
        df_full = pd.read_parquet(CROP_SCORES / f'{det}__adversarial_fake__full.parquet').sort_values('filename')
        stems_full = [Path(f).stem for f in df_full['filename']]
        for v in CROP_VARIANTS:
            df_v = pd.read_parquet(CROP_SCORES / f'{det}__adversarial_fake__{v}.parquet').sort_values('filename')
            full_s = df_full['score'].to_numpy()
            v_s = df_v['score'].to_numpy()
            inst = np.abs(full_s - v_s)
            # 100장만 추출
            col = []
            for k in keys:
                idx = stems_full.index(k) if k in stems_full else -1
                col.append(inst[idx] if idx >= 0 else 0.0)
            feats.append(col)
    return np.array(feats).T

X_eps2 = load_adv_features_for_keys(keys_100)
print(f"  X_eps2: {X_eps2.shape}")

# Real + ε=2 전체 RF (이전 instability 분석과 동일)
def build_train_X(ds):
    feats = []
    for det in DETECTORS:
        df_full = pd.read_parquet(CROP_SCORES / f'{det}__{ds}__full.parquet').sort_values('filename')
        for v in CROP_VARIANTS:
            df_v = pd.read_parquet(CROP_SCORES / f'{det}__{ds}__{v}.parquet').sort_values('filename')
            feats.append(np.abs(df_full['score'].to_numpy() - df_v['score'].to_numpy()))
    return np.stack(feats, axis=1)

X_real = build_train_X('real_reference')
X_adv_all = build_train_X('adversarial_fake')
X_train = np.vstack([X_real, X_adv_all])
y_train = np.array([0]*len(X_real) + [1]*len(X_adv_all))
clf_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, y_train)

prob_eps2 = clf_rf.predict_proba(X_eps2)[:, 1]
prob_eps8 = clf_rf.predict_proba(X_eps8)[:, 1]
det_rate_eps2 = (prob_eps2 >= 0.5).mean()
det_rate_eps8 = (prob_eps8 >= 0.5).mean()
print(f"\n  RF detection rate: ε=2 → {det_rate_eps2*100:.1f}% | ε=8 → {det_rate_eps8*100:.1f}%")
print(f"  RF mean prob:     ε=2 → {prob_eps2.mean():.3f}  | ε=8 → {prob_eps8.mean():.3f}")

# ── 4. Visualizations ────────────────────────────────────────────────
print(f"\n{'='*70}")
print("[4/5] Visualizations")
print("=" * 70)

# 4a. Face/non-face ratio by epsilon
fig, ax = plt.subplots(figsize=(6, 5))
data = [pert_df[pert_df['epsilon']=='eps2']['face_nonface_ratio'].values,
        pert_df[pert_df['epsilon']=='eps8']['face_nonface_ratio'].values]
bp = ax.boxplot(data, labels=['ε=2/255', 'ε=8/255'], patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('#1f77b4'); bp['boxes'][1].set_facecolor('#d62728')
for b in bp['boxes']: b.set_alpha(0.5)
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('face / non-face perturbation ratio')
ax.set_title('Face/non-face ratio by ε')
plt.tight_layout()
plt.savefig(OUT / 'ratio_by_epsilon.png', dpi=130, bbox_inches='tight')
plt.close()

# 4b. Perturbation heatmap comparison (5장)
fig, axes = plt.subplots(5, 3, figsize=(9, 15))
for i in range(5):
    key = keys_100[i]
    clean = cv2.imread(str(CLEAN_DIR / f'{key}.png'))
    eps2i = cv2.imread(str(EPS2_DIR / f'{key}.png'))
    eps8i = cv2.imread(str(EPS8_DIR / f'{key}.png'))
    clean_q80 = apply_jpeg_q80(clean)
    mag2 = np.abs(eps2i.astype(np.float32) - clean_q80.astype(np.float32)).mean(axis=2)
    mag8 = np.abs(eps8i.astype(np.float32) - clean_q80.astype(np.float32)).mean(axis=2)
    vmax = max(mag2.max(), mag8.max(), 1)
    axes[i, 0].imshow(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)); axes[i, 0].set_title(f'{key[:16]}', fontsize=8)
    axes[i, 1].imshow(mag2, cmap='hot', vmin=0, vmax=vmax); axes[i, 1].set_title(f'ε=2 pert', fontsize=8)
    axes[i, 2].imshow(mag8, cmap='hot', vmin=0, vmax=vmax); axes[i, 2].set_title(f'ε=8 pert', fontsize=8)
    for ax in axes[i]: ax.axis('off')
plt.suptitle('Perturbation magnitude: ε=2 vs ε=8 (same colorscale)', fontsize=11)
plt.tight_layout()
plt.savefig(OUT / 'perturbation_heatmap_comparison.png', dpi=100, bbox_inches='tight')
plt.close()

# 4c. RF probability distribution
fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(0, 1, 31)
ax.hist(prob_eps2, bins=bins, alpha=0.6, color='#1f77b4', label=f'ε=2 (det={det_rate_eps2*100:.0f}%)')
ax.hist(prob_eps8, bins=bins, alpha=0.6, color='#d62728', label=f'ε=8 (det={det_rate_eps8*100:.0f}%)')
ax.axvline(0.5, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Adversarial probability (RF)'); ax.set_ylabel('count')
ax.set_title('RF model prediction: ε=2 vs ε=8 (model trained on ε=2)')
ax.legend()
plt.tight_layout()
plt.savefig(OUT / 'rf_probability_by_epsilon.png', dpi=130, bbox_inches='tight')
plt.close()

# 4d. Instability feature scatter (eps2 vs eps8 mean)
fig, ax = plt.subplots(figsize=(7, 5))
for i, fn in enumerate(FEAT_NAMES):
    ax.scatter(X_eps2[:, i].mean(), X_eps8[:, i].mean(), s=50, alpha=0.7)
    ax.annotate(fn.split('__')[0][:4] + '_' + fn.split('__')[1][:4],
                (X_eps2[:, i].mean(), X_eps8[:, i].mean()), fontsize=6)
lim = max(X_eps2.mean(axis=0).max(), X_eps8.mean(axis=0).max()) * 1.1
ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
ax.set_xlabel('ε=2 mean instability'); ax.set_ylabel('ε=8 mean instability')
ax.set_title('Feature-level instability: ε=2 vs ε=8')
plt.tight_layout()
plt.savefig(OUT / 'epsilon_comparison_scatter.png', dpi=130, bbox_inches='tight')
plt.close()

print("  시각화 저장 완료")

# ── 5. Report ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("[5/5] Report")
print("=" * 70)

e2_ratio = pert_df[pert_df['epsilon']=='eps2']['face_nonface_ratio'].mean()
e8_ratio = pert_df[pert_df['epsilon']=='eps8']['face_nonface_ratio'].mean()
e2_pstd = pert_df[pert_df['epsilon']=='eps2']['patch_std'].mean()
e8_pstd = pert_df[pert_df['epsilon']=='eps8']['patch_std'].mean()

print(f"  Q1 (spatial): ε=2 ratio={e2_ratio:.3f}, ε=8 ratio={e8_ratio:.3f}")
if e8_ratio > 1.5:
    v1 = '✓ 큰 ε에서 공격이 localized됨 (EoT 한계)'
elif e8_ratio < 1.25:
    v1 = '✗ EoT가 ε에 무관하게 균일화 유지'
else:
    v1 = '~ 약간 localized'
print(f"    → {v1}")

print(f"  Q2 (RF generalization): ε=8 det rate={det_rate_eps8*100:.1f}%")
if det_rate_eps8 > 0.90:
    v2 = '✓ 기존 모델이 ε=8도 잘 잡음'
elif det_rate_eps8 > 0.60:
    v2 = '~ 부분적 generalization'
else:
    v2 = '✗ ε=8 공격 놓침'
print(f"    → {v2}")

with open(OUT / 'report.md', 'w') as f:
    f.write('# ε Ablation: ε=2/255 vs ε=8/255\n\n')
    f.write(f'> 100 paired images, same PGD pipeline (attack_1002_all_nonface.py)\n')
    f.write(f'> ε=2: α=0.5/255, ε=8: α=2.0/255, 30 iterations, 13-detector ensemble\n\n')
    f.write('---\n\n## 1. Perturbation Statistics\n\n')
    f.write(f'| metric | ε=2 | ε=8 |\n|---|---|---|\n')
    for m in ['linf', 'l1_mean', 'face_nonface_ratio', 'patch_std']:
        e2v = pert_df[pert_df['epsilon']=='eps2'][m].mean()
        e8v = pert_df[pert_df['epsilon']=='eps8'][m].mean()
        f.write(f'| {m} | {e2v:.4f} | {e8v:.4f} |\n')
    f.write(f'\n## 2. Spatial Distribution\n\n')
    f.write(f'- ε=2 face/non-face ratio: **{e2_ratio:.3f}**\n')
    f.write(f'- ε=8 face/non-face ratio: **{e8_ratio:.3f}**\n')
    f.write(f'- **{v1}**\n\n')
    f.write(f'## 3. RF Model Generalization\n\n')
    f.write(f'Model trained on real vs ε=2 adversarial:\n\n')
    f.write(f'| | det rate (>0.5) | mean prob |\n|---|---|---|\n')
    f.write(f'| ε=2 (100장) | {det_rate_eps2*100:.1f}% | {prob_eps2.mean():.3f} |\n')
    f.write(f'| ε=8 (100장) | {det_rate_eps8*100:.1f}% | {prob_eps8.mean():.3f} |\n\n')
    f.write(f'**{v2}**\n\n')
    f.write(f'## 4. Instability Feature Comparison\n\n')
    f.write(f'ε=2 mean instability: {X_eps2.mean():.4f}\n')
    f.write(f'ε=8 mean instability: {X_eps8.mean():.4f}\n')
    f.write(f'Ratio (ε=8/ε=2): {X_eps8.mean()/max(X_eps2.mean(),1e-8):.2f}x\n\n')
    f.write('## 출력 파일\n\n')
    for fn in ['perturbation_comparison.csv', 'ratio_by_epsilon.png',
               'perturbation_heatmap_comparison.png', 'rf_probability_by_epsilon.png',
               'epsilon_comparison_scatter.png']:
        f.write(f'- [{fn}](./{fn})\n')

print(f"\n경과: {time.time()-t0:.1f}s")
print("완료 ✓")
