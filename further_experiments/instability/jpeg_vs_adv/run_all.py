"""
import os
Adversarial vs JPEG-only Real discriminability 검증.
Steps 1-7 all-in-one: generate → inference → classify → t-SNE → report.
"""
import os, sys, io, gc, time, traceback
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
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.manifold import TSNE
from tqdm import tqdm

BASE = Path('/home/student/GEN_AI_GTG')
REAL_DIR = BASE / 'detection/real_reference/center'
CROP_SCORES = BASE / 'detection/crop_ablation/scores'  # train adv features
OUT = BASE / 'detection/instability/jpeg_vs_adv'
OUT.mkdir(parents=True, exist_ok=True)

DETECTORS = ['npr', 'distildire', 'aide_genimage', 'xception',
             'siglip_dino', 'aide_progan', 'haywood', 'aide_sd14']
CROP_VARIANTS = ['face_crop_variable', 'random_crop_192_a', 'center_crop_192']
ALL_VARIANTS = ['full'] + CROP_VARIANTS
FEAT_NAMES = [f'{d}__{v}' for d in DETECTORS for v in CROP_VARIANTS]
JPEG_QS = [50, 60, 70, 80]

t0 = time.time()

# ── Step 1: JPEG-only Real 생성 ──────────────────────────────────────
print("=" * 70)
print("[1/7] JPEG-only Real 생성 (100 real × 4 quality = 400)")
print("=" * 70)

rng = np.random.RandomState(42)
all_pngs = sorted(REAL_DIR.glob('*.png'))
sel_idx = rng.choice(len(all_pngs), size=100, replace=False)
real_100 = [all_pngs[i] for i in sorted(sel_idx)]

VAR_BASE = OUT / 'variants'
fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

for q in JPEG_QS:
    for v in ALL_VARIANTS:
        (VAR_BASE / f'Q{q}' / v).mkdir(parents=True, exist_ok=True)

def gen_crop_variants(img, idx):
    h, w = 256, 256
    out = {'full': img.copy()}
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    lm = None
    if res.multi_face_landmarks:
        lm = [(l.x, l.y) for l in res.multi_face_landmarks[0].landmark]
    # face_crop_variable (center fallback)
    if lm:
        xs = [l[0]*w for l in lm]; ys = [l[1]*h for l in lm]
        x_min, x_max = min(xs), max(xs); y_min, y_max = min(ys), max(ys)
        fw, fh = x_max-x_min, y_max-y_min
        cx, cy = (x_min+x_max)/2, (y_min+y_max)/2
        size = max(fw, fh) * 1.3
        x1 = max(0, int(cx-size/2)); y1 = max(0, int(cy-size/2))
        x2 = min(w, int(cx+size/2)); y2 = min(h, int(cy+size/2))
        crop = img[y1:y2, x1:x2]
        out['face_crop_variable'] = crop if crop.size > 0 else img[32:224, 32:224]
    else:
        out['face_crop_variable'] = img[32:224, 32:224]
    o = (256-192)//2
    out['center_crop_192'] = img[o:o+192, o:o+192]
    r = np.random.RandomState(42 + idx)
    mx = 256 - 192
    ox, oy = int(r.randint(0, mx+1)), int(r.randint(0, mx+1))
    out['random_crop_192_a'] = img[oy:oy+192, ox:ox+192]
    return out

n_created = 0
for idx, rp in enumerate(tqdm(real_100, desc='JPEG aug', unit='img')):
    img_orig = cv2.imread(str(rp))
    if img_orig is None: continue
    if img_orig.shape[:2] != (256, 256):
        img_orig = cv2.resize(img_orig, (256, 256), interpolation=cv2.INTER_LINEAR)
    stem = rp.stem
    for q in JPEG_QS:
        _, buf = cv2.imencode('.jpg', img_orig, [cv2.IMWRITE_JPEG_QUALITY, q])
        img_jpeg = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        variants = gen_crop_variants(img_jpeg, idx)
        for v_name, v_img in variants.items():
            vp = VAR_BASE / f'Q{q}' / v_name / f'{stem}.png'
            if not vp.exists():
                cv2.imwrite(str(vp), v_img)
                n_created += 1
fm.close()
print(f"  생성: {n_created} files\n")

# ── Step 2: Inference ────────────────────────────────────────────────
print("=" * 70)
print("[2/7] Inference (400 imgs × 8 det × 4 var)")
print("=" * 70)

_orig = sys.argv; sys.argv = ['x', '--detectors', 'dummy']
sys.path.insert(0, '/home/student/GEN_AI_GTG/detection/scores')
import run_inference as ri
sys.argv = _orig

import torch
from PIL import Image as PILImage
DEVICE = torch.device('cuda:0')
SCORES_DIR = OUT / 'scores'
SCORES_DIR.mkdir(parents=True, exist_ok=True)

for det_name in DETECTORS:
    all_done = all((SCORES_DIR / f'{det_name}__Q{q}__{v}.parquet').exists()
                   for q in JPEG_QS for v in ALL_VARIANTS)
    if all_done:
        print(f"  ⏭ {det_name} skip"); continue
    spec = ri.DETECTOR_SPECS[det_name]
    print(f"  ▶ {det_name} 로드...", end=' ', flush=True)
    model = spec['build']()
    print("OK")
    for q in JPEG_QS:
        for v in ALL_VARIANTS:
            out_p = SCORES_DIR / f'{det_name}__Q{q}__{v}.parquet'
            if out_p.exists(): continue
            paths = sorted((VAR_BASE / f'Q{q}' / v).glob('*.png'))
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
    print(f"    {det_name} 완료")
print()

# ── Step 3: Feature 추출 ─────────────────────────────────────────────
print("=" * 70)
print("[3/7] Instability features 추출")
print("=" * 70)

# JPEG-only real features (4 quality × 100 = 400)
jpeg_feats = []; jpeg_labels = []
for q in JPEG_QS:
    feats = []
    for det in DETECTORS:
        df_full = pd.read_parquet(SCORES_DIR / f'{det}__Q{q}__full.parquet').sort_values('filename')
        for v in CROP_VARIANTS:
            df_v = pd.read_parquet(SCORES_DIR / f'{det}__Q{q}__{v}.parquet').sort_values('filename')
            feats.append(np.abs(df_full['score'].to_numpy() - df_v['score'].to_numpy()))
    X_q = np.stack(feats, axis=1)  # (100, 24)
    jpeg_feats.append(X_q)
    jpeg_labels.extend([f'Q{q}'] * len(X_q))
X_jpeg = np.vstack(jpeg_feats)  # (400, 24)
print(f"  JPEG-only real: {X_jpeg.shape}")

# save
rows_j = []
for i in range(X_jpeg.shape[0]):
    row = {'idx': i, 'quality': jpeg_labels[i]}
    for j, fn in enumerate(FEAT_NAMES):
        row[fn] = float(X_jpeg[i, j])
    rows_j.append(row)
pd.DataFrame(rows_j).to_parquet(OUT / 'jpeg_only_real_features.parquet', index=False)

# Adversarial fake features (기존 crop_ablation에서)
adv_feats = []
for det in DETECTORS:
    df_full = pd.read_parquet(CROP_SCORES / f'{det}__adversarial_fake__full.parquet').sort_values('filename')
    for v in CROP_VARIANTS:
        df_v = pd.read_parquet(CROP_SCORES / f'{det}__adversarial_fake__{v}.parquet').sort_values('filename')
        adv_feats.append(np.abs(df_full['score'].to_numpy() - df_v['score'].to_numpy()))
X_adv_all = np.stack(adv_feats, axis=1)  # (1002, 24)
# random 400 sampling
adv_idx = rng.choice(len(X_adv_all), size=400, replace=False)
X_adv = X_adv_all[adv_idx]
print(f"  Adversarial (sampled): {X_adv.shape}")

rows_a = []
for i in range(X_adv.shape[0]):
    row = {'idx': i, 'quality': 'adversarial'}
    for j, fn in enumerate(FEAT_NAMES):
        row[fn] = float(X_adv[i, j])
    rows_a.append(row)
pd.DataFrame(rows_a).to_parquet(OUT / 'adv_features_sample.parquet', index=False)

# ── Step 4: 분류 실험 ────────────────────────────────────────────────
print(f"\n{'='*70}")
print("[4/7] Binary classification (JPEG-only real=0, adversarial=1)")
print("=" * 70)

X = np.vstack([X_jpeg, X_adv])
y = np.array([0]*len(X_jpeg) + [1]*len(X_adv))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
for clf_name, clf_cls in [
    ('logistic_regression', lambda: LogisticRegression(max_iter=2000, random_state=42)),
    ('random_forest',       lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
]:
    proba_all = cross_val_predict(clf_cls(), X, y, cv=skf, method='predict_proba')[:, 1]
    pred_all = (proba_all >= 0.5).astype(int)
    auc = roc_auc_score(y, proba_all)
    acc = accuracy_score(y, pred_all)
    print(f"  {clf_name:<24}  AUROC={auc:.4f}  Acc={acc:.4f}")
    results.append({'model': clf_name, 'auroc': round(auc, 4), 'accuracy': round(acc, 4)})

    # per-quality breakdown (JPEG side)
    if clf_name == 'random_forest':
        for qi, q in enumerate(JPEG_QS):
            mask = slice(qi*100, (qi+1)*100)
            q_proba = proba_all[mask]
            q_pred = pred_all[mask]
            fp = int(q_pred.sum())  # JPEG-only real이 adversarial로 오판
            print(f"    Q{q}: {fp}/100 오판 (FP), mean(prob)={q_proba.mean():.3f}")

# ── Step 5: Feature importance ────────────────────────────────────────
print(f"\n{'='*70}")
print("[5/7] Feature importance (RF)")
print("=" * 70)

clf_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X, y)
imp = clf_rf.feature_importances_
top_idx = np.argsort(-imp)[:5]
print(f"  Top 5 features:")
for rank, fi in enumerate(top_idx):
    print(f"    {rank+1}. {FEAT_NAMES[fi]:40s}  importance={imp[fi]:.4f}")

# 이전 real vs adv RF의 top features와 비교
clf_orig = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
# real_ref(original) vs adv
X_real_orig = []
for det in DETECTORS:
    df_full = pd.read_parquet(CROP_SCORES / f'{det}__real_reference__full.parquet').sort_values('filename')
    for v in CROP_VARIANTS:
        df_v = pd.read_parquet(CROP_SCORES / f'{det}__real_reference__{v}.parquet').sort_values('filename')
        X_real_orig.append(np.abs(df_full['score'].to_numpy() - df_v['score'].to_numpy()))
X_real_orig = np.stack(X_real_orig, axis=1)
X_orig = np.vstack([X_real_orig, X_adv_all])
y_orig = np.array([0]*len(X_real_orig) + [1]*len(X_adv_all))
clf_orig.fit(X_orig, y_orig)
imp_orig = clf_orig.feature_importances_
top_orig = np.argsort(-imp_orig)[:5]
print(f"\n  비교: real(원본) vs adv RF top 5:")
for rank, fi in enumerate(top_orig):
    print(f"    {rank+1}. {FEAT_NAMES[fi]:40s}  importance={imp_orig[fi]:.4f}")

overlap = set(top_idx[:5]) & set(top_orig[:5])
print(f"\n  Top-5 겹침: {len(overlap)}/5  → {'동일 feature 의존' if len(overlap) >= 3 else '다른 feature 사용'}")

# ── Step 6: t-SNE ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("[6/7] t-SNE 시각화")
print("=" * 70)

labels_all = jpeg_labels + ['adversarial'] * 400
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
emb = tsne.fit_transform(X)

fig, ax = plt.subplots(figsize=(9, 7))
colors = {'Q50': '#e41a1c', 'Q60': '#ff7f00', 'Q70': '#984ea3', 'Q80': '#377eb8', 'adversarial': '#4daf4a'}
for lbl in ['Q50', 'Q60', 'Q70', 'Q80', 'adversarial']:
    mask = [l == lbl for l in labels_all]
    idx_m = np.where(mask)[0]
    marker = '*' if lbl == 'adversarial' else 'o'
    size = 80 if lbl == 'adversarial' else 25
    ax.scatter(emb[idx_m, 0], emb[idx_m, 1], c=colors[lbl], label=lbl,
              marker=marker, s=size, alpha=0.7,
              edgecolors='black' if lbl == 'adversarial' else 'none', linewidths=0.3)
ax.legend(fontsize=10, frameon=True, fancybox=True)
ax.set_title(f't-SNE: JPEG-only real (Q50-Q80) vs Adversarial fake\n'
             f'RF 5-fold AUROC={results[1]["auroc"]:.3f}', fontsize=12)
plt.tight_layout()
plt.savefig(OUT / 'tsne_jpeg_vs_adv.png', dpi=130, bbox_inches='tight')
plt.close()
print("  저장: tsne_jpeg_vs_adv.png")

# ── Step 7: 판정 + 리포트 ────────────────────────────────────────────
print(f"\n{'='*70}")
print("[7/7] 판정")
print("=" * 70)

best_auc = results[1]['auroc']  # RF
if best_auc > 0.85:
    verdict = '✓ DISCRIMINABLE — JPEG-balanced training will work'
elif best_auc > 0.65:
    verdict = '~ PARTIALLY DISCRIMINABLE — JPEG-balanced training will partially help'
else:
    verdict = '✗ NOT DISCRIMINABLE — Need different approach'
print(f"  RF AUROC = {best_auc:.4f}")
print(f"  판정: {verdict}")

with open(OUT / 'discriminability_results.md', 'w') as f:
    f.write('# Adversarial vs JPEG-only Real Discriminability\n\n')
    f.write(f'> 400 JPEG-only real (100 imgs × Q50/Q60/Q70/Q80) vs 400 adversarial fake\n')
    f.write(f'> 24-dim instability features, 5-fold stratified CV\n\n')
    f.write('## Classification Results\n\n')
    f.write('| Model | AUROC | Accuracy |\n|---|---|---|\n')
    for r in results:
        f.write(f'| {r["model"]} | {r["auroc"]:.4f} | {r["accuracy"]:.4f} |\n')
    f.write(f'\n## 판정: **{verdict}**\n\n')
    f.write('## Feature Importance (RF, JPEG-only vs Adv)\n\n')
    f.write('| rank | feature | importance |\n|---|---|---|\n')
    for rank, fi in enumerate(top_idx[:5]):
        f.write(f'| {rank+1} | {FEAT_NAMES[fi]} | {imp[fi]:.4f} |\n')
    f.write(f'\n### 비교: real(원본) vs adv RF top-5\n\n')
    f.write('| rank | feature | importance |\n|---|---|---|\n')
    for rank, fi in enumerate(top_orig[:5]):
        f.write(f'| {rank+1} | {FEAT_NAMES[fi]} | {imp_orig[fi]:.4f} |\n')
    f.write(f'\nTop-5 겹침: **{len(overlap)}/5**\n\n')
    f.write('## 해석\n\n')
    if best_auc > 0.85:
        f.write('Adversarial fake와 JPEG-only real은 instability feature space에서 **구분 가능**.\n')
        f.write('JPEG-balanced training으로 false positive를 줄일 수 있음.\n')
    elif best_auc > 0.65:
        f.write('부분적으로 구분 가능하지만 겹치는 영역이 있음.\n')
        f.write('JPEG-balanced training으로 일부 개선 가능하나 완전한 분리는 어려움.\n')
    else:
        f.write('구분 불가능. Instability feature 자체가 JPEG artifact에 dominated.\n')
        f.write('다른 접근 필요 (raw image feature, 또는 JPEG quality를 control variable로).\n')
    f.write('\n## 출력 파일\n\n')
    f.write('- [jpeg_only_real_features.parquet](./jpeg_only_real_features.parquet)\n')
    f.write('- [adv_features_sample.parquet](./adv_features_sample.parquet)\n')
    f.write('- [tsne_jpeg_vs_adv.png](./tsne_jpeg_vs_adv.png)\n')

print(f'\n경과: {time.time()-t0:.1f}s')
print('완료 ✓')
