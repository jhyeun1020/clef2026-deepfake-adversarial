"""
import os
Task 1: JPEG Q80 augmentation (20장) + crop variants 생성
Task 2: Test set 24,404장 JPEG quality 추정 (8×8 block artifact)
Task 3: Quality vs adversarial prob 상관분석
"""
import os, sys, io, csv, time, gc, traceback
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
import pandas as pd
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

BASE = Path('/home/student/GEN_AI_GTG')
CTRL = BASE / 'detection/instability/control_experiment'
REAL_DIR = BASE / 'detection/real_reference/center'
TEST_DIR = BASE / 'detection/ImageCLEF2026-DeepFakeDetection-Tes/Data/Images_Detection'
PRED_PQ  = BASE / 'detection/instability/test_application/test_instability_predictions.parquet'

# ── Task 1: JPEG Q80 augmentation + variants ─────────────────────────
print("=" * 70)
print("Task 1: JPEG Q80 augmentation")
print("=" * 70)

AUG_DIR = CTRL / 'augmented_real/jpeg_q80'
VAR_DIR = CTRL / 'variants/jpeg_q80'
VARIANTS = ['full', 'face_crop_variable', 'center_crop_192', 'random_crop_192_a']
AUG_DIR.mkdir(parents=True, exist_ok=True)
for v in VARIANTS:
    (VAR_DIR / v).mkdir(parents=True, exist_ok=True)

# 기존 control experiment와 동일한 20장 sampling
rng = np.random.RandomState(42)
all_pngs = sorted(REAL_DIR.glob('*.png'))
selected = rng.choice(len(all_pngs), size=20, replace=False)
real_paths = [all_pngs[i] for i in sorted(selected)]

fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

print(f"  Real 20장 → Q80 + 4 variants 생성...")
for idx, rp in enumerate(real_paths):
    stem = rp.stem
    img = cv2.imread(str(rp))
    if img.shape[:2] != (256, 256):
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    # Q80
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    img_q80 = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    cv2.imwrite(str(AUG_DIR / f'{stem}.png'), img_q80)

    # face detection
    rgb = cv2.cvtColor(img_q80, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    lm = None
    if res.multi_face_landmarks:
        lm = [(l.x, l.y) for l in res.multi_face_landmarks[0].landmark]

    # full
    cv2.imwrite(str(VAR_DIR / 'full' / f'{stem}.png'), img_q80)

    # face_crop_variable (center fallback)
    h, w = 256, 256
    if lm:
        xs = [l[0]*w for l in lm]; ys = [l[1]*h for l in lm]
        x_min, x_max = min(xs), max(xs); y_min, y_max = min(ys), max(ys)
        fw, fh = x_max-x_min, y_max-y_min
        cx, cy = (x_min+x_max)/2, (y_min+y_max)/2
        size = max(fw, fh) * 1.3
        x1 = max(0, int(cx-size/2)); y1 = max(0, int(cy-size/2))
        x2 = min(w, int(cx+size/2)); y2 = min(h, int(cy+size/2))
        crop = img_q80[y1:y2, x1:x2]
        if crop.size == 0: crop = img_q80[32:224, 32:224]
    else:
        crop = img_q80[32:224, 32:224]
    cv2.imwrite(str(VAR_DIR / 'face_crop_variable' / f'{stem}.png'), crop)

    # center_crop_192
    o = (256-192)//2
    cv2.imwrite(str(VAR_DIR / 'center_crop_192' / f'{stem}.png'), img_q80[o:o+192, o:o+192])

    # random_crop_192_a
    r = np.random.RandomState(42 + idx)
    mx = 256 - 192
    ox, oy = int(r.randint(0, mx+1)), int(r.randint(0, mx+1))
    cv2.imwrite(str(VAR_DIR / 'random_crop_192_a' / f'{stem}.png'), img_q80[oy:oy+192, ox:ox+192])

fm.close()
print(f"  Q80 variants 생성 완료")

# Inference (단일 GPU)
print(f"  Inference 시작 (8 det × 4 var × 20 imgs)...")
os.environ.setdefault('HF_TOKEN', 'REMOVED_TOKEN_USE_ENV_VAR')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
_orig = sys.argv; sys.argv = ['x', '--detectors', 'dummy']
sys.path.insert(0, '/home/student/GEN_AI_GTG/detection/scores')
import run_inference as ri
sys.argv = _orig

import torch
from PIL import Image

DEVICE = torch.device('cuda:0')
SCORES_DIR = CTRL / 'scores'
DETECTORS = ['npr', 'distildire', 'aide_genimage', 'xception',
             'siglip_dino', 'aide_progan', 'haywood', 'aide_sd14']
CROP_VARIANTS = ['face_crop_variable', 'random_crop_192_a', 'center_crop_192']

for det_name in DETECTORS:
    all_done = all((SCORES_DIR / f'{det_name}__jpeg_q80__{v}.parquet').exists() for v in VARIANTS)
    if all_done: continue
    spec = ri.DETECTOR_SPECS[det_name]
    model = spec['build']()
    for v in VARIANTS:
        out_p = SCORES_DIR / f'{det_name}__jpeg_q80__{v}.parquet'
        if out_p.exists(): continue
        paths = sorted((VAR_DIR / v).glob('*.png'))
        pils = [Image.open(p).convert('RGB') for p in paths]
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

# Q80 instability features
feats_q80 = []
for det in DETECTORS:
    df_full = pd.read_parquet(SCORES_DIR / f'{det}__jpeg_q80__full.parquet').sort_values('filename')
    for v in CROP_VARIANTS:
        df_v = pd.read_parquet(SCORES_DIR / f'{det}__jpeg_q80__{v}.parquet').sort_values('filename')
        feats_q80.append(np.abs(df_full['score'].to_numpy() - df_v['score'].to_numpy()))
X_q80 = np.stack(feats_q80, axis=1)  # (20, 24)

# Load existing baselines
SUSPECT_STEMS = ['6488e0306ef3ca2f3c90', 'e819e3aad4a3226de7d0', 'ec634859158b8bcc121a',
                 '1b91da9e460c6a7d0cdc', 'c3957f266d7078d9e959']
ADV_STEMS = ['60feecd82812b3ca8519', '6229ffe049db851dd7f5', '5a647e1d6b497a8f4d0a',
             '6b94443c0a878d394159', '6c112186b0c66912301d']
REAL_STEMS = ['b5b10f28ca5ae7776c14', '4cbba497dc2a7fe4987a', '82b8053a47a4c2f0d74d',
              '986236cec8e28dffb100', '6f6f606dba8d02273619']

TEST_SCORES = BASE / 'detection/instability/test_application/scores'
def load_test_inst(stems):
    feats = []
    for det in DETECTORS:
        df_full = pd.read_parquet(TEST_SCORES / f'{det}__full.parquet').sort_values('filename')
        n2s_full = {Path(r['filename']).stem: float(r['score']) for _, r in df_full.iterrows()}
        for v in CROP_VARIANTS:
            df_v = pd.read_parquet(TEST_SCORES / f'{det}__{v}.parquet').sort_values('filename')
            n2s_v = {Path(r['filename']).stem: float(r['score']) for _, r in df_v.iterrows()}
            feats.append([abs(n2s_full.get(s,0) - n2s_v.get(s,0)) for s in stems])
    return np.array(feats).T

X_suspect = load_test_inst(SUSPECT_STEMS)
X_adv     = load_test_inst(ADV_STEMS)
X_real    = load_test_inst(REAL_STEMS)

def cos1d(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

q80_mean = X_q80.mean(axis=0)
sus_mean = X_suspect.mean(axis=0)
adv_mean = X_adv.mean(axis=0)
real_mean = X_real.mean(axis=0)

cos_q80_sus = cos1d(q80_mean, sus_mean)
cos_q80_adv = cos1d(q80_mean, adv_mean)
cos_q80_real = cos1d(q80_mean, real_mean)

print(f"\n  JPEG Q80 cosine similarity:")
print(f"    Q80 → suspect    : {cos_q80_sus:.3f}")
print(f"    Q80 → adv_baseline: {cos_q80_adv:.3f}")
print(f"    Q80 → real_baseline: {cos_q80_real:.3f}")
if cos_q80_sus > 0.9:
    print(f"    ⚠ Q80 matches adversarial pattern")
else:
    print(f"    ✓ Q80 distinct from adversarial pattern")

# 기존 similarity_matrix.csv에 추가
sim_csv = CTRL / 'similarity_matrix.csv'
sim_df = pd.read_csv(sim_csv)
for tgt_name, tgt_mat in [('suspect', X_suspect), ('adv_baseline', X_adv), ('real_baseline', X_real)]:
    a_n = X_q80 / (np.linalg.norm(X_q80, axis=1, keepdims=True) + 1e-12)
    b_n = tgt_mat / (np.linalg.norm(tgt_mat, axis=1, keepdims=True) + 1e-12)
    C = a_n @ b_n.T
    new_row = pd.DataFrame([{
        'group_a': 'jpeg_q80', 'group_b': tgt_name,
        'n_pairs': int(C.size), 'mean_cosine': round(float(C.mean()), 4),
        'std_cosine': round(float(C.std()), 4),
        'min': round(float(C.min()), 4), 'max': round(float(C.max()), 4),
    }])
    sim_df = pd.concat([sim_df, new_row], ignore_index=True)
sim_df.to_csv(sim_csv, index=False)
print(f"  similarity_matrix.csv 업데이트 완료\n")

# ── Task 2: Test set JPEG quality 추정 ───────────────────────────────
print("=" * 70)
print("Task 2: Test set JPEG quality 추정 (8×8 block artifact)")
print("=" * 70)

def estimate_jpeg_quality(img_bgr):
    """8×8 block boundary artifact 강도로 JPEG 흔적 추정.
    Returns (estimated_quality, trace_strength).
    trace_strength가 높을수록 JPEG 압축 흔적이 강함 (= quality 낮음)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    # 8×8 block boundary에서의 gradient 강도 vs 내부
    boundary_diffs = []
    internal_diffs = []
    for y in range(0, h-1):
        for x in range(0, w-1):
            d = abs(gray[y, x] - gray[y, x+1])
            if (x+1) % 8 == 0:  # block boundary
                boundary_diffs.append(d)
            else:
                internal_diffs.append(d)
    if not boundary_diffs or not internal_diffs:
        return -1, 0.0
    b_mean = np.mean(boundary_diffs)
    i_mean = np.mean(internal_diffs)
    # blocking artifact ratio (BAR)
    # BAR > 1 → block boundaries stand out → JPEG trace
    bar = b_mean / (i_mean + 1e-8)
    # Rough quality estimation from BAR
    # BAR ≈ 1.0 → no JPEG trace (high quality or never compressed)
    # BAR ≈ 1.05~1.15 → Q70-90
    # BAR > 1.2 → Q50 or lower
    if bar < 1.01:
        est_q = 95  # essentially no trace
    elif bar < 1.05:
        est_q = 85
    elif bar < 1.10:
        est_q = 75
    elif bar < 1.15:
        est_q = 65
    elif bar < 1.25:
        est_q = 55
    else:
        est_q = 45
    trace_strength = max(0, bar - 1.0)
    return est_q, round(trace_strength, 4)

# 24,404장 전체 or 6,002 샘플만?
# 전체 → 너무 느림 (full BAR analysis). 대신 sub-sampling으로 근사.
# → 전체 24,404장에 대해 간소화된 BAR (stride=2 sampling)
def estimate_jpeg_quality_fast(img_bgr, stride=2):
    """빠른 BAR 추정 — stride로 샘플링."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    # horizontal block boundaries
    bound = []; inter = []
    for y in range(0, h-1, stride):
        for x in range(0, w-1, stride):
            d = abs(gray[y, x] - gray[y, x+1])
            if (x+1) % 8 == 0: bound.append(d)
            else: inter.append(d)
    if not bound or not inter: return -1, 0.0
    bar = np.mean(bound) / (np.mean(inter) + 1e-8)
    if bar < 1.01:   eq = 95
    elif bar < 1.05: eq = 85
    elif bar < 1.10: eq = 75
    elif bar < 1.15: eq = 65
    elif bar < 1.25: eq = 55
    else:            eq = 45
    return eq, round(max(0, bar - 1.0), 4)

test_paths = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() == '.png')
print(f"  {len(test_paths)}장 분석 중...")
quality_rows = []
for p in tqdm(test_paths, desc='jpeg quality', unit='img'):
    img = cv2.imread(str(p))
    if img is None:
        quality_rows.append({'filename': p.name, 'estimated_quality': -1, 'jpeg_trace_strength': 0})
        continue
    eq, ts = estimate_jpeg_quality_fast(img)
    quality_rows.append({'filename': p.name, 'estimated_quality': eq, 'jpeg_trace_strength': ts})
q_df = pd.DataFrame(quality_rows)
q_df.to_csv(CTRL / 'test_set_jpeg_analysis.csv', index=False)
print(f"  저장: test_set_jpeg_analysis.csv ({len(q_df)} rows)")

# 분포 히스토그램
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
counts = Counter(q_df['estimated_quality'])
qs = sorted(counts.keys())
ax.bar(qs, [counts[q] for q in qs], width=8, color='#4c9ee0')
for q in qs:
    ax.text(q, counts[q] + 200, str(counts[q]), ha='center', fontsize=9)
ax.set_xlabel('Estimated JPEG quality')
ax.set_ylabel('Count')
ax.set_title(f'Test set JPEG quality distribution (n={len(q_df)})')

ax2 = axes[1]
ax2.hist(q_df['jpeg_trace_strength'], bins=50, color='#d62728', edgecolor='none')
ax2.set_xlabel('JPEG trace strength (BAR - 1.0)')
ax2.set_ylabel('Count')
ax2.set_title('Block artifact ratio strength')
plt.tight_layout()
plt.savefig(CTRL / 'test_set_jpeg_distribution.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"  저장: test_set_jpeg_distribution.png")

print(f"\n  Quality 분포:")
for q in sorted(counts.keys()):
    print(f"    Q≈{q}: {counts[q]:6d}  ({counts[q]/len(q_df)*100:5.1f}%)")

# ── Task 3: Quality vs adversarial prob ──────────────────────────────
print(f"\n{'='*70}")
print("Task 3: Quality vs adversarial prob 상관분석")
print("=" * 70)

pred_df = pd.read_parquet(PRED_PQ)
pred_df['filename_stem'] = pred_df['filename'].apply(lambda x: Path(x).stem)

# merge (6002 샘플만 pred가 있음)
q_df['filename_stem'] = q_df['filename'].apply(lambda x: Path(x).stem)
merged = pred_df.merge(q_df[['filename_stem', 'estimated_quality', 'jpeg_trace_strength']],
                       on='filename_stem', how='left')
merged = merged.dropna(subset=['estimated_quality'])
print(f"  merged: {len(merged)} rows")

# Spearman correlation
rho_q, p_q = stats.spearmanr(merged['estimated_quality'], merged['adv_prob_rf'])
rho_t, p_t = stats.spearmanr(merged['jpeg_trace_strength'], merged['adv_prob_rf'])
print(f"  Spearman(quality, adv_prob) = {rho_q:.4f}  p={p_q:.2e}")
print(f"  Spearman(trace_strength, adv_prob) = {rho_t:.4f}  p={p_t:.2e}")

# Quality bin별 adversarial 판정 비율
bins_q = [0, 50, 60, 70, 80, 90, 100]
labels_q = ['<50', '50-60', '60-70', '70-80', '80-90', '90+']
merged['q_bin'] = pd.cut(merged['estimated_quality'], bins=bins_q, labels=labels_q, right=False)
bin_stats = merged.groupby('q_bin', observed=True).agg(
    n=('adv_prob_rf', 'count'),
    adv_rate=('predicted_label_rf', 'mean'),
    mean_prob=('adv_prob_rf', 'mean'),
).reset_index()
print(f"\n  Quality bin별 adversarial 판정:")
print(f"  {'bin':<8} {'n':>6} {'adv_rate':>10} {'mean_prob':>10}")
for _, r in bin_stats.iterrows():
    print(f"  {r['q_bin']:<8} {int(r['n']):>6} {r['adv_rate']:>10.3f} {r['mean_prob']:>10.3f}")

# Scatter/boxplot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.scatter(merged['estimated_quality'] + np.random.uniform(-2, 2, len(merged)),
           merged['adv_prob_rf'], s=3, alpha=0.2, c='#444')
ax.set_xlabel('Estimated JPEG quality'); ax.set_ylabel('Adversarial probability (RF)')
ax.set_title(f'Quality vs Adv prob\nSpearman ρ={rho_q:.3f}')

ax2 = axes[1]
merged.boxplot(column='adv_prob_rf', by='q_bin', ax=ax2)
ax2.set_xlabel('Quality bin'); ax2.set_ylabel('Adversarial probability (RF)')
ax2.set_title(f'Adv prob by quality bin')
plt.suptitle('')
plt.tight_layout()
plt.savefig(CTRL / 'jpeg_quality_vs_adv_prob.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"  저장: jpeg_quality_vs_adv_prob.png")

# 판정
print(f"\n판정:")
if cos_q80_sus > 0.9:
    v1 = "⚠ JPEG Q80 matches adversarial pattern — test set real images with Q80 may be false positives"
else:
    v1 = "✓ JPEG Q80 distinct from adversarial pattern"
print(f"  Task 1: {v1}")

if rho_q < -0.5:
    v3 = "⚠ Strong negative correlation — many 'adversarial' detections may be JPEG compression artifacts"
elif rho_q > -0.2:
    v3 = "✓ Weak correlation — adversarial detection is mostly independent of JPEG quality"
else:
    v3 = "~ Moderate correlation — needs investigation"
print(f"  Task 3: {v3}")

# 리포트
with open(CTRL / 'jpeg_quality_analysis.md', 'w') as f:
    f.write('# JPEG Quality Analysis\n\n')
    f.write('## Task 1: JPEG Q80 Instability Pattern\n\n')
    f.write(f'| target | cosine sim |\n|---|---|\n')
    f.write(f'| Q80 → suspect | {cos_q80_sus:.3f} |\n')
    f.write(f'| Q80 → adv_baseline | {cos_q80_adv:.3f} |\n')
    f.write(f'| Q80 → real_baseline | {cos_q80_real:.3f} |\n\n')
    f.write(f'**판정**: {v1}\n\n')
    f.write('비교 (기존 결과):\n')
    f.write('| augmentation | →suspect |\n|---|---|\n')
    f.write('| jpeg_q50 | 0.921 |\n| jpeg_q70 | 0.964 |\n')
    f.write(f'| **jpeg_q80** | **{cos_q80_sus:.3f}** |\n\n')
    f.write('## Task 2: Test Set JPEG Quality Distribution\n\n')
    f.write(f'전체 {len(q_df)}장 분석 결과:\n\n')
    f.write('| Quality | n | % |\n|---|---|---|\n')
    for q in sorted(counts.keys()):
        f.write(f'| Q≈{q} | {counts[q]} | {counts[q]/len(q_df)*100:.1f}% |\n')
    f.write(f'\n## Task 3: Quality vs Adversarial Probability\n\n')
    f.write(f'- Spearman(quality, adv_prob) = **{rho_q:.4f}** (p={p_q:.2e})\n')
    f.write(f'- Spearman(trace_strength, adv_prob) = **{rho_t:.4f}** (p={p_t:.2e})\n\n')
    f.write(f'**판정**: {v3}\n\n')
    f.write('| Quality bin | n | adv_rate | mean_prob |\n|---|---|---|---|\n')
    for _, r in bin_stats.iterrows():
        f.write(f'| {r["q_bin"]} | {int(r["n"])} | {r["adv_rate"]:.3f} | {r["mean_prob"]:.3f} |\n')
    f.write('\n## 출력 파일\n\n')
    f.write('- [test_set_jpeg_analysis.csv](./test_set_jpeg_analysis.csv)\n')
    f.write('- [test_set_jpeg_distribution.png](./test_set_jpeg_distribution.png)\n')
    f.write('- [jpeg_quality_vs_adv_prob.png](./jpeg_quality_vs_adv_prob.png)\n')

print(f"\n완료 ✓")
