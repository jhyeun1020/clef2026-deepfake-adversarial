"""
Step 4: instability features 생성 + RF 학습 + test 예측 + 분석 + 시각화
"""
import csv, time, json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

BASE = Path('/home/student/GEN_AI_GTG')
APP = BASE / 'detection/instability/test_application'
CROP_SCORES = BASE / 'detection/crop_ablation/scores'  # train data: real_reference + adversarial_fake
TEST_SCORES = APP / 'scores'                            # test data: 6002 × 4 variant
TEST_DIR = BASE / 'detection/ImageCLEF2026-DeepFakeDetection-Tes/Data/Images_Detection'
META_CSV = APP / 'sample_metadata.csv'
PHASH_CSV = BASE / 'detection/phash_matching/fake_gt_matches.csv'

DETECTORS = ['npr', 'distildire', 'aide_genimage', 'xception',
             'siglip_dino', 'aide_progan', 'haywood', 'aide_sd14']
CROP_VARIANTS = ['face_crop_variable', 'random_crop_192_a', 'center_crop_192']

# ── 1. 학습용 features (real + adversarial) ───────────────────────────
print("[1/7] 학습 features 로드 (real_reference 1000 + adversarial_fake 1002)...")

def build_train_features(ds):
    """ds: real_reference 또는 adversarial_fake. crop_ablation/scores 사용."""
    n = None
    feats = []
    for det in DETECTORS:
        df_full = pd.read_parquet(CROP_SCORES / f'{det}__{ds}__full.parquet').sort_values('filename').reset_index(drop=True)
        if n is None: n = len(df_full)
        for v in CROP_VARIANTS:
            df_v = pd.read_parquet(CROP_SCORES / f'{det}__{ds}__{v}.parquet').sort_values('filename').reset_index(drop=True)
            assert len(df_v) == n
            feats.append(np.abs(df_full['score'].to_numpy() - df_v['score'].to_numpy()))
    return np.stack(feats, axis=1)  # (n, 24)

X_real = build_train_features('real_reference')
X_adv  = build_train_features('adversarial_fake')
print(f"  real: {X_real.shape}  adv: {X_adv.shape}")

X = np.vstack([X_real, X_adv])
y = np.array([0]*len(X_real) + [1]*len(X_adv))

# 동일 파라미터로 전체 학습 (test 적용)
clf_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X, y)
clf_lr = LogisticRegression(max_iter=2000, random_state=42).fit(X, y)
print(f"  RF/LR 학습 완료\n")

# Quick CV check
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
clf_check = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
auc_check = roc_auc_score(y_te, clf_check.predict_proba(X_te)[:,1])
print(f"  RF 50/50 split AUROC: {auc_check:.4f}")

# ── 2. Test features (6002장) ─────────────────────────────────────────
print("\n[2/7] Test features 생성 (6,002장)...")
# 모든 (det, variant)를 같은 stem 순서로 정렬해서 row alignment
def load_test_score(det, v):
    df = pd.read_parquet(TEST_SCORES / f'{det}__{v}.parquet').sort_values('filename').reset_index(drop=True)
    return df['filename'].tolist(), df['score'].to_numpy(dtype=np.float64)

# stem 기준으로 모든 detector × variant align
ref_names, _ = load_test_score(DETECTORS[0], 'full')
n_test = len(ref_names)
print(f"  test n: {n_test}")

feats_test = []
for det in DETECTORS:
    names_full, full = load_test_score(det, 'full')
    assert names_full == ref_names
    for v in CROP_VARIANTS:
        names_v, sv = load_test_score(det, v)
        assert names_v == ref_names
        feats_test.append(np.abs(full - sv))
X_test = np.stack(feats_test, axis=1)
print(f"  X_test: {X_test.shape}")

# ── 3. 예측 + 저장 ────────────────────────────────────────────────────
print("\n[3/7] 예측...")
proba_rf = clf_rf.predict_proba(X_test)[:,1]
proba_lr = clf_lr.predict_proba(X_test)[:,1]

stems = [Path(n).stem for n in ref_names]
filenames = [Path(n).name if n.endswith('.png') else stems[i]+'.png' for i,n in enumerate(ref_names)]

feat_cols = [f'{det}__{v}' for det in DETECTORS for v in CROP_VARIANTS]
df_feat = pd.DataFrame(X_test, columns=feat_cols)
df_feat.insert(0, 'filename', filenames)
df_feat.to_parquet(APP / 'test_instability_features.parquet', index=False)

df_pred = pd.DataFrame({
    'filename': filenames,
    'adv_prob_rf': proba_rf,
    'adv_prob_lr': proba_lr,
    'predicted_label_rf': (proba_rf >= 0.5).astype(int),
    'predicted_label_lr': (proba_lr >= 0.5).astype(int),
})
df_pred.to_parquet(APP / 'test_instability_predictions.parquet', index=False)
print(f"  features.parquet, predictions.parquet 저장")

# ── 4. metadata 머지 (category, face_detected) ────────────────────────
print("\n[4/7] metadata 머지...")
meta = pd.read_csv(META_CSV)
df = df_pred.merge(meta, on='filename', how='left')

# 우리 1002장 식별
ours = set()
for r in csv.DictReader(open(PHASH_CSV)):
    ours.add(Path(r['test_image_path']).name)
df['is_ours'] = df['filename'].isin(ours)

# ── 5. 분포 분석 ───────────────────────────────────────────────────────
print("\n[5/7] 분포 분석...")

def bimodality_coef(x):
    n = len(x)
    if n < 4: return float('nan')
    s = stats.skew(x); k = stats.kurtosis(x, fisher=True)
    return float((s**2 + 1) / (k + 3*((n-1)**2)/((n-2)*(n-3))))

bc_rf = bimodality_coef(proba_rf)
bc_lr = bimodality_coef(proba_lr)
print(f"  Bimodality coef (RF): {bc_rf:.4f} ({'bimodal' if bc_rf > 5/9 else 'unimodal'})")
print(f"  Bimodality coef (LR): {bc_lr:.4f} ({'bimodal' if bc_lr > 5/9 else 'unimodal'})")

threshold_table = []
for t in [0.3, 0.5, 0.7, 0.9]:
    n_rf = int((proba_rf >= t).sum())
    n_lr = int((proba_lr >= t).sum())
    threshold_table.append({'threshold': t, 'pred_adv_rf': n_rf, 'pred_adv_lr': n_lr,
                            'pct_rf': n_rf/n_test*100, 'pct_lr': n_lr/n_test*100})

print(f"\n  threshold | pred_adv_rf  pct  | pred_adv_lr  pct")
for r in threshold_table:
    print(f"    {r['threshold']:.1f}     | {r['pred_adv_rf']:6d}  {r['pct_rf']:5.1f}%  | {r['pred_adv_lr']:6d}  {r['pct_lr']:5.1f}%")

# 분포 히스토그램
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, proba, title in [(axes[0], proba_rf, 'Random Forest'), (axes[1], proba_lr, 'Logistic Regression')]:
    ax.hist(proba, bins=50, color='#444', edgecolor='none')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Adversarial probability'); ax.set_ylabel('count')
    bc = bimodality_coef(proba)
    n_adv = int((proba>=0.5).sum())
    ax.set_title(f'{title}\nn={n_test}, BC={bc:.3f}, pred_adv={n_adv} ({n_adv/n_test*100:.1f}%)')
    ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig(APP / 'test_prediction_distribution.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"  저장: test_prediction_distribution.png")

# ── 6. 우리 1002 cross-check ─────────────────────────────────────────
print("\n[6/7] 우리 1002장 cross-check...")
df_ours = df[df['is_ours']].sort_values('adv_prob_rf').reset_index(drop=True)
n_ours = len(df_ours)
n_ours_correct = int((df_ours['adv_prob_rf'] >= 0.5).sum())
print(f"  ours 총 {n_ours}장 | adv 판정: {n_ours_correct} ({n_ours_correct/n_ours*100:.1f}%)")
print(f"  prob 평균: {df_ours['adv_prob_rf'].mean():.3f}, 중앙값: {df_ours['adv_prob_rf'].median():.3f}")

# 우리 1002 prob 히스토그램
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df_ours['adv_prob_rf'], bins=50, color='#d62728', edgecolor='none', alpha=0.85)
ax.axvline(0.5, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Adversarial probability (RF)')
ax.set_ylabel('count')
ax.set_title(f'우리 1002장 prob 분포 (RF)\n판정 성공 {n_ours_correct}/{n_ours} = {n_ours_correct/n_ours*100:.1f}%')
ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig(APP / 'our_1002_prob_distribution.png', dpi=130, bbox_inches='tight')
plt.close()

# misses CSV
df_misses = df_ours[df_ours['adv_prob_rf'] < 0.5][['filename', 'adv_prob_rf']].copy()
df_misses.insert(0, 'rank', range(1, len(df_misses)+1))
df_misses.columns = ['rank', 'filename', 'adversarial_prob']
df_misses.to_csv(APP / 'our_1002_misses.csv', index=False, float_format='%.4f')
print(f"  misses CSV: {len(df_misses)} rows")

# Face detected vs failed 그룹 비교
face_ok = df[df['face_detected'] == True]
face_no = df[df['face_detected'] == False]
print(f"\n  face_detected=True : n={len(face_ok)}, mean(prob)={face_ok['adv_prob_rf'].mean():.3f}, "
      f"adv판정={(face_ok['adv_prob_rf']>=0.5).sum()} ({(face_ok['adv_prob_rf']>=0.5).mean()*100:.1f}%)")
print(f"  face_detected=False: n={len(face_no)}, mean(prob)={face_no['adv_prob_rf'].mean():.3f}, "
      f"adv판정={(face_no['adv_prob_rf']>=0.5).sum()} ({(face_no['adv_prob_rf']>=0.5).mean()*100:.1f}%)")

# ── 7. Sample grids ──────────────────────────────────────────────────
print("\n[7/7] sample grids 생성...")

def make_grid(rows, cols, items, out_path, title='', cell=180, font=12):
    """items: list of dicts with keys filename, label."""
    from PIL import ImageDraw, ImageFont
    pad = 4; label_h = 22
    n = rows * cols
    items = items[:n]
    W = cols * cell + pad * (cols + 1)
    H = rows * (cell + label_h + pad) + pad + 24
    grid = Image.new('RGB', (W, H), (25, 25, 25))
    draw = ImageDraw.Draw(grid)
    try:
        f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', font)
    except Exception:
        f = ImageFont.load_default()
    if title:
        draw.text((pad, 5), title, fill=(255, 220, 100), font=f)
    for i, it in enumerate(items):
        r, c = i // cols, i % cols
        x = pad + c * (cell + pad)
        y = 24 + pad + r * (cell + label_h + pad)
        try:
            im = Image.open(TEST_DIR / it['filename']).convert('RGB').resize((cell, cell))
        except Exception:
            im = Image.new('RGB', (cell, cell), (60, 0, 0))
        grid.paste(im, (x, y))
        short = it['filename'][:8]
        draw.text((x, y + cell + 2), f"{short} p={it['adv_prob_rf']:.2f}",
                  fill=(220, 220, 220), font=f)
    grid.save(out_path)

# (a) High-confidence adversarial — 우리 1002 제외, RF prob 상위 50
not_ours = df[~df['is_ours']].copy()
high_adv = not_ours.sort_values('adv_prob_rf', ascending=False).head(50)
make_grid(5, 10, high_adv.to_dict('records'),
          APP / 'samples_high_confidence_adversarial.png',
          title='High-confidence adversarial (top 50, ours excluded)')

# (b) Mid-confidence — 0.5~0.6
mid = not_ours[(not_ours['adv_prob_rf'] >= 0.5) & (not_ours['adv_prob_rf'] <= 0.6)]
mid_sample = mid.sample(n=min(50, len(mid)), random_state=42)
make_grid(5, 10, mid_sample.to_dict('records'),
          APP / 'samples_mid_confidence_adversarial.png',
          title=f'Mid-confidence adversarial (0.5–0.6, n_sampled={len(mid_sample)} of {len(mid)})')

# (c) Low (high-confidence non-adversarial)
low = not_ours.sort_values('adv_prob_rf').head(50)
make_grid(5, 10, low.to_dict('records'),
          APP / 'samples_high_confidence_clean.png',
          title='High-confidence non-adversarial (bottom 50, ours excluded)')

# (d) Face failed cases with high prob
face_failed_df = df[(df['face_detected'] == False) & (df['adv_prob_rf'] >= 0.5)]
ff_top = face_failed_df.sort_values('adv_prob_rf', ascending=False).head(30)
if len(ff_top) > 0:
    rows_d = (len(ff_top) + 5) // 6
    make_grid(rows_d, 6, ff_top.to_dict('records'),
              APP / 'samples_face_failed_adversarial.png',
              title=f'Face-detection-failed adversarial (top {len(ff_top)})')
else:
    print(f"  ⚠ face-failed에서 adv 판정된 이미지 없음")

# our 1002 misses grid
if len(df_misses) > 0:
    miss_items = df_misses.head(20).to_dict('records')
    miss_items = [{'filename': r['filename'], 'adv_prob_rf': r['adversarial_prob']} for r in miss_items]
    rows_m = max(1, (len(miss_items) + 4) // 5)
    make_grid(rows_m, 5, miss_items,
              APP / 'our_1002_misses.png',
              title=f'Our 1002 misses (lowest prob, n={min(20, len(miss_items))})')

# sample_filenames CSV
sample_rows = []
for r in high_adv.to_dict('records')[:50]:
    sample_rows.append({'category':'high_conf_adv', 'filename':r['filename'], 'adversarial_prob':round(r['adv_prob_rf'],4)})
for r in mid_sample.to_dict('records'):
    sample_rows.append({'category':'mid_conf_adv', 'filename':r['filename'], 'adversarial_prob':round(r['adv_prob_rf'],4)})
for r in low.to_dict('records')[:50]:
    sample_rows.append({'category':'high_conf_clean', 'filename':r['filename'], 'adversarial_prob':round(r['adv_prob_rf'],4)})
for r in ff_top.to_dict('records'):
    sample_rows.append({'category':'face_failed_adv', 'filename':r['filename'], 'adversarial_prob':round(r['adv_prob_rf'],4)})
pd.DataFrame(sample_rows).insert(1, 'rank', None)
df_sf = pd.DataFrame(sample_rows)
df_sf['rank'] = df_sf.groupby('category').cumcount() + 1
df_sf = df_sf[['category','rank','filename','adversarial_prob']]
df_sf.to_csv(APP / 'sample_filenames.csv', index=False)
print(f"  sample_filenames.csv 저장 ({len(df_sf)} rows)")

# ── 8. test_summary.md ───────────────────────────────────────────────
print("\n[8/8] test_summary.md 생성...")
n_total_test = 24404
n_face_failed_total = 1286
n_face_failed_pct = n_face_failed_total / n_total_test * 100

ours_n_correct = int((df_ours['adv_prob_rf'] >= 0.5).sum())
others_df = df[~df['is_ours']]
others_n = len(others_df)
others_adv = int((others_df['adv_prob_rf'] >= 0.5).sum())

with open(APP / 'test_summary.md', 'w') as f:
    f.write('# Test Set Instability Detection — Summary\n\n')
    f.write('> RandomForest 학습: real_reference 1000 + adversarial_fake 1002 (24-dim instability)\n')
    f.write(f'> 50/50 cross-validation AUROC: **{auc_check:.4f}**\n')
    f.write(f'> Test 적용: 6,002장 (우리 1002 + random 4,500 + face-failed 500)\n\n')
    f.write('---\n\n## 1. Face Detection 통계 (test set 24,404 전체)\n\n')
    f.write(f'- Total            : {n_total_test:,}\n')
    f.write(f'- Face detected    : {n_total_test - n_face_failed_total:,} ({(n_total_test-n_face_failed_total)/n_total_test*100:.1f}%)\n')
    f.write(f'- Face NOT detected: {n_face_failed_total:,} ({n_face_failed_pct:.1f}%)\n\n')
    f.write('## 2. Threshold별 adversarial 판정 수 (n=6,002 샘플 기준)\n\n')
    f.write('| threshold | RF | RF % | Logistic | LR % |\n|---|---|---|---|---|\n')
    for r in threshold_table:
        f.write(f'| {r["threshold"]} | {r["pred_adv_rf"]} | {r["pct_rf"]:.1f}% | {r["pred_adv_lr"]} | {r["pct_lr"]:.1f}% |\n')
    f.write(f'\nBimodality coefficient (RF): **{bc_rf:.4f}** ({"bimodal" if bc_rf>5/9 else "unimodal"})\n')
    f.write(f'Bimodality coefficient (LR): **{bc_lr:.4f}** ({"bimodal" if bc_lr>5/9 else "unimodal"})\n\n')
    f.write('## 3. 우리 팀 1002장 cross-check\n\n')
    f.write(f'- 판정 성공: **{ours_n_correct}/{n_ours}** = {ours_n_correct/n_ours*100:.1f}%\n')
    f.write(f'- prob 평균: {df_ours["adv_prob_rf"].mean():.3f}\n')
    f.write(f'- prob 중앙값: {df_ours["adv_prob_rf"].median():.3f}\n')
    f.write(f'- 실패 케이스: {len(df_misses)}장 → [our_1002_misses.csv](./our_1002_misses.csv)\n\n')
    f.write('## 4. 우리 1002장 제외 → 다른 이미지에서 adversarial 판정\n\n')
    f.write(f'- 우리 1002 제외 샘플: {others_n}장\n')
    f.write(f'- adv 판정 (threshold 0.5): **{others_adv}장 ({others_adv/others_n*100:.1f}%)**\n\n')
    f.write('전체 24,404장 기준 추정:\n')
    f.write(f'- 우리 1002 제외 23,402장 중 sampling 4,500장에서 random_4500 풀의 adv 비율을 측정\n')
    rnd_df = df[df['category'].fillna('').str.contains('random_4500', na=False)]
    rnd_adv = int((rnd_df['adv_prob_rf']>=0.5).sum())
    rnd_n = len(rnd_df)
    if rnd_n > 0:
        rnd_rate = rnd_adv / rnd_n
        f.write(f'- random_4500 그룹: {rnd_adv}/{rnd_n} = {rnd_rate*100:.2f}%\n')
        # 23402 * rnd_rate 추정
        est_adv_others = int(round(23402 * rnd_rate))
        f.write(f'- → 23,402장 추정 adv 수: 약 **{est_adv_others}장**\n\n')
    f.write('## 5. Face detected vs failed 그룹 비교\n\n')
    f.write('| group | n | mean(prob) | adv판정 | rate |\n|---|---|---|---|---|\n')
    f.write(f'| face_detected | {len(face_ok)} | {face_ok["adv_prob_rf"].mean():.3f} | {(face_ok["adv_prob_rf"]>=0.5).sum()} | {(face_ok["adv_prob_rf"]>=0.5).mean()*100:.1f}% |\n')
    f.write(f'| face_failed   | {len(face_no)} | {face_no["adv_prob_rf"].mean():.3f} | {(face_no["adv_prob_rf"]>=0.5).sum()} | {(face_no["adv_prob_rf"]>=0.5).mean()*100:.1f}% |\n\n')
    f.write('## 6. 핵심 결론\n\n')
    if rnd_n > 0:
        if rnd_rate < 0.05:
            verdict_q1 = "**다른 팀들은 instability 기반으로 잡히지 않음** — adversarial 공격을 안 썼거나 다른 방식 사용"
        elif rnd_rate < 0.20:
            verdict_q1 = "**일부 다른 팀이 adversarial 공격 사용**한 것으로 추정"
        else:
            verdict_q1 = "**상당수 다른 팀이 adversarial 공격 사용** — instability 기반 분리 가능"
        f.write(f'- **Q1**: {verdict_q1}\n')
    if bc_rf > 5/9:
        f.write(f'- **Q2 (bimodality)**: BC={bc_rf:.3f} — bimodal, 두 그룹 명확히 분리\n')
    else:
        f.write(f'- **Q2 (bimodality)**: BC={bc_rf:.3f} — unimodal, adversarial이 소수\n')
    f.write(f'- **Q3 (우리 1002 회수율)**: {ours_n_correct/n_ours*100:.1f}% — ')
    f.write('generalization 양호\n' if ours_n_correct/n_ours > 0.95 else 'generalization 부분적, 일부 case 놓침\n')
    f.write('- **Q4**: 시각적 분석은 [samples_high_confidence_adversarial.png] 참고\n\n')
    f.write('## 출력 파일\n\n')
    f.write('- [test_instability_features.parquet](./test_instability_features.parquet)\n')
    f.write('- [test_instability_predictions.parquet](./test_instability_predictions.parquet)\n')
    f.write('- [test_prediction_distribution.png](./test_prediction_distribution.png)\n')
    f.write('- [samples_high_confidence_adversarial.png](./samples_high_confidence_adversarial.png)\n')
    f.write('- [samples_mid_confidence_adversarial.png](./samples_mid_confidence_adversarial.png)\n')
    f.write('- [samples_high_confidence_clean.png](./samples_high_confidence_clean.png)\n')
    f.write('- [samples_face_failed_adversarial.png](./samples_face_failed_adversarial.png)\n')
    f.write('- [our_1002_misses.png](./our_1002_misses.png)\n')
    f.write('- [our_1002_prob_distribution.png](./our_1002_prob_distribution.png)\n')
    f.write('- [our_1002_misses.csv](./our_1002_misses.csv)\n')
    f.write('- [sample_filenames.csv](./sample_filenames.csv)\n')

print("\n완료 ✓")
