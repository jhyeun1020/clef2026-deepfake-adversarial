"""
Mahalanobis Distribution Shift 원인 진단 + Fix attempts.
Tasks 1-7 all-in-one.
"""
import pickle, json, csv, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE

BASE = Path('/home/student/GEN_AI_GTG')
OUT = BASE / 'detection/mahalanobis'

t0 = time.time()

# ── Load data ─────────────────────────────────────────────────────────
print("데이터 로드...")
pilot_df = pd.read_parquet(OUT / 'pilot/pilot_features.parquet')
test_df  = pd.read_parquet(OUT / 'test_features.parquet')

# pHash로 우리 1002장 식별
ours_set = set()
for r in csv.DictReader(open(BASE / 'detection/phash_matching/fake_gt_matches.csv')):
    ours_set.add(Path(r['test_image_path']).name)
test_df['is_ours'] = test_df['filename'].isin(ours_set)

# Column mapping
mah_cols_pilot = [f'mah_{i}' for i in range(20)]
mah_cols_test = [c for c in test_df.columns if c.startswith('mah_L')]  # 20 cols
class_names = ['real', 'adv', 'clean_fake', 'jpeg']

# Gaussian params
gp_pilot = pickle.load(open(OUT / 'pilot/gaussian_params_pilot.pkl', 'rb'))
gp_full  = pickle.load(open(OUT / 'gaussian_params_full.pkl', 'rb'))
layer_indices = gp_pilot['layer_indices']

print(f"  pilot: {len(pilot_df)}, test: {len(test_df)}, ours: {test_df['is_ours'].sum()}")

# ══════════════════════════════════════════════════════════════════════
# TASK 1: Distance 분포 직접 비교
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[1/7] Distance 분포 비교")
print("="*70)

# Pilot: min distance per image
pilot_X = pilot_df[mah_cols_pilot].values
pilot_min_dist = pilot_X.min(axis=1)  # closest class distance
pilot_labels = pilot_df['label'].values

# Test: min distance (already computed)
test_min_dist = test_df['min_distance'].values
ours_min_dist = test_df[test_df['is_ours']]['min_distance'].values
notours_min_dist = test_df[~test_df['is_ours']]['min_distance'].values

# Per-pilot-class
for label in ['real', 'adv', 'clean_fake', 'jpeg']:
    mask = pilot_labels == label
    d = pilot_min_dist[mask]
    print(f"  Pilot {label:12s}: mean={d.mean():.1f}  med={np.median(d):.1f}  "
          f"std={d.std():.1f}  range=[{d.min():.1f}, {d.max():.1f}]")

print(f"\n  Test all (24404): mean={test_min_dist.mean():.1f}  med={np.median(test_min_dist):.1f}  "
      f"std={test_min_dist.std():.1f}  range=[{test_min_dist.min():.1f}, {test_min_dist.max():.1f}]")
print(f"  Test ours (1002): mean={ours_min_dist.mean():.1f}  med={np.median(ours_min_dist):.1f}  "
      f"std={ours_min_dist.std():.1f}")
print(f"  Test others:      mean={notours_min_dist.mean():.1f}  med={np.median(notours_min_dist):.1f}")

# Histogram
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(0, max(test_min_dist.max(), pilot_min_dist.max()) * 1.05, 80)
ax.hist(pilot_min_dist[pilot_labels=='real'], bins=bins, alpha=0.5, density=True, label='Pilot real', color='green')
ax.hist(pilot_min_dist[pilot_labels=='adv'], bins=bins, alpha=0.5, density=True, label='Pilot adv', color='red')
ax.hist(test_min_dist, bins=bins, alpha=0.3, density=True, label='Test all (24k)', color='gray')
ax.hist(ours_min_dist, bins=bins, alpha=0.5, density=True, label='Test ours (1002)', color='orange')
ax.set_xlabel('Min Mahalanobis distance to any class'); ax.set_ylabel('Density')
ax.set_title('Distance distribution: Pilot vs Test')
ax.legend()
plt.tight_layout()
plt.savefig(OUT / 'distance_distribution_comparison.png', dpi=130, bbox_inches='tight')
plt.close()

# Stats CSV
stats_rows = []
for name, dists in [('pilot_real', pilot_min_dist[pilot_labels=='real']),
                     ('pilot_adv', pilot_min_dist[pilot_labels=='adv']),
                     ('pilot_clean', pilot_min_dist[pilot_labels=='clean_fake']),
                     ('pilot_jpeg', pilot_min_dist[pilot_labels=='jpeg']),
                     ('test_all', test_min_dist),
                     ('test_ours', ours_min_dist),
                     ('test_others', notours_min_dist)]:
    stats_rows.append({
        'group': name, 'n': len(dists),
        'mean': round(float(dists.mean()), 2), 'median': round(float(np.median(dists)), 2),
        'std': round(float(dists.std()), 2),
        'p10': round(float(np.percentile(dists, 10)), 2),
        'p25': round(float(np.percentile(dists, 25)), 2),
        'p75': round(float(np.percentile(dists, 75)), 2),
        'p90': round(float(np.percentile(dists, 90)), 2),
        'min': round(float(dists.min()), 2), 'max': round(float(dists.max()), 2),
    })
pd.DataFrame(stats_rows).to_csv(OUT / 'distance_stats.csv', index=False)
print("  distance_distribution_comparison.png, distance_stats.csv 저장")

# ══════════════════════════════════════════════════════════════════════
# TASK 2: Class별 distance
# ══════════════════════════════════════════════════════════════════════
print("\n[2/7] Class별 distance 분석")
# avg_dist columns already in test_df
avg_cols = [f'avg_dist_{cn}' for cn in class_names]
ours_avg = test_df[test_df['is_ours']][avg_cols].mean()
others_avg = test_df[~test_df['is_ours']][avg_cols].mean()
print(f"  Our 1002 avg distances: {dict(ours_avg.round(1))}")
print(f"  Others avg distances:   {dict(others_avg.round(1))}")
print(f"  Our 1002 closest: {test_df[test_df['is_ours']]['closest_class'].value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════════
# TASK 3: Feature space t-SNE
# ══════════════════════════════════════════════════════════════════════
print("\n[3/7] Feature space t-SNE...")
rng = np.random.RandomState(42)

# Pilot 400 + test random 500 + ours 100
test_rand_idx = rng.choice(len(test_df[~test_df['is_ours']]), 500, replace=False)
ours_idx = rng.choice(len(test_df[test_df['is_ours']]), min(100, test_df['is_ours'].sum()), replace=False)

X_p = pilot_df[mah_cols_pilot].values  # (400, 20)
X_tr = test_df[~test_df['is_ours']].iloc[test_rand_idx][mah_cols_test].values  # (500, 20)
X_to = test_df[test_df['is_ours']].iloc[ours_idx][mah_cols_test].values  # (100, 20)

X_all = np.vstack([X_p, X_tr, X_to])
labels_all = (list(pilot_df['label'].values) +
              ['test_other'] * len(X_tr) +
              ['test_ours'] * len(X_to))

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
emb = tsne.fit_transform(X_all)

fig, ax = plt.subplots(figsize=(10, 8))
colors = {'real': '#2ca02c', 'adv': '#d62728', 'clean_fake': '#1f77b4', 'jpeg': '#ff7f0e',
          'test_other': '#888888', 'test_ours': '#9467bd'}
markers = {'real': 'o', 'adv': '*', 'clean_fake': 's', 'jpeg': 'D',
           'test_other': '.', 'test_ours': 'X'}
for lbl in ['test_other', 'real', 'jpeg', 'clean_fake', 'adv', 'test_ours']:
    mask = [l == lbl for l in labels_all]; idx = np.where(mask)[0]
    s = 15 if lbl == 'test_other' else (100 if lbl in ['adv', 'test_ours'] else 40)
    ax.scatter(emb[idx, 0], emb[idx, 1], c=colors[lbl], marker=markers[lbl],
              s=s, alpha=0.6, label=f'{lbl} ({len(idx)})', edgecolors='none')
ax.legend(fontsize=9, frameon=True, loc='best')
ax.set_title('t-SNE: Pilot (400) + Test random (500) + Our 1002 (100)')
plt.tight_layout()
plt.savefig(OUT / 'feature_space_visualization.png', dpi=130, bbox_inches='tight')
plt.close()
print("  feature_space_visualization.png 저장")

# ══════════════════════════════════════════════════════════════════════
# TASK 4: RF probability 분포
# ══════════════════════════════════════════════════════════════════════
print("\n[4/7] RF probability 분포")
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_p, (pilot_df['label'] == 'adv').astype(int).values)

prob_pilot = clf.predict_proba(X_p)[:, 1]
prob_test_rand = clf.predict_proba(X_tr)[:, 1]
prob_test_ours = clf.predict_proba(X_to)[:, 1]

# Full test
prob_test_all = clf.predict_proba(test_df[mah_cols_test].values)[:, 1]

print(f"  Pilot prob: mean={prob_pilot.mean():.3f} range=[{prob_pilot.min():.3f}, {prob_pilot.max():.3f}]")
print(f"  Test rand:  mean={prob_test_rand.mean():.3f} range=[{prob_test_rand.min():.3f}, {prob_test_rand.max():.3f}]")
print(f"  Test ours:  mean={prob_test_ours.mean():.3f} range=[{prob_test_ours.min():.3f}, {prob_test_ours.max():.3f}]")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
bins = np.linspace(0, 1, 41)
for ax, data, title in [
    (axes[0], prob_pilot, f'Pilot (400)\nmean={prob_pilot.mean():.3f}'),
    (axes[1], prob_test_all, f'Test all (24k)\nmean={prob_test_all.mean():.3f}'),
    (axes[2], prob_test_ours, f'Test ours (100 sample)\nmean={prob_test_ours.mean():.3f}'),
]:
    ax.hist(data, bins=bins, color='#444', edgecolor='none')
    ax.axvline(0.5, color='red', linestyle='--')
    ax.set_title(title); ax.set_xlabel('P(adversarial)')
plt.suptitle('RF predict_proba distributions')
plt.tight_layout()
plt.savefig(OUT / 'rf_probability_distributions.png', dpi=130, bbox_inches='tight')
plt.close()
print("  rf_probability_distributions.png 저장")

# ══════════════════════════════════════════════════════════════════════
# TASK 5: Covariance scale 문제
# ══════════════════════════════════════════════════════════════════════
print("\n[5/7] Covariance scale 분석")
cov_rows = []
for name, gp in [('pilot', gp_pilot), ('full', gp_full)]:
    for li in layer_indices:
        sinv = gp['sigma_inv'][li]
        # sinv is inverse covariance → cov = inv(sinv)
        try:
            eigs = np.linalg.eigvalsh(sinv)
            det_sinv = float(np.prod(np.sign(eigs)) * np.exp(np.sum(np.log(np.abs(eigs) + 1e-30))))
            cond = float(eigs.max() / (eigs.min() + 1e-30))
            trace = float(sinv.trace())
        except Exception:
            det_sinv = 0; cond = 0; trace = 0
        cov_rows.append({
            'source': name, 'layer': li,
            'sinv_trace': round(trace, 2),
            'sinv_cond': round(cond, 2),
            'sinv_eig_min': round(float(eigs.min()), 6),
            'sinv_eig_max': round(float(eigs.max()), 2),
            'sinv_eig_median': round(float(np.median(eigs)), 4),
        })
        print(f"  {name} layer {li}: trace={trace:.1f} cond={cond:.1f} "
              f"eig_range=[{eigs.min():.4f}, {eigs.max():.1f}]")
cov_df = pd.DataFrame(cov_rows)
cov_df.to_csv(OUT / 'covariance_analysis.csv', index=False)

# ══════════════════════════════════════════════════════════════════════
# TASK 6: Fix attempts
# ══════════════════════════════════════════════════════════════════════
print("\n[6/7] Fix attempts")

# Fix A: Phase 2 full Gaussian으로 pilot feature 재계산
print("\n  Fix A: Phase 2 Gaussian으로 pilot 400장 distance 재계산...")
# Pilot raw features (CLS tokens) — 없음, parquet에는 이미 계산된 distance만 있음
# → pilot raw features를 다시 추출해야 하지만 모델 로드 없이는 불가능
# 대안: test set에서 pilot과 같은 label을 가진 이미지들의 distance로 RF 재학습
# → 불가능 (test에 label 없음)
# Fix A 대안: Phase 2 distance로 test에서 labeled subset 구성
# 우리 1002장 = adv label, real reference의 test 내 위치 모름
# → Fix A는 raw features 없이 불가 → skip
print("    ⚠ Raw features 미저장으로 재계산 불가 — skip")

# Fix B: Normalized distance (z-score per column)
print("\n  Fix B: Z-score normalized distances...")
X_pilot_norm = (X_p - X_p.mean(axis=0)) / (X_p.std(axis=0) + 1e-8)
X_test_norm = (test_df[mah_cols_test].values - X_p.mean(axis=0)) / (X_p.std(axis=0) + 1e-8)

clf_b = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
y_pilot = (pilot_df['label'] == 'adv').astype(int).values
clf_b.fit(X_pilot_norm, y_pilot)
prob_b = clf_b.predict_proba(X_test_norm)[:, 1]
ours_prob_b = prob_b[test_df['is_ours'].values]
recall_b = (ours_prob_b >= 0.5).mean()
det_b = (prob_b >= 0.5).mean()
print(f"    Test det rate: {det_b*100:.1f}%, Our recall: {recall_b*100:.1f}%")

# Fix C: Threshold 재조정 — pilot adv distance의 분위수 기반
print("\n  Fix C: Distance-based threshold (no RF)...")
# "adv class까지의 거리"가 작을수록 adversarial
adv_dist_col = 'avg_dist_adv'
pilot_adv_dists = pilot_df[pilot_df['label']=='adv'][mah_cols_pilot].values
# pilot에서 adv 이미지의 "adv class 까지 거리" → column index for adv
# pilot columns mah_0~19: layer × class 순서 = L4_real, L4_adv, L4_clean, L4_jpeg, L9_real, ...
# adv columns = indices 1, 5, 9, 13, 17
adv_col_indices = [1, 5, 9, 13, 17]
pilot_adv_to_adv = pilot_df[pilot_df['label']=='adv'][[f'mah_{i}' for i in adv_col_indices]].mean(axis=1).values
pilot_real_to_adv = pilot_df[pilot_df['label']=='real'][[f'mah_{i}' for i in adv_col_indices]].mean(axis=1).values
print(f"    Pilot adv→adv_class: mean={pilot_adv_to_adv.mean():.1f} range=[{pilot_adv_to_adv.min():.1f}, {pilot_adv_to_adv.max():.1f}]")
print(f"    Pilot real→adv_class: mean={pilot_real_to_adv.mean():.1f} range=[{pilot_real_to_adv.min():.1f}, {pilot_real_to_adv.max():.1f}]")

# Test set: avg_dist_adv
test_adv_dist = test_df['avg_dist_adv'].values
ours_adv_dist = test_df[test_df['is_ours']]['avg_dist_adv'].values
print(f"    Test all→adv_class: mean={test_adv_dist.mean():.1f} range=[{test_adv_dist.min():.1f}, {test_adv_dist.max():.1f}]")
print(f"    Test ours→adv_class: mean={ours_adv_dist.mean():.1f} range=[{ours_adv_dist.min():.1f}, {ours_adv_dist.max():.1f}]")

# Threshold = pilot adv→adv 의 95th percentile
thresh_c = np.percentile(pilot_adv_to_adv, 95)
print(f"    Threshold (pilot adv p95): {thresh_c:.1f}")
recall_c = (ours_adv_dist <= thresh_c).mean()
det_c = (test_adv_dist <= thresh_c).mean()
print(f"    Test det rate: {det_c*100:.1f}%, Our recall: {recall_c*100:.1f}%")

# Fix D: Relative distance (adv_dist - real_dist) as feature
print("\n  Fix D: Relative distance (adv_dist - real_dist)...")
pilot_rel = pilot_df[mah_cols_pilot].values  # use all 20 features but compute relative
# For pilot: compute diff features (adv_col - real_col per layer)
real_col_indices = [0, 4, 8, 12, 16]
pilot_diff = np.column_stack([
    pilot_df[[f'mah_{ai}' for ai in adv_col_indices]].values -
    pilot_df[[f'mah_{ri}' for ri in real_col_indices]].values
])  # (400, 5)

test_adv_cols = [c for c in mah_cols_test if '_adv' in c]
test_real_cols = [c for c in mah_cols_test if '_real' in c]
test_diff = test_df[test_adv_cols].values - test_df[test_real_cols].values  # (24404, 5)

clf_d = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf_d.fit(pilot_diff, y_pilot)
prob_d = clf_d.predict_proba(test_diff)[:, 1]
ours_prob_d = prob_d[test_df['is_ours'].values]
recall_d = (ours_prob_d >= 0.5).mean()
det_d = (prob_d >= 0.5).mean()
print(f"    Test det rate: {det_d*100:.1f}%, Our recall: {recall_d*100:.1f}%")
# CV on pilot
proba_d_cv = cross_val_predict(clf_d, pilot_diff, y_pilot,
                                cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                method='predict_proba')[:, 1]
auc_d = roc_auc_score(y_pilot, proba_d_cv)
print(f"    Pilot CV AUROC: {auc_d:.4f}")

# Fix E: Use raw avg_dist as simple score (lower avg_dist_adv = more adversarial)
print("\n  Fix E: Simple avg_dist_adv as score (inverted)...")
# lower distance to adv class → more likely adversarial
# Normalize: score = 1 - (dist - min) / (max - min)
d_min, d_max = test_adv_dist.min(), test_adv_dist.max()
score_e = 1.0 - (test_adv_dist - d_min) / (d_max - d_min + 1e-8)
ours_score_e = score_e[test_df['is_ours'].values]
for t in [0.3, 0.5, 0.7, 0.9]:
    recall_e = (ours_score_e >= t).mean()
    det_e = (score_e >= t).mean()
    print(f"    threshold={t}: det={det_e*100:.1f}%, our_recall={recall_e*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════
# TASK 7: Summary
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("[7/7] 종합 진단")
print("="*70)

fix_results = [
    {'fix': 'A (re-fit with Phase2 Gaussian)', 'status': 'skip (raw features unavailable)'},
    {'fix': 'B (z-score normalized)', 'det_rate': f'{det_b*100:.1f}%', 'our_recall': f'{recall_b*100:.1f}%'},
    {'fix': 'C (distance threshold)', 'det_rate': f'{det_c*100:.1f}%', 'our_recall': f'{recall_c*100:.1f}%'},
    {'fix': 'D (relative distance RF)', 'det_rate': f'{det_d*100:.1f}%', 'our_recall': f'{recall_d*100:.1f}%', 'cv_auroc': f'{auc_d:.4f}'},
    {'fix': 'E (simple inverted dist, t=0.5)', 'det_rate': f'{(score_e>=0.5).mean()*100:.1f}%',
     'our_recall': f'{(ours_score_e>=0.5).mean()*100:.1f}%'},
]
for r in fix_results:
    print(f"  {r['fix']}: {r.get('our_recall', 'N/A')} recall, {r.get('det_rate', 'N/A')} det")

with open(OUT / 'diagnosis_report.md', 'w') as f:
    f.write('# Mahalanobis Distribution Shift — Diagnosis Report\n\n')
    f.write('## 1. Distance 분포\n\n')
    f.write(pd.DataFrame(stats_rows).to_string(index=False) + '\n\n')
    f.write('![distance](./distance_distribution_comparison.png)\n\n')
    f.write('## 2. Class별 distance\n\n')
    f.write(f'Our 1002 avg: {dict(ours_avg.round(1))}\n')
    f.write(f'Others avg: {dict(others_avg.round(1))}\n')
    f.write(f'Our closest: {test_df[test_df["is_ours"]]["closest_class"].value_counts().to_dict()}\n\n')
    f.write('## 3. t-SNE\n\n![tsne](./feature_space_visualization.png)\n\n')
    f.write('## 4. RF probability\n\n![rf](./rf_probability_distributions.png)\n\n')
    f.write('## 5. Covariance\n\n')
    f.write(cov_df.to_string(index=False) + '\n\n')
    f.write('## 6. Fix attempts\n\n')
    f.write('| Fix | Det rate | Our recall | Notes |\n|---|---|---|---|\n')
    for r in fix_results:
        f.write(f'| {r["fix"]} | {r.get("det_rate","N/A")} | {r.get("our_recall","N/A")} | {r.get("cv_auroc","")} |\n')
    f.write('\n## 7. 진단 결론\n\n')

    # 핵심 진단
    pilot_scale = pilot_min_dist.mean()
    test_scale = test_min_dist.mean()
    ratio = test_scale / pilot_scale
    f.write(f'- Pilot mean distance: {pilot_scale:.1f}\n')
    f.write(f'- Test mean distance: {test_scale:.1f}\n')
    f.write(f'- Ratio (test/pilot): **{ratio:.2f}x**\n\n')
    if ratio > 2:
        f.write('⚠ Test set distances가 pilot보다 훨씬 큼 → **covariance scale mismatch** 확정\n')
        f.write('Phase 2 full Gaussian이 더 spread-out한 covariance → distance 값 전체적으로 커짐\n')
        f.write('Pilot에서 학습한 RF의 decision boundary가 test에서 의미 없음\n\n')
    elif ratio < 0.5:
        f.write('Test distances가 pilot보다 작음 → 다른 형태의 shift\n\n')
    else:
        f.write('Scale은 비슷 → 다른 원인 조사 필요\n\n')

print(f"\n경과: {time.time()-t0:.1f}s")
print("완료 ✓")
