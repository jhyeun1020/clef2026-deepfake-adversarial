"""
Mahalanobis feature space t-SNE 분석.
Our 1002 vs test others vs pilot 400 — cluster 구조 + 가설 검증.
"""
import csv, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy import stats as sp_stats

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

BASE = Path('/home/student/GEN_AI_GTG')
MAH = BASE / 'detection/mahalanobis'
OUT = MAH / 'tsne_analysis'
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()

# ── 1. Data prep ─────────────────────────────────────────────────────
print("[1/7] Data loading...")
pilot_df = pd.read_parquet(MAH / 'fixed/pilot_distances_phase2scale.parquet')
test_df  = pd.read_parquet(MAH / 'fixed/test_predictions_fixed.parquet')

mah_cols_pilot = [c for c in pilot_df.columns if c.startswith('mah_')]
mah_cols_test  = [c for c in test_df.columns if c.startswith('mah_L')]

ours_set = set()
for r in csv.DictReader(open(BASE / 'detection/phash_matching/fake_gt_matches.csv')):
    ours_set.add(Path(r['test_image_path']).name)
test_df['is_ours'] = test_df['filename'].isin(ours_set)

# Our 1002
ours_df = test_df[test_df['is_ours']].copy()
others_df = test_df[~test_df['is_ours']].copy()

# Random 2000 from others
rng = np.random.RandomState(42)
others_sample = others_df.sample(n=min(2000, len(others_df)), random_state=42)

X_ours = ours_df[mah_cols_test].values            # (1002, 20)
X_others = others_sample[mah_cols_test].values     # (2000, 20)
X_pilot = pilot_df[mah_cols_pilot].values          # (400, 20)

print(f"  ours: {X_ours.shape}, others: {X_others.shape}, pilot: {X_pilot.shape}")

# ── 2. Labels / probabilities ────────────────────────────────────────
prob_ours = ours_df['adv_prob_mah_fixed'].values
prob_others = others_sample['adv_prob_mah_fixed'].values
pilot_labels = pilot_df['label'].values

# Others subgroup
def prob_group(p):
    if p > 0.7: return 'high'
    if p > 0.4: return 'mid'
    return 'low'
others_groups = [prob_group(p) for p in prob_others]

# ── 3. t-SNE ─────────────────────────────────────────────────────────
print("\n[2/7] t-SNE (n=3402)...")
X_all = np.vstack([X_ours, X_others, X_pilot])
n_ours, n_others, n_pilot = len(X_ours), len(X_others), len(X_pilot)
N = n_ours + n_others + n_pilot

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
emb = tsne.fit_transform(X_all)

emb_ours   = emb[:n_ours]
emb_others = emb[n_ours:n_ours+n_others]
emb_pilot  = emb[n_ours+n_others:]
print(f"  t-SNE 완료 ({time.time()-t0:.0f}s)")

# ── 4a. Overview ─────────────────────────────────────────────────────
print("\n[3/7] Visualizations...")

fig, ax = plt.subplots(figsize=(12, 9))
# Others by prob group
for grp, color, ms in [('low', '#cccccc', 8), ('mid', '#ff7f0e', 15), ('high', '#d62728', 25)]:
    mask = [g == grp for g in others_groups]; idx = np.where(mask)[0]
    ax.scatter(emb_others[idx, 0], emb_others[idx, 1], c=color, s=ms, alpha=0.4,
              label=f'others_{grp} ({len(idx)})', marker='o', edgecolors='none')
# Pilot
for lbl, color, marker in [('real', '#2ca02c', 'o'), ('adv', '#9467bd', 'D'),
                             ('clean_fake', '#1f77b4', 's'), ('jpeg', '#bcbd22', '^')]:
    mask = pilot_labels == lbl; idx = np.where(mask)[0]
    ax.scatter(emb_pilot[idx, 0], emb_pilot[idx, 1], c=color, s=50, alpha=0.8,
              label=f'pilot_{lbl} ({len(idx)})', marker=marker, edgecolors='black', linewidths=0.3)
# Ours
ax.scatter(emb_ours[:, 0], emb_ours[:, 1], c='#d62728', s=30, alpha=0.7,
          label=f'our_1002 ({n_ours})', marker='*', edgecolors='darkred', linewidths=0.2)
ax.legend(fontsize=8, loc='best', frameon=True, ncol=2)
ax.set_title('t-SNE overview: Mahalanobis 20-dim feature space')
plt.tight_layout()
plt.savefig(OUT / 'tsne_overview.png', dpi=130, bbox_inches='tight')
plt.close()

# ── 4b. Our vs Pilot adv ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(emb_others[:, 0], emb_others[:, 1], c='#dddddd', s=5, alpha=0.3, label='test_others')
pilot_adv_mask = pilot_labels == 'adv'; pilot_adv_idx = np.where(pilot_adv_mask)[0]
ax.scatter(emb_pilot[pilot_adv_idx, 0], emb_pilot[pilot_adv_idx, 1], c='#9467bd', s=80,
          marker='D', label=f'pilot_adv ({len(pilot_adv_idx)})', edgecolors='black', linewidths=0.5)
ax.scatter(emb_ours[:, 0], emb_ours[:, 1], c='#d62728', s=25, alpha=0.7,
          marker='*', label=f'our_1002 ({n_ours})', edgecolors='darkred', linewidths=0.2)
ax.legend(fontsize=10); ax.set_title('Our 1002 vs Pilot adversarial (test others = background)')
plt.tight_layout()
plt.savefig(OUT / 'tsne_our_vs_pilot.png', dpi=130, bbox_inches='tight')
plt.close()

# ── 4c. High-prob others ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(emb_others[:, 0], emb_others[:, 1], c='#dddddd', s=5, alpha=0.3, label='test_others_low')
high_mask = [g == 'high' for g in others_groups]; high_idx = np.where(high_mask)[0]
mid_mask = [g == 'mid' for g in others_groups]; mid_idx = np.where(mid_mask)[0]
ax.scatter(emb_others[mid_idx, 0], emb_others[mid_idx, 1], c='#ff7f0e', s=20, alpha=0.6,
          marker='o', label=f'others_mid (0.4-0.7, {len(mid_idx)})')
ax.scatter(emb_others[high_idx, 0], emb_others[high_idx, 1], c='red', s=40, alpha=0.8,
          marker='s', label=f'others_high (>0.7, {len(high_idx)})', edgecolors='darkred')
ax.scatter(emb_ours[:, 0], emb_ours[:, 1], c='#d62728', s=15, alpha=0.5, marker='*', label='our_1002')
ax.legend(fontsize=9); ax.set_title('High-prob others (red squares) vs Our 1002 (red stars)')
plt.tight_layout()
plt.savefig(OUT / 'tsne_high_prob_others.png', dpi=130, bbox_inches='tight')
plt.close()

# ── 4d. Mid-prob zoom ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(emb_others[:, 0], emb_others[:, 1], c='#dddddd', s=5, alpha=0.3)
ax.scatter(emb_others[mid_idx, 0], emb_others[mid_idx, 1], c='#ff7f0e', s=25, alpha=0.7,
          marker='o', label=f'mid_prob (0.4-0.7, {len(mid_idx)})')
# Pilot real for reference
pilot_real_idx = np.where(pilot_labels == 'real')[0]
ax.scatter(emb_pilot[pilot_real_idx, 0], emb_pilot[pilot_real_idx, 1], c='#2ca02c', s=50,
          marker='o', label='pilot_real', edgecolors='black', linewidths=0.3)
ax.legend(fontsize=10); ax.set_title('Mid-prob region (0.4-0.7) — real boundary or separate cluster?')
plt.tight_layout()
plt.savefig(OUT / 'tsne_mid_prob.png', dpi=130, bbox_inches='tight')
plt.close()
print("  4 plots saved")

# ── 5. Cluster analysis ──────────────────────────────────────────────
print("\n[4/7] Cluster analysis...")

# 5a. Our 1002 tightness
ours_centroid = emb_ours.mean(axis=0)
ours_dists = np.sqrt(((emb_ours - ours_centroid)**2).sum(axis=1))
print(f"  Our 1002 tightness: mean_dist_to_centroid={ours_dists.mean():.2f}, std={ours_dists.std():.2f}")

# 5b. Nearest neighbor
nn = NearestNeighbors(n_neighbors=5)
nn.fit(emb_others)
dists_nn, idx_nn = nn.kneighbors(emb_ours)

nn_probs = []
for i in range(len(emb_ours)):
    for j in idx_nn[i]:
        nn_probs.append(prob_others[j])
nn_probs = np.array(nn_probs)
print(f"  Our nearest neighbors (in others) prob: mean={nn_probs.mean():.3f}, "
      f"median={np.median(nn_probs):.3f}, >0.5={int((nn_probs>=0.5).sum())}/{len(nn_probs)}")

# 5c. Separation
pilot_real_centroid = emb_pilot[pilot_labels=='real'].mean(axis=0)
pilot_adv_centroid = emb_pilot[pilot_labels=='adv'].mean(axis=0)
high_others_centroid = emb_others[high_idx].mean(axis=0) if len(high_idx) > 0 else np.array([0, 0])

sep_our_real = np.sqrt(((ours_centroid - pilot_real_centroid)**2).sum())
sep_our_pilotadv = np.sqrt(((ours_centroid - pilot_adv_centroid)**2).sum())
sep_our_highothers = np.sqrt(((ours_centroid - high_others_centroid)**2).sum()) if len(high_idx) > 0 else float('nan')
print(f"  Separation: our↔pilot_real={sep_our_real:.1f}, our↔pilot_adv={sep_our_pilotadv:.1f}, "
      f"our↔high_others={sep_our_highothers:.1f}")

with open(OUT / 'cluster_analysis.md', 'w') as f:
    f.write('# Cluster Analysis\n\n')
    f.write(f'## Our 1002 tightness\n- mean dist to centroid: {ours_dists.mean():.2f}\n- std: {ours_dists.std():.2f}\n\n')
    f.write(f'## Nearest neighbors (our→others in t-SNE)\n')
    f.write(f'- 5-NN prob mean: {nn_probs.mean():.3f}, median: {np.median(nn_probs):.3f}\n')
    f.write(f'- 5-NN prob > 0.5: {int((nn_probs>=0.5).sum())}/{len(nn_probs)}\n\n')
    f.write(f'## Separation (centroid distances in t-SNE)\n')
    f.write(f'- our ↔ pilot_real: {sep_our_real:.1f}\n')
    f.write(f'- our ↔ pilot_adv: {sep_our_pilotadv:.1f}\n')
    f.write(f'- our ↔ high_prob_others: {sep_our_highothers:.1f}\n')

# ── 6. Hypothesis analysis ───────────────────────────────────────────
print("\n[5/7] Hypothesis analysis...")
n_high = len(high_idx)
n_mid = len(mid_idx)
frac_high = n_high / n_others * 100
frac_mid = n_mid / n_others * 100

# Check if high-prob others are near our cluster or elsewhere
if n_high > 0:
    high_near_ours = np.sqrt(((emb_others[high_idx] - ours_centroid)**2).sum(axis=1))
    high_near_real = np.sqrt(((emb_others[high_idx] - pilot_real_centroid)**2).sum(axis=1))
    frac_near_ours = (high_near_ours < high_near_real).mean()
else:
    frac_near_ours = 0

with open(OUT / 'hypothesis_analysis.md', 'w') as f:
    f.write('# Hypothesis Analysis\n\n')
    f.write(f'## Test others subgroups\n')
    f.write(f'- high (>0.7): {n_high} ({frac_high:.1f}%)\n')
    f.write(f'- mid (0.4-0.7): {n_mid} ({frac_mid:.1f}%)\n')
    f.write(f'- low (<0.4): {n_others - n_high - n_mid}\n\n')

    f.write('## 가설 검증\n\n')
    f.write(f'### H1: Others 대부분 real → adversarial 없음\n')
    if frac_high < 3:
        f.write(f'**지지됨**: high-prob others가 {frac_high:.1f}%로 매우 소수\n\n')
    else:
        f.write(f'**기각됨**: high-prob others가 {frac_high:.1f}%로 무시 못 함\n\n')

    f.write(f'### H2: 별도 cluster → 다른 스타일 adversarial\n')
    if n_high > 0:
        f.write(f'high-prob others 중 our cluster에 더 가까운 비율: {frac_near_ours*100:.1f}%\n')
        if frac_near_ours > 0.5:
            f.write('→ our cluster 근처에 위치 → **같은 유형의 adversarial일 가능성**\n\n')
        else:
            f.write('→ our cluster와 다른 위치 → **다른 유형이거나 non-adversarial**\n\n')

    f.write(f'### H3: Our 1002만 outlier → overfitting\n')
    f.write(f'our ↔ pilot_adv 거리: {sep_our_pilotadv:.1f}\n')
    f.write(f'our tightness: {ours_dists.mean():.1f}\n')
    if sep_our_pilotadv < ours_dists.mean() * 3:
        f.write('→ pilot_adv와 가까움 → **overfitting 아님, 진짜 adversarial signal**\n\n')
    else:
        f.write('→ pilot_adv와 멀음 → **overfitting 가능성 있음**\n\n')

# ── 7. HDBSCAN ────────────────────────────────────────────────────────
print("\n[6/7] HDBSCAN clustering...")
if HAS_HDBSCAN:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
    cluster_labels = clusterer.fit_predict(emb)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"  Found {n_clusters} clusters (+ noise)")

    # Cluster membership
    membership_rows = []
    for cl in sorted(set(cluster_labels)):
        mask = cluster_labels == cl
        n_total = int(mask.sum())
        n_ours_in = int(mask[:n_ours].sum())
        n_others_in = int(mask[n_ours:n_ours+n_others].sum())
        n_pilot_in = int(mask[n_ours+n_others:].sum())
        # prob stats for others in this cluster
        others_in = mask[n_ours:n_ours+n_others]
        if others_in.sum() > 0:
            probs_in = prob_others[others_in]
            mean_prob = float(probs_in.mean())
        else:
            mean_prob = float('nan')
        membership_rows.append({
            'cluster': cl, 'total': n_total,
            'our_1002': n_ours_in, 'test_others': n_others_in, 'pilot': n_pilot_in,
            'others_mean_prob': round(mean_prob, 3),
        })
    mem_df = pd.DataFrame(membership_rows)
    mem_df.to_csv(OUT / 'cluster_membership.csv', index=False)
    print(mem_df.to_string(index=False))

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_cl = sorted(set(cluster_labels))
    cmap = plt.cm.tab20
    for ci, cl in enumerate(unique_cl):
        mask = cluster_labels == cl
        color = '#cccccc' if cl == -1 else cmap(ci / max(len(unique_cl), 1))
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[color], s=8, alpha=0.5,
                  label=f'cl={cl} (n={int(mask.sum())})')
    ax.scatter(emb_ours[:, 0], emb_ours[:, 1], c='red', s=10, alpha=0.3, marker='*')
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.set_title(f'HDBSCAN clusters ({n_clusters} found)')
    plt.tight_layout()
    plt.savefig(OUT / 'hdbscan_clusters.png', dpi=130, bbox_inches='tight')
    plt.close()
else:
    print("  hdbscan not installed, trying DBSCAN from sklearn...")
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=3.0, min_samples=20)
    cluster_labels = db.fit_predict(emb)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"  DBSCAN found {n_clusters} clusters")

    membership_rows = []
    for cl in sorted(set(cluster_labels)):
        mask = cluster_labels == cl
        n_total = int(mask.sum())
        n_ours_in = int(mask[:n_ours].sum())
        n_others_in = int(mask[n_ours:n_ours+n_others].sum())
        n_pilot_in = int(mask[n_ours+n_others:].sum())
        others_in = mask[n_ours:n_ours+n_others]
        mean_prob = float(prob_others[others_in].mean()) if others_in.sum() > 0 else float('nan')
        membership_rows.append({
            'cluster': cl, 'total': n_total,
            'our_1002': n_ours_in, 'test_others': n_others_in, 'pilot': n_pilot_in,
            'others_mean_prob': round(mean_prob, 3),
        })
    mem_df = pd.DataFrame(membership_rows)
    mem_df.to_csv(OUT / 'cluster_membership.csv', index=False)
    print(mem_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 8))
    for cl in sorted(set(cluster_labels)):
        mask = cluster_labels == cl
        c = '#cccccc' if cl == -1 else plt.cm.tab20(cl / max(n_clusters, 1))
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[c], s=8, alpha=0.5,
                  label=f'cl={cl} (n={int(mask.sum())})')
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f'DBSCAN clusters ({n_clusters})')
    plt.tight_layout()
    plt.savefig(OUT / 'dbscan_clusters.png', dpi=130, bbox_inches='tight')
    plt.close()

print(f"\n[7/7] 완료 ({time.time()-t0:.0f}s)")
