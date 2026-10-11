"""
Step 3: Instability features + cosine similarity + t-SNE + report
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy import stats

BASE = Path('/home/student/GEN_AI_GTG')
CTRL_SCORES = BASE / 'detection/instability/control_experiment/scores'
CROP_SCORES = BASE / 'detection/crop_ablation/scores'
TEST_SCORES = BASE / 'detection/instability/test_application/scores'
OUT = BASE / 'detection/instability/control_experiment'
OUT.mkdir(parents=True, exist_ok=True)

DETECTORS = ['npr', 'distildire', 'aide_genimage', 'xception',
             'siglip_dino', 'aide_progan', 'haywood', 'aide_sd14']
CROP_VARIANTS = ['face_crop_variable', 'random_crop_192_a', 'center_crop_192']
FEAT_NAMES = [f'{d}__{v}' for d in DETECTORS for v in CROP_VARIANTS]
AUG_TYPES = ['original', 'gaussian_noise_002', 'gaussian_noise_005',
             'gaussian_blur_3', 'gaussian_blur_5', 'jpeg_q50', 'jpeg_q70',
             'rotation_15', 'poisson_10', 'rotation_20']

SUSPECT_STEMS = ['6488e0306ef3ca2f3c90', 'e819e3aad4a3226de7d0', 'ec634859158b8bcc121a',
                 '1b91da9e460c6a7d0cdc', 'c3957f266d7078d9e959']
ADV_STEMS = ['60feecd82812b3ca8519', '6229ffe049db851dd7f5', '5a647e1d6b497a8f4d0a',
             '6b94443c0a878d394159', '6c112186b0c66912301d']
REAL_STEMS = ['b5b10f28ca5ae7776c14', '4cbba497dc2a7fe4987a', '82b8053a47a4c2f0d74d',
              '986236cec8e28dffb100', '6f6f606dba8d02273619']

# ── 1. Augmented real instability features (10 aug × 20) ─────────────
print("[1/5] Augmented real instability features 추출...")
aug_features = {}  # aug_type -> (20, 24)
for aug in AUG_TYPES:
    feats = []
    for det in DETECTORS:
        df_full = pd.read_parquet(CTRL_SCORES / f'{det}__{aug}__full.parquet').sort_values('filename')
        for v in CROP_VARIANTS:
            df_v = pd.read_parquet(CTRL_SCORES / f'{det}__{aug}__{v}.parquet').sort_values('filename')
            feats.append(np.abs(df_full['score'].to_numpy() - df_v['score'].to_numpy()))
    aug_features[aug] = np.stack(feats, axis=1)  # (20, 24)
    print(f'  {aug}: {aug_features[aug].shape}')

# ── 2. Baseline features (suspect, adv, real) ─────────────────────────
print("\n[2/5] Baseline features 로드...")

def load_test_instability(stems):
    """test_application/scores 에서 instability 추출."""
    feats = []
    for det in DETECTORS:
        df_full = pd.read_parquet(TEST_SCORES / f'{det}__full.parquet').sort_values('filename')
        name2score_full = {Path(r['filename']).stem: float(r['score'])
                           for _, r in df_full.iterrows()}
        for v in CROP_VARIANTS:
            df_v = pd.read_parquet(TEST_SCORES / f'{det}__{v}.parquet').sort_values('filename')
            name2score_v = {Path(r['filename']).stem: float(r['score'])
                            for _, r in df_v.iterrows()}
            col = [abs(name2score_full.get(s, 0) - name2score_v.get(s, 0)) for s in stems]
            feats.append(col)
    return np.array(feats).T  # (n, 24)

X_suspect = load_test_instability(SUSPECT_STEMS)
X_adv     = load_test_instability(ADV_STEMS)
X_real    = load_test_instability(REAL_STEMS)
print(f'  suspect: {X_suspect.shape}  adv: {X_adv.shape}  real: {X_real.shape}')

# save augmented features parquet
rows = []
for aug in AUG_TYPES:
    for i in range(aug_features[aug].shape[0]):
        row = {'image_idx': i, 'aug_type': aug}
        for j, fn in enumerate(FEAT_NAMES):
            row[fn] = float(aug_features[aug][i, j])
        rows.append(row)
pd.DataFrame(rows).to_parquet(OUT / 'augmented_features.parquet', index=False)

# ── 3. Cosine similarity matrix ──────────────────────────────────────
print("\n[3/5] Cosine similarity 계산...")

def cos_sim(a, b):
    """a: (n, d), b: (m, d) → (n,m) pairwise cosine."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_n @ b_n.T

def mean_cos(group_a, group_b):
    """두 집단 간 평균/std cosine."""
    C = cos_sim(group_a, group_b)
    return float(C.mean()), float(C.std()), float(C.min()), float(C.max()), C.size

targets = {'suspect': X_suspect, 'adv_baseline': X_adv, 'real_baseline': X_real}
sim_rows = []
sources = {aug: aug_features[aug] for aug in AUG_TYPES}
sources['suspect'] = X_suspect
sources['adv_baseline'] = X_adv
sources['real_baseline'] = X_real

for src_name, src_mat in sources.items():
    for tgt_name, tgt_mat in targets.items():
        m, s, mn, mx, n = mean_cos(src_mat, tgt_mat)
        sim_rows.append({
            'group_a': src_name, 'group_b': tgt_name,
            'n_pairs': n, 'mean_cosine': round(m, 4),
            'std_cosine': round(s, 4), 'min': round(mn, 4), 'max': round(mx, 4),
        })
sim_df = pd.DataFrame(sim_rows)
sim_df.to_csv(OUT / 'similarity_matrix.csv', index=False)

# 핵심 출력
print(f"\n{'source':<22} | {'→ suspect':>12} | {'→ adv':>12} | {'→ real':>12}")
print('-' * 68)
for src_name in list(sources.keys()):
    sims = {}
    for tgt in ['suspect', 'adv_baseline', 'real_baseline']:
        r = sim_df[(sim_df['group_a']==src_name) & (sim_df['group_b']==tgt)]
        sims[tgt] = r.iloc[0]['mean_cosine'] if len(r) else float('nan')
    print(f"  {src_name:<20} | {sims['suspect']:>12.3f} | {sims['adv_baseline']:>12.3f} | {sims['real_baseline']:>12.3f}")

# ── 4. Similarity heatmap ────────────────────────────────────────────
print("\n[4/5] 시각화 생성...")
heat_sources = AUG_TYPES + ['suspect', 'adv_baseline', 'real_baseline']
heat_targets = ['suspect', 'adv_baseline', 'real_baseline']
heat_mat = np.zeros((len(heat_sources), len(heat_targets)))
for i, src in enumerate(heat_sources):
    for j, tgt in enumerate(heat_targets):
        r = sim_df[(sim_df['group_a']==src) & (sim_df['group_b']==tgt)]
        heat_mat[i, j] = r.iloc[0]['mean_cosine'] if len(r) else 0

fig, ax = plt.subplots(figsize=(6, 8))
im = ax.imshow(heat_mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax.set_yticks(range(len(heat_sources)))
ax.set_yticklabels(heat_sources, fontsize=9)
ax.set_xticks(range(len(heat_targets)))
ax.set_xticklabels(heat_targets, fontsize=10)
for i in range(heat_mat.shape[0]):
    for j in range(heat_mat.shape[1]):
        ax.text(j, i, f'{heat_mat[i,j]:.3f}', ha='center', va='center', fontsize=9,
                color='white' if heat_mat[i,j] > 0.6 else 'black')
ax.axhline(len(AUG_TYPES)-0.5, color='cyan', linewidth=2, linestyle='--')
ax.set_title('Cosine similarity to baseline groups\n(augmented real vs suspect/adv/real)', fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
plt.tight_layout()
plt.savefig(OUT / 'similarity_heatmap.png', dpi=130, bbox_inches='tight')
plt.close()

# ── Extended instability heatmap ─────────────────────────────────────
# 2 samples per aug (18) + 5 suspect + 5 adv + 5 real = 33
ext_labels = []; ext_vecs = []
for aug in AUG_TYPES:
    for idx in [0, 1]:
        ext_labels.append(f'{aug}[{idx}]')
        ext_vecs.append(aug_features[aug][idx])
for i, s in enumerate(SUSPECT_STEMS):
    ext_labels.append(f'suspect_{s[:8]}'); ext_vecs.append(X_suspect[i])
for i, s in enumerate(ADV_STEMS):
    ext_labels.append(f'adv_{s[:8]}'); ext_vecs.append(X_adv[i])
for i, s in enumerate(REAL_STEMS):
    ext_labels.append(f'real_{s[:8]}'); ext_vecs.append(X_real[i])
ext_mat = np.stack(ext_vecs)

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(ext_mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=max(0.5, ext_mat.max()))
ax.set_yticks(range(len(ext_labels)))
ax.set_yticklabels(ext_labels, fontsize=7, family='monospace')
ax.set_xticks(range(24))
ax.set_xticklabels(FEAT_NAMES, rotation=45, ha='right', fontsize=6.5)
ax.axhline(len(AUG_TYPES)*2 - 0.5, color='cyan', linewidth=2)
ax.axhline(len(AUG_TYPES)*2 + 5 - 0.5, color='lime', linewidth=2)
ax.axhline(len(AUG_TYPES)*2 + 10 - 0.5, color='orange', linewidth=2)
ax.set_title('Extended instability heatmap: aug samples + suspect + adv + real')
fig.colorbar(im, ax=ax, fraction=0.015)
plt.tight_layout()
plt.savefig(OUT / 'extended_instability_heatmap.png', dpi=130, bbox_inches='tight')
plt.close()

# ── t-SNE ────────────────────────────────────────────────────────────
all_vecs = []; all_labels = []; all_groups = []
for aug in AUG_TYPES:
    for i in range(aug_features[aug].shape[0]):
        all_vecs.append(aug_features[aug][i])
        all_labels.append(aug)
        all_groups.append('augmented')
for i in range(X_suspect.shape[0]):
    all_vecs.append(X_suspect[i]); all_labels.append('suspect'); all_groups.append('suspect')
for i in range(X_adv.shape[0]):
    all_vecs.append(X_adv[i]); all_labels.append('adv_baseline'); all_groups.append('adv_baseline')
for i in range(X_real.shape[0]):
    all_vecs.append(X_real[i]); all_labels.append('real_baseline'); all_groups.append('real_baseline')
all_vecs = np.stack(all_vecs)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
emb = tsne.fit_transform(all_vecs)

fig, ax = plt.subplots(figsize=(10, 8))
cmap = plt.cm.tab20
unique_labels = AUG_TYPES + ['suspect', 'adv_baseline', 'real_baseline']
colors = {l: cmap(i / len(unique_labels)) for i, l in enumerate(unique_labels)}
# special colors
colors['suspect'] = '#d62728'
colors['adv_baseline'] = '#ff7f0e'
colors['real_baseline'] = '#2ca02c'

for lbl in unique_labels:
    mask = [l == lbl for l in all_labels]
    idx = np.where(mask)[0]
    marker = '*' if lbl in ['suspect','adv_baseline','real_baseline'] else 'o'
    size = 120 if lbl in ['suspect','adv_baseline','real_baseline'] else 30
    ax.scatter(emb[idx, 0], emb[idx, 1], c=[colors[lbl]], label=lbl,
              marker=marker, s=size, alpha=0.8, edgecolors='black' if marker=='*' else 'none',
              linewidths=0.5)
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
ax.set_title('t-SNE: augmented real vs suspect vs adv vs real (24-dim instability)')
plt.tight_layout()
plt.savefig(OUT / 'tsne_instability.png', dpi=130, bbox_inches='tight')
plt.close()
print("  heatmaps + t-SNE 저장")

# ── 5. Verdict + Report ──────────────────────────────────────────────
print("\n[5/5] Verdict + Report...")
suspect_mean = X_suspect.mean(axis=0)
adv_mean = X_adv.mean(axis=0)
real_mean = X_real.mean(axis=0)

def cos1d(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

verdicts = {}
for aug in AUG_TYPES:
    aug_m = aug_features[aug].mean(axis=0)
    sim_sus = cos1d(aug_m, suspect_mean)
    sim_adv = cos1d(aug_m, adv_mean)
    if sim_sus > 0.90:
        v = f'{aug} is SIMILAR to suspect/adversarial (cos={sim_sus:.3f})'
    elif sim_sus < 0.70:
        v = f'{aug} is DISTINCT from suspect (cos={sim_sus:.3f}) — confirms adversarial-specific'
    else:
        v = f'{aug} is PARTIALLY similar (cos={sim_sus:.3f}) — grey area'
    verdicts[aug] = (v, sim_sus, sim_adv)
    print(f"  {aug:<22}  →suspect {sim_sus:.3f}  →adv {sim_adv:.3f}  | {v.split('—')[0].strip()}")

# 종합 판정
all_sims = [sim_s for aug_k, (_, sim_s, _) in verdicts.items() if aug_k != 'original']
max_sim = max(all_sims)
n_high = sum(1 for s in all_sims if s > 0.85)
n_distinct = sum(1 for s in all_sims if s < 0.70)

if max_sim < 0.85 and n_distinct >= 6:
    overall = 'A'
    overall_text = 'Instability detector is ADVERSARIAL-SPECIFIC. No augmentation matches suspect pattern.'
elif max_sim > 0.90:
    culprit = [aug for aug, (_, s, _) in verdicts.items() if s > 0.90 and aug != 'original']
    overall = 'B'
    overall_text = f'Detector also responds to: {culprit}. Not purely adversarial-specific.'
else:
    overall = 'C'
    overall_text = 'Mixed results. Some augmentations partially overlap adversarial pattern.'

print(f'\n종합 판정: Case {overall}')
print(f'  {overall_text}')

# Write verdict.md
with open(OUT / 'verdict.md', 'w') as f:
    f.write(f'# Augmented Real Control Experiment — Verdict\n\n')
    f.write(f'## 종합: **Case {overall}**\n\n')
    f.write(f'{overall_text}\n\n')
    f.write(f'## Augmentation별 cosine similarity to suspect\n\n')
    f.write('| aug_type | →suspect | →adv | verdict |\n|---|---|---|---|\n')
    for aug in AUG_TYPES:
        v_text, s_sus, s_adv = verdicts[aug]
        short = v_text.split('(')[0].strip()
        f.write(f'| {aug} | {s_sus:.3f} | {s_adv:.3f} | {short} |\n')
    f.write(f'\nmax similarity to suspect (excluding original): **{max_sim:.3f}**\n')
    f.write(f'Augmentations with cos > 0.85: {n_high}\n')
    f.write(f'Augmentations with cos < 0.70: {n_distinct}\n')

# Write full report
with open(OUT / 'control_experiment_report.md', 'w') as f:
    f.write('# Augmented Real — Control Experiment Report\n\n')
    f.write(f'> 9 augmentations × 20 real images + 15 baselines (suspect/adv/real)\n')
    f.write(f'> 8 detectors × 4 variants, 24-dim instability features\n\n')
    f.write('---\n\n## 1. Cosine Similarity (mean across pairs)\n\n')
    f.write(f'{"source":<22} | {"→suspect":>10} | {"→adv":>10} | {"→real":>10}\n')
    f.write('-' * 60 + '\n')
    for src in list(sources.keys()):
        sims = {}
        for tgt in ['suspect', 'adv_baseline', 'real_baseline']:
            r = sim_df[(sim_df['group_a']==src) & (sim_df['group_b']==tgt)]
            sims[tgt] = r.iloc[0]['mean_cosine'] if len(r) else 0
        f.write(f'{src:<22} | {sims["suspect"]:>10.3f} | {sims["adv_baseline"]:>10.3f} | {sims["real_baseline"]:>10.3f}\n')

    f.write('\n## 2. Augmentation별 판정\n\n')
    f.write('| aug_type | cos(→suspect) | cos(→adv) | verdict |\n|---|---|---|---|\n')
    for aug in AUG_TYPES:
        v_text, s_sus, s_adv = verdicts[aug]
        f.write(f'| {aug} | {s_sus:.3f} | {s_adv:.3f} | {v_text} |\n')

    f.write(f'\n## 3. 종합 판정: **Case {overall}**\n\n')
    f.write(f'{overall_text}\n\n')

    # top 3 most/least similar
    ranked = sorted([(aug, s) for aug, (_, s, _) in verdicts.items() if aug != 'original'],
                    key=lambda x: -x[1])
    f.write('### suspect와 가장 유사한 augmentation (상위 3)\n\n')
    for aug, s in ranked[:3]:
        f.write(f'- **{aug}**: cos={s:.3f}\n')
    f.write('\n### suspect와 가장 다른 augmentation (상위 3)\n\n')
    for aug, s in ranked[-3:]:
        f.write(f'- **{aug}**: cos={s:.3f}\n')

    f.write('\n## 4. t-SNE 해석\n\n')
    f.write('![t-SNE](./tsne_instability.png)\n\n')
    f.write('위 t-SNE plot에서:\n')
    f.write('- ★ suspect (빨강)와 ★ adv_baseline (주황)이 같은 cluster에 위치하는지\n')
    f.write('- ○ augmented real이 별도 cluster인지\n')
    f.write('- ★ real_baseline (초록)이 어디에 위치하는지\n')
    f.write('를 확인하면 instability 신호의 구별 능력을 시각적으로 판단 가능.\n\n')

    f.write('## 5. 출력 파일\n\n')
    f.write('- [augmented_features.parquet](./augmented_features.parquet)\n')
    f.write('- [similarity_matrix.csv](./similarity_matrix.csv)\n')
    f.write('- [similarity_heatmap.png](./similarity_heatmap.png)\n')
    f.write('- [extended_instability_heatmap.png](./extended_instability_heatmap.png)\n')
    f.write('- [tsne_instability.png](./tsne_instability.png)\n')
    f.write('- [verdict.md](./verdict.md)\n')
    f.write('- [sample_augmented_images.png](./sample_augmented_images.png)\n')

print(f'\n출력: {OUT}/')
print('완료 ✓')
