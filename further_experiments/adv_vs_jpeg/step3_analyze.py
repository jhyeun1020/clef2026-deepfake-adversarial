"""
Step 3: Features + similarity + t-SNE + classification + report
"""
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.manifold import TSNE

BASE = Path('/home/student/GEN_AI_GTG')
SCORES = BASE / 'detection/adv_vs_jpeg_decomposition/scores'
CROP_SCORES = BASE / 'detection/crop_ablation/scores'
OUT = BASE / 'detection/adv_vs_jpeg_decomposition'

DETECTORS = ['npr', 'distildire', 'aide_genimage', 'xception',
             'siglip_dino', 'aide_progan', 'haywood', 'aide_sd14']
CROP_VARIANTS = ['face_crop_variable', 'random_crop_192_a', 'center_crop_192']
FEAT_NAMES = [f'{d}__{v}' for d in DETECTORS for v in CROP_VARIANTS]

# Load image list
img_list = list(csv.DictReader(open(OUT / 'image_list.csv')))
uid2ver = {r['uid']: r['version'] for r in img_list}
VERSIONS = ['A_pure_adv', 'B_pure_jpeg', 'C_adv_jpeg', 'D_clean_fake', 'D_real']

# ── 1. Features ──────────────────────────────────────────────────────
print("[1/5] Instability features 추출...")
# Load scores and build features
def load_scores(det, v):
    df = pd.read_parquet(SCORES / f'{det}__{v}.parquet').sort_values('filename')
    return {Path(r['filename']).stem: float(r['score']) for _, r in df.iterrows()}

all_uids = sorted(uid2ver.keys())
feats = []
for det in DETECTORS:
    full = load_scores(det, 'full')
    for v in CROP_VARIANTS:
        sv = load_scores(det, v)
        feats.append([abs(full.get(uid, 0) - sv.get(uid, 0)) for uid in all_uids])
X = np.array(feats).T  # (500, 24)
versions = [uid2ver[uid] for uid in all_uids]
print(f"  X: {X.shape}, versions: {len(set(versions))}")

# Save
rows = []
for i, uid in enumerate(all_uids):
    row = {'uid': uid, 'version': uid2ver[uid]}
    for j, fn in enumerate(FEAT_NAMES):
        row[fn] = float(X[i, j])
    rows.append(row)
pd.DataFrame(rows).to_parquet(OUT / 'features.parquet', index=False)

# ── 2. Cosine Similarity ─────────────────────────────────────────────
print("\n[2/5] Cosine similarity matrix...")
ver_means = {}
ver_mats = {}
for ver in VERSIONS:
    mask = [v == ver for v in versions]
    idx = np.where(mask)[0]
    ver_mats[ver] = X[idx]
    ver_means[ver] = X[idx].mean(axis=0)

def cos1d(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

def pairwise_cos(ma, mb):
    an = ma / (np.linalg.norm(ma, axis=1, keepdims=True) + 1e-12)
    bn = mb / (np.linalg.norm(mb, axis=1, keepdims=True) + 1e-12)
    return (an @ bn.T).mean()

sim_rows = []
print(f"\n  {'':>14}", end='')
for vb in VERSIONS: print(f'  {vb[:8]:>8}', end='')
print()
for va in VERSIONS:
    print(f"  {va[:14]:>14}", end='')
    for vb in VERSIONS:
        s = float(pairwise_cos(ver_mats[va], ver_mats[vb]))
        sim_rows.append({'group_a': va, 'group_b': vb, 'mean_cosine': round(s, 4)})
        print(f'  {s:>8.3f}', end='')
    print()
pd.DataFrame(sim_rows).to_csv(OUT / 'similarity_matrix.csv', index=False)

# Key numbers
sim_AB = cos1d(ver_means['A_pure_adv'], ver_means['B_pure_jpeg'])
sim_AC = cos1d(ver_means['A_pure_adv'], ver_means['C_adv_jpeg'])
sim_BC = cos1d(ver_means['B_pure_jpeg'], ver_means['C_adv_jpeg'])
sim_A_clean = cos1d(ver_means['A_pure_adv'], ver_means['D_clean_fake'])
sim_A_real = cos1d(ver_means['A_pure_adv'], ver_means['D_real'])
print(f"\n  Key (mean vector cosine):")
print(f"    A(pure adv) vs B(pure jpeg): {sim_AB:.3f}")
print(f"    A(pure adv) vs C(adv+jpeg):  {sim_AC:.3f}")
print(f"    B(pure jpeg) vs C(adv+jpeg): {sim_BC:.3f}")
print(f"    A(pure adv) vs clean_fake:   {sim_A_clean:.3f}")
print(f"    A(pure adv) vs real:         {sim_A_real:.3f}")

# ── 3. t-SNE ─────────────────────────────────────────────────────────
print("\n[3/5] t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
emb = tsne.fit_transform(X)

fig, ax = plt.subplots(figsize=(9, 7))
colors = {'A_pure_adv': '#d62728', 'B_pure_jpeg': '#1f77b4', 'C_adv_jpeg': '#9467bd',
          'D_clean_fake': '#2ca02c', 'D_real': '#888888'}
markers = {'A_pure_adv': '*', 'B_pure_jpeg': 's', 'C_adv_jpeg': 'D',
           'D_clean_fake': 'o', 'D_real': 'v'}
for ver in VERSIONS:
    mask = [v == ver for v in versions]; idx = np.where(mask)[0]
    size = 80 if ver in ['A_pure_adv', 'C_adv_jpeg'] else 30
    ax.scatter(emb[idx, 0], emb[idx, 1], c=colors[ver], marker=markers[ver],
              s=size, alpha=0.7, label=ver, edgecolors='black' if size > 50 else 'none',
              linewidths=0.3)
ax.legend(fontsize=9, frameon=True)
ax.set_title(f't-SNE: Pure Adv vs Pure JPEG vs Adv+JPEG vs Clean\n'
             f'cos(A↔B)={sim_AB:.3f} cos(A↔C)={sim_AC:.3f} cos(B↔C)={sim_BC:.3f}')
plt.tight_layout()
plt.savefig(OUT / 'tsne_decomposition.png', dpi=130, bbox_inches='tight')
plt.close()

# ── 4. Feature decomposition ─────────────────────────────────────────
print("\n[4/5] Feature decomposition...")
fd_rows = []
for j, fn in enumerate(FEAT_NAMES):
    row = {'feature': fn}
    for ver in VERSIONS:
        mask = [v == ver for v in versions]; idx = np.where(mask)[0]
        row[f'{ver}_mean'] = round(float(X[idx, j].mean()), 4)
        row[f'{ver}_std'] = round(float(X[idx, j].std()), 4)
    fd_rows.append(row)
pd.DataFrame(fd_rows).to_csv(OUT / 'feature_decomposition.csv', index=False)

# ── 5. Classification ────────────────────────────────────────────────
print("\n[5/5] Classification experiments...")

# 5a: Existing RF on pure adversarial
print("  5a: Existing RF model → pure adversarial detection rate")
def build_train_X(ds):
    feats = []
    for det in DETECTORS:
        df_full = pd.read_parquet(CROP_SCORES / f'{det}__{ds}__full.parquet').sort_values('filename')
        for v in CROP_VARIANTS:
            df_v = pd.read_parquet(CROP_SCORES / f'{det}__{ds}__{v}.parquet').sort_values('filename')
            feats.append(np.abs(df_full['score'].to_numpy() - df_v['score'].to_numpy()))
    return np.stack(feats, axis=1)
X_real_train = build_train_X('real_reference')
X_adv_train = build_train_X('adversarial_fake')
X_train = np.vstack([X_real_train, X_adv_train])
y_train = np.array([0]*len(X_real_train) + [1]*len(X_adv_train))
clf_existing = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, y_train)

X_A = ver_mats['A_pure_adv']
X_B = ver_mats['B_pure_jpeg']
X_C = ver_mats['C_adv_jpeg']
prob_A = clf_existing.predict_proba(X_A)[:, 1]
prob_B = clf_existing.predict_proba(X_B)[:, 1]
prob_C = clf_existing.predict_proba(X_C)[:, 1]
det_A = (prob_A >= 0.5).mean()
det_B = (prob_B >= 0.5).mean()
det_C = (prob_C >= 0.5).mean()
print(f"    A (pure adv):  det={det_A*100:.0f}%  mean_prob={prob_A.mean():.3f}")
print(f"    B (pure jpeg): det={det_B*100:.0f}%  mean_prob={prob_B.mean():.3f}")
print(f"    C (adv+jpeg):  det={det_C*100:.0f}%  mean_prob={prob_C.mean():.3f}")

# 5b: 3-class classification
print("\n  5b: 3-class classification (clean vs pure_adv vs pure_jpeg)")
mask_clean = [v in ['D_clean_fake', 'D_real'] for v in versions]
mask_A = [v == 'A_pure_adv' for v in versions]
mask_B = [v == 'B_pure_jpeg' for v in versions]
X_3c = np.vstack([X[np.where(mask_clean)[0]], X[np.where(mask_A)[0]], X[np.where(mask_B)[0]]])
y_3c = np.array([0]*sum(mask_clean) + [1]*sum(mask_A) + [2]*sum(mask_B))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf_3c = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
proba_3c = cross_val_predict(clf_3c, X_3c, y_3c, cv=skf, method='predict_proba')
pred_3c = proba_3c.argmax(axis=1)

# Class 1 vs Class 2 AUROC
mask_12 = (y_3c == 1) | (y_3c == 2)
y_12 = (y_3c[mask_12] == 1).astype(int)
p_12 = proba_3c[mask_12, 1] / (proba_3c[mask_12, 1] + proba_3c[mask_12, 2] + 1e-8)
auc_12 = roc_auc_score(y_12, p_12)
acc_3c = accuracy_score(y_3c, pred_3c)
cm = confusion_matrix(y_3c, pred_3c)
print(f"    3-class accuracy: {acc_3c:.4f}")
print(f"    Class 1(adv) vs Class 2(jpeg) AUROC: {auc_12:.4f}")
print(f"    Confusion matrix:\n{cm}")

# ── Report ────────────────────────────────────────────────────────────
print("\n\nReport 생성...")
if sim_AB > 0.85:
    v_sim = '⚠ 순수 adversarial과 순수 JPEG 매우 유사 — 구분 불가능'
elif sim_AB < 0.65:
    v_sim = '✓ 순수 adversarial과 JPEG 명확히 구분됨'
else:
    v_sim = '~ 중간 — 부분적 구분 가능'

if det_A > 0.80:
    v_det = '✓ 모델이 adversarial 자체를 잘 포착'
elif det_A < 0.50:
    v_det = '⚠ 모델이 주로 JPEG을 잡고 있었음'
else:
    v_det = '~ 모델이 adversarial을 부분적으로 포착'

print(f"  Q1 (sim A↔B): {sim_AB:.3f} → {v_sim}")
print(f"  Q2 (det A):   {det_A*100:.0f}% → {v_det}")
print(f"  Q3 (3-class): AUC(adv vs jpeg) = {auc_12:.3f}")

with open(OUT / 'report.md', 'w') as f:
    f.write('# Pure Adversarial vs JPEG Decomposition\n\n')
    f.write(f'> A: Pure PGD (100), B: Pure JPEG Q80 (100), C: PGD+JPEG (100), D: Clean (200)\n\n')
    f.write('---\n\n## 1. Cosine Similarity Matrix (mean vector)\n\n')
    f.write(f'| | A(pure adv) | B(pure jpeg) | C(adv+jpeg) | clean_fake | real |\n|---|---|---|---|---|---|\n')
    for va in VERSIONS:
        cells = [f'{cos1d(ver_means[va], ver_means[vb]):.3f}' for vb in VERSIONS]
        f.write(f'| {va} | ' + ' | '.join(cells) + ' |\n')
    f.write(f'\n**Key**: A↔B = **{sim_AB:.3f}**, A↔C = **{sim_AC:.3f}**, B↔C = **{sim_BC:.3f}**\n\n')
    f.write(f'## 2. Q1: 순수 adversarial과 순수 JPEG 유사도\n\n')
    f.write(f'**{v_sim}**\n\n')
    f.write(f'## 3. Q2: 기존 모델이 잡는 것은?\n\n')
    f.write(f'| version | det rate | mean prob |\n|---|---|---|\n')
    f.write(f'| A (pure adv) | **{det_A*100:.0f}%** | {prob_A.mean():.3f} |\n')
    f.write(f'| B (pure jpeg) | {det_B*100:.0f}% | {prob_B.mean():.3f} |\n')
    f.write(f'| C (adv+jpeg) | {det_C*100:.0f}% | {prob_C.mean():.3f} |\n\n')
    f.write(f'**{v_det}**\n\n')
    f.write(f'## 4. Q3: 3-class 분리\n\n')
    f.write(f'- 3-class accuracy: **{acc_3c:.4f}**\n')
    f.write(f'- **Adv vs JPEG AUROC: {auc_12:.4f}**\n')
    f.write(f'- Confusion matrix (0=clean, 1=adv, 2=jpeg):\n\n')
    f.write(f'```\n{cm}\n```\n\n')
    if auc_12 > 0.9:
        f.write('→ 순수 adversarial과 JPEG이 **잘 구분됨**. 분리 학습 시 pure adversarial signature 포착 가능.\n\n')
    elif auc_12 > 0.7:
        f.write('→ 부분적 구분 가능. 일부 overlap 존재.\n\n')
    else:
        f.write('→ 구분 어려움. 다른 feature 필요.\n\n')
    f.write(f'## 5. Q4: cos=0.95의 원인 분해\n\n')
    f.write(f'- A↔B (순수 adv ↔ 순수 jpeg): **{sim_AB:.3f}**\n')
    f.write(f'- A↔C (순수 adv ↔ adv+jpeg): **{sim_AC:.3f}**\n')
    f.write(f'- B↔C (순수 jpeg ↔ adv+jpeg): **{sim_BC:.3f}**\n\n')
    if sim_BC > sim_AC:
        f.write('B↔C > A↔C → C의 signature는 **JPEG에 더 가까움** (JPEG contribution dominant)\n')
    else:
        f.write('A↔C > B↔C → C의 signature는 **adversarial에 더 가까움** (adversarial contribution dominant)\n')
    f.write('\n## 출력 파일\n\n')
    for fn in ['features.parquet', 'similarity_matrix.csv', 'tsne_decomposition.png',
               'feature_decomposition.csv']:
        f.write(f'- [{fn}](./{fn})\n')

print('\n완료 ✓')
