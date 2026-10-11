"""
Instability-based detection 검증.
가설: adversarial fake는 crop 전후 score가 크게 변함; clean(real/clean_fake)은 안정적.
이 instability 자체를 판정 feature로 사용 가능한가?

Note: crop_ablation은 8개 탐지기만 inference함 → 8 × 3 variant = 24차원
      (요구사항의 57차원 대신; 더 필요하면 추가 inference 후 재분석)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

BASE = Path('/home/student/GEN_AI_GTG')
SCORES = BASE / 'detection/crop_ablation/scores'
OUT = BASE / 'detection/instability'
OUT.mkdir(parents=True, exist_ok=True)

DETECTORS = [
    'npr', 'distildire', 'aide_genimage', 'xception',
    'siglip_dino', 'aide_progan', 'haywood', 'aide_sd14',
]
CROP_VARIANTS = ['face_crop_variable', 'random_crop_192_a', 'center_crop_192']
DATASETS = ['real_reference', 'clean_fake', 'adversarial_fake']

def load_scores(det, ds, variant):
    """parquet 로드 → numpy. filename으로 정렬해서 row order 통일."""
    df = pd.read_parquet(SCORES / f'{det}__{ds}__{variant}.parquet')
    df = df.sort_values('filename').reset_index(drop=True)
    return df['score'].to_numpy(dtype=np.float64), df['filename'].tolist()

# ── 1. 모든 (det, ds, variant) score 로드 ────────────────────────────
print("[1/5] score parquet 로드 중...")
TBL = {}        # (det, ds, v) -> np.array
FNS = {}        # (ds, v)      -> filename list
for det in DETECTORS:
    for ds in DATASETS:
        for v in ['full'] + CROP_VARIANTS:
            scores, fns = load_scores(det, ds, v)
            TBL[(det, ds, v)] = scores
            if (ds, v) not in FNS:
                FNS[(ds, v)] = fns

# ── filename 정렬이 dataset 내에서 일관된지 확인 ────────────────────
def stem_of(p):
    return Path(p).stem
for ds in DATASETS:
    base_full = [stem_of(p) for p in FNS[(ds, 'full')]]
    for v in CROP_VARIANTS:
        cur = [stem_of(p) for p in FNS[(ds, v)]]
        assert base_full == cur, f"{ds}/{v} stem 순서 불일치"
print(f"  로드 OK ({len(DETECTORS)} det × {len(DATASETS)} ds × 4 variant)\n")

def auroc(neg, pos):
    """neg=class 0 (clean), pos=class 1 (adv)."""
    if len(neg)==0 or len(pos)==0: return float('nan')
    n_neg, n_pos = len(neg), len(pos)
    ranks = stats.rankdata(np.concatenate([neg, pos]))
    return float((ranks[n_neg:].sum() - n_pos*(n_pos+1)/2) / (n_neg*n_pos))

# ── 2. Single-detector instability AUROC ─────────────────────────────
print("[2/5] Single-detector instability (real vs adv)...")
sd_rows = []
for det in DETECTORS:
    for v in CROP_VARIANTS:
        full_real = TBL[(det, 'real_reference', 'full')]
        crop_real = TBL[(det, 'real_reference', v)]
        full_adv  = TBL[(det, 'adversarial_fake', 'full')]
        crop_adv  = TBL[(det, 'adversarial_fake', v)]
        inst_real = np.abs(full_real - crop_real)
        inst_adv  = np.abs(full_adv - crop_adv)
        auc = auroc(inst_real, inst_adv)
        sd_rows.append({
            'detector': det, 'crop_variant': v,
            'auroc': round(auc, 4),
            'real_mean': round(float(inst_real.mean()), 4),
            'real_std':  round(float(inst_real.std()), 4),
            'adv_mean':  round(float(inst_adv.mean()), 4),
            'adv_std':   round(float(inst_adv.std()), 4),
            'separation': round(float(inst_adv.mean() - inst_real.mean()), 4),
        })
sd_df = pd.DataFrame(sd_rows).sort_values('auroc', ascending=False).reset_index(drop=True)
sd_df.to_csv(OUT / 'single_detector_auroc.csv', index=False)

print(f"\n  Top single-detector instability AUROC (real vs adv):")
print(f"  {'detector':<14}  {'variant':<22}  {'AUROC':>6}  {'real_μ':>8}  {'adv_μ':>8}  {'sep':>7}")
print("  " + "-"*78)
for _, r in sd_df.iterrows():
    print(f"  {r['detector']:<14}  {r['crop_variant']:<22}  {r['auroc']:>6.3f}  {r['real_mean']:>8.4f}  {r['adv_mean']:>8.4f}  {r['separation']:>+7.4f}")

# ── 3. Clean fake vs Adversarial fake instability ────────────────────
print("\n[3/5] Clean fake vs Adversarial fake instability...")
cv_rows = []
for det in DETECTORS:
    for v in CROP_VARIANTS:
        full_clean = TBL[(det, 'clean_fake', 'full')]
        crop_clean = TBL[(det, 'clean_fake', v)]
        full_adv   = TBL[(det, 'adversarial_fake', 'full')]
        crop_adv   = TBL[(det, 'adversarial_fake', v)]
        inst_clean = np.abs(full_clean - crop_clean)
        inst_adv   = np.abs(full_adv - crop_adv)
        auc = auroc(inst_clean, inst_adv)
        cv_rows.append({
            'detector': det, 'crop_variant': v,
            'auroc_clean_vs_adv': round(auc, 4),
            'clean_mean': round(float(inst_clean.mean()), 4),
            'clean_std':  round(float(inst_clean.std()), 4),
            'adv_mean':   round(float(inst_adv.mean()), 4),
            'adv_std':    round(float(inst_adv.std()), 4),
        })
cv_df = pd.DataFrame(cv_rows).sort_values('auroc_clean_vs_adv', ascending=False).reset_index(drop=True)
cv_df.to_csv(OUT / 'clean_vs_adv_instability.csv', index=False)

print(f"\n  AUROC (clean fake vs adversarial fake) by instability:")
print(f"  {'detector':<14}  {'variant':<22}  {'AUROC':>6}  {'clean_μ':>8}  {'adv_μ':>8}")
print("  " + "-"*72)
for _, r in cv_df.iterrows():
    print(f"  {r['detector']:<14}  {r['crop_variant']:<22}  {r['auroc_clean_vs_adv']:>6.3f}  {r['clean_mean']:>8.4f}  {r['adv_mean']:>8.4f}")

# ── 4. Multi-detector classification ─────────────────────────────────
print("\n[4/5] Multi-detector classification...")

def build_feature_matrix(ds, n_per_class):
    """Return (n_per_class, D) instability matrix where D = len(DETECTORS)*len(CROP_VARIANTS)."""
    n_full = TBL[(DETECTORS[0], ds, 'full')].shape[0]
    feats = []
    for det in DETECTORS:
        full = TBL[(det, ds, 'full')]
        for v in CROP_VARIANTS:
            crop = TBL[(det, ds, v)]
            feats.append(np.abs(full - crop))
    X = np.stack(feats, axis=1)  # (n, D)
    return X[:n_per_class]

X_real = build_feature_matrix('real_reference', 1000)
X_adv  = build_feature_matrix('adversarial_fake', 1002)
X_clean = build_feature_matrix('clean_fake', 1002)

print(f"  Real instability matrix : {X_real.shape}")
print(f"  Adv  instability matrix : {X_adv.shape}")
print(f"  Clean instability matrix: {X_clean.shape}")

X = np.vstack([X_real, X_adv])
y = np.array([0] * len(X_real) + [1] * len(X_adv))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

results = []
for clf_name, clf in [
    ('logistic_regression', LogisticRegression(max_iter=2000, random_state=42)),
    ('random_forest',       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
]:
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_te, proba)
    acc = accuracy_score(y_te, pred)
    print(f"  {clf_name:<22}  AUROC={auc:.4f}  Acc={acc:.4f}")
    results.append({'model': clf_name, 'test_auroc': round(auc,4),
                    'test_accuracy': round(acc,4),
                    'n_features': X.shape[1],
                    'n_train': len(X_tr), 'n_test': len(X_te)})

# extra: clean fake가 어떻게 분류되는지 (held-out OOD)
print("\n  [Held-out test] clean_fake (1002장) → 어느 클래스로 예측?")
for clf_name, clf in [
    ('logistic_regression', LogisticRegression(max_iter=2000, random_state=42).fit(X, y)),
    ('random_forest',       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X, y)),
]:
    proba_clean = clf.predict_proba(X_clean)[:, 1]
    pred_clean = (proba_clean >= 0.5).astype(int)
    print(f"    {clf_name:<22}  pred=adv: {int(pred_clean.sum())}/{len(pred_clean)} ({pred_clean.mean()*100:.1f}%)  mean(proba)={proba_clean.mean():.3f}")
    results.append({'model': f'{clf_name}__on_clean_fake',
                    'test_auroc': '',
                    'test_accuracy': '',
                    'n_features': X.shape[1],
                    'pred_adv_rate': round(float(pred_clean.mean()), 4),
                    'mean_proba_adv': round(float(proba_clean.mean()), 4)})

pd.DataFrame(results).to_csv(OUT / 'multi_detector_classification.csv', index=False)

# ── 5. Distribution 시각화 (top-3 single detector) ───────────────────
print("\n[5/5] 분포 시각화...")
top3 = sd_df.head(3).to_dict('records')
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
bins = np.linspace(0, 1, 41)
for ax, r in zip(axes, top3):
    det = r['detector']; v = r['crop_variant']
    full_real = TBL[(det, 'real_reference', 'full')]
    crop_real = TBL[(det, 'real_reference', v)]
    full_clean = TBL[(det, 'clean_fake', 'full')]
    crop_clean = TBL[(det, 'clean_fake', v)]
    full_adv  = TBL[(det, 'adversarial_fake', 'full')]
    crop_adv  = TBL[(det, 'adversarial_fake', v)]
    inst_real  = np.abs(full_real  - crop_real)
    inst_clean = np.abs(full_clean - crop_clean)
    inst_adv   = np.abs(full_adv   - crop_adv)
    ax.hist(inst_real,  bins=bins, density=True, color='#2ca02c', alpha=0.55, label=f'real (n={len(inst_real)})')
    ax.hist(inst_clean, bins=bins, density=True, color='#888888', alpha=0.55, label=f'clean fake (n={len(inst_clean)})')
    ax.hist(inst_adv,   bins=bins, density=True, color='#d62728', alpha=0.55, label=f'adv fake (n={len(inst_adv)})')
    ax.set_title(f"{det} × {v}\nAUROC(real vs adv) = {r['auroc']:.3f}", fontsize=10)
    ax.set_xlabel('|score(full) − score(crop)|')
    ax.set_ylabel('density')
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, loc='upper right')
plt.suptitle('Instability distributions: real vs clean fake vs adversarial fake (top-3 by AUROC)',
             fontsize=12)
plt.tight_layout()
plt.savefig(OUT / 'instability_distributions.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"  저장: instability_distributions.png\n")

# ── 종합 요약 ────────────────────────────────────────────────────────
print("="*80)
print("요약")
print("="*80)
print(f"  단일 탐지기 최고 AUROC : {sd_df.iloc[0]['auroc']:.3f}  ({sd_df.iloc[0]['detector']} × {sd_df.iloc[0]['crop_variant']})")
print(f"  Multi-detector ({X.shape[1]}차원) Logistic AUROC : {results[0]['test_auroc']:.4f}")
print(f"  Multi-detector ({X.shape[1]}차원) RandomForest AUROC : {results[1]['test_auroc']:.4f}")
print(f"\n  결과 저장: {OUT}/")
