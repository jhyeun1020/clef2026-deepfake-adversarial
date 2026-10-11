"""
False Positive 의심 케이스 분석
- 5 의심 + 5 adv baseline + 5 real baseline = 15장 비교
- Raw scores, RF feature importance, instability heatmap, 해석 리포트
"""
import csv, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

BASE = Path('/home/student/GEN_AI_GTG')
CROP_SCORES = BASE / 'detection/crop_ablation/scores'   # train (real_ref, adv_fake)
TEST_SCORES = BASE / 'detection/instability/test_application/scores'
TEST_DIR = BASE / 'detection/ImageCLEF2026-DeepFakeDetection-Tes/Data/Images_Detection'
PHASH_CSV = BASE / 'detection/phash_matching/fake_gt_matches.csv'
PRED_PQ = BASE / 'detection/instability/test_application/test_instability_predictions.parquet'
OUT = BASE / 'detection/instability/false_positive_analysis'
OUT.mkdir(parents=True, exist_ok=True)

DETECTORS = ['npr', 'distildire', 'aide_genimage', 'xception',
             'siglip_dino', 'aide_progan', 'haywood', 'aide_sd14']
VARIANTS_ALL = ['full', 'face_crop_variable', 'random_crop_192_a', 'center_crop_192']
CROP_VARIANTS = ['face_crop_variable', 'random_crop_192_a', 'center_crop_192']
FEAT_NAMES = [f'{d}__{v}' for d in DETECTORS for v in CROP_VARIANTS]  # 24 features

SUSPECT_STEMS = [
    '6488e0306ef3ca2f3c90', 'e819e3aad4a3226de7d0', 'ec634859158b8bcc121a',
    '1b91da9e460c6a7d0cdc', 'c3957f266d7078d9e959',
]

# ── 1. Test scores 로드 (6002장) ─────────────────────────────────────
print("[1/6] Test scores 로드...")
test_scores = {}  # (det, var) -> {stem: score}
for det in DETECTORS:
    for v in VARIANTS_ALL:
        df = pd.read_parquet(TEST_SCORES / f'{det}__{v}.parquet')
        test_scores[(det, v)] = {Path(r['filename']).stem: float(r['score'])
                                  for _, r in df.iterrows()}

# ── 2. Train scores 로드 (real_ref 1000 + adv 1002) ──────────────────
print("[2/6] Train scores 로드...")
train_scores = {}
for ds in ['real_reference', 'adversarial_fake']:
    for det in DETECTORS:
        for v in VARIANTS_ALL:
            df = pd.read_parquet(CROP_SCORES / f'{det}__{ds}__{v}.parquet').sort_values('filename')
            train_scores[(det, ds, v)] = df['score'].to_numpy(dtype=np.float64)

# ── 3. RF 재학습 (동일 파라미터) + feature importance ─────────────────
print("[3/6] RF 학습...")
def build_train_X(ds):
    feats = []
    for det in DETECTORS:
        full = train_scores[(det, ds, 'full')]
        for v in CROP_VARIANTS:
            sv = train_scores[(det, ds, v)]
            feats.append(np.abs(full - sv))
    return np.stack(feats, axis=1)

X_real = build_train_X('real_reference')
X_adv  = build_train_X('adversarial_fake')
X = np.vstack([X_real, X_adv])
y = np.array([0]*len(X_real) + [1]*len(X_adv))
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X, y)
feat_imp = clf.feature_importances_  # (24,)
print(f"  RF OK, feature importance sum = {feat_imp.sum():.4f}")

# ── 4. 대조군 선택 ────────────────────────────────────────────────────
print("[4/6] 대조군 선택...")
# 우리 1002 pHash
ours_set = set()
for r in csv.DictReader(open(PHASH_CSV)):
    ours_set.add(Path(r['test_image_path']).stem)

# 예측 결과에서 우리 1002 중 high-confidence adv 5장
pred_df = pd.read_parquet(PRED_PQ)
pred_df['stem'] = pred_df['filename'].apply(lambda x: Path(x).stem)
ours_pred = pred_df[pred_df['stem'].isin(ours_set)].sort_values('adv_prob_rf', ascending=False)
adv_baseline_stems = ours_pred.head(5)['stem'].tolist()

# Real (non-ours) low-confidence 5장
non_ours_pred = pred_df[~pred_df['stem'].isin(ours_set)]
real_baseline_stems = non_ours_pred.sort_values('adv_prob_rf').head(5)['stem'].tolist()

print(f"  의심: {SUSPECT_STEMS}")
print(f"  adv baseline: {adv_baseline_stems}")
print(f"  real baseline: {real_baseline_stems}")

ALL_STEMS = SUSPECT_STEMS + adv_baseline_stems + real_baseline_stems
LABELS = (['suspect']*5 + ['adv_baseline']*5 + ['real_baseline']*5)

# ── 5. Raw scores table ──────────────────────────────────────────────
print("[5/6] Raw scores + instability 추출...")
raw_rows = []
for stem in ALL_STEMS:
    for det in DETECTORS:
        full_s = test_scores[(det, 'full')].get(stem, float('nan'))
        for v in VARIANTS_ALL:
            s = test_scores[(det, v)].get(stem, float('nan'))
            inst = abs(full_s - s) if v != 'full' else 0.0
            raw_rows.append({
                'image_id': stem, 'detector': det, 'variant': v,
                'raw_score': round(s, 4),
                'instability_vs_full': round(inst, 4),
            })
pd.DataFrame(raw_rows).to_csv(OUT / 'raw_scores_table.csv', index=False)

# ── 6. Instability feature vectors (15장) + RF breakdown ─────────────
feat_matrix = []  # (15, 24)
for stem in ALL_STEMS:
    vec = []
    for det in DETECTORS:
        full_s = test_scores[(det, 'full')].get(stem, 0.0)
        for v in CROP_VARIANTS:
            sv = test_scores[(det, v)].get(stem, 0.0)
            vec.append(abs(full_s - sv))
    feat_matrix.append(vec)
feat_matrix = np.array(feat_matrix)  # (15, 24)

# RF decision breakdown
print("[6/6] RF decision breakdown + heatmap + report...")
breakdown_rows = []
for i, stem in enumerate(ALL_STEMS):
    vec = feat_matrix[i]
    prob = clf.predict_proba(vec.reshape(1, -1))[0, 1]
    # local contribution 근사: feature_value × global_importance
    contributions = vec * feat_imp
    ranked = np.argsort(-contributions)
    for rank, j in enumerate(ranked[:5]):
        # percentile of this feature value in train adv/real distributions
        adv_vals = X_adv[:, j]
        real_vals = X_real[:, j]
        pct_adv = stats.percentileofscore(adv_vals, vec[j])
        pct_real = stats.percentileofscore(real_vals, vec[j])
        breakdown_rows.append({
            'image_id': stem,
            'group': LABELS[i],
            'adv_prob': round(prob, 4),
            'rank': rank + 1,
            'feature_name': FEAT_NAMES[j],
            'feature_value': round(float(vec[j]), 4),
            'global_importance': round(float(feat_imp[j]), 4),
            'local_contribution': round(float(contributions[j]), 4),
            'percentile_in_adv': round(pct_adv, 1),
            'percentile_in_real': round(pct_real, 1),
        })
pd.DataFrame(breakdown_rows).to_csv(OUT / 'rf_decision_breakdown.csv', index=False)

# comparison patterns (15 × 24)
comp_rows = []
for i, stem in enumerate(ALL_STEMS):
    row = {'image_id': stem, 'group': LABELS[i]}
    for j, fn in enumerate(FEAT_NAMES):
        row[fn] = round(float(feat_matrix[i, j]), 4)
    comp_rows.append(row)
pd.DataFrame(comp_rows).to_csv(OUT / 'comparison_patterns.csv', index=False)

# ── Heatmap ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(feat_matrix, aspect='auto', cmap='YlOrRd', vmin=0,
               vmax=max(0.5, float(feat_matrix.max())))
ax.set_yticks(range(15))
ylabels = []
for i, (stem, lbl) in enumerate(zip(ALL_STEMS, LABELS)):
    prob = clf.predict_proba(feat_matrix[i].reshape(1,-1))[0, 1]
    ylabels.append(f'{stem[:12]}  [{lbl}]  p={prob:.2f}')
ax.set_yticklabels(ylabels, fontsize=8, family='monospace')
ax.set_xticks(range(24))
ax.set_xticklabels(FEAT_NAMES, rotation=45, ha='right', fontsize=7)
for i in range(15):
    for j in range(24):
        v = feat_matrix[i, j]
        if v > 0.01:
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5.5,
                    color='white' if v > 0.3 else 'black')
# 그룹 구분선
ax.axhline(4.5, color='cyan', linewidth=2, linestyle='--')
ax.axhline(9.5, color='cyan', linewidth=2, linestyle='--')
ax.set_title('Instability heatmap: 5 suspect + 5 adv baseline + 5 real baseline', fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
plt.tight_layout()
plt.savefig(OUT / 'instability_heatmap.png', dpi=130, bbox_inches='tight')
plt.close()

# ── 이미지 비교 grid ────────────────────────────────────────────────
from PIL import Image, ImageDraw, ImageFont
cell = 200; pad = 4; label_h = 36
cols = 5; rows = 3
W = cols * (cell + pad) + pad
H = rows * (cell + label_h + pad) + pad + 24
grid = Image.new('RGB', (W, H), (25, 25, 25))
draw = ImageDraw.Draw(grid)
try:
    fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
except Exception:
    fnt = ImageFont.load_default()

titles = ['SUSPECT (FP?)', 'ADV BASELINE (true adv)', 'REAL BASELINE (true clean)']
for row_i, (group_stems, title) in enumerate(zip(
    [SUSPECT_STEMS, adv_baseline_stems, real_baseline_stems], titles)):
    for col_j, stem in enumerate(group_stems):
        x = pad + col_j * (cell + pad)
        y = 24 + pad + row_i * (cell + label_h + pad)
        try:
            im = Image.open(TEST_DIR / f'{stem}.png').convert('RGB').resize((cell, cell))
        except Exception:
            im = Image.new('RGB', (cell, cell), (60, 0, 0))
        grid.paste(im, (x, y))
        prob = clf.predict_proba(feat_matrix[row_i*5 + col_j].reshape(1,-1))[0, 1]
        # top feature
        vec = feat_matrix[row_i*5 + col_j]
        top_idx = np.argmax(vec * feat_imp)
        top_fn = FEAT_NAMES[top_idx]
        draw.text((x, y + cell + 2),
                  f'{stem[:10]} p={prob:.2f}\ntop: {top_fn}={vec[top_idx]:.2f}',
                  fill=(220, 220, 220), font=fnt)
    draw.text((pad, 24 + pad + row_i * (cell + label_h + pad) - 12),
              title, fill=(255, 220, 100), font=fnt)
grid.save(OUT / 'comparison_grid.png')

# ── 해석 리포트 ──────────────────────────────────────────────────────
with open(OUT / 'FP_analysis.md', 'w') as f:
    f.write('# False Positive Analysis — 5 suspect images\n\n')
    f.write('## 대상 이미지\n\n')
    f.write('| # | image_id | adv_prob (RF) | top feature | top value | %ile in adv | %ile in real |\n')
    f.write('|---|---|---|---|---|---|---|\n')
    bd_df = pd.DataFrame(breakdown_rows)
    for i, stem in enumerate(SUSPECT_STEMS):
        top = bd_df[(bd_df['image_id']==stem) & (bd_df['rank']==1)].iloc[0]
        f.write(f'| {i+1} | {stem} | {top["adv_prob"]:.3f} | {top["feature_name"]} | '
                f'{top["feature_value"]:.4f} | {top["percentile_in_adv"]:.0f}% | {top["percentile_in_real"]:.0f}% |\n')

    f.write('\n## 대조군 — adv baseline (우리 1002 high-prob)\n\n')
    f.write('| # | image_id | adv_prob | top feature | top value |\n|---|---|---|---|---|\n')
    for i, stem in enumerate(adv_baseline_stems):
        top = bd_df[(bd_df['image_id']==stem) & (bd_df['rank']==1)].iloc[0]
        f.write(f'| {i+1} | {stem} | {top["adv_prob"]:.3f} | {top["feature_name"]} | {top["feature_value"]:.4f} |\n')

    f.write('\n## 대조군 — real baseline (low-prob)\n\n')
    f.write('| # | image_id | adv_prob | top feature | top value |\n|---|---|---|---|---|\n')
    for i, stem in enumerate(real_baseline_stems):
        top = bd_df[(bd_df['image_id']==stem) & (bd_df['rank']==1)].iloc[0]
        f.write(f'| {i+1} | {stem} | {top["adv_prob"]:.3f} | {top["feature_name"]} | {top["feature_value"]:.4f} |\n')

    # 패턴 분석
    suspect_mean = feat_matrix[:5].mean(axis=0)
    adv_mean     = feat_matrix[5:10].mean(axis=0)
    real_mean    = feat_matrix[10:15].mean(axis=0)
    cos_sus_adv  = float(np.dot(suspect_mean, adv_mean) /
                         (np.linalg.norm(suspect_mean)*np.linalg.norm(adv_mean) + 1e-12))
    cos_sus_real = float(np.dot(suspect_mean, real_mean) /
                         (np.linalg.norm(suspect_mean)*np.linalg.norm(real_mean) + 1e-12))
    cos_adv_real = float(np.dot(adv_mean, real_mean) /
                         (np.linalg.norm(adv_mean)*np.linalg.norm(real_mean) + 1e-12))

    f.write('\n## 패턴 유사도 (평균 instability vector cosine similarity)\n\n')
    f.write(f'- suspect ↔ adv_baseline : **{cos_sus_adv:.3f}**\n')
    f.write(f'- suspect ↔ real_baseline: **{cos_sus_real:.3f}**\n')
    f.write(f'- adv ↔ real             : {cos_adv_real:.3f}\n\n')

    if cos_sus_adv > 0.7 and cos_sus_adv > cos_sus_real + 0.2:
        verdict = 'A'
        f.write('### 판정: **Case A — 의심 이미지들의 instability 패턴이 adv baseline과 유사**\n\n')
        f.write('시각적 인상과 달리, 이 이미지들은 adversarial 공격을 받았을 가능성이 높습니다.\n')
        f.write('모델이 visually subtle한 adversarial perturbation을 정확하게 포착한 것으로 보입니다.\n')
    elif cos_sus_real > cos_sus_adv:
        verdict = 'B'
        f.write('### 판정: **Case B — 의심 이미지들이 real baseline과 더 유사하나 판정이 반대**\n\n')
        f.write('모델이 일반 perturbation (noise, blur)을 adversarial로 오판하고 있을 가능성.\n')
        f.write('Threshold 상향 또는 학습 데이터 다양화 필요.\n')
    else:
        verdict = 'C'
        f.write('### 판정: **Case C — 혼합 패턴**\n\n')
        f.write('의심 이미지들이 adv/real 어느 쪽과도 완전히 일치하지 않는 고유 패턴.\n')
        f.write('새로운 perturbation type일 가능성. 개별 케이스 분석 필요.\n')

    # 개별 이미지 해석
    f.write('\n## 개별 이미지 해석\n\n')
    for i, stem in enumerate(SUSPECT_STEMS):
        top5 = bd_df[(bd_df['image_id']==stem)].head(5)
        vec_i = feat_matrix[i]
        # adv 분포에서의 위치
        prob_i = clf.predict_proba(vec_i.reshape(1,-1))[0, 1]
        cos_vs_adv = float(np.dot(vec_i, adv_mean) /
                           (np.linalg.norm(vec_i)*np.linalg.norm(adv_mean)+1e-12))
        cos_vs_real = float(np.dot(vec_i, real_mean) /
                            (np.linalg.norm(vec_i)*np.linalg.norm(real_mean)+1e-12))
        if cos_vs_adv > cos_vs_real + 0.15:
            local_verdict = 'adversarial-like'
        elif cos_vs_real > cos_vs_adv:
            local_verdict = 'real-like but noisy'
        else:
            local_verdict = 'ambiguous'

        f.write(f'### {i+1}. `{stem}`\n\n')
        f.write(f'- adv_prob: {prob_i:.3f}\n')
        f.write(f'- cos(vs adv): {cos_vs_adv:.3f} | cos(vs real): {cos_vs_real:.3f}\n')
        f.write(f'- **판정: {local_verdict}**\n')
        f.write(f'- Top features:\n\n')
        f.write('| rank | feature | value | %ile(adv) | %ile(real) |\n|---|---|---|---|---|\n')
        for _, tr in top5.iterrows():
            f.write(f'| {int(tr["rank"])} | {tr["feature_name"]} | {tr["feature_value"]:.4f} | '
                    f'{tr["percentile_in_adv"]:.0f}% | {tr["percentile_in_real"]:.0f}% |\n')
        f.write('\n')

    f.write('## 출력 파일\n\n')
    f.write('- [raw_scores_table.csv](./raw_scores_table.csv)\n')
    f.write('- [rf_decision_breakdown.csv](./rf_decision_breakdown.csv)\n')
    f.write('- [comparison_patterns.csv](./comparison_patterns.csv)\n')
    f.write('- [instability_heatmap.png](./instability_heatmap.png)\n')
    f.write('- [comparison_grid.png](./comparison_grid.png)\n')

print(f"\n패턴 유사도:")
print(f"  suspect ↔ adv  : {cos_sus_adv:.3f}")
print(f"  suspect ↔ real : {cos_sus_real:.3f}")
print(f"  adv ↔ real     : {cos_adv_real:.3f}")
print(f"\n판정: Case {verdict}")
print(f"\n출력: {OUT}/")
print("완료 ✓")
