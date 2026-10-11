"""
6개 시드(multi_seed 5개 + first_sub)에서 ID_frame별 최고 id_sim 이미지 선별
출력: outputs/best_idsim/
"""
import os, json, cv2, shutil
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from insightface.app import FaceAnalysis

BASE_DIR = Path('/home/student/GEN_AI_GTG')

# 소스 디렉토리
SOURCES = {
    'seed_2024': BASE_DIR / 'outputs/multi_seed/seed_2024',
    'seed_314':  BASE_DIR / 'outputs/multi_seed/seed_314',
    'seed_5555': BASE_DIR / 'outputs/multi_seed/seed_5555',
    'seed_7777': BASE_DIR / 'outputs/multi_seed/seed_7777',
    'seed_999':  BASE_DIR / 'outputs/multi_seed/seed_999',
    'first_sub': BASE_DIR / 'submit/first_sub/Images',
}

OUT_DIR = BASE_DIR / 'outputs/best_idsim'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# constraints (for lm_dist)
with open(BASE_DIR / 'images/files/generation_constraints.json') as f:
    constraints_all = json.load(f)

all_keys = sorted(constraints_all.keys())
print(f"총 {len(all_keys)}개 키")

# InsightFace
face_app = FaceAnalysis(name='buffalo_s', root=str(BASE_DIR / 'images/models'), providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1)
video_ref_embs = np.load(str(BASE_DIR / 'models/video_ref_embs.npy'), allow_pickle=True).item()

# MediaPipe
def extract_lm(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        res = fm.process(img_rgb)
    if not res.multi_face_landmarks:
        return None
    return [(l.x, l.y) for l in res.multi_face_landmarks[0].landmark]

def euc_dist(g, r):
    if g is None or len(g) != len(r):
        return 0.5
    return float(np.mean(np.sqrt(np.sum((np.array(g) - np.array(r))**2, axis=1))))

def get_id_sim(img_bgr, id_str):
    ref_emb = video_ref_embs.get(id_str)
    if ref_emb is None:
        return -1.0
    faces = face_app.get(img_bgr)
    if not faces:
        return -1.0
    emb = faces[0].embedding / np.linalg.norm(faces[0].embedding)
    return float(np.dot(emb, ref_emb))

# 선별
rows = []
for ki, key in enumerate(all_keys):
    id_str = key.rsplit('_GT_', 1)[0]
    ref_lm = constraints_all[key]['mediapipe_landmarks_facemesh']

    best_sim = -2.0
    best_src = None
    best_bgr = None

    for src_name, src_dir in SOURCES.items():
        img_path = src_dir / f'{key}.png'
        if not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        # 256x256로 통일
        if img_bgr.shape[0] != 256 or img_bgr.shape[1] != 256:
            img_bgr = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)

        sim = get_id_sim(img_bgr, id_str)
        if sim > best_sim:
            best_sim = sim
            best_src = src_name
            best_bgr = img_bgr

    # lm_dist for best
    if best_bgr is not None:
        gen_lm = extract_lm(best_bgr)
        lm_dist = euc_dist(gen_lm, ref_lm)
        # 저장
        cv2.imwrite(str(OUT_DIR / f'{key}.png'), best_bgr)
    else:
        lm_dist = 0.5

    rows.append({
        'key': key, 'id': id_str,
        'id_sim': round(best_sim, 4),
        'lm_dist': round(lm_dist, 6),
        'source': best_src,
    })

    if (ki + 1) % 100 == 0:
        print(f"  [{ki+1}/{len(all_keys)}]", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / 'best_idsim_results.csv', index=False)

# 통계
print(f"\n{'='*60}")
print(f"선별 결과: {len(df)}장 → {OUT_DIR}")
print(f"{'='*60}")
print(f"  id_sim  평균: {df['id_sim'].mean():.4f}, 중앙값: {df['id_sim'].median():.4f}")
print(f"  lm_dist 평균: {df['lm_dist'].mean():.6f}, 중앙값: {df['lm_dist'].median():.6f}")

# 소스별 분포
print(f"\n소스별 선택 수:")
for src in SOURCES:
    n = (df['source'] == src).sum()
    print(f"  {src:12s}: {n}장 ({n/len(df)*100:.1f}%)")

# 이상치: id_sim < 0.3 또는 lm_dist > 0.02
print(f"\n이상치 (id_sim < 0.3):")
outliers_sim = df[df['id_sim'] < 0.3].sort_values('id_sim')
if len(outliers_sim) == 0:
    print("  없음")
else:
    for _, r in outliers_sim.iterrows():
        print(f"  {r['key']:25s} sim={r['id_sim']:.4f} src={r['source']}")

print(f"\n이상치 (lm_dist > 0.02):")
outliers_lm = df[df['lm_dist'] > 0.02].sort_values('lm_dist', ascending=False)
if len(outliers_lm) == 0:
    print("  없음")
else:
    for _, r in outliers_lm.iterrows():
        print(f"  {r['key']:25s} dist={r['lm_dist']:.6f} src={r['source']}")

# face not detected
no_face = df[df['id_sim'] < 0]
if len(no_face) > 0:
    print(f"\n얼굴 미감지: {len(no_face)}장")
    for _, r in no_face.iterrows():
        print(f"  {r['key']:25s} src={r['source']}")

print(f"\nCSV: {OUT_DIR}/best_idsim_results.csv")
print("완료 ✓")
