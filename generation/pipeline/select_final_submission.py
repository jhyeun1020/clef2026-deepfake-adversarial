"""
7개 소스에서 키별 최고 id_sim 이미지 선별 (최종 1002장)
- P-S: ps_6seeds/seed_{42,2024,7777,314,999,5555}
- Basic: best_idsim (5시드 중 최고 선별 완료)
"""
import json, cv2, numpy as np, pandas as pd, shutil
import mediapipe as mp
from pathlib import Path
from insightface.app import FaceAnalysis

BASE = Path('/home/student/GEN_AI_GTG')

SOURCES = {
    'PS_seed42':   BASE / 'outputs/ps_6seeds/seed_42',
    'PS_seed2024': BASE / 'outputs/ps_6seeds/seed_2024',
    'PS_seed7777': BASE / 'outputs/ps_6seeds/seed_7777',
    'PS_seed314':  BASE / 'outputs/ps_6seeds/seed_314',
    'PS_seed999':  BASE / 'outputs/ps_6seeds/seed_999',
    'PS_seed5555': BASE / 'outputs/ps_6seeds/seed_5555',
    'Basic_best':  BASE / 'outputs/best_idsim',
}

OUT_DIR = BASE / 'outputs/final_1002'
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE / 'images/files/generation_constraints.json') as f:
    constraints = json.load(f)

all_keys = sorted(constraints.keys())
print(f"총 {len(all_keys)}개 키, {len(SOURCES)}개 소스")

# InsightFace
face_app = FaceAnalysis(name='buffalo_s', root=str(BASE/'images/models'), providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1)
ref_embs = np.load(str(BASE/'models/video_ref_embs.npy'), allow_pickle=True).item()

def get_id_sim(img_bgr, id_str):
    ref = ref_embs.get(id_str)
    if ref is None: return -1.0
    faces = face_app.get(img_bgr)
    if not faces: return -1.0
    emb = faces[0].embedding / np.linalg.norm(faces[0].embedding)
    return float(np.dot(emb, ref))

def extract_lm(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        res = fm.process(rgb)
    if not res.multi_face_landmarks: return None
    return [(l.x, l.y) for l in res.multi_face_landmarks[0].landmark]

def euc_dist(g, r):
    if g is None or len(g) != len(r): return 0.5
    return float(np.mean(np.sqrt(np.sum((np.array(g) - np.array(r))**2, axis=1))))

rows = []
for ki, key in enumerate(all_keys):
    id_str = key.rsplit('_GT_', 1)[0]
    ref_lm = constraints[key]['mediapipe_landmarks_facemesh']

    best_sim = -2.0
    best_src = None
    best_path = None

    for src_name, src_dir in SOURCES.items():
        img_path = src_dir / f'{key}.png'
        if not img_path.exists(): continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        if img_bgr.shape[0] != 256 or img_bgr.shape[1] != 256:
            img_bgr = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
        sim = get_id_sim(img_bgr, id_str)
        if sim > best_sim:
            best_sim = sim
            best_src = src_name
            best_path = img_path

    # 선택된 이미지 복사 + lm_dist 계산
    if best_path is not None:
        img_bgr = cv2.imread(str(best_path))
        if img_bgr.shape[0] != 256 or img_bgr.shape[1] != 256:
            img_bgr = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT_DIR / f'{key}.png'), img_bgr)
        gen_lm = extract_lm(img_bgr)
        lm_dist = euc_dist(gen_lm, ref_lm)
    else:
        lm_dist = 0.5

    # 프롬프트 타입과 시드 분리
    if best_src and best_src.startswith('PS_'):
        prompt_type = 'P-S'
        seed = best_src.replace('PS_seed', '')
    else:
        prompt_type = 'Basic'
        seed = 'best5'

    rows.append({
        'key': key, 'id': id_str,
        'id_sim': round(best_sim, 4),
        'lm_dist': round(lm_dist, 6),
        'source': best_src,
        'prompt': prompt_type,
        'seed': seed,
    })

    if (ki + 1) % 100 == 0:
        print(f"  [{ki+1}/{len(all_keys)}]", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / 'final_1002_results.csv', index=False)

# 통계
print(f"\n{'='*60}")
print(f"최종 선별: {len(df)}장 → {OUT_DIR}")
print(f"{'='*60}")
print(f"  id_sim  평균: {df['id_sim'].mean():.4f}, 중앙값: {df['id_sim'].median():.4f}")
print(f"  lm_dist 평균: {df['lm_dist'].mean():.6f}, 중앙값: {df['lm_dist'].median():.6f}")

# 프롬프트별
print(f"\n프롬프트별:")
for pt in ['P-S', 'Basic']:
    sub = df[df['prompt'] == pt]
    print(f"  {pt:6s}: {len(sub):4d}장 ({len(sub)/len(df)*100:.1f}%)  id_sim={sub['id_sim'].mean():.4f}  lm_dist={sub['lm_dist'].mean():.6f}")

# 소스별
print(f"\n소스별:")
for src in SOURCES:
    n = (df['source'] == src).sum()
    if n > 0:
        sub = df[df['source'] == src]
        print(f"  {src:15s}: {n:4d}장 ({n/len(df)*100:.1f}%)  id_sim={sub['id_sim'].mean():.4f}")

# 이상치
print(f"\n이상치 (id_sim < 0.3):")
outliers = df[df['id_sim'] < 0.3].sort_values('id_sim')
if len(outliers) == 0:
    print("  없음")
else:
    for _, r in outliers.iterrows():
        print(f"  {r['key']:25s} sim={r['id_sim']:.4f} src={r['source']}")

print(f"\n이상치 (lm_dist > 0.02):")
outliers_lm = df[df['lm_dist'] > 0.02].sort_values('lm_dist', ascending=False)
if len(outliers_lm) == 0:
    print("  없음")
else:
    for _, r in outliers_lm.iterrows():
        print(f"  {r['key']:25s} dist={r['lm_dist']:.6f} src={r['source']}")

no_face = df[df['id_sim'] < 0]
if len(no_face) > 0:
    print(f"\n얼굴 미감지: {len(no_face)}장")
    for _, r in no_face.iterrows():
        print(f"  {r['key']:25s} src={r['source']}")

print(f"\nCSV: {OUT_DIR}/final_1002_results.csv")
print("완료 ✓")
