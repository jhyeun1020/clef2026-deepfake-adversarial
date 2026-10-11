"""
1002장 유클리디안 거리 + 신원 유사도 평가
- 영상 참조 임베딩: 캐시 저장/로드 (models/video_ref_embs.npy)
- 랜드마크: MediaPipe FaceMesh 478점
- 신원: InsightFace buffalo_s (CPUExecutionProvider, ctx_id=-1)
GPU 2 사용 (MediaPipe는 CPU)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json, sys, time
import numpy as np
import cv2, av
import torch
import mediapipe as mp
from pathlib import Path
from PIL import Image
import pandas as pd

BASE_DIR = Path('/home/student/GEN_AI_GTG')
IMAGES_DIR = BASE_DIR / 'app/input/res/Images'
VIDEO_DIR = BASE_DIR / 'datasets/Development Videos'
CONSTRAINTS_PATH = BASE_DIR / 'images/files/generation_constraints.json'
EMB_CACHE_PATH = BASE_DIR / 'models/video_ref_embs.npy'

with open(CONSTRAINTS_PATH) as f:
    constraints_all = json.load(f)

print(f'Constraints: {len(constraints_all)}키')
print(f'Images: {len(list(IMAGES_DIR.glob("*.png")))}장')

# ── InsightFace (채점 기준과 동일: buffalo_s, CPU, ctx_id=-1) ──
from insightface.app import FaceAnalysis
face_app = FaceAnalysis(name='buffalo_s', root=str(BASE_DIR / 'images/models'),
                        providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1)

# ── 영상 참조 임베딩 (cropped anchor 기반) ──
CROPPED_DIR = BASE_DIR / 'data/gold_training_set/cropped'

def get_anchor_emb(id_str):
    """cropped/{ID}/ 20장에서 평균 임베딩 추출 → anchor"""
    subdir = CROPPED_DIR / id_str
    if not subdir.exists(): return None
    embs = []
    for p in sorted(subdir.glob('*.png')):
        img = cv2.imread(str(p))
        if img is None: continue
        faces = face_app.get(img)
        if not faces: continue
        emb = faces[0].embedding
        norm = np.linalg.norm(emb)
        if norm > 0: embs.append(emb / norm)
    if not embs: return None
    mean = np.mean(embs, axis=0)
    return mean / np.linalg.norm(mean)

def extract_frames_pyav(video_path, n_sample=30):
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    total = stream.frames or 300
    step = max(1, total // n_sample)
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i % step == 0:
            frames.append(frame.to_ndarray(format='bgr24'))
        if len(frames) >= n_sample:
            break
    container.close()
    return frames

def get_video_ref_emb(id_str, anchor_emb, n_sample=30, sim_threshold=0.4):
    """anchor_emb와 가장 유사한 얼굴만 선택하여 영상 참조 임베딩 추출"""
    video_path = VIDEO_DIR / f"{id_str}_dev.mp4"
    if not video_path.exists(): return None
    frames = extract_frames_pyav(video_path, n_sample=n_sample)
    if not frames: return None
    matched_embs = []
    for frame in frames:
        faces = face_app.get(frame)
        if not faces: continue
        # anchor와 가장 유사한 얼굴 선택
        best_sim, best_emb = -1, None
        for face in faces:
            emb = face.embedding
            norm = np.linalg.norm(emb)
            if norm <= 0: continue
            emb = emb / norm
            sim = float(np.dot(emb, anchor_emb))
            if sim > best_sim:
                best_sim, best_emb = sim, emb
        # similarity threshold 적용
        if best_emb is not None and best_sim >= sim_threshold:
            matched_embs.append(best_emb)
    if not matched_embs: return None
    final_emb = np.mean(matched_embs, axis=0)
    final_emb = final_emb / np.linalg.norm(final_emb)
    return final_emb

if EMB_CACHE_PATH.exists():
    print(f'\n영상 임베딩 캐시 로드: {EMB_CACHE_PATH}')
    video_ref_embs = np.load(str(EMB_CACHE_PATH), allow_pickle=True).item()
    print(f'  {len(video_ref_embs)}개 ID 로드 완료')
else:
    print('\n영상 참조 임베딩 추출 중 (cropped anchor 기반, 최초 1회)...')
    all_ids = sorted(set(k.rsplit('_GT_', 1)[0] for k in constraints_all.keys()))
    video_ref_embs = {}
    for i, id_str in enumerate(all_ids):
        anchor = get_anchor_emb(id_str)
        if anchor is None:
            print(f'  [경고] {id_str}: anchor 추출 실패')
            continue
        emb = get_video_ref_emb(id_str, anchor)
        if emb is not None:
            video_ref_embs[id_str] = emb
        else:
            print(f'  [경고] {id_str}: 영상 임베딩 추출 실패')
        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(all_ids)}]', flush=True)
    np.save(str(EMB_CACHE_PATH), video_ref_embs)
    print(f'  캐시 저장: {EMB_CACHE_PATH}')
    print(f'  {len(video_ref_embs)}개 ID')

# ── 얼굴 임베딩 추출 ──
def get_face_embedding(pil_img):
    img_bgr = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_bgr)
    if not faces: return None
    emb = faces[0].embedding
    return emb / np.linalg.norm(emb)

def cosine_sim(e1, e2):
    if e1 is None or e2 is None: return -1.0
    return float(np.dot(e1, e2))

# ── 랜드마크 추출 ──
def extract_landmarks_pil(pil_img):
    img_np = np.array(pil_img.convert('RGB'))
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    ) as fm:
        res = fm.process(img_np)
    if not res.multi_face_landmarks:
        return None
    return [(lm.x, lm.y) for lm in res.multi_face_landmarks[0].landmark]

def euclidean_dist(gen_lm, ref_coords):
    if gen_lm is None or len(gen_lm) != len(ref_coords):
        return 0.5
    gen = np.array(gen_lm)
    ref = np.array(ref_coords)
    return float(np.mean(np.sqrt(np.sum((gen - ref) ** 2, axis=1))))

# ── 1002장 평가 ──
print(f'\n{"="*60}')
print('1002장 평가 시작')
print(f'{"="*60}')

t0 = time.time()
rows = []
missing = []
face_fail = []

keys = sorted(constraints_all.keys())
for i, key in enumerate(keys):
    id_str = key.rsplit('_GT_', 1)[0]
    img_path = IMAGES_DIR / f'{key}.png'

    if not img_path.exists():
        missing.append(key)
        continue

    img = Image.open(img_path).convert('RGB')
    facemesh_lm = constraints_all[key]['mediapipe_landmarks_facemesh']
    ref_emb = video_ref_embs.get(id_str)

    # 랜드마크 거리
    gen_lm = extract_landmarks_pil(img)
    lm_dist = euclidean_dist(gen_lm, facemesh_lm)

    # 신원 유사도
    gen_emb = get_face_embedding(img)
    id_sim = cosine_sim(gen_emb, ref_emb)

    if gen_emb is None:
        face_fail.append(key)

    rows.append({
        'key': key, 'id': id_str,
        'lm_dist': round(lm_dist, 6),
        'id_sim': round(id_sim, 4),
        'face_detected': gen_emb is not None,
    })

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        eta = (elapsed / (i+1)) * (len(keys) - i - 1)
        print(f'  [{i+1}/{len(keys)}] {elapsed:.0f}초 경과, ETA {eta/60:.0f}분', flush=True)

elapsed = time.time() - t0
df = pd.DataFrame(rows)

# ── 결과 저장 ──
OUT = BASE_DIR / 'outputs/eval_1002'
OUT.mkdir(exist_ok=True)
df.to_csv(OUT / 'metrics_1002.csv', index=False)

# ── 요약 ──
print(f'\n{"="*60}')
print('1002장 평가 결과')
print(f'{"="*60}')
print(f'평가: {len(df)}장 / {len(keys)}장')
print(f'누락: {len(missing)}장')
print(f'얼굴 미감지: {len(face_fail)}장')

valid = df[df['face_detected']]
print(f'\n── 신원 유사도 (id_sim) ──')
print(f'  평균: {valid["id_sim"].mean():.4f}')
print(f'  중앙값: {valid["id_sim"].median():.4f}')
print(f'  최소: {valid["id_sim"].min():.4f}')
print(f'  id_sim < 0: {(valid["id_sim"] < 0).sum()}장')

print(f'\n── 랜드마크 거리 (lm_dist) ──')
print(f'  평균: {df["lm_dist"].mean():.6f}')
print(f'  중앙값: {df["lm_dist"].median():.6f}')
print(f'  최대: {df["lm_dist"].max():.6f}')
print(f'  lm_dist > 0.01: {(df["lm_dist"] > 0.01).sum()}장')

# ID별 평균
id_avg = df.groupby('id').agg({'id_sim': 'mean', 'lm_dist': 'mean'}).reset_index()
worst_id = id_avg.sort_values('id_sim').head(10)
print(f'\n── 신원 유사도 최하위 10 ID ──')
print(worst_id.to_string(index=False))

worst_lm = id_avg.sort_values('lm_dist', ascending=False).head(10)
print(f'\n── 랜드마크 거리 최대 10 ID ──')
print(worst_lm.to_string(index=False))

if missing:
    print(f'\n── 누락 이미지 ──')
    for k in missing[:10]:
        print(f'  {k}')
    if len(missing) > 10:
        print(f'  ... 외 {len(missing)-10}장')

if face_fail:
    print(f'\n── 얼굴 미감지 ──')
    for k in face_fail[:10]:
        print(f'  {k}')

print(f'\n소요: {elapsed/60:.1f}분')
print(f'결과 저장: {OUT}/metrics_1002.csv')
print('완료 ✓')
