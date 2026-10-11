"""
Step 2: 6,000장 샘플링 + 4 variant 생성
- 우리 1002장 (pHash 매칭) — 필수 포함
- face-detected 풀에서 random 4,500장 (우리 1002 제외)
- face-failed 풀에서 random 500장
- 4 variants: full, face_crop_variable (center fallback), random_crop_192_a, center_crop_192
"""
import csv, pickle, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

BASE = Path('/home/student/GEN_AI_GTG')
TEST_DIR = BASE / 'detection/ImageCLEF2026-DeepFakeDetection-Tes/Data/Images_Detection'
MATCHES_CSV = BASE / 'detection/phash_matching/fake_gt_matches.csv'
APP = BASE / 'detection/instability/test_application'
LM_CACHE = APP / 'test_face_landmarks.pkl'

VAR_BASE = APP / 'variants'
META_CSV = APP / 'sample_metadata.csv'

VARIANTS = ['full', 'face_crop_variable', 'random_crop_192_a', 'center_crop_192']
for v in VARIANTS:
    (VAR_BASE / v).mkdir(parents=True, exist_ok=True)

# 1. landmarks 캐시 로드
with open(LM_CACHE, 'rb') as f:
    landmarks_cache = pickle.load(f)
print(f'landmarks 로드: {len(landmarks_cache)}')

# 2. 우리 1002장 식별
ours_set = set()
for r in csv.DictReader(open(MATCHES_CSV)):
    ours_set.add(Path(r['test_image_path']).name)
print(f'우리 1002장: {len(ours_set)}')

# 3. 풀 분리
all_files = sorted(landmarks_cache.keys())
detected = [n for n in all_files if landmarks_cache[n] is not None]
failed   = [n for n in all_files if landmarks_cache[n] is None]
detected_set = set(detected); failed_set = set(failed)
print(f'detected: {len(detected)} / failed: {len(failed)}')
print(f'  ours in detected: {len(ours_set & detected_set)}')
print(f'  ours in failed:   {len(ours_set & failed_set)}')

# 4. 샘플링
rng = np.random.RandomState(2026)
ours_list = sorted(ours_set)

# 우리 1002 제외하고 detected에서 4500
detected_pool = [n for n in detected if n not in ours_set]
random_4500 = rng.choice(detected_pool, size=4500, replace=False).tolist()

# failed에서 500 (우리 1002 제외)
failed_pool = [n for n in failed if n not in ours_set]
random_500_failed = rng.choice(failed_pool, size=500, replace=False).tolist()

selected = sorted(set(ours_list) | set(random_4500) | set(random_500_failed))
print(f'\n총 선택: {len(selected)}')
print(f'  ours      : {len(ours_set & set(selected))}')
print(f'  random4500: {len(set(random_4500) & set(selected))}')
print(f'  failed500 : {len(set(random_500_failed) & set(selected))}')

# 메타: per image category
def cat_of(name):
    cats = []
    if name in ours_set: cats.append('ours_1002')
    if name in set(random_4500): cats.append('random_4500')
    if name in set(random_500_failed): cats.append('failed_500')
    return ','.join(cats)

# 5. random crop offsets (per-image, seed=42 고정 — crop_ablation과 동일 분포)
SZ = 192; IMG = 256; max_o = IMG - SZ  # 64
def random_offset(idx):
    r = np.random.RandomState(42 + idx)  # per-image deterministic
    return int(r.randint(0, max_o + 1)), int(r.randint(0, max_o + 1))

# 6. variant 생성
def gen_variants(img, lm, idx):
    out = {}
    h, w = img.shape[:2]  # 256, 256
    out['full'] = img

    # face_crop_variable: 얼굴 검출 성공 → 가변 crop, 실패 → center_192 fallback
    used_fb = False
    if lm:
        xs = [l[0]*w for l in lm]; ys = [l[1]*h for l in lm]
        x_min, x_max = min(xs), max(xs); y_min, y_max = min(ys), max(ys)
        fw, fh = x_max-x_min, y_max-y_min
        cx, cy = (x_min+x_max)/2, (y_min+y_max)/2
        size = max(fw, fh) * 1.3
        x1 = max(0, int(cx - size/2)); y1 = max(0, int(cy - size/2))
        x2 = min(w, int(cx + size/2)); y2 = min(h, int(cy + size/2))
        out['face_crop_variable'] = img[y1:y2, x1:x2]
        if out['face_crop_variable'].size == 0:
            o = (256-192)//2
            out['face_crop_variable'] = img[o:o+192, o:o+192]
            used_fb = True
    else:
        o = (256-192)//2
        out['face_crop_variable'] = img[o:o+192, o:o+192]
        used_fb = True

    # center_crop_192
    o = (256-192)//2
    out['center_crop_192'] = img[o:o+192, o:o+192]

    # random_crop_192_a
    ox, oy = random_offset(idx)
    out['random_crop_192_a'] = img[oy:oy+192, ox:ox+192]

    return out, used_fb

print('\nvariants 생성 중...')
meta_rows = []
t0 = time.time()
for i, name in enumerate(tqdm(selected, desc='variants', unit='img')):
    src = TEST_DIR / name
    img = cv2.imread(str(src))
    if img is None:
        continue
    if img.shape[:2] != (256, 256):
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    lm = landmarks_cache.get(name)
    variants, used_fb = gen_variants(img, lm, i)
    stem = Path(name).stem
    for v_name, v_img in variants.items():
        out_p = VAR_BASE / v_name / f'{stem}.png'
        if not out_p.exists():
            cv2.imwrite(str(out_p), v_img)
    meta_rows.append({
        'filename': name, 'stem': stem,
        'category': cat_of(name),
        'face_detected': lm is not None,
        'used_fallback_face_crop': used_fb,
    })

with open(META_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
    w.writeheader(); w.writerows(meta_rows)
print(f'\nmetadata 저장: {META_CSV} ({len(meta_rows)} rows)')
print(f'경과: {time.time()-t0:.1f}s')

print('\n각 variant 디렉토리 카운트:')
for v in VARIANTS:
    n = len(list((VAR_BASE / v).glob('*.png')))
    print(f'  {v:24s} : {n}')
