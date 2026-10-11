"""
Step 1: 500장 준비 + 4 crop variants 생성
A: Pure PGD (raw, no JPEG) — 100장
B: Pure JPEG Q80 (real + Q80) — 100장
C: Current adversarial (PGD + JPEG Q80) — 100장 (same IDs as A)
D_clean_fake: Clean fake (no attack) — 100장 (same IDs as A)
D_real: Real reference — 100장
"""
import csv
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm

BASE = Path('/home/student/GEN_AI_GTG')
RAW_ADV_DIR   = BASE / 'generation/outputs/attack_1002_all_nonface/raw_png'
ADV_JPEG_DIR  = BASE / 'generation/submit/third_sub/256'
CLEAN_DIR     = BASE / 'generation/outputs/final_1002'
REAL_DIR      = BASE / 'detection/real_reference/center'
OUT = BASE / 'detection/adv_vs_jpeg_decomposition'
VAR_BASE = OUT / 'variants'

VARIANTS = ['full', 'face_crop_variable', 'center_crop_192', 'random_crop_192_a']

rng = np.random.RandomState(42)

# A/C/D_clean: same 100 IDs from clean fake
all_stems = sorted(p.stem for p in CLEAN_DIR.glob('*.png'))
sel_idx = rng.choice(len(all_stems), 100, replace=False)
keys_100 = [all_stems[i] for i in sorted(sel_idx)]

# B/D_real: 100 real images
real_all = sorted(REAL_DIR.glob('*.png'))
real_idx = rng.choice(len(real_all), 200, replace=False)
real_for_jpeg = [real_all[i] for i in real_idx[:100]]  # B
real_for_baseline = [real_all[i] for i in real_idx[100:200]]  # D_real

# Build image list: (uid, version, src_path_or_img)
images = []
for key in keys_100:
    images.append((f'A_{key}', 'A_pure_adv', RAW_ADV_DIR / f'{key}.png', None))
    images.append((f'C_{key}', 'C_adv_jpeg', ADV_JPEG_DIR / f'{key}.png', None))
    images.append((f'Dcf_{key}', 'D_clean_fake', CLEAN_DIR / f'{key}.png', None))

for p in real_for_jpeg:
    images.append((f'B_{p.stem}', 'B_pure_jpeg', p, 'jpeg_q80'))

for p in real_for_baseline:
    images.append((f'Dr_{p.stem}', 'D_real', p, None))

print(f'총 {len(images)}장')
versions = {}
for uid, ver, _, _ in images:
    versions.setdefault(ver, []).append(uid)
for v, uids in versions.items():
    print(f'  {v}: {len(uids)}')

# Create variant dirs
for uid, _, _, _ in images:
    for v in VARIANTS:
        (VAR_BASE / v).mkdir(parents=True, exist_ok=True)

# Face detection + crop
fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def gen_crops(img, idx):
    h, w = 256, 256
    out = {'full': img.copy()}
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if res.multi_face_landmarks:
        lm = [(l.x, l.y) for l in res.multi_face_landmarks[0].landmark]
        xs = [l[0]*w for l in lm]; ys = [l[1]*h for l in lm]
        cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
        size = max(max(xs)-min(xs), max(ys)-min(ys)) * 1.3
        x1, y1 = max(0, int(cx-size/2)), max(0, int(cy-size/2))
        x2, y2 = min(w, int(cx+size/2)), min(h, int(cy+size/2))
        crop = img[y1:y2, x1:x2]
        out['face_crop_variable'] = crop if crop.size > 0 else img[32:224, 32:224]
    else:
        out['face_crop_variable'] = img[32:224, 32:224]
    o = (256-192)//2
    out['center_crop_192'] = img[o:o+192, o:o+192]
    r = np.random.RandomState(42 + idx); mx = 64
    ox, oy = int(r.randint(0, mx+1)), int(r.randint(0, mx+1))
    out['random_crop_192_a'] = img[oy:oy+192, ox:ox+192]
    return out

n_created = 0
for idx, (uid, ver, src_path, transform) in enumerate(tqdm(images, desc='prep', unit='img')):
    # check if all variants already exist
    if all((VAR_BASE / v / f'{uid}.png').exists() for v in VARIANTS):
        continue
    img = cv2.imread(str(src_path))
    if img is None:
        print(f'⚠ read fail: {src_path}'); continue
    if img.shape[:2] != (256, 256):
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    if transform == 'jpeg_q80':
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    crops = gen_crops(img, idx)
    for v_name, v_img in crops.items():
        vp = VAR_BASE / v_name / f'{uid}.png'
        if not vp.exists():
            cv2.imwrite(str(vp), v_img)
            n_created += 1
fm.close()

# Save metadata
with open(OUT / 'image_list.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['uid', 'version'])
    for uid, ver, _, _ in images:
        w.writerow([uid, ver])

print(f'\n생성: {n_created} files')
for v in VARIANTS:
    n = len(list((VAR_BASE / v).glob('*.png')))
    print(f'  {v}: {n}')
print('완료 ✓')
