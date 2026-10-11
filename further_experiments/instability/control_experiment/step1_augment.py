"""
Step 1: 9 augmentations × 20 real images = 180 augmented + crop variants 생성
"""
import io, csv
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm

BASE = Path('/home/student/GEN_AI_GTG')
REAL_DIR = BASE / 'detection/real_reference/center'
OUT_AUG = BASE / 'detection/instability/control_experiment/augmented_real'
OUT_VAR = BASE / 'detection/instability/control_experiment/variants'
META_CSV = BASE / 'detection/instability/control_experiment/aug_metadata.csv'

AUGMENTATIONS = [
    ('original',          {}),
    ('gaussian_noise_002', {'sigma': 0.02}),
    ('gaussian_noise_005', {'sigma': 0.05}),
    ('gaussian_blur_3',    {'kernel': 3}),
    ('gaussian_blur_5',    {'kernel': 5}),
    ('jpeg_q50',           {'quality': 50}),
    ('jpeg_q70',           {'quality': 70}),
    ('rotation_15',        {'angle': 15}),
    ('poisson_10',         {'scale': 10}),
    ('rotation_20',        {'angle': 20}),
]
CROP_VARIANTS = ['full', 'face_crop_variable', 'center_crop_192', 'random_crop_192_a']

# 디렉토리 생성
for aug_name, _ in AUGMENTATIONS:
    (OUT_AUG / aug_name).mkdir(parents=True, exist_ok=True)
    for v in CROP_VARIANTS:
        (OUT_VAR / aug_name / v).mkdir(parents=True, exist_ok=True)

# Real 20장 sampling
rng = np.random.RandomState(42)
all_pngs = sorted(REAL_DIR.glob('*.png'))
selected = rng.choice(len(all_pngs), size=20, replace=False)
real_paths = [all_pngs[i] for i in sorted(selected)]
print(f'Real 20장: {[p.stem for p in real_paths[:3]]}...')

# Augmentation 함수
def apply_aug(img, aug_name, params):
    if aug_name == 'original':
        return img.copy()
    if aug_name.startswith('gaussian_noise'):
        sigma = params['sigma']
        noise = np.random.normal(0, sigma * 255, img.shape)
        return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    if aug_name.startswith('gaussian_blur'):
        k = params['kernel']
        return cv2.GaussianBlur(img, (k, k), 0)
    if aug_name.startswith('jpeg_q'):
        q = params['quality']
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if aug_name.startswith('rotation'):
        angle = params['angle']
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))
    if aug_name.startswith('poisson'):
        scale = params['scale']
        img_f = img.astype(np.float64)
        noisy = np.random.poisson(img_f * scale / 255.0).astype(np.float64) * 255.0 / scale
        return np.clip(noisy, 0, 255).astype(np.uint8)
    raise ValueError(f'Unknown aug: {aug_name}')

# face detection + crop variants
fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def gen_crop_variants(img, idx):
    h, w = img.shape[:2]
    out = {'full': img.copy()}
    # face detection
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    lm = None
    if res.multi_face_landmarks:
        lm = [(l.x, l.y) for l in res.multi_face_landmarks[0].landmark]
    # face_crop_variable (center_192 fallback)
    if lm:
        xs = [l[0]*w for l in lm]; ys = [l[1]*h for l in lm]
        x_min, x_max = min(xs), max(xs); y_min, y_max = min(ys), max(ys)
        fw, fh = x_max-x_min, y_max-y_min
        cx, cy = (x_min+x_max)/2, (y_min+y_max)/2
        size = max(fw, fh) * 1.3
        x1 = max(0, int(cx-size/2)); y1 = max(0, int(cy-size/2))
        x2 = min(w, int(cx+size/2)); y2 = min(h, int(cy+size/2))
        crop = img[y1:y2, x1:x2]
        out['face_crop_variable'] = crop if crop.size > 0 else img[32:224, 32:224]
    else:
        out['face_crop_variable'] = img[32:224, 32:224]  # center_192 fallback
    # center_crop_192
    o = (256-192)//2
    out['center_crop_192'] = img[o:o+192, o:o+192]
    # random_crop_192_a
    r = np.random.RandomState(42 + idx)
    mx = 256 - 192
    ox, oy = int(r.randint(0, mx+1)), int(r.randint(0, mx+1))
    out['random_crop_192_a'] = img[oy:oy+192, ox:ox+192]
    return out, lm is not None

# 생성
meta_rows = []
total = len(AUGMENTATIONS) * len(real_paths)
pbar = tqdm(total=total, desc='aug+crop', unit='img')
for aug_name, params in AUGMENTATIONS:
    for idx, rp in enumerate(real_paths):
        stem = rp.stem
        img_orig = cv2.imread(str(rp))
        if img_orig is None: continue
        if img_orig.shape[:2] != (256, 256):
            img_orig = cv2.resize(img_orig, (256, 256), interpolation=cv2.INTER_LINEAR)
        # augment
        np.random.seed(42 + idx)
        aug_img = apply_aug(img_orig, aug_name, params)
        aug_path = OUT_AUG / aug_name / f'{stem}.png'
        if not aug_path.exists():
            cv2.imwrite(str(aug_path), aug_img)
        # crop variants
        variants, face_ok = gen_crop_variants(aug_img, idx)
        for v_name, v_img in variants.items():
            vp = OUT_VAR / aug_name / v_name / f'{stem}.png'
            if not vp.exists():
                cv2.imwrite(str(vp), v_img)
        meta_rows.append({
            'stem': stem, 'aug_type': aug_name,
            'face_detected': face_ok,
        })
        pbar.update(1)
pbar.close()
fm.close()

with open(META_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
    w.writeheader(); w.writerows(meta_rows)
print(f'\nmetadata: {len(meta_rows)} rows → {META_CSV}')

# sample grid (각 aug 1장)
from PIL import Image, ImageDraw, ImageFont
cell = 200; pad = 4; label_h = 20
cols = 5; rows = 2
W = cols * (cell + pad) + pad
H = rows * (cell + label_h + pad) + pad
grid = Image.new('RGB', (W, H), (25, 25, 25))
draw = ImageDraw.Draw(grid)
try: fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
except: fnt = ImageFont.load_default()
sample_stem = real_paths[0].stem
for i, (aug_name, _) in enumerate(AUGMENTATIONS):
    r, c = i // cols, i % cols
    x = pad + c * (cell + pad)
    y = pad + r * (cell + label_h + pad)
    p = OUT_AUG / aug_name / f'{sample_stem}.png'
    im = Image.open(p).convert('RGB').resize((cell, cell))
    grid.paste(im, (x, y))
    draw.text((x, y + cell + 2), aug_name, fill=(220, 220, 220), font=fnt)
grid.save(BASE / 'detection/instability/control_experiment/sample_augmented_images.png')
print('sample grid 저장')

# variant count
for aug_name, _ in AUGMENTATIONS:
    for v in CROP_VARIANTS:
        n = len(list((OUT_VAR / aug_name / v).glob('*.png')))
        if n != 20:
            print(f'  ⚠ {aug_name}/{v}: {n}')
print('완료 ✓')
