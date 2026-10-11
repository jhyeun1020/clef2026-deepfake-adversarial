"""
Step 1: Test set 24,404장 전체 face detection (MediaPipe FaceMesh)
- 실패 파일명 → face_detection_failures.csv
- 성공 파일은 face landmarks도 함께 캐시 (이후 face_crop_variable 생성에 재사용)
"""
import csv, time, pickle
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm

BASE = Path('/home/student/GEN_AI_GTG')
TEST_DIR = BASE / 'detection/ImageCLEF2026-DeepFakeDetection-Tes/Data/Images_Detection'
OUT = BASE / 'detection/instability/test_application'
OUT.mkdir(parents=True, exist_ok=True)
FAIL_CSV = OUT / 'face_detection_failures.csv'
LM_CACHE = OUT / 'test_face_landmarks.pkl'

paths = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() == '.png')
print(f'test set: {len(paths)}장')

if LM_CACHE.exists():
    print(f'캐시 사용: {LM_CACHE}')
    with open(LM_CACHE, 'rb') as f:
        landmarks_cache = pickle.load(f)
    fails = [n for n, lm in landmarks_cache.items() if lm is None]
else:
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    landmarks_cache = {}
    fails = []
    t0 = time.time()
    for p in tqdm(paths, desc='face detect', unit='img'):
        img = cv2.imread(str(p))
        if img is None:
            landmarks_cache[p.name] = None
            fails.append(p.name); continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            landmarks_cache[p.name] = None
            fails.append(p.name)
        else:
            lm = [(l.x, l.y) for l in res.multi_face_landmarks[0].landmark]
            landmarks_cache[p.name] = lm
    fm.close()
    with open(LM_CACHE, 'wb') as f:
        pickle.dump(landmarks_cache, f)
    print(f'\n경과: {time.time()-t0:.1f}s, 캐시 저장')

# 실패 CSV 저장
with open(FAIL_CSV, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['filename'])
    for n in fails: w.writerow([n])

n_total = len(paths)
n_fail = len(fails)
print(f'\nFace detection 결과:')
print(f'  total       : {n_total}')
print(f'  detected    : {n_total - n_fail}  ({(n_total-n_fail)/n_total*100:.1f}%)')
print(f'  not detected: {n_fail}  ({n_fail/n_total*100:.1f}%)')
print(f'\n실패 리스트 저장: {FAIL_CSV}')
print(f'landmarks 캐시 : {LM_CACHE}')
