"""Stage 1에서 d=4로 매칭됐던 3개 false positive 시각화."""
import pickle
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path('/home/student/GEN_AI_GTG')
OURS_DIR = BASE / 'generation/submit/third_sub/256'
TEST_DIR = BASE / 'detection/ImageCLEF2026-DeepFakeDetection-Tes/Data/Images_Detection'
OUT = BASE / 'detection/phash_matching'

# 진단에서 확인된 3개의 d=4 매칭
fps = [
    ('0efc7c82c6f484559256.png', 'ID_139_GT_frame1.png', 4),
    ('3a406da419d88bdafc12.png', 'ID_180_GT_frame1.png', 4),
    ('dcb43ed634a4c5038830.png', 'ID_116_GT_frame3.png', 4),
]

# 각 our_image가 d=0로 매칭된 진짜 test 이미지도 같이 보여주기
import csv
gt = {}
for r in csv.DictReader(open(OUT / 'fake_gt_matches.csv')):
    our = Path(r['our_image_path']).name
    test = Path(r['test_image_path']).name
    gt[our] = test

cell = 256
pad = 8
label_h = 28
W = cell * 3 + pad * 4
H = (cell + label_h + pad) * len(fps) + pad
grid = Image.new('RGB', (W, H), (30, 30, 30))
draw = ImageDraw.Draw(grid)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
except Exception:
    font = ImageFont.load_default()

# 헤더
header = Image.new('RGB', (W, 24), (50, 50, 50))
hd = ImageDraw.Draw(header)
hd.text((pad, 4), '  TRUE MATCH (d=0)             |   OUR IMAGE                    |   FALSE POSITIVE (d=4)',
        fill=(255, 255, 200), font=font)

for i, (fp_test, our, d) in enumerate(fps):
    y = pad + i * (cell + label_h + pad)
    true_test = gt[our]

    t_img = Image.open(TEST_DIR / true_test).convert('RGB').resize((cell, cell))
    o_img = Image.open(OURS_DIR / our).convert('RGB').resize((cell, cell))
    f_img = Image.open(TEST_DIR / fp_test).convert('RGB').resize((cell, cell))

    grid.paste(t_img, (pad, y))
    grid.paste(o_img, (pad * 2 + cell, y))
    grid.paste(f_img, (pad * 3 + cell * 2, y))

    draw.text((pad, y + cell + 4),
              f'true: {true_test}',
              fill=(150, 255, 150), font=font)
    draw.text((pad * 2 + cell, y + cell + 4),
              f'ours: {our}',
              fill=(255, 255, 255), font=font)
    draw.text((pad * 3 + cell * 2, y + cell + 4),
              f'FP:   {fp_test} (d={d})',
              fill=(255, 150, 150), font=font)

out_path = OUT / 'false_positive_inspection.png'
grid.save(out_path)
print(f'저장: {out_path}')
print('\n3개 false positive 매칭 (d=4):')
for fp_test, our, d in fps:
    print(f'  test  : {fp_test}')
    print(f'  ours  : {our}')
    print(f'  (true : {gt[our]})')
    print()
