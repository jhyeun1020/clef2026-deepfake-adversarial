"""
ImageCLEF 2026 Deepfake Detection — pHash matching
1002 generated (Generation task 3rd submission) → 24404 test set 안에서 동일 이미지 찾기
재실행 가능 (캐시 활용), Stage 1 strict → Stage 2 mutual NN.
"""
import os, sys, json, pickle, time, traceback, random
from pathlib import Path
from PIL import Image
import numpy as np
import imagehash
from tqdm import tqdm

# ── 경로 ────────────────────────────────────────────────────────────────────
BASE = Path('/home/student/GEN_AI_GTG')
OURS_DIR = BASE / 'generation/submit/third_sub/256'
TEST_DIR = BASE / 'detection/ImageCLEF2026-DeepFakeDetection-Tes/Data/Images_Detection'
OUT_DIR  = BASE / 'detection/phash_matching'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_OURS = OUT_DIR / 'phash_cache_ours.pkl'
CACHE_TEST = OUT_DIR / 'phash_cache_test.pkl'
ERR_LOG    = OUT_DIR / 'errors.log'
MATCHES_CSV= OUT_DIR / 'fake_gt_matches.csv'
REPORT_TXT = OUT_DIR / 'matching_report.txt'
GRID_PNG   = OUT_DIR / 'verification_grid.png'

HASH_SIZE = 8  # 8×8 = 64bit DCT pHash
STAGE1_THRESH = 5
STAGE2_THRESH = 10
STAGE1_GOAL   = 700  # Stage 1 매칭 ≥ 이 수면 Stage 2 생략

errors = []

def log_err(path, e):
    errors.append(f"{path}\t{type(e).__name__}: {e}")

# ── pHash 계산 ─────────────────────────────────────────────────────────────
def compute_hashes(image_dir, cache_path, label):
    """image_dir 안 모든 .png에 대해 imagehash.phash 계산. dict[stem] = hash 반환."""
    if cache_path.exists():
        print(f"[{label}] 캐시 로드: {cache_path}", flush=True)
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() == '.png')
    print(f"[{label}] {len(paths)}장 pHash 계산 시작...", flush=True)

    hashes = {}
    for p in tqdm(paths, desc=f'pHash {label}', unit='img'):
        try:
            with Image.open(p) as im:
                h = imagehash.phash(im.convert('L'), hash_size=HASH_SIZE)
            # ImageHash 객체 그대로 저장 (pickle 가능)
            hashes[p.name] = h
        except Exception as e:
            log_err(p, e)

    with open(cache_path, 'wb') as f:
        pickle.dump(hashes, f)
    print(f"[{label}] 캐시 저장: {cache_path} ({len(hashes)}개)", flush=True)
    return hashes

# ── Hamming distance 행렬 (numpy 가속) ─────────────────────────────────────
def hashes_to_bits(hashes_dict):
    """dict[name] = ImageHash → (names_list, np.uint8 [N, 8] packed bits)"""
    names = list(hashes_dict.keys())
    arr = np.stack([np.packbits(h.hash.flatten()) for h in hashes_dict.values()])
    return names, arr  # arr shape (N, 8) uint8

POPCNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

def hamming_matrix(A, B):
    """A: (Na,8), B: (Nb,8) → (Na, Nb) hamming distance matrix.
    XOR 후 popcount. chunk 단위로 메모리 절약."""
    Na, Nb = A.shape[0], B.shape[0]
    out = np.empty((Na, Nb), dtype=np.uint8)
    chunk = 256
    for i in tqdm(range(0, Na, chunk), desc='Hamming', unit='chunk'):
        a = A[i:i+chunk, None, :]               # (c, 1, 8)
        x = np.bitwise_xor(a, B[None, :, :])    # (c, Nb, 8)
        out[i:i+chunk] = POPCNT[x].sum(axis=2)  # (c, Nb)
    return out

# ── 매칭 알고리즘 ──────────────────────────────────────────────────────────
def stage1_strict(test_names, our_names, D, thresh):
    """test_img → 가장 가까운 our_img가 thresh 이하면 매칭.
    같은 our_img에 여러 test가 매칭될 수 있으니 (test, our, dist) 리스트 반환."""
    matches = []
    nearest_our = D.argmin(axis=1)  # for each test, nearest our index
    nearest_dist = D.min(axis=1)
    for ti, (oi, d) in enumerate(zip(nearest_our, nearest_dist)):
        if d <= thresh:
            matches.append((test_names[ti], our_names[oi], int(d)))
    return matches

def stage2_mutual_nn(test_names, our_names, D, thresh):
    """양방향 nearest neighbor + threshold."""
    nearest_our_for_test = D.argmin(axis=1)   # (Nt,)
    nearest_test_for_our = D.argmin(axis=0)   # (No,)
    matches = []
    for ti, oi in enumerate(nearest_our_for_test):
        if nearest_test_for_our[oi] != ti:
            continue  # not mutual
        d = int(D[ti, oi])
        if d <= thresh:
            matches.append((test_names[ti], our_names[oi], int(d)))
    return matches

# ── distance 분포 출력 ─────────────────────────────────────────────────────
def dist_summary(matches, label):
    if not matches:
        print(f"  [{label}] 매칭 0개", flush=True)
        return
    dists = np.array([m[2] for m in matches])
    print(f"  [{label}] 매칭 {len(matches)}개 | dist mean={dists.mean():.2f} median={np.median(dists):.0f} min={dists.min()} max={dists.max()}", flush=True)
    # 히스토그램
    hist, edges = np.histogram(dists, bins=range(0, max(int(dists.max()), 1) + 2))
    bars = ''
    for h, e in zip(hist, edges[:-1]):
        bars += f"    d={e:>2}: {'█' * min(h, 60)} ({h})\n"
    print(bars, flush=True)
    return dists

# ── 검증용 그리드 ──────────────────────────────────────────────────────────
def make_verification_grid(matches, n_random=20, n_extreme=5):
    """매칭 결과 중 무작위 + 거리 양극단을 좌우로 배치한 그리드 PNG 생성."""
    if not matches:
        return
    sorted_m = sorted(matches, key=lambda x: x[2])
    pick = []
    pick += sorted_m[:n_extreme]                              # 가장 가까운
    pick += sorted_m[-n_extreme:] if len(sorted_m) > n_extreme else []  # 가장 먼
    random.seed(42)
    rest = [m for m in sorted_m[n_extreme:-n_extreme]] if len(sorted_m) > 2*n_extreme else []
    if rest:
        pick += random.sample(rest, min(n_random, len(rest)))

    rows = len(pick)
    cell = 256
    pad = 8
    label_h = 24
    W = cell * 2 + pad * 3
    H = (cell + label_h + pad) * rows + pad
    grid = Image.new('RGB', (W, H), (30, 30, 30))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except Exception:
        font = ImageFont.load_default()

    for i, item in enumerate(pick):
        test_name, our_name, d = item[0], item[1], item[2]
        y = pad + i * (cell + label_h + pad)
        try:
            t_img = Image.open(TEST_DIR / test_name).convert('RGB').resize((cell, cell))
            o_img = Image.open(OURS_DIR / our_name).convert('RGB').resize((cell, cell))
            grid.paste(t_img, (pad, y))
            grid.paste(o_img, (pad * 2 + cell, y))
        except Exception as e:
            log_err(test_name, e)
            continue
        draw.text((pad, y + cell + 2),
                  f'TEST: {test_name}   |   OURS: {our_name}   |   d={d}',
                  fill=(255, 255, 255), font=font)

    grid.save(GRID_PNG)
    print(f"검증 그리드 저장: {GRID_PNG} ({rows}쌍)", flush=True)

# ── main ───────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 70)
    print("ImageCLEF 2026 Detection — pHash matching")
    print("=" * 70)
    print(f"OURS_DIR : {OURS_DIR}")
    print(f"TEST_DIR : {TEST_DIR}")
    print(f"OUT_DIR  : {OUT_DIR}\n")

    ours_hashes = compute_hashes(OURS_DIR, CACHE_OURS, 'ours')
    test_hashes = compute_hashes(TEST_DIR, CACHE_TEST, 'test')
    print(f"\nours: {len(ours_hashes)}개 / test: {len(test_hashes)}개\n")

    print("Hamming distance 행렬 계산 중...", flush=True)
    test_names, T_bits = hashes_to_bits(test_hashes)
    our_names,  O_bits = hashes_to_bits(ours_hashes)
    D = hamming_matrix(T_bits, O_bits)  # (Nt, No) uint8
    print(f"  D shape: {D.shape}, dtype={D.dtype}\n")

    # Stage 1 ── strict
    print(f"━━━ Stage 1: strict (threshold={STAGE1_THRESH}) ━━━", flush=True)
    s1 = stage1_strict(test_names, our_names, D, STAGE1_THRESH)
    s1_d = dist_summary(s1, 'Stage 1')

    # 같은 our에 다중 매칭된 our 진단
    from collections import Counter
    our_count_s1 = Counter(o for (_, o, _) in s1)
    s1_dups = sum(1 for v in our_count_s1.values() if v > 1)
    print(f"  같은 our에 다중 매칭된 우리 이미지: {s1_dups}개", flush=True)

    # precision 우선: Stage 1 결과 중 mutual NN 통과한 것만 유지
    print(f"\n━━━ Stage 1 + mutual NN 정제 ━━━", flush=True)
    s1_mutual = stage2_mutual_nn(test_names, our_names, D, STAGE1_THRESH)
    dist_summary(s1_mutual, 'Stage 1 mutual')

    final_matches = [(t, o, d, 1) for (t, o, d) in s1_mutual]
    used_stage = 1

    if len(s1_mutual) < STAGE1_GOAL:
        print(f"\nStage 1 mutual {len(s1_mutual)}개 < goal {STAGE1_GOAL} → Stage 2 진행", flush=True)
        print(f"\n━━━ Stage 2: mutual NN (threshold={STAGE2_THRESH}) ━━━", flush=True)
        s2 = stage2_mutual_nn(test_names, our_names, D, STAGE2_THRESH)
        s2_d = dist_summary(s2, 'Stage 2')

        # Stage 1 mutual은 모두 유지하고, Stage 2에서 추가되는 것만 stage=2
        s1_keys = {(t, o) for (t, o, _) in s1_mutual}
        added = 0
        for (t, o, d) in s2:
            if (t, o) not in s1_keys:
                final_matches.append((t, o, d, 2))
                added += 1
        print(f"\nStage 2에서 추가 매칭: {added}개", flush=True)
        used_stage = 2
    else:
        print(f"\nStage 1 mutual {len(s1_mutual)}개 ≥ goal {STAGE1_GOAL} → Stage 2 생략", flush=True)

    # 결과 저장
    print(f"\n총 최종 매칭: {len(final_matches)}개", flush=True)
    all_d = np.array([m[2] for m in final_matches]) if final_matches else np.array([])
    if len(all_d):
        print(f"  dist mean={all_d.mean():.2f} median={np.median(all_d):.0f} min={all_d.min()} max={all_d.max()}", flush=True)

    import csv
    with open(MATCHES_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['test_image_path', 'our_image_path', 'hamming_distance', 'stage'])
        for (t, o, d, s) in final_matches:
            w.writerow([str(TEST_DIR / t), str(OURS_DIR / o), d, s])
    print(f"매칭 CSV 저장: {MATCHES_CSV}", flush=True)

    # report
    with open(REPORT_TXT, 'w') as f:
        f.write("ImageCLEF 2026 Detection — pHash matching report\n")
        f.write("=" * 60 + "\n")
        f.write(f"OURS_DIR : {OURS_DIR}\n")
        f.write(f"TEST_DIR : {TEST_DIR}\n")
        f.write(f"ours   : {len(ours_hashes)}장\n")
        f.write(f"test   : {len(test_hashes)}장\n\n")
        f.write(f"Stage 1 (strict, thresh={STAGE1_THRESH}): {len(s1)}개 (raw)\n")
        f.write(f"  같은 our에 다중 매칭된 우리 이미지: {s1_dups}개\n")
        f.write(f"Stage 1 (mutual NN, thresh={STAGE1_THRESH}): {len(s1_mutual)}개\n")
        if used_stage == 2:
            stage2_added = len(final_matches) - len(s1)
            f.write(f"Stage 2 (mutual NN, thresh={STAGE2_THRESH}): {stage2_added}개 추가\n")
        f.write(f"\n총 최종 매칭: {len(final_matches)}개\n")
        if len(all_d):
            f.write(f"distance: mean={all_d.mean():.3f} median={np.median(all_d):.1f} ")
            f.write(f"min={all_d.min()} max={all_d.max()}\n")
            hist, edges = np.histogram(all_d, bins=range(0, int(all_d.max())+2))
            f.write("\ndistance 히스토그램:\n")
            for h, e in zip(hist, edges[:-1]):
                f.write(f"  d={e:>2}: {h}\n")
        f.write(f"\n에러: {len(errors)}건 (자세히는 errors.log)\n")
        f.write(f"\n경과 시간: {time.time() - t0:.1f}s\n")
    print(f"리포트 저장: {REPORT_TXT}", flush=True)

    # 검증 그리드
    make_verification_grid(final_matches, n_random=15, n_extreme=5)

    if errors:
        with open(ERR_LOG, 'w') as f:
            f.write('\n'.join(errors))
        print(f"에러 {len(errors)}건 저장: {ERR_LOG}", flush=True)

    print(f"\n전체 경과: {time.time() - t0:.1f}s")
    print("완료 ✓")

if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
