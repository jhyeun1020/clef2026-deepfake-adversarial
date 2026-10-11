"""
ID당 비디오에서 얼굴 크롭 이미지 20장 추출
- InsightFace로 얼굴 감지 + 정사각형 크롭
- 블러 필터(Laplacian 분산)로 선명한 프레임 우선 선택
- det_score × sharpness 복합 점수 기준 정렬
- data/gold_training_set/cropped/{id_str}/ 에 저장
"""
import cv2
import numpy as np
from pathlib import Path
import av as _av

# ── 설정 ──────────────────────────────────────────────────────────
VIDEO_DIR   = Path("/home/student/GEN_AI_GTG/datasets/Development Videos")
OUT_DIR     = Path("/home/student/GEN_AI_GTG/data/gold_training_set/cropped")
MODEL_DIR   = "/home/student/GEN_AI_GTG/images/models"
TARGET_PER_ID  = 20
MIN_DET_SCORE  = 0.4
FRAME_INTERVAL = 5     # 5프레임마다 1장 샘플링
CROP_SIZE      = 256   # 출력 정사각형 크기
CROP_MARGIN    = 0.35  # bbox 기준 상하좌우 여백 비율

from insightface.app import FaceAnalysis

print("InsightFace 초기화 중...")
app = FaceAnalysis(name="buffalo_s", root=MODEL_DIR,
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_thresh=MIN_DET_SCORE)
print("완료\n")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_frames(vid_path):
    """OpenCV(H264) / PyAV(AV1) 자동 선택"""
    cap = cv2.VideoCapture(str(vid_path))
    ret, _ = cap.read()
    if ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
        cap.release()
        return
    cap.release()
    container = _av.open(str(vid_path))
    for frame_idx, frame in enumerate(container.decode(video=0)):
        yield frame_idx, frame.to_ndarray(format="bgr24")
    container.close()


def sharpness(img_bgr):
    """Laplacian 분산 — 높을수록 선명"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def square_crop(frame, bbox, margin=CROP_MARGIN):
    """
    bbox [x1,y1,x2,y2] 기준 정사각형 크롭.
    margin 비율만큼 bbox 확장 후 정사각형으로 맞춤.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin, bh * margin

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half = max(bw + 2 * mx, bh + 2 * my) / 2

    sx = int(max(0, cx - half))
    sy = int(max(0, cy - half))
    ex = int(min(w, cx + half))
    ey = int(min(h, cy + half))

    crop = frame[sy:ey, sx:ex]
    if crop.size == 0:
        return None

    # 정사각형 패딩 (크롭이 이미 거의 정사각형이지만 경계 처리)
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    pad = np.zeros((side, side, 3), dtype=np.uint8)
    oh = (side - ch) // 2
    ow = (side - cw) // 2
    pad[oh:oh+ch, ow:ow+cw] = crop

    return cv2.resize(pad, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LANCZOS4)


def process_video(id_str, vid_path):
    out_subdir = OUT_DIR / id_str
    # 이미 충분히 있으면 스킵
    existing = list(out_subdir.glob("*.png")) if out_subdir.exists() else []
    if len(existing) >= TARGET_PER_ID:
        print(f"[{id_str}] 이미 {len(existing)}장 존재 → 스킵")
        return 0

    candidates = []  # (score, frame_idx, cropped_img)

    for frame_idx, frame in read_frames(vid_path):
        if frame_idx % FRAME_INTERVAL != 0:
            continue
        faces = app.get(frame)
        if not faces:
            continue
        # 가장 큰 얼굴만 사용
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        det_score = float(face.det_score)
        if det_score < MIN_DET_SCORE:
            continue

        cropped = square_crop(frame, face.bbox)
        if cropped is None:
            continue

        sharp = sharpness(cropped)
        # 복합 점수: det_score(신뢰도) × log(sharpness+1) 정규화
        combined = det_score * np.log1p(sharp)
        candidates.append((combined, frame_idx, cropped))

    if not candidates:
        print(f"[{id_str}] 얼굴 감지 실패")
        return 0

    # 복합 점수 내림차순 정렬
    candidates.sort(key=lambda x: x[0], reverse=True)

    # 다양성 확보: 상위 후보 중 시간적으로 분산된 프레임 선택
    selected = []
    used_fidx = []
    MIN_GAP = 15  # 선택된 프레임 간 최소 간격
    for score, fidx, crop in candidates:
        if all(abs(fidx - u) >= MIN_GAP for u in used_fidx):
            selected.append((score, fidx, crop))
            used_fidx.append(fidx)
        if len(selected) >= TARGET_PER_ID:
            break

    # 부족하면 간격 무시하고 점수 순으로 채움
    if len(selected) < TARGET_PER_ID:
        used_set = {fidx for _, fidx, _ in selected}
        for score, fidx, crop in candidates:
            if fidx not in used_set:
                selected.append((score, fidx, crop))
                used_set.add(fidx)
            if len(selected) >= TARGET_PER_ID:
                break

    out_subdir.mkdir(parents=True, exist_ok=True)
    for i, (score, fidx, crop) in enumerate(selected[:TARGET_PER_ID]):
        cv2.imwrite(str(out_subdir / f"{id_str}_crop_{i:02d}_f{fidx}.png"), crop)

    scores = [s for s, _, _ in selected[:TARGET_PER_ID]]
    print(f"[{id_str}] {len(selected[:TARGET_PER_ID])}장 저장 "
          f"(복합점수 {min(scores):.1f}~{max(scores):.1f})")
    return len(selected[:TARGET_PER_ID])


# ── 실행 ──────────────────────────────────────────────────────────
video_paths = sorted(VIDEO_DIR.glob("ID_*.mp4"))
print(f"비디오 수: {len(video_paths)}\n")

total = 0
for vid_path in video_paths:
    id_str = vid_path.stem.replace("_dev", "")
    n = process_video(id_str, vid_path)
    total += n

print(f"\n총 {total}장 추출 완료")
print(f"저장 위치: {OUT_DIR}")
