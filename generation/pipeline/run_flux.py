#!/usr/bin/env python
"""experiment_flux.ipynb 셀 내용을 순서대로 실행"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import json as _json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mediapipe as mp

from diffusers import FluxControlNetPipeline, FluxControlNetModel

assert torch.cuda.is_available()
NUM_GPUS = torch.cuda.device_count()
DTYPE    = torch.bfloat16
SEED     = 42
IMG_SIZE = 512
DEVICE   = "cuda:0"

print(f"GPU {NUM_GPUS}개:")
for i in range(NUM_GPUS):
    print(f"  cuda:{i} — {torch.cuda.get_device_name(i)}")

OUTPUT_DIR       = Path("outputs/flux")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONSTRAINTS_PATH = Path("/home/student/GEN_AI_GTG/images/files/generation_constraints.json")
DATA_DIR         = Path("/home/student/GEN_AI_GTG/data/gold_training_set/train")
HF_TOKEN         = os.environ.get("HF_TOKEN")

with open(CONSTRAINTS_PATH) as f:
    constraints_all = _json.load(f)

print(f"제약조건 키 수: {len(constraints_all)}")

mp_face_mesh = mp.solutions.face_mesh

MP_TO_68 = [
    234, 93, 132, 58, 172, 136, 150, 149, 152, 377, 378, 400, 290, 305, 321, 375, 454,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    168, 197, 195, 5,
    98, 97, 2, 326, 327,
    33, 246, 161, 160, 159, 158,
    362, 398, 384, 385, 386, 387,
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 306,
    78, 191, 80, 81, 82, 312, 311, 310,
]

FACE_CONNECTIONS = (
    list(zip(range(0, 16), range(1, 17))) +
    list(zip(range(17, 21), range(18, 22))) +
    list(zip(range(22, 26), range(23, 27))) +
    list(zip(range(27, 30), range(28, 31))) +
    list(zip(range(31, 35), range(32, 36))) +
    list(zip(range(36, 41), range(37, 42))) + [(41, 36)] +
    list(zip(range(42, 47), range(43, 48))) + [(47, 42)] +
    list(zip(range(48, 59), range(49, 60))) + [(59, 48)] +
    list(zip(range(60, 67), range(61, 68))) + [(67, 60)]
)

KP_COLORS = (
    [(255, 255, 255)] * 17 +
    [(255, 200, 0)]   * 5  +
    [(255, 200, 0)]   * 5  +
    [(255, 128, 0)]   * 4  +
    [(255, 128, 0)]   * 5  +
    [(0, 255, 0)]     * 6  +
    [(0, 255, 0)]     * 6  +
    [(255, 0, 85)]    * 12 +
    [(255, 0, 85)]    * 8
)

def render_facemesh_as_pose(landmarks_xy, canvas_size=512, dot_radius=3, line_thickness=1):
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    all_pts = [(int(x * canvas_size), int(y * canvas_size)) for x, y in landmarks_xy]
    n = len(all_pts)
    kp68 = [all_pts[mp_idx] if mp_idx < n else None for mp_idx in MP_TO_68]
    for i, j in FACE_CONNECTIONS:
        if i < len(kp68) and j < len(kp68) and kp68[i] and kp68[j]:
            cv2.line(canvas, kp68[i], kp68[j], (64, 64, 64), line_thickness)
    for idx, pt in enumerate(kp68):
        if pt is not None:
            color = KP_COLORS[idx] if idx < len(KP_COLORS) else (255, 255, 255)
            cv2.circle(canvas, pt, dot_radius, color, -1)
    return Image.fromarray(canvas)

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

def pick_constraint_key(id_str):
    for suffix in ['_GT_frame1', '_GT_frame2', '_GT_frame3']:
        k = id_str + suffix
        if k in constraints_all:
            return k
    return None

TEST_IDS = ["ID_0", "ID_1", "ID_10", "ID_50", "ID_100"]

PROMPT = (
    "photorealistic portrait photo of a person, natural skin texture, "
    "podcast studio background, soft lighting, high detail"
)
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE      = 3.5

print("Cell 1 완료\n")

# ── Cell 2: 포즈 키포인트 미리보기 저장 ──────────────────────────────────────
fig, axes = plt.subplots(1, len(TEST_IDS), figsize=(3 * len(TEST_IDS), 3))
for ax, id_str in zip(axes, TEST_IDS):
    key = pick_constraint_key(id_str)
    if key is None:
        ax.set_title(f"{id_str}\n(키 없음)"); ax.axis('off'); continue
    coords   = constraints_all[key]["mediapipe_landmarks_facemesh"]
    edge_img = render_facemesh_as_pose(coords, canvas_size=IMG_SIZE)
    ax.imshow(edge_img)
    ax.set_title(f"{id_str}", fontsize=9)
    ax.axis('off')
plt.suptitle("FaceMesh → Pose 키포인트 이미지", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pose_preview.png", dpi=100, bbox_inches="tight")
plt.close()
print("Pose 키포인트 미리보기 저장 완료: outputs/flux/pose_preview.png\n")

# ── Cell 3: 모델 로드 ──────────────────────────────────────────────────────────
print("FLUX ControlNet 로드 중...")
controlnet = FluxControlNetModel.from_pretrained(
    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
    torch_dtype=DTYPE,
    token=HF_TOKEN,
)
print("  ControlNet 완료")

print("FLUX.1-dev 파이프라인 로드 중 (시간 소요)...")
pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    torch_dtype=DTYPE,
    token=HF_TOKEN,
)
pipe.enable_model_cpu_offload(gpu_id=0)
print("파이프라인 로드 완료\n")

# ── Cell 4: CN scale 스윕 ──────────────────────────────────────────────────────
import pandas as pd

CN_SCALES            = [0.6, 0.7, 0.8, 0.9, 1.0]
CONTROL_GUIDANCE_END = 0.65

results = []

for id_str in TEST_IDS:
    key = pick_constraint_key(id_str)
    if key is None:
        print(f"{id_str}: 키 없음, 스킵")
        continue

    coords   = constraints_all[key]["mediapipe_landmarks_facemesh"]
    pose_img = render_facemesh_as_pose(coords, canvas_size=IMG_SIZE)

    for cn_scale in CN_SCALES:
        gen = torch.Generator(device="cpu").manual_seed(SEED)
        try:
            out = pipe(
                prompt=PROMPT,
                control_image=pose_img,
                controlnet_conditioning_scale=float(cn_scale),
                control_guidance_end=CONTROL_GUIDANCE_END,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=IMG_SIZE,
                width=IMG_SIZE,
                generator=gen,
            ).images[0]

            gen_lm  = extract_landmarks_pil(out)
            lm_dist = euclidean_dist(gen_lm, coords)

            d = OUTPUT_DIR / id_str
            d.mkdir(parents=True, exist_ok=True)
            out.save(d / f"cn{cn_scale:.1f}.png")

            row = {"id_str": id_str, "cn_scale": cn_scale, "lm_dist": round(lm_dist, 5)}
            results.append(row)
            print(f"  {id_str} cn={cn_scale:.1f}: lm_dist={lm_dist:.5f}")

        except Exception as e:
            print(f"  [ERROR] {id_str} cn={cn_scale}: {e}")
            import traceback; traceback.print_exc()

df = pd.DataFrame(results)
print("\n=== CN scale별 lm_dist 평균 ===")
if not df.empty:
    print(df.groupby("cn_scale")["lm_dist"].mean().to_string())
df.to_csv(OUTPUT_DIR / "results.csv", index=False)
print("\n스윕 완료 → outputs/flux/results.csv")

# ── Cell 5: 시각화 ─────────────────────────────────────────────────────────────
SHOW_CN = [CN_SCALES[0], CN_SCALES[-1]]
n_ids   = len(TEST_IDS)
n_cols  = 2 + len(SHOW_CN)

fig, axes = plt.subplots(n_ids, n_cols, figsize=(3.5 * n_cols, 3.5 * n_ids))

for ri, id_str in enumerate(TEST_IDS):
    key = pick_constraint_key(id_str)
    if key is None:
        continue
    coords   = constraints_all[key]["mediapipe_landmarks_facemesh"]
    pose_img = render_facemesh_as_pose(coords, canvas_size=IMG_SIZE)

    ref_paths = sorted(DATA_DIR.glob(f"{id_str}_*.png"))
    ref_paths = [p for p in ref_paths if not any(k in p.name for k in ["_map","_mesh","depth"])]
    ref_img = Image.open(ref_paths[0]).resize((IMG_SIZE, IMG_SIZE)) if ref_paths else None

    axes[ri][0].imshow(ref_img if ref_img else np.zeros((IMG_SIZE, IMG_SIZE, 3)))
    axes[ri][0].set_title(f"{id_str}\n참조", fontsize=9)
    axes[ri][0].axis("off")
    axes[ri][1].imshow(pose_img)
    axes[ri][1].set_title("Pose KP", fontsize=9)
    axes[ri][1].axis("off")

    for ci, cn_scale in enumerate(SHOW_CN):
        p = OUTPUT_DIR / id_str / f"cn{cn_scale:.1f}.png"
        if not df.empty:
            row = df[(df["id_str"] == id_str) & (df["cn_scale"] == cn_scale)]
            lmd = row.iloc[0]["lm_dist"] if not row.empty else float("nan")
        else:
            lmd = float("nan")
        if p.exists():
            axes[ri][2 + ci].imshow(Image.open(p))
            axes[ri][2 + ci].set_title(f"CN={cn_scale}\nlm={lmd:.4f}", fontsize=8)
        else:
            axes[ri][2 + ci].set_title(f"CN={cn_scale}\n(없음)", fontsize=8)
        axes[ri][2 + ci].axis("off")

plt.suptitle(
    "FLUX.1-dev + Pose ControlNet (Shakker-Labs Union-Pro-2.0)\n"
    "랜드마크 제약조건 준수 테스트",
    fontsize=11
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "flux_lm_test.png", dpi=120, bbox_inches="tight")
plt.close()
print("시각화 저장 완료: outputs/flux/flux_lm_test.png")
