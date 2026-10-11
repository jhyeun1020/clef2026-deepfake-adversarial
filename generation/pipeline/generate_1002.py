"""
1002장 전체 이미지 생성 스크립트
A+H2 파라미터: cn=0.85, iw=0.8, guidance=3.5, end=0.65, seed=42
프롬프트: "candid photo of a single person, taken with a phone camera, dim indoor lighting, plain background"

출력: app/input/res/Images/ID_X_GT_frameN.png (512×512)

사용법:
  python generate_1002.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json, sys, time
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
import mediapipe as mp

# ── 설정 ──
DTYPE    = torch.bfloat16
SEED     = 42
IMG_SIZE = 512
HF_TOKEN = os.environ.get("HF_TOKEN")  # set via environment variable
BASE_DIR = Path("/home/student/GEN_AI_GTG")

CONSTRAINTS_PATH = BASE_DIR / "images/files/generation_constraints.json"
CROPPED_DIR      = BASE_DIR / "data/gold_training_set/cropped"
OUTPUT_DIR       = BASE_DIR / "app/input/res/Images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMS = dict(
    cn=0.85, iw=0.8, guidance=3.5, end=0.65, seed=42,
    prompt="candid photo of a single person, taken with a phone camera, dim indoor lighting, plain background",
)

with open(CONSTRAINTS_PATH) as f:
    constraints_all = json.load(f)

print(f"Total keys: {len(constraints_all)}")
print(f"Output: {OUTPUT_DIR}")

# ── 이미 생성된 파일 체크 (이어하기 지원) ──
existing = set(p.stem for p in OUTPUT_DIR.glob("*.png"))
todo_keys = [k for k in sorted(constraints_all.keys()) if k not in existing]
print(f"이미 생성: {len(existing)}장, 남은: {len(todo_keys)}장")

if len(todo_keys) == 0:
    print("모든 이미지가 이미 생성되어 있습니다.")
    sys.exit(0)

# ── FLUX + ControlNet 파이프라인 로드 ──
from diffusers import FluxControlNetPipeline, FluxControlNetModel

print("\nControlNet 로드 중...")
controlnet = FluxControlNetModel.from_pretrained(
    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
    torch_dtype=DTYPE, token=HF_TOKEN)

print("FLUX.1-dev 파이프라인 로드 중...")
pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet, torch_dtype=DTYPE, token=HF_TOKEN)
pipe.enable_model_cpu_offload(gpu_id=0)
print("파이프라인 로드 완료")

# ── PuLID 로드 + Hook 등록 ──
sys.path.insert(0, str(BASE_DIR / 'pulid_pkg'))
os.environ['HF_TOKEN'] = HF_TOKEN

from huggingface_hub import hf_hub_download
from pulid.pipeline_flux import PuLIDPipeline

ckpt_path = hf_hub_download(
    'guozinan/PuLID', 'pulid_flux_v0.9.1.safetensors',
    local_dir=str(BASE_DIR / 'models'), token=HF_TOKEN)
pulid_model = PuLIDPipeline(pipe.transformer, device='cuda:0', weight_dtype=DTYPE)
pulid_model.load_pretrain(pretrain_path=ckpt_path)

double_interval = pipe.transformer.pulid_double_interval
single_interval = pipe.transformer.pulid_single_interval
n_double = len(pipe.transformer.transformer_blocks)
n_double_ca = sum(1 for i in range(n_double) if i % double_interval == 0)

_PULID_STATE = {'id_emb': None, 'id_weight': 1.0}

def make_double_hook(block_idx):
    def hook_fn(module, input, output):
        if _PULID_STATE['id_emb'] is None or block_idx % double_interval != 0:
            return output
        ca_idx = block_idx // double_interval
        if ca_idx >= len(pipe.transformer.pulid_ca): return output
        enc_hs, hs = output
        ca = pipe.transformer.pulid_ca[ca_idx]
        hs = hs + _PULID_STATE['id_weight'] * ca(_PULID_STATE['id_emb'].to(hs.dtype), hs)
        return enc_hs, hs
    return hook_fn

def make_single_hook(block_idx):
    def hook_fn(module, input, output):
        if _PULID_STATE['id_emb'] is None or block_idx % single_interval != 0:
            return output
        ca_idx = n_double_ca + block_idx // single_interval
        if ca_idx >= len(pipe.transformer.pulid_ca): return output
        enc_hs, hs = output
        ca = pipe.transformer.pulid_ca[ca_idx]
        hs = hs + _PULID_STATE['id_weight'] * ca(_PULID_STATE['id_emb'].to(hs.dtype), hs)
        return enc_hs, hs
    return hook_fn

for i, block in enumerate(pipe.transformer.transformer_blocks):
    block.register_forward_hook(make_double_hook(i))
for i, block in enumerate(pipe.transformer.single_transformer_blocks):
    block.register_forward_hook(make_single_hook(i))

print("PuLID Hook 등록 완료")

# ── InsightFace (참조 이미지 정렬용) ──
from insightface.app import FaceAnalysis
face_app = FaceAnalysis(name='buffalo_s', root=str(BASE_DIR / 'images/models'),
                        providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1)

# ── 포즈 렌더링 함수 (experiment_evasion.ipynb Cell 54 원본) ──
MP_TO_68_FIXED = [
    234,  93, 132,  58, 172, 136, 150, 149, 152,
    377, 378, 379, 365, 397, 288, 323, 454,
    70,  63, 105,  66, 107, 336, 296, 334, 293, 300,
    168, 197, 195,   5,  98,  97,   2, 326, 327,
     33, 246, 161, 160, 159, 158, 362, 398, 384, 385, 386, 387,
     61, 185,  40,  39,  37,   0, 267, 269, 270, 409, 291, 306,
     78, 191,  80,  81,  82, 312, 311, 310,
]
_LIPS_FM = set()
for _conn in mp.solutions.face_mesh.FACEMESH_LIPS:
    _LIPS_FM.update(_conn)

_MP33_TO_OP18 = [0, None, 12, 14, 16, 11, 13, 15, 24, 26, 28, 23, 25, 27, 5, 2, 8, 7]
_OPJ_COLORS = [
    (255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),(0,255,0),
    (0,255,85),(0,255,170),(0,255,255),(0,170,255),(0,85,255),(0,0,255),
    (85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85),
]
_OPB_SEQ = [
    (1,2,(153,0,0)),(1,5,(153,51,0)),(2,3,(153,102,0)),(3,4,(153,153,0)),
    (5,6,(102,153,0)),(6,7,(51,153,0)),(1,8,(0,153,0)),(8,9,(0,153,51)),
    (9,10,(0,153,102)),(1,11,(0,153,153)),(11,12,(0,102,153)),(12,13,(0,51,153)),
    (1,0,(0,0,153)),(0,14,(51,0,153)),(14,16,(102,0,153)),(0,15,(153,0,153)),(15,17,(153,0,102)),
]
_FACE_OP_TO_FM = {0:1, 14:468, 15:473, 16:234, 17:454}

def _mp33_to_op18_px(pose_lm, canvas_size):
    kpts = {}
    for op_idx, mp_idx in enumerate(_MP33_TO_OP18):
        if op_idx == 1:
            x0, y0 = pose_lm[11][0], pose_lm[11][1]
            x1, y1 = pose_lm[12][0], pose_lm[12][1]
            if x0 is None or y0 is None or x1 is None or y1 is None: continue
            x = (x0 + x1) / 2
            y = (y0 + y1) / 2
        elif mp_idx is None: continue
        else:
            x, y = pose_lm[mp_idx][0], pose_lm[mp_idx][1]
            if x is None or y is None: continue
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            kpts[op_idx] = (int(x * canvas_size), int(y * canvas_size))
    return kpts

def _draw_body_and_colored_joints(canvas, pose_lm, facemesh_lm, canvas_size):
    if not pose_lm or len(pose_lm) < 29: return
    kpts = _mp33_to_op18_px(pose_lm, canvas_size)
    sw = max(1, canvas_size // 512); dot_r = max(2, canvas_size // 170)
    if facemesh_lm:
        fm_n = len(facemesh_lm)
        for op_idx, fm_idx in _FACE_OP_TO_FM.items():
            if fm_idx < fm_n:
                x, y = facemesh_lm[fm_idx]
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    kpts[op_idx] = (int(x * canvas_size), int(y * canvas_size))
                elif op_idx in kpts: del kpts[op_idx]
    for i, j, color in _OPB_SEQ:
        if i not in kpts or j not in kpts: continue
        x1,y1=kpts[i]; x2,y2=kpts[j]
        length=((x2-x1)**2+(y2-y1)**2)**0.5
        if length < 1: continue
        angle=np.degrees(np.arctan2(y2-y1,x2-x1))
        cx,cy=(x1+x2)//2,(y1+y2)//2
        cv2.ellipse(canvas,(cx,cy),(max(1,int(length/2)),sw),angle,0,360,color,-1)
    for op_idx, pt in kpts.items():
        color=_OPJ_COLORS[op_idx] if op_idx<len(_OPJ_COLORS) else (255,255,255)
        cv2.circle(canvas, pt, dot_r, color, -1)

def _draw_face_dots_fixed(canvas, facemesh_lm, canvas_size, extra_indices=None):
    if not facemesh_lm: return
    _fm_skip = set(_FACE_OP_TO_FM.values())
    _LOWER_EYE_R = [33,7,163,144,145,153,154,155,133]
    _LOWER_EYE_L = [362,382,381,380,374,373,390,249,263]
    _base = list(dict.fromkeys(MP_TO_68_FIXED + _LOWER_EYE_R + _LOWER_EYE_L))
    if extra_indices: _base = list(dict.fromkeys(_base + list(extra_indices)))
    n = len(facemesh_lm)
    for mi in _base:
        if mi in _fm_skip or mi >= n: continue
        x, y = facemesh_lm[mi]
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            cv2.circle(canvas,(int(x*canvas_size),int(y*canvas_size)),2,(255,255,255),-1)

def render_off_full_lips_fixed(pose_lm, facemesh_lm, canvas_size=512):
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    _draw_face_dots_fixed(canvas, facemesh_lm, canvas_size, extra_indices=_LIPS_FM)
    _draw_body_and_colored_joints(canvas, pose_lm, facemesh_lm, canvas_size)
    return Image.fromarray(canvas)

# ── 참조 이미지 헬퍼 (다양성 기반 선택) ──
def get_diverse_ref_crops(id_str, max_n=20):
    subdir = CROPPED_DIR / id_str
    if not subdir.exists(): return []
    candidates = []
    for p in sorted(subdir.glob('*.png')):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None: continue
        faces = face_app.get(img_bgr)
        if not faces: continue
        emb = faces[0].embedding
        emb = emb / np.linalg.norm(emb)
        candidates.append((float(faces[0].det_score), emb, p))
    if not candidates: return []
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected_indices = [0]
    selected_embs = [candidates[0][1]]
    remaining = list(range(1, len(candidates)))
    while len(selected_indices) < min(max_n, len(candidates)) and remaining:
        best_idx, best_min_dist = None, -1.0
        for r in remaining:
            min_dist = min(1.0 - float(np.dot(candidates[r][1], s)) for s in selected_embs)
            if min_dist > best_min_dist:
                best_min_dist, best_idx = min_dist, r
        selected_indices.append(best_idx)
        selected_embs.append(candidates[best_idx][1])
        remaining.remove(best_idx)
    return [Image.open(candidates[i][2]).convert('RGB') for i in selected_indices]

def get_multi_ref_embedding(ref_imgs, n_refs):
    imgs = ref_imgs[:n_refs]
    embs = []
    for img in imgs:
        emb, _ = pulid_model.get_id_embedding(np.array(img.convert('RGB')), cal_uncond=False)
        embs.append(emb)
    if len(embs) == 1: return embs[0]
    return torch.mean(torch.stack(embs), dim=0)

# ── PuLID 임베딩 캐시 (디스크 저장/로드) ──
EMB_CACHE_PATH = BASE_DIR / "models" / "pulid_emb_cache.pt"

if EMB_CACHE_PATH.exists():
    print(f"\n임베딩 캐시 로드: {EMB_CACHE_PATH}")
    emb_cache = torch.load(str(EMB_CACHE_PATH), map_location='cpu')
    print(f"  {len(emb_cache)}개 ID 로드 완료")
else:
    print("\nPuLID 임베딩 캐시 생성 중...")
    all_ids = sorted(set(k.rsplit('_GT_', 1)[0] for k in constraints_all.keys()))
    emb_cache = {}
    for i, id_str in enumerate(all_ids):
        crops = get_diverse_ref_crops(id_str, max_n=20)
        if not crops:
            print(f"  [경고] {id_str}: 크롭 없음, 건너뜀")
            continue
        emb_cache[id_str] = get_multi_ref_embedding(crops, min(20, len(crops)))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(all_ids)}] 임베딩 완료")
    torch.save(emb_cache, str(EMB_CACHE_PATH))
    print(f"  캐시 저장: {EMB_CACHE_PATH}")

print(f"임베딩 캐시: {len(emb_cache)}개 ID")

# ── 생성 루프 ──
print(f"\n{'='*60}")
print(f"생성 시작: {len(todo_keys)}장")
print(f"파라미터: cn={PARAMS['cn']}, iw={PARAMS['iw']}, guidance={PARAMS['guidance']}, end={PARAMS['end']}, seed={PARAMS['seed']}")
print(f"{'='*60}")

t0 = time.time()
done = 0
errors = []

for key in todo_keys:
    id_str = key.rsplit('_GT_', 1)[0]

    if id_str not in emb_cache:
        errors.append((key, "임베딩 없음"))
        continue

    entry = constraints_all[key]
    facemesh_lm = entry['mediapipe_landmarks_facemesh']
    pose_lm = entry.get('mediapipe_pose', [])

    ctrl_img = render_off_full_lips_fixed(pose_lm, facemesh_lm, canvas_size=IMG_SIZE)

    _PULID_STATE['id_emb'] = emb_cache[id_str]
    _PULID_STATE['id_weight'] = PARAMS['iw']

    try:
        gen = pipe(
            prompt=PARAMS['prompt'],
            control_image=ctrl_img,
            controlnet_conditioning_scale=PARAMS['cn'],
            control_guidance_end=PARAMS['end'],
            num_inference_steps=28,
            guidance_scale=PARAMS['guidance'],
            height=IMG_SIZE, width=IMG_SIZE,
            generator=torch.Generator(device='cpu').manual_seed(PARAMS['seed']),
        ).images[0]
    except Exception as e:
        errors.append((key, str(e)))
        print(f"  [에러] {key}: {e}")
        continue
    finally:
        _PULID_STATE['id_emb'] = None

    gen.save(OUTPUT_DIR / f"{key}.png")
    done += 1

    elapsed = time.time() - t0
    eta = (elapsed / done) * (len(todo_keys) - done) if done > 0 else 0

    if done % 10 == 0 or done == len(todo_keys):
        print(f"  [{done}/{len(todo_keys)}] {key} — {elapsed:.0f}초 경과, ETA {eta/60:.0f}분")

elapsed_total = time.time() - t0
print(f"\n{'='*60}")
print(f"생성 완료: {done}장 / {len(todo_keys)}장")
print(f"총 소요: {elapsed_total/60:.1f}분 ({elapsed_total/done:.1f}초/장)")
print(f"저장: {OUTPUT_DIR}")
if errors:
    print(f"에러: {len(errors)}건")
    for k, e in errors[:5]:
        print(f"  {k}: {e}")
print(f"{'='*60}")
