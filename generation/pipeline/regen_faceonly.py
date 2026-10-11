"""
ID_88_GT_frame3 재생성: 포즈맵에서 얼굴 랜드마크만 사용 (body skeleton 제거)
GPU 2 사용
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json, sys, time
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
import mediapipe as mp

DTYPE = torch.bfloat16
IMG_SIZE = 512
HF_TOKEN = os.environ.get("HF_TOKEN")  # set via environment variable
BASE_DIR = Path("/home/student/GEN_AI_GTG")

KEY = "ID_207_GT_frame3"

with open(BASE_DIR / "images/files/generation_constraints.json") as f:
    constraints_all = json.load(f)

FIXED_CONSTRAINTS_PATH = BASE_DIR / "models/fixed_constraints_4.json"
if FIXED_CONSTRAINTS_PATH.exists():
    with open(FIXED_CONSTRAINTS_PATH) as f:
        fixed_constraints = json.load(f)
else:
    fixed_constraints = {}

# ── 렌더링 함수 (얼굴만, body 없음) ──
_LIPS_FM = set()
for _conn in mp.solutions.face_mesh.FACEMESH_LIPS: _LIPS_FM.update(_conn)
MP_TO_68_FIXED = [234,93,132,58,172,136,150,149,152,377,378,379,365,397,288,323,454,
    70,63,105,66,107,336,296,334,293,300,168,197,195,5,98,97,2,326,327,
    33,246,161,160,159,158,362,398,384,385,386,387,
    61,185,40,39,37,0,267,269,270,409,291,306,78,191,80,81,82,312,311,310]
_FACE_OP_TO_FM = {0:1, 14:468, 15:473, 16:234, 17:454}
_OPJ_COLORS = [(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),
    (0,255,0),(0,255,85),(0,255,170),(0,255,255),(0,170,255),(0,85,255),
    (0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85)]

def _draw_face(canvas, facemesh_lm, cs, extra=None):
    if not facemesh_lm: return
    _skip = set(_FACE_OP_TO_FM.values())
    _LR = [33,7,163,144,145,153,154,155,133]
    _LL = [362,382,381,380,374,373,390,249,263]
    _base = list(dict.fromkeys(MP_TO_68_FIXED + _LR + _LL))
    if extra: _base = list(dict.fromkeys(_base + list(extra)))
    for mi in _base:
        if mi in _skip or mi >= len(facemesh_lm): continue
        x, y = facemesh_lm[mi]
        if x is not None and y is not None and 0<=x<=1 and 0<=y<=1:
            cv2.circle(canvas, (int(x*cs), int(y*cs)), 2, (255,255,255), -1)

def _draw_face_joints(canvas, facemesh_lm, cs):
    """얼굴 관절 5개(코끝, 홍채×2, 귀×2)를 컬러 dot으로"""
    dr = max(3, cs // 170)
    for oi, fi in _FACE_OP_TO_FM.items():
        if fi < len(facemesh_lm):
            x, y = facemesh_lm[fi]
            if x is not None and y is not None and 0<=x<=1 and 0<=y<=1:
                c = _OPJ_COLORS[oi] if oi < len(_OPJ_COLORS) else (255,255,255)
                cv2.circle(canvas, (int(x*cs), int(y*cs)), dr, c, -1)

def render_face_only(facemesh_lm, canvas_size=512):
    """얼굴 랜드마크만 렌더링 (body skeleton 없음)"""
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    _draw_face(canvas, facemesh_lm, canvas_size, extra=_LIPS_FM)
    _draw_face_joints(canvas, facemesh_lm, canvas_size)
    return Image.fromarray(canvas)


# ── 파이프라인 로드 ──
print("파이프라인 로드 중...")
from diffusers import FluxControlNetPipeline, FluxControlNetModel
controlnet = FluxControlNetModel.from_pretrained(
    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0", torch_dtype=DTYPE, token=HF_TOKEN)
pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", controlnet=controlnet, torch_dtype=DTYPE, token=HF_TOKEN)
pipe.enable_model_cpu_offload(gpu_id=0)

sys.path.insert(0, str(BASE_DIR / 'pulid_pkg'))
os.environ['HF_TOKEN'] = HF_TOKEN
from huggingface_hub import hf_hub_download
from pulid.pipeline_flux import PuLIDPipeline

ckpt_path = hf_hub_download('guozinan/PuLID', 'pulid_flux_v0.9.1.safetensors',
    local_dir=str(BASE_DIR / 'models'), token=HF_TOKEN)
pulid_model = PuLIDPipeline(pipe.transformer, device='cuda:0', weight_dtype=DTYPE)
pulid_model.load_pretrain(pretrain_path=ckpt_path)

# Hook
double_interval = pipe.transformer.pulid_double_interval
single_interval = pipe.transformer.pulid_single_interval
n_double = len(pipe.transformer.transformer_blocks)
n_double_ca = sum(1 for i in range(n_double) if i % double_interval == 0)
_PULID_STATE = {'id_emb': None, 'id_weight': 1.0}

def make_double_hook(block_idx):
    def hook_fn(module, input, output):
        if _PULID_STATE['id_emb'] is None or block_idx % double_interval != 0: return output
        ca_idx = block_idx // double_interval
        if ca_idx >= len(pipe.transformer.pulid_ca): return output
        enc_hs, hs = output
        ca = pipe.transformer.pulid_ca[ca_idx].to(hs.device)
        id_emb = _PULID_STATE['id_emb'].to(dtype=hs.dtype, device=hs.device)
        hs = hs + _PULID_STATE['id_weight'] * ca(id_emb, hs)
        return enc_hs, hs
    return hook_fn

def make_single_hook(block_idx):
    def hook_fn(module, input, output):
        if _PULID_STATE['id_emb'] is None or block_idx % single_interval != 0: return output
        ca_idx = n_double_ca + block_idx // single_interval
        if ca_idx >= len(pipe.transformer.pulid_ca): return output
        enc_hs, hs = output
        ca = pipe.transformer.pulid_ca[ca_idx].to(hs.device)
        id_emb = _PULID_STATE['id_emb'].to(dtype=hs.dtype, device=hs.device)
        hs = hs + _PULID_STATE['id_weight'] * ca(id_emb, hs)
        return enc_hs, hs
    return hook_fn

for i, b in enumerate(pipe.transformer.transformer_blocks): b.register_forward_hook(make_double_hook(i))
for i, b in enumerate(pipe.transformer.single_transformer_blocks): b.register_forward_hook(make_single_hook(i))

emb_cache = torch.load(str(BASE_DIR / 'models/pulid_emb_cache.pt'), map_location='cpu')
print("파이프라인 로드 완료")

# ── 생성 ──
PARAMS = dict(
    cn=0.85, iw=0.85, guidance=4.5, end=0.65,
    prompt="a person in front of a light blue painted wall background, natural lighting, casual photo"
)
SEEDS = [42, 2024, 7777, 314, 999, 5555, 4321, 7070]

id_str = KEY.rsplit('_GT_', 1)[0]
entry = fixed_constraints.get(KEY, constraints_all[KEY])
facemesh_lm = entry['mediapipe_landmarks_facemesh']

# 얼굴만 포즈맵
ctrl_img = render_face_only(facemesh_lm, canvas_size=IMG_SIZE)
ctrl_img.save(BASE_DIR / f"outputs/{KEY}_faceonly_posemap.png")
print(f"포즈맵 저장: outputs/{KEY}_faceonly_posemap.png")

OUT_DIR = BASE_DIR / "outputs/faceonly_regen"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for seed in SEEDS:
    print(f"  Seed {seed} 생성 중...", end=" ", flush=True)
    _PULID_STATE['id_emb'] = emb_cache[id_str]
    _PULID_STATE['id_weight'] = PARAMS['iw']
    try:
        gen = pipe(
            prompt=PARAMS['prompt'], control_image=ctrl_img,
            controlnet_conditioning_scale=PARAMS['cn'],
            control_guidance_end=PARAMS['end'],
            num_inference_steps=28, guidance_scale=PARAMS['guidance'],
            height=IMG_SIZE, width=IMG_SIZE,
            generator=torch.Generator(device='cpu').manual_seed(seed),
        ).images[0]
    except Exception as e:
        print(f"에러: {e}")
        continue
    finally:
        _PULID_STATE['id_emb'] = None

    gen.save(OUT_DIR / f"{KEY}_seed{seed}.png")
    print(f"저장 완료")

print(f"\n완료! 결과: {OUT_DIR}")
print("완료 ✓")
