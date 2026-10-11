"""
전이 탐지기 평가: 공격 후 1002장 (full + crop)
AIDE, DistilDIRE, RECCE, UCF
GPU 1 사용
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys, torch, cv2, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
import mediapipe as mp

os.environ['HF_TOKEN'] = 'REMOVED_TOKEN_USE_ENV_VAR'
DEVICE = torch.device('cuda:0')
BASE_DIR = Path('/home/student/GEN_AI_GTG')
OUT = BASE_DIR / 'outputs/attack_1002_all'

# 기존 결과 로드
df = pd.read_csv(OUT / 'results_adaptive.csv')
all_keys = df['key'].tolist()
N = len(all_keys)
print(f"키: {N}장")

# 이미지 로드
print("이미지 로드...")
final_pils = []
for key in all_keys:
    p = OUT / 'jpeg_q80' / f'{key}.png'
    final_pils.append(Image.open(p).convert('RGB'))

# 얼굴 크롭
print("얼굴 크롭...")
def crop_face_pil(pil_img, margin=0.3):
    img_np = np.array(pil_img.convert('RGB'))
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        res = fm.process(img_np)
    if not res.multi_face_landmarks: return None
    lm = res.multi_face_landmarks[0].landmark
    h, w = img_np.shape[:2]
    xs = [l.x * w for l in lm]; ys = [l.y * h for l in lm]
    x_min, x_max = min(xs), max(xs); y_min, y_max = min(ys), max(ys)
    fw = x_max - x_min; fh = y_max - y_min
    cx = (x_min + x_max) / 2; cy = (y_min + y_max) / 2
    size = max(fw, fh) * (1 + margin)
    x1 = max(0, int(cx - size/2)); y1 = max(0, int(cy - size/2))
    x2 = min(w, int(cx + size/2)); y2 = min(h, int(cy + size/2))
    return Image.fromarray(img_np[y1:y2, x1:x2])

crop_pils = []
for i, key in enumerate(all_keys):
    cp = OUT / 'face_crop' / f'{key}.png'
    if cp.exists():
        crop_pils.append(Image.open(cp).convert('RGB'))
    else:
        cropped = crop_face_pil(final_pils[i])
        crop_pils.append(cropped if cropped else final_pils[i])
    if (i+1) % 200 == 0:
        print(f"  크롭 [{i+1}/{N}]")
print(f"  크롭 완료")

# ── AIDE ──
sys.path.insert(0, str(BASE_DIR / 'AIDE'))
from data.dct import DCT_base_Rec_Module
from models.AIDE import AIDE

dct_module = DCT_base_Rec_Module()
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def prepare_aide_input(pil_img):
    img_t = T.ToTensor()(pil_img.convert('RGB'))
    img_t = F.interpolate(img_t.unsqueeze(0), (256, 256), mode='bilinear', align_corners=False)[0]
    x_minmin, x_maxmax, x_minmin1, x_maxmax1 = dct_module(img_t)
    min1 = normalize(F.interpolate(x_minmin.unsqueeze(0), (256, 256), mode='bilinear', align_corners=False)[0])
    max1 = normalize(F.interpolate(x_maxmax.unsqueeze(0), (256, 256), mode='bilinear', align_corners=False)[0])
    min2 = normalize(F.interpolate(x_minmin1.unsqueeze(0), (256, 256), mode='bilinear', align_corners=False)[0])
    max2 = normalize(F.interpolate(x_maxmax1.unsqueeze(0), (256, 256), mode='bilinear', align_corners=False)[0])
    tokens = normalize(img_t)
    return torch.stack([min1, max1, min2, max2, tokens], dim=0).unsqueeze(0)

AIDE_CKPTS = {
    'aide_progan': str(BASE_DIR / 'AIDE/weights/progan_train.pth'),
    'aide_genimage': str(BASE_DIR / 'AIDE/weights/GenImage_train.pth'),
    'aide_sd14': str(BASE_DIR / 'AIDE/weights/sd14_train.pth'),
}

def score_aide(pils, ckpt_name, ckpt_path):
    print(f"  {ckpt_name} 로드...")
    model = AIDE(resnet_path=None, convnext_path='laion2b_s34b_b82k_augreg_soup')
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'model' in sd: sd = sd['model']
    model.load_state_dict({k.replace('module.',''):v for k,v in sd.items()}, strict=False)
    model.to(DEVICE).eval()
    scores = []
    with torch.no_grad():
        for i, pil in enumerate(pils):
            inp = prepare_aide_input(pil).to(DEVICE)
            logits = model(inp)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            scores.append(prob)
            if (i+1) % 200 == 0:
                print(f"    [{i+1}/{len(pils)}]")
    model.cpu(); del model; torch.cuda.empty_cache()
    return scores

# ── DistilDIRE ──
sys.path.insert(0, str(BASE_DIR / 'DistilDIRE'))
from networks.distill_model import DistilDIRE
from guided_diffusion.compute_dire_eps import dire_get_first_step_noise
from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion, model_and_diffusion_defaults, dict_parse)

WEIGHTS_DIR = BASE_DIR / 'weights'

def score_distildire(pils, label=''):
    print(f"  DistilDIRE 로드... ({label})", flush=True)
    # ADM model for eps computation
    adm_defaults = model_and_diffusion_defaults()
    adm_args = dict(
        clip_denoised=True, use_ddim=True, real_step=0,
        attention_resolutions='32,16,8', class_cond=False, diffusion_steps=1000,
        image_size=256, learn_sigma=True,
        model_path=str(BASE_DIR / 'models/256x256-adm.pt'),
        noise_schedule='linear', num_channels=256, num_head_channels=64,
        num_res_blocks=2, resblock_updown=True, use_fp16=True,
        use_scale_shift_norm=True, timestep_respacing='ddim20')
    adm_full = {**adm_defaults, **adm_args}
    adm_model, diffusion = create_model_and_diffusion(**dict_parse(adm_full, model_and_diffusion_defaults().keys()))
    adm_model.load_state_dict(torch.load(adm_full['model_path'], map_location='cpu'))
    adm_model.to(DEVICE).eval()
    if adm_full.get('use_fp16'): adm_model.convert_to_fp16()

    dd_ckpt = torch.load(str(WEIGHTS_DIR / 'distildire.pth'), map_location='cpu')
    distildire = DistilDIRE('cuda').to(DEVICE)
    distildire.load_state_dict({k.replace('module.',''):v for k,v in dd_ckpt.get('model', dd_ckpt).items()})
    distildire.eval()

    dire_transform = T.Compose([T.Resize(256, antialias=True), T.CenterCrop((256, 256))])
    scores = []
    with torch.no_grad():
        for i, pil in enumerate(pils):
            import torchvision.transforms.functional as TFF
            img = TFF.to_tensor(pil.convert('RGB'))
            img = dire_transform(img).to(DEVICE) * 2 - 1
            img = img.unsqueeze(0)
            eps = dire_get_first_step_noise(img, adm_model, diffusion, adm_full, 'cuda')
            score = float(distildire(img, eps)['logit'].sigmoid().item())
            scores.append(score)
            if (i+1) % 100 == 0: print(f"    [{i+1}/{len(pils)}]", flush=True)
    distildire.cpu(); adm_model.cpu()
    del distildire, adm_model, diffusion
    torch.cuda.empty_cache()
    return scores

# ── RECCE ──
from timm.models import xception as _timm_xc

class RECCE_Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.backbone = _timm_xc(pretrained=False, num_classes=2)
    def forward(s, x):
        return s.backbone(x)

def score_recce(pils, label=''):
    print(f"  RECCE 로드... ({label})")
    recce = RECCE_Net()
    sd = torch.load(str(WEIGHTS_DIR / 'recce_best.pth'), map_location='cpu', weights_only=False)
    recce.load_state_dict({k.replace('module.',''):v for k,v in sd.items()}, strict=False)
    recce.to(DEVICE).eval()
    rc_tf = T.Compose([T.Resize((299,299)), T.ToTensor(), T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    scores = []
    with torch.no_grad():
        for pil in pils:
            inp = rc_tf(pil.convert('RGB')).unsqueeze(0).to(DEVICE)
            logits = recce(inp)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            scores.append(prob)
    recce.cpu(); del recce; torch.cuda.empty_cache()
    return scores

# ── UCF ──
class UCFModel(nn.Module):
    def __init__(s):
        super().__init__()
        s.backbone = _timm_xc(pretrained=False, num_classes=2)
    def forward(s, x):
        return s.backbone(x)

def score_ucf(pils, label=''):
    print(f"  UCF 로드... ({label})")
    ucf = UCFModel()
    sd = torch.load(str(WEIGHTS_DIR / 'ucf_best.pth'), map_location='cpu', weights_only=False)
    ucf.load_state_dict({k.replace('module.',''):v for k,v in sd.items()}, strict=False)
    ucf.to(DEVICE).eval()
    uc_tf = T.Compose([T.Resize((299,299)), T.ToTensor(), T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    scores = []
    with torch.no_grad():
        for pil in pils:
            inp = uc_tf(pil.convert('RGB')).unsqueeze(0).to(DEVICE)
            logits = ucf(inp)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            scores.append(prob)
    ucf.cpu(); del ucf; torch.cuda.empty_cache()
    return scores

# ── 실행 ──
print("\n전이 탐지기 평가 시작...", flush=True)
transfer_results = {}

for ckpt_name, ckpt_path in AIDE_CKPTS.items():
    transfer_results[f'{ckpt_name}_full'] = score_aide(final_pils, ckpt_name, ckpt_path)
    transfer_results[f'{ckpt_name}_crop'] = score_aide(crop_pils, ckpt_name, ckpt_path)
    # 중간 저장
    for col, vals in transfer_results.items():
        df[col] = [round(v, 4) for v in vals]
    df.to_csv(OUT / 'results_adaptive.csv', index=False)
    print(f"  {ckpt_name} 중간 저장 완료", flush=True)

print("\nDistilDIRE...", flush=True)
transfer_results['distildire_full'] = score_distildire(final_pils, 'full')
transfer_results['distildire_crop'] = score_distildire(crop_pils, 'crop')

print("\nRECCE...")
transfer_results['recce_full'] = score_recce(final_pils, 'full')
transfer_results['recce_crop'] = score_recce(crop_pils, 'crop')

print("\nUCF...")
transfer_results['ucf_full'] = score_ucf(final_pils, 'full')
transfer_results['ucf_crop'] = score_ucf(crop_pils, 'crop')

# CSV에 추가
for col, vals in transfer_results.items():
    df[col] = [round(v, 4) for v in vals]
df.to_csv(OUT / 'results_adaptive.csv', index=False)

# 결과 출력
transfer_names = ['aide_progan','aide_genimage','aide_sd14','distildire','recce','ucf']

print(f"\n{'='*80}")
print(f"전이 탐지기 (full + crop) ({N}장)")
print(f"{'='*80}")
print(f"{'탐지기':15s} {'full_avg':>9s} {'crop_avg':>9s}  {'f회피':>7s} {'c회피':>7s}")
print('-' * 70)

for det in transfer_names:
    f_scores = transfer_results[f'{det}_full']
    c_scores = transfer_results[f'{det}_crop']
    f_ev = sum(1 for s in f_scores if s < 0.5)
    c_ev = sum(1 for s in c_scores if s < 0.5)
    print(f"  {det:15s} {np.mean(f_scores):9.4f} {np.mean(c_scores):9.4f}  {f_ev:>4d}/{N} {c_ev:>4d}/{N}")

print(f"\n결과: {OUT}/results_adaptive.csv")
print("완료 ✓")
