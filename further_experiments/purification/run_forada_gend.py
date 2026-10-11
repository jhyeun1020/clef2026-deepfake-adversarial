"""
import os
Re-run ForAda and GenD with patches:
- ForAda: CLIP loader hardcoded /data/cuixinjie/weights → patch to user cache
- GenD: torch.compiler.is_compiling missing in torch 2.2 → stub it
"""
import os, sys, json, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter
import cv2

# === GenD compatibility patch (must run BEFORE importing GenD) ===
if not hasattr(torch.compiler, 'is_compiling'):
    torch.compiler.is_compiling = lambda: False

os.environ.setdefault('HF_TOKEN', 'REMOVED_TOKEN_USE_ENV_VAR')

BASE = Path('/home/s20204116/GEN_AI_GTG')
OUT = BASE / 'detection/purification_comparison'
DEVICE = torch.device('cuda:0')
LOG = OUT / 'run_forada_gend.log'

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def host(p):
    return Path(str(p).replace('/home/student/', '/home/s20204116/'))

# Build same test data
log("Building test data...")
rng = np.random.default_rng(42)
with open(BASE / 'detection/gaussian_fitting/data_split.json') as f:
    split = json.load(f)
real_paths = [host(p) for p in split['real_holdout'][:100]]
clean_all = sorted((BASE / 'generation/outputs/final_1002').glob('*.png'))
clean_paths = [clean_all[i] for i in sorted(rng.choice(len(clean_all), 100, replace=False))]

ADV_BASE = BASE / 'detection/multi_attack/images'
def collect(target):
    dirs = sorted([d for d in ADV_BASE.glob(f'{target}__*__pgd_eps4') if not d.name.endswith('jpeg80')])
    imgs = []
    for d in dirs:
        imgs.extend(sorted(d.glob('*.png')))
    return imgs

forada_pool = collect('forada')
adv_forada_paths = [forada_pool[i] for i in sorted(rng.choice(len(forada_pool), 100, replace=False))]
siglip_pool = collect('siglip')
adv_siglip_paths = [siglip_pool[i] for i in sorted(rng.choice(len(siglip_pool), 100, replace=False))]

def apply_jpeg_q80(pil_img):
    arr = np.array(pil_img); bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Image.fromarray(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))

def load_pil(p):
    return Image.open(p).convert('RGB').resize((256,256), Image.LANCZOS)

real_pils       = [load_pil(p) for p in real_paths]
clean_pils      = [load_pil(p) for p in clean_paths]
adv_forada_pils = [load_pil(p) for p in adv_forada_paths]
adv_siglip_pils = [load_pil(p) for p in adv_siglip_paths]
jpeg_q80_pils   = [apply_jpeg_q80(p) for p in real_pils]

def med3(pils): return [p.filter(ImageFilter.MedianFilter(size=3)) for p in pils]

CATEGORIES = [
    ('real',       real_pils,       med3(real_pils),       real_paths),
    ('clean_fake', clean_pils,      med3(clean_pils),      clean_paths),
    ('jpeg_q80',   jpeg_q80_pils,   med3(jpeg_q80_pils),   real_paths),
    ('adv_forada', adv_forada_pils, med3(adv_forada_pils), adv_forada_paths),
    ('adv_siglip', adv_siglip_pils, med3(adv_siglip_pils), adv_siglip_paths),
]
log(f"Total: {sum(len(c[1]) for c in CATEGORIES)}")

SAFE_EPS = 1e-6
def logit(p):
    p = np.clip(p, SAFE_EPS, 1-SAFE_EPS)
    return np.log(p/(1-p))

def measure(name, predict):
    log(f"  Running {name}...")
    rows = []
    for cat, pils, m3, paths in CATEGORIES:
        t0 = time.time()
        po = predict(pils)
        pm = predict(m3)
        for i in range(len(pils)):
            rows.append({'category':cat, 'image_id':Path(paths[i]).stem,
                         'prob_orig':float(po[i]), 'prob_med3':float(pm[i]),
                         'delta_logit':float(logit(pm[i])-logit(po[i]))})
        log(f"    {cat} ({len(pils)}): {time.time()-t0:.1f}s, p_orig={np.mean(po):.3f} p_med3={np.mean(pm):.3f}")
    return pd.DataFrame(rows)


# ============== ForAda with CLIP cache patch ==============
def run_forada():
    log("\n=== ForAda (with CLIP cache patch) ===")
    FA_ROOT = BASE / 'detection/ForensicsAdapter'
    sys.path.insert(0, str(FA_ROOT))

    # Patch CLIP load to use writable cache
    USER_CLIP_CACHE = str(Path.home() / '.cache/clip')
    os.makedirs(USER_CLIP_CACHE, exist_ok=True)
    from model.clip import clip as fa_clip
    _orig = fa_clip.load
    def _patched(name, device='cuda', jit=False, download_root=None):
        log(f"  CLIP load patched: {download_root} → {USER_CLIP_CACHE}")
        return _orig(name, device=device, jit=jit, download_root=USER_CLIP_CACHE)
    fa_clip.load = _patched

    from model.ds import DS as ForAda_DS
    fa_model = ForAda_DS(clip_name='ViT-L/14', adapter_vit_name='vit_tiny_patch16_224',
        num_quires=128, fusion_map={0:0,1:1,2:8,3:15},
        mlp_dim=256, mlp_out_dim=128, head_num=16, device=str(DEVICE))
    fa_ckpt = torch.load(str(FA_ROOT / 'weights/ckpt_best.pth'), map_location='cpu')
    fa_model.load_state_dict(fa_ckpt, strict=False)
    fa_model.to(DEVICE).eval()
    fa_mean = torch.tensor([0.48145466,0.4578275,0.40821073]).view(1,3,1,1).to(DEVICE)
    fa_std  = torch.tensor([0.26862954,0.26130258,0.27577711]).view(1,3,1,1).to(DEVICE)

    @torch.no_grad()
    def predict(pils, bs=16):
        probs = []
        for i in range(0, len(pils), bs):
            chunk = pils[i:i+bs]
            x = torch.stack([T.ToTensor()(p) for p in chunk]).to(DEVICE)
            xn = (x - fa_mean) / fa_std
            B = xn.shape[0]
            dd = {'image': xn, 'label': torch.zeros(B,dtype=torch.long,device=DEVICE),
                  'mask': None, 'landmark': None, 'if_boundary': torch.ones(B,256,device=DEVICE)}
            pred = fa_model(dd, inference=True)
            probs.extend(pred['prob'].cpu().numpy().tolist())
            fa_model.prob=[]; fa_model.label=[]; fa_model.correct=0; fa_model.total=0
        return probs

    df = measure('ForAda', predict)
    df.to_csv(OUT / 'forada_results.csv', index=False)
    log(f"  Saved forada_results.csv ({len(df)} rows)")
    fa_model.cpu(); del fa_model
    torch.cuda.empty_cache(); gc.collect()


# ============== GenD with torch.compiler patch ==============
def run_gend():
    log("\n=== GenD_DINOv3 (with torch.compiler patch) ===")
    sys.path.insert(0, str(BASE / 'detection/GenD'))
    from src.hf.modeling_gend import GenD
    m = GenD.from_pretrained('yermandy/GenD_DINOv3_L')
    m = m.to(DEVICE).eval()

    @torch.no_grad()
    def predict(pils, bs=8):
        probs = []
        for i in range(0, len(pils), bs):
            chunk = pils[i:i+bs]
            tensors = torch.stack([m.feature_extractor.preprocess(p) for p in chunk]).to(DEVICE)
            logits = m(tensors)
            p = torch.softmax(logits, dim=-1)
            # GenD: index 1 = fake (verify by inspecting on real images)
            probs.extend(p[:, 1].cpu().numpy().tolist())
        return probs

    df = measure('GenD_DINOv3', predict)
    df.to_csv(OUT / 'gend_results.csv', index=False)
    log(f"  Saved gend_results.csv ({len(df)} rows)")
    m.cpu(); del m
    torch.cuda.empty_cache(); gc.collect()


run_forada()
run_gend()
log("DONE")
