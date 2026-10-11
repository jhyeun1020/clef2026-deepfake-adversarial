"""
import os
Purification adversarial detection — 4-model comparison.
Models: SigLIP+DINOv2, GenD_DINOv3, ForAda, EFFORT
Data: real(100) + clean_fake(100) + jpeg_q80(100) + adv_forada(100) + adv_siglip(100)
Method: median3 filter, delta_logit = logit(p_med3) - logit(p_orig)
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

os.environ.setdefault('HF_TOKEN', 'REMOVED_TOKEN_USE_ENV_VAR')

BASE = Path('/home/s20204116/GEN_AI_GTG')
OUT = BASE / 'detection/purification_comparison'
OUT.mkdir(exist_ok=True)
DEVICE = torch.device('cuda:0')
WEIGHTS = BASE / 'generation/weights'
LOG = OUT / 'run.log'

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def host(p):
    return Path(str(p).replace('/home/student/', '/home/s20204116/'))

# ============================================================
# Phase 1: Build test data
# ============================================================
log("=" * 60)
log("Building test data (5 categories × 100 = 500)")
log("=" * 60)

rng = np.random.default_rng(42)

# Real (100) from holdout split
with open(BASE / 'detection/gaussian_fitting/data_split.json') as f:
    split = json.load(f)
real_paths = [host(p) for p in split['real_holdout'][:100]]

# Clean fake (100) from generation/outputs/final_1002
clean_all = sorted((BASE / 'generation/outputs/final_1002').glob('*.png'))
clean_idx = sorted(rng.choice(len(clean_all), 100, replace=False))
clean_paths = [clean_all[i] for i in clean_idx]

# Adv ForAda target (100) — sample 100 from forada__*__pgd_eps4 dirs
ADV_BASE = BASE / 'detection/multi_attack/images'
forada_dirs = sorted([d for d in ADV_BASE.glob('forada__*__pgd_eps4') if not d.name.endswith('jpeg80')])
forada_imgs_all = []
for d in forada_dirs:
    forada_imgs_all.extend(sorted(d.glob('*.png')))
log(f"  ForAda adv pool: {len(forada_imgs_all)} from {len(forada_dirs)} dirs")
forada_idx = sorted(rng.choice(len(forada_imgs_all), 100, replace=False))
adv_forada_paths = [forada_imgs_all[i] for i in forada_idx]

# Adv SigLIP target (100)
siglip_dirs = sorted([d for d in ADV_BASE.glob('siglip__*__pgd_eps4') if not d.name.endswith('jpeg80')])
siglip_imgs_all = []
for d in siglip_dirs:
    siglip_imgs_all.extend(sorted(d.glob('*.png')))
log(f"  SigLIP adv pool: {len(siglip_imgs_all)} from {len(siglip_dirs)} dirs")
siglip_idx = sorted(rng.choice(len(siglip_imgs_all), 100, replace=False))
adv_siglip_paths = [siglip_imgs_all[i] for i in siglip_idx]

# Verify all paths exist
def check(paths, name):
    missing = [p for p in paths if not p.exists()]
    if missing:
        log(f"  WARN {name}: {len(missing)} missing, e.g. {missing[0]}")
    else:
        log(f"  ✓ {name}: {len(paths)} all exist")

check(real_paths, 'real')
check(clean_paths, 'clean_fake')
check(adv_forada_paths, 'adv_forada')
check(adv_siglip_paths, 'adv_siglip')


def apply_jpeg_q80(pil_img):
    arr = np.array(pil_img); bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Image.fromarray(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))

def load_pil_256(p):
    return Image.open(p).convert('RGB').resize((256, 256), Image.LANCZOS)

# Load all PILs at 256×256
log("Loading PIL images at 256×256...")
real_pils       = [load_pil_256(p) for p in real_paths]
clean_pils      = [load_pil_256(p) for p in clean_paths]
adv_forada_pils = [load_pil_256(p) for p in adv_forada_paths]
adv_siglip_pils = [load_pil_256(p) for p in adv_siglip_paths]
jpeg_q80_pils   = [apply_jpeg_q80(p) for p in real_pils]

# Apply median3 filter to each
log("Applying median3 filter...")
def med3(pils):
    return [p.filter(ImageFilter.MedianFilter(size=3)) for p in pils]

real_med3       = med3(real_pils)
clean_med3      = med3(clean_pils)
jpeg_med3       = med3(jpeg_q80_pils)
adv_forada_med3 = med3(adv_forada_pils)
adv_siglip_med3 = med3(adv_siglip_pils)

CATEGORIES = [
    ('real',       real_pils,       real_med3,       real_paths),
    ('clean_fake', clean_pils,      clean_med3,      clean_paths),
    ('jpeg_q80',   jpeg_q80_pils,   jpeg_med3,       real_paths),  # base = real, jpeg applied
    ('adv_forada', adv_forada_pils, adv_forada_med3, adv_forada_paths),
    ('adv_siglip', adv_siglip_pils, adv_siglip_med3, adv_siglip_paths),
]

log(f"Total samples: {sum(len(c[1]) for c in CATEGORIES)}")


# ============================================================
# Helpers
# ============================================================
SAFE_EPS = 1e-6
def logit(p):
    p = np.clip(p, SAFE_EPS, 1 - SAFE_EPS)
    return np.log(p / (1 - p))


def measure_model(model_name, predict_pils_fn):
    """Run prob_fake on (orig, med3) for all categories. Return DataFrame."""
    log(f"  Running {model_name} on {sum(len(c[1]) for c in CATEGORIES)} samples × 2 (orig + med3)...")
    rows = []
    for cat, pils, med3_pils, paths in CATEGORIES:
        t0 = time.time()
        p_orig = predict_pils_fn(pils)
        p_med3 = predict_pils_fn(med3_pils)
        for i in range(len(pils)):
            rows.append({
                'category': cat,
                'image_id': Path(paths[i]).stem,
                'prob_orig': float(p_orig[i]),
                'prob_med3': float(p_med3[i]),
                'delta_logit': float(logit(p_med3[i]) - logit(p_orig[i])),
            })
        log(f"    {cat} ({len(pils)}): {time.time()-t0:.1f}s, mean p_orig={np.mean(p_orig):.3f}, p_med3={np.mean(p_med3):.3f}")
    return pd.DataFrame(rows)


# ============================================================
# Model 1: ForAda
# ============================================================
def run_forada():
    log("Loading ForAda...")
    FA_ROOT = BASE / 'detection/ForensicsAdapter'
    sys.path.insert(0, str(FA_ROOT))
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
            dd = {'image': xn, 'label': torch.zeros(B, dtype=torch.long, device=DEVICE),
                  'mask': None, 'landmark': None, 'if_boundary': torch.ones(B, 256, device=DEVICE)}
            pred = fa_model(dd, inference=True)
            probs.extend(pred['prob'].cpu().numpy().tolist())
            fa_model.prob=[]; fa_model.label=[]; fa_model.correct=0; fa_model.total=0
        return probs

    df = measure_model('ForAda', predict)
    df.to_csv(OUT / 'forada_results.csv', index=False)
    log(f"  Saved forada_results.csv ({len(df)} rows)")
    fa_model.cpu(); del fa_model
    sys.path.remove(str(FA_ROOT))
    torch.cuda.empty_cache(); gc.collect()
    return df


# ============================================================
# Model 2: EFFORT
# ============================================================
def run_effort():
    log("Loading EFFORT...")
    from transformers import CLIPModel

    # Replicate _asvd from run_inference.py inline
    class _SVDLinear(nn.Module):
        def __init__(s, lin, r):
            super().__init__()
            W = lin.weight.data; b = lin.bias.data if lin.bias is not None else None
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(r, S.numel())
            s.register_buffer('weight_main', U[:, :r] @ torch.diag(S[:r]) @ Vt[:r])
            s.U_residual = nn.Parameter(U[:, r:].clone())
            s.S_residual = nn.Parameter(S[r:].clone())
            s.V_residual = nn.Parameter(Vt[r:].clone())
            s.bias = nn.Parameter(b.clone()) if b is not None else None
        def forward(s, x):
            return F.linear(x, s.weight_main + s.U_residual @ torch.diag(s.S_residual) @ s.V_residual, s.bias)

    def _asvd(model, r):
        for name, module in model.named_children():
            if 'self_attn' in name:
                for sn, sm in module.named_modules():
                    if isinstance(sm, nn.Linear):
                        parent = module; parts = sn.split('.')
                        for p in parts[:-1]: parent = getattr(parent, p)
                        setattr(parent, parts[-1], _SVDLinear(sm, r))
            else:
                _asvd(module, r)
        return model

    _clip = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    _clip.vision_model = _asvd(_clip.vision_model, r=1023)
    eff_bb = _clip.vision_model
    eff_hd = nn.Linear(1024, 2)
    es = torch.load(str(WEIGHTS/'effort_clip_L14_trainOn_sdv14.pth'), map_location='cpu', weights_only=False)
    esd = {k.replace('module.',''):v for k,v in es.items()}
    eff_bb.load_state_dict({k.replace('backbone.',''):v for k,v in esd.items() if k.startswith('backbone.')})
    eff_hd.load_state_dict({k.replace('head.',''):v for k,v in esd.items() if k.startswith('head.')})
    del _clip, es, esd
    eff_bb = eff_bb.to(DEVICE).eval()
    eff_hd = eff_hd.to(DEVICE).eval()

    eff_mean = torch.tensor([.48145466,.4578275,.40821073]).view(1,3,1,1).to(DEVICE)
    eff_std  = torch.tensor([.26862954,.26130258,.27577711]).view(1,3,1,1).to(DEVICE)

    @torch.no_grad()
    def predict(pils, bs=32):
        probs = []
        for i in range(0, len(pils), bs):
            chunk = pils[i:i+bs]
            x = torch.stack([T.ToTensor()(p) for p in chunk]).to(DEVICE)
            x = F.interpolate(x, (224,224), mode='bilinear', align_corners=False)
            inp = (x - eff_mean) / eff_std
            feat = eff_bb(inp).pooler_output
            logits = eff_hd(feat)  # [B, 2], idx 0 = AI/fake based on ai_idx=0 in run_inference
            p = torch.softmax(logits, dim=-1)[:, 0].cpu().numpy()
            probs.extend(p.tolist())
        return probs

    df = measure_model('EFFORT', predict)
    df.to_csv(OUT / 'effort_results.csv', index=False)
    log(f"  Saved effort_results.csv ({len(df)} rows)")
    eff_bb.cpu(); eff_hd.cpu(); del eff_bb, eff_hd
    torch.cuda.empty_cache(); gc.collect()
    return df


# ============================================================
# Model 3: SigLIP+DINOv2
# ============================================================
def run_siglip():
    log("Loading SigLIP+DINOv2...")
    from huggingface_hub import hf_hub_download
    from transformers import AutoProcessor
    import importlib.util
    REPO = 'Bombek1/ai-image-detector-siglip-dinov2'
    model_py = hf_hub_download(repo_id=REPO, filename='model.py')
    spec = importlib.util.spec_from_file_location('ai_det', model_py)
    ai_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(ai_mod)
    wt = hf_hub_download(repo_id=REPO, filename='pytorch_model.pt')
    ckpt = torch.load(wt, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    sig_m = ai_mod.create_model_with_lora(
        siglip_model_name=cfg.get('siglip_model','google/siglip2-so400m-patch14-384'),
        dinov2_model_name=cfg.get('dinov2_model','vit_large_patch14_dinov2.lvd142m'),
        image_size=cfg.get('image_size', 392))
    sig_m.load_state_dict(ckpt['model_state_dict'])
    proc = AutoProcessor.from_pretrained('google/siglip2-so400m-patch14-384')
    sig_m = sig_m.to(DEVICE).eval()

    sig_mean = torch.tensor(proc.image_processor.image_mean).view(1,3,1,1).to(DEVICE)
    sig_std  = torch.tensor(proc.image_processor.image_std).view(1,3,1,1).to(DEVICE)
    sig_size = (proc.image_processor.size.get('height', 384), proc.image_processor.size.get('width', 384))
    dino_mean = torch.tensor([.485,.456,.406]).view(1,3,1,1).to(DEVICE)
    dino_std  = torch.tensor([.229,.224,.225]).view(1,3,1,1).to(DEVICE)

    @torch.no_grad()
    def predict(pils, bs=4):
        probs = []
        for i in range(0, len(pils), bs):
            chunk = pils[i:i+bs]
            x = torch.stack([T.ToTensor()(p) for p in chunk]).to(DEVICE)
            si = (F.interpolate(x, sig_size, mode='bilinear', align_corners=False) - sig_mean) / sig_std
            d  = (F.interpolate(x, (392,392), mode='bilinear', align_corners=False) - dino_mean) / dino_std
            with torch.amp.autocast(device_type='cuda'):
                logits, _, _ = sig_m(si, d)
            logits = logits.float()
            if logits.dim() == 1:
                logits = logits.unsqueeze(-1)
            if logits.shape[-1] == 1:
                p = torch.sigmoid(logits).reshape(-1).cpu().numpy()
            else:
                p = torch.softmax(logits, dim=-1)[:, 0].cpu().numpy()
            probs.extend(p.tolist())
        return probs

    df = measure_model('SigLIP+DINOv2', predict)
    df.to_csv(OUT / 'siglip_results.csv', index=False)
    log(f"  Saved siglip_results.csv ({len(df)} rows)")
    sig_m.cpu(); del sig_m
    torch.cuda.empty_cache(); gc.collect()
    return df


# ============================================================
# Model 4: GenD_DINOv3
# ============================================================
def run_gend():
    log("Loading GenD_DINOv3...")
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
            # GenD uses [real, fake] convention, fake is index 1
            probs.extend(p[:, 1].cpu().numpy().tolist())
        return probs

    df = measure_model('GenD_DINOv3', predict)
    df.to_csv(OUT / 'gend_results.csv', index=False)
    log(f"  Saved gend_results.csv ({len(df)} rows)")
    m.cpu(); del m
    sys.path.remove(str(BASE / 'detection/GenD'))
    torch.cuda.empty_cache(); gc.collect()
    return df


# ============================================================
# Run all
# ============================================================
log("\n" + "=" * 60)
log("Phase 2: Running 4 models")
log("=" * 60)

t_total = time.time()
results = {}
for name, fn in [('ForAda', run_forada), ('EFFORT', run_effort),
                 ('SigLIP+DINOv2', run_siglip), ('GenD_DINOv3', run_gend)]:
    log(f"\n--- {name} ---")
    try:
        results[name] = fn()
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"  FAIL {name}: {e}")

log(f"\nTotal: {time.time()-t_total:.1f}s")
log("DONE")
