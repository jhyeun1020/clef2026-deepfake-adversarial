"""
import os
Shared utilities for paper_experiment phases.
- Model loading (6 models): linear_probe, lora, full_finetune, gend_clip, forada, effort
- Image loading + median3 + jpeg_q80
- BCE forward, PGD attack
"""
import os, sys, time, gc, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageFilter
import cv2

# === GenD torch.compiler patch (must be before importing GenD) ===
if not hasattr(torch.compiler, 'is_compiling'):
    torch.compiler.is_compiling = lambda: False

os.environ.setdefault('HF_TOKEN', 'REMOVED_TOKEN_USE_ENV_VAR')

BASE = Path('/home/s20204116/GEN_AI_GTG')
OUT = BASE / 'detection/paper_experiment'
FF_TRAIN = BASE / 'ff_frames/train'
FF_VAL   = BASE / 'ff_frames/val'
FF_TEST  = BASE / 'ff_frames/test'
WEIGHTS = BASE / 'generation/weights'
LOG_FILE = OUT / 'run_log.txt'

DEVICE = torch.device('cuda:0')

CLIP_MEAN = torch.tensor([.48145466,.4578275,.40821073]).view(1,3,1,1)
CLIP_STD  = torch.tensor([.26862954,.26130258,.27577711]).view(1,3,1,1)

FAKE_TYPES = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


# ============== Classifier head (shared across our 3 trained models) ==============
class Classifier(nn.Module):
    def __init__(s, in_dim=1024, hidden=256):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )
    def forward(s, x): return s.net(x).squeeze(-1)


# ============== Image dataset (FF++ frames) ==============
class FFDataset(Dataset):
    """Loads PNG at 256x256 + label. Optional filter (median3 or jpeg)."""
    def __init__(s, paths, labels, filt=None):
        s.paths = paths
        s.labels = labels
        s.filt = filt  # None | 'med3' | 'jpeg80'
    def __len__(s): return len(s.paths)
    def __getitem__(s, i):
        p = s.paths[i]
        img = Image.open(p).convert('RGB').resize((256,256), Image.LANCZOS)
        if s.filt == 'med3':
            img = img.filter(ImageFilter.MedianFilter(size=3))
        elif s.filt == 'jpeg80':
            img = apply_jpeg_q80(img)
        x = T.ToTensor()(img)
        return x, float(s.labels[i]), str(p)


def apply_jpeg_q80(pil_img):
    arr = np.array(pil_img); bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Image.fromarray(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))


# ============== Gather paths from ff_frames split ==============
def gather_paths(split_dir):
    """Returns (paths, labels) where labels: 0=real, 1=fake."""
    paths, labels = [], []
    for p in sorted((split_dir / 'original').glob('*.png')):
        paths.append(p); labels.append(0)
    for ft in FAKE_TYPES:
        for p in sorted((split_dir / ft).glob('*.png')):
            paths.append(p); labels.append(1)
    return paths, labels


# ============== Model loaders ==============
def load_clip_backbone():
    from transformers import CLIPModel
    clip = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    vision = clip.vision_model
    del clip
    return vision


def forward_clip_pooler(vision_model, x_raw, mean=None, std=None):
    """x_raw [B,3,256,256] in [0,1] → pooler_output [B,1024]."""
    if mean is None:
        mean = CLIP_MEAN.to(x_raw.device)
        std = CLIP_STD.to(x_raw.device)
    x224 = F.interpolate(x_raw, (224,224), mode='bilinear', align_corners=False)
    inp = (x224 - mean) / std
    return vision_model(inp).pooler_output


# --- Our 3 trained models ---
def load_linear_probe():
    vision = load_clip_backbone().to(DEVICE).eval()
    for p in vision.parameters(): p.requires_grad = False
    ckpt = torch.load(OUT / 'models/linear_probe_v2.pt', map_location=DEVICE)
    clf = Classifier(in_dim=ckpt['in_dim'], hidden=ckpt['hidden']).to(DEVICE).eval()
    clf.load_state_dict(ckpt['classifier_state_dict'])
    for p in clf.parameters(): p.requires_grad = False
    return ('linear_probe', vision, clf)


def load_lora():
    from peft import PeftModel
    vision = load_clip_backbone()
    for p in vision.parameters(): p.requires_grad = False
    vision_lora = PeftModel.from_pretrained(vision, str(OUT / 'models/lora_v2')).to(DEVICE).eval()
    ckpt = torch.load(OUT / 'models/lora_v2/classifier.pt', map_location=DEVICE)
    clf = Classifier(in_dim=ckpt['in_dim'], hidden=ckpt['hidden']).to(DEVICE).eval()
    clf.load_state_dict(ckpt['classifier_state_dict'])
    for p in clf.parameters(): p.requires_grad = False
    return ('lora', vision_lora, clf)


def load_full_finetune():
    vision = load_clip_backbone().to(DEVICE).eval()
    ckpt = torch.load(OUT / 'models/full_finetune.pt', map_location=DEVICE)
    vision.load_state_dict(ckpt['vision_state_dict'])
    for p in vision.parameters(): p.requires_grad = False
    clf = Classifier(in_dim=ckpt['in_dim'], hidden=ckpt['hidden']).to(DEVICE).eval()
    clf.load_state_dict(ckpt['classifier_state_dict'])
    for p in clf.parameters(): p.requires_grad = False
    return ('full_finetune', vision, clf)


def forward_ours(model_pack, x_raw):
    """Returns logits [B]."""
    name, vision, clf = model_pack
    feat = forward_clip_pooler(vision, x_raw)
    return clf(feat)


# --- GenD CLIP ---
def load_gend_clip():
    sys.path.insert(0, str(BASE / 'detection/GenD'))
    from src.hf.modeling_gend import GenD
    m = GenD.from_pretrained('yermandy/GenD_CLIP_L_14')
    m = m.to(DEVICE).eval()
    return ('gend_clip', m, None)


def forward_gend(model_pack, x_raw):
    """GenD has its own preprocess. x_raw [B,3,256,256] in [0,1]."""
    name, m, _ = model_pack
    pils = [T.ToPILImage()(x.cpu()) for x in x_raw]
    tensors = torch.stack([m.feature_extractor.preprocess(p) for p in pils]).to(DEVICE)
    logits = m(tensors)  # [B, 2]
    # GenD: index 1 = fake → return as scalar logit (log-ratio)
    return logits[:, 1] - logits[:, 0]  # fake_logit - real_logit


def forward_gend_grad(model_pack, x_raw):
    """Grad-friendly version that doesn't break differentiability. Uses interpolate to GenD's expected size."""
    name, m, _ = model_pack
    # GenD's feature_extractor uses 224x224 with imagenet mean/std for DINOv3-based; for CLIP-L variant similar
    # Try a generic approach: 224, normalize CLIP-style
    mean = CLIP_MEAN.to(x_raw.device)
    std = CLIP_STD.to(x_raw.device)
    x224 = F.interpolate(x_raw, (224,224), mode='bilinear', align_corners=False)
    inp = (x224 - mean) / std
    logits = m(inp)
    return logits[:, 1] - logits[:, 0]


# --- ForAda ---
def load_forada():
    FA_ROOT = BASE / 'detection/ForensicsAdapter'
    sys.path.insert(0, str(FA_ROOT))
    # Patch CLIP cache path
    USER_CLIP_CACHE = str(Path.home() / '.cache/clip')
    os.makedirs(USER_CLIP_CACHE, exist_ok=True)
    from model.clip import clip as fa_clip
    if not hasattr(fa_clip, '_orig_load'):
        fa_clip._orig_load = fa_clip.load
        def _patched(name, device='cuda', jit=False, download_root=None):
            return fa_clip._orig_load(name, device=device, jit=jit, download_root=USER_CLIP_CACHE)
        fa_clip.load = _patched
    from model.ds import DS as ForAda_DS
    fa = ForAda_DS(clip_name='ViT-L/14', adapter_vit_name='vit_tiny_patch16_224',
        num_quires=128, fusion_map={0:0,1:1,2:8,3:15},
        mlp_dim=256, mlp_out_dim=128, head_num=16, device=str(DEVICE))
    fa_ckpt = torch.load(str(FA_ROOT / 'weights/ckpt_best.pth'), map_location='cpu')
    fa.load_state_dict(fa_ckpt, strict=False)
    fa = fa.to(DEVICE).eval()
    return ('forada', fa, None)


def forward_forada(model_pack, x_raw):
    """Returns scalar logit (fake side)."""
    name, fa, _ = model_pack
    mean = CLIP_MEAN.to(x_raw.device)
    std = CLIP_STD.to(x_raw.device)
    xn = (x_raw - mean) / std  # ForAda uses 256, normalize directly
    B = xn.shape[0]
    dd = {'image': xn, 'label': torch.zeros(B, dtype=torch.long, device=DEVICE),
          'mask': None, 'landmark': None, 'if_boundary': torch.ones(B, 256, device=DEVICE)}
    pred = fa(dd, inference=True)
    fa.prob=[]; fa.label=[]; fa.correct=0; fa.total=0
    # pred['cls']: [B, 2]. fake index is 1.
    cls = pred['cls']
    return cls[:, 1] - cls[:, 0]


# --- EFFORT ---
def load_effort():
    from transformers import CLIPModel as _CLIPModel
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

    _clip = _CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
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
    return ('effort', (eff_bb, eff_hd), None)


def forward_effort(model_pack, x_raw):
    name, (bb, hd), _ = model_pack
    mean = CLIP_MEAN.to(x_raw.device)
    std = CLIP_STD.to(x_raw.device)
    x224 = F.interpolate(x_raw, (224,224), mode='bilinear', align_corners=False)
    inp = (x224 - mean) / std
    feat = bb(inp).pooler_output
    logits = hd(feat)  # [B, 2]. EFFORT ai_idx=0
    return logits[:, 0] - logits[:, 1]  # ai - real


# --- Unified forward dispatcher ---
def forward_model(model_pack, x_raw):
    name = model_pack[0]
    if name in ['linear_probe', 'lora', 'full_finetune']:
        return forward_ours(model_pack, x_raw)
    elif name == 'gend_clip':
        return forward_gend_grad(model_pack, x_raw)
    elif name == 'forada':
        return forward_forada(model_pack, x_raw)
    elif name == 'effort':
        return forward_effort(model_pack, x_raw)
    else:
        raise ValueError(f"Unknown model: {name}")


def prob_fake(model_pack, x_raw):
    """Returns prob_fake in [0, 1]."""
    logits = forward_model(model_pack, x_raw)
    return torch.sigmoid(logits)


def unload(model_pack):
    name = model_pack[0]
    try:
        if name == 'effort':
            bb, hd = model_pack[1]
            bb.cpu(); hd.cpu(); del bb, hd
        else:
            if hasattr(model_pack[1], 'cpu'):
                model_pack[1].cpu()
        del model_pack
    except Exception:
        pass
    torch.cuda.empty_cache(); gc.collect()


# ============== Inference batch helper ==============
@torch.no_grad()
def batch_prob_fake(model_pack, paths, batch_size=32, filt=None):
    """Returns numpy array of prob_fake for each path. filt: None|'med3'|'jpeg80'."""
    labels_dummy = [0]*len(paths)
    ds = FFDataset(paths, labels_dummy, filt=filt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    probs = []
    for x, _, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        p = prob_fake(model_pack, x).cpu().numpy()
        probs.extend(p.tolist())
    return np.array(probs, dtype=np.float32)


# ============== Common interfaces for phases ==============
ALL_MODEL_NAMES = ['linear_probe', 'lora', 'full_finetune', 'gend_clip', 'forada', 'effort']

def load_by_name(name):
    if name == 'linear_probe':  return load_linear_probe()
    if name == 'lora':          return load_lora()
    if name == 'full_finetune': return load_full_finetune()
    if name == 'gend_clip':     return load_gend_clip()
    if name == 'forada':        return load_forada()
    if name == 'effort':        return load_effort()
    raise ValueError(name)
