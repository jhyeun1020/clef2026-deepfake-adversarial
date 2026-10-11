"""
import os
Step 1c: PGD adversarial generation targeting trained Linear Probe.
- Source: 100 AI-gen images sampled from multi_attack source pool (our_flux_pulid + bitmind_*)
- Attack: PGD eps=4/255, steps=40, alpha=eps/10
- Goal: push prob_fake → 0 (toward REAL)
- Output: linear_probe_adv/*.png (100 images)
"""
import os, sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import CLIPModel

os.environ.setdefault('HF_TOKEN', 'REMOVED_TOKEN_USE_ENV_VAR')

BASE = Path('/home/s20204116/GEN_AI_GTG')
OUT = BASE / 'detection/finetuning_ablation'
ADV_DIR = OUT / 'linear_probe_adv'
ADV_DIR.mkdir(exist_ok=True)
DEVICE = torch.device('cuda:0')
LOG = OUT / 'logs/step1_adv.log'

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

# ============== Build full model (CLIP backbone + classifier) ==============
log("Loading CLIP ViT-L/14 + Linear Probe classifier...")
clip_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(DEVICE)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

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

ckpt = torch.load(OUT / 'models/linear_probe.pt', map_location=DEVICE)
clf = Classifier(in_dim=ckpt['in_dim'], hidden=ckpt['hidden']).to(DEVICE)
clf.load_state_dict(ckpt['classifier_state_dict'])
clf.eval()
for p in clf.parameters():
    p.requires_grad = False
log(f"  Loaded. Val acc on FF++: {ckpt['val_acc']:.3f}")

CLIP_MEAN = torch.tensor([.48145466,.4578275,.40821073]).view(1,3,1,1).to(DEVICE)
CLIP_STD  = torch.tensor([.26862954,.26130258,.27577711]).view(1,3,1,1).to(DEVICE)

def forward_full(x_raw):
    """x_raw in [0,1] at 256×256 → logits."""
    x224 = F.interpolate(x_raw, (224,224), mode='bilinear', align_corners=False)
    inp = (x224 - CLIP_MEAN) / CLIP_STD
    feat = clip_model.vision_model(inp).pooler_output
    return clf(feat)


# ============== Source images (AI-gen, same 600 used by multi_attack) ==============
rng = np.random.default_rng(42)
ai_dirs = [
    BASE / 'generation/outputs/final_1002',
    BASE / 'detection/detection_dataset/genimage_fitting_data/sdxl',
    BASE / 'detection/detection_dataset/genimage_fitting_data/flux',
]
ai_pool = []
for d in ai_dirs:
    imgs = sorted(d.glob('*.png'))
    sel = sorted(rng.choice(len(imgs), 200, replace=False)) if len(imgs) > 200 else range(len(imgs))
    ai_pool.extend([imgs[i] for i in sel])
log(f"AI-gen pool: {len(ai_pool)} (200 from each of 3 sources)")

# Pick 100 for adversarial
adv_idx = sorted(rng.choice(len(ai_pool), 100, replace=False))
adv_paths = [ai_pool[i] for i in adv_idx]
log(f"Adversarial source: {len(adv_paths)} images")


# ============== PGD attack ==============
EPS = 4 / 255
ALPHA = EPS / 10
STEPS = 40

def pgd_attack(x_clean):
    """x_clean: [B,3,256,256] in [0,1]. Push prob_fake → 0 (target REAL)."""
    x = x_clean.clone().detach()
    # Random init within eps ball
    delta = torch.empty_like(x).uniform_(-EPS, EPS)
    delta = (x + delta).clamp(0, 1) - x
    delta = delta.detach().requires_grad_(True)

    for step in range(STEPS):
        x_adv = x + delta
        logits = forward_full(x_adv)  # [B]
        # Target REAL → minimize prob_fake = sigmoid(logit). Equivalently minimize logit.
        loss = logits.sum()
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta - ALPHA * grad.sign()
            delta = delta.clamp(-EPS, EPS)
            delta = (x + delta).clamp(0, 1) - x
        delta = delta.detach().requires_grad_(True)
    return (x + delta).clamp(0, 1).detach()


to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

# ============== Generate ==============
log(f"PGD attack: eps={EPS:.4f} (4/255), alpha={ALPHA:.5f}, steps={STEPS}")
log(f"Generating {len(adv_paths)} adversarial samples...")

BATCH = 8
t0 = time.time()
success_decrease = 0
total = 0
for bi in range(0, len(adv_paths), BATCH):
    chunk = adv_paths[bi:bi+BATCH]
    pils = [Image.open(p).convert('RGB').resize((256,256), Image.LANCZOS) for p in chunk]
    x = torch.stack([to_tensor(p) for p in pils]).to(DEVICE)

    # Original logit
    with torch.no_grad():
        l_orig = torch.sigmoid(forward_full(x)).cpu().numpy()

    x_adv = pgd_attack(x)

    with torch.no_grad():
        l_adv = torch.sigmoid(forward_full(x_adv)).cpu().numpy()

    for j, p in enumerate(chunk):
        adv_pil = to_pil(x_adv[j].cpu())
        adv_pil.save(ADV_DIR / f"{p.stem}.png")
        if l_adv[j] < l_orig[j]:
            success_decrease += 1
        total += 1

    log(f"  batch {bi}-{bi+len(chunk)}: prob_fake {np.mean(l_orig):.3f} → {np.mean(l_adv):.3f}")

log(f"Done in {time.time()-t0:.1f}s. Success rate (prob decreased): {success_decrease}/{total}")
log("DONE Step 1c")
