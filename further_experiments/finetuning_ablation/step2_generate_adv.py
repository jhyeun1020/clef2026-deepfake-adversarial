"""
import os
Step 2c: PGD adversarial against trained LoRA model.

First, check whether existing SigLIP/ForAda-target adv fools LoRA.
If LoRA already sees them as REAL (low prob_fake): reuse those.
Else: generate new PGD eps=2/255 adv on AI-gen 100 images.

Output: linear_probe_adv vs lora_adv comparison + lora_adversarial/*.png (if needed)
"""
import os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import CLIPModel
from peft import PeftModel

os.environ.setdefault('HF_TOKEN', 'REMOVED_TOKEN_USE_ENV_VAR')

BASE = Path('/home/s20204116/GEN_AI_GTG')
OUT = BASE / 'detection/finetuning_ablation'
ADV_DIR = OUT / 'lora_adversarial'
ADV_DIR.mkdir(exist_ok=True)
DEVICE = torch.device('cuda:0')
LOG = OUT / 'logs/step2_adv.log'

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

# ============== Load trained LoRA model ==============
log("Loading CLIP + LoRA + classifier...")
clip = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
vision = clip.vision_model
del clip
for p in vision.parameters(): p.requires_grad = False

vision_lora = PeftModel.from_pretrained(vision, str(OUT / 'models/lora_adapter'))
vision_lora = vision_lora.to(DEVICE).eval()

class Classifier(nn.Module):
    def __init__(s, in_dim=1024, hidden=256):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(hidden, 1))
    def forward(s, x): return s.net(x).squeeze(-1)

ckpt = torch.load(OUT / 'models/lora_classifier.pt', map_location=DEVICE)
clf = Classifier(in_dim=ckpt['in_dim'], hidden=ckpt['hidden']).to(DEVICE).eval()
clf.load_state_dict(ckpt['classifier_state_dict'])
for p in clf.parameters(): p.requires_grad = False
log(f"  Val acc on FF++: {ckpt['val_acc']:.3f} (real {ckpt['val_real_acc']:.2f} / fake {ckpt['val_fake_acc']:.2f})")

CLIP_MEAN = torch.tensor([.48145466,.4578275,.40821073]).view(1,3,1,1).to(DEVICE)
CLIP_STD  = torch.tensor([.26862954,.26130258,.27577711]).view(1,3,1,1).to(DEVICE)

def forward_full(x_raw):
    x224 = F.interpolate(x_raw, (224,224), mode='bilinear', align_corners=False)
    inp = (x224 - CLIP_MEAN) / CLIP_STD
    feat = vision_lora(inp).pooler_output
    return clf(feat)


# ============== Step 1: Check if existing adversarial fools LoRA ==============
log("\n=== Check 1: Does existing SigLIP/ForAda adv fool LoRA? ===")
ADV_BASE = BASE / 'detection/multi_attack/images'
to_tensor = T.ToTensor()

@torch.no_grad()
def measure(paths, label):
    pils = [Image.open(p).convert('RGB').resize((256,256), Image.LANCZOS) for p in paths]
    probs = []
    for i in range(0, len(pils), 16):
        chunk = pils[i:i+16]
        x = torch.stack([to_tensor(p) for p in chunk]).to(DEVICE)
        probs.extend(torch.sigmoid(forward_full(x)).cpu().numpy().tolist())
    log(f"  {label}: mean prob_fake={np.mean(probs):.3f}, < 0.5 ratio={np.mean(np.array(probs)<0.5):.0%}")
    return np.array(probs)

# Sample 100 from existing adv (siglip target) and (forada target)
rng = np.random.default_rng(42)
def collect(target):
    dirs = sorted([d for d in ADV_BASE.glob(f'{target}__*__pgd_eps4') if not d.name.endswith('jpeg80')])
    imgs = []
    for d in dirs: imgs.extend(sorted(d.glob('*.png')))
    sel = sorted(rng.choice(len(imgs), 100, replace=False))
    return [imgs[i] for i in sel]

adv_sig_paths = collect('siglip')
adv_fa_paths  = collect('forada')

# Same AI-gen pool as Step 1 (clean baseline)
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
clean_paths = [ai_pool[i] for i in sorted(rng.choice(len(ai_pool), 100, replace=False))]

probs_clean = measure(clean_paths,    "clean AI-gen     ")
probs_sig   = measure(adv_sig_paths,  "adv_siglip target")
probs_fa    = measure(adv_fa_paths,   "adv_forada target")

# 기존 adv가 LoRA를 충분히 속이는지 판단
# 기준: 기존 adv의 mean prob_fake가 clean보다 0.3 이상 낮으면 "충분히 속음"
sig_drop = float(np.mean(probs_clean) - np.mean(probs_sig))
fa_drop  = float(np.mean(probs_clean) - np.mean(probs_fa))
log(f"\n  prob drop (clean → siglip-adv): {sig_drop:+.3f}")
log(f"  prob drop (clean → forada-adv): {fa_drop:+.3f}")

REUSE_THRESHOLD = 0.3
if sig_drop >= REUSE_THRESHOLD or fa_drop >= REUSE_THRESHOLD:
    log("  → 기존 adversarial이 LoRA를 충분히 속임. 재활용 가능 (그래도 own-target 새로 생성으로 정확한 측정).")
else:
    log("  → 기존 adversarial이 LoRA를 별로 안 속임. 새로 생성 필요.")
log("\n→ 어느 쪽이든, own-target adv를 새로 생성합니다 (apples-to-apples 비교).")


# ============== Step 2: Generate LoRA-target PGD adversarial ==============
log("\n=== Generate LoRA-target PGD eps=2/255 ===")
EPS = 2 / 255
ALPHA = 0.5 / 255
STEPS = 40

def pgd_attack(x_clean):
    """Push prob_fake → 0 (target REAL)."""
    x = x_clean.clone().detach()
    delta = torch.empty_like(x).uniform_(-EPS, EPS)
    delta = (x + delta).clamp(0, 1) - x
    delta = delta.detach().requires_grad_(True)

    for step in range(STEPS):
        x_adv = x + delta
        logits = forward_full(x_adv)  # [B]
        loss = logits.sum()  # minimize logit (sigmoid(logit) → 0)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta - ALPHA * grad.sign()
            delta = delta.clamp(-EPS, EPS)
            delta = (x + delta).clamp(0, 1) - x
        delta = delta.detach().requires_grad_(True)
    return (x + delta).clamp(0, 1).detach()


to_pil = T.ToPILImage()
log(f"PGD: eps={EPS:.4f} (2/255), alpha={ALPHA:.5f} (0.5/255), steps={STEPS}")
log(f"Generating {len(clean_paths)} adversarial samples...")

BATCH = 8
t0 = time.time()
success_decrease = 0
for bi in range(0, len(clean_paths), BATCH):
    chunk = clean_paths[bi:bi+BATCH]
    pils = [Image.open(p).convert('RGB').resize((256,256), Image.LANCZOS) for p in chunk]
    x = torch.stack([to_tensor(p) for p in pils]).to(DEVICE)
    with torch.no_grad():
        l_orig = torch.sigmoid(forward_full(x)).cpu().numpy()
    x_adv = pgd_attack(x)
    with torch.no_grad():
        l_adv = torch.sigmoid(forward_full(x_adv)).cpu().numpy()
    for j, p in enumerate(chunk):
        adv_pil = to_pil(x_adv[j].cpu())
        adv_pil.save(ADV_DIR / f"{p.stem}.png")
        if l_adv[j] < l_orig[j]: success_decrease += 1
    log(f"  batch {bi}-{bi+len(chunk)}: prob_fake {np.mean(l_orig):.3f} → {np.mean(l_adv):.3f}")

log(f"\nDone in {time.time()-t0:.1f}s. Success rate (prob decreased): {success_decrease}/{len(clean_paths)}")
log("DONE Step 2c")
