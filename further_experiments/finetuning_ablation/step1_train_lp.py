"""
import os
Step 1a-b: Linear Probe training
- CLIP ViT-L/14 backbone (FROZEN)
- Classifier head: Linear(1024, 256) → GELU → Dropout(0.3) → Linear(256, 1) → Sigmoid
  (Note: using pooler_output=1024 instead of projected=768 for stronger feature)
- Train data: FF++ 5,500 (Real 1,500 + Fake 4,000) — same as previous calibration set
- 90/10 train/val, seed=42, BCE loss, 10 epochs, LR 1e-3, batch 32

Outputs: models/linear_probe.pt (classifier head weights)
"""
import os, sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

os.environ.setdefault('HF_TOKEN', 'REMOVED_TOKEN_USE_ENV_VAR')

BASE = Path('/home/s20204116/GEN_AI_GTG')
OUT = BASE / 'detection/finetuning_ablation'
DEVICE = torch.device('cuda:0')
LOG = OUT / 'logs/step1_train.log'

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

# ============== Build train/val data ==============
# Same FF++ structure as model_comparison/run_gend.py
DD = BASE / 'detection/detection_dataset/ff_fitting_data'
SOURCES = {
    'real_original':  (DD / 'real/original',                   0),  # 0=real
    'real_dfd':       (DD / 'real/dfd_original',               0),
    'deepfakes':      (DD / 'fake_swap/deepfakes',             1),  # 1=fake
    'faceswap':       (DD / 'fake_swap/faceswap',              1),
    'faceshifter':    (DD / 'fake_swap/faceshifter',           1),
    'face2face':      (DD / 'fake_reenactment/face2face',      1),
    'neuraltextures': (DD / 'fake_reenactment/neuraltextures', 1),
}

all_paths = []
all_labels = []
for name, (path, label) in SOURCES.items():
    imgs = sorted(path.glob('*.png'))
    for p in imgs:
        all_paths.append(p)
        all_labels.append(label)
log(f"Total: {len(all_paths)}  Real: {all_labels.count(0)}  Fake: {all_labels.count(1)}")

# 90/10 split with seed=42
rng = np.random.default_rng(42)
idx = np.arange(len(all_paths))
rng.shuffle(idx)
n_train = int(0.9 * len(idx))
train_idx = idx[:n_train]
val_idx = idx[n_train:]
log(f"Train: {len(train_idx)}  Val: {len(val_idx)}")


# ============== Pre-extract CLIP features (frozen backbone) ==============
log("Loading CLIP ViT-L/14...")
clip_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(DEVICE)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

CLIP_MEAN = torch.tensor([.48145466,.4578275,.40821073]).view(1,3,1,1).to(DEVICE)
CLIP_STD  = torch.tensor([.26862954,.26130258,.27577711]).view(1,3,1,1).to(DEVICE)

@torch.no_grad()
def extract_features(paths, batch=32):
    """Returns features tensor [N, 1024] (pooler_output)."""
    feats = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i+batch]
        pils = [Image.open(p).convert('RGB').resize((256,256), Image.LANCZOS) for p in chunk]
        x = torch.stack([T.ToTensor()(p) for p in pils]).to(DEVICE)
        x224 = F.interpolate(x, (224,224), mode='bilinear', align_corners=False)
        inp = (x224 - CLIP_MEAN) / CLIP_STD
        f = clip_model.vision_model(inp).pooler_output  # [B, 1024]
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)

CACHE = OUT / 'models/features.pt'
if CACHE.exists():
    log(f"Loading cached features {CACHE}")
    cached = torch.load(CACHE)
    features = cached['features']
    cached_labels = cached['labels']
    assert len(features) == len(all_paths), f"cache mismatch: {len(features)} vs {len(all_paths)}"
else:
    log("Extracting features for 5,500 images...")
    t0 = time.time()
    features = extract_features(all_paths)
    log(f"  done {time.time()-t0:.1f}s, shape={features.shape}")
    torch.save({'features': features, 'labels': torch.tensor(all_labels)}, CACHE)
    log(f"  saved to {CACHE}")

labels_t = torch.tensor(all_labels, dtype=torch.float32)
train_feats = features[train_idx]
train_labels = labels_t[train_idx]
val_feats = features[val_idx]
val_labels = labels_t[val_idx]


# ============== Classifier head ==============
class Classifier(nn.Module):
    def __init__(s, in_dim=1024, hidden=256):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )
    def forward(s, x):
        return s.net(x).squeeze(-1)  # logits [B]

clf = Classifier().to(DEVICE)
opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
n_real = sum(1 for l in all_labels if l == 0)
n_fake = sum(1 for l in all_labels if l == 1)
pos_weight = torch.tensor(n_real / n_fake).to(DEVICE)  # downweight fake (majority)
log(f"pos_weight (Real/Fake balance): {pos_weight.item():.3f}")
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ============== Train ==============
train_feats = train_feats.to(DEVICE)
train_labels = train_labels.to(DEVICE)
val_feats = val_feats.to(DEVICE)
val_labels = val_labels.to(DEVICE)

best_val_loss = float('inf')
best_state = None
patience = 3
patience_counter = 0
BATCH = 32
EPOCHS = 10

log(f"Training Linear Probe (epochs={EPOCHS}, batch={BATCH}, lr=1e-3)...")
for epoch in range(EPOCHS):
    clf.train()
    perm = torch.randperm(len(train_feats), device=DEVICE)
    total_loss = 0.0
    n_batches = 0
    correct = 0
    for i in range(0, len(perm), BATCH):
        b = perm[i:i+BATCH]
        x = train_feats[b]
        y = train_labels[b]
        logits = clf(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n_batches += 1
        with torch.no_grad():
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == y).sum().item()
    train_acc = correct / len(train_feats)
    train_loss = total_loss / n_batches

    clf.eval()
    with torch.no_grad():
        v_logits = clf(val_feats)
        v_loss = loss_fn(v_logits, val_labels).item()
        v_pred = (torch.sigmoid(v_logits) > 0.5).float()
        v_acc = (v_pred == val_labels).float().mean().item()

    scheduler.step()
    log(f"  Epoch {epoch+1}: train_loss={train_loss:.4f} acc={train_acc:.3f} | val_loss={v_loss:.4f} acc={v_acc:.3f}")

    if v_loss < best_val_loss:
        best_val_loss = v_loss
        best_state = {k: v.clone() for k, v in clf.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            log(f"  Early stop at epoch {epoch+1}")
            break

clf.load_state_dict(best_state)

# Final eval
clf.eval()
with torch.no_grad():
    v_logits = clf(val_feats)
    v_pred = (torch.sigmoid(v_logits) > 0.5).float()
    v_acc = (v_pred == val_labels).float().mean().item()
    # Per-class
    real_mask = (val_labels == 0)
    fake_mask = (val_labels == 1)
    real_acc = (v_pred[real_mask] == 0).float().mean().item() if real_mask.any() else 0
    fake_acc = (v_pred[fake_mask] == 1).float().mean().item() if fake_mask.any() else 0
log(f"Final val: acc={v_acc:.3f}  real_acc={real_acc:.3f}  fake_acc={fake_acc:.3f}")

# Save classifier
torch.save({
    'classifier_state_dict': clf.state_dict(),
    'in_dim': 1024,
    'hidden': 256,
    'val_acc': v_acc,
    'val_real_acc': real_acc,
    'val_fake_acc': fake_acc,
}, OUT / 'models/linear_probe.pt')
log(f"Saved classifier to models/linear_probe.pt")
log("DONE Step 1a-b")
