"""
import os
Step 2a-b: CLIP ViT-L/14 + LoRA fine-tuning.

LoRA: r=32, target=["q_proj","v_proj"], alpha=64, dropout=0.1
Classifier: Linear(1024→256) + GELU + Dropout(0.3) + Linear(256→1)
  (Note: spec says 768 input but using 1024 pooler_output to match Step 1)
Train: FF++ 5,500 (Real 1,500 + Fake 4,000), 90/10 split, seed=42
Loss: BCEWithLogitsLoss with pos_weight (class balance)
LR: 5e-5 (LoRA), 1e-3 (head). Epochs: 10, batch: 32. Early stop on val_loss.

Output: models/lora_adapter/ + models/lora_classifier.pt
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
from peft import LoraConfig, get_peft_model

os.environ.setdefault('HF_TOKEN', 'REMOVED_TOKEN_USE_ENV_VAR')

BASE = Path('/home/s20204116/GEN_AI_GTG')
OUT = BASE / 'detection/finetuning_ablation'
DEVICE = torch.device('cuda:0')
LOG = OUT / 'logs/step2_train.log'

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

# ============== Build train/val data (same as Step 1) ==============
DD = BASE / 'detection/detection_dataset/ff_fitting_data'
SOURCES = {
    'real_original':  (DD / 'real/original',                   0),
    'real_dfd':       (DD / 'real/dfd_original',               0),
    'deepfakes':      (DD / 'fake_swap/deepfakes',             1),
    'faceswap':       (DD / 'fake_swap/faceswap',              1),
    'faceshifter':    (DD / 'fake_swap/faceshifter',           1),
    'face2face':      (DD / 'fake_reenactment/face2face',      1),
    'neuraltextures': (DD / 'fake_reenactment/neuraltextures', 1),
}
all_paths, all_labels = [], []
for name, (path, label) in SOURCES.items():
    for p in sorted(path.glob('*.png')):
        all_paths.append(p); all_labels.append(label)
log(f"Total: {len(all_paths)}  Real: {all_labels.count(0)}  Fake: {all_labels.count(1)}")

rng = np.random.default_rng(42)
idx = np.arange(len(all_paths))
rng.shuffle(idx)
n_train = int(0.9 * len(idx))
train_idx, val_idx = idx[:n_train], idx[n_train:]
log(f"Train: {len(train_idx)}  Val: {len(val_idx)}")


# ============== Load CLIP + apply LoRA ==============
log("Loading CLIP ViT-L/14...")
clip = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
vision = clip.vision_model
del clip  # only need vision encoder

# Freeze all
for p in vision.parameters():
    p.requires_grad = False

# Apply LoRA to q_proj, v_proj (vision encoder attention)
lora_cfg = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
)
vision_lora = get_peft_model(vision, lora_cfg)
vision_lora.to(DEVICE)
vision_lora.print_trainable_parameters()

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
        return s.net(x).squeeze(-1)

clf = Classifier().to(DEVICE)
log(f"Classifier params: {sum(p.numel() for p in clf.parameters()):,}")


# ============== Forward ==============
CLIP_MEAN = torch.tensor([.48145466,.4578275,.40821073]).view(1,3,1,1).to(DEVICE)
CLIP_STD  = torch.tensor([.26862954,.26130258,.27577711]).view(1,3,1,1).to(DEVICE)

def forward_full(x_raw, training=True):
    """x_raw [B,3,256,256] in [0,1] → logits [B]."""
    x224 = F.interpolate(x_raw, (224,224), mode='bilinear', align_corners=False)
    inp = (x224 - CLIP_MEAN) / CLIP_STD
    out = vision_lora(inp)
    feat = out.pooler_output  # [B, 1024]
    return clf(feat)


# ============== Optimizer with 2 param groups ==============
lora_params = [p for n, p in vision_lora.named_parameters() if p.requires_grad]
head_params = list(clf.parameters())
log(f"LoRA trainable: {sum(p.numel() for p in lora_params):,}")
log(f"Head trainable: {sum(p.numel() for p in head_params):,}")

optimizer = torch.optim.AdamW([
    {'params': lora_params, 'lr': 5e-5},
    {'params': head_params, 'lr': 1e-3},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Class balance
n_real = sum(1 for l in all_labels if l == 0)
n_fake = sum(1 for l in all_labels if l == 1)
pos_weight = torch.tensor(n_real / n_fake).to(DEVICE)
log(f"pos_weight (Real/Fake balance): {pos_weight.item():.3f}")
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ============== Image loading helpers ==============
to_tensor = T.ToTensor()

def load_batch(paths_subset):
    pils = [Image.open(p).convert('RGB').resize((256,256), Image.LANCZOS) for p in paths_subset]
    return torch.stack([to_tensor(p) for p in pils])


# ============== Train loop ==============
BATCH = 32
EPOCHS = 10
patience = 3
best_val_loss = float('inf')
best_state = None
patience_counter = 0

train_paths = [all_paths[i] for i in train_idx]
train_labels_t = torch.tensor([all_labels[i] for i in train_idx], dtype=torch.float32)
val_paths = [all_paths[i] for i in val_idx]
val_labels_t = torch.tensor([all_labels[i] for i in val_idx], dtype=torch.float32)

log(f"\nTraining (epochs={EPOCHS}, batch={BATCH}, LR LoRA=5e-5, head=1e-3)...")
log(f"Estimated ~{len(train_idx)//BATCH} iters/epoch, ~30-60 min total")

for epoch in range(EPOCHS):
    vision_lora.train(); clf.train()
    perm = np.random.default_rng(epoch).permutation(len(train_idx))
    total_loss = 0.0; n_batches = 0; correct = 0
    t0 = time.time()
    for i in range(0, len(perm), BATCH):
        b = perm[i:i+BATCH]
        x = load_batch([train_paths[j] for j in b]).to(DEVICE)
        y = train_labels_t[b].to(DEVICE)
        logits = forward_full(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item(); n_batches += 1
        with torch.no_grad():
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == y).sum().item()
    train_acc = correct / len(train_idx)
    train_loss = total_loss / n_batches

    # Val
    vision_lora.eval(); clf.eval()
    val_loss_total = 0.0; val_correct = 0; val_real_correct = 0; val_fake_correct = 0
    val_real_n = 0; val_fake_n = 0
    with torch.no_grad():
        for i in range(0, len(val_idx), BATCH):
            x = load_batch(val_paths[i:i+BATCH]).to(DEVICE)
            y = val_labels_t[i:i+BATCH].to(DEVICE)
            logits = forward_full(x, training=False)
            val_loss_total += loss_fn(logits, y).item() * len(y)
            pred = (torch.sigmoid(logits) > 0.5).float()
            val_correct += (pred == y).sum().item()
            real_mask = (y == 0); fake_mask = (y == 1)
            val_real_correct += ((pred == y) & real_mask).sum().item()
            val_fake_correct += ((pred == y) & fake_mask).sum().item()
            val_real_n += real_mask.sum().item(); val_fake_n += fake_mask.sum().item()
    val_loss = val_loss_total / len(val_idx)
    val_acc = val_correct / len(val_idx)
    val_real_acc = val_real_correct / max(val_real_n, 1)
    val_fake_acc = val_fake_correct / max(val_fake_n, 1)
    scheduler.step()

    log(f"  Epoch {epoch+1} ({time.time()-t0:.0f}s): tr_loss={train_loss:.4f} acc={train_acc:.3f} "
        f"| val_loss={val_loss:.4f} acc={val_acc:.3f} (real {val_real_acc:.2f} / fake {val_fake_acc:.2f})")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {
            'lora': {k: v.cpu().clone() for k, v in vision_lora.state_dict().items() if 'lora' in k.lower()},
            'classifier': {k: v.cpu().clone() for k, v in clf.state_dict().items()},
            'val_loss': val_loss, 'val_acc': val_acc,
            'val_real_acc': val_real_acc, 'val_fake_acc': val_fake_acc,
        }
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            log(f"  Early stop at epoch {epoch+1}")
            break

# ============== Save ==============
log(f"\nBest val_loss={best_state['val_loss']:.4f} acc={best_state['val_acc']:.3f} "
    f"(real {best_state['val_real_acc']:.2f} / fake {best_state['val_fake_acc']:.2f})")

save_dir = OUT / 'models/lora_adapter'
save_dir.mkdir(parents=True, exist_ok=True)
# Restore best state and save full vision_lora adapter
vision_lora.load_state_dict({**vision_lora.state_dict(), **best_state['lora']}, strict=False)
vision_lora.save_pretrained(str(save_dir))
log(f"Saved LoRA adapter -> {save_dir}")

torch.save({
    'classifier_state_dict': best_state['classifier'],
    'in_dim': 1024, 'hidden': 256,
    'val_loss': best_state['val_loss'],
    'val_acc': best_state['val_acc'],
    'val_real_acc': best_state['val_real_acc'],
    'val_fake_acc': best_state['val_fake_acc'],
}, OUT / 'models/lora_classifier.pt')
log(f"Saved classifier -> models/lora_classifier.pt")
log("DONE Step 2a-b")
