"""
Phase 1: Train Linear Probe + LoRA + Full FT on FF++ train set.
"""
import os, sys, time, json, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).parent))
from common_utils import (
    log, BASE, OUT, FF_TRAIN, FF_VAL, DEVICE, FAKE_TYPES,
    Classifier, FFDataset, gather_paths,
    CLIP_MEAN, CLIP_STD,
)

def make_train_loader(paths, labels, batch_size=64):
    """WeightedRandomSampler to balance real/fake."""
    labels_arr = np.array(labels)
    n_per_class = np.bincount(labels_arr)
    weights = 1.0 / n_per_class[labels_arr]
    sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True)
    ds = FFDataset(paths, labels)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      num_workers=8, pin_memory=True, persistent_workers=True)


def make_val_loader(paths, labels, batch_size=64):
    ds = FFDataset(paths, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=8, pin_memory=True, persistent_workers=True)


# ============== Linear Probe (frozen CLIP + classifier) ==============
def train_linear_probe():
    log("=" * 60)
    log("Phase 1A: Linear Probe (frozen CLIP)")
    log("=" * 60)
    if (OUT / 'models/linear_probe_v2.pt').exists():
        log("Already trained. Skipping.")
        return
    train_paths, train_labels = gather_paths(FF_TRAIN)
    val_paths, val_labels = gather_paths(FF_VAL)
    log(f"Train: {len(train_paths)}  Val: {len(val_paths)}")

    clip = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    vision = clip.vision_model.to(DEVICE).eval()
    del clip
    for p in vision.parameters(): p.requires_grad = False

    mean = CLIP_MEAN.to(DEVICE); std = CLIP_STD.to(DEVICE)

    @torch.no_grad()
    def featurize(loader):
        feats, labs = [], []
        for x, y, _ in loader:
            x = x.to(DEVICE, non_blocking=True)
            x224 = F.interpolate(x, (224,224), mode='bilinear', align_corners=False)
            inp = (x224 - mean) / std
            f = vision(inp).pooler_output  # [B, 1024]
            feats.append(f.cpu()); labs.append(y)
        return torch.cat(feats, dim=0), torch.cat(labs, dim=0)

    # Pre-extract features (no augmentation, frozen backbone)
    cache_path = OUT / 'models/linear_probe_features.pt'
    if cache_path.exists():
        log(f"Loading cached features from {cache_path}")
        cached = torch.load(cache_path)
        train_feats, train_labs = cached['train_feats'], cached['train_labs']
        val_feats, val_labs = cached['val_feats'], cached['val_labs']
    else:
        log("Extracting train features...")
        t0 = time.time()
        train_loader_lin = DataLoader(FFDataset(train_paths, train_labels),
                                       batch_size=64, shuffle=False,
                                       num_workers=8, pin_memory=True)
        train_feats, train_labs = featurize(train_loader_lin)
        log(f"  Train feats: {train_feats.shape} ({time.time()-t0:.0f}s)")
        log("Extracting val features...")
        val_loader_lin = DataLoader(FFDataset(val_paths, val_labels),
                                     batch_size=64, shuffle=False,
                                     num_workers=8, pin_memory=True)
        val_feats, val_labs = featurize(val_loader_lin)
        log(f"  Val feats: {val_feats.shape}")
        torch.save({'train_feats': train_feats, 'train_labs': train_labs,
                    'val_feats': val_feats, 'val_labs': val_labs}, cache_path)

    vision.cpu(); del vision
    torch.cuda.empty_cache(); gc.collect()

    # Train classifier
    clf = Classifier(in_dim=1024, hidden=256).to(DEVICE)
    train_feats = train_feats.to(DEVICE)
    train_labs = train_labs.float().to(DEVICE)
    val_feats = val_feats.to(DEVICE)
    val_labs = val_labs.float().to(DEVICE)

    n_real = (train_labs == 0).sum().item()
    n_fake = (train_labs == 1).sum().item()
    pos_weight = torch.tensor(n_real / n_fake).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    EPOCHS, BATCH = 20, 64
    patience = 3
    best_auc, best_state, pc = 0, None, 0
    history = []
    for epoch in range(EPOCHS):
        clf.train()
        perm = torch.randperm(len(train_feats), device=DEVICE)
        total_loss = 0; n = 0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i+BATCH]
            x, y = train_feats[b], train_labs[b]
            logits = clf(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1
        train_loss = total_loss / n
        sched.step()

        clf.eval()
        with torch.no_grad():
            val_logits = clf(val_feats).cpu().numpy()
            val_y = val_labs.cpu().numpy()
            val_auc = roc_auc_score(val_y, val_logits)
        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_auc': val_auc})
        log(f"  Epoch {epoch+1}/{EPOCHS}: train_loss={train_loss:.4f}  val_AUROC={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                log(f"  Early stop at epoch {epoch+1}")
                break

    log(f"Best val AUROC: {best_auc:.4f}")
    if best_auc < 0.6:
        log("WARNING: val AUROC < 0.6 even after training. Logged but proceeding.")

    torch.save({
        'classifier_state_dict': best_state,
        'in_dim': 1024, 'hidden': 256,
        'val_auc': best_auc,
    }, OUT / 'models/linear_probe_v2.pt')
    import pandas as pd
    pd.DataFrame(history).to_csv(OUT / 'models/linear_probe_v2_log.csv', index=False)
    log("Saved linear_probe_v2.pt")


# ============== LoRA ==============
def train_lora():
    log("=" * 60)
    log("Phase 1B: LoRA (CLIP + LoRA r=32)")
    log("=" * 60)
    if (OUT / 'models/lora_v2/adapter_config.json').exists() and (OUT / 'models/lora_v2/classifier.pt').exists():
        log("Already trained. Skipping.")
        return
    train_paths, train_labels = gather_paths(FF_TRAIN)
    val_paths, val_labels = gather_paths(FF_VAL)
    log(f"Train: {len(train_paths)}  Val: {len(val_paths)}")

    clip = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    vision = clip.vision_model
    del clip
    for p in vision.parameters(): p.requires_grad = False

    lora_cfg = LoraConfig(r=32, lora_alpha=64,
                          target_modules=["q_proj", "v_proj"],
                          lora_dropout=0.1, bias="none")
    vision_lora = get_peft_model(vision, lora_cfg).to(DEVICE)
    vision_lora.print_trainable_parameters()

    clf = Classifier(in_dim=1024, hidden=256).to(DEVICE)
    mean = CLIP_MEAN.to(DEVICE); std = CLIP_STD.to(DEVICE)

    n_real = sum(1 for l in train_labels if l == 0)
    n_fake = sum(1 for l in train_labels if l == 1)
    pos_weight = torch.tensor(n_real / n_fake).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    lora_params = [p for n, p in vision_lora.named_parameters() if p.requires_grad]
    head_params = list(clf.parameters())
    opt = torch.optim.AdamW([
        {'params': lora_params, 'lr': 5e-5},
        {'params': head_params, 'lr': 1e-3},
    ], weight_decay=1e-2)

    EPOCHS, BATCH, GRAD_ACCUM = 10, 32, 4
    n_iters = len(train_paths) // BATCH
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    train_loader = make_train_loader(train_paths, train_labels, batch_size=BATCH)
    val_loader = make_val_loader(val_paths, val_labels, batch_size=BATCH)

    best_auc, best_state, pc = 0, None, 0
    patience = 3
    history = []
    for epoch in range(EPOCHS):
        vision_lora.train(); clf.train()
        t0 = time.time()
        total_loss = 0; n = 0
        opt.zero_grad()
        for it, (x, y, _) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            x224 = F.interpolate(x, (224,224), mode='bilinear', align_corners=False)
            inp = (x224 - mean) / std
            feat = vision_lora(inp).pooler_output
            logits = clf(feat)
            loss = loss_fn(logits, y) / GRAD_ACCUM
            loss.backward()
            if (it+1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_([*lora_params, *head_params], 1.0)
                opt.step(); opt.zero_grad()
            total_loss += loss.item() * GRAD_ACCUM; n += 1
            if it >= n_iters: break
        train_loss = total_loss / max(n, 1)
        sched.step()

        # Val
        vision_lora.eval(); clf.eval()
        val_logits_all, val_y_all = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(DEVICE, non_blocking=True)
                x224 = F.interpolate(x, (224,224), mode='bilinear', align_corners=False)
                inp = (x224 - mean) / std
                feat = vision_lora(inp).pooler_output
                logits = clf(feat)
                val_logits_all.append(logits.cpu().numpy())
                val_y_all.append(y.numpy())
        val_logits = np.concatenate(val_logits_all)
        val_y = np.concatenate(val_y_all)
        val_auc = roc_auc_score(val_y, val_logits)
        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_auc': val_auc, 'time_sec': time.time()-t0})
        log(f"  Epoch {epoch+1}/{EPOCHS} ({time.time()-t0:.0f}s): tr_loss={train_loss:.4f}  val_AUROC={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {
                'lora': {k: v.cpu().clone() for k, v in vision_lora.state_dict().items() if 'lora' in k.lower()},
                'classifier': {k: v.cpu().clone() for k, v in clf.state_dict().items()},
            }
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                log(f"  Early stop at epoch {epoch+1}")
                break

    log(f"Best val AUROC: {best_auc:.4f}")
    # Restore best and save
    save_dir = OUT / 'models/lora_v2'
    save_dir.mkdir(parents=True, exist_ok=True)
    vision_lora.load_state_dict({**vision_lora.state_dict(), **best_state['lora']}, strict=False)
    vision_lora.save_pretrained(str(save_dir))
    torch.save({
        'classifier_state_dict': best_state['classifier'],
        'in_dim': 1024, 'hidden': 256,
        'val_auc': best_auc,
    }, save_dir / 'classifier.pt')
    import pandas as pd
    pd.DataFrame(history).to_csv(OUT / 'models/lora_v2_log.csv', index=False)
    log("Saved lora_v2/")


# ============== Full Fine-tuning ==============
def train_full_finetune():
    log("=" * 60)
    log("Phase 1C: Full Fine-tuning")
    log("=" * 60)
    if (OUT / 'models/full_finetune.pt').exists():
        log("Already trained. Skipping.")
        return
    train_paths, train_labels = gather_paths(FF_TRAIN)
    val_paths, val_labels = gather_paths(FF_VAL)
    log(f"Train: {len(train_paths)}  Val: {len(val_paths)}")

    clip = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    vision = clip.vision_model.to(DEVICE)
    del clip
    for p in vision.parameters(): p.requires_grad = True

    clf = Classifier(in_dim=1024, hidden=256).to(DEVICE)
    mean = CLIP_MEAN.to(DEVICE); std = CLIP_STD.to(DEVICE)

    n_real = sum(1 for l in train_labels if l == 0)
    n_fake = sum(1 for l in train_labels if l == 1)
    pos_weight = torch.tensor(n_real / n_fake).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW([
        {'params': vision.parameters(), 'lr': 1e-5},
        {'params': clf.parameters(),    'lr': 1e-3},
    ], weight_decay=1e-2)

    EPOCHS, BATCH, GRAD_ACCUM = 10, 32, 4
    n_iters = len(train_paths) // BATCH
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    train_loader = make_train_loader(train_paths, train_labels, batch_size=BATCH)
    val_loader = make_val_loader(val_paths, val_labels, batch_size=BATCH)

    best_auc, best_state, pc = 0, None, 0
    patience = 3
    history = []
    for epoch in range(EPOCHS):
        vision.train(); clf.train()
        t0 = time.time()
        total_loss = 0; n = 0
        opt.zero_grad()
        for it, (x, y, _) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            x224 = F.interpolate(x, (224,224), mode='bilinear', align_corners=False)
            inp = (x224 - mean) / std
            feat = vision(inp).pooler_output
            logits = clf(feat)
            loss = loss_fn(logits, y) / GRAD_ACCUM
            loss.backward()
            if (it+1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_([*vision.parameters(), *clf.parameters()], 1.0)
                opt.step(); opt.zero_grad()
            total_loss += loss.item() * GRAD_ACCUM; n += 1
            if it >= n_iters: break
        train_loss = total_loss / max(n, 1)
        sched.step()

        vision.eval(); clf.eval()
        val_logits_all, val_y_all = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(DEVICE, non_blocking=True)
                x224 = F.interpolate(x, (224,224), mode='bilinear', align_corners=False)
                inp = (x224 - mean) / std
                feat = vision(inp).pooler_output
                logits = clf(feat)
                val_logits_all.append(logits.cpu().numpy())
                val_y_all.append(y.numpy())
        val_logits = np.concatenate(val_logits_all)
        val_y = np.concatenate(val_y_all)
        val_auc = roc_auc_score(val_y, val_logits)
        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_auc': val_auc, 'time_sec': time.time()-t0})
        log(f"  Epoch {epoch+1}/{EPOCHS} ({time.time()-t0:.0f}s): tr_loss={train_loss:.4f}  val_AUROC={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {
                'vision': {k: v.cpu().clone() for k, v in vision.state_dict().items()},
                'classifier': {k: v.cpu().clone() for k, v in clf.state_dict().items()},
            }
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                log(f"  Early stop at epoch {epoch+1}")
                break

    log(f"Best val AUROC: {best_auc:.4f}")
    torch.save({
        'vision_state_dict': best_state['vision'],
        'classifier_state_dict': best_state['classifier'],
        'in_dim': 1024, 'hidden': 256,
        'val_auc': best_auc,
    }, OUT / 'models/full_finetune.pt')
    import pandas as pd
    pd.DataFrame(history).to_csv(OUT / 'models/full_finetune_log.csv', index=False)
    log("Saved full_finetune.pt")


if __name__ == '__main__':
    try: train_linear_probe()
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"FAIL Linear Probe: {e}")
    torch.cuda.empty_cache(); gc.collect()
    try: train_lora()
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"FAIL LoRA: {e}")
    torch.cuda.empty_cache(); gc.collect()
    try: train_full_finetune()
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"FAIL Full FT: {e}")
    log("Phase 1 DONE")
