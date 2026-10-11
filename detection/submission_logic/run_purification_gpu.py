"""
import os
Parallel purification signal extraction on single GPU.
Usage: CUDA_VISIBLE_DEVICES=X python run_purification_gpu.py --start START --end END
"""
import os, sys, time, json, argparse
os.environ.setdefault('HF_TOKEN', 'REMOVED_TOKEN_USE_ENV_VAR')

from pathlib import Path
import numpy as np, pandas as pd, torch, cv2
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from numpy.linalg import norm

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--end', type=int, required=True)
parser.add_argument('--gpu', type=int, required=True)
args = parser.parse_args()

BASE = Path('/home/student/GEN_AI_GTG')
OUT = BASE / 'detection/submission'
TEST_DIR = BASE / 'detection/ImageCLEF2026-DeepFakeDetection-Tes/Data/Images_Detection'
DEVICE = torch.device('cuda:0')
RESULT_FILE = OUT / f'purification_gpu{args.gpu}.json'

def log(msg):
    print(f'[GPU{args.gpu} {time.strftime("%H:%M:%S")}] {msg}', flush=True)

purifications = {
    'median3': lambda arr: cv2.medianBlur(arr, 3),
    'median5': lambda arr: cv2.medianBlur(arr, 5),
    'gaussian1': lambda arr: cv2.GaussianBlur(arr, (3,3), 1.0),
    'bitdepth4': lambda arr: (arr // 16) * 16,
}

# Load SigLIP
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor
import importlib.util
mp = hf_hub_download(repo_id='Bombek1/ai-image-detector-siglip-dinov2', filename='model.py')
spec = importlib.util.spec_from_file_location('m', mp)
md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
wt = hf_hub_download(repo_id='Bombek1/ai-image-detector-siglip-dinov2', filename='pytorch_model.pt')
ckpt = torch.load(wt, map_location='cpu', weights_only=False); cfg = ckpt.get('config',{})
sig_model = md.create_model_with_lora(
    siglip_model_name=cfg.get('siglip_model','google/siglip2-so400m-patch14-384'),
    dinov2_model_name=cfg.get('dinov2_model','vit_large_patch14_dinov2.lvd142m'),
    image_size=cfg.get('image_size',392))
sig_model.load_state_dict(ckpt['model_state_dict']); sig_model.to(DEVICE).eval()
proc = AutoProcessor.from_pretrained('google/siglip2-so400m-patch14-384')
ss = (proc.image_processor.size.get('height',384), proc.image_processor.size.get('width',384))
sm = torch.tensor(proc.image_processor.image_mean).view(1,3,1,1).to(DEVICE)
ssd = torch.tensor(proc.image_processor.image_std).view(1,3,1,1).to(DEVICE)
dm = torch.tensor([.485,.456,.406]).view(1,3,1,1).to(DEVICE)
ds_ = torch.tensor([.229,.224,.225]).view(1,3,1,1).to(DEVICE)
dbb = sig_model.dinov2; dbl = list(dbb.blocks)
sig_hook = {}
def sig_mk(li):
    def fn(m, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        sig_hook[li] = t[:, 0, :].detach().cpu()
    return fn
dbl[19].register_forward_hook(sig_mk(19))

# Load GenD
sys.path.insert(0, str(BASE / 'detection/GenD'))
from src.hf.modeling_gend import GenD
gend_model = GenD.from_pretrained('yermandy/GenD_DINOv3_L').to(DEVICE).eval()
gend_t = T.Compose([T.Resize((518,518), interpolation=T.InterpolationMode.BICUBIC),
                     T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
gend_layers = gend_model.feature_extractor.backbone.layer
gend_hook = {}
def gend_mk(li):
    def fn(mod, inp, out):
        if isinstance(out, tuple): gend_hook[li] = out[0][:, 0, :].detach().cpu()
        else: gend_hook[li] = out[:, 0, :].detach().cpu()
    return fn
gend_layers[14].register_forward_hook(gend_mk(14))

log("Models loaded.")

@torch.no_grad()
def sig_forward(pils, bs=16):
    all_logits = []; all_f19 = []
    for i in range(0, len(pils), bs):
        ch = pils[i:i+bs]
        imgs = torch.stack([T.ToTensor()(p.convert('RGB')) for p in ch]).to(DEVICE)
        sig_hook.clear()
        di = (F.interpolate(imgs, (392,392), mode='bilinear', align_corners=False) - dm) / ds_
        _ = dbb.forward_features(di)
        all_f19.append(sig_hook[19].numpy().astype(np.float32))
        si = (F.interpolate(imgs, ss, mode='bilinear', align_corners=False) - sm) / ssd
        with torch.amp.autocast(device_type='cuda'):
            lo, _, _ = sig_model(si, di)
        all_logits.extend(lo.float().cpu().numpy().flatten().tolist())
    return np.array(all_logits), np.concatenate(all_f19, 0)

@torch.no_grad()
def gend_forward(pils, bs=16):
    all_logits = []; all_f14 = []
    for i in range(0, len(pils), bs):
        ch = pils[i:i+bs]
        tensors = torch.stack([gend_t(img) for img in ch]).to(DEVICE)
        gend_hook.clear()
        logits = gend_model(tensors)
        all_logits.extend(logits[:, 1].cpu().numpy().tolist())
        all_f14.append(gend_hook[14].numpy().astype(np.float32))
    return np.array(all_logits), np.concatenate(all_f14, 0)

# Process assigned range
test_paths = sorted(TEST_DIR.glob('*.png'))[args.start:args.end]
log(f"Processing {len(test_paths)} images (idx {args.start}-{args.end})")

results = {}
CHUNK = 100

for ci in range(0, len(test_paths), CHUNK):
    chunk_paths = test_paths[ci:ci+CHUNK]
    chunk_ids = [p.stem for p in chunk_paths]
    pils_orig = [Image.open(p).convert('RGB') for p in chunk_paths]
    arrs_orig = [np.array(img) for img in pils_orig]

    sig_lo_orig, sig_f19_orig = sig_forward(pils_orig)
    gend_lo_orig, gend_f14_orig = gend_forward(pils_orig)

    for pname, pfunc in purifications.items():
        pils_pur = [Image.fromarray(pfunc(arr)) for arr in arrs_orig]
        sig_lo_pur, sig_f19_pur = sig_forward(pils_pur)
        gend_lo_pur, gend_f14_pur = gend_forward(pils_pur)

        for j, cid in enumerate(chunk_ids):
            if cid not in results:
                results[cid] = {}
            results[cid][f'delta_logit_siglip_{pname}'] = float(sig_lo_pur[j] - sig_lo_orig[j])
            results[cid][f'delta_L19_siglip_{pname}'] = float(norm(sig_f19_orig[j] - sig_f19_pur[j]))
            results[cid][f'delta_logit_gend_{pname}'] = float(gend_lo_pur[j] - gend_lo_orig[j])
            results[cid][f'delta_L14_gend_{pname}'] = float(norm(gend_f14_orig[j] - gend_f14_pur[j]))

    torch.cuda.empty_cache()
    if (ci // CHUNK) % 5 == 0:
        log(f"  {ci+len(chunk_paths)}/{len(test_paths)}")

# Save
RESULT_FILE.write_text(json.dumps(results))
log(f"Done: {len(results)} images saved to {RESULT_FILE.name}")
