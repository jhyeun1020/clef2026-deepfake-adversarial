"""Step 2: 8 det × 500 imgs × 4 variants inference."""
import os
import os, sys, argparse, time, gc, traceback
os.environ.setdefault('HF_TOKEN', 'REMOVED_TOKEN_USE_ENV_VAR')

parser = argparse.ArgumentParser()
parser.add_argument('--detectors', required=True)
parser.add_argument('--gpu_label', default='?')
a = parser.parse_args()

_o = sys.argv; sys.argv = ['x', '--detectors', 'dummy']
sys.path.insert(0, '/home/student/GEN_AI_GTG/detection/scores')
import run_inference as ri; sys.argv = _o

import torch, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image

BASE = Path('/home/student/GEN_AI_GTG')
VAR = BASE / 'detection/adv_vs_jpeg_decomposition/variants'
OUT = BASE / 'detection/adv_vs_jpeg_decomposition/scores'
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / 'detection/adv_vs_jpeg_decomposition/inference_log.txt'
VARIANTS = ['full', 'face_crop_variable', 'random_crop_192_a', 'center_crop_192']

def log(m):
    l = f'[{time.strftime("%H:%M:%S")}] [GPU{a.gpu_label}] {m}'
    print(l, flush=True)
    with open(LOG, 'a') as f: f.write(l+'\n')

for det in [d.strip() for d in a.detectors.split(',') if d.strip()]:
    if all((OUT/f'{det}__{v}.parquet').exists() for v in VARIANTS):
        log(f'⏭ {det}'); continue
    spec = ri.DETECTOR_SPECS[det]; log(f'▶ {det}...')
    model = spec['build']()
    for v in VARIANTS:
        op = OUT/f'{det}__{v}.parquet'
        if op.exists(): continue
        paths = sorted((VAR/v).glob('*.png'))
        pils = [Image.open(p).convert('RGB') for p in paths]
        b = 1 if v == 'face_crop_variable' else spec.get('batch', 32)
        with torch.no_grad():
            k = spec['kind']
            if k == 'wrapper':     s = ri.score_simple_wrapper(model, pils, spec['ai_idx'], b)
            elif k == 'recce_ucf': s = ri.score_recce_ucf(model, pils, b)
            elif k == 'aide':      s = ri.score_aide(model, pils)
            elif k == 'distildire':s = ri.score_distildire(model, pils)
        pd.DataFrame({'filename': [p.name for p in paths], 'score': s.astype(np.float32)}).to_parquet(op, index=False)
        log(f'  {det}/{v} n={len(s)} μ={s.mean():.4f}')
    if spec['kind']=='distildire':
        try: model[0].cpu(); model[1].cpu()
        except: pass
    else:
        try: model.cpu()
        except: pass
    del model; torch.cuda.empty_cache(); gc.collect()
    log(f'✓ {det}')
log('종료')
