"""
import os
Step 2: 10 aug_types × 20 images × 8 detectors × 4 variants inference
"""
import os, sys, argparse, time, gc, traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--detectors', required=True)
parser.add_argument('--gpu_label', default='?')
my_args = parser.parse_args()

os.environ.setdefault('HF_TOKEN', 'REMOVED_TOKEN_USE_ENV_VAR')
_orig_argv = sys.argv
sys.argv = ['run_inference.py', '--detectors', 'dummy']
sys.path.insert(0, '/home/student/GEN_AI_GTG/detection/scores')
import run_inference as ri
sys.argv = _orig_argv

import torch, numpy as np, pandas as pd
from PIL import Image

BASE = Path('/home/student/GEN_AI_GTG')
VAR_BASE = BASE / 'detection/instability/control_experiment/variants'
OUT = BASE / 'detection/instability/control_experiment/scores'
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / 'detection/instability/control_experiment/inference_log.txt'

AUG_TYPES = ['original', 'gaussian_noise_002', 'gaussian_noise_005',
             'gaussian_blur_3', 'gaussian_blur_5', 'jpeg_q50', 'jpeg_q70',
             'rotation_15', 'poisson_10', 'rotation_20']
VARIANTS = ['full', 'face_crop_variable', 'random_crop_192_a', 'center_crop_192']

def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] [GPU{my_args.gpu_label}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

def list_pngs(d): return sorted(p for p in d.iterdir() if p.suffix.lower() == '.png')

def run_det(spec, model, pils, variant):
    kind = spec['kind']
    batch = 1 if variant == 'face_crop_variable' else spec.get('batch', 32)
    if kind == 'wrapper':   return ri.score_simple_wrapper(model, pils, spec['ai_idx'], batch)
    if kind == 'recce_ucf': return ri.score_recce_ucf(model, pils, batch)
    if kind == 'aide':      return ri.score_aide(model, pils)
    if kind == 'distildire':return ri.score_distildire(model, pils)
    raise RuntimeError(f'unknown: {kind}')

def unload(spec, model):
    if spec['kind'] == 'distildire':
        try: model[0].cpu(); model[1].cpu()
        except: pass
    else:
        try: model.cpu()
        except: pass
    del model; torch.cuda.empty_cache(); gc.collect()

def main():
    detectors = [d.strip() for d in my_args.detectors.split(',') if d.strip()]
    log(f'시작: {detectors}')
    for det_name in detectors:
        spec = ri.DETECTOR_SPECS[det_name]
        all_done = all((OUT / f'{det_name}__{aug}__{v}.parquet').exists()
                       for aug in AUG_TYPES for v in VARIANTS)
        if all_done:
            log(f'  ⏭ {det_name}: skip'); continue
        log(f'  ▶ {det_name} 로드...')
        t = time.time()
        model = spec['build']()
        log(f'  ✓ {det_name} ({time.time()-t:.1f}s)')
        for aug in AUG_TYPES:
            for v in VARIANTS:
                out_p = OUT / f'{det_name}__{aug}__{v}.parquet'
                if out_p.exists(): continue
                paths = list_pngs(VAR_BASE / aug / v)
                if not paths: continue
                t = time.time()
                with torch.no_grad():
                    pils = [Image.open(p).convert('RGB') for p in paths]
                    scores = run_det(spec, model, pils, v)
                df = pd.DataFrame({
                    'filename': [Path(p).name for p in paths],
                    'score': scores.astype(np.float32),
                })
                df.to_parquet(out_p, index=False)
                log(f'    {det_name}/{aug}/{v} n={len(scores)} μ={scores.mean():.4f} ({time.time()-t:.1f}s)')
        unload(spec, model)
        log(f'  ✓ {det_name} 언로드')
    log('종료')

if __name__ == '__main__':
    try: main()
    except: traceback.print_exc(); sys.exit(1)
