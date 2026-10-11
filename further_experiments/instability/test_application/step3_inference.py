"""
import os
Step 3: 6,002장 × 8 detectors × 4 variants inference
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
import run_inference as ri  # noqa: E402
sys.argv = _orig_argv

import torch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

BASE = Path('/home/student/GEN_AI_GTG')
APP = BASE / 'detection/instability/test_application'
VAR_BASE = APP / 'variants'
OUT = APP / 'scores'
OUT.mkdir(parents=True, exist_ok=True)
LOG = APP / 'inference_log.txt'
FAIL = APP / 'failed.log'

VARIANTS = ['full', 'face_crop_variable', 'random_crop_192_a', 'center_crop_192']

def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] [GPU{my_args.gpu_label}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def log_fail(det, v, err):
    with open(FAIL, 'a') as f:
        f.write(f'{det}\t{v}\t{type(err).__name__}: {err}\n')

def list_pngs(d):
    return sorted(p for p in d.iterdir() if p.suffix.lower() == '.png')

def run_detector_on(spec, model, pils, variant):
    kind = spec['kind']
    batch = 1 if variant == 'face_crop_variable' else spec.get('batch', 32)
    if kind == 'wrapper':
        return ri.score_simple_wrapper(model, pils, spec['ai_idx'], batch)
    if kind == 'recce_ucf':
        return ri.score_recce_ucf(model, pils, batch)
    if kind == 'aide':
        return ri.score_aide(model, pils)
    if kind == 'distildire':
        return ri.score_distildire(model, pils)
    raise RuntimeError(f'unknown kind: {kind}')

def unload(spec, model):
    if spec['kind'] == 'distildire':
        dd, am, *_ = model
        try: dd.cpu(); am.cpu()
        except Exception: pass
    else:
        try: model.cpu()
        except Exception: pass
    del model
    torch.cuda.empty_cache(); gc.collect()

def main():
    detectors = [d.strip() for d in my_args.detectors.split(',') if d.strip()]
    log(f'시작: detectors={detectors}')
    for det_name in detectors:
        if det_name not in ri.DETECTOR_SPECS:
            log(f'  ❌ unknown: {det_name}'); continue
        spec = ri.DETECTOR_SPECS[det_name]
        all_done = all((OUT / f'{det_name}__{v}.parquet').exists() for v in VARIANTS)
        if all_done:
            log(f'  ⏭ {det_name}: 모든 variant 완료, skip'); continue

        log(f'  ▶ {det_name} 로드 중...')
        t = time.time()
        try:
            model = spec['build']()
        except Exception as e:
            traceback.print_exc()
            log(f'  ❌ {det_name} 로드 실패: {e}')
            for v in VARIANTS: log_fail(det_name, v, e)
            continue
        log(f'  ✓ {det_name} 로드 ({time.time()-t:.1f}s)')

        for v in VARIANTS:
            out_p = OUT / f'{det_name}__{v}.parquet'
            if out_p.exists(): continue
            paths = list_pngs(VAR_BASE / v)
            t = time.time()
            try:
                with torch.no_grad():
                    pils = [Image.open(p).convert('RGB') for p in paths]
                    scores = run_detector_on(spec, model, pils, v)
                if len(scores) != len(paths):
                    raise RuntimeError(f'count: {len(scores)} vs {len(paths)}')
                df = pd.DataFrame({
                    'filename': [Path(p).name for p in paths],
                    'score': scores.astype(np.float32),
                    'detector': det_name, 'variant': v,
                })
                df.to_parquet(out_p, index=False)
                log(f'    ✓ {det_name}/{v}  n={len(scores)}  μ={float(scores.mean()):.4f}  ({time.time()-t:.1f}s)')
            except Exception as e:
                traceback.print_exc()
                log(f'    ❌ {det_name}/{v}: {e}')
                log_fail(det_name, v, e)
        unload(spec, model)
        log(f'  ✓ {det_name} 언로드')
    log('종료')

if __name__ == '__main__':
    try: main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
