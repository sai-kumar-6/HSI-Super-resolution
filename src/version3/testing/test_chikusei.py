"""
test_chikusei.py  —  V3 Edition
M.Tech Thesis – Hyperspectral Pansharpening

Evaluates all three VMamba variants on Chikusei test patches and prints
a side-by-side comparison table.

Usage
─────
  # Quick smoke test (random weights):
  python testing/test_chikusei.py

  # With trained checkpoints:
  python testing/test_chikusei.py \
      --checkpoint_old      comparison/checkpoints/old_best.pth \
      --checkpoint_improved comparison/checkpoints/improved_best.pth \
      --checkpoint_v3       comparison/checkpoints/v3_best.pth
"""

import sys
import os
import json
import time
import argparse

_HERE    = os.path.dirname(os.path.abspath(__file__))
_V3      = os.path.dirname(_HERE)
_PROJECT = os.path.dirname(_V3)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, 'scripts'))
sys.path.insert(0, _V3)

import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

from dataset_loader_overlap import create_dataloaders_overlap
from baseline_models import create_vmamba_model, count_parameters
from loss_functions  import compute_psnr, compute_sam_metric, compute_ergas

# ── Config ───────────────────────────────────────────────────────────────────
CHIKUSEI_PATH = os.path.join(_PROJECT, 'chikusei', 'chikusei.mat')
RESULTS_DIR   = os.path.join(_HERE, 'results')
IN_CHANNELS   = 128
SCALE         = 4
D_MODEL       = 32
D_STATE       = 8
NUM_BLOCKS    = [1, 1, 1, 1]
MAX_PATCHES   = 20
PATCH_SIZE    = 32   # HR patch (LR = 8×8)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_model(variant, ckpt_path, device):
    model = create_vmamba_model(
        variant=variant, in_ch=IN_CHANNELS, out_ch=IN_CHANNELS,
        scale=SCALE, d_model=D_MODEL, d_state=D_STATE, num_blocks=NUM_BLOCKS,
    ).to(device)
    if ckpt_path and os.path.isfile(ckpt_path):
        st = torch.load(ckpt_path, map_location=device)
        if isinstance(st, dict) and 'model_state_dict' in st:
            st = st['model_state_dict']
        model.load_state_dict(st)
        print(f"  [OK] {variant} ← {ckpt_path}")
    else:
        print(f"  [INFO] {variant}: no checkpoint — random weights")
    return model.eval()


@torch.no_grad()
def evaluate(model, loader, device, label):
    ps, ss, es, ts = [], [], [], []
    for batch in tqdm(loader, desc=f"  Eval {label}", leave=False):
        lr_hsi = batch['lr_hsi'].to(device)
        hr_pan = batch['hr_pan'].to(device)
        hr_hsi = batch['hr_hsi'].to(device)
        t0  = time.perf_counter()
        out = model(lr_hsi, hr_pan)
        t1  = time.perf_counter()
        ps.append(compute_psnr(out, hr_hsi))
        ss.append(compute_sam_metric(out, hr_hsi))
        es.append(compute_ergas(out, hr_hsi, scale=SCALE))
        ts.append((t1 - t0) / lr_hsi.size(0) * 1000)
    avg = lambda l: sum(l) / max(len(l), 1)
    return {'psnr': avg(ps), 'sam': avg(ss), 'ergas': avg(es), 'inference_ms': avg(ts)}


def print_table(results: dict, models: dict):
    variants = list(results.keys())
    labels   = {'old': 'Old VMamba', 'improved': 'Improved VMamba', 'v3': 'V3 VMamba'}
    col_w    = 16

    header = f"{'Metric':<18}" + "".join(f"{labels[v]:>{col_w}}" for v in variants)
    sep    = '-' * len(header)
    print('\n' + sep + '\n' + header + '\n' + sep)

    metrics = {
        'PSNR  (dB)':        ('psnr',         True ),
        'SAM   (deg)':       ('sam',          False),
        'ERGAS':             ('ergas',        False),
        'Inference (ms/img)':('inference_ms', False),
    }
    for display, (key, higher_better) in metrics.items():
        row_vals = [results[v][key] for v in variants]
        best_idx = row_vals.index(max(row_vals) if higher_better else min(row_vals))
        row = f"{display:<18}"
        for i, v in enumerate(variants):
            val_str = f"{row_vals[i]:.3f}"
            row += f"{val_str:>{col_w}}"
        print(row)

    print(sep)
    params = {v: count_parameters(models[v])[0] for v in variants}
    row = f"{'Params (M)':<18}"
    for v in variants:
        row += f"{params[v]/1e6:>{col_w}.2f}"
    print(row)
    print(sep + '\n')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_old',      default=None)
    parser.add_argument('--checkpoint_improved', default=None)
    parser.add_argument('--checkpoint_v3',       default=None)
    parser.add_argument('--batch_size',  default=1,           type=int)
    parser.add_argument('--max_patches', default=MAX_PATCHES, type=int)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")

    # Dataset
    print("\nLoading Chikusei test set …")
    _, _, test_l = create_dataloaders_overlap(
        dataset='chikusei', mat_path=CHIKUSEI_PATH,
        batch_size=1, patch_size=PATCH_SIZE,
        overlap=PATCH_SIZE // 2, scale=SCALE, num_workers=0,
    )
    n      = min(len(test_l.dataset), args.max_patches)
    subset = torch.utils.data.Subset(test_l.dataset, list(range(n)))
    loader = DataLoader(subset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    print(f"  Test patches : {n}")

    # Models
    ckpts = {
        'old':      args.checkpoint_old,
        'improved': args.checkpoint_improved,
        'v3':       args.checkpoint_v3,
    }
    print("\nBuilding models …")
    models = {v: load_model(v, ckpts[v], device) for v in ['old', 'improved', 'v3']}

    # Evaluate
    print("\nRunning evaluation …")
    results = {v: evaluate(models[v], loader, device, v) for v in models}

    # Print table
    print_table(results, models)

    # Save JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'test_results.json')
    params_info = {v: count_parameters(models[v])[0] for v in models}
    with open(save_path, 'w') as f:
        json.dump({
            'results': results,
            'params':  params_info,
            'dataset': CHIKUSEI_PATH,
            'n_patches': n, 'scale': SCALE, 'd_model': D_MODEL,
        }, f, indent=2)
    print(f"Results saved → {save_path}")


if __name__ == '__main__':
    main()
