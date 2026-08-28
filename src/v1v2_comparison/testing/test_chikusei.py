"""
test_chikusei.py
M.Tech Thesis – Hyperspectral Pansharpening

Evaluates both VMamba variants on the Chikusei dataset and prints a
side-by-side comparison table.

Usage
-----
# Quick test with randomly-initialised weights:
    python testing/test_chikusei.py

# With saved checkpoints:
    python testing/test_chikusei.py \\
        --checkpoint_old   path/to/old_best.pth \\
        --checkpoint_improved path/to/improved_best.pth
"""

import sys
import os
import json
import time
import argparse

# Make the project root importable from this subfolder
_HERE       = os.path.dirname(os.path.abspath(__file__))
_PROJECT    = os.path.dirname(_HERE)
sys.path.insert(0, _PROJECT)

import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):          # fallback if tqdm not installed
        return x

from dataset_loader import load_chikusei_dataset, PansharpeningDataset
from baseline_models import create_vmamba_model, count_parameters
from loss_functions  import compute_psnr, compute_sam_metric, compute_ergas


# ============================================================================
# Configuration
# ============================================================================

CHIKUSEI_PATH = os.path.join(_PROJECT, 'chikusei', 'chikusei.mat')
RESULTS_DIR   = os.path.join(_HERE, 'results')
IN_CHANNELS   = 128
SCALE         = 4
D_MODEL       = 32          # OOM-safe default
NUM_BLOCKS    = [1, 1, 1, 1]  # OOM-safe default
MAX_PATCHES   = 20          # max test patches for a quick evaluation run


# ============================================================================
# Helpers
# ============================================================================

def load_model(variant: str, checkpoint_path: str | None, device: torch.device) -> torch.nn.Module:
    """Create model and optionally load checkpoint weights."""
    model = create_vmamba_model(
        variant    = variant,
        in_ch      = IN_CHANNELS,
        out_ch     = IN_CHANNELS,
        scale      = SCALE,
        d_model    = D_MODEL,
        num_blocks = NUM_BLOCKS,
    ).to(device)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        # Support both bare state-dicts and checkpoint dicts
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        print(f"  [OK] Loaded {variant} weights from {checkpoint_path}")
    else:
        print(f"  [INFO] {variant}: no checkpoint found – using random weights")

    model.eval()
    return model


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, label: str):
    """Run inference on the loader and return averaged metrics."""
    psnr_list, sam_list, ergas_list, time_list = [], [], [], []

    for batch in tqdm(loader, desc=f"Evaluating {label}", leave=False):
        lr_hsi = batch['lr_hsi'].to(device)
        hr_pan = batch['hr_pan'].to(device)
        hr_hsi = batch['hr_hsi'].to(device)

        t0  = time.perf_counter()
        out = model(lr_hsi, hr_pan)
        t1  = time.perf_counter()

        psnr_list.append( compute_psnr(out, hr_hsi) )
        sam_list.append(  compute_sam_metric(out, hr_hsi) )
        ergas_list.append(compute_ergas(out, hr_hsi, scale=SCALE) )
        time_list.append( (t1 - t0) / lr_hsi.size(0) )   # per-image

    return {
        'psnr'          : float(sum(psnr_list)  / len(psnr_list)),
        'sam'           : float(sum(sam_list)   / len(sam_list)),
        'ergas'         : float(sum(ergas_list) / len(ergas_list)),
        'inference_ms'  : float(sum(time_list)  / len(time_list) * 1000),
    }


def print_table(results: dict):
    """Pretty-print comparison table."""
    header = f"{'Metric':<18}{'Old VMamba':>14}{'Improved VMamba':>18}{'Delta':>10}"
    sep    = '-' * len(header)
    print('\n' + sep)
    print(header)
    print(sep)

    metrics = {
        'PSNR  (dB)':        ('psnr',  True,   '+.3f'),   # higher is better
        'SAM   (deg)':       ('sam',   False,  '+.3f'),   # lower  is better
        'ERGAS':             ('ergas', False,  '+.3f'),   # lower  is better
        'Inference (ms/img)':('inference_ms', False, '+.1f'),
    }

    r_old = results['old']
    r_imp = results['improved']

    for display, (key, higher_better, fmt) in metrics.items():
        old_v = r_old[key]
        imp_v = r_imp[key]
        delta = imp_v - old_v
        sign  = '+' if delta >= 0 else ''
        arrow = '↑' if (delta > 0) == higher_better else ('↓' if delta != 0 else '=')
        print(f"{display:<18}{old_v:>14.3f}{imp_v:>18.3f}  {sign}{delta:.3f} {arrow}")

    print(sep)

    # Parameter comparison
    p_old, _ = count_parameters(results['_model_old'])
    p_imp, _ = count_parameters(results['_model_imp'])
    print(f"{'Params (M)':<18}{p_old/1e6:>14.2f}{p_imp/1e6:>18.2f}")
    print(sep + '\n')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test VMamba models on Chikusei')
    parser.add_argument('--checkpoint_old',      default=None, type=str,
                        help='Path to Old VMamba checkpoint (.pth)')
    parser.add_argument('--checkpoint_improved', default=None, type=str,
                        help='Path to Improved VMamba checkpoint (.pth)')
    parser.add_argument('--batch_size',  default=1,   type=int)
    parser.add_argument('--max_patches', default=MAX_PATCHES, type=int,
                        help='Maximum number of test patches (default 20)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")
    print(f"Dataset: {CHIKUSEI_PATH}")

    # ---- Load dataset ----
    print("\nLoading Chikusei dataset …")
    _, test_hsi = load_chikusei_dataset(CHIKUSEI_PATH, train_ratio=0.8)

    test_dataset = PansharpeningDataset(
        test_hsi,
        patch_size = 32,          # HR patch → 32×32, LR=32/4=8×8 (min for backbone)
        scale      = SCALE,
        overlap    = 0.25,
        augment    = False,
    )

    # Limit to MAX_PATCHES for speed
    n_patches = min(len(test_dataset), args.max_patches)
    subset    = torch.utils.data.Subset(test_dataset, list(range(n_patches)))
    loader    = DataLoader(
        subset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )
    print(f"Test patches : {n_patches}  (batch_size={args.batch_size})")

    # ---- Build models ----
    print("\nBuilding models …")
    model_old = load_model('old',      args.checkpoint_old,      device)
    model_imp = load_model('improved', args.checkpoint_improved, device)

    total_old, _ = count_parameters(model_old)
    total_imp, _ = count_parameters(model_imp)
    print(f"  Old      params : {total_old:,}")
    print(f"  Improved params : {total_imp:,}")

    # ---- Evaluate ----
    print("\nRunning evaluation …")
    results_old = evaluate(model_old, loader, device, 'Old VMamba')
    results_imp = evaluate(model_imp, loader, device, 'Improved VMamba')

    results = {
        'old':      results_old,
        'improved': results_imp,
        # Store model references for param count in table (not serialised)
        '_model_old': model_old,
        '_model_imp': model_imp,
    }

    # ---- Print table ----
    print_table(results)

    # ---- Save JSON ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'test_results.json')
    serialisable = {
        'old':      results_old,
        'improved': results_imp,
        'params':   {'old': total_old, 'improved': total_imp},
        'dataset':  CHIKUSEI_PATH,
        'n_patches': n_patches,
        'scale':    SCALE,
        'd_model':  D_MODEL,
    }
    with open(save_path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"Results saved → {save_path}")


if __name__ == '__main__':
    main()
