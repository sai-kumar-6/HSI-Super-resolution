"""
test_chikusei.py  —  V5 Edition
Evaluates all 5 model variants on Chikusei test patches.

Usage
─────
  cd version5
  python testing/test_chikusei.py
  python testing/test_chikusei.py \
      --checkpoint_old      comparison/checkpoints/old_best.pth \
      --checkpoint_improved comparison/checkpoints/improved_best.pth \
      --checkpoint_v3       comparison/checkpoints/v3_best.pth \
      --checkpoint_v4       comparison/checkpoints/v4_best.pth \
      --checkpoint_v5       comparison/checkpoints/v5_best.pth
"""
from __future__ import annotations
import sys, os, argparse

_HERE    = os.path.dirname(os.path.abspath(__file__))   # testing/
_V5      = os.path.dirname(_HERE)                        # version5/
_PROJECT = os.path.dirname(_V5)

sys.path.insert(0, os.path.join(_PROJECT, 'version3'))  # lowest priority
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, 'scripts'))
sys.path.insert(0, os.path.join(_PROJECT, 'version4'))
sys.path.insert(0, _V5)                                  # highest priority

import torch

from dataset_loader_overlap import create_dataloaders_overlap
from baseline_models import create_vmamba_model, count_parameters
from loss_functions_v5 import compute_psnr, compute_sam_metric, compute_ergas

CHIKUSEI_PATH = os.path.join(_PROJECT, 'chikusei', 'chikusei.mat')
IN_CHANNELS   = 128
SCALE         = 4
D_MODEL       = 32
D_STATE       = 8
NUM_BLOCKS    = [1, 1, 1, 1]
PATCH_SIZE    = 32

VARIANT_LABELS = {
    'old':      'V1 Old VMamba',
    'improved': 'V2 Improved VMamba',
    'v3':       'V3 VMamba',
    'v4':       'V4 VMamba (Spectral Fix)',
    'v5':       'V5 VMamba (True Parallel Scan)',
}


def load_model(variant, ckpt_path, device):
    model = create_vmamba_model(
        variant=variant, in_ch=IN_CHANNELS, out_ch=IN_CHANNELS,
        scale=SCALE, d_model=D_MODEL, d_state=D_STATE,
        num_blocks=NUM_BLOCKS,
    ).to(device)
    if ckpt_path and os.path.isfile(ckpt_path):
        st = torch.load(ckpt_path, map_location=device)
        if isinstance(st, dict) and 'model_state_dict' in st:
            st = st['model_state_dict']
        model.load_state_dict(st)
        print(f"  [OK] {variant} <- {ckpt_path}")
    else:
        print(f"  [INFO] {variant}: random weights (no checkpoint)")
    return model.eval()


def evaluate(model, test_l, device, max_patches):
    psnr_sum = sam_sum = ergas_sum = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(test_l):
            if i >= max_patches:
                break
            lr_hsi = batch['lr_hsi'].to(device)
            hr_pan = batch['hr_pan'].to(device)
            hr_hsi = batch['hr_hsi']
            pred   = model(lr_hsi, hr_pan).cpu()

            psnr_sum  += compute_psnr(pred, hr_hsi)
            sam_sum   += compute_sam_metric(pred, hr_hsi)
            ergas_sum += compute_ergas(pred, hr_hsi, scale=SCALE)
            n += 1

    if n == 0:
        return 0.0, 0.0, 0.0
    return psnr_sum / n, sam_sum / n, ergas_sum / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_old',      default=None)
    parser.add_argument('--checkpoint_improved', default=None)
    parser.add_argument('--checkpoint_v3',       default=None)
    parser.add_argument('--checkpoint_v4',       default=None)
    parser.add_argument('--checkpoint_v5',       default=None)
    parser.add_argument('--max_patches', default=20, type=int)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print("Loading Chikusei test set …")
    _, _, test_l = create_dataloaders_overlap(
        dataset='chikusei', mat_path=CHIKUSEI_PATH,
        batch_size=1, patch_size=PATCH_SIZE,
        overlap=PATCH_SIZE // 2, scale=SCALE, num_workers=0,
    )
    print(f"  Test patches: {len(test_l.dataset)}  (evaluating up to {args.max_patches})")

    ckpts = {
        'old':      args.checkpoint_old,
        'improved': args.checkpoint_improved,
        'v3':       args.checkpoint_v3,
        'v4':       args.checkpoint_v4,
        'v5':       args.checkpoint_v5,
    }

    print("\nLoading models …")
    results = {}
    for v, ck in ckpts.items():
        m = load_model(v, ck, device)
        total, _ = count_parameters(m)
        psnr, sam, ergas = evaluate(m, test_l, device, args.max_patches)
        results[v] = {'psnr': psnr, 'sam': sam, 'ergas': ergas, 'params': total}

    print("\n" + "=" * 78)
    print(f"{'Variant':<30} {'PSNR (↑)':>10} {'SAM (↓)':>10} "
          f"{'ERGAS (↓)':>10} {'Params':>10}")
    print("=" * 78)
    for v, m in results.items():
        print(f"  {VARIANT_LABELS[v]:<28} {m['psnr']:>10.3f} "
              f"{m['sam']:>10.3f} {m['ergas']:>10.3f} "
              f"{m['params']/1e6:>9.3f}M")
    print("=" * 78)

    # Save results
    res_dir = os.path.join(_HERE, 'results')
    os.makedirs(res_dir, exist_ok=True)
    import json
    out_path = os.path.join(res_dir, 'test_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == '__main__':
    main()
