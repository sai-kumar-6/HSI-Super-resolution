"""
test_chikusei.py  —  V6 Edition
Evaluates V6 model on Chikusei test patches.

Usage
─────
  cd version6
  python testing/test_chikusei.py
  python testing/test_chikusei.py --checkpoint_v6 comparison/checkpoints/v6_best.pth
  python testing/test_chikusei.py \
      --checkpoint_v5 ../version5/comparison/checkpoints/v5_best.pth \
      --checkpoint_v6 comparison/checkpoints/v6_best.pth
"""
from __future__ import annotations
import sys, os, argparse, json

_HERE    = os.path.dirname(os.path.abspath(__file__))   # testing/
_V6      = os.path.dirname(_HERE)                        # version6/
_PROJECT = os.path.dirname(_V6)
_V5      = os.path.join(_PROJECT, 'version5')

import importlib.util

for p in [
    os.path.join(_PROJECT, 'version1'),
    os.path.join(_PROJECT, 'version3'),
    _PROJECT,
    os.path.join(_PROJECT, 'scripts'),
    os.path.join(_PROJECT, 'version4'),
    _V5,
    _V6,
    os.path.join(_V6, 'model'),
]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_module(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_bm = _load_module('baseline_models_v6',    os.path.join(_V6, 'model', 'baselines.py'))
_lf = _load_module('loss_functions_v6_mod', os.path.join(_V6, 'model', 'losses.py'))

create_vmamba_model = _bm.create_vmamba_model
count_parameters    = _bm.count_parameters
compute_psnr        = _lf.compute_psnr
compute_sam_metric  = _lf.compute_sam_metric
compute_ergas       = _lf.compute_ergas
compute_ssim        = _lf.compute_ssim

import torch
from dataset_loader_overlap import create_dataloaders_overlap

CHIKUSEI_PATH = os.path.join(_PROJECT, 'chikusei', 'chikusei.mat')
IN_CHANNELS   = 128
SCALE         = 4
D_MODEL       = 32
D_STATE       = 8
NUM_BLOCKS    = [1, 1, 1, 1]
PATCH_SIZE    = 32

VARIANT_LABELS = {
    'v5': 'V5 VMamba (True Parallel Scan)',
    'v6': 'V6 VMamba (Spectral-Focused Loss)',
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
        print(f"  [INFO] {variant}: random weights (no checkpoint found)")
    return model.eval()


def evaluate(model, test_l, device, max_patches):
    psnr_sum = sam_sum = ergas_sum = ssim_sum = 0.0
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
            ssim_sum  += compute_ssim(pred, hr_hsi)
            n += 1

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    return psnr_sum/n, sam_sum/n, ergas_sum/n, ssim_sum/n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_v5', default=None,
                        help='Path to V5 checkpoint for comparison')
    parser.add_argument('--checkpoint_v6', default=None,
                        help='Path to V6 checkpoint (default: comparison/checkpoints/v6_best.pth)')
    parser.add_argument('--max_patches',   default=20, type=int)
    args = parser.parse_args()

    # Default V6 checkpoint
    if args.checkpoint_v6 is None:
        args.checkpoint_v6 = os.path.join(_V6, 'comparison', 'checkpoints', 'v6_best.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print("Loading Chikusei test set ...")
    _, _, test_l = create_dataloaders_overlap(
        dataset='chikusei', mat_path=CHIKUSEI_PATH,
        batch_size=1, patch_size=PATCH_SIZE,
        overlap=PATCH_SIZE // 2, scale=SCALE, num_workers=0,
    )
    print(f"  Test patches: {len(test_l.dataset)}  (evaluating up to {args.max_patches})")

    ckpts = {'v6': args.checkpoint_v6}
    if args.checkpoint_v5:
        ckpts = {'v5': args.checkpoint_v5, 'v6': args.checkpoint_v6}

    print("\nLoading models ...")
    results = {}
    for v, ck in ckpts.items():
        m = load_model(v, ck, device)
        total, _ = count_parameters(m)
        psnr, sam, ergas, ssim = evaluate(m, test_l, device, args.max_patches)
        results[v] = {'psnr': psnr, 'sam': sam, 'ergas': ergas, 'ssim': ssim, 'params': total}

    print("\n" + "=" * 82)
    print(f"{'Variant':<35} {'PSNR (dB)':>10} {'SAM (deg)':>10} {'ERGAS':>8} {'SSIM':>8} {'Params':>9}")
    print("=" * 82)
    for v, m in results.items():
        label = VARIANT_LABELS.get(v, v)
        print(f"  {label:<33} {m['psnr']:>10.3f} {m['sam']:>10.3f} "
              f"{m['ergas']:>8.3f} {m['ssim']:>8.4f} {m['params']/1e6:>8.3f}M")
    print("=" * 82)

    if 'v5' in results and 'v6' in results:
        print("\n  V6 vs V5 improvement:")
        print(f"    PSNR  : {results['v6']['psnr']  - results['v5']['psnr']:+.3f} dB")
        print(f"    SAM   : {results['v6']['sam']   - results['v5']['sam']:+.3f} deg  (negative = better)")
        print(f"    ERGAS : {results['v6']['ergas'] - results['v5']['ergas']:+.3f}    (negative = better)")
        print(f"    SSIM  : {results['v6']['ssim']  - results['v5']['ssim']:+.4f}")

    res_dir = os.path.join(_HERE, 'results')
    os.makedirs(res_dir, exist_ok=True)
    out_path = os.path.join(res_dir, 'test_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == '__main__':
    main()
