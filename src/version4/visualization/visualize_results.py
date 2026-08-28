"""
visualize_results.py  —  V4 Edition
Generates publication-quality output images and spectral plots for all 4 models.

Saves to version4/visualization/outputs/:
  - false_colour_comparison.png      — RGB composites for all models
  - error_map_comparison.png         — absolute error maps
  - spectral_curves.png              — spectral profile at selected pixel
  - individual/lr_hsi.png            — individual images
  - individual/hr_pan.png
  - individual/gt_hr_hsi.png
  - individual/<variant>_pred.png    — each model's output
  - individual/<variant>_error.png   — each model's error map

Usage
─────
  # With trained checkpoints:
  python visualization/visualize_results.py \\
      --checkpoint_old      comparison/checkpoints/old_best.pth \\
      --checkpoint_improved comparison/checkpoints/improved_best.pth \\
      --checkpoint_v3       comparison/checkpoints/v3_best.pth \\
      --checkpoint_v4       comparison/checkpoints/v4_best.pth
"""

import sys
import os
import argparse
import numpy as np

_HERE    = os.path.dirname(os.path.abspath(__file__))
_V4      = os.path.dirname(_HERE)
_PROJECT = os.path.dirname(_V4)
sys.path.insert(0, os.path.join(_PROJECT, 'version3'))  # lowest priority
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, 'scripts'))
sys.path.insert(0, _V4)                                  # highest priority
sys.path.insert(0, os.path.join(_V4, 'model'))

import torch
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    print("[WARN] matplotlib not found — no plots will be generated")
    HAS_MPL = False

from dataset_overlap import create_dataloaders_overlap
from baselines       import create_vmamba_model, count_parameters
from losses          import compute_psnr, compute_sam_metric, compute_ergas

CHIKUSEI_PATH = os.path.join(_PROJECT, 'chikusei', 'chikusei.mat')
OUT_DIR       = os.path.join(_HERE, 'outputs')
INDIV_DIR     = os.path.join(OUT_DIR, 'individual')
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
}
COLOURS = {
    'old': 'tab:blue', 'improved': 'tab:orange',
    'v3':  'tab:green', 'v4': 'tab:purple',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def hsi_to_rgb(t, bands=(30, 20, 10)):
    """(C,H,W) → (H,W,3) numpy float32 in [0,1]."""
    C = t.shape[0]
    r = t[min(bands[0], C-1)].cpu().numpy()
    g = t[min(bands[1], C-1)].cpu().numpy()
    b = t[min(bands[2], C-1)].cpu().numpy()
    rgb = np.stack([r, g, b], axis=2)
    vmin, vmax = rgb.min(), rgb.max()
    if vmax > vmin:
        rgb = (rgb - vmin) / (vmax - vmin)
    return np.clip(rgb, 0, 1)


def save_single(arr, title, path, cmap='viridis'):
    """Save a single panel image."""
    if not HAS_MPL:
        return
    plt.figure(figsize=(5, 5))
    if arr.ndim == 3:
        plt.imshow(arr)
    else:
        plt.imshow(arr, cmap=cmap)
    plt.title(title, fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


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
        print(f"  [INFO] {variant}: random weights")
    return model.eval()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_old',      default=None)
    parser.add_argument('--checkpoint_improved', default=None)
    parser.add_argument('--checkpoint_v3',       default=None)
    parser.add_argument('--checkpoint_v4',       default=None)
    parser.add_argument('--patch_idx', default=0, type=int,
                        help='Which test patch to visualise')
    parser.add_argument('--pixel_y',   default=16, type=int)
    parser.add_argument('--pixel_x',   default=16, type=int)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(INDIV_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading Chikusei …")
    _, _, test_l = create_dataloaders_overlap(
        dataset='chikusei', mat_path=CHIKUSEI_PATH,
        batch_size=1, patch_size=PATCH_SIZE,
        overlap=PATCH_SIZE // 2, scale=SCALE, num_workers=0,
    )
    n = len(test_l.dataset)
    idx = min(args.patch_idx, n - 1)
    batch = test_l.dataset[idx]
    # Add batch dim
    lr_hsi = batch['lr_hsi'].unsqueeze(0).to(device)  # (1,128,8,8)
    hr_pan = batch['hr_pan'].unsqueeze(0).to(device)  # (1,1,32,32)
    hr_hsi = batch['hr_hsi'].unsqueeze(0)             # (1,128,32,32) on CPU for GT

    print(f"  Using patch #{idx} of {n}")

    # ── Load models ───────────────────────────────────────────────────────────
    ckpts = {
        'old':      args.checkpoint_old,
        'improved': args.checkpoint_improved,
        'v3':       args.checkpoint_v3,
        'v4':       args.checkpoint_v4,
    }
    models = {}
    for v, ck in ckpts.items():
        models[v] = load_model(v, ck, device)

    # ── Run inference ─────────────────────────────────────────────────────────
    preds = {}
    metrics = {}
    with torch.no_grad():
        for v, m in models.items():
            pred = m(lr_hsi, hr_pan).cpu()
            preds[v] = pred
            metrics[v] = {
                'psnr':  compute_psnr(pred, hr_hsi),
                'sam':   compute_sam_metric(pred, hr_hsi),
                'ergas': compute_ergas(pred, hr_hsi, scale=SCALE),
                'params': count_parameters(m)[0],
            }

    # ── Save individual images ────────────────────────────────────────────────
    lr_up = F.interpolate(lr_hsi.cpu(), scale_factor=SCALE,
                          mode='bicubic', align_corners=False)

    save_single(hsi_to_rgb(lr_up[0]), 'LR-HSI (Bicubic upsampled)',
                os.path.join(INDIV_DIR, 'lr_hsi.png'))
    save_single(hr_pan[0, 0].cpu().numpy(), 'HR-PAN',
                os.path.join(INDIV_DIR, 'hr_pan.png'), cmap='gray')
    save_single(hsi_to_rgb(hr_hsi[0]), 'Ground Truth HR-HSI',
                os.path.join(INDIV_DIR, 'gt_hr_hsi.png'))

    for v, pred in preds.items():
        m_info = metrics[v]
        label = (f"{VARIANT_LABELS[v]}\n"
                 f"PSNR={m_info['psnr']:.2f}dB  "
                 f"SAM={m_info['sam']:.3f}°  "
                 f"ERGAS={m_info['ergas']:.3f}")
        save_single(hsi_to_rgb(pred[0]), label,
                    os.path.join(INDIV_DIR, f'{v}_pred.png'))

        err = (pred[0] - hr_hsi[0]).abs().mean(0).numpy()
        save_single(err, f'{VARIANT_LABELS[v]}\nMean Abs Error',
                    os.path.join(INDIV_DIR, f'{v}_error.png'), cmap='hot')

    if not HAS_MPL:
        print("matplotlib not available — skipping composite figures.")
        return

    # ── Composite: false colour comparison ───────────────────────────────────
    VARIANTS = list(preds.keys())
    n_cols = 3 + len(VARIANTS)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5))

    axes[0].imshow(hsi_to_rgb(lr_up[0]))
    axes[0].set_title('LR-HSI\n(Bicubic)', fontsize=9)
    axes[0].axis('off')

    axes[1].imshow(hr_pan[0, 0].cpu().numpy(), cmap='gray')
    axes[1].set_title('HR-PAN', fontsize=9)
    axes[1].axis('off')

    axes[2].imshow(hsi_to_rgb(hr_hsi[0]))
    axes[2].set_title('GT HR-HSI', fontsize=9)
    axes[2].axis('off')

    for ci, v in enumerate(VARIANTS, 3):
        m_info = metrics[v]
        axes[ci].imshow(hsi_to_rgb(preds[v][0].clamp(0, 1)))
        axes[ci].set_title(
            f"{VARIANT_LABELS[v]}\n"
            f"PSNR={m_info['psnr']:.2f}  SAM={m_info['sam']:.3f}",
            fontsize=7,
        )
        axes[ci].axis('off')

    plt.suptitle('False Colour Comparison — Bands (30, 20, 10)', fontsize=11)
    plt.tight_layout()
    fcc_path = os.path.join(OUT_DIR, 'false_colour_comparison.png')
    plt.savefig(fcc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {fcc_path}")

    # ── Composite: error maps ─────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, len(VARIANTS), figsize=(5 * len(VARIANTS), 4))
    if len(VARIANTS) == 1:
        axes2 = [axes2]

    all_err_max = max(
        (preds[v][0] - hr_hsi[0]).abs().mean(0).numpy().max()
        for v in VARIANTS
    )

    for ax, v in zip(axes2, VARIANTS):
        err = (preds[v][0] - hr_hsi[0]).abs().mean(0).numpy()
        im = ax.imshow(err, cmap='hot', vmin=0, vmax=all_err_max, aspect='auto')
        ax.set_title(
            f"{VARIANT_LABELS[v]}\n"
            f"PSNR={metrics[v]['psnr']:.2f}  ERGAS={metrics[v]['ergas']:.3f}",
            fontsize=8,
        )
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Spectral-Mean Absolute Error (same colour scale)', fontsize=11)
    plt.tight_layout()
    err_path = os.path.join(OUT_DIR, 'error_map_comparison.png')
    plt.savefig(err_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {err_path}")

    # ── Spectral curves ───────────────────────────────────────────────────────
    py = min(args.pixel_y, hr_hsi.shape[2] - 1)
    px = min(args.pixel_x, hr_hsi.shape[3] - 1)
    gt_spec = hr_hsi[0, :, py, px].numpy()

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # Left: all curves overlaid
    ax = axes3[0]
    bands = np.arange(IN_CHANNELS)
    ax.plot(bands, gt_spec, 'k-', linewidth=2.5, label='Ground Truth', zorder=5)
    for v in VARIANTS:
        spec = preds[v][0, :, py, px].numpy()
        ax.plot(bands, spec, '--', color=COLOURS[v],
                linewidth=1.5, label=VARIANT_LABELS[v])
    ax.set_xlabel('Band index', fontsize=11)
    ax.set_ylabel('Reflectance', fontsize=11)
    ax.set_title(f'Spectral Curve at Pixel ({py}, {px})', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: spectral gradient (band differences)
    ax2 = axes3[1]
    gt_grad = np.diff(gt_spec)
    ax2.plot(bands[1:], gt_grad, 'k-', linewidth=2.5, label='Ground Truth', zorder=5)
    for v in VARIANTS:
        spec = preds[v][0, :, py, px].numpy()
        grad = np.diff(spec)
        ax2.plot(bands[1:], grad, '--', color=COLOURS[v],
                 linewidth=1.5, label=VARIANT_LABELS[v])
    ax2.axhline(0, color='gray', linewidth=0.8, linestyle=':')
    ax2.set_xlabel('Band index', fontsize=11)
    ax2.set_ylabel('∇c (band difference)', fontsize=11)
    ax2.set_title('Spectral Gradient ∇c(H) — shape preservation', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Spectral Analysis', fontsize=13)
    plt.tight_layout()
    spec_path = os.path.join(OUT_DIR, 'spectral_curves.png')
    plt.savefig(spec_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {spec_path}")

    # ── Metric summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Variant':<28} {'PSNR (↑)':>10} {'SAM (↓)':>10} "
          f"{'ERGAS (↓)':>10} {'Params':>10}")
    print("=" * 70)
    for v in VARIANTS:
        m = metrics[v]
        print(f"  {VARIANT_LABELS[v]:<26} {m['psnr']:>10.3f} "
              f"{m['sam']:>10.3f} {m['ergas']:>10.3f} "
              f"{m['params']/1e6:>9.3f}M")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
