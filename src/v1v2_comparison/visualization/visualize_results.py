"""
visualize_results.py
M.Tech Thesis – Hyperspectral Pansharpening

Generates a rich visualisation figure comparing Old VMamba vs Improved VMamba
on a single test patch from the Chikusei dataset.

Output layout
-------------
Row 1 (5 columns):
    LR-HSI (RGB)  |  HR-PAN  |  GT HR-HSI (RGB)  |  Old output (RGB)  |  Improved output (RGB)

Row 2 (4 columns + 1 wide):
    Error map – Old  |  Error map – Improved  |  Spectral profile (centre pixel)  |  Band-50 comparison (GT / Old / Improved)

Usage
-----
    python visualization/visualize_results.py
    python visualization/visualize_results.py \\
        --checkpoint_old   comparison/checkpoints/old_best.pth \\
        --checkpoint_improved comparison/checkpoints/improved_best.pth \\
        --patch_idx 5
"""

import sys
import os
import argparse
from datetime import datetime

# Make the project root importable from this subfolder
_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, 'scripts'))

import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dataset_loader  import load_chikusei_dataset, PansharpeningDataset
from baseline_models import create_vmamba_model
from loss_functions  import compute_psnr, compute_sam_metric


# ============================================================================
# Configuration
# ============================================================================

CHIKUSEI_PATH = os.path.join(_PROJECT, 'chikusei', 'chikusei.mat')
OUTPUT_DIR    = os.path.join(_HERE, 'outputs')
IN_CHANNELS   = 128
SCALE         = 4
D_MODEL       = 64
NUM_BLOCKS    = [2, 3, 3, 2]

# Pseudo-RGB bands for Chikusei (128 bands, ~380–1050 nm)
#   Band 30 ≈ Red, Band 20 ≈ Green, Band 10 ≈ Blue (approximate)
RGB_BANDS = [30, 20, 10]

# Band to show in the per-band comparison plot
VIS_BAND = 50


# ============================================================================
# Helpers
# ============================================================================

def load_model(variant: str, checkpoint_path, device):
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
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        print(f"  [OK] Loaded {variant} weights from {checkpoint_path}")
    else:
        print(f"  [INFO] {variant}: using random weights (no checkpoint)")

    model.eval()
    return model


def to_rgb(tensor_chw: torch.Tensor, bands=None) -> np.ndarray:
    """
    Extract 3 bands from (C, H, W) tensor and return a uint8 (H, W, 3) image.
    Values are normalised to [0, 1] before display.
    """
    if bands is None:
        bands = RGB_BANDS
    bands = [min(b, tensor_chw.shape[0] - 1) for b in bands]
    rgb   = tensor_chw[bands].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    # Normalise per-channel for display
    for c in range(3):
        lo, hi = rgb[:, :, c].min(), rgb[:, :, c].max()
        if hi > lo:
            rgb[:, :, c] = (rgb[:, :, c] - lo) / (hi - lo)
        else:
            rgb[:, :, c] = 0.0
    return np.clip(rgb, 0, 1)


def error_map(pred: torch.Tensor, gt: torch.Tensor) -> np.ndarray:
    """Mean absolute error across bands → (H, W) numpy float."""
    err = (pred - gt).abs().mean(dim=0).cpu().numpy()
    return err


def get_spectral_profile(tensor_chw: torch.Tensor, h: int, w: int) -> np.ndarray:
    """Return spectral vector at pixel (h, w)."""
    return tensor_chw[:, h, w].cpu().numpy()


# ============================================================================
# Main visualisation
# ============================================================================

def visualise(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice  : {device}")
    print(f"Dataset : {CHIKUSEI_PATH}")

    # ---- Dataset ----
    print("Loading Chikusei dataset …")
    _, test_hsi = load_chikusei_dataset(CHIKUSEI_PATH, train_ratio=0.8)

    test_dataset = PansharpeningDataset(
        test_hsi,
        patch_size = 16,
        scale      = SCALE,
        overlap    = 0.25,
        augment    = False,
    )
    n_patches = len(test_dataset)
    patch_idx = max(0, min(args.patch_idx, n_patches - 1))
    print(f"Total test patches : {n_patches}  |  Visualising patch #{patch_idx}")

    sample = test_dataset[patch_idx]
    lr_hsi = sample['lr_hsi'].unsqueeze(0).to(device)   # (1, C, H,  W)
    hr_pan = sample['hr_pan'].unsqueeze(0).to(device)   # (1, 1, rH, rW)
    hr_hsi = sample['hr_hsi'].unsqueeze(0).to(device)   # (1, C, rH, rW)

    # ---- Models ----
    print("Building models …")
    model_old = load_model('old',      args.checkpoint_old,      device)
    model_imp = load_model('improved', args.checkpoint_improved, device)

    # ---- Inference ----
    with torch.no_grad():
        out_old = model_old(lr_hsi, hr_pan)   # (1, C, rH, rW)
        out_imp = model_imp(lr_hsi, hr_pan)

    # Move to cpu / squeeze batch dim
    lr_cpu  = lr_hsi[0].cpu()    # (C, H, W)
    pan_cpu = hr_pan[0, 0].cpu() # (rH, rW)
    gt_cpu  = hr_hsi[0].cpu()    # (C, rH, rW)
    old_cpu = out_old[0].cpu()
    imp_cpu = out_imp[0].cpu()

    # Clamp predictions to [0,1] for display
    old_cpu = old_cpu.clamp(0, 1)
    imp_cpu = imp_cpu.clamp(0, 1)

    _, rH, rW = gt_cpu.shape
    cx, cy = rH // 2, rW // 2    # centre pixel

    # ---- Metrics ----
    psnr_old = compute_psnr(out_old, hr_hsi)
    psnr_imp = compute_psnr(out_imp, hr_hsi)
    sam_old  = compute_sam_metric(out_old, hr_hsi)
    sam_imp  = compute_sam_metric(out_imp, hr_hsi)
    print(f"\nPatch #{patch_idx} metrics:")
    print(f"  Old VMamba    PSNR={psnr_old:.2f} dB   SAM={sam_old:.3f}°")
    print(f"  Improved VMamba PSNR={psnr_imp:.2f} dB   SAM={sam_imp:.3f}°")

    # ---- Build figure ----
    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(
        f'Chikusei Patch #{patch_idx} – Old vs Improved VMamba\n'
        f'Old: PSNR={psnr_old:.2f} dB, SAM={sam_old:.3f}°  |  '
        f'Improved: PSNR={psnr_imp:.2f} dB, SAM={sam_imp:.3f}°',
        fontsize=13, fontweight='bold'
    )

    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.3)

    # ---- Row 1 ----
    # LR-HSI (upsampled for display)
    ax = fig.add_subplot(gs[0, 0])
    lr_rgb = to_rgb(lr_cpu)
    lr_rgb_up = np.kron(lr_rgb, np.ones((SCALE, SCALE, 1)))[:rH, :rW, :]
    ax.imshow(lr_rgb_up)
    ax.set_title('LR-HSI (RGB)', fontsize=9)
    ax.axis('off')

    # HR-PAN
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(pan_cpu.numpy(), cmap='gray')
    ax.set_title('HR-PAN', fontsize=9)
    ax.axis('off')

    # GT HR-HSI
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(to_rgb(gt_cpu))
    ax.set_title('GT HR-HSI (RGB)', fontsize=9)
    ax.axis('off')

    # Old VMamba output
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(to_rgb(old_cpu))
    ax.set_title(f'Old VMamba\nPSNR={psnr_old:.2f} dB', fontsize=9)
    ax.axis('off')

    # Improved VMamba output
    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(to_rgb(imp_cpu))
    ax.set_title(f'Improved VMamba\nPSNR={psnr_imp:.2f} dB', fontsize=9)
    ax.axis('off')

    # ---- Row 2 ----
    # Error map – Old
    err_old = error_map(old_cpu, gt_cpu)
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(err_old, cmap='hot', vmin=0)
    ax.set_title('Error Map – Old\n(|pred−GT| mean over bands)', fontsize=9)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Error map – Improved
    err_imp = error_map(imp_cpu, gt_cpu)
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(err_imp, cmap='hot', vmin=0,
                   vmax=max(err_old.max(), err_imp.max()))
    ax.set_title('Error Map – Improved\n(|pred−GT| mean over bands)', fontsize=9)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Spectral profile at centre pixel
    ax = fig.add_subplot(gs[1, 2])
    bands_x = np.arange(IN_CHANNELS)
    ax.plot(bands_x, get_spectral_profile(gt_cpu,  cx, cy), label='GT',       linewidth=1.5, color='black')
    ax.plot(bands_x, get_spectral_profile(old_cpu, cx, cy), label='Old',      linewidth=1.2, color='steelblue',  linestyle='--')
    ax.plot(bands_x, get_spectral_profile(imp_cpu, cx, cy), label='Improved', linewidth=1.2, color='darkorange', linestyle='-.')
    ax.set_title(f'Spectral Profile at\ncentre pixel ({cx},{cy})', fontsize=9)
    ax.set_xlabel('Band index')
    ax.set_ylabel('Reflectance')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Band-VIS_BAND comparison
    vb = min(VIS_BAND, IN_CHANNELS - 1)
    ax = fig.add_subplot(gs[1, 3])
    vmin = gt_cpu[vb].min().item()
    vmax = gt_cpu[vb].max().item()
    ax.imshow(gt_cpu[vb].numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f'GT  – Band {vb}', fontsize=9)
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 4])
    # Split the cell to show old and improved side-by-side
    # Use text annotations to label; show improved only (space limited)
    ax.imshow(imp_cpu[vb].numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f'Improved – Band {vb}', fontsize=9)
    ax.axis('off')

    # ---- Save ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(OUTPUT_DIR, f'comparison_{timestamp}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nMain figure saved → {save_path}")

    # ---- Band comparison figure (GT / Old / Improved for VIS_BAND) ----
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    fig2.suptitle(f'Band {vb} Comparison – Chikusei Patch #{patch_idx}', fontsize=12)
    for ax2, img, title in zip(
        axes2,
        [gt_cpu[vb], old_cpu[vb], imp_cpu[vb]],
        ['Ground Truth', 'Old VMamba', 'Improved VMamba']
    ):
        ax2.imshow(img.numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax2.set_title(title, fontsize=10)
        ax2.axis('off')
    plt.tight_layout()
    band_path = os.path.join(OUTPUT_DIR, f'band{vb}_comparison_{timestamp}.png')
    fig2.savefig(band_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Band figure  saved → {band_path}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualise Old vs Improved VMamba on a Chikusei test patch'
    )
    parser.add_argument('--checkpoint_old',      default=None, type=str,
                        help='Path to Old VMamba checkpoint (.pth)')
    parser.add_argument('--checkpoint_improved', default=None, type=str,
                        help='Path to Improved VMamba checkpoint (.pth)')
    parser.add_argument('--patch_idx',           default=0,    type=int,
                        help='Index of the test patch to visualise (default 0)')
    args = parser.parse_args()
    visualise(args)


if __name__ == '__main__':
    main()
