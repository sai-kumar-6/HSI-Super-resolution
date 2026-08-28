"""
inference_visual.py  —  V6 Edition
Runs inference on a single test patch and visualises:
  1. False-colour composite  : GT | Predicted | LR (bicubic)
  2. Per-pixel error map     : |GT - Pred| averaged across bands
  3. Spectral curve          : GT vs Pred at a chosen pixel
  4. Metrics summary table   : PSNR / SAM / ERGAS / SSIM

Usage
-----
  cd version6
  python inference_visual.py                          # patch index 0
  python inference_visual.py --patch_idx 5            # specific patch
  python inference_visual.py --patch_idx 10 --bands 30 60 90
"""
from __future__ import annotations
import sys, os, argparse, importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F

_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_V5      = os.path.join(_PROJECT, 'version5')

for p in [os.path.join(_PROJECT,'version1'), os.path.join(_PROJECT,'version3'),
          _PROJECT, os.path.join(_PROJECT, 'scripts'), os.path.join(_PROJECT,'version4'), _V5, _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _load(name, path):
    if name in sys.modules: return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_bm = _load('baseline_models_v6',    os.path.join(_HERE, 'model', 'baselines.py'))
_lf = _load('loss_functions_v6_mod', os.path.join(_HERE, 'model', 'losses.py'))

create_vmamba_model = _bm.create_vmamba_model
compute_psnr        = _lf.compute_psnr
compute_sam_metric  = _lf.compute_sam_metric
compute_ergas       = _lf.compute_ergas
compute_ssim        = _lf.compute_ssim

from dataset_loader_overlap import create_dataloaders_overlap

CHIKUSEI_PATH = os.path.join(_PROJECT, 'chikusei', 'chikusei.mat')
CKPT_PATH     = os.path.join(_HERE, 'comparison', 'checkpoints', 'v6_best.pth')
OUT_DIR       = os.path.join(_HERE, 'inference_results')
IN_CHANNELS   = 128
SCALE         = 4
D_MODEL       = 64
D_STATE       = 8
NUM_BLOCKS    = [1, 1, 1, 1]
PATCH_SIZE    = 32


def false_colour(tensor, b_r=60, b_g=30, b_b=10):
    """Tensor (C,H,W) -> RGB uint8 image using 3 bands."""
    img = torch.stack([tensor[b_r], tensor[b_g], tensor[b_b]], dim=0)  # (3,H,W)
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return (img.permute(1,2,0).numpy() * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_idx', default=0,  type=int,
                        help='Which test patch to visualise (0-based)')
    parser.add_argument('--bands',     default=[60, 30, 10], nargs=3, type=int,
                        help='Three band indices for false-colour R G B')
    parser.add_argument('--pixel_r',   default=None, type=int,
                        help='Row of pixel for spectral curve (default: centre)')
    parser.add_argument('--pixel_c',   default=None, type=int,
                        help='Col of pixel for spectral curve (default: centre)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print('Loading Chikusei test patches ...')
    _, _, test_l = create_dataloaders_overlap(
        dataset='chikusei', mat_path=CHIKUSEI_PATH,
        batch_size=1, patch_size=PATCH_SIZE,
        overlap=PATCH_SIZE // 2, scale=SCALE, num_workers=0,
    )
    total = len(test_l.dataset)
    print(f'  Test patches: {total}')

    idx = args.patch_idx % total
    batch = test_l.dataset[idx]
    lr_hsi = batch['lr_hsi'].unsqueeze(0)   # (1,128,32,32)
    hr_pan = batch['hr_pan'].unsqueeze(0)   # (1,1,128,128)
    hr_hsi = batch['hr_hsi']                # (128,128,128)

    # ── Load model ────────────────────────────────────────────────────────────
    print('Loading V6 model ...')
    model = create_vmamba_model(
        variant='v6', in_ch=IN_CHANNELS, out_ch=IN_CHANNELS,
        scale=SCALE, d_model=D_MODEL, d_state=D_STATE, num_blocks=NUM_BLOCKS,
    ).to(device)

    if os.path.isfile(CKPT_PATH):
        st = torch.load(CKPT_PATH, map_location=device)
        if isinstance(st, dict) and 'model_state_dict' in st:
            st = st['model_state_dict']
        model.load_state_dict(st)
        print(f'  Checkpoint: {CKPT_PATH}')
    else:
        print(f'  [WARNING] No checkpoint found — using random weights')

    model.eval()
    with torch.no_grad():
        pred = model(lr_hsi.to(device), hr_pan.to(device)).cpu().squeeze(0)  # (128,128,128)

    # ── Bicubic baseline ──────────────────────────────────────────────────────
    bicubic = F.interpolate(
        lr_hsi, scale_factor=SCALE, mode='bicubic', align_corners=False
    ).squeeze(0)   # (128,128,128)

    # ── Metrics ───────────────────────────────────────────────────────────────
    pred_b  = pred.unsqueeze(0)
    gt_b    = hr_hsi.unsqueeze(0)
    bic_b   = bicubic.unsqueeze(0)

    metrics_v6  = dict(
        psnr  = compute_psnr(pred_b, gt_b),
        sam   = compute_sam_metric(pred_b, gt_b),
        ergas = compute_ergas(pred_b, gt_b, scale=SCALE),
        ssim  = compute_ssim(pred_b, gt_b),
    )
    metrics_bic = dict(
        psnr  = compute_psnr(bic_b, gt_b),
        sam   = compute_sam_metric(bic_b, gt_b),
        ergas = compute_ergas(bic_b, gt_b, scale=SCALE),
        ssim  = compute_ssim(bic_b, gt_b),
    )

    print(f'\n  Patch #{idx}')
    print(f'  {"Metric":<8}  {"Bicubic":>10}  {"V6 Pred":>10}')
    print(f'  {"-"*32}')
    for k in ['psnr','sam','ergas','ssim']:
        fmt = '.4f' if k == 'ssim' else '.3f'
        print(f'  {k.upper():<8}  {metrics_bic[k]:>10{fmt}}  {metrics_v6[k]:>10{fmt}}')

    # ── False-colour images ───────────────────────────────────────────────────
    br, bg, bb = args.bands
    fc_gt   = false_colour(hr_hsi,  br, bg, bb)
    fc_pred = false_colour(pred,    br, bg, bb)
    fc_bic  = false_colour(bicubic, br, bg, bb)

    # ── Error map ─────────────────────────────────────────────────────────────
    err_v6  = (pred  - hr_hsi).abs().mean(dim=0).numpy()   # (128,128)
    err_bic = (bicubic - hr_hsi).abs().mean(dim=0).numpy()

    # Shared colour scale
    vmax = max(err_v6.max(), err_bic.max())

    # ── Spectral curves ───────────────────────────────────────────────────────
    H, W = hr_hsi.shape[1], hr_hsi.shape[2]
    pr = args.pixel_r if args.pixel_r is not None else H // 2
    pc = args.pixel_c if args.pixel_c is not None else W // 2

    bands   = np.arange(IN_CHANNELS)
    sp_gt   = hr_hsi[:, pr, pc].numpy()
    sp_pred = pred[:,   pr, pc].numpy()
    sp_bic  = bicubic[:, pr, pc].numpy()

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 13), facecolor='white')
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # Row 0: False-colour images
    ax_gt   = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_bic  = fig.add_subplot(gs[0, 2])
    ax_met  = fig.add_subplot(gs[0, 3])

    # Row 1: Error maps
    ax_ev6  = fig.add_subplot(gs[1, 0])
    ax_ebic = fig.add_subplot(gs[1, 1])
    ax_spec = fig.add_subplot(gs[1, 2:])

    # Row 2: Band-by-band difference
    ax_bd_v6  = fig.add_subplot(gs[2, 0:2])
    ax_bd_bic = fig.add_subplot(gs[2, 2:])

    fig.suptitle(
        f'V6 Inference — Test Patch #{idx}   '
        f'(LR: {PATCH_SIZE}×{PATCH_SIZE}  HR: {H}×{W}  Scale: {SCALE}×)',
        fontsize=14, fontweight='bold', color='#1a1a1a', y=1.01,
    )

    def clean_ax(ax):
        ax.set_facecolor('white')
        for sp in ax.spines.values():
            sp.set_color('#cccccc'); sp.set_linewidth(0.8)
        ax.tick_params(colors='#333', labelsize=8)

    # ── False-colour panels ───────────────────────────────────────────────────
    for ax, img, ttl in [(ax_gt, fc_gt, 'Ground Truth (HR)'),
                          (ax_pred, fc_pred, 'V6 Prediction'),
                          (ax_bic, fc_bic, 'Bicubic Baseline')]:
        ax.imshow(img)
        ax.set_title(ttl, fontsize=10, fontweight='bold', pad=4)
        ax.axis('off')
        # Mark spectral curve pixel
        ax.scatter([pc], [pr], c='red', s=50, marker='+', linewidths=1.5, zorder=5)

    # ── Metrics table panel ───────────────────────────────────────────────────
    ax_met.set_facecolor('white')
    ax_met.axis('off')
    rows = [['Metric', 'Bicubic', 'V6 Pred', 'Gain'],
            ['PSNR (dB) ↑',
             f"{metrics_bic['psnr']:.3f}",
             f"{metrics_v6['psnr']:.3f}",
             f"{metrics_v6['psnr']-metrics_bic['psnr']:+.3f}"],
            ['SAM (°) ↓',
             f"{metrics_bic['sam']:.3f}",
             f"{metrics_v6['sam']:.3f}",
             f"{metrics_v6['sam']-metrics_bic['sam']:+.3f}"],
            ['ERGAS ↓',
             f"{metrics_bic['ergas']:.3f}",
             f"{metrics_v6['ergas']:.3f}",
             f"{metrics_v6['ergas']-metrics_bic['ergas']:+.3f}"],
            ['SSIM ↑',
             f"{metrics_bic['ssim']:.4f}",
             f"{metrics_v6['ssim']:.4f}",
             f"{metrics_v6['ssim']-metrics_bic['ssim']:+.4f}"]]
    tbl = ax_met.table(cellText=rows[1:], colLabels=rows[0],
                       loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)
    # Style header
    for j in range(4):
        tbl[(0, j)].set_facecolor('#1565c0')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    # Style gain column: green if improved, red if worse
    gain_col = 3
    for i, k in enumerate(['psnr','sam','ergas','ssim'], start=1):
        gain = metrics_v6[k] - metrics_bic[k]
        better = gain > 0 if k in ['psnr','ssim'] else gain < 0
        tbl[(i, gain_col)].set_facecolor('#c8e6c9' if better else '#ffcdd2')
    ax_met.set_title('Patch Metrics', fontsize=10, fontweight='bold', pad=4)

    # ── Error maps ────────────────────────────────────────────────────────────
    im1 = ax_ev6.imshow(err_v6,  cmap='hot', vmin=0, vmax=vmax)
    ax_ev6.set_title('Error Map — V6 Prediction\n(mean |GT - Pred| across bands)',
                     fontsize=9, fontweight='bold', pad=4)
    ax_ev6.axis('off')
    plt.colorbar(im1, ax=ax_ev6, fraction=0.046, pad=0.04)

    im2 = ax_ebic.imshow(err_bic, cmap='hot', vmin=0, vmax=vmax)
    ax_ebic.set_title('Error Map — Bicubic Baseline\n(mean |GT - Pred| across bands)',
                      fontsize=9, fontweight='bold', pad=4)
    ax_ebic.axis('off')
    plt.colorbar(im2, ax=ax_ebic, fraction=0.046, pad=0.04)

    # ── Spectral curve ────────────────────────────────────────────────────────
    clean_ax(ax_spec)
    ax_spec.plot(bands, sp_gt,   color='#1a1a1a', linewidth=2.0, label='Ground Truth', zorder=3)
    ax_spec.plot(bands, sp_pred, color='#1565c0', linewidth=1.8, linestyle='-',
                 label='V6 Prediction', zorder=4)
    ax_spec.plot(bands, sp_bic,  color='#e53935', linewidth=1.4, linestyle='--',
                 label='Bicubic', zorder=2, alpha=0.8)
    ax_spec.set_title(f'Spectral Curve at Pixel ({pr}, {pc})',
                      fontsize=10, fontweight='bold', pad=4)
    ax_spec.set_xlabel('Band Index', fontsize=9, color='#444')
    ax_spec.set_ylabel('Reflectance', fontsize=9, color='#444')
    ax_spec.legend(fontsize=8.5, framealpha=0.95, edgecolor='#ccc')
    ax_spec.grid(True, color='#e0e0e0', linestyle='--', linewidth=0.7)

    # ── Band-by-band absolute error ───────────────────────────────────────────
    mean_err_v6  = (pred   - hr_hsi).abs().mean(dim=[1,2]).numpy()
    mean_err_bic = (bicubic - hr_hsi).abs().mean(dim=[1,2]).numpy()

    clean_ax(ax_bd_v6)
    ax_bd_v6.bar(bands, mean_err_v6, color='#1565c0', alpha=0.75, width=0.9)
    ax_bd_v6.set_title('V6: Mean Absolute Error per Band',
                        fontsize=10, fontweight='bold', pad=4)
    ax_bd_v6.set_xlabel('Band Index', fontsize=9, color='#444')
    ax_bd_v6.set_ylabel('Mean |error|', fontsize=9, color='#444')
    ax_bd_v6.grid(True, axis='y', color='#e0e0e0', linestyle='--', linewidth=0.7)

    clean_ax(ax_bd_bic)
    ax_bd_bic.bar(bands, mean_err_bic, color='#e53935', alpha=0.75, width=0.9)
    ax_bd_bic.set_title('Bicubic: Mean Absolute Error per Band',
                         fontsize=10, fontweight='bold', pad=4)
    ax_bd_bic.set_xlabel('Band Index', fontsize=9, color='#444')
    ax_bd_bic.set_ylabel('Mean |error|', fontsize=9, color='#444')
    ax_bd_bic.grid(True, axis='y', color='#e0e0e0', linestyle='--', linewidth=0.7)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, f'inference_patch{idx:03d}.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\n  Saved: {out_path}')


if __name__ == '__main__':
    main()
