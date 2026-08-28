"""
compare_inference.py
====================
Runs inference on a single test patch with ALL methods and generates
a side-by-side visual comparison.

Methods compared:
  Bicubic | PanNet | PanGAN | PSGAN | Panformer | V6 (ours)

Usage
-----
  cd version6/baselines
  python compare_inference.py                    # patch index 0
  python compare_inference.py --patch_idx 5
  python compare_inference.py --patch_idx 5 --bands 60 30 10
"""

import sys, os, argparse, importlib.util
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_HERE    = os.path.dirname(os.path.abspath(__file__))
_V6      = os.path.dirname(_HERE)
_PROJECT = os.path.dirname(_V6)
_V5      = os.path.join(_PROJECT, 'version5')

for p in [_PROJECT, os.path.join(_PROJECT, 'scripts'), _V6, _V5,
          os.path.join(_PROJECT,'version1'),
          os.path.join(_PROJECT,'version3'),
          os.path.join(_PROJECT,'version4'),
          _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)

from baseline_architectures import create_baseline, ALL_BASELINES
from dataset_loader_overlap  import create_dataloaders_overlap

CHIKUSEI_PATH = os.path.join(_PROJECT, 'chikusei', 'chikusei.mat')
CKPT_DIR      = os.path.join(_HERE,    'checkpoints')
V6_CKPT       = os.path.join(_V6,      'comparison', 'checkpoints', 'v6_best.pth')
OUT_DIR       = os.path.join(_V6,      'inference_results', 'comparison')
IN_CH, SCALE  = 128, 4
PATCH_SIZE    = 32

os.makedirs(OUT_DIR, exist_ok=True)


def _load_module(name, path):
    if name in sys.modules: return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m    = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_v6(device):
    _bm  = _load_module('baseline_models_v6',    os.path.join(_V6, 'baseline_models.py'))
    _lf  = _load_module('loss_functions_v6_mod', os.path.join(_V6, 'loss_functions_v6.py'))
    model = _bm.create_vmamba_model(
        'v6', in_ch=IN_CH, out_ch=IN_CH, scale=SCALE,
        d_model=64, d_state=8, num_blocks=[1,1,1,1],
    ).to(device)
    if os.path.isfile(V6_CKPT):
        st = torch.load(V6_CKPT, map_location=device)
        if isinstance(st, dict) and 'model_state_dict' in st:
            st = st['model_state_dict']
        model.load_state_dict(st)
        print(f'  V6 checkpoint loaded')
    else:
        print(f'  [!] V6 checkpoint not found — random weights')
    return model, _lf


def load_baseline(method, device):
    model = create_baseline(method, IN_CH, SCALE).to(device)
    ckpt  = os.path.join(CKPT_DIR, f'{method}_best.pth')
    if os.path.isfile(ckpt):
        st = torch.load(ckpt, map_location=device)
        model.load_state_dict(st)
        print(f'  {method.upper():12s} checkpoint loaded')
    else:
        print(f'  {method.upper():12s} [!] no checkpoint — random/bicubic weights')
    return model


def false_colour(t, r=60, g=30, b=10):
    img = torch.stack([t[r], t[g], t[b]])
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)


def metrics(pred, gt, scale, lf):
    p = pred.unsqueeze(0); q = gt.unsqueeze(0)
    return {
        'PSNR':  round(lf.compute_psnr(p, q), 3),
        'SAM':   round(lf.compute_sam_metric(p, q), 4),
        'ERGAS': round(lf.compute_ergas(p, q, scale=scale), 3),
        'SSIM':  round(lf.compute_ssim(p, q), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_idx', default=0,        type=int)
    parser.add_argument('--bands',     default=[60,30,10], nargs=3, type=int)
    parser.add_argument('--pixel_r',   default=None,     type=int)
    parser.add_argument('--pixel_c',   default=None,     type=int)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # ── Data ─────────────────────────────────────────────────────────────────
    print('Loading test patches ...')
    _, _, test_l = create_dataloaders_overlap(
        dataset='chikusei', mat_path=CHIKUSEI_PATH,
        batch_size=1, patch_size=PATCH_SIZE,
        overlap=PATCH_SIZE//2, scale=SCALE, num_workers=0,
    )
    total = len(test_l.dataset)
    idx   = args.patch_idx % total
    batch = test_l.dataset[idx]

    lr_hsi = batch['lr_hsi'].unsqueeze(0).to(device)
    hr_pan = batch['hr_pan'].unsqueeze(0).to(device)
    hr_hsi = batch['hr_hsi'].to(device)
    print(f'Patch #{idx}  LR={tuple(lr_hsi.shape)}  HR={tuple(hr_hsi.shape)}\n')

    pr = args.pixel_r if args.pixel_r is not None else hr_hsi.shape[1] // 2
    pc = args.pixel_c if args.pixel_c is not None else hr_hsi.shape[2] // 2

    # ── Models ────────────────────────────────────────────────────────────────
    print('Loading models ...')
    v6_model, lf = load_v6(device)
    baseline_models = {m: load_baseline(m, device) for m in ALL_BASELINES}
    all_models = {**baseline_models, 'V6 (Ours)': v6_model}

    # Method display order and colors
    METHOD_INFO = {
        'bicubic':   ('Bicubic',    '#555555'),
        'pannet':    ('PanNet',     '#1565c0'),
        'pangan':    ('PanGAN',     '#c62828'),
        'psgan':     ('PSGAN',      '#6a1b9a'),
        'panformer': ('Panformer',  '#00695c'),
        'V6 (Ours)': ('V6 VMamba\n(Ours)', '#e65100'),
    }

    # ── Inference ─────────────────────────────────────────────────────────────
    print('\nRunning inference ...')
    results = {}
    for key, model in all_models.items():
        model.eval()
        with torch.no_grad():
            pred = model(lr_hsi, hr_pan).squeeze(0)
        pred = pred.clamp(0, 1)
        m    = metrics(pred, hr_hsi, SCALE, lf)
        results[key] = {'pred': pred, 'metrics': m}
        label = METHOD_INFO[key][0].replace('\n', ' ')
        print(f'  {label:<18}  PSNR={m["PSNR"]:.2f}  SAM={m["SAM"]:.4f}  '
              f'ERGAS={m["ERGAS"]:.3f}  SSIM={m["SSIM"]:.4f}')

    # ── Figure ────────────────────────────────────────────────────────────────
    n_methods = len(all_models) + 1   # +1 for GT
    br, bg, bb = args.bands

    fig = plt.figure(figsize=(4.2 * n_methods, 14), facecolor='white')
    gs  = gridspec.GridSpec(4, n_methods, figure=fig,
                            hspace=0.55, wspace=0.18)

    def clean_ax(ax):
        ax.set_facecolor('white')
        for sp in ax.spines.values():
            sp.set_color('#cccccc'); sp.set_linewidth(0.7)
        ax.tick_params(colors='#333', labelsize=8)

    # ── Row 0: False-colour composites ────────────────────────────────────────
    # GT first
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_gt.imshow(false_colour(hr_hsi.cpu(), br, bg, bb))
    ax_gt.set_title('Ground Truth\n(HR-HSI)', fontsize=10,
                    fontweight='bold', color='#1a1a1a')
    ax_gt.scatter([pc], [pr], c='red', s=60, marker='+', linewidths=2, zorder=5)
    ax_gt.axis('off')

    for col, (key, info) in enumerate(results.items(), start=1):
        pred  = info['pred'].cpu()
        label, color = METHOD_INFO[key]
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(false_colour(pred, br, bg, bb))
        ax.set_title(label, fontsize=10, fontweight='bold', color=color)
        ax.scatter([pc], [pr], c='red', s=60, marker='+', linewidths=2, zorder=5)
        ax.axis('off')

    # ── Row 1: Error maps ─────────────────────────────────────────────────────
    all_errs = [np.zeros_like(hr_hsi.cpu()[0].numpy())]  # GT placeholder
    for key, info in results.items():
        err = (info['pred'].cpu() - hr_hsi.cpu()).abs().mean(0).numpy()
        all_errs.append(err)
    vmax = max(e.max() for e in all_errs[1:])

    # GT: just show zeros
    ax0 = fig.add_subplot(gs[1, 0])
    ax0.imshow(np.zeros_like(all_errs[1]), cmap='hot', vmin=0, vmax=vmax)
    ax0.set_title('Error Map', fontsize=9, color='#1a1a1a')
    ax0.axis('off')

    for col, (key, info) in enumerate(results.items(), start=1):
        err   = (info['pred'].cpu() - hr_hsi.cpu()).abs().mean(0).numpy()
        label, color = METHOD_INFO[key]
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(err, cmap='hot', vmin=0, vmax=vmax)
        ax.set_title(f'Error — {label.split(chr(10))[0]}',
                     fontsize=9, color=color)
        ax.axis('off')
        if col == n_methods - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Row 2: Spectral curve at (pr, pc) ─────────────────────────────────────
    ax_spec = fig.add_subplot(gs[2, :])
    clean_ax(ax_spec)
    bands = np.arange(IN_CH)

    sp_gt = hr_hsi.cpu()[:, pr, pc].numpy()
    ax_spec.plot(bands, sp_gt, color='#1a1a1a', linewidth=2.5,
                 label='Ground Truth', zorder=10)

    for key, info in results.items():
        pred  = info['pred'].cpu()
        label, color = METHOD_INFO[key]
        lw    = 2.2 if 'V6' in key else 1.4
        ls    = '-'  if 'V6' in key else '--'
        alpha = 1.0  if 'V6' in key else 0.75
        ax_spec.plot(bands, pred[:, pr, pc].numpy(),
                     color=color, linewidth=lw, linestyle=ls,
                     alpha=alpha, label=label.replace('\n', ' '))

    ax_spec.set_title(f'Spectral Curve at Pixel ({pr}, {pc})',
                      fontsize=11, fontweight='bold', color='#1a1a1a')
    ax_spec.set_xlabel('Band Index', fontsize=9, color='#444')
    ax_spec.set_ylabel('Reflectance', fontsize=9, color='#444')
    ax_spec.legend(fontsize=8.5, ncol=4, framealpha=0.95,
                   edgecolor='#ccc', loc='best')
    ax_spec.grid(True, color='#e0e0e0', linestyle='--', linewidth=0.7)

    # ── Row 3: Metrics bar chart ──────────────────────────────────────────────
    metric_keys = ['PSNR', 'SAM', 'ERGAS', 'SSIM']
    axes_m = [fig.add_subplot(gs[3, i]) for i in range(4)]
    # Hide remaining cols in row 3
    for i in range(4, n_methods):
        fig.add_subplot(gs[3, i]).axis('off')

    method_labels = [METHOD_INFO[k][0].replace('\n',' ') for k in results]
    method_colors = [METHOD_INFO[k][1] for k in results]
    x = np.arange(len(results))

    for ax_m, mk in zip(axes_m, metric_keys):
        vals = [results[k]['metrics'][mk] for k in results]
        bars = ax_m.bar(x, vals, color=method_colors, width=0.6, alpha=0.85)
        ax_m.set_xticks(x)
        ax_m.set_xticklabels(method_labels, rotation=30, ha='right', fontsize=7.5)
        ax_m.set_title(mk, fontsize=10, fontweight='bold', color='#1a1a1a')
        clean_ax(ax_m)
        ax_m.grid(True, axis='y', color='#e0e0e0', linestyle='--', linewidth=0.7)
        # Value label on each bar
        for bar, v in zip(bars, vals):
            ax_m.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                      f'{v:.3f}', ha='center', va='bottom',
                      fontsize=7, fontweight='bold', color='#222')

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f'Pansharpening Methods Comparison — Chikusei Dataset\n'
        f'Test Patch #{idx}  |  False-colour bands R={br} G={bg} B={bb}  '
        f'|  Scale={SCALE}x  LR={PATCH_SIZE}×{PATCH_SIZE}  HR={PATCH_SIZE*SCALE}×{PATCH_SIZE*SCALE}',
        fontsize=13, fontweight='bold', color='#1a1a1a', y=1.01,
    )

    out = os.path.join(OUT_DIR, f'comparison_patch{idx:03d}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nSaved: {out}')

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f'\n{"Method":<18}  {"PSNR":>8}  {"SAM":>8}  {"ERGAS":>8}  {"SSIM":>8}')
    print('-' * 55)
    for key, info in results.items():
        m = info['metrics']
        label = METHOD_INFO[key][0].replace('\n', ' ')
        print(f'{label:<18}  {m["PSNR"]:>8.3f}  {m["SAM"]:>8.4f}  '
              f'{m["ERGAS"]:>8.3f}  {m["SSIM"]:>8.4f}')


if __name__ == '__main__':
    main()
