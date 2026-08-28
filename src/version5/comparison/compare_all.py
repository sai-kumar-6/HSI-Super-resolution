"""
compare_all.py  —  V5 Edition
Trains all 5 model variants and saves a side-by-side comparison.

Usage
─────
  cd version5
  python comparison/compare_all.py --epochs 5 --batch_size 1 --patch_size 32
  python comparison/compare_all.py --epochs 30 --skip_old --skip_improved --skip_v3 --skip_v4
"""
from __future__ import annotations
import sys, os, time, argparse, json

_HERE    = os.path.dirname(os.path.abspath(__file__))   # comparison/
_V5      = os.path.dirname(_HERE)                        # version5/
_PROJECT = os.path.dirname(_V5)

sys.path.insert(0, os.path.join(_PROJECT, 'version3'))  # lowest priority
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, 'scripts'))
sys.path.insert(0, os.path.join(_PROJECT, 'version4'))
sys.path.insert(0, _V5)                                  # highest priority

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset_loader_overlap import create_dataloaders_overlap
from baseline_models import create_vmamba_model, count_parameters
from loss_functions_v5 import CompositeLossV5, compute_psnr, compute_sam_metric, compute_ergas, compute_ssim

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import openpyxl
    HAS_XLSX = True
except ImportError:
    HAS_XLSX = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kw): return x

CHIKUSEI_PATH = os.path.join(_PROJECT, 'chikusei', 'chikusei.mat')
CKP_DIR      = os.path.join(_HERE, 'checkpoints')
IMG_DIR      = os.path.join(_HERE, 'saved_images')
PLOT_DIR     = os.path.join(_HERE, 'plots')
RES_DIR      = os.path.join(_HERE, 'results')
METRIC_PLOT_DIR = os.path.join(_HERE, 'metric_plots')
IN_CHANNELS = 128
SCALE       = 4

VARIANT_LABELS = {
    'old':      'V1 Old VMamba',
    'improved': 'V2 Improved VMamba',
    'v3':       'V3 VMamba',
    'v4':       'V4 VMamba (Spectral Fix)',
    'v5':       'V5 VMamba (True Parallel Scan)',
}
COLOURS = {
    'old': 'tab:blue', 'improved': 'tab:orange',
    'v3':  'tab:green', 'v4': 'tab:purple', 'v5': 'tab:red',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def hsi_to_rgb(t, bands=(30, 20, 10)):
    """(C,H,W) tensor → (H,W,3) numpy float32 in [0,1]."""
    import numpy as np
    C = t.shape[0]
    r = t[min(bands[0], C-1)].cpu().numpy()
    g = t[min(bands[1], C-1)].cpu().numpy()
    b = t[min(bands[2], C-1)].cpu().numpy()
    rgb = np.stack([r, g, b], axis=2)
    vmin, vmax = rgb.min(), rgb.max()
    if vmax > vmin:
        rgb = (rgb - vmin) / (vmax - vmin)
    import numpy as np
    return np.clip(rgb, 0, 1)


def save_images(epoch, variant, lr_hsi, hr_pan, pred, gt, out_dir):
    """Save false-colour and error images for one variant at one epoch."""
    if not HAS_MPL:
        return
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    lr_up = F.interpolate(lr_hsi.cpu(), scale_factor=SCALE,
                          mode='bicubic', align_corners=False)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(hsi_to_rgb(lr_up[0]));       axes[0].set_title('LR-HSI (bicubic)'); axes[0].axis('off')
    axes[1].imshow(hr_pan[0, 0].cpu().numpy(), cmap='gray'); axes[1].set_title('HR-PAN'); axes[1].axis('off')
    axes[2].imshow(hsi_to_rgb(gt[0].cpu()));    axes[2].set_title('Ground Truth'); axes[2].axis('off')
    axes[3].imshow(hsi_to_rgb(pred[0].cpu().clamp(0, 1))); axes[3].set_title(f'{VARIANT_LABELS[variant]}\nEp{epoch}'); axes[3].axis('off')
    err = (pred[0].cpu() - gt[0].cpu()).abs().mean(0).numpy()
    im = axes[4].imshow(err, cmap='hot'); axes[4].set_title('Error'); axes[4].axis('off')
    plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{variant}_ep{epoch:03d}.png'), dpi=120, bbox_inches='tight')
    plt.close()


def save_spectral_curves(epoch, variant, pred, gt, out_dir, py=16, px=16):
    """Save spectral profile at pixel (py, px)."""
    if not HAS_MPL:
        return
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    gt_spec   = gt[0, :, py, px].cpu().numpy()
    pred_spec = pred[0, :, py, px].detach().cpu().numpy()
    bands = np.arange(IN_CHANNELS)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bands, gt_spec, 'k-', linewidth=2, label='GT')
    ax.plot(bands, pred_spec, '--', color=COLOURS[variant], linewidth=1.5, label=VARIANT_LABELS[variant])
    ax.set_xlabel('Band'); ax.set_ylabel('Reflectance')
    ax.set_title(f'Spectral Curve — {VARIANT_LABELS[variant]}  Ep{epoch}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{variant}_spectral_ep{epoch:03d}.png'), dpi=120, bbox_inches='tight')
    plt.close()


def save_metric_plots(variant, history, epoch, out_dir):
    """
    Save a 5-panel figure (Loss, PSNR, SAM, ERGAS, SSIM vs epoch) to out_dir.
    Called every `plot_every` epochs and at the end of training.
    File name: <variant>_metrics_ep<epoch:04d>.png
    """
    if not HAS_MPL or not history:
        return
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)

    epochs = [h['epoch']      for h in history]
    loss   = [h['train_loss'] for h in history]
    psnr   = [h['val_psnr']   for h in history]
    sam    = [h['val_sam']    for h in history]
    ergas  = [h['val_ergas']  for h in history]
    ssim   = [h['val_ssim']   for h in history]

    color = COLOURS.get(variant, 'tab:blue')
    label = VARIANT_LABELS.get(variant, variant)

    metrics = [
        (loss,  'Training Loss',     'Loss',      '↓', '#e74c3c'),
        (psnr,  'Validation PSNR',   'PSNR (dB)', '↑', '#2980b9'),
        (sam,   'Validation SAM',    'SAM (°)',   '↓', '#27ae60'),
        (ergas, 'Validation ERGAS',  'ERGAS',     '↓', '#f39c12'),
        (ssim,  'Validation SSIM',   'SSIM',      '↑', '#8e44ad'),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(26, 4.5))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(f'{label}  —  Epoch {epoch}', fontsize=14,
                 color='white', y=1.01)

    for ax, (vals, title, ylabel, better, c) in zip(axes, metrics):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#aaa', labelsize=9)
        for sp in ax.spines.values():
            sp.set_color('#333')

        # Plot valid values; mark NaN regions as semi-transparent red band
        valid_mask = [not (v != v) for v in vals]  # NaN check (works with plain float)
        ax.plot(epochs, vals, color=c, linewidth=1.8, zorder=3)

        # Highlight NaN region
        nan_eps = [e for e, v, ok in zip(epochs, vals, valid_mask) if not ok]
        if nan_eps:
            ax.axvspan(min(nan_eps), max(nan_eps) + 1,
                       alpha=0.18, color='red', label='NaN region', zorder=1)

        # Mark best point (ignore NaN)
        valid_pairs = [(e, v) for e, v, ok in zip(epochs, vals, valid_mask) if ok]
        if valid_pairs:
            if better == '↑':
                best_e, best_v = max(valid_pairs, key=lambda x: x[1])
            else:
                best_e, best_v = min(valid_pairs, key=lambda x: x[1])
            ax.axvline(best_e, color='#ffcc00', linewidth=1.2,
                       linestyle='--', alpha=0.8, zorder=2)
            ax.scatter([best_e], [best_v], color='#ffcc00', zorder=5, s=50)
            ax.text(best_e + 0.3, best_v, f'{best_v:.3f}',
                    fontsize=8, color='#ffcc00', va='center', zorder=6)

        ax.set_title(f'{title}  ({better})', fontsize=11, color='white', pad=5)
        ax.set_xlabel('Epoch', fontsize=10, color='#aaa')
        ax.set_ylabel(ylabel, fontsize=10, color='#aaa')
        ax.grid(True, alpha=0.2, color='#444', linestyle='--')

    plt.tight_layout()
    fname = f'{variant}_metrics_ep{epoch:04d}.png'
    save_path = os.path.join(out_dir, fname)
    fig.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close(fig)
    print(f'    [Metric plot saved → {save_path}]')


def train_variant(variant, args, device, train_l, val_l):
    """Train one variant and return best metrics + checkpoint path."""
    print(f"\n{'='*60}")
    print(f"  Training: {VARIANT_LABELS[variant]}")
    print(f"{'='*60}")

    model = create_vmamba_model(
        variant=variant, in_ch=IN_CHANNELS, out_ch=IN_CHANNELS,
        scale=SCALE, d_model=args.d_model, d_state=args.d_state,
        num_blocks=[args.num_blocks]*4,
    ).to(device)

    total_params, _ = count_parameters(model)
    print(f"  Parameters: {total_params:,}")

    criterion = CompositeLossV5()
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    ckp_path = os.path.join(CKP_DIR, f'{variant}_best.pth')
    os.makedirs(CKP_DIR, exist_ok=True)

    img_dir  = os.path.join(IMG_DIR, variant)
    spec_dir = os.path.join(IMG_DIR, variant, 'spectral')

    best_psnr = -float('inf')
    history   = []

    epoch_iter = tqdm(range(1, args.epochs + 1),
                      desc=f"[{variant}] epochs", unit='ep') if HAS_TQDM else range(1, args.epochs + 1)
    for epoch in epoch_iter:
        model.train()
        train_loss = 0.0
        t0 = time.time()

        train_pbar = tqdm(train_l, desc=f"  Train {epoch:3d}/{args.epochs}",
                          leave=False, unit='batch') if HAS_TQDM else train_l
        for batch in train_pbar:
            lr_hsi = batch['lr_hsi'].to(device)
            hr_pan = batch['hr_pan'].to(device)
            hr_hsi = batch['hr_hsi'].to(device)

            optimizer.zero_grad()
            pred = model(lr_hsi, hr_pan)
            loss = criterion(pred, hr_hsi)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()
            if HAS_TQDM:
                train_pbar.set_postfix(loss=f'{train_loss/max(train_pbar.n,1):.4f}')

        scheduler.step()
        train_loss /= max(len(train_l), 1)

        # Validation
        model.eval()
        val_psnr = val_sam = val_ergas = val_ssim = 0.0
        n_val = 0
        sample_pred = sample_lr = sample_pan = sample_gt = None

        val_pbar = tqdm(val_l, desc="  Val  ", leave=False, unit='batch') if HAS_TQDM else val_l
        with torch.no_grad():
            for batch in val_pbar:
                lr_hsi = batch['lr_hsi'].to(device)
                hr_pan = batch['hr_pan'].to(device)
                hr_hsi = batch['hr_hsi'].to(device)
                pred   = model(lr_hsi, hr_pan)

                val_psnr  += compute_psnr(pred.cpu(), hr_hsi.cpu())
                val_sam   += compute_sam_metric(pred.cpu(), hr_hsi.cpu())
                val_ergas += compute_ergas(pred.cpu(), hr_hsi.cpu(), scale=SCALE)
                val_ssim  += compute_ssim(pred.cpu(), hr_hsi.cpu())
                n_val += 1
                if HAS_TQDM:
                    val_pbar.set_postfix(psnr=f'{val_psnr/n_val:.2f}', ssim=f'{val_ssim/n_val:.4f}')

                if sample_pred is None:
                    sample_pred = pred
                    sample_lr   = lr_hsi
                    sample_pan  = hr_pan
                    sample_gt   = hr_hsi

        n_val = max(n_val, 1)
        val_psnr  /= n_val
        val_sam   /= n_val
        val_ergas /= n_val
        val_ssim  /= n_val

        elapsed = time.time() - t0
        msg = (f"  Ep {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  "
               f"PSNR={val_psnr:.2f}  SAM={val_sam:.3f}  "
               f"ERGAS={val_ergas:.3f}  SSIM={val_ssim:.4f}  [{elapsed:.1f}s]")
        if HAS_TQDM:
            tqdm.write(msg)
            epoch_iter.set_postfix(psnr=f'{val_psnr:.2f}', ssim=f'{val_ssim:.4f}')
        else:
            print(msg)

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_psnr': val_psnr, 'val_sam': val_sam, 'val_ergas': val_ergas,
            'val_ssim': val_ssim,
        })

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch, 'psnr': val_psnr}, ckp_path)
            print(f"    [Saved best checkpoint  PSNR={best_psnr:.2f}]")

        if epoch % args.save_img_every == 0 and sample_pred is not None:
            save_images(epoch, variant, sample_lr, sample_pan, sample_pred, sample_gt, img_dir)
            save_spectral_curves(epoch, variant, sample_pred, sample_gt, spec_dir)

        # Save metric plots every plot_every epochs
        if epoch % args.plot_every == 0:
            v_plot_dir = os.path.join(METRIC_PLOT_DIR, variant)
            save_metric_plots(variant, history, epoch, v_plot_dir)

    # Final metric plot at end of training
    v_plot_dir = os.path.join(METRIC_PLOT_DIR, variant)
    save_metric_plots(variant, history, history[-1]['epoch'] if history else 0, v_plot_dir)

    # Save full training history to JSON
    if history:
        os.makedirs(RES_DIR, exist_ok=True)
        hist_path = os.path.join(RES_DIR, f'{variant}_training_history.json')
        with open(hist_path, 'w') as _f:
            json.dump(history, _f, indent=2)

    # Save training curve
    if HAS_MPL and history:
        import numpy as np
        eps   = [h['epoch']     for h in history]
        psnrs = [h['val_psnr']  for h in history]
        sams  = [h['val_sam']   for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(eps, psnrs, color=COLOURS[variant])
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('PSNR (dB)'); ax1.set_title(f'{VARIANT_LABELS[variant]} — PSNR')
        ax1.grid(True, alpha=0.3)
        ax2.plot(eps, sams, color=COLOURS[variant])
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('SAM (deg)'); ax2.set_title(f'{VARIANT_LABELS[variant]} — SAM')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOT_DIR, f'{variant}_training_curve.png'), dpi=120)
        plt.close()

    best_rec = max(history, key=lambda h: h['val_psnr']) if history else {}
    return {
        'variant': variant,
        'label':   VARIANT_LABELS[variant],
        'params':  total_params,
        'best_psnr':  best_rec.get('val_psnr', 0),
        'best_sam':   best_rec.get('val_sam', 0),
        'best_ergas': best_rec.get('val_ergas', 0),
        'best_ssim':  best_rec.get('val_ssim', 0),
        'checkpoint': ckp_path,
    }


def make_comparison_plot(results: list[dict]):
    if not HAS_MPL:
        return
    import numpy as np
    os.makedirs(PLOT_DIR, exist_ok=True)
    names  = [r['label']      for r in results]
    psnrs  = [r['best_psnr']  for r in results]
    sams   = [r['best_sam']   for r in results]
    ergas  = [r['best_ergas'] for r in results]
    params = [r['params']/1e6 for r in results]
    colors = [COLOURS[r['variant']] for r in results]
    x = np.arange(len(results))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, vals, ylabel, title in zip(
        axes,
        [psnrs, sams, ergas, params],
        ['PSNR (dB)', 'SAM (deg)', 'ERGAS', 'Params (M)'],
        ['PSNR (higher is better)', 'SAM (lower is better)',
         'ERGAS (lower is better)', 'Model Parameters'],
    ):
        bars = ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=7)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(vals),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('V5 Model Comparison — All 5 Variants', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'comparison_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(PLOT_DIR, 'comparison_summary.png')}")


def save_excel(results: list[dict]):
    if not HAS_XLSX:
        print("[INFO] openpyxl not installed — skipping Excel export")
        return
    import openpyxl
    os.makedirs(RES_DIR, exist_ok=True)
    wb  = openpyxl.Workbook()
    ws  = wb.active
    ws.title = 'V5 Comparison'
    headers = ['Variant', 'Label', 'Params', 'Best PSNR', 'Best SAM', 'Best ERGAS', 'Checkpoint']
    ws.append(headers)
    for r in results:
        ws.append([r['variant'], r['label'], r['params'],
                   r['best_psnr'], r['best_sam'], r['best_ergas'], r['checkpoint']])
    path = os.path.join(RES_DIR, 'comparison_results.xlsx')
    wb.save(path)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='V5 Compare all variants')
    parser.add_argument('--epochs',        default=5,  type=int)
    parser.add_argument('--batch_size',    default=1,  type=int)
    parser.add_argument('--d_model',       default=32, type=int)
    parser.add_argument('--d_state',       default=8,  type=int)
    parser.add_argument('--num_blocks',    default=1,  type=int)
    parser.add_argument('--patch_size',    default=32, type=int)
    parser.add_argument('--save_img_every', default=5, type=int)
    parser.add_argument('--plot_every',     default=50, type=int,
                        help='Save metric plots every N epochs (default: 50)')
    parser.add_argument('--skip_old',      action='store_true')
    parser.add_argument('--skip_improved', action='store_true')
    parser.add_argument('--skip_v3',       action='store_true')
    parser.add_argument('--skip_v4',       action='store_true')
    parser.add_argument('--skip_v5',       action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print("Loading Chikusei dataset …")
    train_l, val_l, _ = create_dataloaders_overlap(
        dataset='chikusei', mat_path=CHIKUSEI_PATH,
        batch_size=args.batch_size, patch_size=args.patch_size,
        overlap=args.patch_size // 2, scale=SCALE, num_workers=0,
    )
    print(f"  Train patches: {len(train_l.dataset)}  Val patches: {len(val_l.dataset)}")

    variants_to_run = []
    if not args.skip_old:      variants_to_run.append('old')
    if not args.skip_improved: variants_to_run.append('improved')
    if not args.skip_v3:       variants_to_run.append('v3')
    if not args.skip_v4:       variants_to_run.append('v4')
    if not args.skip_v5:       variants_to_run.append('v5')

    results = []
    for v in variants_to_run:
        r = train_variant(v, args, device, train_l, val_l)
        results.append(r)

    # Final summary
    print("\n" + "=" * 80)
    print(f"{'Variant':<30} {'PSNR (+)':>10} {'SAM (-)':>10} "
          f"{'ERGAS (-)':>10} {'Params (M)':>12}")
    print("=" * 80)
    for r in results:
        print(f"  {r['label']:<28} {r['best_psnr']:>10.3f} "
              f"{r['best_sam']:>10.3f} {r['best_ergas']:>10.3f} "
              f"{r['params']/1e6:>11.3f}")
    print("=" * 80)

    make_comparison_plot(results)
    save_excel(results)

    # Save JSON summary
    os.makedirs(RES_DIR, exist_ok=True)
    json_path = os.path.join(RES_DIR, 'comparison_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")


if __name__ == '__main__':
    main()
