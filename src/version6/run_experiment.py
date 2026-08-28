"""
run_experiment.py  —  V6 Edition
Single-variant training script (trains V6 only by default).

Usage
-----
  cd version6
  python run_experiment.py
  python run_experiment.py --epochs 200 --d_model 64 --patch_size 32
  python run_experiment.py --epochs 100 --patch_size 64
"""
from __future__ import annotations
import sys, os, argparse, time, json, importlib.util

_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_V5      = os.path.join(_PROJECT, 'version5')

# ── Path setup: V6 must win over every other version ─────────────────────────
# Insert in reverse priority order (last insert = highest priority)
for p in [
    os.path.join(_PROJECT, 'version1'),
    os.path.join(_PROJECT, 'version3'),
    _PROJECT,
    os.path.join(_PROJECT, 'version4'),
    _V5,
    _HERE,   # HIGHEST priority — version6/ wins
]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Explicit import of V6 modules by file path (avoids any shadowing) ─────────
def _load(name, filepath):
    spec   = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

_bm  = _load('baseline_models_v6',    os.path.join(_HERE, 'baseline_models.py'))
_lf  = _load('loss_functions_v6_mod', os.path.join(_HERE, 'loss_functions_v6.py'))

create_vmamba_model = _bm.create_vmamba_model
count_parameters    = _bm.count_parameters
CompositeLossV6     = _lf.CompositeLossV6
compute_psnr        = _lf.compute_psnr
compute_sam_metric  = _lf.compute_sam_metric
compute_ergas       = _lf.compute_ergas
compute_ssim        = _lf.compute_ssim

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset_loader_overlap import create_dataloaders_overlap

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kw): return x

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

CHIKUSEI_PATH = os.path.join(_PROJECT, 'chikusei', 'chikusei.mat')
IN_CHANNELS   = 128
SCALE         = 4

VARIANT_LABELS = {
    'v5': 'V5 VMamba (True Parallel Scan)',
    'v6': 'V6 VMamba (Spectral-Focused)',
}


def save_metric_plots(variant, history, epoch, out_dir):
    if not HAS_MPL or not history:
        return
    os.makedirs(out_dir, exist_ok=True)
    epochs = [h['epoch']      for h in history]
    metrics = [
        ([h['train_loss'] for h in history], 'Training Loss',    'Loss',      False, '#e74c3c'),
        ([h['val_psnr']   for h in history], 'Validation PSNR',  'PSNR (dB)', True,  '#2980b9'),
        ([h['val_sam']    for h in history], 'Validation SAM',   'SAM (deg)', False, '#27ae60'),
        ([h['val_ergas']  for h in history], 'Validation ERGAS', 'ERGAS',     False, '#f39c12'),
        ([h['val_ssim']   for h in history], 'Validation SSIM',  'SSIM',      True,  '#8e44ad'),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(26, 4.5))
    fig.patch.set_facecolor('white')
    label = VARIANT_LABELS.get(variant, variant)
    fig.suptitle(f'{label}  --  Epoch {epoch}', fontsize=13, color='black', y=1.01)
    for ax, (vals, title, ylabel, higher, c) in zip(axes, metrics):
        ax.set_facecolor('white')
        ax.tick_params(colors='black', labelsize=9)
        for sp in ax.spines.values(): sp.set_color('#cccccc')
        ax.plot(epochs, vals, color=c, linewidth=1.8)
        valid = [(e, v) for e, v in zip(epochs, vals) if v == v]
        if valid:
            best_e, best_v = max(valid, key=lambda x: x[1]) if higher else min(valid, key=lambda x: x[1])
            ax.axvline(best_e, color='#e67e00', linewidth=1.0, linestyle='--', alpha=0.8)
            ax.scatter([best_e], [best_v], color='#e67e00', zorder=5, s=40)
            ax.text(best_e + 0.5, best_v, f'{best_v:.3f}', fontsize=8, color='#e67e00')
        ax.set_title(f'{title}', fontsize=10, color='black', pad=4)
        ax.set_xlabel('Epoch', fontsize=9, color='#333333')
        ax.set_ylabel(ylabel, fontsize=9, color='#333333')
        ax.grid(True, alpha=0.4, color='#dddddd', linestyle='--')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{variant}_metrics_ep{epoch:04d}.png'),
                dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='V6 Training')
    parser.add_argument('--variant',        default='v6',  type=str, choices=['v5','v6'])
    parser.add_argument('--epochs',         default=200,   type=int)
    parser.add_argument('--batch_size',     default=1,     type=int)
    parser.add_argument('--d_model',        default=64,    type=int)
    parser.add_argument('--d_state',        default=8,     type=int)
    parser.add_argument('--num_blocks',     default=1,     type=int)
    parser.add_argument('--patch_size',     default=32,    type=int, choices=[32, 64])
    parser.add_argument('--overlap',        default=16,    type=int)
    parser.add_argument('--scale',          default=4,     type=int, choices=[2, 4, 8])
    parser.add_argument('--lr',             default=3e-4,  type=float)
    parser.add_argument('--grad_clip',      default=0.5,   type=float)
    parser.add_argument('--save_img_every', default=20,    type=int)
    parser.add_argument('--plot_every',     default=50,    type=int)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label  = VARIANT_LABELS.get(args.variant, args.variant)
    hr_size = args.patch_size * args.scale

    print(f"\n{'='*60}")
    print(f"  V6 Training")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    print(f"  Variant     : {label}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  d_model     : {args.d_model}")
    print(f"  LR patch    : {args.patch_size}x{args.patch_size}")
    print(f"  HR patch    : {hr_size}x{hr_size}  (scale={args.scale}x)")
    print(f"  Loss        : CompositeLossV6  (lambda_sam=0.40, lambda_spec=0.10)")
    print(f"  n_groups    : 32  (4 bands/token)")
    print(f"  alpha_init  : 0.05  beta_init: 0.3")
    print(f"{'='*60}\n")

    # Directories
    ckp_dir  = os.path.join(_HERE, 'comparison', 'checkpoints')
    res_dir  = os.path.join(_HERE, 'comparison', 'results')
    plot_dir = os.path.join(_HERE, 'comparison', 'metric_plots', args.variant)
    img_dir  = os.path.join(_HERE, 'comparison', 'saved_images', args.variant)
    for d in [ckp_dir, res_dir, plot_dir, img_dir]:
        os.makedirs(d, exist_ok=True)

    # Dataset
    print("Loading Chikusei dataset ...")
    train_l, val_l, _ = create_dataloaders_overlap(
        dataset     = 'chikusei',
        mat_path    = CHIKUSEI_PATH,
        batch_size  = args.batch_size,
        patch_size  = args.patch_size,
        overlap     = args.overlap,
        scale       = args.scale,
        num_workers = 0,
    )
    print(f"  Train: {len(train_l.dataset)} patches   Val: {len(val_l.dataset)} patches\n")

    # Model
    model = create_vmamba_model(
        variant    = args.variant,
        in_ch      = IN_CHANNELS,
        out_ch     = IN_CHANNELS,
        scale      = args.scale,
        d_model    = args.d_model,
        d_state    = args.d_state,
        num_blocks = [args.num_blocks] * 4,
    ).to(device)

    total, _ = count_parameters(model)
    print(f"  Parameters: {total:,}  ({total/1e6:.3f} M)\n")

    criterion = CompositeLossV6()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    ckp_path  = os.path.join(ckp_dir, f'{args.variant}_best.pth')
    best_psnr = -float('inf')
    history   = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        t0 = time.time()

        loader = tqdm(train_l, desc=f"Ep {epoch:3d}/{args.epochs} [train]",
                      leave=False) if HAS_TQDM else train_l
        for batch in loader:
            lr_hsi = batch['lr_hsi'].to(device)
            hr_pan = batch['hr_pan'].to(device)
            hr_hsi = batch['hr_hsi'].to(device)
            optimizer.zero_grad()
            pred = model(lr_hsi, hr_pan)
            loss = criterion(pred, hr_hsi)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(len(train_l), 1)

        # Validate
        model.eval()
        val_psnr = val_sam = val_ergas = val_ssim = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_l:
                lr_hsi = batch['lr_hsi'].to(device)
                hr_pan = batch['hr_pan'].to(device)
                hr_hsi = batch['hr_hsi'].to(device)
                pred   = model(lr_hsi, hr_pan)
                val_psnr  += compute_psnr(pred.cpu(), hr_hsi.cpu())
                val_sam   += compute_sam_metric(pred.cpu(), hr_hsi.cpu())
                val_ergas += compute_ergas(pred.cpu(), hr_hsi.cpu(), scale=args.scale)
                val_ssim  += compute_ssim(pred.cpu(), hr_hsi.cpu())
                n_val += 1

        n_val = max(n_val, 1)
        val_psnr  /= n_val;  val_sam   /= n_val
        val_ergas /= n_val;  val_ssim  /= n_val

        elapsed = time.time() - t0
        msg = (f"Ep {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  "
               f"PSNR={val_psnr:.2f}  SAM={val_sam:.3f}  "
               f"ERGAS={val_ergas:.3f}  SSIM={val_ssim:.4f}  [{elapsed:.1f}s]")
        if HAS_TQDM:
            tqdm.write(msg)
        else:
            print(msg)

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_psnr': val_psnr, 'val_sam': val_sam,
            'val_ergas': val_ergas, 'val_ssim': val_ssim,
        })

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch, 'psnr': val_psnr,
                        'sam': val_sam}, ckp_path)
            print(f"    [Saved best  PSNR={best_psnr:.2f}  SAM={val_sam:.3f}]")

        if epoch % args.plot_every == 0:
            save_metric_plots(args.variant, history, epoch, plot_dir)

    # Final plot + history save
    save_metric_plots(args.variant, history, args.epochs, plot_dir)
    hist_path = os.path.join(res_dir, f'{args.variant}_training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved: {hist_path}")

    best_rec = max(history, key=lambda h: h['val_psnr'])
    best_sam = min(history, key=lambda h: h['val_sam'])
    print(f"\n{'='*55}")
    print(f"  Best PSNR : {best_rec['val_psnr']:.4f} dB  @ epoch {best_rec['epoch']}")
    print(f"  Best SAM  : {best_sam['val_sam']:.4f} deg @ epoch {best_sam['epoch']}")
    print(f"  Final SAM : {history[-1]['val_sam']:.4f} deg")
    print(f"{'='*55}")
    print(f"\nCheckpoint : {ckp_path}")


if __name__ == '__main__':
    main()
