"""
train.py  —  Version 2
Train V2VMambaPansharp for a given scale factor (2 or 4).

Usage:
  cd version2
  python train.py --scale 4 --epochs 30 --batch_size 2 --d_model 32
  python train.py --scale 2 --epochs 30 --batch_size 2 --d_model 32

All logs, checkpoints, and images are saved under runs/scale<N>/.
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import json
import math

_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'model'))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kw): return x

from dataset          import create_dataloaders
from model            import V2VMambaPansharp, count_parameters
from losses           import CompositeLoss, compute_psnr, compute_sam, compute_ergas, compute_ssim
from logger           import ExperimentLogger

CHIKUSEI_PATH = os.path.join(_PROJECT, 'chikusei', 'chikusei.mat')


# ── Argument parser ───────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='V2 Training')
    p.add_argument('--scale',       type=int,   default=4)
    p.add_argument('--epochs',      type=int,   default=30)
    p.add_argument('--batch_size',  type=int,   default=2)
    p.add_argument('--patch_size',  type=int,   default=32)
    p.add_argument('--overlap',     type=int,   default=16)
    p.add_argument('--d_model',     type=int,   default=32)
    p.add_argument('--d_state',     type=int,   default=8)
    p.add_argument('--num_blocks',  type=int,   default=1,
                   help='Blocks per stage (same for all 4 stages)')
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--weight_decay',type=float, default=1e-4)
    p.add_argument('--max_rows',    type=int,   default=512)
    p.add_argument('--save_img_every', type=int, default=5)
    p.add_argument('--num_workers', type=int,   default=0)
    p.add_argument('--run_dir',     type=str,   default=None,
                   help='Override default runs/scale<N>/')
    return p.parse_args()


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch=0, epochs=0):
    model.train()
    total_loss = 0.0
    comp_sums  = dict(l1=0.0, sam=0.0, edge=0.0, ssim=0.0, spec=0.0)
    n = 0

    desc = f"  Train {epoch:3d}/{epochs}" if epochs else "  Train"
    pbar = tqdm(loader, desc=desc, leave=False, unit='batch') if HAS_TQDM else loader
    for batch in pbar:
        lr_hsi = batch['lr_hsi'].to(device)
        hr_pan = batch['hr_pan'].to(device)
        hr_hsi = batch['hr_hsi'].to(device)

        optimizer.zero_grad()
        pred = model(lr_hsi, hr_pan)
        loss, comps = criterion(pred, hr_hsi, return_components=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for k in comp_sums:
            comp_sums[k] += comps.get(k, 0.0)
        n += 1

        if HAS_TQDM:
            pbar.set_postfix(loss=f'{total_loss/n:.4f}')

    n = max(n, 1)
    return {
        'total': total_loss / n,
        **{k: v/n for k, v in comp_sums.items()}
    }


@torch.no_grad()
def validate(model, loader, device, scale):
    model.eval()
    psnr_sum = sam_sum = ergas_sum = ssim_sum = 0.0
    n = 0
    pbar = tqdm(loader, desc="  Val  ", leave=False, unit='batch') if HAS_TQDM else loader
    for batch in pbar:
        lr_hsi = batch['lr_hsi'].to(device)
        hr_pan = batch['hr_pan'].to(device)
        hr_hsi = batch['hr_hsi']
        pred   = model(lr_hsi, hr_pan).cpu()

        psnr_sum  += compute_psnr(pred, hr_hsi)
        sam_sum   += compute_sam(pred, hr_hsi)
        ergas_sum += compute_ergas(pred, hr_hsi, scale=scale)
        ssim_sum  += compute_ssim(pred, hr_hsi)
        n += 1
        if HAS_TQDM:
            pbar.set_postfix(psnr=f'{psnr_sum/n:.2f}', ssim=f'{ssim_sum/n:.4f}')

    n = max(n, 1)
    return dict(psnr=psnr_sum/n, sam=sam_sum/n, ergas=ergas_sum/n, ssim=ssim_sum/n)


@torch.no_grad()
def test(model, loader, device, scale, logger, tag='test'):
    """Full test pass with timing and visual logging."""
    model.eval()
    psnr_sum = sam_sum = ergas_sum = ssim_sum = 0.0
    n = 0
    t0 = time.perf_counter()
    sample = None

    pbar = tqdm(loader, desc="  Test ", leave=False, unit='batch') if HAS_TQDM else loader
    for batch in pbar:
        lr_hsi = batch['lr_hsi'].to(device)
        hr_pan = batch['hr_pan'].to(device)
        hr_hsi = batch['hr_hsi']
        pred   = model(lr_hsi, hr_pan).cpu()

        psnr_sum  += compute_psnr(pred, hr_hsi)
        sam_sum   += compute_sam(pred, hr_hsi)
        ergas_sum += compute_ergas(pred, hr_hsi, scale=scale)
        ssim_sum  += compute_ssim(pred, hr_hsi)
        n += 1
        if HAS_TQDM:
            pbar.set_postfix(psnr=f'{psnr_sum/n:.2f}', ssim=f'{ssim_sum/n:.4f}')
        if sample is None:
            sample = (pred, hr_hsi, lr_hsi.cpu(), hr_pan.cpu())

    elapsed_ms = (time.perf_counter() - t0) / max(n,1) * 1000
    n = max(n, 1)
    metrics = dict(psnr=psnr_sum/n, sam=sam_sum/n, ergas=ergas_sum/n, ssim=ssim_sum/n)

    print(f"\n  Test ({tag})  PSNR={metrics['psnr']:.3f}dB  "
          f"SAM={metrics['sam']:.3f}  ERGAS={metrics['ergas']:.3f}  "
          f"SSIM={metrics['ssim']:.4f}  ({elapsed_ms:.1f}ms/patch)")

    if sample is not None:
        pred, gt, lr, pan = sample
        logger.save_test_visuals(pred, gt, lr, pan, metrics, tag=tag)

    logger.log_test_results(metrics, scale=scale, inference_ms=round(elapsed_ms,2))
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    run_dir = args.run_dir or os.path.join(_HERE, 'runs', f'scale{args.scale}')
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"  V2 VMamba Pansharpening — Scale {args.scale}")
    print(f"{'='*70}")
    print(f"  Device     : {device}")
    print(f"  Run dir    : {run_dir}")

    hparams = vars(args)
    hparams['device'] = str(device)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading data …")
    train_dl, val_dl, test_dl = create_dataloaders(
        mat_path   = CHIKUSEI_PATH,
        batch_size = args.batch_size,
        patch_size = args.patch_size,
        overlap    = args.overlap,
        scale      = args.scale,
        num_workers= args.num_workers,
        max_rows   = args.max_rows,
    )
    hparams['train_patches'] = len(train_dl.dataset)
    hparams['val_patches']   = len(val_dl.dataset)
    hparams['test_patches']  = len(test_dl.dataset)

    # ── Model ─────────────────────────────────────────────────────────────────
    nb = [args.num_blocks] * 4
    model = V2VMambaPansharp(
        in_channels=128, out_channels=128,
        d_model=args.d_model, d_state=args.d_state,
        scale=args.scale, num_blocks=nb,
    ).to(device)

    total, trainable = count_parameters(model)
    hparams['params_total']    = total
    hparams['params_trainable']= trainable
    print(f"  Parameters : {total:,} total  {trainable:,} trainable")

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = ExperimentLogger(run_dir)

    # Sample for FLOPs + dataset visualisation
    sample_batch = next(iter(train_dl))
    sample_lr    = sample_batch['lr_hsi'][:1].to(device)
    sample_pan   = sample_batch['hr_pan'][:1].to(device)
    sample_hr    = sample_batch['hr_hsi'][:1]

    logger.log_config(hparams, model, sample_lr, sample_pan)
    logger.save_dataset_sample(sample_hr, name='dataset_sample')

    # Inference timing (before training)
    inf_info = logger.time_inference(model, sample_lr, sample_pan, n_runs=5)
    print(f"  Inference  : {inf_info['inference_ms_per_sample']:.2f} ms/sample")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    criterion = CompositeLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01)

    ckp_dir  = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckp_dir, exist_ok=True)
    best_psnr = -float('inf')

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs ...\n")
    epoch_iter = tqdm(range(1, args.epochs + 1), desc=f"Epochs scale={args.scale}",
                      unit='ep') if HAS_TQDM else range(1, args.epochs + 1)
    for epoch in epoch_iter:
        t0 = time.time()

        train_losses = train_one_epoch(model, train_dl, criterion, optimizer, device,
                                       epoch=epoch, epochs=args.epochs)
        val_metrics  = validate(model, val_dl, device, args.scale)
        scheduler.step()

        current_lr  = scheduler.get_last_lr()[0]
        epoch_time  = time.time() - t0

        logger.log_epoch(epoch, train_losses, val_metrics, current_lr, epoch_time)

        psnr = val_metrics['psnr']
        msg = (f"  Ep {epoch:3d}/{args.epochs}  "
               f"loss={train_losses['total']:.4f}  "
               f"PSNR={psnr:.2f}  SAM={val_metrics['sam']:.3f}  "
               f"ERGAS={val_metrics['ergas']:.3f}  SSIM={val_metrics['ssim']:.4f}  [{epoch_time:.1f}s]")
        if HAS_TQDM:
            tqdm.write(msg)
            epoch_iter.set_postfix(psnr=f'{psnr:.2f}', loss=f'{train_losses["total"]:.4f}')
        else:
            print(msg)

        # Save best
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch, 'psnr': psnr, 'scale': args.scale,
                        'hparams': hparams},
                       os.path.join(ckp_dir, 'best.pth'))
            print(f"    [Saved best  PSNR={best_psnr:.2f}]")

        # Periodic image snapshot
        if epoch % args.save_img_every == 0:
            with torch.no_grad():
                snap_pred = model(sample_lr, sample_pan).cpu()
            logger.save_train_snapshot(epoch, snap_pred, sample_hr, sample_lr.cpu())

    # ── Final evaluation ──────────────────────────────────────────────────────
    # Load best
    best_ckp = os.path.join(ckp_dir, 'best.pth')
    if os.path.isfile(best_ckp):
        st = torch.load(best_ckp, map_location=device)
        model.load_state_dict(st['model_state_dict'])
        print(f"\n  Loaded best checkpoint (ep {st['epoch']}, PSNR={st['psnr']:.2f})")

    test_metrics = test(model, test_dl, device, args.scale, logger, tag='best')

    # ── Save plots ─────────────────────────────────────────────────────────────
    logger.save_learning_curves(title_suffix=f'scale={args.scale}')
    logger.save_loss_components()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Scale {args.scale} Training Complete")
    print(f"  Best Val PSNR : {best_psnr:.3f} dB")
    print(f"  Test PSNR     : {test_metrics['psnr']:.3f} dB")
    print(f"  Test SAM      : {test_metrics['sam']:.3f}")
    print(f"  Test ERGAS    : {test_metrics['ergas']:.3f}")
    print(f"  Test SSIM     : {test_metrics['ssim']:.4f}")
    print(f"  Run dir       : {run_dir}")
    print(f"{'='*70}")

    return test_metrics


if __name__ == '__main__':
    main()
