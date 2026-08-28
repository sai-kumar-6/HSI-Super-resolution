"""
train_baselines.py
==================
Trains each baseline (PanNet, PanGAN, PSGAN, Panformer) on Chikusei
using the same Wald's Protocol data as V6.

Usage
-----
  cd version6/baselines
  python train_baselines.py --method pannet    --epochs 100
  python train_baselines.py --method pangan    --epochs 100
  python train_baselines.py --method psgan     --epochs 100
  python train_baselines.py --method panformer --epochs 100
  python train_baselines.py --method all       --epochs 100  # train all
"""

import sys, os, argparse, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

_HERE    = os.path.dirname(os.path.abspath(__file__))
_V6      = os.path.dirname(_HERE)
_PROJECT = os.path.dirname(_V6)

for p in [_PROJECT, _V6,
          os.path.join(_PROJECT,'version1'),
          os.path.join(_PROJECT,'version3'),
          os.path.join(_PROJECT,'version4'),
          os.path.join(_PROJECT,'version5'),
          _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)

from baseline_architectures import create_baseline
from dataset_loader_overlap  import create_dataloaders_overlap

CKPT_DIR      = os.path.join(_HERE, 'checkpoints')
CHIKUSEI_PATH = os.path.join(_PROJECT, 'chikusei', 'chikusei.mat')
IN_CH, SCALE  = 128, 4
PATCH_SIZE    = 32
os.makedirs(CKPT_DIR, exist_ok=True)


def compute_psnr(pred, gt, max_val=1.0):
    mse = F.mse_loss(pred, gt)
    return 10 * torch.log10(max_val**2 / (mse + 1e-8))


def sam_loss(pred, gt):
    dot  = (pred * gt).sum(dim=1, keepdim=True)
    n_p  = pred.norm(dim=1, keepdim=True).clamp(min=1e-8)
    n_g  = gt.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos  = (dot / (n_p * n_g)).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos).mean()


def composite_loss(pred, gt):
    return F.l1_loss(pred, gt) + 0.1 * sam_loss(pred, gt)


def train_standard(model, train_l, val_l, epochs, device, ckpt_path, method):
    """Standard supervised training (for PanNet, Panformer)."""
    opt   = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best_psnr = -1
    history   = []

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0

        for batch in tqdm(train_l, desc=f'Ep {ep}/{epochs} [train]', leave=False):
            lr_hsi = batch['lr_hsi'].to(device)
            hr_pan = batch['hr_pan'].to(device)
            hr_hsi = batch['hr_hsi'].to(device)

            pred = model(lr_hsi, hr_pan)
            loss = composite_loss(pred, hr_hsi)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            total_loss += loss.item()

        sched.step()
        avg_loss = total_loss / len(train_l)

        # Validation
        model.eval()
        psnr_sum, sam_sum, n = 0, 0, 0
        with torch.no_grad():
            for batch in val_l:
                lr_hsi = batch['lr_hsi'].to(device)
                hr_pan = batch['hr_pan'].to(device)
                hr_hsi = batch['hr_hsi'].to(device)
                pred   = model(lr_hsi, hr_pan)
                psnr_sum += compute_psnr(pred, hr_hsi).item()
                sam_sum  += sam_loss(pred, hr_hsi).item() * 180 / 3.14159
                n += 1

        val_psnr = psnr_sum / n
        val_sam  = sam_sum / n
        dt = time.time() - t0

        print(f'  Ep {ep:3d}/{epochs}  loss={avg_loss:.4f}  '
              f'PSNR={val_psnr:.2f}  SAM={val_sam:.3f}  [{dt:.1f}s]')

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), ckpt_path)
            print(f'    [Saved best PSNR={best_psnr:.2f}]')

        history.append({'epoch': ep, 'train_loss': avg_loss,
                        'val_psnr': val_psnr, 'val_sam': val_sam})

    return history


def train_gan(model, train_l, val_l, epochs, device, ckpt_path, method):
    """Adversarial training for PanGAN and PSGAN."""
    G = model.generator
    D = model.discriminator

    opt_G = Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    sched_G = CosineAnnealingLR(opt_G, T_max=epochs, eta_min=1e-5)
    sched_D = CosineAnnealingLR(opt_D, T_max=epochs, eta_min=1e-5)

    bce     = nn.BCEWithLogitsLoss()
    best_psnr = -1
    history   = []

    for ep in range(1, epochs + 1):
        G.train(); D.train()
        t0 = time.time()
        loss_G_sum, loss_D_sum = 0, 0

        for batch in tqdm(train_l, desc=f'Ep {ep}/{epochs} [GAN]', leave=False):
            lr_hsi = batch['lr_hsi'].to(device)
            hr_pan = batch['hr_pan'].to(device)
            hr_hsi = batch['hr_hsi'].to(device)
            B = lr_hsi.size(0)

            # ── Train Discriminator ──────────────────────────────────────────
            with torch.no_grad():
                fake = G(lr_hsi, hr_pan)

            if method == 'psgan':
                real_logit = D(hr_hsi, hr_pan)
                fake_logit = D(fake.detach(), hr_pan)
            else:
                real_logit = D(hr_hsi)
                fake_logit = D(fake.detach())

            real_lbl = torch.ones_like(real_logit)
            fake_lbl = torch.zeros_like(fake_logit)
            loss_D   = (bce(real_logit, real_lbl) + bce(fake_logit, fake_lbl)) * 0.5

            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # ── Train Generator ──────────────────────────────────────────────
            fake = G(lr_hsi, hr_pan)
            if method == 'psgan':
                adv = bce(D(fake, hr_pan), torch.ones_like(D(fake, hr_pan)))
            else:
                adv = bce(D(fake), torch.ones_like(D(fake)))

            loss_G = adv + composite_loss(fake, hr_hsi)
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            loss_G_sum += loss_G.item()
            loss_D_sum += loss_D.item()

        sched_G.step(); sched_D.step()

        # Validation
        G.eval()
        psnr_sum, n = 0, 0
        with torch.no_grad():
            for batch in val_l:
                lr_hsi = batch['lr_hsi'].to(device)
                hr_pan = batch['hr_pan'].to(device)
                hr_hsi = batch['hr_hsi'].to(device)
                fake   = G(lr_hsi, hr_pan)
                psnr_sum += compute_psnr(fake, hr_hsi).item()
                n += 1

        val_psnr = psnr_sum / n
        dt = time.time() - t0

        print(f'  Ep {ep:3d}/{epochs}  '
              f'G={loss_G_sum/len(train_l):.4f}  '
              f'D={loss_D_sum/len(train_l):.4f}  '
              f'PSNR={val_psnr:.2f}  [{dt:.1f}s]')

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), ckpt_path)
            print(f'    [Saved best PSNR={best_psnr:.2f}]')

        history.append({'epoch': ep, 'val_psnr': val_psnr,
                        'loss_G': loss_G_sum/len(train_l),
                        'loss_D': loss_D_sum/len(train_l)})

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',     default='all',
                        choices=['all','pannet','pangan','psgan','panformer'])
    parser.add_argument('--epochs',     default=100, type=int)
    parser.add_argument('--batch_size', default=4,   type=int)
    parser.add_argument('--patch_size', default=32,  type=int)
    parser.add_argument('--scale',      default=4,   type=int)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Dataset ───────────────────────────────────────────────────────────────
    print('\nLoading Chikusei ...')
    train_l, val_l, _ = create_dataloaders_overlap(
        dataset='chikusei', mat_path=CHIKUSEI_PATH,
        batch_size=args.batch_size, patch_size=args.patch_size,
        overlap=args.patch_size // 2, scale=args.scale, num_workers=0,
    )
    print(f'  Train: {len(train_l.dataset)} patches  Val: {len(val_l.dataset)} patches')

    methods = ['pannet','pangan','psgan','panformer'] if args.method == 'all' else [args.method]

    for method in methods:
        print(f'\n{"="*60}')
        print(f'  Training: {method.upper()}  ({args.epochs} epochs)')
        print(f'{"="*60}')

        model     = create_baseline(method, IN_CH, args.scale).to(device)
        ckpt_path = os.path.join(CKPT_DIR, f'{method}_best.pth')
        n_params  = sum(p.numel() for p in model.parameters())
        print(f'  Parameters: {n_params:,}  ({n_params/1e6:.3f} M)')

        is_gan = method in ('pangan', 'psgan')
        fn     = train_gan if is_gan else train_standard

        history = fn(model, train_l, val_l, args.epochs, device, ckpt_path, method)

        hist_path = os.path.join(CKPT_DIR, f'{method}_history.json')
        with open(hist_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f'\n  History: {hist_path}')
        print(f'  Best ckpt: {ckpt_path}')

    print('\nAll done.')


if __name__ == '__main__':
    main()
