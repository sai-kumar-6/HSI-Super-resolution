"""
logger.py  —  Version 2
Comprehensive experiment logger.

Logs per epoch:
  - wall-clock time (epoch, cumulative, inference)
  - train loss (total + L1, SAM, Edge, SSIM, SpectralGrad components)
  - val metrics (PSNR, SAM, ERGAS)
  - learning rate

Logs once per run:
  - all hyperparameters
  - model architecture string
  - parameter counts (total, trainable)
  - FLOPs estimate (thop if available, else manual)
  - block counts per stage
  - dataset statistics (patch counts per split)
  - sample dataset visualisation  (3-band false colour)
  - 3 tested image visuals per evaluation (false colour, error map, spectral)

Outputs (in run_dir/):
  - log.json       complete structured log (append per epoch)
  - metrics.csv    one row per epoch (easy to import in Excel/pandas)
  - run_config.json  hyperparams + architecture snapshot
  - images/        training snapshots & final test images
  - plots/         learning curves
"""
from __future__ import annotations
import os
import csv
import json
import time
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def hsi_to_rgb(t, bands=(30, 20, 10)):
    """(C,H,W) tensor/numpy → (H,W,3) float32 in [0,1]."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    C = t.shape[0]
    b0, b1, b2 = [min(b, C-1) for b in bands]
    rgb = np.stack([t[b0], t[b1], t[b2]], axis=2).astype(np.float32)
    lo, hi = rgb.min(), rgb.max()
    if hi > lo:
        rgb = (rgb - lo) / (hi - lo)
    return np.clip(rgb, 0, 1)


# ── Main logger class ─────────────────────────────────────────────────────────

class ExperimentLogger:
    """
    Attach to a training run; call the hook methods at the right points.

    Usage:
        logger = ExperimentLogger(run_dir='runs/scale4')
        logger.log_config(hparams, model, sample_lr, sample_pan)
        for epoch in range(epochs):
            ...training...
            logger.log_epoch(epoch, train_losses, val_metrics, lr, epoch_time)
        logger.log_test_results(test_metrics, pred, gt, lr_hsi, hr_pan, scale)
        logger.save_comparison_plot()
    """

    def __init__(self, run_dir: str):
        self.run_dir  = run_dir
        self.img_dir  = os.path.join(run_dir, 'images')
        self.plot_dir = os.path.join(run_dir, 'plots')
        os.makedirs(self.img_dir,  exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.log_path     = os.path.join(run_dir, 'log.json')
        self.csv_path     = os.path.join(run_dir, 'metrics.csv')
        self.config_path  = os.path.join(run_dir, 'run_config.json')

        self.history: list[dict] = []
        self.t_start  = time.time()

        # CSV header
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'epoch_time_s', 'cumulative_time_s',
                'train_loss', 'train_l1', 'train_sam', 'train_edge',
                'train_ssim', 'train_spec',
                'val_psnr', 'val_sam', 'val_ergas', 'val_ssim', 'lr',
            ])

    # ── Configuration ─────────────────────────────────────────────────────────

    def log_config(
        self,
        hparams:    dict,
        model:      nn.Module,
        sample_lr:  torch.Tensor,   # (1,C,h,w) for FLOPs
        sample_pan: torch.Tensor,   # (1,1,H,W)
    ):
        """Log hyperparameters, architecture, parameter counts, FLOPs."""
        total    = sum(p.numel() for p in model.parameters())
        trainable= sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Architecture string (truncated to 4000 chars)
        arch = str(model)[:4000]

        # Block counts
        block_counts = {}
        for name, mod in model.named_modules():
            cname = type(mod).__name__
            block_counts[cname] = block_counts.get(cname, 0) + 1

        # FLOPs
        flops_info = self._estimate_flops(model, sample_lr, sample_pan)

        config = {
            'timestamp':    time.strftime('%Y-%m-%d %H:%M:%S'),
            'hparams':      hparams,
            'params_total': total,
            'params_train': trainable,
            'flops':        flops_info,
            'block_counts': block_counts,
            'architecture': arch,
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"  [Logger] Config saved -> {self.config_path}")
        print(f"  [Logger] Params total={total:,}  trainable={trainable:,}")
        print(f"  [Logger] FLOPs: {flops_info}")

    def _estimate_flops(self, model, lr, pan):
        if HAS_THOP:
            try:
                with torch.no_grad():
                    macs, params = thop_profile(model, inputs=(lr, pan), verbose=False)
                return {'macs': int(macs), 'params_thop': int(params),
                        'source': 'thop'}
            except Exception as e:
                pass
        # Manual estimate based on parameter count (rough)
        total = sum(p.numel() for p in model.parameters())
        L = lr.shape[2] * lr.shape[3] * model.scale**2
        est = int(total * L * 2)
        return {'macs_estimate': est, 'source': 'manual (2*params*L)'}

    # ── Epoch logging ──────────────────────────────────────────────────────────

    def log_epoch(
        self,
        epoch:        int,
        train_losses: dict,   # keys: total, l1, sam, edge, ssim, spec
        val_metrics:  dict,   # keys: psnr, sam, ergas
        lr:           float,
        epoch_time:   float,
    ):
        cumulative = time.time() - self.t_start
        row = {
            'epoch':           epoch,
            'epoch_time_s':    round(epoch_time, 2),
            'cumulative_time_s': round(cumulative, 2),
            **{f'train_{k}': v for k, v in train_losses.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
            'lr': lr,
        }
        self.history.append(row)

        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, round(epoch_time,2), round(cumulative,2),
                _fmt(train_losses.get('total',0)),
                _fmt(train_losses.get('l1',0)),
                _fmt(train_losses.get('sam',0)),
                _fmt(train_losses.get('edge',0)),
                _fmt(train_losses.get('ssim',0)),
                _fmt(train_losses.get('spec',0)),
                _fmt(val_metrics.get('psnr',0)),
                _fmt(val_metrics.get('sam',0)),
                _fmt(val_metrics.get('ergas',0)),
                _fmt(val_metrics.get('ssim',0)),
                _fmt(lr),
            ])

        # Overwrite full JSON log
        with open(self.log_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    # ── Inference timing ───────────────────────────────────────────────────────

    def time_inference(self, model: nn.Module, lr: torch.Tensor,
                       pan: torch.Tensor, n_runs: int = 10) -> dict:
        """Measure per-sample inference time and store in config."""
        device = next(model.parameters()).device
        lr  = lr.to(device)
        pan = pan.to(device)
        model.eval()
        # Warm up
        with torch.no_grad():
            _ = model(lr, pan)
        # Time
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                model(lr, pan)
        elapsed = (time.perf_counter() - t0) / n_runs * 1000  # ms
        info = {'inference_ms_per_sample': round(elapsed, 3), 'n_runs': n_runs}
        # Append to config
        if os.path.exists(self.config_path):
            with open(self.config_path) as f:
                cfg = json.load(f)
            cfg['inference_timing'] = info
            with open(self.config_path, 'w') as f:
                json.dump(cfg, f, indent=2, default=str)
        return info

    # ── Image snapshots ────────────────────────────────────────────────────────

    def save_dataset_sample(self, hr: torch.Tensor, name: str = 'dataset_sample'):
        """Save a 3-band visualisation of one GT patch."""
        if not HAS_MPL:
            return
        hr_np = hr[0].detach().cpu().numpy() if hr.ndim == 4 else hr.detach().cpu().numpy()
        C = hr_np.shape[0]
        # Pick 3 bands spread across the spectral range
        b1, b2, b3 = C//6, C//2, 5*C//6
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor('#1a1a2e')
        titles = [f'Band {b1}', f'Band {b2}', f'Band {b3}']
        for ax, b, title in zip(axes, [b1, b2, b3], titles):
            band = hr_np[b]
            ax.imshow(band, cmap='viridis')
            ax.set_title(title, color='white', fontsize=10)
            ax.axis('off')
        plt.suptitle('Dataset Sample — 3 Spectral Bands', color='white', fontsize=12)
        plt.tight_layout()
        path = os.path.join(self.img_dir, f'{name}.png')
        plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        print(f"  [Logger] Dataset sample saved -> {path}")

    def save_train_snapshot(self, epoch: int, pred: torch.Tensor,
                             gt: torch.Tensor, lr_hsi: torch.Tensor):
        """Save false-colour comparison at a training epoch."""
        if not HAS_MPL:
            return
        import torch.nn.functional as F
        lr_up = F.interpolate(lr_hsi.cpu(), scale_factor=float(pred.shape[2]//lr_hsi.shape[2]),
                              mode='bicubic', align_corners=False)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#111')
        for ax, img, title in zip(axes,
            [lr_up[0], pred[0].detach(), gt[0]],
            ['LR-HSI (bicubic)', f'Predicted (ep {epoch})', 'Ground Truth']):
            ax.imshow(hsi_to_rgb(img)); ax.set_title(title, color='white', fontsize=9)
            ax.axis('off')
        plt.tight_layout()
        path = os.path.join(self.img_dir, f'train_ep{epoch:04d}.png')
        plt.savefig(path, dpi=100, bbox_inches='tight', facecolor='#111')
        plt.close()

    def save_test_visuals(self, pred: torch.Tensor, gt: torch.Tensor,
                           lr_hsi: torch.Tensor, hr_pan: torch.Tensor,
                           metrics: dict, tag: str = 'test'):
        """Save 3 standard test visuals: false colour, error map, spectral curve."""
        if not HAS_MPL:
            return
        import torch.nn.functional as F
        p  = pred[0].detach().cpu()
        g  = gt[0].cpu()
        lr = lr_hsi[0].cpu()
        pan= hr_pan[0, 0].cpu().numpy()

        lr_up = F.interpolate(lr.unsqueeze(0), size=p.shape[1:],
                              mode='bicubic', align_corners=False)[0]

        # 1. False colour comparison
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.patch.set_facecolor('#111')
        for ax, img, title in zip(axes,
            [lr_up, p, g, None],
            ['LR-HSI (bicubic)', 'Predicted', 'Ground Truth', 'HR-PAN']):
            if img is None:
                ax.imshow(pan, cmap='gray')
                ax.set_title('HR-PAN', color='white', fontsize=9)
            else:
                ax.imshow(hsi_to_rgb(img))
                ax.set_title(title, color='white', fontsize=9)
            ax.axis('off')
        plt.suptitle(f'PSNR={metrics.get("psnr",0):.2f}dB  '
                     f'SAM={metrics.get("sam",0):.3f}°  '
                     f'ERGAS={metrics.get("ergas",0):.3f}',
                     color='white', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.img_dir, f'{tag}_false_colour.png'),
                    dpi=130, bbox_inches='tight', facecolor='#111')
        plt.close()

        # 2. Error map
        err = (p - g).abs().mean(0).numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(err, cmap='hot')
        ax.set_title('Mean Abs Error per Pixel', color='white', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.patch.set_facecolor('#111')
        plt.tight_layout()
        plt.savefig(os.path.join(self.img_dir, f'{tag}_error_map.png'),
                    dpi=130, bbox_inches='tight', facecolor='#111')
        plt.close()

        # 3. Spectral curve at centre pixel
        py, px = p.shape[1]//2, p.shape[2]//2
        bands  = np.arange(p.shape[0])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#111')
        for ax in axes:
            ax.set_facecolor('#0a0a1a')
            ax.tick_params(colors='white')
            for sp in ax.spines.values():
                sp.set_color('grey')

        axes[0].plot(bands, g[:, py, px].numpy(), 'w-',  lw=2.5, label='GT')
        axes[0].plot(bands, p[:, py, px].numpy(), 'r--', lw=1.5, label='Predicted')
        axes[0].set_xlabel('Band', color='white'); axes[0].set_ylabel('Reflectance', color='white')
        axes[0].set_title(f'Spectral Curve @ ({py},{px})', color='white')
        axes[0].legend(fontsize=8, labelcolor='white', facecolor='#1a1a2e', edgecolor='grey')
        axes[0].grid(True, alpha=0.2, color='grey')

        axes[1].plot(bands[1:], np.diff(g[:, py, px].numpy()), 'w-',  lw=2.5, label='GT grad')
        axes[1].plot(bands[1:], np.diff(p[:, py, px].numpy()), 'r--', lw=1.5, label='Pred grad')
        axes[1].axhline(0, color='grey', lw=0.8)
        axes[1].set_xlabel('Band', color='white'); axes[1].set_ylabel('dC', color='white')
        axes[1].set_title('Spectral Gradient (shape preservation)', color='white')
        axes[1].legend(fontsize=8, labelcolor='white', facecolor='#1a1a2e', edgecolor='grey')
        axes[1].grid(True, alpha=0.2, color='grey')

        plt.suptitle('Spectral Analysis', color='white', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.img_dir, f'{tag}_spectral.png'),
                    dpi=130, bbox_inches='tight', facecolor='#111')
        plt.close()
        print(f"  [Logger] Test visuals saved -> {self.img_dir}/{tag}_*.png")

    # ── Learning curves ────────────────────────────────────────────────────────

    def save_learning_curves(self, title_suffix: str = ''):
        """Plot train loss and val metrics vs epoch."""
        if not HAS_MPL or not self.history:
            return
        eps   = [h['epoch']           for h in self.history]
        loss  = [h.get('train_total', h.get('train_loss', 0)) for h in self.history]
        psnrs = [h.get('val_psnr',  0) for h in self.history]
        sams  = [h.get('val_sam',   0) for h in self.history]
        ergas = [h.get('val_ergas', 0) for h in self.history]

        fig, axes = plt.subplots(1, 4, figsize=(22, 4))
        fig.patch.set_facecolor('#111')
        for ax in axes:
            ax.set_facecolor('#0a0a1a')
            ax.tick_params(colors='white')
            for sp in ax.spines.values(): sp.set_color('grey')

        pairs = [
            (axes[0], loss,  'Train Loss',     'Composite Loss'),
            (axes[1], psnrs, 'Val PSNR (dB)',  'PSNR (higher=better)'),
            (axes[2], sams,  'Val SAM (deg)',   'SAM (lower=better)'),
            (axes[3], ergas, 'Val ERGAS',       'ERGAS (lower=better)'),
        ]
        colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        for (ax, vals, ylabel, title), c in zip(pairs, colors):
            ax.plot(eps, vals, color=c, lw=1.5)
            ax.set_xlabel('Epoch', color='white')
            ax.set_ylabel(ylabel, color='white')
            ax.set_title(title + (' ' + title_suffix if title_suffix else ''),
                         color='white', fontsize=9)
            ax.grid(True, alpha=0.2, color='grey')

        plt.tight_layout()
        path = os.path.join(self.plot_dir, 'learning_curves.png')
        plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='#111')
        plt.close()
        print(f"  [Logger] Learning curves saved -> {path}")

    def save_loss_components(self):
        """Plot individual loss components over epochs."""
        if not HAS_MPL or not self.history:
            return
        eps = [h['epoch'] for h in self.history]
        components = ['l1', 'sam', 'edge', 'ssim', 'spec']
        colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#111')
        ax.set_facecolor('#0a0a1a')
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_color('grey')
        for comp, c in zip(components, colors):
            vals = [h.get(f'train_{comp}', 0) for h in self.history]
            ax.plot(eps, vals, color=c, lw=1.5, label=comp.upper())
        ax.set_xlabel('Epoch', color='white'); ax.set_ylabel('Loss', color='white')
        ax.set_title('Loss Components per Epoch', color='white')
        ax.legend(fontsize=8, labelcolor='white', facecolor='#1a1a2e', edgecolor='grey')
        ax.grid(True, alpha=0.2, color='grey')
        plt.tight_layout()
        path = os.path.join(self.plot_dir, 'loss_components.png')
        plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='#111')
        plt.close()

    # ── Final test results ─────────────────────────────────────────────────────

    def log_test_results(self, metrics: dict, scale: int, inference_ms: float):
        """Append test results to the run config JSON."""
        if os.path.exists(self.config_path):
            with open(self.config_path) as f:
                cfg = json.load(f)
        else:
            cfg = {}
        cfg.setdefault('test_results', {})[f'scale{scale}'] = {
            **metrics, 'inference_ms': inference_ms
        }
        with open(self.config_path, 'w') as f:
            json.dump(cfg, f, indent=2, default=str)
        print(f"  [Logger] Test results (scale={scale}) -> {self.config_path}")
