"""
plot_v5_metrics.py
Plots each training metric from v5_training_history.json in a separate PNG.
Output folder: training_plots/v5/
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Load data ─────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON    = os.path.join(HERE, 'version5', 'comparison', 'results', 'v5_training_history.json')
OUT_DIR = os.path.join(HERE, 'training_plots', 'v5')
os.makedirs(OUT_DIR, exist_ok=True)

with open(JSON) as f:
    history = json.load(f)

epochs     = [h['epoch']      for h in history]
train_loss = [h['train_loss'] for h in history]
val_psnr   = [h['val_psnr']   for h in history]
val_sam    = [h['val_sam']    for h in history]
val_ergas  = [h['val_ergas']  for h in history]
val_ssim   = [h['val_ssim']   for h in history]

# ── Style ──────────────────────────────────────────────────────────────────────
BG   = 'white'
AX   = 'white'
GRID = '#dddddd'
TEXT = '#111111'
GOLD = '#e65100'

STYLE = {
    'figure.facecolor': BG,
    'axes.facecolor':   AX,
    'axes.edgecolor':   '#333333',
    'axes.labelcolor':  TEXT,
    'xtick.color':      TEXT,
    'ytick.color':      TEXT,
    'grid.color':       GRID,
    'text.color':       TEXT,
    'legend.facecolor': 'white',
    'legend.edgecolor': '#cccccc',
}

# ── Helper ────────────────────────────────────────────────────────────────────
def save_plot(x, y, ylabel, title, filename, color, better_up=True):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))

        # Thin line + markers every 10 epochs
        ax.plot(x, y, color='#e53935', linewidth=2.0, alpha=0.9,
                marker='o', markersize=3.5,
                markevery=10, markerfacecolor='#e53935', zorder=3,
                label='Per-epoch value')

        # Smoothed trend (moving average window=10)
        if len(y) >= 10:
            kernel = np.ones(10) / 10
            smooth = np.convolve(y, kernel, mode='valid')
            x_smooth = x[9:]  # align
            ax.plot(x_smooth, smooth, color='#555555', linewidth=1.2,
                    linestyle='--', alpha=0.6, label='Moving avg (10)', zorder=2)

        # Best point
        best_idx = int(np.argmax(y)) if better_up else int(np.argmin(y))
        best_ep  = x[best_idx]
        best_val = y[best_idx]
        ax.axvline(best_ep, color=GOLD, linewidth=1.2, linestyle='--', alpha=0.7)
        ax.scatter([best_ep], [best_val], color=GOLD, zorder=5, s=90, label=f'Best: {best_val:.4f} @ ep{best_ep}')
        ax.text(best_ep + max(len(x)*0.015, 1), best_val,
                f' {best_val:.4f}\n ep {best_ep}',
                fontsize=9, color=GOLD, va='center')

        # Axis labels
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        better_sym = '↑ higher is better' if better_up else '↓ lower is better'
        ax.set_title(f'{title}  —  V5 VMamba    ({better_sym})',
                     fontsize=13, pad=8, color=TEXT)
        ax.grid(True, alpha=0.35, linestyle='--')
        ax.set_xlim(min(x) - 1, max(x) + 1)
        ax.legend(fontsize=9, loc='best')
        ax.tick_params(labelsize=10)

        fig.tight_layout()
        path = os.path.join(OUT_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Saved: {path}')

# ── Plot each metric ──────────────────────────────────────────────────────────
print(f'\nPlotting V5 metrics ({len(epochs)} epochs) …\n')

save_plot(epochs, train_loss,
          ylabel='Loss',
          title='Training Loss',
          filename='v5_train_loss.png',
          color='#e53935',
          better_up=False)

save_plot(epochs, val_psnr,
          ylabel='PSNR (dB)',
          title='Validation PSNR',
          filename='v5_val_psnr.png',
          color='#e53935',
          better_up=True)

save_plot(epochs, val_sam,
          ylabel='SAM (°)',
          title='Validation SAM',
          filename='v5_val_sam.png',
          color='#e53935',
          better_up=False)

save_plot(epochs, val_ergas,
          ylabel='ERGAS',
          title='Validation ERGAS',
          filename='v5_val_ergas.png',
          color='#e53935',
          better_up=False)

save_plot(epochs, val_ssim,
          ylabel='SSIM',
          title='Validation SSIM',
          filename='v5_val_ssim.png',
          color='#e53935',
          better_up=True)

# ── Summary table ─────────────────────────────────────────────────────────────
print('\n' + '='*55)
print(f"  {'Metric':<20} {'Best Value':>12}  {'@ Epoch':>8}")
print('='*55)
metrics = [
    ('Train Loss  (low)', train_loss, False),
    ('Val PSNR   (high)', val_psnr,   True),
    ('Val SAM     (low)', val_sam,    False),
    ('Val ERGAS   (low)', val_ergas,  False),
    ('Val SSIM   (high)', val_ssim,   True),
]
for name, vals, up in metrics:
    idx = int(np.argmax(vals)) if up else int(np.argmin(vals))
    print(f"  {name:<20} {vals[idx]:>12.5f}  ep {epochs[idx]:>4}")
print('='*55)
print(f'\nAll plots saved to: {OUT_DIR}')
