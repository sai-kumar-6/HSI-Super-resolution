"""
regenerate_metric_plots.py
Regenerates metric plots with white background and highlighted values
for V5 (200 epochs) and V6 (50 epochs).
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

_PROJECT = os.path.dirname(os.path.abspath(__file__))

CONFIGS = [
    {
        'label'    : 'V5 VMamba (True Parallel Scan)',
        'history'  : os.path.join(_PROJECT, 'version5', 'comparison', 'results', 'v5_training_history.json'),
        'out_dir'  : os.path.join(_PROJECT, 'version5', 'comparison', 'metric_plots', 'v5'),
        'prefix'   : 'v5',
    },
    {
        'label'    : 'V6 VMamba (Spectral-Focused Loss)',
        'history'  : os.path.join(_PROJECT, 'version6', 'comparison', 'results', 'v6_training_history.json'),
        'out_dir'  : os.path.join(_PROJECT, 'version6', 'comparison', 'metric_plots', 'v6'),
        'prefix'   : 'v6',
    },
]

METRICS = [
    # (key,          title,              ylabel,       higher_better, line_color)
    ('train_loss', 'Training Loss',    'Loss',        False, '#e53935'),
    ('val_psnr',   'Validation PSNR',  'PSNR (dB)',   True,  '#1565c0'),
    ('val_sam',    'Validation SAM',   'SAM (deg)',   False, '#2e7d32'),
    ('val_ergas',  'Validation ERGAS', 'ERGAS',       False, '#e65100'),
    ('val_ssim',   'Validation SSIM',  'SSIM',        True,  '#6a1b9a'),
]


def moving_avg(vals, w=5):
    result = []
    for i in range(len(vals)):
        lo = max(0, i - w + 1)
        result.append(np.mean(vals[lo:i+1]))
    return result


def make_combined_plot(cfg, history):
    epochs = [h['epoch'] for h in history]
    os.makedirs(cfg['out_dir'], exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle(cfg['label'], fontsize=14, fontweight='bold', color='#1a1a1a', y=1.02)

    for ax, (key, title, ylabel, higher, color) in zip(axes, METRICS):
        vals = [h[key] for h in history]
        ma   = moving_avg(vals, w=max(1, len(vals)//20))

        ax.set_facecolor('white')
        for sp in ax.spines.values():
            sp.set_color('#cccccc')
            sp.set_linewidth(0.8)
        ax.tick_params(colors='#333333', labelsize=8.5, length=3)
        ax.grid(True, color='#e0e0e0', linestyle='--', linewidth=0.7, alpha=0.9)

        # Main line
        ax.plot(epochs, vals, color=color, linewidth=1.6, alpha=0.35, label='Per epoch')
        # Moving average
        ax.plot(epochs, ma, color=color, linewidth=2.2, label='Moving avg')

        # Best value
        best_idx = (np.argmax(vals) if higher else np.argmin(vals))
        best_e, best_v = epochs[best_idx], vals[best_idx]

        # Vertical dashed line at best epoch
        ax.axvline(best_e, color='#ff6f00', linewidth=1.2, linestyle='--', alpha=0.85)
        # Bold dot at best
        ax.scatter([best_e], [best_v], color='#ff6f00', zorder=6, s=60, linewidths=0)

        # Highlighted box annotation
        ax.annotate(
            f'Best: {best_v:.4f}\n@ ep {best_e}',
            xy=(best_e, best_v),
            xytext=(0.97, 0.97 if higher else 0.03),
            textcoords='axes fraction',
            ha='right', va='top' if higher else 'bottom',
            fontsize=8.5, fontweight='bold', color='#b71c1c',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#fff9c4',
                      edgecolor='#ff6f00', linewidth=1.2, alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='#ff6f00',
                            lw=1.2, connectionstyle='arc3,rad=0.2'),
        )

        # First and last value labels on x-axis
        ax.annotate(f'{vals[0]:.3f}', xy=(epochs[0], vals[0]),
                    xytext=(epochs[0], vals[0]),
                    fontsize=7.5, color='#555555', ha='left', va='bottom')
        ax.annotate(f'{vals[-1]:.3f}', xy=(epochs[-1], vals[-1]),
                    xytext=(epochs[-1], vals[-1]),
                    fontsize=7.5, color='#555555', ha='right', va='top')

        # Highlight every 25th epoch value with a tick mark
        for h in history:
            if h['epoch'] % 25 == 0:
                ax.scatter([h['epoch']], [h[key]], color=color,
                           s=22, zorder=5, marker='D', linewidths=0, alpha=0.8)
                ax.annotate(
                    f"{h[key]:.3f}",
                    xy=(h['epoch'], h[key]),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=7, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='#bbbbbb', linewidth=0.8, alpha=0.9),
                )

        ax.set_title(title, fontsize=11, fontweight='bold', color='#1a1a1a', pad=6)
        ax.set_xlabel('Epoch', fontsize=9, color='#444444')
        ax.set_ylabel(ylabel, fontsize=9, color='#444444')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))

    plt.tight_layout(pad=1.5)
    out = os.path.join(cfg['out_dir'], f"{cfg['prefix']}_metrics_ep{epochs[-1]:04d}.png")
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


def make_individual_plots(cfg, history):
    """Also save each metric as a separate full-size plot."""
    epochs = [h['epoch'] for h in history]
    ind_dir = os.path.join(cfg['out_dir'], 'individual')
    os.makedirs(ind_dir, exist_ok=True)

    for key, title, ylabel, higher, color in METRICS:
        vals = [h[key] for h in history]
        ma   = moving_avg(vals, w=max(1, len(vals)//20))

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        for sp in ax.spines.values():
            sp.set_color('#cccccc'); sp.set_linewidth(0.8)
        ax.tick_params(colors='#333333', labelsize=9, length=3)
        ax.grid(True, color='#e0e0e0', linestyle='--', linewidth=0.8, alpha=0.9)

        ax.plot(epochs, vals, color=color, linewidth=1.4, alpha=0.3, label='Per epoch')
        ax.plot(epochs, ma,   color=color, linewidth=2.4, label='Moving avg')

        best_idx = (np.argmax(vals) if higher else np.argmin(vals))
        best_e, best_v = epochs[best_idx], vals[best_idx]

        ax.axvline(best_e, color='#ff6f00', linewidth=1.4, linestyle='--', alpha=0.85)
        ax.scatter([best_e], [best_v], color='#ff6f00', zorder=6, s=80, linewidths=0)
        ax.annotate(
            f'Best: {best_v:.4f}\n@ epoch {best_e}',
            xy=(best_e, best_v),
            xytext=(0.97, 0.97 if higher else 0.05),
            textcoords='axes fraction',
            ha='right', va='top' if higher else 'bottom',
            fontsize=10, fontweight='bold', color='#b71c1c',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#fff9c4',
                      edgecolor='#ff6f00', linewidth=1.5, alpha=0.97),
            arrowprops=dict(arrowstyle='->', color='#ff6f00',
                            lw=1.4, connectionstyle='arc3,rad=0.2'),
        )

        # Highlight every 25th epoch
        for h in history:
            if h['epoch'] % 25 == 0:
                v = h[key]
                ax.scatter([h['epoch']], [v], color=color,
                           s=35, zorder=5, marker='D', linewidths=0, alpha=0.85)
                ax.annotate(
                    f"ep{h['epoch']}\n{v:.3f}",
                    xy=(h['epoch'], v),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='#222222',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#aaaaaa', linewidth=1.0, alpha=0.93),
                )

        ax.set_title(f'{cfg["label"]} — {title}', fontsize=12,
                     fontweight='bold', color='#1a1a1a', pad=8)
        ax.set_xlabel('Epoch', fontsize=10, color='#444444')
        ax.set_ylabel(ylabel, fontsize=10, color='#444444')
        ax.legend(fontsize=9, framealpha=0.9, edgecolor='#cccccc')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))

        plt.tight_layout()
        safe_key = key.replace('_', '-')
        out = os.path.join(ind_dir, f"{cfg['prefix']}_{safe_key}.png")
        fig.savefig(out, dpi=130, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'    {out.split(os.sep)[-1]}')


if __name__ == '__main__':
    for cfg in CONFIGS:
        if not os.path.isfile(cfg['history']):
            print(f'[SKIP] {cfg["prefix"]} — history not found: {cfg["history"]}')
            continue
        history = json.load(open(cfg['history']))
        print(f'\n[{cfg["prefix"].upper()}] {len(history)} epochs')
        make_combined_plot(cfg, history)
        make_individual_plots(cfg, history)

    print('\nAll plots saved.')
