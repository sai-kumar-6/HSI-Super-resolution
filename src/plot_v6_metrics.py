"""
plot_v6_metrics.py
Generates white-background metric plots from v6_training_history.json.
Saves: one combined PNG + 5 individual PNGs.
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

_PROJECT  = os.path.dirname(os.path.abspath(__file__))
HIST_PATH = os.path.join(_PROJECT, 'version6', 'comparison', 'results', 'v6_training_history.json')
OUT_DIR   = os.path.join(_PROJECT, 'version6', 'v6_metric_plots')
LABEL     = 'V6 VMamba — Spectral-Focused Loss'

METRICS = [
    # (json_key,      plot_title,         y_label,      higher_better, line_color)
    ('train_loss', 'Training Loss',    'Loss',        False, '#e53935'),
    ('val_psnr',   'Validation PSNR',  'PSNR (dB)',   True,  '#1565c0'),
    ('val_sam',    'Validation SAM',   'SAM (deg)',   False, '#2e7d32'),
    ('val_ergas',  'Validation ERGAS', 'ERGAS',       False, '#e65100'),
    ('val_ssim',   'Validation SSIM',  'SSIM',        True,  '#6a1b9a'),
]


def moving_avg(vals, w=5):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - w + 1)
        out.append(np.mean(vals[lo:i+1]))
    return out


def style_ax(ax):
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_color('#cccccc')
        sp.set_linewidth(0.8)
    ax.tick_params(colors='#333333', labelsize=9, length=3)
    ax.grid(True, color='#e0e0e0', linestyle='--', linewidth=0.7, alpha=1.0)


def annotate_best(ax, epochs, vals, higher, color):
    best_idx  = np.argmax(vals) if higher else np.argmin(vals)
    best_e    = epochs[best_idx]
    best_v    = vals[best_idx]
    ax.axvline(best_e, color='#ff6f00', linewidth=1.3, linestyle='--', alpha=0.9, zorder=3)
    ax.scatter([best_e], [best_v], color='#ff6f00', s=70, zorder=7, linewidths=0)
    ax.annotate(
        f'Best: {best_v:.4f}\n@ epoch {best_e}',
        xy=(best_e, best_v),
        xytext=(0.97, 0.97 if higher else 0.05),
        textcoords='axes fraction',
        ha='right', va='top' if higher else 'bottom',
        fontsize=9, fontweight='bold', color='#b71c1c',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff9c4',
                  edgecolor='#ff6f00', linewidth=1.3, alpha=0.97),
        arrowprops=dict(arrowstyle='->', color='#ff6f00',
                        lw=1.3, connectionstyle='arc3,rad=0.2'),
        zorder=10,
    )
    return best_e, best_v


def mark_milestones(ax, history, key, color, step=10):
    for h in history:
        if h['epoch'] % step == 0:
            v = h[key]
            ax.scatter([h['epoch']], [v], color=color, s=28,
                       zorder=6, marker='D', linewidths=0, alpha=0.9)
            ax.annotate(
                f"{v:.3f}",
                xy=(h['epoch'], v),
                xytext=(4, 4), textcoords='offset points',
                fontsize=7.5, color='#222222',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='#aaaaaa', linewidth=0.8, alpha=0.95),
                zorder=9,
            )


# ── 1. Combined plot (1 row × 5 metrics) ─────────────────────────────────────
def make_combined(history):
    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 5, figsize=(28, 5.2))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'{LABEL}  |  {len(history)} Epochs',
                 fontsize=13, fontweight='bold', color='#1a1a1a', y=1.02)

    for ax, (key, title, ylabel, higher, color) in zip(axes, METRICS):
        vals = [h[key] for h in history]

        style_ax(ax)
        ax.plot(epochs, vals, color=color, linewidth=1.8)
        annotate_best(ax, epochs, vals, higher, color)
        mark_milestones(ax, history, key, color, step=10)

        ax.set_title(title, fontsize=10.5, fontweight='bold', color='#1a1a1a', pad=5)
        ax.set_xlabel('Epoch', fontsize=9, color='#444444')
        ax.set_ylabel(ylabel,  fontsize=9, color='#444444')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))

    plt.tight_layout(pad=1.5)
    out = os.path.join(OUT_DIR, f'v6_metrics_ep{epochs[-1]:04d}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Combined : {out}')


# ── 2. Individual plots (one per metric) ──────────────────────────────────────
def make_individual(history):
    epochs  = [h['epoch'] for h in history]
    for key, title, ylabel, higher, color in METRICS:
        vals = [h[key] for h in history]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        fig.patch.set_facecolor('white')
        style_ax(ax)

        ax.plot(epochs, vals, color=color, linewidth=2.0)
        annotate_best(ax, epochs, vals, higher, color)
        mark_milestones(ax, history, key, color, step=10)

        # Start and end value labels
        ax.annotate(f'Start\n{vals[0]:.4f}', xy=(epochs[0], vals[0]),
                    xytext=(6, 6), textcoords='offset points',
                    fontsize=8, color='#555555',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#cccccc', linewidth=0.8, alpha=0.9))
        ax.annotate(f'End\n{vals[-1]:.4f}', xy=(epochs[-1], vals[-1]),
                    xytext=(-6, 6), textcoords='offset points',
                    fontsize=8, color='#555555', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#cccccc', linewidth=0.8, alpha=0.9))

        ax.set_title(f'{LABEL}\n{title}', fontsize=12,
                     fontweight='bold', color='#1a1a1a', pad=8)
        ax.set_xlabel('Epoch', fontsize=10, color='#444444')
        ax.set_ylabel(ylabel,  fontsize=10, color='#444444')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))

        plt.tight_layout()
        fname = os.path.join(OUT_DIR, f'v6_{key.replace("_","-")}.png')
        fig.savefig(fname, dpi=130, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Individual: {os.path.basename(fname)}')


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    history = json.load(open(HIST_PATH))
    print(f'Loaded {len(history)} epochs from v6_training_history.json\n')
    make_combined(history)
    make_individual(history)
    print('\nDone.')
