"""
plot_training_metrics.py
────────────────────────
Reads training history and draws per-metric graphs.

Two input modes
───────────────
1. Auto-detect JSON files saved by compare_all.py:
       python plot_training_metrics.py

2. Parse a terminal log file you saved with:
       python ... > training_log.txt 2>&1
   Then:
       python plot_training_metrics.py --log training_log.txt

3. Specify a single JSON history file directly:
       python plot_training_metrics.py --json version5/comparison/results/v5_training_history.json

Output
──────
   training_plots/
       <variant>_all_metrics.png   — 5-panel grid (loss, PSNR, SAM, ERGAS, SSIM)
       <variant>_loss.png
       <variant>_psnr.png
       <variant>_sam.png
       <variant>_ergas.png
       <variant>_ssim.png
       all_variants_psnr.png       — overlay of all variants on one PSNR chart
       all_variants_ergas.png
"""

from __future__ import annotations
import os
import re
import sys
import json
import argparse
import glob as _glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

_HERE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(_HERE, 'training_plots')

# Colour per variant (matches compare_all.py)
COLOURS = {
    'old':      '#1f77b4',
    'improved': '#ff7f0e',
    'v3':       '#2ca02c',
    'v4':       '#9467bd',
    'v5':       '#d62728',
}
LABELS = {
    'old':      'V1 Old VMamba',
    'improved': 'V2 Improved VMamba',
    'v3':       'V3 VMamba',
    'v4':       'V4 VMamba (Spectral Fix)',
    'v5':       'V5 VMamba (True Parallel Scan)',
}

# Regex that matches lines like:
#   Ep  10/50  loss=0.0306  PSNR=43.07  SAM=6.739  ERGAS=7.773  SSIM=0.9622  [309.6s]
_EP_RE = re.compile(
    r'Ep\s+(\d+)/\d+\s+'
    r'loss=([^\s]+)\s+'
    r'PSNR=([^\s]+)\s+'
    r'SAM=([^\s]+)\s+'
    r'ERGAS=([^\s]+)\s+'
    r'SSIM=([^\s]+)',
    re.IGNORECASE,
)


# ── Parsers ──────────────────────────────────────────────────────────────────

def parse_log_file(path: str) -> dict[str, list[dict]]:
    """
    Parse a terminal log file that may contain runs for one or more variants.
    Variant sections start with a line containing the variant label, e.g.:
        Training: V5 VMamba (True Parallel Scan)
    If no variant header is found, all epochs are stored under 'unknown'.
    """
    label_to_key = {v: k for k, v in LABELS.items()}
    histories: dict[str, list] = {}
    current = 'unknown'

    with open(path, 'r', errors='replace') as f:
        for line in f:
            # Detect variant section header
            for label, key in label_to_key.items():
                if label.lower() in line.lower() and 'training' in line.lower():
                    current = key
                    if current not in histories:
                        histories[current] = []
                    break
            m = _EP_RE.search(line)
            if m:
                ep, loss, psnr, sam, ergas, ssim = m.groups()
                try:
                    rec = {
                        'epoch':      int(ep),
                        'train_loss': float(loss) if loss.lower() != 'nan' else float('nan'),
                        'val_psnr':   float(psnr) if psnr.lower() != 'nan' else float('nan'),
                        'val_sam':    float(sam)  if sam.lower()  != 'nan' else float('nan'),
                        'val_ergas':  float(ergas) if ergas.lower() != 'nan' else float('nan'),
                        'val_ssim':   float(ssim) if ssim.lower() != 'nan' else float('nan'),
                    }
                    if current not in histories:
                        histories[current] = []
                    histories[current].append(rec)
                except ValueError:
                    pass

    return histories


def load_json_history(path: str, variant: str | None = None) -> dict[str, list[dict]]:
    with open(path) as f:
        data = json.load(f)
    if variant is None:
        # Guess variant from filename
        base = os.path.basename(path)
        for k in COLOURS:
            if k in base:
                variant = k
                break
        if variant is None:
            variant = 'unknown'
    return {variant: data}


def find_all_json_histories() -> dict[str, list[dict]]:
    """Search all known compare_all results folders for history JSONs."""
    search_dirs = [
        os.path.join(_HERE, 'version3', 'comparison', 'results'),
        os.path.join(_HERE, 'version4', 'comparison', 'results'),
        os.path.join(_HERE, 'version5', 'comparison', 'results'),
    ]
    histories: dict[str, list] = {}
    for d in search_dirs:
        for fp in _glob.glob(os.path.join(d, '*_training_history.json')):
            # Extract variant name from filename: v5_training_history.json → v5
            base = os.path.splitext(os.path.basename(fp))[0]  # e.g. v5_training_history
            variant = base.replace('_training_history', '')
            try:
                with open(fp) as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    histories[variant] = data
                    print(f"  Loaded: {fp}  ({len(data)} epochs)")
            except Exception as e:
                print(f"  [WARN] Could not read {fp}: {e}")
    return histories


# ── Plotting ──────────────────────────────────────────────────────────────────

METRIC_CFG = [
    # (key,           ylabel,            title,                   better)
    ('train_loss', 'Loss',             'Training Loss',          '↓'),
    ('val_psnr',   'PSNR (dB)',        'Validation PSNR',        '↑'),
    ('val_sam',    'SAM (°)',           'Validation SAM',         '↓'),
    ('val_ergas',  'ERGAS',            'Validation ERGAS',       '↓'),
    ('val_ssim',   'SSIM',             'Validation SSIM',        '↑'),
]

STYLE = {
    'figure.facecolor':  '#0f0f1a',
    'axes.facecolor':    '#1a1a2e',
    'axes.edgecolor':    '#444',
    'axes.labelcolor':   '#e0e0e0',
    'xtick.color':       '#aaa',
    'ytick.color':       '#aaa',
    'grid.color':        '#333',
    'text.color':        '#e0e0e0',
    'legend.facecolor':  '#1a1a2e',
    'legend.edgecolor':  '#555',
}


def _apply_style(ax, title, ylabel, better):
    ax.set_title(f'{title}  ({better})', fontsize=13, pad=6, color='#e0e0e0')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)


def plot_single_variant(variant: str, history: list[dict], out_dir: str):
    """Draw a 5-panel grid for one variant and also individual PNGs."""
    epochs = [h['epoch'] for h in history]
    color  = COLOURS.get(variant, '#aaaaaa')
    label  = LABELS.get(variant, variant)

    os.makedirs(out_dir, exist_ok=True)

    # ── 5-panel combined figure ────────────────────────────────────────────
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 5, figsize=(28, 5))
        fig.suptitle(f'Training History — {label}', fontsize=15, y=1.02, color='#e0e0e0')

        for ax, (key, ylabel, title, better) in zip(axes, METRIC_CFG):
            vals = [h.get(key, float('nan')) for h in history]
            ax.plot(epochs, vals, color=color, linewidth=2.0, marker='o',
                    markersize=3, markevery=max(1, len(epochs)//20))
            # Mark best point
            valid = [(e, v) for e, v in zip(epochs, vals) if not np.isnan(v)]
            if valid:
                if better == '↑':
                    best_e, best_v = max(valid, key=lambda x: x[1])
                else:
                    best_e, best_v = min(valid, key=lambda x: x[1])
                ax.axvline(best_e, color='#ffcc00', linewidth=1, linestyle='--', alpha=0.7)
                ax.scatter([best_e], [best_v], color='#ffcc00', zorder=5, s=60)
                ax.text(best_e, best_v, f'  {best_v:.3f}\n  ep{best_e}',
                        fontsize=8.5, color='#ffcc00', va='center')
            _apply_style(ax, title, ylabel, better)

        plt.tight_layout()
        combo_path = os.path.join(out_dir, f'{variant}_all_metrics.png')
        fig.savefig(combo_path, dpi=140, bbox_inches='tight',
                    facecolor=STYLE['figure.facecolor'])
        plt.close(fig)
        print(f'  Saved: {combo_path}')

    # ── Individual per-metric figures ──────────────────────────────────────
    for key, ylabel, title, better in METRIC_CFG:
        vals = [h.get(key, float('nan')) for h in history]
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, vals, color=color, linewidth=2.2, marker='o',
                    markersize=4, markevery=max(1, len(epochs)//15))
            valid = [(e, v) for e, v in zip(epochs, vals) if not np.isnan(v)]
            if valid:
                if better == '↑':
                    best_e, best_v = max(valid, key=lambda x: x[1])
                else:
                    best_e, best_v = min(valid, key=lambda x: x[1])
                ax.axvline(best_e, color='#ffcc00', linewidth=1.5,
                           linestyle='--', alpha=0.8, label=f'Best ep={best_e} → {best_v:.4f}')
                ax.scatter([best_e], [best_v], color='#ffcc00', zorder=5, s=80)
            ax.legend(fontsize=10, loc='best')
            _apply_style(ax, f'{title} — {label}', ylabel, better)
            fig.tight_layout()
            save_path = os.path.join(out_dir, f'{variant}_{key.replace("val_","")}.png')
            fig.savefig(save_path, dpi=140, bbox_inches='tight',
                        facecolor=STYLE['figure.facecolor'])
            plt.close(fig)
            print(f'  Saved: {save_path}')


def plot_overlay(histories: dict[str, list[dict]], out_dir: str):
    """Overlay all variants on PSNR and ERGAS charts."""
    os.makedirs(out_dir, exist_ok=True)
    overlay_metrics = [
        ('val_psnr',  'PSNR (dB)', 'PSNR Comparison — All Variants', '↑'),
        ('val_ergas', 'ERGAS',     'ERGAS Comparison — All Variants', '↓'),
        ('val_ssim',  'SSIM',      'SSIM Comparison — All Variants',  '↑'),
        ('train_loss','Loss',      'Training Loss — All Variants',     '↓'),
    ]
    for key, ylabel, title, better in overlay_metrics:
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(11, 6))
            for variant, history in histories.items():
                epochs = [h['epoch'] for h in history]
                vals   = [h.get(key, float('nan')) for h in history]
                color  = COLOURS.get(variant, '#aaaaaa')
                label  = LABELS.get(variant, variant)
                ax.plot(epochs, vals, color=color, linewidth=2.2,
                        label=label, marker='o', markersize=3,
                        markevery=max(1, len(epochs)//15))
            ax.legend(fontsize=10, loc='best')
            _apply_style(ax, title, ylabel, better)
            fig.tight_layout()
            fname = key.replace('val_', '').replace('train_', '')
            save_path = os.path.join(out_dir, f'all_variants_{fname}.png')
            fig.savefig(save_path, dpi=140, bbox_inches='tight',
                        facecolor=STYLE['figure.facecolor'])
            plt.close(fig)
            print(f'  Saved: {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot VMamba training metrics')
    parser.add_argument('--log',     type=str, default=None,
                        help='Path to a terminal log file (plain text with Ep ... lines)')
    parser.add_argument('--json',    type=str, default=None,
                        help='Path to a single *_training_history.json file')
    parser.add_argument('--variant', type=str, default=None,
                        help='Variant name when using --json (old/improved/v3/v4/v5)')
    parser.add_argument('--out',     type=str, default=OUT_DIR,
                        help=f'Output directory (default: {OUT_DIR})')
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # ── Collect histories ──────────────────────────────────────────────────
    if args.log:
        print(f'Parsing log file: {args.log}')
        histories = parse_log_file(args.log)
        if not histories:
            print('[ERROR] No "Ep ... loss= ..." lines found in the log file.')
            sys.exit(1)

    elif args.json:
        print(f'Loading JSON: {args.json}')
        histories = load_json_history(args.json, args.variant)

    else:
        print('Auto-searching for training history JSON files …')
        histories = find_all_json_histories()
        if not histories:
            print('\n[INFO] No history JSON files found yet.')
            print('Options:')
            print('  1. Redirect terminal output: python ... > log.txt 2>&1')
            print('     Then: python plot_training_metrics.py --log log.txt')
            print('  2. After training completes, JSON files will be auto-saved to:')
            print('     version5/comparison/results/<variant>_training_history.json')
            sys.exit(0)

    if not histories:
        print('[ERROR] No training data found.')
        sys.exit(1)

    print(f'\nPlotting {len(histories)} variant(s): {list(histories.keys())}')
    print(f'Output directory: {out_dir}\n')

    # ── Per-variant plots ──────────────────────────────────────────────────
    for variant, history in histories.items():
        if not history:
            print(f'  [SKIP] {variant}: empty history')
            continue
        # Filter out nan-only epochs for cleaner display but keep them for context
        non_nan = sum(1 for h in history if not np.isnan(h.get('val_psnr', float('nan'))))
        print(f'  Plotting {variant}: {len(history)} epochs, {non_nan} with valid PSNR')
        plot_single_variant(variant, history, out_dir)

    # ── Overlay comparison ─────────────────────────────────────────────────
    if len(histories) > 1:
        print('\n  Drawing overlay comparison charts …')
        plot_overlay(histories, out_dir)

    print(f'\nDone. All plots saved to: {out_dir}')


if __name__ == '__main__':
    main()
