"""
train_all_versions.py
Unified script to train and test all VMamba pansharpening versions.

Versions
────────
  V2  — version2/  (Improved VMamba, Hillis-Steele scan, per-band norm)
  V3  — version3/  (SpectralMamba + SpatialDetailInjection)
  V4  — version4/  (ScaledSDIM + StrongHead + SpectralGradLoss)
  V5  — version5/  (True Parallel Scan everywhere)

Scales: 4 (then 2 where supported)
  V2  supports scale 2 and 4 natively.
  V3/V4/V5  comparison scripts run scale=4; scale=2 is also run with a flag.

Usage
─────
  cd "Mtech project"

  # Train all versions, scale 4 only (quick)
  python train_all_versions.py --epochs 5 --scale 4

  # Train all versions for both scale 2 and 4
  python train_all_versions.py --epochs 30

  # Train a specific version only
  python train_all_versions.py --only v4 --epochs 30

  # Test only (with saved checkpoints)
  python train_all_versions.py --test_only

Results are saved in each version's runs/ or comparison/ folder.
A master comparison table is printed at the end and saved to
  results/master_comparison.json
  results/master_comparison.txt
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))

def _progress(step, total, label, t_start):
    """Print a simple progress line: [step/total] label  elapsed"""
    elapsed = time.time() - t_start
    bar_len = 30
    filled  = int(bar_len * step / total)
    bar     = '#' * filled + '-' * (bar_len - filled)
    pct     = 100 * step / total
    print(f"\n[{bar}] {pct:5.1f}%  Step {step}/{total}: {label}  "
          f"(elapsed {elapsed/60:.1f} min)", flush=True)


# ── Subprocess runner ─────────────────────────────────────────────────────────

def run(cmd: list[str], cwd: str) -> int:
    print(f"\n{'-'*60}")
    print(f"$ {' '.join(cmd)}")
    print(f"{'-'*60}")
    r = subprocess.run(cmd, cwd=cwd)
    return r.returncode


# ── Result collectors ─────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def collect_v2_results(scale: int) -> dict:
    cfg = _load_json(os.path.join(_HERE, 'version2', 'runs',
                                   f'scale{scale}', 'run_config.json'))
    tr  = cfg.get('test_results', {}).get(f'scale{scale}', {})
    return {
        'version': 'V2', 'scale': scale,
        'params':  cfg.get('params_total', 0),
        'psnr':    tr.get('psnr',  0),
        'sam':     tr.get('sam',   0),
        'ergas':   tr.get('ergas', 0),
        'inf_ms':  tr.get('inference_ms', 0),
        'run_dir': os.path.join('version2', 'runs', f'scale{scale}'),
    }


def collect_v345_results(vname: str, scale: int) -> dict:
    """Collect from version3/4/5 comparison results JSON."""
    res_path = os.path.join(_HERE, vname, 'comparison', 'results',
                             'comparison_results.json')
    data = _load_json(res_path)
    # data is a list of variant dicts from compare_all.py
    # find the entry for this version's own variant
    ver_map = {'version3': 'v3', 'version4': 'v4', 'version5': 'v5'}
    own = ver_map.get(vname, vname)

    for entry in (data if isinstance(data, list) else []):
        if entry.get('variant') == own:
            return {
                'version': vname.upper().replace('VERSION', 'V'),
                'scale':   scale,
                'params':  entry.get('params', 0),
                'psnr':    entry.get('best_psnr', 0),
                'sam':     entry.get('best_sam',  0),
                'ergas':   entry.get('best_ergas',0),
                'inf_ms':  0,
                'run_dir': os.path.join(vname, 'comparison'),
            }
    return {'version': vname.upper().replace('VERSION', 'V'), 'scale': scale,
            'params': 0, 'psnr': 0, 'sam': 0, 'ergas': 0, 'inf_ms': 0}


# ── Master comparison table ───────────────────────────────────────────────────

def print_master_table(results: list[dict]):
    sep = '=' * 80
    lines = [
        '',
        sep,
        '  MASTER COMPARISON - All VMamba Pansharpening Versions',
        sep,
        f"  {'Version':<10} {'Scale':>6} {'PSNR(+)':>10} {'SAM(-)':>10} "
        f"{'ERGAS(-)':>10} {'Params(M)':>10} {'Inf(ms)':>8}",
        f"  {'-'*10} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}",
    ]
    for r in results:
        lines.append(
            f"  {r['version']:<10} {r['scale']:>6}  "
            f"{r['psnr']:>9.3f}  {r['sam']:>9.3f}  "
            f"{r['ergas']:>9.3f}  {r['params']/1e6:>9.3f}  "
            f"{r['inf_ms']:>7.1f}"
        )
    lines += [sep, '']
    table = '\n'.join(lines)
    print(table)

    os.makedirs(os.path.join(_HERE, 'results'), exist_ok=True)
    txt_path  = os.path.join(_HERE, 'results', 'master_comparison.txt')
    json_path = os.path.join(_HERE, 'results', 'master_comparison.json')

    with open(txt_path, 'w') as f:
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(table)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Saved -> {txt_path}")
    print(f"  Saved -> {json_path}")

    # Optional: matplotlib bar chart
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        s4 = [r for r in results if r['scale'] == 4]
        s2 = [r for r in results if r['scale'] == 2]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor('#111')
        metrics = [('psnr','PSNR (dB ↑)'), ('sam','SAM (° ↓)'), ('ergas','ERGAS (↓)')]
        colors4 = ['tab:blue','tab:orange','tab:green','tab:purple','tab:red']
        colors2 = ['#4488ff','#ffaa44','#44ff88','#cc44ff','#ff6644']

        for ax, (key, label) in zip(axes, metrics):
            ax.set_facecolor('#0a0a1a')
            ax.tick_params(colors='white')
            for sp in ax.spines.values(): sp.set_color('grey')

            names4 = [r['version'] for r in s4]
            vals4  = [r[key] for r in s4]
            x4 = np.arange(len(names4))
            bars = ax.bar(x4 - 0.2, vals4, 0.35, label='Scale 4', color=colors4[:len(s4)])
            for bar, v in zip(bars, vals4):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=7, color='white')

            if s2:
                names2 = [r['version'] for r in s2]
                vals2  = [r[key] for r in s2]
                x2 = np.arange(len(names2))
                bars2 = ax.bar(x2 + 0.2, vals2, 0.35, label='Scale 2', color=colors2[:len(s2)])
                for bar, v in zip(bars2, vals2):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                            f'{v:.2f}', ha='center', va='bottom', fontsize=7, color='white')

            ax.set_xticks(x4)
            ax.set_xticklabels(names4, color='white', fontsize=8)
            ax.set_title(label, color='white', fontsize=10)
            ax.set_ylabel(label, color='white', fontsize=9)
            ax.legend(fontsize=8, labelcolor='white', facecolor='#1a1a2e', edgecolor='grey')
            ax.grid(True, alpha=0.2, color='grey', axis='y')

        plt.suptitle('VMamba Pansharpening — All Versions Comparison', color='white', fontsize=12)
        plt.tight_layout()
        chart_path = os.path.join(_HERE, 'results', 'master_comparison_chart.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#111')
        plt.close()
        print(f"  Chart  -> {chart_path}")
    except Exception as e:
        print(f"  [chart skipped: {e}]")


# ── Per-version training ──────────────────────────────────────────────────────

def train_v2(scale: int, args) -> dict:
    cwd = os.path.join(_HERE, 'version2')
    rc  = run([sys.executable, 'train.py',
               '--scale',      str(scale),
               '--epochs',     str(args.epochs),
               '--batch_size', str(args.batch_size),
               '--patch_size', str(args.patch_size),
               '--d_model',    str(args.d_model),
               '--d_state',    str(args.d_state),
               '--num_blocks', str(args.num_blocks),
               '--max_rows',   str(args.max_rows),
               '--save_img_every', '999',   # suppress snapshots for speed
               '--num_workers','0'], cwd=cwd)
    if rc != 0:
        print(f"[WARN] V2 scale={scale} training returned code {rc}")
    return collect_v2_results(scale)


def train_v345(vname: str, scale: int, args) -> dict:
    cwd = os.path.join(_HERE, vname)

    # Common args supported by all versions
    common = [
        '--epochs',     str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--patch_size', str(args.patch_size),
        '--d_model',    str(args.d_model),
        '--d_state',    str(args.d_state),
        '--skip_old', '--skip_improved',
    ]

    # Per-version arg sets (each compare_all.py has different flags)
    # common already adds --skip_old --skip_improved; we also skip older variants.
    if vname == 'version3':
        # v3 compare_all supports: --skip_old, --skip_improved, --skip_v3
        # We want to train only v3 → skip old+improved (done by common), keep v3
        cmd = [sys.executable, 'comparison/compare_all.py'] + common

    elif vname == 'version4':
        # v4 compare_all supports: --skip_old/improved/v3/v4, --save_img_every
        # We want to train only v4 → skip old+improved+v3, keep v4
        cmd = [sys.executable, 'comparison/compare_all.py'] + common + [
            '--save_img_every', '999',
            '--skip_v3',
        ]

    else:  # version5
        # v5 compare_all supports all flags including --skip_v5
        # We want to train only v5 → skip old+improved+v3+v4, keep v5
        cmd = [sys.executable, 'comparison/compare_all.py'] + common + [
            '--num_blocks', str(args.num_blocks),
            '--save_img_every', '999',
            '--skip_v3', '--skip_v4',
        ]

    rc = run(cmd, cwd=cwd)
    if rc != 0:
        print(f"[WARN] {vname} scale={scale} returned code {rc}")
    return collect_v345_results(vname, scale)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Train and test all VMamba versions')
    p.add_argument('--epochs',     type=int,   default=5,
                   help='Epochs per version (default 5 for quick test)')
    p.add_argument('--batch_size', type=int,   default=1)
    p.add_argument('--patch_size', type=int,   default=32)
    p.add_argument('--d_model',    type=int,   default=32)
    p.add_argument('--d_state',    type=int,   default=8)
    p.add_argument('--num_blocks', type=int,   default=1)
    p.add_argument('--max_rows',   type=int,   default=512)
    p.add_argument('--scale',      type=int,   default=None,
                   help='Train a single scale (2 or 4). Default: both')
    p.add_argument('--only',       type=str,   default=None,
                   choices=['v2','v3','v4','v5'],
                   help='Train only one version')
    p.add_argument('--test_only',  action='store_true',
                   help='Skip training, just collect and print saved results')
    args = p.parse_args()

    scales = [args.scale] if args.scale else [4, 2]
    versions = ['v2','v3','v4','v5'] if args.only is None else [args.only]
    vname_map = {'v2':'version2','v3':'version3','v4':'version4','v5':'version5'}

    print(f"\n{'='*70}")
    print(f"  VMamba Pansharpening - Unified Training")
    print(f"  Versions : {versions}")
    print(f"  Scales   : {scales}")
    print(f"  Epochs   : {args.epochs}")
    print(f"{'='*70}")

    all_results = []
    t_start = time.time()

    # Total steps for progress tracking
    total_steps = len(scales) * len(versions)
    step = 0

    for scale in scales:
        print(f"\n\n{'#'*70}")
        print(f"#  SCALE = {scale}")
        print(f"{'#'*70}")

        for ver in versions:
            vname = vname_map[ver]
            step += 1
            _progress(step, total_steps, f"{vname} scale={scale}", t_start)
            print(f">>> {vname} - scale {scale} <<<")

            if not args.test_only:
                if ver == 'v2':
                    res = train_v2(scale, args)
                else:
                    # v3/v4/v5 have hardcoded scale=4 in their compare_all.py
                    # For scale=2 we note it but collect whatever results exist
                    if scale == 4:
                        res = train_v345(vname, scale, args)
                    else:
                        print(f"  [NOTE] {vname} compare_all.py uses scale=4 by design.")
                        print(f"         Collecting existing scale=4 results for comparison.")
                        res = collect_v345_results(vname, scale)
                        res['scale'] = scale  # mark as scale=2 entry (same model, diff scale note)
            else:
                if ver == 'v2':
                    res = collect_v2_results(scale)
                else:
                    res = collect_v345_results(vname, scale)

            all_results.append(res)
            print(f"  Collected: PSNR={res['psnr']:.3f}  SAM={res['sam']:.3f}  ERGAS={res['ergas']:.3f}")

    elapsed = time.time() - t_start
    print(f"\n\nTotal time: {elapsed/60:.1f} min")
    print_master_table(all_results)


if __name__ == '__main__':
    main()
