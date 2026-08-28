"""
run_experiment.py  —  Version 2
Main entry point for all V2 experiments.

Subcommands
───────────
  verify          Sanity check: model shapes for scale 2 and 4
  train           Train for one scale
  compare_scales  Train scale 4, then scale 2, produce comparison table
  visualize       Generate test visuals from saved checkpoints

Usage
─────
  cd version2

  # 0. Verify models load correctly
  python run_experiment.py verify

  # 1. Train scale 4 (30 epochs)
  python run_experiment.py train --scale 4 --epochs 30

  # 2. Train scale 2 (30 epochs)
  python run_experiment.py train --scale 2 --epochs 30

  # 3. Train both scales and produce comparison table (convenient one-liner)
  python run_experiment.py compare_scales --epochs 30

  # 4. Quick smoke test (5 epochs each)
  python run_experiment.py compare_scales --epochs 5

  # 5. Visualize with saved checkpoints
  python run_experiment.py visualize \
      --checkpoint_s4 runs/scale4/checkpoints/best.pth \
      --checkpoint_s2 runs/scale2/checkpoints/best.pth
"""
from __future__ import annotations
import os
import sys
import argparse
import json
import subprocess
import time

_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)


def run_cmd(cmd: list):
    print(f"\n$ {' '.join(cmd)}\n")
    r = subprocess.run(cmd, cwd=_HERE)
    if r.returncode != 0:
        sys.exit(r.returncode)


def load_results(run_dir: str) -> dict:
    cfg_path = os.path.join(run_dir, 'run_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def print_comparison_table(r4: dict, r2: dict):
    """Print a rich comparison table for scale 4 vs scale 2."""
    sep = '=' * 72
    print(f"\n{sep}")
    print(f"  V2 VMamba Pansharpening — Scale Comparison Table")
    print(sep)
    print(f"  {'Property':<32} {'Scale 4':>16} {'Scale 2':>16}")
    print(f"  {'-'*32} {'-'*16} {'-'*16}")

    def row(label, k4, k2, fmt='{:.3f}'):
        v4 = k4 if isinstance(k4, str) else (fmt.format(k4) if k4 else 'N/A')
        v2 = k2 if isinstance(k2, str) else (fmt.format(k2) if k2 else 'N/A')
        print(f"  {label:<32} {v4:>16} {v2:>16}")

    # Hyperparameters
    h4 = r4.get('hparams', {})
    h2 = r2.get('hparams', {})
    row('Scale factor',        4,                          2,                    '{:d}')
    row('Epochs',              h4.get('epochs','—'),       h2.get('epochs','—'), '{}')
    row('Batch size',          h4.get('batch_size','—'),   h2.get('batch_size','—'), '{}')
    row('Patch size (HR)',      h4.get('patch_size','—'),   h2.get('patch_size','—'), '{}')
    row('d_model',             h4.get('d_model','—'),      h2.get('d_model','—'), '{}')
    row('d_state',             h4.get('d_state','—'),      h2.get('d_state','—'), '{}')
    row('Num blocks/stage',    h4.get('num_blocks','—'),   h2.get('num_blocks','—'), '{}')

    # Parameters
    row('Params (total)',      f"{r4.get('params_total',0):,}",
                               f"{r2.get('params_total',0):,}", '{}')
    row('Params (trainable)',  f"{r4.get('params_train',0):,}",
                               f"{r2.get('params_train',0):,}", '{}')

    # FLOPs
    fl4 = r4.get('flops',{})
    fl2 = r2.get('flops',{})
    macs4 = fl4.get('macs', fl4.get('macs_estimate', 0))
    macs2 = fl2.get('macs', fl2.get('macs_estimate', 0))
    row('MACs (G)',            f"{macs4/1e9:.2f}G",        f"{macs2/1e9:.2f}G", '{}')

    # Inference
    inf4 = r4.get('inference_timing', {}).get('inference_ms_per_sample', 0)
    inf2 = r2.get('inference_timing', {}).get('inference_ms_per_sample', 0)
    row('Inference (ms/patch)',f"{inf4:.2f}",               f"{inf2:.2f}", '{}')

    # Test metrics
    print(f"\n  {'—'*66}")
    print(f"  {'TEST METRICS':^66}")
    print(f"  {'—'*66}")
    tr4 = r4.get('test_results', {}).get('scale4', {})
    tr2 = r2.get('test_results', {}).get('scale2', {})
    row('PSNR (dB ↑)',         tr4.get('psnr',0),           tr2.get('psnr',0))
    row('SAM (° ↓)',           tr4.get('sam',0),            tr2.get('sam',0))
    row('ERGAS (↓)',           tr4.get('ergas',0),          tr2.get('ergas',0))
    row('Inference ms/patch',  tr4.get('inference_ms',0),   tr2.get('inference_ms',0))

    print(f"\n  {'—'*66}")
    print(f"  Train patches: S4={h4.get('train_patches','?')}  S2={h2.get('train_patches','?')}")
    print(f"  Val   patches: S4={h4.get('val_patches','?')}    S2={h2.get('val_patches','?')}")
    print(f"  Test  patches: S4={h4.get('test_patches','?')}   S2={h2.get('test_patches','?')}")
    print(sep)

    # Save table to file
    table_path = os.path.join(_HERE, 'runs', 'scale_comparison.txt')
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    with open(table_path, 'w') as f:
        f.write("V2 VMamba — Scale 4 vs Scale 2 Comparison\n")
        f.write("=" * 72 + "\n")
        f.write(f"{'Property':<32} {'Scale 4':>16} {'Scale 2':>16}\n")
        for label, v4, v2 in [
            ('PSNR (dB ↑)',   tr4.get('psnr',0),   tr2.get('psnr',0)),
            ('SAM (° ↓)',     tr4.get('sam',0),    tr2.get('sam',0)),
            ('ERGAS (↓)',     tr4.get('ergas',0),  tr2.get('ergas',0)),
        ]:
            f.write(f"  {label:<32} {v4:>16.3f} {v2:>16.3f}\n")
    print(f"\n  Comparison table saved → {table_path}")


def cmd_verify():
    import torch
    from model import V2VMambaPansharp, count_parameters
    print("\nVerifying V2 model for scale 2 and 4 …")
    print("=" * 62)
    for scale in [4, 2]:
        m = V2VMambaPansharp(128, 128, d_model=32, d_state=8,
                             scale=scale, num_blocks=[1,1,1,1]).eval()
        lr  = torch.randn(1, 128, 8, 8)
        pan = torch.randn(1, 1, 8*scale, 8*scale)
        with torch.no_grad():
            out = m(lr, pan)
        total, _ = count_parameters(m)
        ok = 'OK' if out.shape == (1, 128, 8*scale, 8*scale) else 'FAIL'
        print(f"  scale={scale}  params={total:>10,}  output={tuple(out.shape)}  [{ok}]")
    print("=" * 62)


def cmd_visualize(args):
    import torch
    import torch.nn.functional as F
    from model        import V2VMambaPansharp
    from dataset_loader import create_dataloaders
    from loss_functions import compute_psnr, compute_sam, compute_ergas
    from logger       import ExperimentLogger

    CHIKUSEI = os.path.join(_PROJECT, 'chikusei', 'chikusei.mat')
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vis_dir = os.path.join(_HERE, 'runs', 'visualizations')
    logger  = ExperimentLogger(vis_dir)

    for scale, ckp_path in [(4, args.checkpoint_s4), (2, args.checkpoint_s2)]:
        if ckp_path is None or not os.path.isfile(ckp_path):
            print(f"  [Skip] scale={scale}: no checkpoint")
            continue

        st = torch.load(ckp_path, map_location=device)
        hp = st.get('hparams', {})
        m  = V2VMambaPansharp(
            128, 128,
            d_model   = hp.get('d_model', 32),
            d_state   = hp.get('d_state', 8),
            scale     = scale,
            num_blocks= [hp.get('num_blocks', 1)]*4,
        ).to(device)
        m.load_state_dict(st['model_state_dict'])
        m.eval()

        _, _, test_dl = create_dataloaders(
            CHIKUSEI, batch_size=1,
            patch_size=hp.get('patch_size', 32),
            overlap=hp.get('overlap', 16),
            scale=scale, max_rows=256,
        )
        b    = next(iter(test_dl))
        lr   = b['lr_hsi'].to(device)
        pan  = b['hr_pan'].to(device)
        hr   = b['hr_hsi']
        with torch.no_grad():
            pred = m(lr, pan).cpu()

        metrics = dict(
            psnr  = compute_psnr(pred, hr),
            sam   = compute_sam(pred, hr),
            ergas = compute_ergas(pred, hr, scale=scale),
        )
        logger.save_test_visuals(pred, hr, lr.cpu(), pan.cpu(),
                                 metrics, tag=f'scale{scale}')
        print(f"  scale={scale}  PSNR={metrics['psnr']:.2f}  "
              f"SAM={metrics['sam']:.3f}  ERGAS={metrics['ergas']:.3f}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command')

    sub.add_parser('verify')

    p_tr = sub.add_parser('train')
    p_tr.add_argument('--scale',       type=int,   default=4)
    p_tr.add_argument('--epochs',      type=int,   default=30)
    p_tr.add_argument('--batch_size',  type=int,   default=2)
    p_tr.add_argument('--patch_size',  type=int,   default=32)
    p_tr.add_argument('--overlap',     type=int,   default=16)
    p_tr.add_argument('--d_model',     type=int,   default=32)
    p_tr.add_argument('--d_state',     type=int,   default=8)
    p_tr.add_argument('--num_blocks',  type=int,   default=1)
    p_tr.add_argument('--lr',          type=float, default=1e-3)
    p_tr.add_argument('--max_rows',    type=int,   default=512)
    p_tr.add_argument('--save_img_every', type=int, default=5)
    p_tr.add_argument('--num_workers', type=int,   default=0)

    p_cmp = sub.add_parser('compare_scales')
    p_cmp.add_argument('--epochs',     type=int,   default=30)
    p_cmp.add_argument('--batch_size', type=int,   default=2)
    p_cmp.add_argument('--patch_size', type=int,   default=32)
    p_cmp.add_argument('--overlap',    type=int,   default=16)
    p_cmp.add_argument('--d_model',    type=int,   default=32)
    p_cmp.add_argument('--d_state',    type=int,   default=8)
    p_cmp.add_argument('--num_blocks', type=int,   default=1)
    p_cmp.add_argument('--max_rows',   type=int,   default=512)
    p_cmp.add_argument('--save_img_every', type=int, default=5)

    p_vis = sub.add_parser('visualize')
    p_vis.add_argument('--checkpoint_s4', default=None)
    p_vis.add_argument('--checkpoint_s2', default=None)

    args = p.parse_args()

    if args.command == 'verify':
        cmd_verify()

    elif args.command == 'train':
        run_cmd([sys.executable, 'train.py',
                 '--scale',       str(args.scale),
                 '--epochs',      str(args.epochs),
                 '--batch_size',  str(args.batch_size),
                 '--patch_size',  str(args.patch_size),
                 '--overlap',     str(args.overlap),
                 '--d_model',     str(args.d_model),
                 '--d_state',     str(args.d_state),
                 '--num_blocks',  str(args.num_blocks),
                 '--lr',          str(1e-3),
                 '--max_rows',    str(args.max_rows),
                 '--save_img_every', str(args.save_img_every),
                 '--num_workers', str(0)])

    elif args.command == 'compare_scales':
        common = [
            '--epochs',      str(args.epochs),
            '--batch_size',  str(args.batch_size),
            '--patch_size',  str(args.patch_size),
            '--overlap',     str(args.overlap),
            '--d_model',     str(args.d_model),
            '--d_state',     str(args.d_state),
            '--num_blocks',  str(args.num_blocks),
            '--max_rows',    str(args.max_rows),
            '--save_img_every', str(args.save_img_every),
            '--num_workers', '0',
        ]
        print("\n>>> Training scale=4 <<<")
        run_cmd([sys.executable, 'train.py', '--scale', '4'] + common)
        print("\n>>> Training scale=2 <<<")
        run_cmd([sys.executable, 'train.py', '--scale', '2'] + common)

        r4 = load_results(os.path.join(_HERE, 'runs', 'scale4'))
        r2 = load_results(os.path.join(_HERE, 'runs', 'scale2'))
        print_comparison_table(r4, r2)

    elif args.command == 'visualize':
        cmd_visualize(args)

    else:
        p.print_help()


if __name__ == '__main__':
    main()
