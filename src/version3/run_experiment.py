"""
run_experiment.py  —  V3 Edition
Main runner for the Version 3 experiments.

Subcommands
───────────
  verify     Quick model sanity check (no data)
  test       Evaluate all 3 models on test patches
  compare    Train all 3 models and compare
  visualize  Generate RGB + error map + spectral plots

Usage
─────
  cd version3

  # 0. Verify models
  python run_experiment.py verify

  # 1. Quick test (random weights, no training)
  python run_experiment.py test

  # 2. Train and compare (OOM-safe CPU defaults)
  python run_experiment.py compare --epochs 5 --batch_size 1 --patch_size 32 --d_model 32

  # 3. Full training
  python run_experiment.py compare --epochs 30 --batch_size 1 --patch_size 32 --d_model 32

  # 4. Test with trained checkpoints
  python run_experiment.py test \
    --checkpoint_old      comparison/checkpoints/old_best.pth \
    --checkpoint_improved comparison/checkpoints/improved_best.pth \
    --checkpoint_v3       comparison/checkpoints/v3_best.pth

  # 5. Visualize
  python run_experiment.py visualize \
    --checkpoint_old      comparison/checkpoints/old_best.pth \
    --checkpoint_improved comparison/checkpoints/improved_best.pth \
    --checkpoint_v3       comparison/checkpoints/v3_best.pth
"""

import sys
import os
import subprocess
import argparse

_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, 'scripts'))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'model'))


def run(cmd):
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=_HERE)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description='V3 VMamba experiment runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command')

    # ── verify ────────────────────────────────────────────────────────────────
    sub.add_parser('verify', help='Quick model sanity check (no data)')

    # ── test ──────────────────────────────────────────────────────────────────
    p_test = sub.add_parser('test', help='Evaluate on Chikusei test patches')
    p_test.add_argument('--checkpoint_old',      default=None)
    p_test.add_argument('--checkpoint_improved', default=None)
    p_test.add_argument('--checkpoint_v3',       default=None)
    p_test.add_argument('--max_patches', default=20, type=int)

    # ── compare ───────────────────────────────────────────────────────────────
    p_cmp = sub.add_parser('compare', help='Train and compare all 3 variants')
    p_cmp.add_argument('--epochs',     default=5,  type=int)
    p_cmp.add_argument('--batch_size', default=1,  type=int)
    p_cmp.add_argument('--d_model',    default=32, type=int)
    p_cmp.add_argument('--d_state',    default=8,  type=int)
    p_cmp.add_argument('--patch_size', default=32, type=int)
    p_cmp.add_argument('--skip_old',      action='store_true')
    p_cmp.add_argument('--skip_improved', action='store_true')
    p_cmp.add_argument('--skip_v3',       action='store_true')

    # ── visualize ─────────────────────────────────────────────────────────────
    p_vis = sub.add_parser('visualize', help='Generate output images')
    p_vis.add_argument('--checkpoint_old',      default=None)
    p_vis.add_argument('--checkpoint_improved', default=None)
    p_vis.add_argument('--checkpoint_v3',       default=None)
    p_vis.add_argument('--patch_idx', default=0, type=int)

    args = parser.parse_args()

    if args.command == 'verify':
        import torch
        from baselines import create_vmamba_model, count_parameters
        print("\nVerifying all three models …")
        print("=" * 55)
        for v in ['old', 'improved', 'v3']:
            m = create_vmamba_model(v, 128, 128, scale=4, d_model=32,
                                    d_state=4, num_blocks=[1,1,1,1]).eval()
            lr  = torch.randn(1, 128, 8, 8)
            pan = torch.randn(1, 1, 32, 32)
            with torch.no_grad():
                out = m(lr, pan)
            total, _ = count_parameters(m)
            status = 'OK' if out.shape == (1, 128, 32, 32) else 'FAIL'
            print(f"  {v:<10}  params={total:>10,}  output={tuple(out.shape)}  [{status}]")
        print("=" * 55)

    elif args.command == 'test':
        cmd = [sys.executable, 'testing/test_chikusei.py',
               '--max_patches', str(args.max_patches)]
        for flag, val in [('--checkpoint_old',      args.checkpoint_old),
                           ('--checkpoint_improved', args.checkpoint_improved),
                           ('--checkpoint_v3',       args.checkpoint_v3)]:
            if val:
                cmd += [flag, val]
        run(cmd)

    elif args.command == 'compare':
        cmd = [sys.executable, 'comparison/compare_all.py',
               '--epochs',     str(args.epochs),
               '--batch_size', str(args.batch_size),
               '--d_model',    str(args.d_model),
               '--d_state',    str(args.d_state),
               '--patch_size', str(args.patch_size)]
        if args.skip_old:      cmd.append('--skip_old')
        if args.skip_improved: cmd.append('--skip_improved')
        if args.skip_v3:       cmd.append('--skip_v3')
        run(cmd)

    elif args.command == 'visualize':
        cmd = [sys.executable, 'visualization/visualize_results.py',
               '--patch_idx', str(args.patch_idx)]
        for flag, val in [('--checkpoint_old',      args.checkpoint_old),
                           ('--checkpoint_improved', args.checkpoint_improved),
                           ('--checkpoint_v3',       args.checkpoint_v3)]:
            if val:
                cmd += [flag, val]
        run(cmd)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
