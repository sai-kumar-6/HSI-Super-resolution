"""
run_experiment.py  —  V5 Edition
Main runner for Version 5 experiments (True Parallel Scan).

Subcommands
───────────
  verify     Quick model sanity check for all 5 variants (no data)
  test       Evaluate all 5 models on test patches
  compare    Train all 5 models and compare (saves images + Excel)
  visualize  Generate RGB + error map + spectral plots

Usage
─────
  cd version5

  # 0. Verify all 5 models load and produce correct output shapes
  python run_experiment.py verify

  # 1. Quick smoke test (random weights, no training)
  python run_experiment.py test

  # 2. Train and compare (OOM-safe CPU defaults, 5 epochs)
  python run_experiment.py compare --epochs 5 --batch_size 1 --patch_size 32

  # 3. Full training (30 epochs)
  python run_experiment.py compare --epochs 30 --batch_size 1 --patch_size 32

  # 4. Train only V5 (skip older variants)
  python run_experiment.py compare --skip_old --skip_improved --skip_v3 --skip_v4 --epochs 30

  # 5. Test with trained checkpoints
  python run_experiment.py test \\
      --checkpoint_old      comparison/checkpoints/old_best.pth \\
      --checkpoint_improved comparison/checkpoints/improved_best.pth \\
      --checkpoint_v3       comparison/checkpoints/v3_best.pth \\
      --checkpoint_v4       comparison/checkpoints/v4_best.pth \\
      --checkpoint_v5       comparison/checkpoints/v5_best.pth

  # 6. Visualize results
  python run_experiment.py visualize \\
      --checkpoint_v4 comparison/checkpoints/v4_best.pth \\
      --checkpoint_v5 comparison/checkpoints/v5_best.pth
"""

import sys
import os
import argparse

_HERE    = os.path.dirname(os.path.abspath(__file__))   # version5/
_PROJECT = os.path.dirname(_HERE)

sys.path.insert(0, os.path.join(_PROJECT, 'version3'))  # lowest priority
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, 'scripts'))
sys.path.insert(0, os.path.join(_PROJECT, 'version4'))
sys.path.insert(0, _HERE)                               # highest priority
sys.path.insert(0, os.path.join(_HERE, 'model'))

import subprocess


def run(cmd):
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=_HERE)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description='V5 VMamba experiment runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command')

    # ── verify ────────────────────────────────────────────────────────────────
    sub.add_parser('verify', help='Quick model sanity check (no data)')

    # ── test ──────────────────────────────────────────────────────────────────
    p_test = sub.add_parser('test', help='Evaluate all 5 models on test patches')
    p_test.add_argument('--checkpoint_old',      default=None)
    p_test.add_argument('--checkpoint_improved', default=None)
    p_test.add_argument('--checkpoint_v3',       default=None)
    p_test.add_argument('--checkpoint_v4',       default=None)
    p_test.add_argument('--checkpoint_v5',       default=None)
    p_test.add_argument('--max_patches', default=20, type=int)

    # ── compare ───────────────────────────────────────────────────────────────
    p_cmp = sub.add_parser('compare', help='Train and compare all 5 variants')
    p_cmp.add_argument('--epochs',        default=5,  type=int)
    p_cmp.add_argument('--batch_size',    default=1,  type=int)
    p_cmp.add_argument('--d_model',       default=32, type=int)
    p_cmp.add_argument('--d_state',       default=8,  type=int)
    p_cmp.add_argument('--patch_size',    default=32, type=int)
    p_cmp.add_argument('--skip_old',      action='store_true')
    p_cmp.add_argument('--skip_improved', action='store_true')
    p_cmp.add_argument('--skip_v3',       action='store_true')
    p_cmp.add_argument('--skip_v4',       action='store_true')
    p_cmp.add_argument('--skip_v5',       action='store_true')
    p_cmp.add_argument('--save_img_every', default=5, type=int)

    # ── visualize ─────────────────────────────────────────────────────────────
    p_vis = sub.add_parser('visualize', help='Generate output images and spectral plots')
    p_vis.add_argument('--checkpoint_old',      default=None)
    p_vis.add_argument('--checkpoint_improved', default=None)
    p_vis.add_argument('--checkpoint_v3',       default=None)
    p_vis.add_argument('--checkpoint_v4',       default=None)
    p_vis.add_argument('--checkpoint_v5',       default=None)
    p_vis.add_argument('--patch_idx', default=0, type=int)
    p_vis.add_argument('--pixel_y',   default=16, type=int)
    p_vis.add_argument('--pixel_x',   default=16, type=int)

    args = parser.parse_args()

    if args.command == 'verify':
        import torch
        from baselines import create_vmamba_model, count_parameters
        print("\nVerifying all five models …")
        print("=" * 66)
        for v in ['old', 'improved', 'v3', 'v4', 'v5']:
            m = create_vmamba_model(
                v, 128, 128, scale=4, d_model=32,
                d_state=4, num_blocks=[1, 1, 1, 1],
            ).eval()
            lr  = torch.randn(1, 128, 8, 8)
            pan = torch.randn(1, 1, 32, 32)
            with torch.no_grad():
                out = m(lr, pan)
            total, _ = count_parameters(m)
            ok = 'OK' if out.shape == (1, 128, 32, 32) else 'FAIL'
            print(f"  {v:<12}  params={total:>10,}  "
                  f"output={tuple(out.shape)}  [{ok}]")

            if v == 'v4':
                print(f"             alpha={m.sdim.alpha.item():.4f}  "
                      f"beta={m.recon.beta.item():.4f}")
            if v == 'v5':
                print(f"             alpha={m.sdim.alpha.item():.4f}  "
                      f"beta={m.recon.beta.item():.4f}  "
                      f"(parallel scan: log2(32)=5 iters)")
        print("=" * 66)

    elif args.command == 'test':
        cmd = [sys.executable, 'testing/test_chikusei.py',
               '--max_patches', str(args.max_patches)]
        for flag, val in [
            ('--checkpoint_old',      args.checkpoint_old),
            ('--checkpoint_improved', args.checkpoint_improved),
            ('--checkpoint_v3',       args.checkpoint_v3),
            ('--checkpoint_v4',       args.checkpoint_v4),
            ('--checkpoint_v5',       args.checkpoint_v5),
        ]:
            if val:
                cmd += [flag, val]
        run(cmd)

    elif args.command == 'compare':
        cmd = [sys.executable, 'comparison/compare_all.py',
               '--epochs',     str(args.epochs),
               '--batch_size', str(args.batch_size),
               '--d_model',    str(args.d_model),
               '--d_state',    str(args.d_state),
               '--patch_size', str(args.patch_size),
               '--save_img_every', str(args.save_img_every)]
        if args.skip_old:      cmd.append('--skip_old')
        if args.skip_improved: cmd.append('--skip_improved')
        if args.skip_v3:       cmd.append('--skip_v3')
        if args.skip_v4:       cmd.append('--skip_v4')
        if args.skip_v5:       cmd.append('--skip_v5')
        run(cmd)

    elif args.command == 'visualize':
        cmd = [sys.executable, 'visualization/visualize_results.py',
               '--patch_idx', str(args.patch_idx),
               '--pixel_y',   str(args.pixel_y),
               '--pixel_x',   str(args.pixel_x)]
        for flag, val in [
            ('--checkpoint_old',      args.checkpoint_old),
            ('--checkpoint_improved', args.checkpoint_improved),
            ('--checkpoint_v3',       args.checkpoint_v3),
            ('--checkpoint_v4',       args.checkpoint_v4),
            ('--checkpoint_v5',       args.checkpoint_v5),
        ]:
            if val:
                cmd += [flag, val]
        run(cmd)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
