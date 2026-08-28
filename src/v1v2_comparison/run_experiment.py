"""
run_experiment.py
M.Tech Thesis – Hyperspectral Pansharpening

Top-level runner script.  Delegates to sub-scripts via subprocess.

Subcommands
-----------
    train       Train both VMamba models (calls compare_vmamba.py)
    test        Evaluate on Chikusei test patches (calls test_chikusei.py)
    compare     Full training + comparison (calls compare_vmamba.py)
    visualize   Generate visualisation figures (calls visualize_results.py)

Examples
--------
    python run_experiment.py train
    python run_experiment.py train --epochs 50 --batch_size 4
    python run_experiment.py test  --checkpoint_old comparison/checkpoints/old_best.pth \\
                                   --checkpoint_improved comparison/checkpoints/improved_best.pth
    python run_experiment.py compare --epochs 30 --d_model 64
    python run_experiment.py visualize --patch_idx 3 \\
                                       --checkpoint_old comparison/checkpoints/old_best.pth \\
                                       --checkpoint_improved comparison/checkpoints/improved_best.pth
"""

import sys
import os
import subprocess
import argparse

_PROJECT = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# Script paths
# ============================================================================

SCRIPTS = {
    'train'    : os.path.join(_PROJECT, 'comparison',    'compare_vmamba.py'),
    'compare'  : os.path.join(_PROJECT, 'comparison',    'compare_vmamba.py'),
    'test'     : os.path.join(_PROJECT, 'testing',       'test_chikusei.py'),
    'visualize': os.path.join(_PROJECT, 'visualization', 'visualize_results.py'),
}


# ============================================================================
# Banner
# ============================================================================

BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║          Hyperspectral Pansharpening – M.Tech Thesis             ║
║                  VMamba Experiment Runner                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

INSTRUCTIONS = """
Available subcommands:
  train      Train both Old and Improved VMamba models
  test       Evaluate on Chikusei test patches and print metrics
  compare    Full training + comparison plots + summary JSON
  visualize  Generate visualisation figures for a test patch

Common flags (pass after the subcommand):
  --epochs N            Training epochs        (default 30)
  --batch_size N        Batch size             (default 2)
  --d_model N           Feature dimension      (default 64)
  --patch_idx N         Patch to visualise     (default 0)
  --checkpoint_old PATH      Path to Old VMamba checkpoint
  --checkpoint_improved PATH Path to Improved VMamba checkpoint

Dataset : C:\\Users\\s.saikumar\\Desktop\\Mtech project\\chikusei\\chikusei.mat

Examples:
  python run_experiment.py train
  python run_experiment.py train --epochs 50 --batch_size 4
  python run_experiment.py test
  python run_experiment.py test --checkpoint_old comparison/checkpoints/old_best.pth \\
                                --checkpoint_improved comparison/checkpoints/improved_best.pth
  python run_experiment.py compare --epochs 30
  python run_experiment.py visualize --patch_idx 5
"""


# ============================================================================
# Run a sub-script
# ============================================================================

def run_script(script_path: str, extra_args: list):
    """Launch script_path with sys.executable, passing extra_args through."""
    cmd = [sys.executable, script_path] + extra_args
    print(f"\n[RUN]  {' '.join(cmd)}\n{'─'*60}")
    result = subprocess.run(cmd, cwd=_PROJECT)
    if result.returncode != 0:
        print(f"\n[ERROR] Script exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n{'─'*60}")
    print("[DONE]  Script completed successfully.")


# ============================================================================
# Argument parsing  (first arg = subcommand, rest forwarded to sub-script)
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Hyperspectral Pansharpening experiment runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=INSTRUCTIONS,
    )
    parser.add_argument(
        'subcommand',
        choices=['train', 'test', 'compare', 'visualize'],
        help='Action to perform'
    )
    # All other arguments are forwarded verbatim
    parser.add_argument('extra', nargs=argparse.REMAINDER,
                        help='Additional arguments forwarded to the sub-script')
    return parser


# ============================================================================
# Main
# ============================================================================

def main():
    print(BANNER)

    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(INSTRUCTIONS)
        sys.exit(0)

    parser = build_parser()
    args   = parser.parse_args()
    sub    = args.subcommand
    extra  = args.extra

    # Strip leading '--' separator that argparse sometimes inserts
    if extra and extra[0] == '--':
        extra = extra[1:]

    script = SCRIPTS.get(sub)
    if script is None or not os.path.isfile(script):
        print(f"[ERROR] Cannot find script for subcommand '{sub}'.")
        print(f"  Expected path: {script}")
        sys.exit(1)

    print(f"Subcommand : {sub}")
    print(f"Script     : {script}")
    if extra:
        print(f"Extra args : {extra}")

    run_script(script, extra)


if __name__ == '__main__':
    main()
