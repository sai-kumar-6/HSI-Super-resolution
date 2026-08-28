"""
Quick Start Training Script with Overlapping Patches
Run this script directly - no command line arguments needed!
"""

import sys
sys.path.insert(0, '.')

from train_vmamba_pansharp import Trainer
from train_config_overlap import PAVIA_CONFIG, CHIKUSEI_CONFIG, SAFE_CONFIG

print("=" * 80)
print("VMAMBA-PANSHARP TRAINING - OVERLAPPING PATCHES (SCALE=2)")
print("=" * 80)
print("\nAvailable configurations:")
print("  1. PAVIA (Standard)     - batch_size=8, d_model=32, patch=64x64")
print("  2. CHIKUSEI (Standard)  - batch_size=8, d_model=32, patch=64x64")
print("  3. SAFE (16GB GPU)      - batch_size=1, d_model=24, patch=32x32")
print("")

# Choose configuration
choice = input("Select configuration (1/2/3) [default=1]: ").strip() or "1"

if choice == "1":
    config = PAVIA_CONFIG.copy()
    print("\n[SELECTED] Pavia University dataset")
elif choice == "2":
    config = CHIKUSEI_CONFIG.copy()
    print("\n[SELECTED] Chikusei dataset")
elif choice == "3":
    config = SAFE_CONFIG.copy()
    print("\n[SELECTED] Safe training for 16GB GPU")
else:
    print("[ERROR] Invalid choice, using Pavia")
    config = PAVIA_CONFIG.copy()

print("\nConfiguration:")
print(f"  Dataset: {config['dataset']}")
print(f"  Patch size: {config['patch_size']} (overlap={config['overlap']})")
print(f"  Scale: {config['scale']}")
print(f"  Batch size: {config['batch_size']}")
print(f"  Model: d_model={config['d_model']}, d_state={config['d_state']}")
print(f"  Blocks: {config['num_blocks']}")
print(f"  Epochs: {config['epochs']}")
print("")

# Confirm
proceed = input("Start training? (y/n) [default=y]: ").strip().lower() or "y"

if proceed != 'y':
    print("Training cancelled.")
    sys.exit(0)

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80 + "\n")

# Start training
trainer = Trainer(config)
trainer.train()

print("\n" + "=" * 80)
print("TRAINING COMPLETED!")
print("=" * 80)
print(f"\nCheckpoints saved in: experiments/{config['exp_name']}/checkpoints/")
print(f"Logs saved in: experiments/{config['exp_name']}/logs/")
print("\nTo evaluate the model:")
print(f"  python evaluate_vmamba_pansharp.py --checkpoint experiments/{config['exp_name']}/checkpoints/best.pth")
