"""
SAFE MODEL COMPARISON CONFIGURATION
Optimized for Tesla T4 GPU with minimal memory usage

Key fixes:
- 16x16 patches (instead of 64x64) to prevent OOM errors
- 8x8 overlap for validation/testing
- Batch size 1 for safety
- Scale 2 (easier on memory than scale 4)
"""

config_comparison = {
    # Dataset
    'dataset': 'pavia',
    'dataset_path': 'pavia/PaviaU.mat',

    # CRITICAL: Patch configuration (prevents OOM)
    'patch_size': 16,       # 16x16 HR patches (8x8 LR with scale=2)
    'overlap': 8,           # 8x8 overlap for val/test
    'scale': 2,             # Scale factor (2 is memory-safe)

    # Memory-safe training parameters
    'batch_size': 1,        # Must be 1 for safety
    'd_model': 32,          # Reduced model dimension
    'num_blocks': [2, 3, 3, 2],  # Reduced blocks

    # Training
    'epochs': 20,           # Reduced for quick comparison
    'lr': 1e-4,
    'weight_decay': 1e-2,
    'grad_clip': 1.0,

    # Loss weights
    'lambda_l1': 1.0,
    'lambda_sam': 0.1,
    'lambda_edge': 0.05,
    'lambda_ssim': 0.1,

    # System
    'num_workers': 0,       # 0 for Windows compatibility
}

# Validation/Test configuration (different patch settings)
config_val_test = {
    **config_comparison,
    'patch_size': 16,       # 16x16 for validation
    'overlap': 8,           # 8x8 overlap
    'batch_size': 1,        # Batch size 1 for val/test
}

# Alternative: Even smaller for extreme memory constraints
config_minimal = {
    **config_comparison,
    'patch_size': 8,        # 8x8 HR patches (4x4 LR)
    'overlap': 4,           # 4x4 overlap
    'd_model': 16,          # Minimal model size
    'num_blocks': [1, 2, 2, 1],
}

if __name__ == "__main__":
    print("=" * 80)
    print("SAFE MODEL COMPARISON CONFIGURATION")
    print("=" * 80)
    print(f"Dataset: {config_comparison['dataset']}")
    print(f"Patch size: {config_comparison['patch_size']}x{config_comparison['patch_size']}")
    print(f"Overlap: {config_comparison['overlap']} pixels")
    print(f"Scale: {config_comparison['scale']}x")
    print(f"Batch size: {config_comparison['batch_size']}")
    print(f"Model: d_model={config_comparison['d_model']}, blocks={config_comparison['num_blocks']}")
    print("")
    print("MEMORY-SAFE FEATURES:")
    print("  [OK] Small 16x16 patches")
    print("  [OK] Batch size 1")
    print("  [OK] Reduced model dimensions")
    print("  [OK] One model at a time training")
    print("")
    print("Usage:")
    print("  python compare_models.py --dataset pavia --epochs 20")
    print("=" * 80)
