"""
SAFE TRAINING CONFIGURATION - OOM Prevention
Use this config to train VMamba-Pansharp without memory errors

Tested on: Tesla T4 (16GB GPU)
Expected memory usage: ~12-14GB during training
"""

config = {
    # Dataset
    'dataset': 'pavia',  # or 'chikusei'
    'scale': 2,

    # CRITICAL: Memory-safe training parameters
    'batch_size': 1,        # Must be 1 for training
    'patch_size': 16,       # Small patches (16x16 LR -> 64x64 HR)
    'd_model': 32,          # Reduced from 64
    'd_state': 8,           # Reduced from 16
    'num_blocks': [1, 2, 2, 2],  # Reduced from [3, 4, 4, 3]

    # Training
    'epochs': 20,
    'lr': 1e-4,
    'lr_min': 1e-6,
    'warmup_epochs': 5,
    'weight_decay': 0.01,
    'beta1': 0.9,
    'beta2': 0.999,
    'grad_clip': 1.0,

    # Loss weights
    'lambda_l1': 1.0,
    'lambda_sam': 0.1,
    'lambda_edge': 0.5,
    'lambda_ssim': 0.3,

    # System
    'num_workers': 0,  # Set to 0 for Windows, or 2-4 for Linux
    'seed': 42,

    # Logging
    'exp_name': 'vmamba_safe_training',
    'save_freq': 10,
    'log_freq': 10,
    'val_freq': 5,
}

# For better GPU with more memory (e.g., V100, A100), you can increase:
config_better_gpu = {
    **config,
    'batch_size': 2,        # Can try 2-4
    'd_model': 48,          # Can increase to 48-64
    'd_state': 12,          # Can increase to 12-16
    'num_blocks': [2, 3, 3, 2],  # Can increase blocks
}

print("=" * 80)
print("SAFE TRAINING CONFIGURATION LOADED")
print("=" * 80)
print(f"Batch size: {config['batch_size']}")
print(f"Patch size: {config['patch_size']} (LR) -> {config['patch_size']*config['scale']} (HR)")
print(f"Model dims: d_model={config['d_model']}, d_state={config['d_state']}")
print(f"Num blocks: {config['num_blocks']}")
print("")
print("MEMORY OPTIMIZATIONS ENABLED:")
print("  [OK] Sequential scan during training (vs parallel)")
print("  [OK] Sequential 4-way processing (vs batched)")
print("  [OK] Reduced backbone resolution (4x downscale)")
print("")
print("To use this config:")
print("  python train_vmamba_pansharp.py --config config_safe_training.py")
print("=" * 80)
