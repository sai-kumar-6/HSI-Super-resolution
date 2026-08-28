"""
Training Configuration for Overlapping Patches (Scale=2)
Ready-to-use configuration with memory optimizations
"""

import argparse

def get_config():
    """Get training configuration"""
    parser = argparse.ArgumentParser(description='Train VMamba-Pansharp with overlapping patches')

    # Dataset
    parser.add_argument('--dataset', type=str, default='pavia', choices=['pavia', 'chikusei'],
                        help='Dataset to use')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='HR patch size (default: 16 for memory safety)')
    parser.add_argument('--overlap', type=int, default=8,
                        help='Overlap between patches (default: 8 for val/test)')
    parser.add_argument('--scale', type=int, default=2,
                        help='Super-resolution scale factor')

    # Model architecture (memory-safe defaults)
    parser.add_argument('--d_model', type=int, default=32,
                        help='Model dimension')
    parser.add_argument('--d_state', type=int, default=8,
                        help='State dimension')
    parser.add_argument('--num_blocks', type=int, nargs=4, default=[2, 3, 3, 2],
                        help='Number of blocks per stage')

    # Training
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (default: 1 for memory safety)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping')

    # Loss weights
    parser.add_argument('--lambda_l1', type=float, default=1.0,
                        help='L1 loss weight')
    parser.add_argument('--lambda_sam', type=float, default=0.1,
                        help='SAM loss weight')
    parser.add_argument('--lambda_edge', type=float, default=0.5,
                        help='Edge loss weight')
    parser.add_argument('--lambda_ssim', type=float, default=0.3,
                        help='SSIM loss weight')

    # System
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Logging
    parser.add_argument('--exp_name', type=str, default='vmamba_overlap_scale2',
                        help='Experiment name')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint frequency')
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Logging frequency')
    parser.add_argument('--val_freq', type=int, default=5,
                        help='Validation frequency')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Convert to dict
    config = vars(args)

    return config


# Pre-defined configurations

# Memory-safe config (recommended for Tesla T4 / 16GB GPU)
PAVIA_CONFIG = {
    'dataset': 'pavia',
    'patch_size': 16,           # 16x16 HR patches (memory-safe)
    'overlap': 8,               # 8x8 overlap for val/test
    'scale': 2,
    'd_model': 32,
    'd_state': 8,
    'num_blocks': [2, 3, 3, 2],
    'batch_size': 1,            # Batch size 1 for safety
    'epochs': 20,
    'lr': 1e-4,
    'lr_min': 1e-6,
    'warmup_epochs': 5,
    'weight_decay': 0.01,
    'beta1': 0.9,
    'beta2': 0.999,
    'grad_clip': 1.0,
    'lambda_l1': 1.0,
    'lambda_sam': 0.1,
    'lambda_edge': 0.5,
    'lambda_ssim': 0.3,
    'num_workers': 0,
    'seed': 42,
    'exp_name': 'pavia_overlap_scale2',
    'save_freq': 10,
    'log_freq': 10,
    'val_freq': 5,
    'resume': None
}

CHIKUSEI_CONFIG = {
    **PAVIA_CONFIG,
    'dataset': 'chikusei',
    'exp_name': 'chikusei_overlap_scale2'
}

# Minimal config for extreme memory constraints
MINIMAL_CONFIG = {
    **PAVIA_CONFIG,
    'batch_size': 1,
    'd_model': 16,
    'd_state': 4,
    'num_blocks': [1, 2, 2, 1],
    'patch_size': 8,            # 8x8 HR patches
    'overlap': 4,               # 4x4 overlap
    'exp_name': 'minimal_training_scale2'
}


if __name__ == '__main__':
    config = get_config()
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 80)
