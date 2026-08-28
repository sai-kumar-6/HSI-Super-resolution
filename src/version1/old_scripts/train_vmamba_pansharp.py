"""
Training Script for VMamba-Pansharp
Implements training strategy from the paper:
- AdamW optimizer with cosine annealing and warmup
- Composite loss function
- Gradient clipping
- Checkpointing and logging
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from vmamba_pansharp import VMambaPansharp
from loss_functions import CompositeLoss, compute_psnr, compute_sam_metric, compute_ergas
from dataset_loader_overlap import create_dataloaders_overlap


class CosineAnnealingWarmup:
    """
    Cosine annealing learning rate scheduler with warm-up
    η_t = η_0 * t / T_warm for t < T_warm
    η_t = η_min + 0.5 * (η_0 - η_min) * (1 + cos(π * (t - T_warm) / (T_max - T_warm)))
    """
    def __init__(self, optimizer, eta_0, eta_min, T_warm, T_max):
        self.optimizer = optimizer
        self.eta_0 = eta_0
        self.eta_min = eta_min
        self.T_warm = T_warm
        self.T_max = T_max
        self.current_epoch = 0

    def step(self):
        """Update learning rate"""
        if self.current_epoch < self.T_warm:
            # Warmup phase
            lr = self.eta_0 * self.current_epoch / self.T_warm
        else:
            # Cosine annealing phase
            progress = (self.current_epoch - self.T_warm) / (self.T_max - self.T_warm)
            lr = self.eta_min + 0.5 * (self.eta_0 - self.eta_min) * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1
        return lr

    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


class Trainer:
    """Training manager for VMamba-Pansharp"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directory
        self.exp_dir = os.path.join('experiments', config['exp_name'])
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, 'checkpoints'), exist_ok=True)

        # Save config
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        # Initialize tensorboard
        self.writer = SummaryWriter(os.path.join(self.exp_dir, 'logs'))

        # Create dataloaders
        print(f"\nLoading {config['dataset']} dataset...")

        # Get dataset path
        if config['dataset'] == 'pavia':
            mat_path = 'pavia/PaviaU.mat'
        elif config['dataset'] == 'chikusei':
            mat_path = 'chikusei/chikusei.mat'
        else:
            raise ValueError(f"Unknown dataset: {config['dataset']}")

        # Use overlapping patches with spatial separation
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders_overlap(
            dataset=config['dataset'],
            mat_path=mat_path,
            batch_size=config['batch_size'],
            patch_size=config['patch_size'],
            overlap=config.get('overlap', config['patch_size'] // 2),  # Default: 50% overlap
            scale=config['scale'],
            num_workers=config['num_workers']
        )

        # Check dataset sizes
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")

        if len(self.val_loader) == 0:
            print("\n" + "="*70)
            print("[WARNING] Validation set is EMPTY!")
            print("="*70)
            print("Possible causes:")
            print("  1. Dataset is too small for the patch size")
            print("  2. Train/val split leaves too little data for validation")
            print("")
            print("Solutions:")
            print("  - Use smaller patch_size (e.g., 8 or 16)")
            print("  - Increase train_ratio in dataset_loader.py")
            print("  - Use a larger dataset")
            print("="*70 + "\n")

        # Get number of channels from first batch
        sample_batch = next(iter(self.train_loader))
        num_channels = sample_batch['lr_hsi'].shape[1]
        print(f"Number of spectral bands: {num_channels}")

        # Create model
        print("\nCreating VMamba-Pansharp model...")
        self.model = VMambaPansharp(
            in_channels=num_channels,
            out_channels=num_channels,
            d_model=config['d_model'],
            scale=config['scale'],
            num_blocks=config['num_blocks']
        ).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Create loss function
        self.criterion = CompositeLoss(
            lambda_l1=config['lambda_l1'],
            lambda_sam=config['lambda_sam'],
            lambda_edge=config['lambda_edge'],
            lambda_ssim=config['lambda_ssim']
        )

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            betas=(config['beta1'], config['beta2'])
        )

        # Create learning rate scheduler
        self.scheduler = CosineAnnealingWarmup(
            self.optimizer,
            eta_0=config['lr'],
            eta_min=config['lr_min'],
            T_warm=config['warmup_epochs'],
            T_max=config['epochs']
        )

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_psnr = 0.0

        # Load checkpoint if resuming
        if config.get('resume'):
            self.load_checkpoint(config['resume'])

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_components = {'l1': 0.0, 'sam': 0.0, 'edge': 0.0, 'ssim': 0.0}

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["epochs"]}')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            lr_hsi = batch['lr_hsi'].to(self.device)
            hr_pan = batch['hr_pan'].to(self.device)
            hr_hsi = batch['hr_hsi'].to(self.device)

            # Forward pass
            pred_hsi = self.model(lr_hsi, hr_pan)

            # Compute loss
            loss, loss_dict = self.criterion(pred_hsi, hr_hsi, return_components=True)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

            self.optimizer.step()

            # Accumulate losses
            epoch_loss += loss_dict['total']
            for key in epoch_components:
                epoch_components[key] += loss_dict[key]

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}", 'lr': f"{self.scheduler.get_lr():.6f}"})

        # Average losses
        num_batches = len(self.train_loader)
        epoch_loss /= num_batches
        for key in epoch_components:
            epoch_components[key] /= num_batches

        return epoch_loss, epoch_components

    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_components = {'l1': 0.0, 'sam': 0.0, 'edge': 0.0, 'ssim': 0.0}
        val_psnr = 0.0
        val_sam = 0.0
        val_ergas = 0.0

        for batch in tqdm(self.val_loader, desc='Validation'):
            lr_hsi = batch['lr_hsi'].to(self.device)
            hr_pan = batch['hr_pan'].to(self.device)
            hr_hsi = batch['hr_hsi'].to(self.device)

            # Forward pass
            pred_hsi = self.model(lr_hsi, hr_pan)

            # Compute loss
            loss, loss_dict = self.criterion(pred_hsi, hr_hsi, return_components=True)

            # Accumulate losses
            val_loss += loss_dict['total']
            for key in val_components:
                val_components[key] += loss_dict[key]

            # Compute metrics
            val_psnr += compute_psnr(pred_hsi, hr_hsi)
            val_sam += compute_sam_metric(pred_hsi, hr_hsi)
            val_ergas += compute_ergas(pred_hsi, hr_hsi, scale=self.config['scale'])

        # Average
        num_batches = len(self.val_loader)

        # Handle empty validation set
        if num_batches == 0:
            print("[WARNING] Validation loader is empty! Returning zeros.")
            return 0.0, {k: 0.0 for k in val_components}, 0.0, 0.0, 0.0

        val_loss /= num_batches
        for key in val_components:
            val_components[key] /= num_batches
        val_psnr /= num_batches
        val_sam /= num_batches
        val_ergas /= num_batches

        return val_loss, val_components, val_psnr, val_sam, val_ergas

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_epoch': self.scheduler.current_epoch,
            'best_val_loss': self.best_val_loss,
            'best_psnr': self.best_psnr,
            'config': self.config
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.exp_dir, 'checkpoints', 'latest.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.exp_dir, 'checkpoints', 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint at epoch {epoch}")

        # Save periodic checkpoint
        if epoch % self.config['save_freq'] == 0:
            periodic_path = os.path.join(self.exp_dir, 'checkpoints', f'epoch_{epoch}.pth')
            torch.save(checkpoint, periodic_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.current_epoch = checkpoint['scheduler_epoch']
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_psnr = checkpoint['best_psnr']

        print(f"Resumed from epoch {checkpoint['epoch']}")

    def train(self):
        """Main training loop"""
        print(f"\nStarting training on {self.device}")
        print(f"Experiment: {self.config['exp_name']}")
        print(f"Total epochs: {self.config['epochs']}")
        print(f"Warmup epochs: {self.config['warmup_epochs']}\n")

        for epoch in range(self.start_epoch, self.config['epochs']):
            # Update learning rate
            current_lr = self.scheduler.step()

            # Train
            train_loss, train_components = self.train_epoch(epoch)

            # Validate
            val_loss, val_components, val_psnr, val_sam, val_ergas = self.validate(epoch)

            # Log to tensorboard
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Metrics/PSNR', val_psnr, epoch)
            self.writer.add_scalar('Metrics/SAM', val_sam, epoch)
            self.writer.add_scalar('Metrics/ERGAS', val_ergas, epoch)

            for key in train_components:
                self.writer.add_scalar(f'Train/{key}', train_components[key], epoch)
                self.writer.add_scalar(f'Val/{key}', val_components[key], epoch)

            # Print summary
            print(f"\nEpoch {epoch}/{self.config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  PSNR: {val_psnr:.2f} dB | SAM: {val_sam:.2f}° | ERGAS: {val_ergas:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Save checkpoint
            is_best = val_psnr > self.best_psnr
            if is_best:
                self.best_psnr = val_psnr
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, is_best)

        print("\nTraining completed!")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train VMamba-Pansharp')
    parser.add_argument('--dataset', type=str, default='pavia', choices=['pavia', 'chikusei'],
                       help='Dataset to use')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (default: 1 for memory safety)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='HR patch size (default: 16 for memory safety)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers (0 for Windows)')

    args = parser.parse_args()

    # Create config
    config = {
        # Experiment
        'exp_name': args.exp_name or f"vmamba_pansharp_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'dataset': args.dataset,

        # Data
        'patch_size': args.patch_size,
        'overlap': args.patch_size // 2,  # 50% overlap by default
        'scale': 2,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,

        # Model
        'd_model': 64,
        'num_blocks': [3, 4, 4, 3],

        # Loss weights (from paper)
        'lambda_l1': 1.0,
        'lambda_sam': 0.1,
        'lambda_edge': 0.05,
        'lambda_ssim': 0.1,

        # Optimizer
        'lr': args.lr,
        'lr_min': 1e-6,
        'weight_decay': 1e-2,
        'beta1': 0.9,
        'beta2': 0.999,

        # Training
        'epochs': args.epochs,
        'warmup_epochs': 10,
        'grad_clip': 1.0,
        'save_freq': 50,

        # Resume
        'resume': args.resume
    }

    # Validate scale (must be 2 for overlapping patches dataset)
    assert config['scale'] in [2, 4], f"Scale must be 2 or 4, got {config['scale']}"
    if config['scale'] != 2:
        print(f"\n{'='*70}")
        print(f"WARNING: Using scale={config['scale']}, but overlapping dataset is optimized for scale=2")
        print(f"{'='*70}\n")

    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
