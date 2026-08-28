"""
Model Comparison Script
Trains and compares VMamba-Pansharp with baseline models
Generates comprehensive comparison graphs and statistics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime
import time

from vmamba_pansharp import VMambaPansharp
from baseline_models import create_model, count_parameters
from loss_functions import CompositeLoss, compute_psnr, compute_sam_metric, compute_ergas
from dataset_loader_overlap import create_dataloaders_overlap


# ============================================================
# Model Comparator
# ============================================================

class ModelComparator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}

        self.exp_dir = os.path.join('comparison_results', config['exp_name'])
        os.makedirs(self.exp_dir, exist_ok=True)

        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        print(f"\nLoading dataset: {config['dataset']}")

        # Get dataset path
        if config['dataset'] == 'pavia':
            mat_path = 'pavia/PaviaU.mat'
        elif config['dataset'] == 'chikusei':
            mat_path = 'chikusei/chikusei.mat'
        else:
            raise ValueError(f"Unknown dataset: {config['dataset']}")

        self.train_loader, self.val_loader, _ = create_dataloaders_overlap(
            dataset=config['dataset'],
            mat_path=mat_path,
            batch_size=config['batch_size'],
            patch_size=config['patch_size'],
            overlap=config.get('overlap', config['patch_size'] // 2),
            scale=config['scale'],
            num_workers=config['num_workers']
        )

        sample = next(iter(self.train_loader))
        self.num_channels = sample['lr_hsi'].shape[1]

        self.criterion = CompositeLoss(
            lambda_l1=config['lambda_l1'],
            lambda_sam=config['lambda_sam'],
            lambda_edge=config['lambda_edge'],
            lambda_ssim=config['lambda_ssim']
        )

        self.model_configs = {
            'CNN': 'cnn',
            'Transformer': 'transformer',
            'U-Net': 'unet',
            'VMamba': 'vmamba'
        }

        for name in self.model_configs:
            self.models[name] = {
                'model': None,
                'optimizer': None,
                'scheduler': None,
                'params': 0,
                'train_losses': [],
                'val_losses': [],
                'val_psnr': [],
                'val_sam': [],
                'val_ergas': [],
                'train_times': []
            }

    # ========================================================
    # Model Loader
    # ========================================================

    def get_or_create_model(self, name):
        if self.models[name]['model'] is not None:
            return self.models[name]

        print(f"\nLoading model: {name}")
        model_type = self.model_configs[name]

        if model_type == 'vmamba':
            model = VMambaPansharp(
                in_channels=self.num_channels,
                out_channels=self.num_channels,
                d_model=self.config['d_model'],
                scale=self.config['scale'],
                num_blocks=self.config['num_blocks'],
                use_parallel_scan=False,
                use_parallel_4way=False
            ).to(self.device)
        else:
            model = create_model(
                model_type,
                self.num_channels,
                self.num_channels,
                scale=self.config['scale']
            ).to(self.device)

        total_params, trainable = count_parameters(model)

        lr_map = {
            'CNN': 1e-3,
            'Transformer': 5e-4,
            'U-Net': 1e-3,
            'VMamba': 1e-4
        }

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr_map[name],
            weight_decay=self.config['weight_decay']
        )

        # ✅ FIXED scheduler logic
        if name == 'VMamba':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config['epochs'], eta_min=1e-6
            )
        elif name == 'Transformer':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config['epochs'], eta_min=1e-5
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, self.config['epochs'] // 3),
                gamma=0.5
            )

        self.models[name].update({
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'params': total_params
        })

        print(f"  Parameters: {total_params:,}")
        return self.models[name]

    def unload_model(self, name):
        if self.models[name]['model'] is not None:
            del self.models[name]['model']
            del self.models[name]['optimizer']
            del self.models[name]['scheduler']
            self.models[name]['model'] = None
            torch.cuda.empty_cache()

    # ========================================================
    # Training / Validation
    # ========================================================

    def train_epoch(self, model_name, epoch):
        m = self.get_or_create_model(model_name)
        model, optimizer = m['model'], m['optimizer']
        model.train()

        total_loss = 0.0
        start = time.time()

        for batch in tqdm(self.train_loader, desc=f"{model_name} Epoch {epoch}", leave=False):
            lr_hsi = batch['lr_hsi'].to(self.device)
            hr_pan = batch['hr_pan'].to(self.device)
            hr_hsi = batch['hr_hsi'].to(self.device)

            pred = model(lr_hsi, hr_pan)
            loss = self.criterion(pred, hr_hsi)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip'])
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - start
        avg_loss = total_loss / len(self.train_loader)

        m['train_losses'].append(avg_loss)
        m['train_times'].append(epoch_time)

        return avg_loss, epoch_time

    @torch.no_grad()
    def validate(self, model_name):
        m = self.get_or_create_model(model_name)
        model = m['model']
        model.eval()

        loss_sum, psnr_sum, sam_sum, ergas_sum = 0, 0, 0, 0

        for batch in self.val_loader:
            lr_hsi = batch['lr_hsi'].to(self.device)
            hr_pan = batch['hr_pan'].to(self.device)
            hr_hsi = batch['hr_hsi'].to(self.device)

            pred = model(lr_hsi, hr_pan)

            loss_sum += self.criterion(pred, hr_hsi).item()
            psnr_sum += compute_psnr(pred, hr_hsi)
            sam_sum += compute_sam_metric(pred, hr_hsi)
            ergas_sum += compute_ergas(pred, hr_hsi, scale=self.config['scale'])

        n = len(self.val_loader)

        # Handle empty validation set
        if n == 0:
            print("[WARNING] Validation loader is empty! Returning zeros.")
            return 0.0, 0.0, 0.0, 0.0

        m['val_losses'].append(loss_sum / n)
        m['val_psnr'].append(psnr_sum / n)
        m['val_sam'].append(sam_sum / n)
        m['val_ergas'].append(ergas_sum / n)

        return m['val_losses'][-1], m['val_psnr'][-1], m['val_sam'][-1], m['val_ergas'][-1]

    # ========================================================
    # Main Comparison Loop
    # ========================================================

    def compare(self):
        for name in self.models:
            print(f"\n{'='*60}\nTRAINING {name}\n{'='*60}")
            for epoch in range(self.config['epochs']):
                tr_loss, tr_time = self.train_epoch(name, epoch)
                val_loss, psnr, sam, ergas = self.validate(name)

                self.models[name]['scheduler'].step()

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(
                        f"Epoch {epoch+1:3d} | "
                        f"Train {tr_loss:.4f} | Val {val_loss:.4f} | "
                        f"PSNR {psnr:.2f} | SAM {sam:.2f} | ERGAS {ergas:.3f}"
                    )

            self.unload_model(name)

        self.generate_summary()

    # ========================================================
    # Summary
    # ========================================================

    def generate_summary(self):
        summary = {}
        for name, m in self.models.items():
            summary[name] = {
                'params': m['params'],
                'best_psnr': max(m['val_psnr']),
                'best_sam': min(m['val_sam']),
                'best_ergas': min(m['val_ergas'])
            }

        with open(os.path.join(self.exp_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

        print("\nFINAL RESULTS")
        for k, v in summary.items():
            print(f"{k}: PSNR={v['best_psnr']:.2f}, SAM={v['best_sam']:.2f}, ERGAS={v['best_ergas']:.3f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='pavia', choices=['pavia', 'chikusei'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size for HR images (default: 16 for memory safety)')
    parser.add_argument('--overlap', type=int, default=8,
                       help='Overlap between patches (default: 8)')
    args = parser.parse_args()

    config = {
        'exp_name': f'comparison_{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': 1,           # MUST be 1 for memory safety
        'patch_size': args.patch_size,  # 16x16 patches (memory-safe)
        'overlap': args.overlap,        # 8x8 overlap for val/test
        'scale': 2,                # Scale 2 (memory-safe)
        'num_workers': 0,
        'd_model': 32,             # Reduced for memory
        'num_blocks': [2, 3, 3, 2],  # Reduced for memory
        'lambda_l1': 1.0,
        'lambda_sam': 0.1,
        'lambda_edge': 0.05,
        'lambda_ssim': 0.1,
        'weight_decay': 1e-2,
        'grad_clip': 1.0
    }

    print("\n" + "="*70)
    print("MODEL COMPARISON - MEMORY-SAFE CONFIGURATION")
    print("="*70)
    print(f"Dataset: {config['dataset']}")
    print(f"Patch size: {config['patch_size']}x{config['patch_size']} (HR)")
    print(f"Overlap: {config['overlap']} pixels")
    print(f"Scale: {config['scale']}x")
    print(f"LR patch size: {config['patch_size']//config['scale']}x{config['patch_size']//config['scale']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Model: d_model={config['d_model']}, blocks={config['num_blocks']}")
    print("="*70 + "\n")

    # Validate scale
    assert config['scale'] in [2, 4], f"Scale must be 2 or 4, got {config['scale']}"
    if config['scale'] != 2:
        print(f"WARNING: Scale={config['scale']}, but scale=2 is recommended for memory\n")

    comparator = ModelComparator(config)
    comparator.compare()


if __name__ == '__main__':
    main()
