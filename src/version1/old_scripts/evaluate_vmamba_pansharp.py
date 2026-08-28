"""
Evaluation and Testing Script for VMamba-Pansharp
Evaluates trained model and generates visualizations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from tqdm import tqdm
import scipy.io as sio

from vmamba_pansharp import VMambaPansharp
from loss_functions import compute_psnr, compute_sam_metric, compute_ergas
from dataset_loader_overlap import create_dataloaders_overlap


class Evaluator:
    """Evaluation manager for VMamba-Pansharp"""

    def __init__(self, checkpoint_path, output_dir='results'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']

        # Get number of channels
        if self.config['dataset'] == 'pavia':
            num_channels = 103  # Pavia has 103 bands
        elif self.config['dataset'] == 'chikusei':
            num_channels = 128  # Chikusei has 128 bands
        else:
            num_channels = 102  # Default

        # Create model
        print("Creating model...")
        self.model = VMambaPansharp(
            in_channels=num_channels,
            out_channels=num_channels,
            d_model=self.config['d_model'],
            scale=self.config['scale'],
            num_blocks=self.config['num_blocks']
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Loaded model from epoch {checkpoint['epoch']}")
        print(f"Best PSNR: {checkpoint.get('best_psnr', 'N/A'):.2f} dB")

        # Create dataloader
        print(f"\nLoading {self.config['dataset']} dataset...")

        # Get dataset path
        if self.config['dataset'] == 'pavia':
            mat_path = 'pavia/PaviaU.mat'
        elif self.config['dataset'] == 'chikusei':
            mat_path = 'chikusei/chikusei.mat'
        else:
            raise ValueError(f"Unknown dataset: {self.config['dataset']}")

        _, _, self.test_loader = create_dataloaders_overlap(
            dataset=self.config['dataset'],
            mat_path=mat_path,
            batch_size=1,  # Evaluate one image at a time
            patch_size=self.config['patch_size'],
            overlap=self.config.get('overlap', self.config['patch_size'] // 2),
            scale=self.config['scale'],
            num_workers=0
        )

    @torch.no_grad()
    def evaluate(self):
        """Evaluate model on test set"""
        print("\nEvaluating model...")

        metrics = {
            'psnr': [],
            'sam': [],
            'ergas': []
        }

        for batch_idx, batch in enumerate(tqdm(self.test_loader)):
            lr_hsi = batch['lr_hsi'].to(self.device)
            hr_pan = batch['hr_pan'].to(self.device)
            hr_hsi = batch['hr_hsi'].to(self.device)

            # Forward pass
            pred_hsi = self.model(lr_hsi, hr_pan)

            # Compute metrics
            psnr = compute_psnr(pred_hsi, hr_hsi)
            sam = compute_sam_metric(pred_hsi, hr_hsi)
            ergas = compute_ergas(pred_hsi, hr_hsi, scale=self.config['scale'])

            metrics['psnr'].append(psnr)
            metrics['sam'].append(sam)
            metrics['ergas'].append(ergas)

            # Save first few results for visualization
            if batch_idx < 5:
                self.save_visualization(
                    lr_hsi[0], hr_pan[0], pred_hsi[0], hr_hsi[0],
                    batch_idx, psnr, sam, ergas
                )

        # Compute statistics
        results = {
            'psnr_mean': np.mean(metrics['psnr']),
            'psnr_std': np.std(metrics['psnr']),
            'sam_mean': np.mean(metrics['sam']),
            'sam_std': np.std(metrics['sam']),
            'ergas_mean': np.mean(metrics['ergas']),
            'ergas_std': np.std(metrics['ergas']),
            'num_samples': len(metrics['psnr'])
        }

        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Dataset: {self.config['dataset']}")
        print(f"Number of samples: {results['num_samples']}")
        print(f"\nPSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
        print(f"SAM:  {results['sam_mean']:.2f} ± {results['sam_std']:.2f}°")
        print(f"ERGAS: {results['ergas_mean']:.4f} ± {results['ergas_std']:.4f}")
        print("="*60)

        # Save results
        results_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {results_path}")

        return results

    def save_visualization(self, lr_hsi, hr_pan, pred_hsi, gt_hsi, idx, psnr, sam, ergas):
        """
        Save visualization comparing LR, prediction, and ground truth
        """
        # Move to CPU and convert to numpy
        lr_hsi = lr_hsi.cpu().numpy()
        hr_pan = hr_pan.cpu().numpy()
        pred_hsi = pred_hsi.cpu().numpy()
        gt_hsi = gt_hsi.cpu().numpy()

        # Select RGB bands for visualization (approximate)
        # Pavia: bands [55, 41, 12] ~ [R, G, B]
        # Chikusei: bands [60, 40, 20] ~ [R, G, B]
        if self.config['dataset'] == 'pavia':
            rgb_bands = [55, 41, 12] if lr_hsi.shape[0] > 55 else [min(55, lr_hsi.shape[0]-1),
                                                                     min(41, lr_hsi.shape[0]-1),
                                                                     min(12, lr_hsi.shape[0]-1)]
        else:
            rgb_bands = [60, 40, 20] if lr_hsi.shape[0] > 60 else [min(60, lr_hsi.shape[0]-1),
                                                                     min(40, lr_hsi.shape[0]-1),
                                                                     min(20, lr_hsi.shape[0]-1)]

        def normalize_rgb(img, bands):
            """Extract and normalize RGB bands"""
            rgb = img[bands, :, :].transpose(1, 2, 0)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            return np.clip(rgb, 0, 1)

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Row 1: LR upsampled, PAN, Prediction
        lr_rgb = normalize_rgb(lr_hsi, rgb_bands)
        pred_rgb = normalize_rgb(pred_hsi, rgb_bands)
        gt_rgb = normalize_rgb(gt_hsi, rgb_bands)
        pan = hr_pan[0]
        pan = (pan - pan.min()) / (pan.max() - pan.min() + 1e-8)

        # Upsample LR for visualization
        import cv2
        lr_rgb_up = cv2.resize(lr_rgb, (pred_rgb.shape[1], pred_rgb.shape[0]),
                               interpolation=cv2.INTER_CUBIC)

        axes[0, 0].imshow(lr_rgb_up)
        axes[0, 0].set_title('LR-HSI (Upsampled)', fontsize=12)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(pan, cmap='gray')
        axes[0, 1].set_title('HR-PAN', fontsize=12)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(pred_rgb)
        axes[0, 2].set_title(f'Predicted HR-HSI\nPSNR: {psnr:.2f} dB', fontsize=12)
        axes[0, 2].axis('off')

        # Row 2: Ground Truth, Error Map, Spectral Signature
        axes[1, 0].imshow(gt_rgb)
        axes[1, 0].set_title('Ground Truth HR-HSI', fontsize=12)
        axes[1, 0].axis('off')

        # Error map (mean absolute error across bands)
        error_map = np.mean(np.abs(pred_hsi - gt_hsi), axis=0)
        im = axes[1, 1].imshow(error_map, cmap='hot')
        axes[1, 1].set_title(f'Error Map\nSAM: {sam:.2f}°, ERGAS: {ergas:.4f}', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

        # Spectral signature comparison (center pixel)
        h, w = pred_hsi.shape[1:]
        center_y, center_x = h // 2, w // 2
        pred_spectrum = pred_hsi[:, center_y, center_x]
        gt_spectrum = gt_hsi[:, center_y, center_x]

        axes[1, 2].plot(gt_spectrum, 'b-', label='Ground Truth', linewidth=2)
        axes[1, 2].plot(pred_spectrum, 'r--', label='Predicted', linewidth=2)
        axes[1, 2].set_xlabel('Band Index', fontsize=10)
        axes[1, 2].set_ylabel('Reflectance', fontsize=10)
        axes[1, 2].set_title('Spectral Signature (Center Pixel)', fontsize=12)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        save_path = os.path.join(self.output_dir, f'sample_{idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization to {save_path}")

    def export_results(self, output_path='vmamba_pansharp_results.mat'):
        """
        Export pansharpened results for the entire dataset
        """
        print(f"\nExporting results to {output_path}...")

        results = {
            'lr_hsi': [],
            'hr_pan': [],
            'pred_hsi': [],
            'gt_hsi': []
        }

        for batch in tqdm(self.test_loader, desc='Processing'):
            lr_hsi = batch['lr_hsi'].to(self.device)
            hr_pan = batch['hr_pan'].to(self.device)
            hr_hsi = batch['hr_hsi'].to(self.device)

            # Forward pass
            pred_hsi = self.model(lr_hsi, hr_pan)

            # Append to results
            results['lr_hsi'].append(lr_hsi[0].cpu().numpy())
            results['hr_pan'].append(hr_pan[0].cpu().numpy())
            results['pred_hsi'].append(pred_hsi[0].cpu().numpy())
            results['gt_hsi'].append(hr_hsi[0].cpu().numpy())

        # Convert lists to arrays
        for key in results:
            results[key] = np.array(results[key])

        # Save to .mat file
        sio.savemat(os.path.join(self.output_dir, output_path), results)
        print(f"Results exported to {os.path.join(self.output_dir, output_path)}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VMamba-Pansharp')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--export', action='store_true',
                       help='Export results to .mat file')

    args = parser.parse_args()

    # Create evaluator
    evaluator = Evaluator(args.checkpoint, args.output_dir)

    # Evaluate
    results = evaluator.evaluate()

    # Export if requested
    if args.export:
        evaluator.export_results()

    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
