"""
Visualization Utilities for Model Comparison
Creates publication-quality plots and visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import cv2


# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


class ComparisonVisualizer:
    """Create comprehensive comparison visualizations"""

    def __init__(self, results_dict, output_dir='visualizations'):
        """
        Args:
            results_dict: Dictionary with model results
                {
                    'model_name': {
                        'train_losses': [...],
                        'val_losses': [...],
                        'val_psnr': [...],
                        'val_sam': [...],
                        'val_ergas': [...],
                        'params': int,
                        'train_times': [...]
                    }
                }
            output_dir: Directory to save visualizations
        """
        self.results = results_dict
        self.output_dir = output_dir
        self.colors = {
            'CNN': '#FF6B6B',
            'Transformer': '#4ECDC4',
            'U-Net': '#45B7D1',
            'VMamba': '#96CEB4'
        }

    def plot_training_curves(self, save_path=None):
        """Plot training and validation loss curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Training loss
        for name, data in self.results.items():
            axes[0].plot(data['train_losses'], label=name,
                        linewidth=2.5, color=self.colors.get(name, '#000000'))

        axes[0].set_xlabel('Epoch', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Training Loss', fontsize=14, fontweight='bold')
        axes[0].set_title('Training Loss Comparison', fontsize=16, fontweight='bold')
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Validation loss
        for name, data in self.results.items():
            axes[1].plot(data['val_losses'], label=name,
                        linewidth=2.5, color=self.colors.get(name, '#000000'))

        axes[1].set_xlabel('Epoch', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Validation Loss', fontsize=14, fontweight='bold')
        axes[1].set_title('Validation Loss Comparison', fontsize=16, fontweight='bold')
        axes[1].legend(fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_metrics_evolution(self, save_path=None):
        """Plot evolution of all metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        metrics = [
            ('val_psnr', 'PSNR (dB)', 'Higher is Better', 0),
            ('val_sam', 'SAM (degrees)', 'Lower is Better', 1),
            ('val_ergas', 'ERGAS', 'Lower is Better', 2)
        ]

        for metric_key, ylabel, subtitle, idx in metrics:
            for name, data in self.results.items():
                axes[idx].plot(data[metric_key], label=name,
                             linewidth=2.5, color=self.colors.get(name, '#000000'))

            axes[idx].set_xlabel('Epoch', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel(ylabel, fontsize=14, fontweight='bold')
            axes[idx].set_title(f'{ylabel} - {subtitle}', fontsize=14, fontweight='bold')
            axes[idx].legend(fontsize=12)
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_final_comparison(self, save_path=None):
        """Bar chart comparison of final metrics"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        model_names = list(self.results.keys())
        x = np.arange(len(model_names))
        width = 0.6

        # 1. PSNR
        ax1 = fig.add_subplot(gs[0, 0])
        psnr_vals = [self.results[name]['val_psnr'][-1] for name in model_names]
        bars = ax1.bar(x, psnr_vals, width,
                      color=[self.colors.get(name, '#000000') for name in model_names],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('PSNR (dB)', fontsize=13, fontweight='bold')
        ax1.set_title('Final PSNR Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=0, fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        # Add values on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

        # 2. SAM
        ax2 = fig.add_subplot(gs[0, 1])
        sam_vals = [self.results[name]['val_sam'][-1] for name in model_names]
        bars = ax2.bar(x, sam_vals, width,
                      color=[self.colors.get(name, '#000000') for name in model_names],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('SAM (degrees)', fontsize=13, fontweight='bold')
        ax2.set_title('Final SAM Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=0, fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

        # 3. Parameters
        ax3 = fig.add_subplot(gs[1, 0])
        params_vals = [self.results[name]['params'] / 1e6 for name in model_names]
        bars = ax3.bar(x, params_vals, width,
                      color=[self.colors.get(name, '#000000') for name in model_names],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Parameters (Millions)', fontsize=13, fontweight='bold')
        ax3.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names, rotation=0, fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}M', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

        # 4. Training Speed
        ax4 = fig.add_subplot(gs[1, 1])
        time_vals = [np.mean(self.results[name]['train_times']) for name in model_names]
        bars = ax4.bar(x, time_vals, width,
                      color=[self.colors.get(name, '#000000') for name in model_names],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Time per Epoch (s)', fontsize=13, fontweight='bold')
        ax4.set_title('Training Speed Comparison', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_names, rotation=0, fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}s', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

        plt.suptitle('Model Performance Summary', fontsize=18, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_performance_efficiency(self, save_path=None):
        """Scatter plot of performance vs efficiency"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        model_names = list(self.results.keys())

        # PSNR vs Parameters
        for name in model_names:
            psnr = self.results[name]['val_psnr'][-1]
            params = self.results[name]['params'] / 1e6
            axes[0].scatter(params, psnr, s=300, alpha=0.7,
                          color=self.colors.get(name, '#000000'),
                          edgecolors='black', linewidth=2, label=name)

        axes[0].set_xlabel('Parameters (Millions)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Final PSNR (dB)', fontsize=14, fontweight='bold')
        axes[0].set_title('Performance vs Model Size', fontsize=16, fontweight='bold')
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # PSNR vs Training Time
        for name in model_names:
            psnr = self.results[name]['val_psnr'][-1]
            time = np.mean(self.results[name]['train_times'])
            axes[1].scatter(time, psnr, s=300, alpha=0.7,
                          color=self.colors.get(name, '#000000'),
                          edgecolors='black', linewidth=2, label=name)

        axes[1].set_xlabel('Avg. Training Time (s/epoch)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Final PSNR (dB)', fontsize=14, fontweight='bold')
        axes[1].set_title('Performance vs Training Speed', fontsize=16, fontweight='bold')
        axes[1].legend(fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def create_summary_table(self, save_path=None):
        """Create a summary table figure"""
        model_names = list(self.results.keys())

        # Prepare data
        data = []
        for name in model_names:
            row = [
                name,
                f"{self.results[name]['params']/1e6:.1f}M",
                f"{self.results[name]['val_psnr'][-1]:.2f}",
                f"{max(self.results[name]['val_psnr']):.2f}",
                f"{self.results[name]['val_sam'][-1]:.2f}",
                f"{min(self.results[name]['val_sam']):.2f}",
                f"{self.results[name]['val_ergas'][-1]:.4f}",
                f"{np.mean(self.results[name]['train_times']):.1f}s"
            ]
            data.append(row)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.axis('tight')
        ax.axis('off')

        columns = ['Model', 'Params', 'Final PSNR', 'Best PSNR',
                  'Final SAM', 'Best SAM', 'Final ERGAS', 'Avg. Time']

        table = ax.table(cellText=data, colLabels=columns, cellLoc='center',
                        loc='center', bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)

        # Color header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color rows alternately
        for i in range(1, len(data) + 1):
            if i % 2 == 0:
                for j in range(len(columns)):
                    table[(i, j)].set_facecolor('#F0F0F0')

        plt.title('Model Comparison Summary', fontsize=16, fontweight='bold', pad=20)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


def plot_visual_comparison(lr_hsi, hr_pan, predictions, gt_hsi, model_names,
                          rgb_bands=[55, 41, 12], save_path=None):
    """
    Create visual comparison of model outputs

    Args:
        lr_hsi: (C, H, W) - Low resolution HSI
        hr_pan: (1, rH, rW) - High resolution PAN
        predictions: dict of {model_name: (C, rH, rW)} - Model predictions
        gt_hsi: (C, rH, rW) - Ground truth
        model_names: List of model names
        rgb_bands: Indices for RGB visualization
        save_path: Path to save figure
    """
    def normalize_rgb(img, bands):
        """Extract and normalize RGB bands"""
        rgb = img[bands, :, :].transpose(1, 2, 0)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        return np.clip(rgb, 0, 1)

    num_models = len(model_names)
    fig, axes = plt.subplots(2, num_models + 2, figsize=(4 * (num_models + 2), 8))

    # Row 1: Inputs and predictions
    # LR-HSI (upsampled)
    lr_rgb = normalize_rgb(lr_hsi, rgb_bands)
    scale = gt_hsi.shape[1] // lr_hsi.shape[1]
    lr_rgb_up = cv2.resize(lr_rgb, (gt_hsi.shape[2], gt_hsi.shape[1]),
                          interpolation=cv2.INTER_CUBIC)
    axes[0, 0].imshow(lr_rgb_up)
    axes[0, 0].set_title('LR-HSI\n(Bicubic)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # HR-PAN
    pan = hr_pan[0]
    pan = (pan - pan.min()) / (pan.max() - pan.min() + 1e-8)
    axes[0, 1].imshow(pan, cmap='gray')
    axes[0, 1].set_title('HR-PAN', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Model predictions
    for idx, name in enumerate(model_names):
        pred_rgb = normalize_rgb(predictions[name], rgb_bands)
        axes[0, idx + 2].imshow(pred_rgb)
        axes[0, idx + 2].set_title(f'{name}\nPrediction', fontsize=12, fontweight='bold')
        axes[0, idx + 2].axis('off')

    # Row 2: Ground truth and error maps
    # Ground truth
    gt_rgb = normalize_rgb(gt_hsi, rgb_bands)
    axes[1, 0].imshow(gt_rgb)
    axes[1, 0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Placeholder for consistency
    axes[1, 1].axis('off')

    # Error maps
    for idx, name in enumerate(model_names):
        error_map = np.mean(np.abs(predictions[name] - gt_hsi), axis=0)
        im = axes[1, idx + 2].imshow(error_map, cmap='hot', vmin=0, vmax=error_map.max())
        axes[1, idx + 2].set_title(f'{name}\nError Map', fontsize=12, fontweight='bold')
        axes[1, idx + 2].axis('off')
        plt.colorbar(im, ax=axes[1, idx + 2], fraction=0.046)

    plt.suptitle('Visual Comparison of Model Outputs', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


if __name__ == '__main__':
    # Test visualizations with dummy data
    print("Testing visualization utilities...")

    # Create dummy results
    epochs = 50
    results = {
        'CNN': {
            'train_losses': np.linspace(0.5, 0.1, epochs) + np.random.normal(0, 0.01, epochs),
            'val_losses': np.linspace(0.6, 0.15, epochs) + np.random.normal(0, 0.01, epochs),
            'val_psnr': np.linspace(25, 32, epochs) + np.random.normal(0, 0.5, epochs),
            'val_sam': np.linspace(10, 5, epochs) + np.random.normal(0, 0.2, epochs),
            'val_ergas': np.linspace(5, 2, epochs) + np.random.normal(0, 0.1, epochs),
            'params': 15e6,
            'train_times': [45 + np.random.normal(0, 2) for _ in range(epochs)]
        },
        'Transformer': {
            'train_losses': np.linspace(0.5, 0.08, epochs) + np.random.normal(0, 0.01, epochs),
            'val_losses': np.linspace(0.6, 0.12, epochs) + np.random.normal(0, 0.01, epochs),
            'val_psnr': np.linspace(25, 34, epochs) + np.random.normal(0, 0.5, epochs),
            'val_sam': np.linspace(10, 4.5, epochs) + np.random.normal(0, 0.2, epochs),
            'val_ergas': np.linspace(5, 1.8, epochs) + np.random.normal(0, 0.1, epochs),
            'params': 25e6,
            'train_times': [65 + np.random.normal(0, 3) for _ in range(epochs)]
        },
        'U-Net': {
            'train_losses': np.linspace(0.5, 0.12, epochs) + np.random.normal(0, 0.01, epochs),
            'val_losses': np.linspace(0.6, 0.18, epochs) + np.random.normal(0, 0.01, epochs),
            'val_psnr': np.linspace(25, 30, epochs) + np.random.normal(0, 0.5, epochs),
            'val_sam': np.linspace(10, 5.5, epochs) + np.random.normal(0, 0.2, epochs),
            'val_ergas': np.linspace(5, 2.2, epochs) + np.random.normal(0, 0.1, epochs),
            'params': 8e6,
            'train_times': [35 + np.random.normal(0, 2) for _ in range(epochs)]
        },
        'VMamba': {
            'train_losses': np.linspace(0.5, 0.07, epochs) + np.random.normal(0, 0.01, epochs),
            'val_losses': np.linspace(0.6, 0.10, epochs) + np.random.normal(0, 0.01, epochs),
            'val_psnr': np.linspace(25, 35, epochs) + np.random.normal(0, 0.5, epochs),
            'val_sam': np.linspace(10, 4.0, epochs) + np.random.normal(0, 0.2, epochs),
            'val_ergas': np.linspace(5, 1.5, epochs) + np.random.normal(0, 0.1, epochs),
            'params': 18e6,
            'train_times': [55 + np.random.normal(0, 3) for _ in range(epochs)]
        }
    }

    # Create visualizer
    viz = ComparisonVisualizer(results)

    # Generate plots
    print("Generating training curves...")
    viz.plot_training_curves(save_path='test_training_curves.png')

    print("Generating metrics evolution...")
    viz.plot_metrics_evolution(save_path='test_metrics_evolution.png')

    print("Generating final comparison...")
    viz.plot_final_comparison(save_path='test_final_comparison.png')

    print("Generating performance-efficiency plot...")
    viz.plot_performance_efficiency(save_path='test_performance_efficiency.png')

    print("Generating summary table...")
    viz.create_summary_table(save_path='test_summary_table.png')

    print("\n✓ All visualizations generated successfully!")
