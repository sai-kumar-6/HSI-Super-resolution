"""
Model Output Visualization Script
Visualizes VMamba-Pansharp predictions with detailed analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import os
import argparse

from vmamba_pansharp import VMambaPansharp
from dataset_loader_overlap import create_dataloaders_overlap
from loss_functions import compute_psnr, compute_sam_metric, compute_ergas


def visualize_model_predictions(model, val_loader, device, num_samples=5, save_dir='visualizations'):
    """
    Create comprehensive visualization of model predictions

    Args:
        model: Trained VMamba-Pansharp model
        val_loader: Validation data loader
        device: torch device
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    print(f"\nGenerating visualizations for {num_samples} samples...")

    with torch.no_grad():
        for sample_idx, batch in enumerate(tqdm(val_loader, desc="Processing samples", total=num_samples)):
            if sample_idx >= num_samples:
                break

            # Get data
            lr_hsi = batch['lr_hsi'].to(device)
            hr_pan = batch['hr_pan'].to(device)
            hr_hsi_gt = batch['hr_hsi'].to(device)

            # Forward pass
            hr_hsi_pred = model(lr_hsi, hr_pan)

            # Compute metrics
            # Dynamically compute scale from data (don't hard-code)
            scale = hr_hsi_gt.shape[-1] * 1.0 / lr_hsi.shape[-1]  # HR_width / LR_width
            if abs(scale - int(scale)) < 0.01:  # If close to integer
                scale = int(scale)
            psnr = compute_psnr(hr_hsi_pred, hr_hsi_gt)
            sam = compute_sam_metric(hr_hsi_pred, hr_hsi_gt)
            ergas = compute_ergas(hr_hsi_pred, hr_hsi_gt, scale=scale)

            # Convert to numpy for visualization
            lr_hsi_np = lr_hsi[0].cpu().numpy()
            hr_pan_np = hr_pan[0].cpu().numpy()
            hr_pred_np = hr_hsi_pred[0].cpu().numpy()
            hr_gt_np = hr_hsi_gt[0].cpu().numpy()

            # Create visualization
            visualize_single_prediction(
                lr_hsi_np, hr_pan_np, hr_pred_np, hr_gt_np,
                psnr, sam, ergas, scale, sample_idx, save_dir
            )

    print(f"\n✓ Saved {num_samples} visualizations to {save_dir}/")


def visualize_single_prediction(lr_hsi, hr_pan, hr_pred, hr_gt, psnr, sam, ergas, scale, sample_idx, save_dir):
    """Create detailed visualization for a single prediction"""

    C, H, W = hr_gt.shape

    # Select RGB-like bands
    if C >= 60:
        rgb_bands = [int(C*0.7), int(C*0.5), int(C*0.2)]
    else:
        rgb_bands = [min(50, C-1), min(30, C-1), min(10, C-1)]

    def normalize(img):
        """Normalize for display"""
        img = img.astype(np.float32)
        if len(img.shape) == 3 and img.shape[0] < 10:  # Channels first
            img = img.transpose(1, 2, 0)
        return np.clip((img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1)

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.25)

    # Row 1: Inputs
    # LR-HSI (upsampled for visualization)
    ax1 = fig.add_subplot(gs[0, 0])
    import cv2
    lr_rgb = normalize(lr_hsi[rgb_bands, :, :])
    lr_rgb_up = cv2.resize(lr_rgb, (W, H), interpolation=cv2.INTER_CUBIC)
    ax1.imshow(lr_rgb_up)
    ax1.set_title(f'Input: LR-HSI (Upsampled)\n{lr_hsi.shape}', fontweight='bold', fontsize=11)
    ax1.axis('off')

    # HR-PAN
    ax2 = fig.add_subplot(gs[0, 1])
    pan = normalize(hr_pan)
    ax2.imshow(pan, cmap='gray')
    ax2.set_title(f'Input: HR-PAN\n{hr_pan.shape}', fontweight='bold', fontsize=11)
    ax2.axis('off')

    # Prediction
    ax3 = fig.add_subplot(gs[0, 2])
    pred_rgb = normalize(hr_pred[rgb_bands, :, :])
    ax3.imshow(pred_rgb)
    ax3.set_title(f'VMamba Prediction\n{hr_pred.shape}\nPSNR: {psnr:.2f}dB',
                 fontweight='bold', fontsize=11)
    ax3.axis('off')

    # Ground Truth
    ax4 = fig.add_subplot(gs[0, 3])
    gt_rgb = normalize(hr_gt[rgb_bands, :, :])
    ax4.imshow(gt_rgb)
    ax4.set_title(f'Ground Truth\n{hr_gt.shape}', fontweight='bold', fontsize=11)
    ax4.axis('off')

    # Error Map
    ax5 = fig.add_subplot(gs[0, 4])
    error_rgb = np.abs(pred_rgb - gt_rgb)
    error_mag = np.mean(error_rgb, axis=2)
    im5 = ax5.imshow(error_mag, cmap='hot', vmin=0, vmax=error_mag.max())
    ax5.set_title(f'RGB Error Map\nMean Error: {error_mag.mean():.4f}',
                 fontweight='bold', fontsize=11)
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    # Row 2: Single Band Comparisons
    band_idx = rgb_bands[1]  # Green band

    ax6 = fig.add_subplot(gs[1, 0])
    lr_band = normalize(lr_hsi[band_idx:band_idx+1, :, :])
    lr_band_up = cv2.resize(lr_band, (W, H), interpolation=cv2.INTER_CUBIC)
    ax6.imshow(lr_band_up, cmap='viridis')
    ax6.set_title(f'LR Band {band_idx} (Up)', fontweight='bold', fontsize=11)
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 1])
    ax7.imshow(normalize(hr_pan), cmap='gray')
    ax7.set_title('HR-PAN Detail', fontweight='bold', fontsize=11)
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 2])
    im8 = ax8.imshow(normalize(hr_pred[band_idx:band_idx+1, :, :]), cmap='viridis')
    ax8.set_title(f'Pred Band {band_idx}', fontweight='bold', fontsize=11)
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046)

    ax9 = fig.add_subplot(gs[1, 3])
    im9 = ax9.imshow(normalize(hr_gt[band_idx:band_idx+1, :, :]), cmap='viridis')
    ax9.set_title(f'GT Band {band_idx}', fontweight='bold', fontsize=11)
    ax9.axis('off')
    plt.colorbar(im9, ax=ax9, fraction=0.046)

    ax10 = fig.add_subplot(gs[1, 4])
    band_error = np.abs(hr_pred[band_idx, :, :] - hr_gt[band_idx, :, :])
    im10 = ax10.imshow(band_error, cmap='hot')
    ax10.set_title(f'Band {band_idx} Error', fontweight='bold', fontsize=11)
    ax10.axis('off')
    plt.colorbar(im10, ax=ax10, fraction=0.046)

    # Row 3: Analysis
    # Spectral signatures (multiple pixels)
    ax11 = fig.add_subplot(gs[2, :2])
    num_pixels = 5
    for i in range(num_pixels):
        y, x = H//2 + i*10, W//2 + i*10
        if y < H and x < W:
            ax11.plot(hr_gt[:, y, x], alpha=0.7, linewidth=2, label=f'GT Pixel {i}')
            ax11.plot(hr_pred[:, y, x], '--', alpha=0.7, linewidth=2, label=f'Pred Pixel {i}')
    ax11.set_xlabel('Band Index', fontweight='bold')
    ax11.set_ylabel('Reflectance', fontweight='bold')
    ax11.set_title('Spectral Signatures Comparison', fontweight='bold', fontsize=11)
    ax11.legend(ncol=2, fontsize=8)
    ax11.grid(True, alpha=0.3)

    # Per-band metrics
    ax12 = fig.add_subplot(gs[2, 2:4])
    band_errors = np.mean(np.abs(hr_pred - hr_gt), axis=(1, 2))
    ax12.bar(range(C), band_errors, color='steelblue', alpha=0.7, edgecolor='black')
    ax12.set_xlabel('Band Index', fontweight='bold')
    ax12.set_ylabel('Mean Absolute Error', fontweight='bold')
    ax12.set_title('Per-Band Reconstruction Error', fontweight='bold', fontsize=11)
    ax12.grid(True, alpha=0.3, axis='y')

    # Metrics summary
    ax13 = fig.add_subplot(gs[2, 4])
    ax13.axis('off')
    metrics_text = f"""
    Quality Metrics:

    PSNR: {psnr:.2f} dB
    {'Excellent' if psnr > 35 else 'Good' if psnr > 30 else 'Fair'}

    SAM: {sam:.2f}°
    {'Excellent' if sam < 3 else 'Good' if sam < 5 else 'Fair'}

    ERGAS: {ergas:.4f}
    {'Excellent' if ergas < 2 else 'Good' if ergas < 4 else 'Fair'}

    Shape Info:
    Input LR: {lr_hsi.shape}
    Input PAN: {hr_pan.shape}
    Output: {hr_pred.shape}
    Scale: {scale}×
    """
    ax13.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax13.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'VMamba-Pansharp Output Analysis - Sample {sample_idx}',
                fontsize=16, fontweight='bold')

    # Save
    save_path = os.path.join(save_dir, f'model_output_sample_{sample_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def compare_with_baselines(checkpoint_paths, val_loader, device, save_dir='visualizations'):
    """Compare VMamba with baseline models"""
    from baseline_models import CNNPansharp, TransformerPansharp, UNetPansharp

    os.makedirs(save_dir, exist_ok=True)
    print("\nComparing with baseline models...")

    # Get sample
    batch = next(iter(val_loader))
    lr_hsi = batch['lr_hsi'].to(device)
    hr_pan = batch['hr_pan'].to(device)
    hr_hsi_gt = batch['hr_hsi'].to(device)

    num_channels = lr_hsi.shape[1]
    # Compute scale dynamically (don't hard-code)
    scale = hr_hsi_gt.shape[-1] * 1.0 / lr_hsi.shape[-1]
    if abs(scale - int(scale)) < 0.01:
        scale = int(scale)

    # Create models
    models = {
        'VMamba': VMambaPansharp(num_channels, num_channels, d_model=64, scale=scale),
        'CNN': CNNPansharp(num_channels, num_channels, scale=scale),
        'Transformer': TransformerPansharp(num_channels, num_channels, scale=scale),
        'U-Net': UNetPansharp(num_channels, num_channels, scale=scale)
    }

    predictions = {}
    metrics = {}

    # Load checkpoints and predict
    with torch.no_grad():
        for name, model in tqdm(models.items(), desc="Running models"):
            model = model.to(device)

            # CRITICAL: Load checkpoint before inference
            # Without this, all baseline results are fake (random initialization)
            if name in checkpoint_paths and checkpoint_paths[name]:
                print(f"  Loading {name} checkpoint from {checkpoint_paths[name]}")
                try:
                    checkpoint = torch.load(checkpoint_paths[name], map_location=device)
                    # Handle both direct state_dict and nested checkpoint format
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    print(f"  ✓ Loaded {name} checkpoint successfully")
                except Exception as e:
                    print(f"  ⚠ Warning: Could not load {name} checkpoint: {e}")
                    print(f"  ⚠ Using randomly initialized {name} (results will be meaningless!)")
            else:
                print(f"  ⚠ Warning: No checkpoint provided for {name}")
                print(f"  ⚠ Using randomly initialized {name} (results will be meaningless!)")

            model.eval()

            # Predict
            pred = model(lr_hsi, hr_pan)
            predictions[name] = pred[0].cpu().numpy()

            # Compute metrics
            metrics[name] = {
                'psnr': compute_psnr(pred, hr_hsi_gt),
                'sam': compute_sam_metric(pred, hr_hsi_gt),
                'ergas': compute_ergas(pred, hr_hsi_gt, scale=scale)
            }

    # Visualize comparison
    visualize_model_comparison(
        lr_hsi[0].cpu().numpy(),
        hr_pan[0].cpu().numpy(),
        predictions,
        hr_hsi_gt[0].cpu().numpy(),
        metrics,
        save_dir
    )


def visualize_model_comparison(lr_hsi, hr_pan, predictions, hr_gt, metrics, save_dir):
    """Create side-by-side comparison of all models"""

    C, H, W = hr_gt.shape
    rgb_bands = [int(C*0.7), int(C*0.5), int(C*0.2)] if C >= 60 else [min(50, C-1), min(30, C-1), min(10, C-1)]

    def normalize(img):
        if len(img.shape) == 3 and img.shape[0] < 10:
            img = img.transpose(1, 2, 0)
        return np.clip((img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1)

    model_names = list(predictions.keys())
    num_models = len(model_names)

    fig, axes = plt.subplots(3, num_models + 1, figsize=(5*(num_models+1), 12))

    # Row 1: RGB predictions
    for idx, name in enumerate(model_names):
        pred_rgb = normalize(predictions[name][rgb_bands, :, :])
        axes[0, idx].imshow(pred_rgb)
        axes[0, idx].set_title(f'{name}\nPSNR: {metrics[name]["psnr"]:.2f}dB',
                              fontweight='bold', fontsize=11)
        axes[0, idx].axis('off')

    # Ground truth
    gt_rgb = normalize(hr_gt[rgb_bands, :, :])
    axes[0, -1].imshow(gt_rgb)
    axes[0, -1].set_title('Ground Truth', fontweight='bold', fontsize=11)
    axes[0, -1].axis('off')

    # Row 2: Error maps
    for idx, name in enumerate(model_names):
        pred_rgb = normalize(predictions[name][rgb_bands, :, :])
        error = np.mean(np.abs(pred_rgb - gt_rgb), axis=2)
        im = axes[1, idx].imshow(error, cmap='hot', vmin=0, vmax=0.2)
        axes[1, idx].set_title(f'{name} Error\nSAM: {metrics[name]["sam"]:.2f}°',
                              fontweight='bold', fontsize=11)
        axes[1, idx].axis('off')
        plt.colorbar(im, ax=axes[1, idx], fraction=0.046)

    axes[1, -1].axis('off')

    # Row 3: Spectral signatures
    center_y, center_x = H//2, W//2
    for idx, name in enumerate(model_names):
        axes[2, idx].plot(hr_gt[:, center_y, center_x], 'b-', linewidth=2, label='GT', alpha=0.7)
        axes[2, idx].plot(predictions[name][:, center_y, center_x], 'r--',
                         linewidth=2, label='Pred', alpha=0.7)
        axes[2, idx].set_title(f'{name}\nERGAS: {metrics[name]["ergas"]:.4f}',
                              fontweight='bold', fontsize=11)
        axes[2, idx].set_xlabel('Band', fontsize=9)
        axes[2, idx].set_ylabel('Reflectance', fontsize=9)
        axes[2, idx].legend(fontsize=8)
        axes[2, idx].grid(True, alpha=0.3)

    # Metrics comparison
    axes[2, -1].axis('off')
    comparison_text = "Model Comparison:\n\n"
    for name in model_names:
        comparison_text += f"{name}:\n"
        comparison_text += f"  PSNR: {metrics[name]['psnr']:.2f}dB\n"
        comparison_text += f"  SAM: {metrics[name]['sam']:.2f}°\n"
        comparison_text += f"  ERGAS: {metrics[name]['ergas']:.4f}\n\n"

    axes[2, -1].text(0.1, 0.5, comparison_text, fontsize=9, family='monospace',
                    verticalalignment='center', transform=axes[2, -1].transAxes)

    plt.suptitle('Model Comparison - Visual Quality Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'models_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize VMamba-Pansharp outputs')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='pavia', choices=['pavia', 'chikusei'])
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Output directory')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")

    # Get dataset path
    if args.dataset == 'pavia':
        mat_path = 'pavia/PaviaU.mat'
    elif args.dataset == 'chikusei':
        mat_path = 'chikusei/chikusei.mat'
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    _, val_loader, _ = create_dataloaders_overlap(
        dataset=args.dataset,
        mat_path=mat_path,
        batch_size=1,
        patch_size=64,
        overlap=32,
        scale=2,
        num_workers=0
    )

    # Get number of channels and compute actual scale from data
    sample = next(iter(val_loader))
    num_channels = sample['lr_hsi'].shape[1]

    # Compute scale dynamically from data (don't hard-code)
    lr_w = sample['lr_hsi'].shape[-1]
    hr_w = sample['hr_hsi'].shape[-1]
    scale = hr_w * 1.0 / lr_w
    if abs(scale - int(scale)) < 0.01:
        scale = int(scale)
    print(f"Detected scale factor from data: {scale}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = VMambaPansharp(
        in_channels=num_channels,
        out_channels=num_channels,
        d_model=64,
        scale=scale,  # Use computed scale
        num_blocks=[3, 4, 4, 3]
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")

    # Visualize predictions
    visualize_model_predictions(model, val_loader, device, args.num_samples, args.output_dir)

    print(f"\n{'='*70}")
    print("✓ VISUALIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nGenerated visualizations in: {args.output_dir}/")
    print("  - model_output_sample_*.png")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
