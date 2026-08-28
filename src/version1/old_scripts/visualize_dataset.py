"""
Dataset Visualization Script
Visualizes Pavia and Chikusei datasets with detailed analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import os


def load_dataset(dataset_name='pavia'):
    """Load dataset from .mat file"""
    print(f"\nLoading {dataset_name.upper()} dataset...")

    if dataset_name.lower() == 'pavia':
        path = 'pavia/PaviaU.mat'
        data = sio.loadmat(path)
        hsi = data['paviaU'] if 'paviaU' in data else data[list(data.keys())[-1]]
        wavelengths = np.linspace(430, 860, hsi.shape[2])  # nm
        name = 'Pavia University'
    else:  # chikusei
        path = 'chikusei/HyperspecVNIR_Chikusei_20140729.mat'
        data = sio.loadmat(path)
        hsi = data['chikusei'] if 'chikusei' in data else data[list(data.keys())[-1]]
        wavelengths = np.linspace(363, 1018, hsi.shape[2])  # nm
        name = 'Chikusei'

    print(f"✓ Loaded {name}: {hsi.shape}")
    return hsi, wavelengths, name


def visualize_dataset_overview(hsi, wavelengths, dataset_name, save_dir='visualizations'):
    """Create comprehensive dataset visualization"""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nGenerating dataset overview for {dataset_name}...")

    h, w, c = hsi.shape
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Normalize for display
    def normalize(img):
        img = img.astype(np.float32)
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    # 1. RGB Composite (bands approximating R, G, B)
    print("  Creating RGB composite...")
    if c >= 60:
        rgb_bands = [int(c*0.7), int(c*0.5), int(c*0.2)]  # Approximate RGB
    else:
        rgb_bands = [min(50, c-1), min(30, c-1), min(10, c-1)]

    ax1 = fig.add_subplot(gs[0, 0])
    rgb = normalize(hsi[:, :, rgb_bands])
    ax1.imshow(rgb)
    ax1.set_title(f'{dataset_name} - RGB Composite\n(Bands {rgb_bands})',
                 fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. False Color Composite
    print("  Creating false color composite...")
    ax2 = fig.add_subplot(gs[0, 1])
    false_bands = [int(c*0.8), int(c*0.4), int(c*0.1)]
    false_color = normalize(hsi[:, :, false_bands])
    ax2.imshow(false_color)
    ax2.set_title(f'False Color Composite\n(Bands {false_bands})',
                 fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 3. Single Band (Near-IR)
    print("  Creating single band visualization...")
    ax3 = fig.add_subplot(gs[0, 2])
    nir_band = int(c * 0.6)
    im3 = ax3.imshow(normalize(hsi[:, :, nir_band]), cmap='viridis')
    ax3.set_title(f'Single Band (Band {nir_band})\nλ≈{wavelengths[nir_band]:.0f}nm',
                 fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # 4. PAN Simulation (Average all bands)
    print("  Creating PAN simulation...")
    ax4 = fig.add_subplot(gs[0, 3])
    pan = np.mean(hsi, axis=2)
    im4 = ax4.imshow(normalize(pan), cmap='gray')
    ax4.set_title('Simulated PAN\n(Average of all bands)',
                 fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    # 5. Spectral Signatures (random pixels)
    print("  Plotting spectral signatures...")
    ax5 = fig.add_subplot(gs[1, :2])
    np.random.seed(42)
    num_samples = 10
    for i in tqdm(range(num_samples), desc="    Sampling pixels", leave=False):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        spectrum = hsi[y, x, :]
        ax5.plot(wavelengths, spectrum, alpha=0.6, linewidth=1.5)

    ax5.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Reflectance', fontsize=12, fontweight='bold')
    ax5.set_title(f'Spectral Signatures ({num_samples} random pixels)',
                 fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(wavelengths[0], wavelengths[-1])

    # 6. Mean Spectrum
    print("  Computing mean spectrum...")
    ax6 = fig.add_subplot(gs[1, 2:])
    mean_spectrum = np.mean(hsi.reshape(-1, c), axis=0)
    std_spectrum = np.std(hsi.reshape(-1, c), axis=0)
    ax6.plot(wavelengths, mean_spectrum, 'b-', linewidth=2, label='Mean')
    ax6.fill_between(wavelengths, mean_spectrum - std_spectrum,
                     mean_spectrum + std_spectrum, alpha=0.3, label='±1 Std')
    ax6.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Reflectance', fontsize=12, fontweight='bold')
    ax6.set_title('Average Spectral Signature', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(wavelengths[0], wavelengths[-1])

    # 7. Band Statistics Heatmap
    print("  Creating band statistics...")
    ax7 = fig.add_subplot(gs[2, 0])
    band_stats = np.array([
        np.mean(hsi[:, :, i]) for i in tqdm(range(c), desc="    Computing stats", leave=False)
    ])
    ax7.plot(range(c), band_stats, 'g-', linewidth=2)
    ax7.set_xlabel('Band Index', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Mean Reflectance', fontsize=12, fontweight='bold')
    ax7.set_title('Band-wise Mean Reflectance', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # 8. Correlation Matrix (sample of bands)
    print("  Computing band correlations...")
    sample_bands = np.linspace(0, c-1, min(20, c), dtype=int)
    corr_matrix = np.corrcoef(hsi[:, :, sample_bands].reshape(-1, len(sample_bands)).T)

    ax8 = fig.add_subplot(gs[2, 1])
    im8 = ax8.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax8.set_title(f'Band Correlation Matrix\n({len(sample_bands)} sampled bands)',
                 fontsize=12, fontweight='bold')
    ax8.set_xlabel('Band Index', fontsize=10)
    ax8.set_ylabel('Band Index', fontsize=10)
    plt.colorbar(im8, ax=ax8, fraction=0.046)

    # 9. Data Distribution
    print("  Computing data distribution...")
    ax9 = fig.add_subplot(gs[2, 2])
    flat_data = hsi.flatten()
    ax9.hist(flat_data, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax9.set_xlabel('Pixel Value', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax9.set_title('Pixel Value Distribution', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='y')
    ax9.set_yscale('log')

    # 10. Dataset Info
    ax10 = fig.add_subplot(gs[2, 3])
    ax10.axis('off')
    info_text = f"""
    Dataset: {dataset_name}

    Dimensions:
    • Height: {h} pixels
    • Width: {w} pixels
    • Bands: {c}
    • Total pixels: {h*w:,}

    Wavelength Range:
    • Min: {wavelengths[0]:.1f} nm
    • Max: {wavelengths[-1]:.1f} nm
    • Step: {(wavelengths[-1]-wavelengths[0])/c:.1f} nm

    Data Statistics:
    • Min value: {hsi.min():.4f}
    • Max value: {hsi.max():.4f}
    • Mean: {hsi.mean():.4f}
    • Std: {hsi.std():.4f}

    Memory:
    • Size: {hsi.nbytes / 1e6:.1f} MB
    """
    ax10.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax10.transAxes)

    plt.suptitle(f'{dataset_name} - Comprehensive Dataset Analysis',
                fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    save_path = os.path.join(save_dir, f'{dataset_name.lower().replace(" ", "_")}_overview.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {save_path}")
    plt.show()
    plt.close()


def visualize_walds_protocol(hsi, wavelengths, dataset_name, save_dir='visualizations'):
    """Visualize Wald's protocol data generation"""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nVisualizing Wald's Protocol for {dataset_name}...")

    from scipy.ndimage import gaussian_filter
    import cv2

    # Take a patch for demonstration
    patch_size = 256
    h, w, c = hsi.shape
    y, x = h//2 - patch_size//2, w//2 - patch_size//2
    hr_hsi = hsi[y:y+patch_size, x:x+patch_size, :].copy()

    print("  Step 1: Ground Truth (HR-HSI)")

    # Step 2: Generate LR-HSI
    print("  Step 2: Generating LR-HSI (Gaussian blur + decimation)...")
    lr_hsi = gaussian_filter(hr_hsi, sigma=[1.0, 1.0, 0])
    lr_hsi = cv2.resize(lr_hsi, (patch_size // 4, patch_size // 4),
                       interpolation=cv2.INTER_CUBIC)

    # Step 3: Generate HR-PAN
    print("  Step 3: Generating HR-PAN (spectral averaging)...")
    hr_pan = np.mean(hr_hsi, axis=2)

    # Create visualization
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

    def normalize(img):
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    # RGB bands
    rgb_bands = [int(c*0.7), int(c*0.5), int(c*0.2)]

    # Row 1: Original HR-HSI
    ax1 = fig.add_subplot(gs[0, 0])
    hr_rgb = normalize(hr_hsi[:, :, rgb_bands])
    ax1.imshow(hr_rgb)
    ax1.set_title(f'Ground Truth (HR-HSI)\n{hr_hsi.shape}', fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(normalize(hr_hsi[:, :, rgb_bands[0]]), cmap='Reds')
    ax2.set_title(f'Red Band\n{hr_hsi.shape[0]}×{hr_hsi.shape[1]}', fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(normalize(hr_hsi[:, :, rgb_bands[1]]), cmap='Greens')
    ax3.set_title(f'Green Band\n{hr_hsi.shape[0]}×{hr_hsi.shape[1]}', fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(normalize(hr_hsi[:, :, rgb_bands[2]]), cmap='Blues')
    ax4.set_title(f'Blue Band\n{hr_hsi.shape[0]}×{hr_hsi.shape[1]}', fontweight='bold')
    ax4.axis('off')

    # Row 2: Degraded LR-HSI
    ax5 = fig.add_subplot(gs[1, 0])
    lr_rgb = normalize(lr_hsi[:, :, rgb_bands])
    ax5.imshow(lr_rgb)
    ax5.set_title(f'LR-HSI (Degraded)\n{lr_hsi.shape}', fontweight='bold')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(normalize(lr_hsi[:, :, rgb_bands[0]]), cmap='Reds')
    ax6.set_title(f'Red Band (LR)\n{lr_hsi.shape[0]}×{lr_hsi.shape[1]}', fontweight='bold')
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(normalize(lr_hsi[:, :, rgb_bands[1]]), cmap='Greens')
    ax7.set_title(f'Green Band (LR)\n{lr_hsi.shape[0]}×{lr_hsi.shape[1]}', fontweight='bold')
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 3])
    ax8.imshow(normalize(lr_hsi[:, :, rgb_bands[2]]), cmap='Blues')
    ax8.set_title(f'Blue Band (LR)\n{lr_hsi.shape[0]}×{lr_hsi.shape[1]}', fontweight='bold')
    ax8.axis('off')

    # Row 3: HR-PAN and Spectral Comparison
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.imshow(normalize(hr_pan), cmap='gray')
    ax9.set_title(f'HR-PAN (Averaged)\n{hr_pan.shape}', fontweight='bold')
    ax9.axis('off')

    ax10 = fig.add_subplot(gs[2, 1:3])
    center_y, center_x = hr_hsi.shape[0]//2, hr_hsi.shape[1]//2
    hr_spectrum = hr_hsi[center_y, center_x, :]
    lr_spectrum = lr_hsi[center_y//4, center_x//4, :]
    ax10.plot(wavelengths, hr_spectrum, 'b-', linewidth=2, label='HR-HSI')
    ax10.plot(wavelengths, lr_spectrum, 'r--', linewidth=2, label='LR-HSI')
    ax10.set_xlabel('Wavelength (nm)', fontweight='bold')
    ax10.set_ylabel('Reflectance', fontweight='bold')
    ax10.set_title('Spectral Signature Comparison (Center Pixel)', fontweight='bold')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    ax11 = fig.add_subplot(gs[2, 3])
    ax11.axis('off')
    protocol_text = f"""
    Wald's Protocol:

    1. Ground Truth
       HR-HSI: {hr_hsi.shape}

    2. LR-HSI Generation
       • Gaussian blur
         σ = 1.0
       • Decimation ×4
       Result: {lr_hsi.shape}

    3. HR-PAN Generation
       • Average {c} bands
       Result: {hr_pan.shape}

    Scale Factor: 4×
    """
    ax11.text(0.1, 0.5, protocol_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax11.transAxes)

    plt.suptitle(f"Wald's Protocol - {dataset_name}", fontsize=16, fontweight='bold')

    save_path = os.path.join(save_dir, f'{dataset_name.lower().replace(" ", "_")}_walds_protocol.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to: {save_path}")
    plt.show()
    plt.close()


def main():
    """Main visualization function"""
    print("="*70)
    print("HYPERSPECTRAL DATASET VISUALIZATION")
    print("="*70)

    # Create visualization directory
    os.makedirs('visualizations', exist_ok=True)

    # Try Pavia first
    try:
        print("\n" + "="*70)
        print("PAVIA UNIVERSITY DATASET")
        print("="*70)
        hsi, wavelengths, name = load_dataset('pavia')

        # Dataset overview
        visualize_dataset_overview(hsi, wavelengths, name)

        # Wald's protocol
        visualize_walds_protocol(hsi, wavelengths, name)

    except FileNotFoundError:
        print("\n⚠ Pavia dataset not found at pavia/PaviaU.mat")

    # Try Chikusei
    try:
        print("\n" + "="*70)
        print("CHIKUSEI DATASET")
        print("="*70)
        hsi, wavelengths, name = load_dataset('chikusei')

        # Dataset overview
        visualize_dataset_overview(hsi, wavelengths, name)

        # Wald's protocol
        visualize_walds_protocol(hsi, wavelengths, name)

    except FileNotFoundError:
        print("\n⚠ Chikusei dataset not found at chikusei/HyperspecVNIR_Chikusei_20140729.mat")

    print("\n" + "="*70)
    print("✓ VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated visualizations:")
    print("  - visualizations/*_overview.png")
    print("  - visualizations/*_walds_protocol.png")
    print("="*70)


if __name__ == '__main__':
    main()
