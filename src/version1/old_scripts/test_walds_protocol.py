"""
Test and verify Wald's Protocol implementation
Demonstrates correct synthetic data generation
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.pyplot as plt


def test_walds_protocol():
    """Test Wald's protocol with a synthetic example"""
    print("="*70)
    print("Testing Wald's Protocol Implementation")
    print("="*70)

    # Create synthetic HR-HSI for testing
    # Simulate Pavia-like dimensions: 610×340×103
    print("\n1. Creating synthetic HR-HSI (Ground Truth)...")
    h, w, c = 610, 340, 103
    hr_hsi = np.random.randn(h, w, c) * 0.1 + 0.5
    hr_hsi = np.clip(hr_hsi, 0, 1)
    print(f"   Ground Truth (H_GT): {hr_hsi.shape}")
    print(f"   - Height: {h}")
    print(f"   - Width: {w}")
    print(f"   - Spectral bands: {c}")

    # Step 2: Generate LR-HSI
    print("\n2. Generating LR-HSI...")
    print("   Step 2a: Applying Gaussian blur (kernel 8×8, σ=1.0)...")
    lr_hsi = gaussian_filter(hr_hsi, sigma=[1.0, 1.0, 0])
    print(f"   - After blur: {lr_hsi.shape}")

    print("   Step 2b: Decimating by factor 4 (bicubic)...")
    scale = 4
    lr_hsi = cv2.resize(lr_hsi, (w // scale, h // scale),
                       interpolation=cv2.INTER_CUBIC)
    print(f"   Low Resolution (H_LR): {lr_hsi.shape}")
    print(f"   - Height: {lr_hsi.shape[0]} = {h}/4")
    print(f"   - Width: {lr_hsi.shape[1]} = {w}/4")
    print(f"   - Spectral bands: {lr_hsi.shape[2]} (preserved)")

    # Step 3: Generate HR-PAN
    print("\n3. Generating HR-PAN...")
    print(f"   Averaging all {c} spectral bands...")
    hr_pan = np.mean(hr_hsi, axis=2, keepdims=True)
    print(f"   High Resolution PAN (P_HR): {hr_pan.shape}")
    print(f"   - Height: {hr_pan.shape[0]} (same as H_GT)")
    print(f"   - Width: {hr_pan.shape[1]} (same as H_GT)")
    print(f"   - Spectral bands: {hr_pan.shape[2]} (averaged)")

    # Verify dimensions
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    print("\nTraining Triplet Dimensions:")
    print(f"  Input 1 (LR-HSI):  {lr_hsi.shape}")
    print(f"  Input 2 (HR-PAN):  {hr_pan.shape}")
    print(f"  Output  (HR-HSI):  {hr_hsi.shape}")

    print("\nDimension Checks:")
    expected_lr_h, expected_lr_w = h // scale, w // scale
    lr_check = (lr_hsi.shape[0] == expected_lr_h and
               lr_hsi.shape[1] == expected_lr_w and
               lr_hsi.shape[2] == c)
    pan_check = (hr_pan.shape[0] == h and
                hr_pan.shape[1] == w and
                hr_pan.shape[2] == 1)
    gt_check = (hr_hsi.shape[0] == h and
               hr_hsi.shape[1] == w and
               hr_hsi.shape[2] == c)

    print(f"  ✓ LR-HSI dimensions correct: {lr_check}")
    print(f"  ✓ HR-PAN dimensions correct: {pan_check}")
    print(f"  ✓ HR-HSI (GT) dimensions correct: {gt_check}")

    print("\nScale Factor:")
    scale_h = hr_hsi.shape[0] / lr_hsi.shape[0]
    scale_w = hr_hsi.shape[1] / lr_hsi.shape[1]
    print(f"  Height scale: {scale_h:.1f}× (expected 4×)")
    print(f"  Width scale: {scale_w:.1f}× (expected 4×)")
    print(f"  ✓ Scale factor correct: {scale_h == 4.0 and scale_w == 4.0}")

    print("\n" + "="*70)
    if all([lr_check, pan_check, gt_check, scale_h == 4.0, scale_w == 4.0]):
        print("✅ ALL CHECKS PASSED - Wald's Protocol Correctly Implemented!")
    else:
        print("❌ SOME CHECKS FAILED - Please review implementation")
    print("="*70)

    # Create visualization
    print("\n4. Creating visualization...")
    visualize_protocol(hr_hsi, lr_hsi, hr_pan)
    print("   Saved to 'walds_protocol_test.png'")

    return lr_hsi, hr_pan, hr_hsi


def visualize_protocol(hr_hsi, lr_hsi, hr_pan):
    """Visualize the Wald's protocol process"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Select a few bands for visualization (RGB-like)
    rgb_bands = [50, 30, 10]  # Approximate RGB

    # Helper function to normalize for display
    def normalize(img):
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Row 1: Ground Truth views
    # RGB composite from GT
    gt_rgb = normalize(hr_hsi[:, :, rgb_bands])
    axes[0, 0].imshow(gt_rgb)
    axes[0, 0].set_title(f'Ground Truth (HR-HSI)\n{hr_hsi.shape}', fontweight='bold')
    axes[0, 0].axis('off')

    # Single band from GT
    axes[0, 1].imshow(normalize(hr_hsi[:, :, 50]), cmap='viridis')
    axes[0, 1].set_title(f'GT Band 50\n{hr_hsi.shape[0]}×{hr_hsi.shape[1]}', fontweight='bold')
    axes[0, 1].axis('off')

    # PAN from GT
    axes[0, 2].imshow(normalize(hr_pan[:, :, 0]), cmap='gray')
    axes[0, 2].set_title(f'HR-PAN (Averaged)\n{hr_pan.shape[0]}×{hr_pan.shape[1]}', fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: Degraded views
    # RGB composite from LR
    lr_rgb = normalize(lr_hsi[:, :, rgb_bands])
    axes[1, 0].imshow(lr_rgb)
    axes[1, 0].set_title(f'LR-HSI (Degraded)\n{lr_hsi.shape}', fontweight='bold')
    axes[1, 0].axis('off')

    # Single band from LR
    axes[1, 1].imshow(normalize(lr_hsi[:, :, 50]), cmap='viridis')
    axes[1, 1].set_title(f'LR Band 50\n{lr_hsi.shape[0]}×{lr_hsi.shape[1]}', fontweight='bold')
    axes[1, 1].axis('off')

    # Comparison
    axes[1, 2].text(0.5, 0.7, 'Wald\'s Protocol', ha='center', va='center',
                   fontsize=16, fontweight='bold', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.5, 0.5, f'Scale Factor: 4×', ha='center', va='center',
                   fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.5, 0.35, f'LR: {lr_hsi.shape[0]}×{lr_hsi.shape[1]}×{lr_hsi.shape[2]}',
                   ha='center', va='center', fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.5, 0.25, f'HR-PAN: {hr_pan.shape[0]}×{hr_pan.shape[1]}×{hr_pan.shape[2]}',
                   ha='center', va='center', fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.5, 0.15, f'HR-HSI: {hr_hsi.shape[0]}×{hr_hsi.shape[1]}×{hr_hsi.shape[2]}',
                   ha='center', va='center', fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')

    plt.suptitle('Wald\'s Protocol for Synthetic Ground Truth Generation',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('walds_protocol_test.png', dpi=150, bbox_inches='tight')
    plt.close()


def test_with_dataset_loader():
    """Test with actual dataset loader"""
    print("\n" + "="*70)
    print("Testing with Dataset Loader")
    print("="*70)

    try:
        from dataset_loader import PansharpeningDataset
        import scipy.io as sio

        # Try to load Pavia dataset
        print("\nAttempting to load Pavia dataset...")
        try:
            data = sio.loadmat('pavia/PaviaU.mat')
            hsi_key = 'paviaU' if 'paviaU' in data else list(data.keys())[-1]
            hsi_data = data[hsi_key]
            print(f"✓ Loaded Pavia dataset: {hsi_data.shape}")

            # Create dataset
            print("\nCreating PansharpeningDataset...")
            dataset = PansharpeningDataset(
                hsi_data,
                patch_size=64,
                scale=4,
                overlap=0.5,
                augment=False,
                normalize=False
            )
            print(f"✓ Created dataset with {len(dataset)} patches")

            # Get a sample
            print("\nGetting sample triplet...")
            sample = dataset[0]
            print(f"  LR-HSI: {sample['lr_hsi'].shape}")
            print(f"  HR-PAN: {sample['hr_pan'].shape}")
            print(f"  HR-HSI: {sample['hr_hsi'].shape}")

            # Verify
            lr_shape = sample['lr_hsi'].shape
            pan_shape = sample['hr_pan'].shape
            hr_shape = sample['hr_hsi'].shape

            checks = [
                lr_shape[1] * 4 == hr_shape[1],  # Width scale
                lr_shape[2] * 4 == hr_shape[2],  # Height scale
                pan_shape[0] == 1,  # PAN is single band
                pan_shape[1:] == hr_shape[1:]  # PAN and HR same spatial size
            ]

            if all(checks):
                print("\n✅ Dataset loader correctly implements Wald's protocol!")
            else:
                print("\n❌ Dataset loader has issues")

        except FileNotFoundError:
            print("⚠ Pavia dataset not found at pavia/PaviaU.mat")
            print("  Please download and place the dataset in the correct location")

    except ImportError:
        print("⚠ Could not import dataset_loader module")


if __name__ == '__main__':
    # Test 1: Basic protocol
    lr_hsi, hr_pan, hr_hsi = test_walds_protocol()

    # Test 2: With dataset loader
    test_with_dataset_loader()

    print("\n" + "="*70)
    print("Testing Complete!")
    print("="*70)
    print("\nFor more information, see:")
    print("  - WALDS_PROTOCOL.md: Detailed protocol documentation")
    print("  - dataset_loader.py: Implementation code")
    print("  - VMAMBA_PANSHARP_README.md: Usage guide")
    print("="*70)
