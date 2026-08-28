# Wald's Protocol for Synthetic Ground Truth Generation

## Overview

Since acquiring true high-resolution hyperspectral (HR-HSI) images with corresponding low-resolution hyperspectral (LR-HSI) and high-resolution panchromatic (HR-PAN) images is practically impossible, we use **Wald's protocol** to generate synthetic training data from available HR-HSI datasets.

## Why Wald's Protocol?

**Problem**: In real-world scenarios, we cannot capture the same scene simultaneously at:
- Low spatial resolution with many spectral bands (LR-HSI)
- High spatial resolution with one spectral band (HR-PAN)
- High spatial resolution with many spectral bands (HR-HSI - this is what we want!)

**Solution**: Wald's protocol creates synthetic LR-HSI and HR-PAN from existing HR-HSI data, allowing us to:
1. Train models with known ground truth
2. Evaluate performance quantitatively
3. Use standard benchmark datasets

## Protocol Steps

### Step 1: Original HR-HSI (Ground Truth)

The full-resolution hyperspectral image serves as ground truth **H_GT**.

**Pavia University Dataset:**
```
H_GT ∈ R^(610 × 340 × 103)
├── Height: 610 pixels
├── Width: 340 pixels
└── Spectral bands: 103 (wavelength 430-860 nm)
```

**Chikusei Dataset:**
```
H_GT ∈ R^(2517 × 2335 × 128)
├── Height: 2517 pixels
├── Width: 2335 pixels
└── Spectral bands: 128 (wavelength 363-1018 nm)
```

### Step 2: LR-HSI Generation

Simulate low-resolution hyperspectral acquisition through degradation.

**Process:**
1. **Gaussian Blur**: Apply low-pass filter
   - Kernel size: 8×8
   - Standard deviation: σ = 1.0
   - Applied to spatial dimensions only (not spectral)

2. **Decimation**: Downsample by factor r=4
   - Method: Bicubic interpolation
   - Reduces spatial resolution while preserving spectral bands

**Results:**

**Pavia:**
```
H_LR ∈ R^(152 × 85 × 103)
├── Height: 152 = 610/4
├── Width: 85 = 340/4
└── Spectral bands: 103 (unchanged)
```

**Chikusei:**
```
H_LR ∈ R^(629 × 584 × 128)
├── Height: 629 = 2517/4
├── Width: 584 = 2335/4
└── Spectral bands: 128 (unchanged)
```

**Mathematical Formula:**
```
H_LR = Decimate(Blur(H_GT, σ=1.0), factor=4)
```

### Step 3: HR-PAN Generation

Simulate panchromatic sensor by averaging all spectral bands.

**Formula:**
```
P_HR = (1/C) × Σ(k=1 to C) H_GT^(k)

where:
- C = number of spectral bands
- H_GT^(k) = k-th spectral band of ground truth
```

**For Pavia (103 bands):**
```
P_HR = (1/103) × Σ(k=1 to 103) H_GT^(k)
```

**Results:**

**Pavia:**
```
P_HR ∈ R^(610 × 340)
├── Height: 610 (same as H_GT)
├── Width: 340 (same as H_GT)
└── Spectral bands: 1 (averaged)
```

**Chikusei:**
```
P_HR ∈ R^(2517 × 2335)
├── Height: 2517 (same as H_GT)
├── Width: 2335 (same as H_GT)
└── Spectral bands: 1 (averaged)
```

## Training Triplets

### Pavia University

```
Input 1 (LR-HSI):  H_LR ∈ R^(152 × 85 × 103)
Input 2 (HR-PAN):  P_HR ∈ R^(610 × 340 × 1)
Output (HR-HSI):   H_GT ∈ R^(610 × 340 × 103)

Scale factor: r = 4
```

### Chikusei

```
Input 1 (LR-HSI):  H_LR ∈ R^(629 × 584 × 128)
Input 2 (HR-PAN):  P_HR ∈ R^(2517 × 2335 × 1)
Output (HR-HSI):   H_GT ∈ R^(2517 × 2335 × 128)

Scale factor: r = 4
```

## Visual Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   Original HR-HSI (H_GT)                    │
│              Pavia: 610 × 340 × 103 bands                   │
│                        ↓                                    │
│                    ┌───┴───┐                                │
│                    ↓       ↓                                │
│          LR-HSI Generation    HR-PAN Generation             │
│                    ↓              ↓                         │
│              ┌─────────┐    ┌──────────┐                    │
│              │ Blur σ=1│    │  Average │                    │
│              │ 8×8 kern│    │  bands   │                    │
│              └────↓────┘    └────↓─────┘                    │
│              ┌─────────┐         ↓                          │
│              │Decimate │    P_HR (610×340×1)                │
│              │factor 4 │                                    │
│              └────↓────┘                                    │
│                   ↓                                         │
│         H_LR (152×85×103)                                   │
│                   ↓                                         │
│                   ↓                                         │
│         ┌─────────────────────┐                             │
│         │   VMamba-Pansharp   │                             │
│         │       Model         │                             │
│         └─────────────────────┘                             │
│                   ↓                                         │
│         Predicted HR-HSI (610×340×103)                      │
│                   ↓                                         │
│         Compare with H_GT for loss                          │
└─────────────────────────────────────────────────────────────┘
```

## Implementation in Code

```python
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

def walds_protocol(hr_hsi, scale=4):
    """
    Generate training triplet using Wald's protocol

    Args:
        hr_hsi: (H, W, C) - High resolution HSI (ground truth)
        scale: Upsampling factor (default: 4)

    Returns:
        lr_hsi: (H/4, W/4, C) - Low resolution HSI
        hr_pan: (H, W, 1) - High resolution PAN
        hr_hsi_gt: (H, W, C) - Ground truth (same as input)
    """
    h, w, c = hr_hsi.shape

    # Step 1: Ground truth
    hr_hsi_gt = hr_hsi.copy()

    # Step 2: Generate LR-HSI
    # Apply Gaussian blur (kernel 8×8, σ=1.0)
    lr_hsi = gaussian_filter(hr_hsi, sigma=[1.0, 1.0, 0])
    # Decimate by factor 4
    lr_hsi = cv2.resize(lr_hsi, (w // scale, h // scale),
                       interpolation=cv2.INTER_CUBIC)

    # Step 3: Generate HR-PAN
    # Average all spectral bands
    hr_pan = np.mean(hr_hsi, axis=2, keepdims=True)

    return lr_hsi, hr_pan, hr_hsi_gt


# Example usage for Pavia dataset
import scipy.io as sio

# Load Pavia dataset
data = sio.loadmat('pavia/PaviaU.mat')
hr_hsi = data['paviaU']  # Shape: (610, 340, 103)

# Generate training triplet
lr_hsi, hr_pan, hr_hsi_gt = walds_protocol(hr_hsi, scale=4)

print(f"Ground Truth (HR-HSI): {hr_hsi_gt.shape}")  # (610, 340, 103)
print(f"Low Res (LR-HSI):      {lr_hsi.shape}")     # (152, 85, 103)
print(f"High Res (HR-PAN):     {hr_pan.shape}")     # (610, 340, 1)
```

## Advantages of Wald's Protocol

1. **Realistic Training Data**
   - Simulates actual degradation in hyperspectral sensors
   - Maintains spatial-spectral relationships

2. **Quantitative Evaluation**
   - Known ground truth enables PSNR, SAM, ERGAS calculation
   - Fair comparison across different methods

3. **Standard Benchmark**
   - Widely used in pansharpening research
   - Enables reproducible results

4. **No Special Hardware Required**
   - Only need existing HR-HSI datasets
   - No synchronized multi-sensor acquisition

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Gaussian kernel | 8×8 | Low-pass filter size |
| Gaussian σ | 1.0 | Blur strength |
| Scale factor | 4 | Spatial downsampling ratio |
| Interpolation | Bicubic | Smooth downsampling |
| PAN bands | All (averaged) | Simulate panchromatic |

## Validation

The protocol ensures:
- ✅ Spatial degradation matches typical sensor ratios (4×)
- ✅ Spectral information preserved in LR-HSI
- ✅ Spatial details available in HR-PAN
- ✅ Ground truth available for training supervision
- ✅ Realistic simulation of sensor characteristics

## References

Wald, L., Ranchin, T., & Mangolini, M. (1997). Fusion of satellite images of different spatial resolutions: Assessing the quality of resulting images. *Photogrammetric Engineering and Remote Sensing*, 63(6), 691-699.

## Citation

If you use this implementation, please cite both the original Wald's protocol paper and this implementation:

```bibtex
@article{wald1997fusion,
  title={Fusion of satellite images of different spatial resolutions: Assessing the quality of resulting images},
  author={Wald, Lucien and Ranchin, Thierry and Mangolini, Marc},
  journal={Photogrammetric engineering and remote sensing},
  volume={63},
  number={6},
  pages={691--699},
  year={1997}
}
```
