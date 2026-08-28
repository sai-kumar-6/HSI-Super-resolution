# Version 6 — VMamba Hyperspectral Pansharpening
## Spectral-Focused Architecture with Reduced SAM

**M.Tech Thesis | Hyperspectral Image Pansharpening using State-Space Models**

---

## Table of Contents

1. [What V6 Solves](#1-what-v6-solves)
2. [Files in This Directory](#2-files-in-this-directory)
3. [Dataset & Data Pipeline](#3-dataset--data-pipeline)
4. [Full Model Pipeline — Step by Step with Tensor Shapes](#4-full-model-pipeline)
5. [Module-by-Module Architecture](#5-module-by-module-architecture)
6. [Loss Function](#6-loss-function)
7. [Key Hyperparameter Changes from V5](#7-key-hyperparameter-changes-from-v5)
8. [Training & Testing Commands](#8-training--testing-commands)
9. [Expected Results](#9-expected-results)
10. [Version Evolution Summary](#10-version-evolution-summary)

---

## 1. What V6 Solves

V5 achieved excellent PSNR (43.34 dB) but had high spectral distortion (SAM = 6.74°).
V6 specifically targets SAM reduction through four coordinated changes:

| Root Cause (V5) | V6 Fix | Expected SAM Gain |
|---|---|---|
| SAM loss weight too low (λ=0.10) | Raised to λ_sam=0.40 (+300%) | 0.5–1.0° |
| Spectral tokens too coarse (16 bands/token) | n_groups 8→32 (4 bands/token) | 0.7–1.2° |
| PAN injection too strong (α=0.10) | Reduced α_init=0.05 | 0.3–0.6° |
| Bicubic residual contribution too low (β=0.50) | Reduced β_init=0.3 | 0.2–0.4° |

**Target:** V5 SAM = 6.22° → V6 SAM = 3.0–4.5°

---

## 2. Files in This Directory

```
version6/
├── vmamba_pansharp_v6.py        # Model architecture (V6VMambaPansharp)
├── loss_functions_v6.py         # CompositeLossV6 (spectral-focused weights)
├── baseline_models.py           # Factory: creates V1–V6 models
├── run_experiment.py            # Training script
├── comparison/
│   ├── compare_all.py           # Compare V5 vs V6 side-by-side
│   ├── checkpoints/             # Saved model weights (v6_best.pth)
│   ├── results/                 # Training history JSON
│   ├── metric_plots/            # Training curve plots
│   └── saved_images/            # Visual output samples
└── testing/
    └── test_chikusei.py         # Evaluation on Chikusei test set
```

**Shared files (project root):**
```
dataset_loader_overlap.py        # Data pipeline with Wald's Protocol
chikusei/chikusei.mat            # Dataset (2517×2335×128 bands)
```

---

## 3. Dataset & Data Pipeline

### 3.1 Dataset: Chikusei

| Property | Value |
|---|---|
| Source | Airborne VNIR hyperspectral sensor (Japan) |
| Full image size | 2517 × 2335 pixels |
| Spectral bands | 128 (visible to near-infrared) |
| Wavelength range | 363 nm – 1018 nm |
| File format | MATLAB v7.3 HDF5 (.mat) |
| Loaded size in RAM | ~1435 MB (float16 storage) |

### 3.2 Train / Val / Test Split (NO spatial leakage)

The image is divided into non-overlapping spatial regions:

```
Full image: 2517 × 2335 × 128
│
├── Train region : 2432 × 1792 × 128  (rows: 0–2431,  cols: 0–1791)
├── Val   region : 2432 ×  256 × 128  (rows: 0–2431,  cols: 1792–2047)
└── Test  region : 2432 ×  256 × 128  (rows: 0–2431,  cols: 2048–2303)
```

Spatial separation ensures no pixel from train appears in val or test.

### 3.3 Patch Extraction

With `patch_size=32, scale=4`:
- HR stride = 64 pixels
- Train: 999 HR patches, Val: 111 patches, Test: 111 patches

### 3.4 Wald's Protocol (LR generation)

Real LR-HSI sensors are simulated from HR-HSI using Wald's Protocol:

```
HR-HSI patch (128×128×128)
        │
        ▼  Step 1: Gaussian blur — scipy.ndimage.gaussian_filter(sigma=[1,1,0])
           (smooths spatial dims only, not spectral; simulates sensor PSF)
        │
        ▼  Step 2: Bicubic downsample × (1/scale) — torch.nn.functional.interpolate
           (HR 128×128 → LR 32×32 for scale=4)
        │
        ▼  LR-HSI patch (32×32×128)

PAN generation:
HR-HSI patch (128×128×128)
        │
        ▼  Spectral mean across all 128 bands → single-channel
        │
        ▼  HR-PAN patch (128×128×1)
```

### 3.5 Final DataLoader Output Shapes

```
patch_size=32, scale=4:

  lr_hsi : (B, 128,  32,  32)   — LR hyperspectral input (model input)
  hr_pan : (B,   1, 128, 128)   — HR panchromatic input  (model input)
  hr_hsi : (B, 128, 128, 128)   — HR hyperspectral GT    (training target)

patch_size=64, scale=4:

  lr_hsi : (B, 128,  64,  64)
  hr_pan : (B,   1, 256, 256)
  hr_hsi : (B, 128, 256, 256)
```

---

## 4. Full Model Pipeline

All shapes shown for default config: `patch_size=32, scale=4, d_model=64, B=1`.

```
════════════════════════════════════════════════════════════════════════════════
INPUTS
════════════════════════════════════════════════════════════════════════════════

  LR-HSI  →  (B, 128, 32, 32)    Low-resolution 128-band hyperspectral image
  HR-PAN  →  (B,   1, 128, 128)  High-resolution single-band panchromatic image

════════════════════════════════════════════════════════════════════════════════
STEP 1 — HSI Encoder  [ImprovedHSIEncoder]
════════════════════════════════════════════════════════════════════════════════

  Input  : (B, 128, 32, 32)

  1a. Unsqueeze channel dim
      (B, 128, 32, 32)  →  (B, 1, 128, 32, 32)

  1b. Conv3d(1→8, k=3, pad=1) + GroupNorm + ReLU
      Joint spectral-spatial feature extraction in 3D
      (B, 1, 128, 32, 32)  →  (B, 8, 128, 32, 32)

  1c. Reshape: collapse 3D features into 2D
      (B, 8, 128, 32, 32)  →  (B, 1024, 32, 32)   [8×128=1024 channels]

  1d. Conv2d(1024→64, k=1) + ReLU
      Spectral aggregation: 1024 channels → d_model=64
      (B, 1024, 32, 32)  →  (B, 64, 32, 32)

  1e. ResidualUpsample2x  ×2  (for scale=4)
      Each stage: bilinear 2× + residual Conv
      (B, 64, 32, 32)  →  (B, 64, 64, 64)  →  (B, 64, 128, 128)

  Output : (B, 64, 128, 128)   F_hsi — HSI features at HR resolution

════════════════════════════════════════════════════════════════════════════════
STEP 2 — PAN Encoder  [ImprovedPANEncoder]
════════════════════════════════════════════════════════════════════════════════

  Input  : (B, 1, 128, 128)

  2a. Conv2d(1→32, k=3, pad=1) + GroupNorm(8) + ReLU
      (B, 1, 128, 128)  →  (B, 32, 128, 128)

  2b. Conv2d(32→64, k=3, pad=1) + GroupNorm(8) + ReLU
      (B, 32, 128, 128)  →  (B, 64, 128, 128)

  2c. Conv2d(64→64, k=3, pad=1) + GroupNorm(8) + ReLU
      (B, 64, 128, 128)  →  (B, 64, 128, 128)

  Output : (B, 64, 128, 128)   F_pan — PAN features at HR resolution

════════════════════════════════════════════════════════════════════════════════
STEP 3 — Scaled Spatial Detail Injection  [ScaledSpatialDetailInjection]
════════════════════════════════════════════════════════════════════════════════

  Inputs : F_hsi = (B, 64, 128, 128)
           F_pan = (B, 64, 128, 128)

  3a. Edge extraction from PAN features
      F_pan  →  Conv(64→64, k=3) + GroupNorm + ReLU  →  edges
      edges  : (B, 64, 128, 128)

  3b. Dual-stream attention gate
      Concat([F_hsi, F_pan])  : (B, 128, 128, 128)
      Conv(128→64, k=1) + GroupNorm + Sigmoid  →  gate
      gate   : (B, 64, 128, 128)   values ∈ [0, 1]

  3c. Scaled injection
      α = clamp(alpha_param, 0, 1)  →  scalar  [α_init=0.05 in V6, was 0.1]

      F_out = F_hsi + α × gate × edges

      Interpretation:
        - α=0.05 means PAN detail initially contributes only 5% of full strength
        - gate controls WHICH spatial locations receive PAN detail
        - α controls HOW STRONGLY detail is injected everywhere
        - Low α prevents spectral band corruption from PAN noise

  3d. Output normalization: GroupNorm

  Output : (B, 64, 128, 128)   F_fused — HSI + PAN fused features

════════════════════════════════════════════════════════════════════════════════
STEP 4 — V6 Multi-Scale Backbone  [V6MultiScaleBackbone]
════════════════════════════════════════════════════════════════════════════════

  Channel dimensions:
    d1 = d_model     = 64
    d2 = d_model × 2 = 128
    d3 = d_model × 4 = 256   (capped at _MAX_CH=256)
    d4 = d3          = 256

  ── ENCODER PATH ──────────────────────────────────────────────────────────────

  Input x: (B, 64, 128, 128)

  Stage 1: V6SpectralSpatialBlock(dim=64) × num_blocks[0]
    Input  : (B, 64, 128, 128)
    [See Section 5.1 for detailed internals]
    Spatial SSM: operates at 32×32 (downsampled, then upsampled back to 128×128)
    Spectral SSM: 32 groups of 4 bands each, at 128×128 spatial resolution
    Output : s1 = (B, 64, 128, 128)          ← skip connection saved

  Down 1: Conv2d(64→128, k=2, stride=2)
    (B, 64, 128, 128)  →  (B, 128, 64, 64)

  Stage 2: V6SpectralSpatialBlock(dim=128) × num_blocks[1]
    Input  : (B, 128, 64, 64)
    Spatial SSM: operates at 32×32 (downsampled, then upsampled back to 64×64)
    Spectral SSM: 32 groups, at 64×64 spatial resolution
    Output : s2 = (B, 128, 64, 64)            ← skip connection saved

  Down 2: Conv2d(128→256, k=2, stride=2)
    (B, 128, 64, 64)  →  (B, 256, 32, 32)

  Stage 3: V6SpectralSpatialBlock(dim=256) × num_blocks[2]
    Input  : (B, 256, 32, 32)
    Spatial SSM: operates at 32×32 (native resolution, no downsampling needed)
    Spectral SSM: 32 groups, at 32×32 spatial resolution
    Output : s3 = (B, 256, 32, 32)            ← skip connection saved

  Down 3: Conv2d(256→256, k=2, stride=2)
    (B, 256, 32, 32)  →  (B, 256, 16, 16)

  Stage 4 (Bottleneck): V6SpectralSpatialBlock(dim=256) × num_blocks[3]
    Input  : (B, 256, 16, 16)
    Spatial SSM: operates at 16×16 (native, L=256 tokens — very efficient)
    Spectral SSM: 32 groups, at 16×16 spatial resolution
    Output : s4 = (B, 256, 16, 16)

  ── DECODER PATH ──────────────────────────────────────────────────────────────

  Up 3: ConvTranspose2d(256→256, k=2, stride=2)
    s4: (B, 256, 16, 16)  →  (B, 256, 32, 32)

  Merge 3: Concat + Conv + GroupNorm
    Concat([up3(s4), s3])  : (B, 512, 32, 32)   [256+256]
    Conv2d(512→256, k=1) + GroupNorm  →  (B, 256, 32, 32)

  Dec 3: V6SpectralSpatialBlock(dim=256)
    Input  : (B, 256, 32, 32)
    Output : d3 = (B, 256, 32, 32)

  Up 2: ConvTranspose2d(256→128, k=2, stride=2)
    d3: (B, 256, 32, 32)  →  (B, 128, 64, 64)

  Merge 2: Concat + Conv + GroupNorm
    Concat([up2(d3), s2])  : (B, 256, 64, 64)   [128+128]
    Conv2d(256→128, k=1) + GroupNorm  →  (B, 128, 64, 64)

  Dec 2: V6SpectralSpatialBlock(dim=128)
    Input  : (B, 128, 64, 64)
    Spatial SSM: operates at 32×32 (downsampled, upsampled back to 64×64)
    Output : d2 = (B, 128, 64, 64)

  Up 1: ConvTranspose2d(128→64, k=2, stride=2)
    d2: (B, 128, 64, 64)  →  (B, 64, 128, 128)

  Merge 1: Concat + Conv + GroupNorm
    Concat([up1(d2), s1])  : (B, 128, 128, 128)  [64+64]
    Conv2d(128→64, k=1) + GroupNorm  →  (B, 64, 128, 128)

  Dec 1: V6SpectralSpatialBlock(dim=64)
    Input  : (B, 64, 128, 128)
    Spatial SSM: operates at 32×32 (downsampled, upsampled back to 128×128)
    Output : d1 = (B, 64, 128, 128)

  Backbone Output: (B, 64, 128, 128)   F_out

════════════════════════════════════════════════════════════════════════════════
STEP 5 — Strong Reconstruction Head  [StrongReconstructionHead]
════════════════════════════════════════════════════════════════════════════════

  Inputs : F_out  = (B, 64, 128, 128)    backbone features
           lr_hsi = (B, 128, 32, 32)     original LR-HSI (spectral anchor)

  5a. 3-layer feature refiner (Conv3×3 with residual character)
      F_out  →  Conv(64→64, k=3)+GN+ReLU  →  Conv(64→64, k=3)+GN+ReLU
             →  Conv(64→64, k=3)+GN+ReLU  →  Conv(64→128, k=1)
      residual : (B, 128, 128, 128)

  5b. Bicubic baseline (spectral anchor)
      LR-HSI bicubic upsample ×4
      (B, 128, 32, 32)  →  (B, 128, 128, 128)   [bicubic, spectrally perfect]

  5c. Beta-scaled combination
      β = clamp(beta_param, 0, 1)   [β_init=0.3 in V6, was 0.5]

      HR-HSI = bicubic_baseline + β × residual

      Interpretation:
        - bicubic_baseline: preserves ALL 128 spectral bands perfectly (SAM=0 by itself)
        - residual: backbone adds spatial detail but may distort spectra
        - β=0.3 means backbone contributes only 30% initially
        - Low β keeps spectral bands close to the clean bicubic baseline
        - β is learnable: model can increase it if spectral distortion stays low

  Output : (B, 128, 128, 128)   HR-HSI prediction

════════════════════════════════════════════════════════════════════════════════
FINAL OUTPUT
════════════════════════════════════════════════════════════════════════════════

  (B, 128, 128, 128)  — predicted HR hyperspectral image
  Same spatial resolution as HR-PAN input (128×128)
  All 128 spectral bands preserved
```

---

## 5. Module-by-Module Architecture

### 5.1 V6SpectralSpatialBlock

The fundamental processing unit. Each block applies two sequential SSMs:

```
Input: (B, C, H, W)
│
├── [SPATIAL SSM BRANCH]
│   │
│   ├── If H > 32 or W > 32:  (memory-efficient path)
│   │    AdaptiveAvgPool2d → (B, C, 32, 32)          [downsample]
│   │    Permute → (B, 32, 32, C)
│   │    LayerNorm(C)
│   │    TrueParallelSS2D (4-direction Hillis-Steele scan at L=1024)
│   │    → (B, 32, 32, C)
│   │    Permute → (B, C, 32, 32)
│   │    Bilinear upsample → (B, C, H, W)              [upsample back]
│   │
│   └── If H <= 32 and W <= 32:  (native resolution path)
│        Permute → (B, H, W, C)
│        LayerNorm(C)
│        TrueParallelSS2D (4-direction scan at L=H×W)
│        → (B, H, W, C)
│        Permute → (B, C, H, W)
│
│   x = x + spatial_output    [residual connection]
│
└── [SPECTRAL SSM BRANCH]
    V6SpectralSSM1D(n_groups=32)
    x = spectral_ssm(x)       [residual inside spectral SSM]

Output: (B, C, H, W)
```

**Why cap spatial at 32×32?**

| Spatial tokens L | Memory per Hillis-Steele iteration | 14 iterations × 4 dirs |
|---|---|---|
| 128×128 = 16,384 | ~64 MB | **~13 GB** (OOM) |
| 64×64 = 4,096 | ~16 MB | ~3.5 GB |
| **32×32 = 1,024** | **~4 MB** | **~0.9 GB** (safe) |

Spatial detail lost by downsampling is recovered through U-Net skip connections.

---

### 5.2 TrueParallelSS2D (Hillis-Steele 4-direction scan, from V5)

Processes the 2D spatial map in 4 scanning directions:

```
Input: (B, H, W, C)
│
├── Direction 0: Row-forward   (B, H×W, C) left→right, row by row
├── Direction 1: Row-reverse   (B, H×W, C) right→left, row by row
├── Direction 2: Col-forward   (B, H×W, C) top→bottom, col by col
└── Direction 3: Col-reverse   (B, H×W, C) bottom→top, col by col

Each direction:
  x_seq → TrueParallelSelectiveSSM (Hillis-Steele O(log L) scan)
  y → direction-specific adapter Conv → y_adapted

Output = x + sum(y_adapted for all 4 directions)
```

---

### 5.3 V6SpectralSSM1D (NEW in V6 — n_groups=32)

Captures correlations between adjacent spectral bands:

```
Input: (B, d_model, H, W)   e.g. (B, 64, 128, 128)
│
├── Permute → (B, H, W, d_model)
├── LayerNorm(d_model)
├── Reshape → (N, d_model)              where N = B×H×W = 16384
├── Reshape → (N, 32, 2)                32 groups of 2 features each
│                                       [each group = 4 consecutive bands]
│
├── TrueParallelSelectiveSSM
│   Input:  (N, 32, 2)    [sequence of 32 spectral tokens]
│   d_model = d_per_grp = 2
│   d_state = 4
│   Output: (N, 32, 2)    [spectral correlations captured]
│
├── Reshape → (N, 64)  →  (B, H, W, 64)  →  (B, 64, H, W)
├── GroupNorm(8, 64)
│
└── x = x + spectral_output   [residual]

Output: (B, d_model, H, W)
```

**V5 vs V6 spectral resolution:**
```
V5: n_groups= 8  →  128 bands / 8  = 16 bands per token  (coarse)
V6: n_groups=32  →  128 bands / 32 =  4 bands per token  (fine)

Example — what each token sees:
  V5 group 0: bands 0–15   (16 bands merged into 1 token)
  V6 group 0: bands 0–3    (4 bands merged into 1 token)

Finer tokens = better spectral discrimination = lower SAM
```

---

### 5.4 TrueParallelSelectiveSSM — Hillis-Steele Scan Core (from V5)

The SSM state equation: `h[t] = a[t] * h[t-1] + b[t]`

Standard sequential scan (V1–V4): O(L) depth — each step depends on previous.

Hillis-Steele parallel scan (V5, V6): O(log L) depth.

```
Standard (O(L) depth):          Hillis-Steele (O(log L) depth):
                                 
t=0: h[0] = b[0]               Pass 1 (stride=1):
t=1: h[1] = a[1]*h[0]+b[1]        b'[t] = a[t]*b[t-1] + b[t]   (all t parallel)
t=2: h[2] = a[2]*h[1]+b[2]     Pass 2 (stride=2):
...                                b''[t] = a[t]*b'[t-2] + b'[t] (all t parallel)
t=L: O(L) sequential steps      ...
                                 log2(L) passes total

For L=1024 (32×32): 10 passes vs 1024 sequential steps
```

---

## 6. Loss Function

### CompositeLossV6

```
L_total = λ_l1   × L_L1
        + λ_sam  × L_SAM
        + λ_edge × L_Edge
        + λ_ssim × L_SSIM
        + λ_spec × L_SpectralGrad
```

| Term | Weight (V6) | Weight (V5) | Change | Formula |
|---|---|---|---|---|
| L1 (pixel MAE) | **1.00** | 1.00 | — | `mean(|pred - gt|)` |
| SAM (spectral angle) | **0.40** | 0.10 | +300% | `mean(arccos(dot(p,g) / (|p||g|)))` |
| Edge preservation | **0.05** | 0.05 | — | Sobel edge MAE on mean band |
| SSIM | **0.01** | 0.01 | — | `1 - SSIM(pred, gt)` |
| SpectralGrad | **0.10** | 0.05 | +100% | `mean(|∇_c(pred) - ∇_c(gt)|)` |

**SpectralGradientLoss detail:**
```
∇_c(H)[b] = H[b+1] - H[b]    (adjacent band difference, b = 0..126)
L_spec = (1/127) × Σ_b mean(|∇_c(pred)[b] - ∇_c(gt)[b]|)

Penalises wrong spectral curve shapes (slopes between bands).
Even if absolute values are close, wrong slopes = wrong spectral signature.
```

---

## 7. Key Hyperparameter Changes from V5

| Parameter | V5 | V6 | Why changed |
|---|---|---|---|
| `n_groups` | 8 | **32** | 4 bands/token vs 16 — finer spectral resolution |
| `d_model` | 32 | **64** | Required for n_groups=32 (d_per_grp = 64/32 = 2) |
| `alpha_init` | 0.10 | **0.05** | Less PAN contamination into spectral bands |
| `beta_init` | 0.50 | **0.30** | Bicubic baseline contributes more (SAM-safe) |
| `λ_sam` | 0.10 | **0.40** | Directly optimise the SAM metric |
| `λ_spec` | 0.05 | **0.10** | Penalise wrong band-to-band slopes |
| `_SCAN_MAX` | N/A | **32** | Cap spatial SSM to 32×32 (prevents GPU OOM) |
| Parameters | 1.77M | **6.78M** | Increased d_model 32→64 |

---

## 8. Training & Testing Commands

### Training (recommended)

```bash
conda activate deepfake
cd "c:\Users\s.saikumar\Desktop\Mtech project\version6"

# Full training (recommended — batch_size=4 uses ~4-5 GB GPU memory)
python run_experiment.py --epochs 200 --d_model 64 --patch_size 32 --scale 4 --batch_size 4

# If GPU OOM, reduce to batch_size=2
python run_experiment.py --epochs 200 --d_model 64 --patch_size 32 --scale 4 --batch_size 2
```

### All command-line arguments

| Argument | Default | Options | Description |
|---|---|---|---|
| `--epochs` | 200 | any int | Training epochs |
| `--d_model` | 64 | 32, 64, 128 | Feature dimension (use 64 for V6) |
| `--d_state` | 8 | 4, 8, 16 | SSM state dimension |
| `--patch_size` | 32 | 32, 64 | LR input patch size |
| `--scale` | 4 | 2, 4, 8 | Spatial upsampling factor |
| `--batch_size` | 1 | 1–8 | Training batch size |
| `--overlap` | 16 | any int | Patch extraction overlap (LR space) |
| `--lr` | 3e-4 | float | Learning rate (AdamW) |
| `--grad_clip` | 0.5 | float | Gradient clipping norm |
| `--num_blocks` | 1 | 1–4 | Blocks per stage |

### Testing

```bash
# Test V6 only
python testing/test_chikusei.py --checkpoint_v6 comparison/checkpoints/v6_best.pth

# Test V5 vs V6 comparison
python testing/test_chikusei.py \
    --checkpoint_v5 ../version5/comparison/checkpoints/v5_best.pth \
    --checkpoint_v6 comparison/checkpoints/v6_best.pth \
    --max_patches 50
```

### Quick self-test (verify model runs)

```bash
python vmamba_pansharp_v6.py
```

Expected output:
```
Testing V6VMambaPansharp ...
  Parameters : 6,778,698  (6.779 M)
  LR (32x32)  -> OUT (128x128)  [OK]
  LR (64x64)  -> OUT (256x256)  [OK]
  [OK] CompositeLossV6  lambda_sam=0.40  lambda_spec=0.10
  alpha=0.0500  beta=0.3000
```

---

## 9. Expected Results

Based on V5 baseline (43.34 dB / 6.74° SAM) and the targeted improvements:

| Metric | V5 (50 ep) | V6 Target | Direction |
|---|---|---|---|
| PSNR (dB) | 43.34 | ~43–45 | higher is better |
| SAM (°) | 6.74 | **3.0–4.5** | lower is better |
| ERGAS | 5.61 | ~4–6 | lower is better |
| SSIM | 0.9622 | ~0.97–0.98 | higher is better |

Note: Training on CPU is slow (~5–15 min/epoch for 999 patches). Use GPU (auto-detected) for reasonable training time (~30–90 sec/epoch on CUDA).

---

## 10. Version Evolution Summary

| Version | Key Innovation | Best PSNR | Best SAM | Params |
|---|---|---|---|---|
| V1 | CrossAttention + PixelShuffle (baseline) | 44.65 | 6.74° | 1.84M |
| V2 | CrossMamba O(N) + learnable HF injection | 44.99 | 6.77° | 0.67M |
| V3 | SpectralMambaBlock + 4-stage U-Net | 47.35 | 6.61° | 1.52M |
| V4 | Scaled SDIM (α) + StrongReconHead (β) + SpectralGradLoss | **49.37** | **6.32°** | 1.53M |
| V5 | Hillis-Steele O(log L) true parallel scan | 43.34 | 6.74° | 1.77M |
| **V6** | **n_groups=32 + λ_sam=0.40 + α=0.05 + β=0.3** | TBD | **target ~4°** | **6.78M** |

---

*Generated for M.Tech thesis: "VMamba-Based Hyperspectral Image Pansharpening"*
