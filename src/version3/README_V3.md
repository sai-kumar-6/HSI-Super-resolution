# VMamba-Pansharp — Version 3
### M.Tech Thesis | Hyperspectral Pansharpening | Chikusei Dataset

---

## Three New Improvements (V3 over V2)

| # | Component | V2 (Improved) | V3 (New) | Benefit |
|---|-----------|---------------|----------|---------|
| 1 | **Spectral Mamba Block** | Spatial SS2D only (H,W) | Spatial SS2D **+** Spectral SSM (H,W,**C**) | Models inter-band correlations |
| 2 | **Multi-scale Backbone** | 3-stage U-Net (½, ¼ res) | **4-stage** U-Net (½, ¼, **⅛** res) | Deeper context hierarchy |
| 3 | **Fusion Gate** | LearnableHFInjection (PAN gate) | **SpatialDetailInjection** ([HSI+PAN] gate) | Context-aware, avoids spectral distortion |

---

## Improvement 1 — SpectralMambaBlock

**V2 backbone**: Only spatial modeling (scans H, W)

**V3 backbone**: Spatial + Spectral modeling (scans H, W, C)

```
Each block:
  x → SpatialSS2D  (existing SS2D, scans H × W per channel)
    → SpectralSSM1D (NEW: scans channel groups at each pixel)
    → output

SpectralSSM1D internals:
  x: (B, d_model, H, W)
  Reshape: (B*H*W, n_groups=8, d_per_group)   ← group channels
  SelectiveSSM: scan across 8 spectral groups  ← O(N) SSM
  Reshape back: (B, d_model, H, W)
  Residual: x + y
```

This models how feature channels depend on each other — capturing spectral band correlations not visible to a purely spatial scan.

---

## Improvement 2 — MultiScaleSpectralBackbone

**V2**: 3-stage U-Net (input at ½ resolution)

**V3**: 4-stage U-Net (input at full resolution)

```
Input (B, d, H, W)       ← full resolution (e.g. 32×32)
  │
  Stage 1: SpectralSpatialMamba × num_blocks[0]     (d,   H,   W  )
  ↓ Down1 (stride=2)
  Stage 2: SpectralSpatialMamba × num_blocks[1]     (2d,  H/2, W/2)
  ↓ Down2 (stride=2)
  Stage 3: SpectralSpatialMamba × num_blocks[2]     (4d*, H/4, W/4)
  ↓ Down3 (stride=2)
  Stage 4: SpectralSpatialMamba × num_blocks[3]     (4d*, H/8, W/8) ← bottleneck
  ↑ Up3 + skip3
  Dec3: SpectralSpatialMamba                         (4d*, H/4, W/4)
  ↑ Up2 + skip2
  Dec2: SpectralSpatialMamba                         (2d,  H/2, W/2)
  ↑ Up1 + skip1
  Dec1: SpectralSpatialMamba                         (d,   H,   W  )
  Output

* 4d capped at 128 for OOM safety
```

Channel schedule (d_model=32): **32 → 64 → 128 → 128**

---

## Improvement 3 — SpatialDetailInjection (SDIM)

**V2 LearnableHFInjection** (gate from PAN only):
```
Gate = sigmoid(Conv1×1(F_pan))
F_out = F_hsi + Gate × Conv3×3(F_pan)
```

**V3 SpatialDetailInjection** (gate from both HSI + PAN):
```
Edges = Conv3×3(F_pan)                         ← extract PAN detail
Gate  = sigmoid(Conv1×1([F_hsi ‖ F_pan]))      ← BOTH streams
F_out = F_hsi + Gate × Edges                   ← context-aware injection
```

The joint gate adapts to the current spectral state of the HSI features — preventing over-injection of PAN edges into spectrally sensitive regions.

---

## V3 Architecture (Full)

```
LR-HSI (B, 128, H, W)
  │
  ├── ImprovedHSIEncoder (V2, unchanged)
  │     Conv3D + spectral aggregation + 2× ResidualUpsample2×
  │                           → F_hsi (B, d, rH, rW)
  │
HR-PAN (B, 1, rH, rW)
  │
  ├── ImprovedPANEncoder (V2, unchanged)
  │     3× Conv2d + GroupNorm + ReLU
  │                           → F_pan (B, d, rH, rW)
  │
  ├── SpatialDetailInjection  ← V3 NEW
  │     Edges = Conv3×3(F_pan)
  │     Gate  = sigmoid(Conv1×1([F_hsi ‖ F_pan]))
  │     F_fused = F_hsi + Gate × Edges
  │                           → F_fused (B, d, rH, rW)
  │
  ├── MultiScaleSpectralBackbone  ← V3 NEW
  │     4-stage U-Net, each stage = SpectralSpatialMambaBlock
  │     SpectralSpatialMambaBlock = SpatialSS2D + SpectralSSM1D
  │                           → F_out (B, d, rH, rW)
  │
  └── ReconstructionHead (V1/V2, unchanged)
        Conv → residual + Bicubic(LR-HSI)
                              → HR-HSI (B, 128, rH, rW)
```

---

## Parameter Count

| Model | Parameters | Notes |
|-------|-----------|-------|
| V1 Old VMamba (d=32) | 1,840,352 | CrossAttention + Sobel + PixelShuffle |
| V2 Improved VMamba (d=32) | 674,984 | CrossMamba + LearnableHF + ResidualUps |
| **V3 VMamba (d=32)** | **1,509,256** | SDIM + MultiScale + SpectralSSM |

V3 is larger than V2 due to the 4-stage backbone, but smaller than V1. The extra parameters bring:
- Deeper multi-scale feature extraction (4 vs 3 stages)
- Joint spatial+spectral modeling in every block
- Context-aware dual-stream gating

---

## Folder Structure

```
version3/
├── vmamba_pansharp_v3.py          ← V3 model (3 new components)
├── baseline_models.py             ← Factory: create_vmamba_model('old'|'improved'|'v3')
├── dataset_loader.py              ← copy
├── dataset_loader_overlap.py      ← copy
├── loss_functions.py              ← copy
├── run_experiment.py              ← MAIN RUNNER
│
├── comparison/
│   ├── compare_all.py             ← Train + compare all 3 variants
│   ├── checkpoints/               ← old_best.pth, improved_best.pth, v3_best.pth
│   ├── plots/                     ← comparison_all.png
│   └── results/                   ← summary_all.json, training_log_*.xlsx
│
├── testing/
│   ├── test_chikusei.py           ← Evaluate on test patches
│   └── results/                   ← test_results.json
│
├── visualization/
│   └── outputs/
│
└── README_V3.md                   ← THIS FILE
```

---

## Quick Commands

```bash
cd version3

# 0. Verify all 3 models load
python run_experiment.py verify

# 1. Quick smoke test (random weights, no training, ~2 min)
python run_experiment.py test

# 2. Train and compare (5 epochs, OOM-safe)
python run_experiment.py compare --epochs 5 --batch_size 1 --patch_size 32 --d_model 32

# 3. Full training (30 epochs)
python run_experiment.py compare --epochs 30 --batch_size 1 --patch_size 32 --d_model 32

# 4. Test with trained weights
python run_experiment.py test \
  --checkpoint_old      comparison/checkpoints/old_best.pth \
  --checkpoint_improved comparison/checkpoints/improved_best.pth \
  --checkpoint_v3       comparison/checkpoints/v3_best.pth
```

---

## OOM-Safe Settings (CPU / 8 GB RAM)

| Setting | Value | Notes |
|---------|-------|-------|
| `--patch_size` | 32 | HR=32×32, LR=8×8, bottleneck=4×4 |
| `--batch_size` | 1 | Minimal memory |
| `--d_model` | 32 | V3: 1.5M params |
| `--epochs` | 5 | Quick smoke test |

> **Important:** Do NOT use `--patch_size` below 32. The 4-stage backbone needs at least 4×4 at the bottleneck. With scale=4 and patch_size=32: bottleneck = 32/8 = 4×4 (minimum safe).

---

## Excel Log

Training automatically saves per-epoch metrics to:
```
comparison/results/training_log_YYYYMMDD_HHMMSS.xlsx
```

- **Sheet 1 "Epoch Log"**: Model, Epoch, Train Loss, Val PSNR, Val SAM, Val ERGAS, LR, Time, Best?
- **Sheet 2 "Summary"**: Best metrics for all 3 variants side-by-side with colour coding

---

## What's Reused vs New

| Component | Source | Status |
|-----------|--------|--------|
| `ImprovedHSIEncoder` | V2 | Unchanged |
| `ImprovedPANEncoder` | V2 | Unchanged |
| `ReconstructionHead` | V1 | Unchanged |
| `MambaVisionBlockOptimized` | V1 | Used inside SpectralSpatialMambaBlock |
| `SelectiveSSMOptimized` | V1 | Used inside SpectralSSM1D |
| `SpectralSSM1D` | **V3 NEW** | Scans channels at each pixel |
| `SpectralSpatialMambaBlock` | **V3 NEW** | Spatial + Spectral combined |
| `SpatialDetailInjection` | **V3 NEW** | Dynamic dual-stream gating |
| `MultiScaleSpectralBackbone` | **V3 NEW** | 4-stage U-Net with spectral blocks |
| `V3VMambaPansharp` | **V3 NEW** | Full model |

---

*Previous versions: see `../version1/README_V1.md` (V1) and `../README_V2.md` (V2).*
