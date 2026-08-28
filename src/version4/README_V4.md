# VMamba-Pansharp — Version 4
### M.Tech Thesis | Hyperspectral Pansharpening | Chikusei Dataset

---

## Four Fixes (V4 over V3) — Addressing Spectral Distortion

| # | Problem in V3 | V4 Fix | Metric Impact |
|---|---------------|--------|---------------|
| 1 | **PAN injection too strong** — Gate×Edges can corrupt spectral signatures | `F_out = F_hsi + α × Gate × Edges` (learnable α, init=0.1) | SAM ↓, ERGAS ↓ |
| 2 | **Weak reconstruction head** — 2 conv layers, residual may dominate bicubic | 3× Conv3×3 head + learnable β (init=0.5) for residual; bicubic is primary spectral anchor | PSNR ↑, SAM ↓ |
| 3 | **No spectral gradient loss** — L1/PSNR don't penalise wrong spectral slope | `L_spec = ||∇_c(H_pred) − ∇_c(H_gt)||_1` (band differences) | SAM ↓ |
| 4 | **Wrong loss weights** — λ_SAM=0.05 too small, λ_SSIM=0.05 too large | `L = 1.0*L1 + 0.10*SAM + 0.05*Edge + 0.01*SSIM + 0.05*L_spec` | SAM ↓, ERGAS ↓ |

---

## Fix 1 — ScaledSpatialDetailInjection

```
V3:  F_out = F_hsi + Gate × Edges
V4:  F_out = F_hsi + α × Gate × Edges      α = learnable scalar, init=0.1

Where:
  Edges = Conv3×3(F_pan)
  Gate  = sigmoid(Conv1×1([F_hsi ‖ F_pan]))  ← dual-stream (same as V3)
  α     ∈ [0, 1]  clamped during forward pass
```

α starts at 0.1 — the network injects very little PAN detail initially and learns the safe injection strength per training step.

---

## Fix 2 — StrongReconstructionHead

```
V1-V3:
  Residual = Conv3 → Conv3 → Conv1×1
  HR = Residual + Bicubic(LR-HSI)       ← equal weight

V4:
  Residual = Conv3 → Conv3 → Conv3 → Conv1×1   (one extra layer)
  HR = Bicubic(LR-HSI) + β × Residual           ← bicubic is anchor
  β = learnable scalar, init=0.5, clamped [0,1]
```

The bicubic term preserves all 128 spectral bands exactly. The residual adds only spatial detail the network has learned is spectrally safe.

---

## Fix 3 — Spectral Gradient Loss

```
∇_c(H) = H[:, 1:, :, :] - H[:, :-1, :, :]     (band-adjacent differences)
L_spec  = || ∇_c(H_pred) - ∇_c(H_gt) ||_1
```

Why band differences matter:
```
GT:     [0.5, 0.7, 0.9]   ← rising curve
Pred A: [0.5, 0.7, 0.9]   ← correct, low L_spec
Pred B: [0.5, 0.9, 0.7]   ← same L1 as A, but wrong spectral slope!
```
`Pred B` has same L1 loss as `Pred A` vs GT but `L_spec` detects it.

---

## Fix 4 — Loss Weight Rebalancing

| Term | V1–V3 | V4 | Reason |
|------|-------|----|--------|
| L1 | 1.00 | 1.00 | unchanged |
| SAM | 0.05 | **0.10** | spectral fidelity is primary |
| Edge | 0.05 | 0.05 | unchanged |
| SSIM | 0.05 | **0.01** | reduce purely spatial bias |
| L_spec | — | **0.05** | NEW: spectral gradient loss |

---

## V4 Architecture (Full)

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
  ├── ScaledSpatialDetailInjection  ← V4 Fix 1
  │     Edges = Conv3×3(F_pan)
  │     Gate  = sigmoid(Conv1×1([F_hsi ‖ F_pan]))
  │     α     = learnable scalar (init=0.1)
  │     F_fused = F_hsi + α × Gate × Edges
  │                           → F_fused (B, d, rH, rW)
  │
  ├── MultiScaleSpectralBackbone  (V3, unchanged)
  │     4-stage U-Net, each stage = SpectralSpatialMambaBlock
  │     SpectralSpatialMambaBlock = SpatialSS2D + SpectralSSM1D
  │                           → F_out (B, d, rH, rW)
  │
  └── StrongReconstructionHead  ← V4 Fix 2
        Conv3 → Conv3 → Conv3 → Conv1×1 → Residual
        β = learnable scalar (init=0.5)
        HR-HSI = Bicubic(LR-HSI) + β × Residual
                              → HR-HSI (B, 128, rH, rW)

Loss (V4 Fix 3 + 4):
  L = 1.0*L1 + 0.10*SAM + 0.05*Edge + 0.01*SSIM + 0.05*L_spec
```

---

## Parameter Count

| Model | Parameters | Notes |
|-------|-----------|-------|
| V1 Old VMamba | 1,840,352 | CrossAttention + Sobel + PixelShuffle |
| V2 Improved | 674,984 | CrossMamba + LearnableHF + ResidualUps |
| V3 VMamba | 1,509,256 | SDIM + MultiScale + SpectralSSM |
| **V4 VMamba** | **~1,523,000** | ScaledSDIM + StrongHead + SpectralGradLoss |

V4 adds ~14K params over V3:
- `ScaledSpatialDetailInjection`: +1 param (α scalar)
- `StrongReconstructionHead`: +1 param (β scalar) + one extra Conv3×3 layer

---

## Folder Structure

```
version4/
├── vmamba_pansharp_v4.py          ← V4 model (2 new components)
├── loss_functions_v4.py           ← SpectralGradientLoss + CompositeLossV4
├── baseline_models.py             ← Factory: 'old'|'improved'|'v3'|'v4'
├── dataset_loader.py              ← copy
├── dataset_loader_overlap.py      ← copy
├── run_experiment.py              ← MAIN RUNNER
│
├── comparison/
│   ├── compare_all.py             ← Train + compare all 4 variants
│   ├── checkpoints/               ← *_best.pth
│   ├── plots/                     ← comparison_all.png
│   ├── results/                   ← summary_all.json, training_log_*.xlsx
│   └── saved_images/              ← per-epoch input/output image snapshots
│       ├── epoch0005_old.png
│       ├── epoch0005_v4.png
│       ├── epoch0005_final_all.png
│       ├── epoch0005_spectral_curve.png
│       └── epoch0005_*_error.png
│
├── testing/
│   ├── test_chikusei.py           ← Evaluate all 4 on test patches
│   └── results/                   ← test_results.json
│
├── visualization/
│   ├── visualize_results.py       ← Publication-quality figures
│   └── outputs/
│       ├── false_colour_comparison.png
│       ├── error_map_comparison.png
│       ├── spectral_curves.png
│       └── individual/            ← lr_hsi.png, gt_hr_hsi.png, v4_pred.png …
│
└── README_V4.md                   ← THIS FILE
```

---

## Quick Commands

```bash
cd version4

# 0. Verify all 4 models load
python run_experiment.py verify

# 1. Quick smoke test (random weights, no training)
python run_experiment.py test

# 2. Train all 4 and compare (5 epochs, OOM-safe)
python run_experiment.py compare --epochs 5 --batch_size 1 --patch_size 32

# 3. Full training (30 epochs)
python run_experiment.py compare --epochs 30 --batch_size 1 --patch_size 32

# 4. Train only V4 (skip V1–V3 if already trained)
python run_experiment.py compare --skip_old --skip_improved --skip_v3 --epochs 30

# 5. Test with trained weights
python run_experiment.py test \
  --checkpoint_old      comparison/checkpoints/old_best.pth \
  --checkpoint_improved comparison/checkpoints/improved_best.pth \
  --checkpoint_v3       comparison/checkpoints/v3_best.pth \
  --checkpoint_v4       comparison/checkpoints/v4_best.pth

# 6. Generate publication figures
python run_experiment.py visualize \
  --checkpoint_old      comparison/checkpoints/old_best.pth \
  --checkpoint_improved comparison/checkpoints/improved_best.pth \
  --checkpoint_v3       comparison/checkpoints/v3_best.pth \
  --checkpoint_v4       comparison/checkpoints/v4_best.pth
```

---

## OOM-Safe Settings (CPU / 8 GB RAM)

| Setting | Value | Notes |
|---------|-------|-------|
| `--patch_size` | 32 | HR=32×32, LR=8×8, bottleneck=4×4 |
| `--batch_size` | 1 | Minimal memory |
| `--d_model` | 32 | V4: ~1.52M params |
| `--epochs` | 5 | Quick smoke test |

> **Important:** Do NOT use `--patch_size` below 32. The 4-stage backbone needs ≥4×4 at bottleneck.

---

## Saved Outputs

### During training (`comparison/saved_images/`):
- `epoch{N}_{variant}.png` — false colour + metrics for one model at epoch N
- `epoch{N}_{variant}_error.png` — error map heatmap
- `epoch{N}_final_all.png` — all models side by side (final epoch)
- `epoch{N}_spectral_curve.png` — spectral curves GT vs all models

### After training (`visualization/outputs/`):
- `false_colour_comparison.png` — side-by-side false colour (bands 30,20,10)
- `error_map_comparison.png` — absolute error heatmaps (same scale)
- `spectral_curves.png` — spectral curve + gradient at selected pixel
- `individual/*.png` — individual high-res images for each model

---

## What's Reused vs New

| Component | Source | Status |
|-----------|--------|--------|
| `ImprovedHSIEncoder` | V2 | Unchanged |
| `ImprovedPANEncoder` | V2 | Unchanged |
| `SpectralSSM1D` | V3 | Unchanged |
| `SpectralSpatialMambaBlock` | V3 | Unchanged |
| `MultiScaleSpectralBackbone` | V3 | Unchanged |
| `ScaledSpatialDetailInjection` | **V4 NEW** | SDIM + learnable α |
| `StrongReconstructionHead` | **V4 NEW** | 3-layer + learnable β |
| `SpectralGradientLoss` | **V4 NEW** | ∇_c band difference loss |
| `CompositeLossV4` | **V4 NEW** | Rebalanced weights + L_spec |
| `V4VMambaPansharp` | **V4 NEW** | Full model |

---

*Previous versions: see `../version1/README_V1.md` (V1), `../README_V2.md` (V2), `../version3/README_V3.md` (V3).*
