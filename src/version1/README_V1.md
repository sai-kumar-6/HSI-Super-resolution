# VMamba-Pansharp — Version 1 (Previous Architecture)
### M.Tech Thesis | Hyperspectral Image Pansharpening

---

## What this version contains

This folder archives **all results, experiments, and documentation** from the
**first development phase** of the project, which used CNN, U-Net, and
Transformer baselines compared against the original VMamba-Pansharp model.

---

## V1 Architecture

```
LR-HSI
  │
  ├── 3D Conv Spectral Encoder
  │       └── PixelShuffle ×2 (4× upsampling)  ← checkerboard risk
  │
HR-PAN
  │
  ├── Lightweight Spatial Encoder
  │       └── Sobel EdgeEnhancement             ← fixed filter, not adaptive
  │
  ├── CrossAttentionFusion (O(N²))              ← bottleneck for large images
  │
  ├── Hierarchical VMamba Backbone (4-stage U-Net)
  │
  └── Reconstruction Head → HR-HSI
```

### V1 Loss Function
```
L = 1.0×L1 + 0.1×SAM + 0.05×Edge + 0.1×SSIM
```

---

## V1 Baseline Models (compared against VMamba)

| Model | Architecture | Notes |
|-------|-------------|-------|
| **CNN** | 8× ResidualBlock + fusion | Simple, fast |
| **U-Net** | Encoder-decoder + skip connections | Memory heavy |
| **Transformer** | Swin-style window attention | O(N²) in window |
| **VMamba (V1)** | CrossAttention + Sobel + PixelShuffle | Proposed V1 |

---

## V1 Files (archived here)

All V1 scripts have been moved to `version1/old_scripts/`.

| File | Purpose |
|------|---------|
| `old_scripts/vmamba_pansharp.py` | V1 model (still at root for comparison use) |
| `old_scripts/train_vmamba_pansharp.py` | V1 training script |
| `old_scripts/evaluate_vmamba_pansharp.py` | V1 evaluation |
| `old_scripts/compare_models.py` | V1 comparison (CNN vs UNet vs Transformer vs VMamba) |
| `old_scripts/visualization_utils.py` | V1 visualization helpers |
| `old_scripts/quick_start.py` | V1 quick start |
| `old_scripts/test_*.py` | V1 unit tests (10 files) |
| `old_scripts/config_*.py` | V1 training configs |
| `old_scripts/run_training_overlap.py` | V1 overlap training |
| `old_scripts/*.ipynb` | V1 analysis notebooks |

---

## V1 Archived Results

```
version1/
├── comparison_results/          ← 18 Pavia comparison runs (CNN/UNet/Transformer/VMamba)
│   ├── comparison_pavia_20251221_*/
│   └── ...
├── experiments/                 ← 15 VMamba training runs (Pavia + Chikusei)
│   ├── vmamba_pansharp_pavia_*/
│   ├── vmamba_pansharp_chikusei_*/
│   └── pavia_overlap_scale2/
├── visualizations/              ← Dataset overview figures
│   ├── pavia_university_overview.png
│   └── pavia_university_walds_protocol.png
├── old_scripts/                 ← 26 archived V1 Python scripts + notebooks
├── old_folders/                 ← Archived empty/unused folders (pavia, dataset1, etc.)
└── docs/                        ← All V1 documentation (19 .md files)
    ├── EXECUTION_GUIDE.md
    ├── WALDS_PROTOCOL.md
    ├── MEMORY_FIX_README.md
    ├── MEMORY_FIX_SUMMARY.md
    ├── MEMORY_OPTIMIZATIONS.md
    ├── NAN_GRADIENT_FIX_SUMMARY.md
    ├── OPTIMIZATIONS.md
    ├── OPTIMIZATION_SUMMARY.md
    ├── OPTIMIZED_INTEGRATION_REPORT.md
    ├── PATCHIFY_REMOVAL_SUMMARY.md
    ├── SHAPE_FIX_FINAL.md
    ├── INTEGRATION_CHECK_REPORT.md
    ├── INTEGRATION_COMPLETE.md
    ├── DATASET_SETUP.md
    ├── EXECUTION_SEQUENCE.md
    ├── VISUALIZATION_GUIDE.md
    ├── VMAMBA_PANSHARP_README.md
    ├── QUICK_START_MEMORY_SAFE.md
    └── README.md
```

---

## V1 Known Issues (fixed in V2)

| Issue | Root Cause | Fix in V2 |
|-------|-----------|-----------|
| PixelShuffle checkerboard artefacts | Sub-pixel periodic grouping | Residual Upsampling |
| Fixed Sobel edges | Non-adaptive to scene content | Learnable HF Injection |
| CrossAttention O(N²) memory | QK⊤ dense attention matrix | Cross-Mamba O(N) |
| OOM during fusion at full HR | Attention over 256×256 tokens | SSM row/column scan |
| NaN gradients in training | Unstable sqrt in Sobel | Learnable conv replaced |
| PixelShuffle channel explosion | 64×C intermediate channels | 1×1 spectral aggregation |

---

## V1 Quick Commands (still runnable from project root)

```bash
# V1 model forward pass test
python vmamba_pansharp.py

# V1 baseline comparison (CNN / UNet / Transformer)
python compare_models.py --dataset pavia --epochs 5

# V1 training
python train_vmamba_pansharp.py --dataset pavia --epochs 30

# V1 evaluation
python evaluate_vmamba_pansharp.py \
    --checkpoint version1/experiments/vmamba_pansharp_pavia_20251222_040706/checkpoints/best.pth
```

---

## Datasets Used in V1

| Dataset | Bands | Spatial | Used for |
|---------|-------|---------|---------|
| Pavia University | 103 | 610×610 | All V1 runs |
| Chikusei | 128 | 2517×2335 | 2 early runs only |

---

*Version 1 archived on completion of Version 2 development.*
*See `README_V2.md` in the project root for the current improved architecture.*
