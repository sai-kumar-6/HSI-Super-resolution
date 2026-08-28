# QUICKSTART — VMamba Hyperspectral Pansharpening
### M.Tech Thesis | Chikusei Dataset | CPU-safe

---

## Prerequisites

```bash
pip install torch torchvision einops scipy h5py opencv-python matplotlib seaborn tqdm
```

**Dataset required:** `chikusei/chikusei.mat` (place in project root)

---

## Project Structure

```
Mtech project/
├── QUICKSTART.md                  ← THIS FILE
├── vmamba_pansharp.py             ← V1 (Old) model
├── vmamba_pansharp_improved.py    ← V2 (Improved) model
├── baseline_models.py             ← Model factory (old / improved)
├── dataset_loader_overlap.py      ← Data loading + patch extraction
├── loss_functions.py              ← Loss + metrics (PSNR, SAM, ERGAS)
├── run_experiment.py              ← MAIN RUNNER
│
├── comparison/compare_vmamba.py   ← Train & compare both models
├── testing/test_chikusei.py       ← Evaluate on test patches
├── visualization/visualize_results.py  ← Generate output figures
│
├── chikusei/chikusei.mat          ← Dataset (put here)
└── version1/                      ← Archived V1 results
```

---

## OOM-Safe Settings (CPU / 8 GB RAM)

| Setting        | Value  | Why                                  |
|----------------|--------|--------------------------------------|
| `--patch_size` | 32     | HR=32×32, LR=8×8 (min for backbone) |
| `--batch_size` | 1      | Minimal memory                       |
| `--d_model`    | 32     | ~677K params (improved) / ~1.8M (old)|
| `--epochs`     | 5      | Quick smoke test                     |
| MAX_ROWS       | 512    | Loads 512×2335×128 ≈ 584 MB float32  |

> DO NOT use `--patch_size` below 32 — LR patches become 2×2 which breaks the backbone.

---

## Step-by-Step Execution

### Step 0 — Verify models load correctly

```bash
python baseline_models.py
```

Expected output:
```
Variant: old       Parameters: ~1,840,352   Output: (1, 128, 32, 32)  [OK]
Variant: improved  Parameters:   ~677,288   Output: (1, 128, 32, 32)  [OK]
```

---

### Step 1 — Smoke test (no training, random weights)

```bash
python testing/test_chikusei.py
```

- Loads 20 test patches from Chikusei
- Runs both models with random weights
- Prints PSNR / SAM / ERGAS comparison table
- Saves `testing/results/test_results.json`

> Runs in ~1–2 minutes on CPU.

---

### Step 2 — Train & compare (5 epochs)

```bash
python run_experiment.py compare \
  --epochs 5 \
  --batch_size 1 \
  --patch_size 32 \
  --d_model 32
```

Outputs saved to:
- `comparison/checkpoints/old_best.pth`
- `comparison/checkpoints/improved_best.pth`
- `comparison/plots/comparison_YYYYMMDD.png`
- `comparison/results/summary.json`

> Estimated time: ~30–60 min on CPU for 5 epochs.

---

### Step 3 — Full training (30 epochs)

```bash
python run_experiment.py train \
  --epochs 30 \
  --batch_size 1 \
  --patch_size 32 \
  --d_model 32
```

> Estimated time: ~3–6 hours on CPU.

---

### Step 4 — Evaluate with trained weights

```bash
python run_experiment.py test \
  --checkpoint_old      comparison/checkpoints/old_best.pth \
  --checkpoint_improved comparison/checkpoints/improved_best.pth
```

Prints side-by-side metrics table:
```
--------------------------------------------------
Metric             Old VMamba    Improved VMamba   Delta
--------------------------------------------------
PSNR  (dB)             XX.XX              XX.XX  +X.XXX ↑
SAM   (deg)             X.XX               X.XX  -X.XXX ↑
ERGAS                   X.XX               X.XX  -X.XXX ↑
Inference (ms/img)      X.XX               X.XX
--------------------------------------------------
Params (M)              1.84               0.68
--------------------------------------------------
```

---

### Step 5 — Visualize results

```bash
python run_experiment.py visualize \
  --checkpoint_old      comparison/checkpoints/old_best.pth \
  --checkpoint_improved comparison/checkpoints/improved_best.pth \
  --patch_idx 3
```

Saves `visualization/outputs/comparison_YYYYMMDD_HHMMSS.png` with:
- RGB false-colour outputs (Old vs Improved vs Ground Truth)
- Per-pixel error maps
- Spectral profile comparison (band 50)

---

## Run from new_model/ folder (self-contained copy)

```bash
cd new_model

# Compare
python run_experiment.py compare --epochs 5 --batch_size 1 --patch_size 32 --d_model 32

# Test
python run_experiment.py test \
  --checkpoint_old      comparison/checkpoints/old_best.pth \
  --checkpoint_improved comparison/checkpoints/improved_best.pth

# Visualize
python run_experiment.py visualize \
  --checkpoint_old      comparison/checkpoints/old_best.pth \
  --checkpoint_improved comparison/checkpoints/improved_best.pth
```

Dataset path is resolved automatically as `../chikusei/chikusei.mat`.

---

## What Changed: V1 → V2

| # | Component        | V1 (Old)                  | V2 (Improved)             | Benefit             |
|---|------------------|---------------------------|---------------------------|---------------------|
| 1 | Fusion           | CrossAttention — O(N²)    | Cross-Mamba — O(N)        | Less memory, faster |
| 2 | Edge extraction  | Fixed Sobel filter        | Learnable HF Injection    | Adapts to scene     |
| 3 | Upsampling       | PixelShuffle (checkerboard)| Residual Upsample        | No artefacts        |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `RuntimeError: Kernel size can't be greater than actual input` | Use `--patch_size 32` (not 8) |
| `Unable to allocate X GiB` | Already fixed — MAX_ROWS=512 limits load to 584 MB |
| `ModuleNotFoundError: einops` | `pip install einops` |
| `ModuleNotFoundError: h5py` | `pip install h5py` |
| `not enough values to unpack` | Run from project root directory, not a subfolder |
| Validation set empty | Normal for very small datasets — ignore warning |

---

## Metrics Reference

| Metric  | Formula                          | Better  |
|---------|----------------------------------|---------|
| PSNR    | `20·log₁₀(1/RMSE)` dB           | Higher ↑ |
| SAM     | Spectral angle in degrees        | Lower ↓  |
| ERGAS   | `100·scale·√(mean(MSE/μ²))`     | Lower ↓  |

---

*See `README_V2.md` for full architecture details.*
*See `version1/README_V1.md` for the previous version.*
