# Hyperspectral Pansharpening – M.Tech Thesis
## Improved VMamba vs Old VMamba on Chikusei Dataset

---

## Project Structure

```
Mtech project/
├── vmamba_pansharp.py           ← Old model (CrossAttention + Sobel + PixelShuffle)
├── vmamba_pansharp_improved.py  ← NEW improved model (CrossMamba + LH-Injection + ResidualUpsample)
├── baseline_models.py           ← Factory: create_vmamba_model('old'|'improved', ...)
├── dataset_loader.py            ← Chikusei / Pavia dataset loading
├── dataset_loader_overlap.py    ← Overlap-patch dataloader with train/val/test split
├── loss_functions.py            ← L1 + SAM + Edge + SSIM composite loss
├── run_experiment.py            ← Main runner (train / test / compare / visualize)
│
├── testing/
│   └── test_chikusei.py         ← Evaluate both models, print metrics table
├── comparison/
│   └── compare_vmamba.py        ← Train + compare, save plots + JSON
├── visualization/
│   └── visualize_results.py     ← Visual output figures
│
├── version1/                    ← Previous results (CNN/UNet/Transformer era)
│   ├── comparison_results/
│   ├── experiments/
│   └── visualizations/
│
└── chikusei/
    └── chikusei.mat             ← Dataset (128 bands, 2517×2335 pixels)
```

---

## Three Key Improvements

| # | Component | Old | Improved |
|---|-----------|-----|----------|
| 1 | **Fusion** | CrossAttention O(N²) | **Cross-Mamba** O(N) — PAN drives SSM params |
| 2 | **Edge extraction** | Fixed Sobel filter | **Learnable HF Injection** — `HSI + σ(Conv1×1) * Conv3×3(PAN)` |
| 3 | **Upsampling** | PixelShuffle (checkerboard) | **Residual Upsample** — `Bilinear + Conv + residual` |

---

## OOM-Safe Configuration (CPU & low-memory GPU)

> **This machine runs CPU-only** (PyTorch 2.9.1+cpu).
> All defaults below are safe for 8 GB RAM.
> If you have a GPU (≥ 6 GB VRAM), see the GPU section at the bottom.

| Parameter | CPU-safe value | Notes |
|-----------|---------------|-------|
| `--patch_size` | **8** | LR patch → HR = 8×4 = **32×32** |
| `--scale` | **4** | Standard pansharpening scale |
| `--batch_size` | **1** | 1 image per step |
| `--d_model` | **32** | Feature channels |
| `--epochs` | **5** | Enough for a quick sanity check |
| `--num_blocks` | `[1,1,1,1]` | Lightest backbone |

---

## Step-by-step Commands

### 1. Verify both models load correctly
```bash
python baseline_models.py
```
Expected output:
```
Variant: old      Parameters: ~2.2M   Output shape: (2, 128, 64, 64)  [OK]
Variant: improved Parameters: ~1.0M   Output shape: (2, 128, 64, 64)  [OK]
```

### 2. Quick test (no training, random weights)
```bash
python testing/test_chikusei.py
```
Runs inference on 20 test patches with random weights.
Results saved to `testing/results/test_results.json`.

### 3. Quick comparison training (5 epochs, CPU-safe)
```bash
python run_experiment.py compare --epochs 5 --batch_size 1 --patch_size 8 --d_model 32
```
Trains both models for 5 epochs, then:
- Saves checkpoints → `comparison/checkpoints/old_best.pth` and `improved_best.pth`
- Saves 6-panel plot  → `comparison/plots/comparison_YYYYMMDD_HHMMSS.png`
- Saves JSON summary  → `comparison/results/summary.json`

### 4. Full training (30 epochs, CPU — takes ~1–2 hours)
```bash
python run_experiment.py train --epochs 30 --batch_size 1 --patch_size 8 --d_model 32
```

### 5. Test with trained checkpoints
```bash
python run_experiment.py test \
  --checkpoint_old      comparison/checkpoints/old_best.pth \
  --checkpoint_improved comparison/checkpoints/improved_best.pth
```

### 6. Visualize results
```bash
python run_experiment.py visualize \
  --checkpoint_old      comparison/checkpoints/old_best.pth \
  --checkpoint_improved comparison/checkpoints/improved_best.pth \
  --patch_idx 3
```
Output saved to `visualization/outputs/comparison_YYYYMMDD_HHMMSS.png`.

---

## Run Everything in One Shot

```bash
# Step 1: train
python run_experiment.py train --epochs 30 --batch_size 1 --patch_size 8 --d_model 32

# Step 2: test
python run_experiment.py test \
  --checkpoint_old comparison/checkpoints/old_best.pth \
  --checkpoint_improved comparison/checkpoints/improved_best.pth

# Step 3: visualize
python run_experiment.py visualize \
  --checkpoint_old comparison/checkpoints/old_best.pth \
  --checkpoint_improved comparison/checkpoints/improved_best.pth
```

---

## GPU Configuration (if available)

| Parameter | GPU ≥ 6 GB | GPU ≥ 12 GB |
|-----------|-----------|------------|
| `--patch_size` | 16 | 32 |
| `--batch_size` | 2 | 4 |
| `--d_model` | 64 | 64 |
| `--epochs` | 30 | 50 |

```bash
# Example for 8 GB GPU
python run_experiment.py compare \
  --epochs 30 --batch_size 2 --patch_size 16 --d_model 64
```

---

## Output Files Reference

| Script | Output Location |
|--------|----------------|
| `compare_vmamba.py` | `comparison/plots/*.png` · `comparison/results/summary.json` · `comparison/checkpoints/*.pth` |
| `test_chikusei.py` | `testing/results/test_results.json` |
| `visualize_results.py` | `visualization/outputs/comparison_*.png` |

---

## Previous Results (CNN / UNet / Transformer era)

All old results are archived in `version1/`:

```
version1/
├── comparison_results/   ← 18 old Pavia comparison runs
├── experiments/          ← 15 old training experiments
└── visualizations/       ← Pavia overview + Wald's protocol figures
```

---

## Metrics Used

| Metric | Formula | Better when |
|--------|---------|-------------|
| **PSNR** | `20·log10(1/RMSE)` | Higher ↑ |
| **SAM** | Spectral angle (degrees) | Lower ↓ |
| **ERGAS** | Relative global error | Lower ↓ |

---

## Expected Improvement (after training)

Based on the 3 architectural improvements:

| Metric | Old VMamba | Improved VMamba |
|--------|-----------|----------------|
| PSNR   | baseline  | +0.5 – 1.5 dB |
| SAM    | baseline  | −0.5 – 1.0°   |
| ERGAS  | baseline  | lower          |
| Params | ~2.2 M    | ~1.0 M (−55%)  |
| Fusion complexity | O(N²) | O(N) |

---

## Troubleshooting

**`ModuleNotFoundError: einops`**
```bash
pip install einops
```

**`ModuleNotFoundError: h5py`**
```bash
pip install h5py
```

**`ModuleNotFoundError: tqdm`**
```bash
pip install tqdm
```

**`RuntimeError: CUDA out of memory`**
Reduce `--patch_size` to 8 and `--d_model` to 32, set `--batch_size 1`.

**`ValueError: not enough values to unpack`**
Make sure you are running scripts from the project root directory.

**Slow CPU training**
The CrossMamba sequential scan is O(N) but runs in Python loops on CPU.
For faster training, use a GPU or reduce `--patch_size` to 4 (HR = 16×16).

---

*M.Tech Thesis — Hyperspectral Image Pansharpening using VMamba*
*Dataset: Chikusei (128 bands, 2517×2335, Headwall Photonics VNIR)*
