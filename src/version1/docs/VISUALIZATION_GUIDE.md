# Visualization Guide

Complete guide to visualizing datasets, Wald's protocol, and model outputs.

## 📊 Available Visualizations

### 1. Dataset Visualization
Comprehensive analysis of Pavia and Chikusei datasets

### 2. Wald's Protocol Visualization
Step-by-step visualization of synthetic data generation

### 3. Model Output Visualization
Detailed analysis of VMamba-Pansharp predictions

---

## 🎨 1. Dataset Visualization

### Command

```bash
python visualize_dataset.py
```

### What It Shows

#### For Each Dataset (Pavia/Chikusei):

**Overview Visualization (10 subplots):**

1. **RGB Composite** - Natural color image using 3 bands
2. **False Color Composite** - Enhanced visualization
3. **Single Band** - Near-IR band visualization
4. **PAN Simulation** - Averaged spectral bands
5. **Spectral Signatures** - 10 random pixel spectra
6. **Mean Spectrum** - Average across all pixels with std
7. **Band Statistics** - Mean reflectance per band
8. **Correlation Matrix** - Inter-band correlations
9. **Data Distribution** - Histogram of pixel values
10. **Dataset Info** - Dimensions, statistics, memory

**Output Files:**
```
visualizations/
├── pavia_university_overview.png
└── chikusei_overview.png
```

### Example Output

```
==================================================================
HYPERSPECTRAL DATASET VISUALIZATION
==================================================================

==================================================================
PAVIA UNIVERSITY DATASET
==================================================================

Loading PAVIA dataset...
✓ Loaded Pavia University: (610, 340, 103)

Generating dataset overview for Pavia University...
  Creating RGB composite...
  Creating false color composite...
  Creating single band visualization...
  Creating PAN simulation...
  Plotting spectral signatures...
    Sampling pixels: 100%|██████████| 10/10
  Computing mean spectrum...
  Creating band statistics...
    Computing stats: 100%|██████████| 103/103
  Computing band correlations...
  Computing data distribution...

✓ Saved to: visualizations/pavia_university_overview.png
```

---

## 🔬 2. Wald's Protocol Visualization

### Command

```bash
python visualize_dataset.py  # Automatically includes Wald's protocol
```

Or test separately:
```bash
python test_walds_protocol.py
```

### What It Shows

**Wald's Protocol Steps (12 subplots):**

**Row 1: Ground Truth (HR-HSI)**
1. RGB Composite
2. Red Band
3. Green Band
4. Blue Band

**Row 2: Degraded (LR-HSI)**
5. LR RGB Composite (152×85 for Pavia)
6. LR Red Band
7. LR Green Band
8. LR Blue Band

**Row 3: Analysis**
9. HR-PAN (averaged bands)
10. Spectral Comparison (HR vs LR)
11. Protocol Diagram

**Shows:**
- ✓ Gaussian blur effect (σ=1.0)
- ✓ 4× decimation result
- ✓ Spectral averaging for PAN
- ✓ Dimension changes at each step

**Output Files:**
```
visualizations/
├── pavia_university_walds_protocol.png
├── chikusei_walds_protocol.png
└── walds_protocol_test.png  # From test script
```

### Example Output

```
Visualizing Wald's Protocol for Pavia University...
  Step 1: Ground Truth (HR-HSI)
  Step 2: Generating LR-HSI (Gaussian blur + decimation)...
  Step 3: Generating HR-PAN (spectral averaging)...

✓ Saved to: visualizations/pavia_university_walds_protocol.png
```

---

## 🎯 3. Model Output Visualization

### Command

```bash
python visualize_model_output.py \
    --checkpoint experiments/YOUR_EXP/checkpoints/best.pth \
    --dataset pavia \
    --num_samples 5 \
    --output_dir visualizations
```

### Arguments

- `--checkpoint`: Path to trained model checkpoint **(required)**
- `--dataset`: Dataset name ('pavia' or 'chikusei'), default: 'pavia'
- `--num_samples`: Number of samples to visualize, default: 5
- `--output_dir`: Output directory, default: 'visualizations'

### What It Shows

**For Each Sample (15 subplots):**

**Row 1: Inputs and Outputs**
1. LR-HSI (upsampled for display)
2. HR-PAN input
3. VMamba Prediction with PSNR
4. Ground Truth
5. RGB Error Map

**Row 2: Single Band Analysis**
6. LR Band (upsampled)
7. HR-PAN detail
8. Predicted Band
9. Ground Truth Band
10. Band Error Map

**Row 3: Detailed Analysis**
11. Spectral Signatures (5 pixels, GT vs Pred)
12. Per-Band Reconstruction Error (bar chart)
13. Quality Metrics Summary (PSNR, SAM, ERGAS)

**Output Files:**
```
visualizations/
├── model_output_sample_0.png
├── model_output_sample_1.png
├── model_output_sample_2.png
├── model_output_sample_3.png
└── model_output_sample_4.png
```

### Example Output

```
Using device: cuda

Loading pavia dataset...
Pavia Dataset loaded:
  Training shape: (488, 340, 103)
  Validation shape: (122, 340, 103)
  Spectral bands: 103

Dataloader created:
  Validation samples: 23

Loading model from experiments/pavia_100epochs/checkpoints/best.pth...
✓ Loaded model from epoch 99

Generating visualizations for 5 samples...
Processing samples: 100%|███████████████| 5/5

✓ Saved 5 visualizations to visualizations/

==================================================================
✓ VISUALIZATION COMPLETE
==================================================================

Generated visualizations in: visualizations/
  - model_output_sample_*.png
==================================================================
```

---

## 📈 Quality Metrics in Visualizations

### PSNR (Peak Signal-to-Noise Ratio)
- **Range**: 0-50+ dB (higher is better)
- **Interpretation**:
  - >35 dB: Excellent
  - 30-35 dB: Good
  - <30 dB: Poor

### SAM (Spectral Angle Mapper)
- **Range**: 0-90° (lower is better)
- **Interpretation**:
  - <3°: Excellent spectral preservation
  - 3-5°: Good spectral preservation
  - >5°: Poor spectral preservation

### ERGAS
- **Range**: 0-10+ (lower is better)
- **Interpretation**:
  - <2: Excellent
  - 2-4: Good
  - >4: Poor

---

## 🎨 Complete Visualization Workflow

### Step 1: Visualize Dataset

```bash
# Visualize both Pavia and Chikusei datasets
python visualize_dataset.py
```

**Outputs:**
- Dataset overviews (RGB, bands, spectra, statistics)
- Wald's protocol demonstrations

### Step 2: Verify Protocol

```bash
# Test Wald's protocol implementation
python test_walds_protocol.py
```

**Outputs:**
- Protocol verification results
- Dimension checks
- Test visualization

### Step 3: Train Model

```bash
# Train VMamba-Pansharp
python train_vmamba_pansharp.py \
    --dataset pavia \
    --epochs 100 \
    --batch_size 8
```

### Step 4: Visualize Model Outputs

```bash
# Visualize predictions from trained model
python visualize_model_output.py \
    --checkpoint experiments/pavia_vmamba_100ep/checkpoints/best.pth \
    --dataset pavia \
    --num_samples 10
```

**Outputs:**
- Detailed prediction analysis for 10 samples
- Input-output comparisons
- Error maps
- Spectral signature comparisons
- Per-band error analysis

---

## 📊 Visualization Examples

### Dataset Overview

Shows:
- ✅ RGB and false-color composites
- ✅ Individual band visualizations
- ✅ Spectral signatures from multiple pixels
- ✅ Statistical distributions
- ✅ Inter-band correlations
- ✅ Complete dataset metadata

### Wald's Protocol

Shows:
- ✅ Original HR-HSI (ground truth)
- ✅ Degraded LR-HSI (blurred + downsampled)
- ✅ Generated HR-PAN (spectral average)
- ✅ Side-by-side band comparisons
- ✅ Spectral signature preservation
- ✅ Dimension transformations (610×340×103 → 152×85×103)

### Model Output

Shows:
- ✅ Input LR-HSI and HR-PAN
- ✅ VMamba prediction
- ✅ Ground truth comparison
- ✅ Error visualization (RGB and per-band)
- ✅ Spectral accuracy (SAM metric)
- ✅ Per-pixel spectral comparisons
- ✅ Quality metrics (PSNR, SAM, ERGAS)

---

## 💡 Tips for Best Results

### High-Quality Figures

All visualizations are saved at **300 DPI** for publication quality.

To customize:
```python
# In visualization scripts, modify:
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Current
plt.savefig(save_path, dpi=600, bbox_inches='tight')  # Higher quality
```

### Color Maps

- **RGB composites**: Natural colors
- **Single bands**: 'viridis' (perceptually uniform)
- **PAN images**: 'gray'
- **Error maps**: 'hot' (highlights errors)

### Batch Visualization

To visualize many samples:
```bash
python visualize_model_output.py \
    --checkpoint PATH \
    --num_samples 20  # Visualize 20 samples
```

---

## 🔍 Understanding Visualizations

### RGB Composites

**Bands used for RGB:**
- Pavia (103 bands): [72, 52, 21] ≈ [R, G, B]
- Chikusei (128 bands): [90, 64, 26] ≈ [R, G, B]

These approximate true color by selecting bands at:
- Red: ~70% through spectrum
- Green: ~50% through spectrum
- Blue: ~20% through spectrum

### Error Maps

**Color interpretation:**
- Dark blue/black: Low error (good)
- Yellow/orange: Medium error
- Red/white: High error (poor)

### Spectral Signatures

Shows reflectance across all bands for a single pixel.

**Good match:**
- Prediction closely follows ground truth
- Low SAM (< 3°)
- Similar peak positions

**Poor match:**
- Large deviations
- High SAM (> 5°)
- Different spectral shape

---

## 📁 Output Directory Structure

```
visualizations/
├── Dataset Visualizations
│   ├── pavia_university_overview.png
│   ├── pavia_university_walds_protocol.png
│   ├── chikusei_overview.png
│   └── chikusei_walds_protocol.png
│
└── Model Output Visualizations
    ├── model_output_sample_0.png
    ├── model_output_sample_1.png
    ├── model_output_sample_2.png
    ├── ...
    └── model_output_sample_N.png
```

---

## 🎓 Research Use

### For Papers/Presentations

1. **Dataset Description** → Use dataset overview
2. **Methodology** → Use Wald's protocol visualization
3. **Results** → Use model output visualizations
4. **Comparison** → Use side-by-side model comparisons

### Figure Captions Example

```
Figure 1: Pavia University hyperspectral dataset overview showing
(a) RGB composite, (b) false-color composite, (c) near-IR band,
(d) simulated PAN, (e) spectral signatures, and (f) mean spectrum
with standard deviation.

Figure 2: Wald's protocol for synthetic training data generation.
Top row shows ground truth HR-HSI, middle row shows degraded LR-HSI
after Gaussian blur (σ=1.0) and 4× decimation, bottom row shows
generated HR-PAN and spectral comparison.

Figure 3: VMamba-Pansharp prediction results. (a) Input LR-HSI,
(b) Input HR-PAN, (c) Predicted HR-HSI (PSNR: 35.2dB, SAM: 3.8°),
(d) Ground truth, (e) Error map showing mean absolute error per pixel.
```

---

## 🐛 Troubleshooting

### Issue: Visualization script fails

**Check dataset:**
```bash
ls -la pavia/PaviaU.mat
ls -la chikusei/HyperspecVNIR_Chikusei_20140729.mat
```

### Issue: Model visualization fails

**Check checkpoint:**
```bash
ls -la experiments/YOUR_EXP/checkpoints/best.pth
```

### Issue: Out of memory

**Reduce samples:**
```bash
python visualize_model_output.py --num_samples 3
```

---

## 📚 Related Documentation

- **EXECUTION_GUIDE.md** - How to train models
- **VMAMBA_PANSHARP_README.md** - Architecture details
- **WALDS_PROTOCOL.md** - Protocol documentation

---

**All visualizations are publication-ready at 300 DPI!** 📊
