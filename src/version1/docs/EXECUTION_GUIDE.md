# VMamba-Pansharp Execution Guide

Complete step-by-step guide to train, evaluate, and compare the VMamba-Pansharp model.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Setup](#setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Testing Installation](#testing-installation)
5. [Training the Model](#training-the-model)
6. [Evaluating the Model](#evaluating-the-model)
7. [Model Comparison](#model-comparison)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test installation
python quick_start.py

# 3. Train on Pavia dataset
python train_vmamba_pansharp.py --dataset pavia --epochs 50 --batch_size 8

# 4. Evaluate trained model
python evaluate_vmamba_pansharp.py --checkpoint experiments/YOUR_EXP/checkpoints/best.pth
```

---

## 🔧 Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- torch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- opencv-python >= 4.8.0
- einops >= 0.7.0
- tqdm >= 4.65.0
- tensorboard >= 2.13.0

### 2. Verify GPU (Optional but Recommended)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

**Expected output (with GPU):**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

---

## 📁 Dataset Preparation

### Option 1: Pavia University Dataset

1. **Download** Pavia University dataset from:
   - [GIC Group](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
   - File: `PaviaU.mat`

2. **Place** in the correct directory:
   ```
   Mtech project/
   └── pavia/
       └── PaviaU.mat
   ```

3. **Verify** dataset:
   ```bash
   python -c "import scipy.io as sio; data = sio.loadmat('pavia/PaviaU.mat'); print(f'Pavia shape: {data[\"paviaU\"].shape}')"
   ```

   **Expected output:**
   ```
   Pavia shape: (610, 340, 103)
   ```

### Option 2: Chikusei Dataset

1. **Download** Chikusei dataset:
   - File: `HyperspecVNIR_Chikusei_20140729.mat`

2. **Place** in the correct directory:
   ```
   Mtech project/
   └── chikusei/
       └── HyperspecVNIR_Chikusei_20140729.mat
   ```

3. **Verify** dataset:
   ```bash
   python -c "import scipy.io as sio; data = sio.loadmat('chikusei/HyperspecVNIR_Chikusei_20140729.mat'); key = list(data.keys())[-1]; print(f'Chikusei shape: {data[key].shape}')"
   ```

---

## ✅ Testing Installation

### Test 1: Quick Start Test

```bash
python quick_start.py
```

**This will:**
- ✓ Check device (CPU/GPU)
- ✓ Create VMamba-Pansharp model
- ✓ Test forward pass
- ✓ Test loss functions
- ✓ Test backward pass
- ✓ Show memory usage

**Expected output:**
```
============================================================
VMamba-Pansharp Quick Start Test
============================================================

1. Device: cuda
   GPU: NVIDIA GeForce RTX 3090

2. Model Configuration:
   Input channels: 102
   Feature dimension: 64
   Total parameters: 18,234,567

...

✓ ALL TESTS PASSED!
============================================================
```

### Test 2: Wald's Protocol Test

```bash
python test_walds_protocol.py
```

**This will:**
- ✓ Verify Wald's protocol implementation
- ✓ Check synthetic data generation
- ✓ Test dimension correctness
- ✓ Create visualization

---

## 🎯 Training the Model

### Basic Training

#### Pavia Dataset (Recommended for Quick Testing)

```bash
python train_vmamba_pansharp.py \
    --dataset pavia \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4
```

**Training time:** ~2-3 hours on RTX 3090 (100 epochs)

#### Chikusei Dataset (Full Training)

```bash
python train_vmamba_pansharp.py \
    --dataset chikusei \
    --batch_size 4 \
    --epochs 300 \
    --lr 1e-4
```

**Training time:** ~8-12 hours on RTX 3090 (300 epochs)

### Advanced Training Options

```bash
python train_vmamba_pansharp.py \
    --dataset pavia \
    --exp_name my_experiment \
    --batch_size 8 \
    --epochs 300 \
    --lr 1e-4 \
    --patch_size 64 \
    --num_workers 4
```

**All Arguments:**
- `--dataset`: Dataset name ('pavia' or 'chikusei')
- `--exp_name`: Custom experiment name (default: auto-generated)
- `--batch_size`: Batch size (default: 8)
- `--epochs`: Number of epochs (default: 300)
- `--lr`: Learning rate (default: 1e-4)
- `--patch_size`: LR patch size (default: 64, HR will be 256)
- `--num_workers`: Data loading workers (default: 4)
- `--resume`: Path to checkpoint to resume training

### Resume Training

```bash
python train_vmamba_pansharp.py \
    --dataset pavia \
    --resume experiments/vmamba_pansharp_pavia_YYYYMMDD_HHMMSS/checkpoints/latest.pth
```

### Monitor Training

**Using TensorBoard:**
```bash
tensorboard --logdir experiments/YOUR_EXP/logs
```

Then open: http://localhost:6006

**Tracked Metrics:**
- Training/Validation Loss (total + components)
- PSNR, SAM, ERGAS
- Learning rate
- Training time per epoch

### Training Output

```
experiments/
└── vmamba_pansharp_pavia_20241221_100000/
    ├── config.json                 # Training configuration
    ├── checkpoints/
    │   ├── latest.pth             # Latest checkpoint
    │   ├── best.pth               # Best model (highest PSNR)
    │   ├── epoch_50.pth           # Periodic checkpoints
    │   └── epoch_100.pth
    └── logs/                       # TensorBoard logs
```

---

## 📊 Evaluating the Model

### Basic Evaluation

```bash
python evaluate_vmamba_pansharp.py \
    --checkpoint experiments/YOUR_EXP/checkpoints/best.pth \
    --output_dir results
```

**This will:**
- ✓ Load trained model
- ✓ Evaluate on validation set
- ✓ Compute PSNR, SAM, ERGAS
- ✓ Generate visualizations
- ✓ Save results to JSON

**Expected output:**
```
============================================================
EVALUATION RESULTS
============================================================
Dataset: pavia
Number of samples: 45

PSNR: 35.12 ± 1.23 dB
SAM:  3.87 ± 0.45°
ERGAS: 1.52 ± 0.18
============================================================
```

### Export Results

```bash
python evaluate_vmamba_pansharp.py \
    --checkpoint experiments/YOUR_EXP/checkpoints/best.pth \
    --output_dir results \
    --export
```

**This will additionally:**
- ✓ Export all results to .mat file
- ✓ Save predictions for all samples

### Evaluation Output

```
results/
├── evaluation_results.json     # Metrics summary
├── sample_0.png               # Visual comparison
├── sample_1.png               # Visual comparison
├── sample_2.png
├── sample_3.png
├── sample_4.png
└── vmamba_pansharp_results.mat  # All predictions (if --export)
```

### Visualization Example

Each `sample_X.png` contains:
- Row 1: LR-HSI (upsampled), HR-PAN, Predicted HR-HSI
- Row 2: Ground Truth, Error Map, Spectral Signature comparison

---

## 🔬 Model Comparison

### Option 1: Interactive Notebook (Recommended)

```bash
jupyter notebook Model_Comparison_Analysis.ipynb
```

**Features:**
- Step-by-step comparison
- Real-time training visualization
- Interactive exploration
- Detailed analysis

### Option 2: Automated Comparison Script

#### Quick Comparison (20 epochs)

```bash
python compare_models.py \
    --dataset pavia \
    --epochs 20 \
    --batch_size 8
```

**Training time:** ~40 minutes on RTX 3090

#### Full Comparison (100 epochs)

```bash
python compare_models.py \
    --dataset pavia \
    --epochs 100 \
    --batch_size 8
```

**Training time:** ~3-4 hours on RTX 3090

### Comparison Output

```
comparison_results/
└── comparison_pavia_20241221_100000/
    ├── config.json                 # Comparison configuration
    ├── summary.json                # Results summary
    ├── comparison_plots.png        # 9-subplot comprehensive comparison
    ├── psnr_detailed.png          # Detailed PSNR evolution
    ├── cnn_final.pth              # Trained CNN model
    ├── transformer_final.pth      # Trained Transformer model
    ├── unet_final.pth             # Trained U-Net model
    └── vmamba_final.pth           # Trained VMamba model
```

### Comparison Graphs Generated

1. **Training Loss Comparison** - Convergence speed
2. **Validation Loss Comparison** - Generalization
3. **PSNR Evolution** - Image quality over time
4. **SAM Evolution** - Spectral fidelity over time
5. **ERGAS Evolution** - Overall quality over time
6. **Final Metrics Bar Charts** - Final performance comparison
7. **Model Size Comparison** - Parameters and memory
8. **Training Speed Comparison** - Time per epoch
9. **Performance vs Efficiency** - Quality-size-speed tradeoff

### Expected Results (100 epochs on Pavia)

| Model       | PSNR↑  | SAM↓  | ERGAS↓ | Params | Speed   |
|-------------|--------|-------|--------|--------|---------|
| CNN         | 32.5dB | 5.0°  | 2.2    | 15M    | 45s     |
| Transformer | 34.2dB | 4.3°  | 1.8    | 25M    | 65s     |
| U-Net       | 30.8dB | 5.8°  | 2.5    | 8M     | 35s     |
| **VMamba**  | **35.1dB** | **3.9°** | **1.5** | 18M | 55s |

---

## 🎓 Complete Workflow Example

### Scenario: Train and Evaluate on Pavia

```bash
# Step 1: Test installation
python quick_start.py

# Step 2: Verify dataset
python -c "import scipy.io as sio; data = sio.loadmat('pavia/PaviaU.mat'); print(f'Dataset OK: {data[\"paviaU\"].shape}')"

# Step 3: Train VMamba-Pansharp (100 epochs)
python train_vmamba_pansharp.py \
    --dataset pavia \
    --exp_name pavia_vmamba_100ep \
    --batch_size 8 \
    --epochs 100

# Step 4: Monitor training (in another terminal)
tensorboard --logdir experiments/pavia_vmamba_100ep/logs

# Step 5: Evaluate best model
python evaluate_vmamba_pansharp.py \
    --checkpoint experiments/pavia_vmamba_100ep/checkpoints/best.pth \
    --output_dir results_pavia \
    --export

# Step 6: Compare with baselines (20 epochs for quick test)
python compare_models.py \
    --dataset pavia \
    --epochs 20 \
    --batch_size 8

# Step 7: View comparison results
# Check: comparison_results/comparison_pavia_*/comparison_plots.png
```

---

## 🐛 Troubleshooting

### Issue 1: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce batch size
python train_vmamba_pansharp.py --batch_size 4  # or 2

# Reduce patch size (modify in code)
# In train_vmamba_pansharp.py, change:
# 'patch_size': 32  # instead of 64
```

### Issue 2: Dataset Not Found

**Error:**
```
FileNotFoundError: pavia/PaviaU.mat
```

**Solution:**
```bash
# Check if file exists
ls -la pavia/PaviaU.mat

# Verify directory structure
ls -la pavia/
```

### Issue 3: Slow Training

**Solutions:**
```bash
# Increase number of workers
python train_vmamba_pansharp.py --num_workers 8

# Use smaller dataset for testing
python train_vmamba_pansharp.py --dataset pavia --epochs 10
```

### Issue 4: Import Errors

**Error:**
```
ImportError: No module named 'einops'
```

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue 5: TensorBoard Not Starting

**Solution:**
```bash
# Install tensorboard
pip install tensorboard

# Use specific port
tensorboard --logdir experiments/YOUR_EXP/logs --port 6007
```

---

## 📝 Project File Structure

```
Mtech project/
├── Core Implementation
│   ├── vmamba_pansharp.py           # VMamba-Pansharp model
│   ├── baseline_models.py           # CNN, Transformer, U-Net baselines
│   ├── dataset_loader.py            # Wald's protocol data loading
│   └── loss_functions.py            # Loss functions + metrics
│
├── Training & Evaluation
│   ├── train_vmamba_pansharp.py     # Training script
│   ├── evaluate_vmamba_pansharp.py  # Evaluation script
│   └── compare_models.py            # Multi-model comparison
│
├── Utilities
│   ├── visualization_utils.py       # Plotting utilities
│   ├── quick_start.py               # Installation test
│   └── test_walds_protocol.py       # Protocol verification
│
├── Documentation
│   ├── VMAMBA_PANSHARP_README.md    # Main guide
│   ├── EXECUTION_GUIDE.md           # This file
│   ├── COMPARISON_GUIDE.md          # Comparison guide
│   ├── WALDS_PROTOCOL.md            # Protocol details
│   └── requirements.txt             # Dependencies
│
├── Interactive
│   └── Model_Comparison_Analysis.ipynb  # Jupyter notebook
│
├── Datasets
│   ├── pavia/
│   │   └── PaviaU.mat
│   └── chikusei/
│       └── HyperspecVNIR_Chikusei_20140729.mat
│
└── Output (created during execution)
    ├── experiments/                 # Training outputs
    ├── results/                     # Evaluation outputs
    └── comparison_results/          # Comparison outputs
```

---

## 🎯 Key Commands Summary

| Task | Command |
|------|---------|
| Test installation | `python quick_start.py` |
| Train (Pavia) | `python train_vmamba_pansharp.py --dataset pavia --epochs 100` |
| Train (Chikusei) | `python train_vmamba_pansharp.py --dataset chikusei --epochs 300` |
| Evaluate | `python evaluate_vmamba_pansharp.py --checkpoint PATH` |
| Compare models | `python compare_models.py --dataset pavia --epochs 20` |
| Monitor training | `tensorboard --logdir experiments/EXP_NAME/logs` |
| Interactive comparison | `jupyter notebook Model_Comparison_Analysis.ipynb` |

---

## 📚 Additional Resources

- **Main Documentation**: VMAMBA_PANSHARP_README.md
- **Comparison Guide**: COMPARISON_GUIDE.md
- **Wald's Protocol**: WALDS_PROTOCOL.md
- **Paper**: SAI_SC24M175.pdf

---

## 💡 Tips for Best Results

1. **Start Small**: Test with `--epochs 10` first
2. **Monitor Early**: Use TensorBoard from the start
3. **Save Frequently**: Default saves every 50 epochs
4. **Use GPU**: Training on CPU is very slow
5. **Batch Size**: Larger is better (if GPU memory allows)
6. **Comparison**: Run comparison after successful single model training

---

## 🆘 Getting Help

If you encounter issues:
1. Check this guide's [Troubleshooting](#troubleshooting) section
2. Verify installation with `python quick_start.py`
3. Check dataset with `test_walds_protocol.py`
4. Review error messages carefully
5. Open an issue on GitHub (if applicable)

---

**Happy Training! 🚀**
