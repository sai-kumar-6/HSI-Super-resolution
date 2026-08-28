# VMamba-Pansharp: Vision Mamba for Hyperspectral Pansharpening

Implementation of the VMamba-Pansharp architecture for hyperspectral image pansharpening as described in the paper.

## Architecture Overview

VMamba-Pansharp combines low-resolution hyperspectral images (LR-HSI) with high-resolution panchromatic images (HR-PAN) to produce high-resolution hyperspectral images (HR-HSI) using Vision Mamba.

### Key Components

1. **HSI Encoder**: 3D convolution + progressive upsampling (2×2× = 4×)
2. **PAN Encoder**: 2D convolution + edge enhancement using Sobel operators
3. **Cross-Attention Fusion**: Multi-head attention where PAN queries HSI features
4. **VMamba Backbone**: 4-stage hierarchical architecture with 2D-Selective-Scan (SS2D)
5. **Reconstruction Head**: Residual learning for final HR-HSI output

### Loss Functions

Composite loss with multiple objectives:
- **L1 Loss** (λ₁=1.0): Pixel-wise reconstruction
- **SAM Loss** (λ₂=0.1): Spectral angle preservation
- **Edge Loss** (λ₃=0.05): Sharp spatial details
- **SSIM Loss** (λ₄=0.1): Structural similarity

## File Structure

```
.
├── vmamba_pansharp.py           # Main model architecture
├── dataset_loader.py             # Dataset loading with Wald's protocol
├── loss_functions.py             # All loss functions and metrics
├── train_vmamba_pansharp.py     # Training script
├── evaluate_vmamba_pansharp.py  # Evaluation and visualization
├── chikusei/                     # Chikusei dataset
│   └── HyperspecVNIR_Chikusei_20140729.mat
└── pavia/                        # Pavia University dataset
    └── PaviaU.mat
```

## Requirements

```bash
pip install torch torchvision tensorboard
pip install numpy scipy matplotlib opencv-python
pip install einops tqdm
```

## Dataset Preparation

### Pavia University Dataset
- **Dimensions**: 610×340×103 (height × width × bands)
- **Wavelength**: 430-860 nm
- **Spatial resolution**: 1.3 m

### Chikusei Dataset
- **Dimensions**: 2517×2335×128 (height × width × bands)
- **Wavelength**: 363-1018 nm
- **Spatial resolution**: 2.5 m

Place the `.mat` files in their respective directories:
- `pavia/PaviaU.mat`
- `chikusei/HyperspecVNIR_Chikusei_20140729.mat`

### Synthetic Ground Truth Generation (Wald's Protocol)

Since true high-resolution hyperspectral ground truth is unavailable in practice, we employ **Wald's protocol** for synthetic data generation. This protocol creates realistic training triplets from available HR-HSI data:

#### Protocol Steps:

1. **Original HR-HSI**: The full resolution dataset serves as ground truth **H_GT**
   - Pavia: 610×340×103
   - Chikusei: 2517×2335×128

2. **LR-HSI Generation**: Apply degradation to simulate low-resolution acquisition
   - **Gaussian blur**: kernel size 8×8, σ=1.0
   - **Decimation**: downsample by factor 4
   - Result: **H_LR** ∈ R^(152×85×103) for Pavia

3. **HR-PAN Generation**: Average all spectral bands to create single-band panchromatic
   ```
   P_HR = (1/C) × Σ(k=1 to C) H_GT^(k)
   ```
   - For Pavia: P_HR = (1/103) × Σ(k=1 to 103) H_GT^(k)
   - Result: **P_HR** ∈ R^(610×340) for Pavia

#### Training Triplets:

**Pavia University:**
- LR-HSI: H_LR ∈ R^(152×85×103)
- HR-PAN: P_HR ∈ R^(610×340)
- HR-HSI (GT): H_GT ∈ R^(610×340×103)

**Chikusei:**
- LR-HSI: H_LR ∈ R^(629×584×128)
- HR-PAN: P_HR ∈ R^(2517×2335)
- HR-HSI (GT): H_GT ∈ R^(2517×2335×128)

#### Implementation Details:

The dataset loader (`dataset_loader.py`) automatically implements this protocol:

```python
from dataset_loader import create_dataloaders

# Creates triplets following Wald's protocol
train_loader, val_loader = create_dataloaders(
    dataset_name='pavia',
    batch_size=8,
    patch_size=64,      # LR patch size (HR will be 256=64×4)
    scale=4,            # Upsampling factor
)
```

**Key Features:**
- ✅ Gaussian blur with σ=1.0 before downsampling
- ✅ 4× decimation (bicubic interpolation)
- ✅ Spectral averaging for PAN generation
- ✅ Patch extraction with 50% overlap for training
- ✅ Data augmentation (flips, rotations, brightness, noise)
- ✅ Per-band normalization

**Advantages of Wald's Protocol:**
1. Creates realistic training data from available HR-HSI
2. Maintains spatial-spectral consistency
3. Standard protocol for pansharpening benchmarks
4. Enables quantitative evaluation with known ground truth

## Usage

### 1. Training

Train on Pavia dataset:
```bash
python train_vmamba_pansharp.py --dataset pavia --batch_size 8 --epochs 300
```

Train on Chikusei dataset:
```bash
python train_vmamba_pansharp.py --dataset chikusei --batch_size 4 --epochs 300
```

**Training Arguments:**
- `--dataset`: Dataset to use ('pavia' or 'chikusei')
- `--exp_name`: Experiment name (auto-generated if not specified)
- `--batch_size`: Batch size (default: 8)
- `--epochs`: Number of training epochs (default: 300)
- `--lr`: Initial learning rate (default: 1e-4)
- `--patch_size`: LR patch size (default: 64, HR will be 256)
- `--resume`: Path to checkpoint to resume training
- `--num_workers`: Number of data loading workers (default: 4)

### 2. Evaluation

Evaluate a trained model:
```bash
python evaluate_vmamba_pansharp.py --checkpoint experiments/YOUR_EXP/checkpoints/best.pth --output_dir results
```

Export results to .mat file:
```bash
python evaluate_vmamba_pansharp.py --checkpoint experiments/YOUR_EXP/checkpoints/best.pth --export
```

**Evaluation Arguments:**
- `--checkpoint`: Path to model checkpoint
- `--output_dir`: Output directory for results (default: 'results')
- `--export`: Export results to .mat file

### 3. Testing the Model

Quick test of model architecture:
```bash
python vmamba_pansharp.py
```

Test loss functions:
```bash
python loss_functions.py
```

Test dataset loader:
```bash
python dataset_loader.py
```

## Model Architecture Details

### HSI Encoder
```
Input: (B, C, H, W) - LR-HSI
↓
Conv3D (1, 64, kernel=3×3×3)
↓
BatchNorm3D + ReLU
↓
PixelShuffle (2×) + Conv2D
↓
PixelShuffle (2×) + Conv2D
↓
Projection to d_model
↓
Output: (B, d_model, 4H, 4W)
```

### PAN Encoder
```
Input: (B, 1, rH, rW) - HR-PAN
↓
Conv2D (1→32) + Conv2D (32→64)
↓
Residual Block
↓
Edge Enhancement (Sobel operators, α=0.5)
↓
Projection to d_model
↓
Output: (B, d_model, rH, rW)
```

### VMamba Backbone (4 stages)
```
Stage 1: rh×rw, d_model channels, 3 VMamba blocks
↓ Downsample 2×
Stage 2: rh/2×rw/2, 2d_model channels, 4 VMamba blocks
↓ Downsample 2×
Stage 3: rh/4×rw/4, 4d_model channels, 4 VMamba blocks (bottleneck)
↓ Upsample 2× + Skip from Stage 2
Stage 4a: rh/2×rw/2, 2d_model channels, 1-2 VMamba blocks
↓ Upsample 2× + Skip from Stage 1
Stage 4b: rh×rw, d_model channels, 1-2 VMamba blocks
```

### VMamba Block
```
Input: (B, C, H, W)
↓
LayerNorm + SS2D (4-directional scan)
↓ + Residual
LayerNorm + MLP (expansion factor 4)
↓ + Residual
Output: (B, C, H, W)
```

## Training Strategy

### Optimizer: AdamW
- Learning rate: η₀ = 1×10⁻⁴
- Weight decay: λ = 1×10⁻²
- Momentum: β₁ = 0.9, β₂ = 0.999

### Learning Rate Schedule
Cosine annealing with warmup:
- Warmup: 10 epochs (linear increase)
- Total: 300 epochs
- Min LR: η_min = 1×10⁻⁶

### Data Augmentation
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random 90° rotation
- Brightness adjustment (α ∼ U(0.9, 1.1))
- Gaussian noise (σ=0.01)

### Gradient Clipping
- Max norm: 1.0

## Evaluation Metrics

1. **PSNR** (Peak Signal-to-Noise Ratio): Higher is better
   - Measures reconstruction quality in dB

2. **SAM** (Spectral Angle Mapper): Lower is better
   - Measures spectral fidelity in degrees
   - Invariant to illumination changes

3. **ERGAS** (Erreur Relative Globale Adimensionnelle de Synthèse): Lower is better
   - Global relative error
   - Considers all spectral bands

## Output Structure

After training, the experiment directory will contain:
```
experiments/
└── vmamba_pansharp_pavia_YYYYMMDD_HHMMSS/
    ├── config.json                    # Training configuration
    ├── checkpoints/
    │   ├── latest.pth                 # Latest checkpoint
    │   ├── best.pth                   # Best model (highest PSNR)
    │   └── epoch_*.pth                # Periodic checkpoints
    └── logs/                          # TensorBoard logs
```

After evaluation:
```
results/
├── evaluation_results.json     # Quantitative metrics
├── sample_0.png               # Visualization of sample 0
├── sample_1.png               # Visualization of sample 1
├── ...
└── vmamba_pansharp_results.mat  # Exported results (if --export)
```

## Model Parameters

Default configuration:
- **d_model**: 64 (feature dimension)
- **num_blocks**: [3, 4, 4, 3] (blocks per stage)
- **d_state**: 16 (state space dimension)
- **num_heads**: 4 (attention heads)
- **scale**: 4 (upsampling factor)

Estimated parameters: ~10-20M (depends on input channels)

## Visualization

The evaluation script generates visualizations with:
1. **LR-HSI (upsampled)**: Bicubic interpolation of input
2. **HR-PAN**: High-resolution panchromatic image
3. **Predicted HR-HSI**: Model output (RGB bands)
4. **Ground Truth HR-HSI**: Reference image (RGB bands)
5. **Error Map**: Per-pixel absolute error
6. **Spectral Signature**: Center pixel comparison

## Monitoring Training

Use TensorBoard to monitor training:
```bash
tensorboard --logdir experiments/YOUR_EXP/logs
```

Metrics tracked:
- Training/Validation loss (total and components)
- PSNR, SAM, ERGAS
- Learning rate

## Tips for Best Results

1. **Batch Size**: Use largest batch size that fits in GPU memory
   - Pavia: 8-16 recommended
   - Chikusei: 2-4 recommended (larger images)

2. **Patch Size**: Balance between context and memory
   - Larger patches: More context, higher memory
   - Smaller patches: Less memory, more samples

3. **Training Time**: Full training takes ~24-48 hours on single GPU
   - Use `--resume` to continue interrupted training

4. **Checkpointing**: Best model is saved based on validation PSNR
   - Periodic checkpoints saved every 50 epochs

## Citation

If you use this implementation, please cite the original paper:
```
[Add paper citation here]
```

## License

[Specify license]

## Acknowledgments

- Implementation based on the VMamba-Pansharp architecture
- Mamba components adapted from the original Mamba paper
- Datasets: Pavia University, Chikusei

## Contact

For questions or issues, please open an issue in the repository.
