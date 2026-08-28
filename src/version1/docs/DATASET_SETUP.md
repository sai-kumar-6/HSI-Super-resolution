# Dataset Setup Guide

## ✅ Your Datasets Are Ready!

### Dataset Files

**Pavia University:**
```
pavia/
├── PaviaU.mat                 ✓ USED - Hyperspectral image (610×340×103)
└── PaviaU_gt.mat              ✗ NOT USED - Classification labels (for land cover)
```

**Chikusei:**
```
chikusei/
├── HyperspecVNIR_Chikusei_20140729.mat              ✓ USED - Hyperspectral (2517×2335×128)
└── HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat ✗ NOT USED - Classification labels
```

### Important Notes

1. **You only need the hyperspectral image files** (`PaviaU.mat` and `HyperspecVNIR_Chikusei_20140729.mat`)

2. **The `_gt.mat` files are NOT needed** for pansharpening - they contain land cover classification labels (0-9 for different classes like asphalt, meadows, trees, etc.)

3. **Wald's Protocol generates training data automatically:**
   - Takes HR-HSI (high-resolution hyperspectral)
   - Generates LR-HSI by blurring and downsampling
   - Generates HR-PAN by averaging spectral bands
   - Uses original HR-HSI as ground truth

### How Training Works

```
Your Dataset (PaviaU.mat):
   610×340×103 bands
         ↓
   Wald's Protocol (automatic)
         ↓
    ┌────┴────┐
    ↓         ↓
LR-HSI    HR-PAN      → Model Training → Reconstruct HR-HSI
152×85×103  610×340×1                     (610×340×103)
```

**No separate ground truth files needed!**

### Dataset Processing Details

**Pavia:**
- Input: `PaviaU.mat` (610×340×103)
- Auto-split: 80% train (488×340×103), 20% val (122×340×103)
- Wald's protocol generates:
  - LR-HSI: 152×85×103 (4× downsampled)
  - HR-PAN: 610×340×1 (spectral average)

**Chikusei:**
- Input: `HyperspecVNIR_Chikusei_20140729.mat` (2517×2335×128)
- Auto-split: 80% train, 20% val
- Wald's protocol generates:
  - LR-HSI: 629×584×128 (4× downsampled)
  - HR-PAN: 2517×2335×1 (spectral average)

### Fixed: MATLAB v7.3 Support

**Issue:** Chikusei dataset uses MATLAB v7.3 format (requires h5py)

**Solution:** Updated code to automatically detect and handle both:
- Old MATLAB format (scipy.io.loadmat)
- MATLAB v7.3 format (h5py)

### Installation

**Install new dependency:**
```bash
pip install h5py>=3.8.0
```

Or reinstall all requirements:
```bash
pip install -r requirements.txt
```

### Verify Your Setup

**Check Pavia dataset:**
```bash
python -c "import scipy.io as sio; data = sio.loadmat('pavia/PaviaU.mat'); print('Pavia shape:', data['paviaU'].shape)"
```
Expected output: `Pavia shape: (610, 340, 103)`

**Check dataset loader works:**
```bash
python test_walds_protocol.py
```

### Next Steps

1. **Install h5py:**
   ```bash
   pip install h5py>=3.8.0
   ```

2. **Test installation:**
   ```bash
   python quick_start.py
   ```

3. **Visualize your datasets:**
   ```bash
   python visualize_dataset.py
   ```

4. **Start training:**
   ```bash
   # Pavia (quick test)
   python train_vmamba_pansharp.py --dataset pavia --epochs 100 --batch_size 8

   # Chikusei (full training)
   python train_vmamba_pansharp.py --dataset chikusei --epochs 300 --batch_size 4
   ```

5. **Monitor training:**
   ```bash
   tensorboard --logdir experiments/YOUR_EXP/logs
   ```

### Summary

✅ **Pavia dataset** - Ready to use
✅ **Chikusei dataset** - Ready to use (after installing h5py)
✅ **Wald's protocol** - Implemented and tested
✅ **Data loading** - Handles both MATLAB formats
✅ **Visualization** - Complete dataset and model output visualization

**Your project is ready for training!**
