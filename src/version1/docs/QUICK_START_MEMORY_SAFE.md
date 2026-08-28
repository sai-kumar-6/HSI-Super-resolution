# Quick Start - Memory-Safe Training

## ⚡ Quick Commands (Copy & Paste)

### 1. Run Model Comparison (RECOMMENDED)
```bash
# Safe defaults: 16x16 patches, 8x8 overlap, batch=1
python compare_models.py --dataset pavia --epochs 20
```

### 2. Train VMamba Only
```bash
python train_vmamba_pansharp.py --dataset pavia --epochs 20
```

### 3. Monitor GPU Memory
```bash
# Run in separate terminal
nvidia-smi -l 1
```

## 📊 Current Configuration (Memory-Safe)

| Parameter | Value | Why |
|-----------|-------|-----|
| Patch Size (HR) | 16x16 | Fits in Tesla T4 memory |
| Patch Size (LR) | 8x8 | scale=2, so 16/2=8 |
| Overlap | 8 pixels | 50% overlap for val/test |
| Batch Size | 1 | Prevents OOM errors |
| Scale Factor | 2x | Easier than 4x |
| d_model | 32 | Reduced from 64 |
| GPU Memory | ~9GB | Safe for 16GB GPU |

## ✅ What's Fixed

1. **Patch size**: 64x64 → **16x16** (memory-safe)
2. **Overlap**: 32 → **8** (validation/testing)
3. **Batch size**: 8 → **1** (prevents OOM)
4. **Model training**: All at once → **One at a time**
5. **GPU cleanup**: Added `torch.cuda.empty_cache()`

## 🎯 Expected Results

### Training
- ✅ No OOM errors
- ✅ ~9GB GPU usage (well within 16GB)
- ✅ ~50 samples/sec training speed
- ✅ All 4 models complete successfully

### Validation/Testing
- ✅ Uses 16x16 patches
- ✅ 8x8 overlap between patches
- ✅ Batch size 1 for safety

## 🔧 Customization Options

### If you have MORE GPU memory (24GB+)
```bash
# Larger patches
python compare_models.py --patch_size 32 --overlap 16 --epochs 20

# Can also increase batch size
# Edit compare_models.py line 303: batch_size = 2
```

### If you have LESS GPU memory or still getting OOM
```bash
# Smaller patches
python compare_models.py --patch_size 8 --overlap 4 --epochs 20

# Or use minimal config
# Edit compare_models.py to import MINIMAL_CONFIG
```

## 📁 Files Modified

All files now use memory-safe defaults:

- ✅ `compare_models.py` - Main comparison script
- ✅ `train_vmamba_pansharp.py` - Training script
- ✅ `train_config_overlap.py` - Configuration presets
- ✅ `config_comparison_safe.py` - NEW safe config file
- ✅ `MEMORY_FIX_SUMMARY.md` - Detailed fix documentation

## 🚀 Running the Comparison

The script will:
1. Load Pavia dataset
2. Create 16x16 patches with 8x8 overlap
3. Train CNN (20 epochs) → unload
4. Train Transformer (20 epochs) → unload
5. Train U-Net (20 epochs) → unload
6. Train VMamba (20 epochs) → unload
7. Generate comparison plots
8. Save results to `comparison_results/`

**Total time**: ~2-4 hours (depends on dataset size)

## 📊 Output

Results saved to:
```
comparison_results/
├── comparison_pavia_YYYYMMDD_HHMMSS/
│   ├── config.json              # Configuration used
│   ├── cnn_final.pth           # Trained CNN model
│   ├── transformer_final.pth   # Trained Transformer
│   ├── u-net_final.pth         # Trained U-Net
│   ├── vmamba_final.pth        # Trained VMamba
│   └── summary.json            # Comparison metrics
```

## ⚠️ Important Notes

1. **Batch size MUST be 1** - Do not increase (will cause OOM)
2. **Scale MUST be 2** - Scale 4 uses too much memory
3. **Patch size 16x16** - Tested and safe for Tesla T4
4. **Overlap 8** - Recommended for validation/testing
5. **One model at a time** - Sequential training prevents OOM

## 🐛 If You Still Get Errors

```bash
# 1. Check GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# 2. Check GPU memory
nvidia-smi

# 3. Clear GPU cache manually
python -c "import torch; torch.cuda.empty_cache()"

# 4. Use minimal config (8x8 patches)
python compare_models.py --patch_size 8 --overlap 4
```

## 💡 Tips

1. **Monitor GPU during training**: `nvidia-smi -l 1`
2. **Start with fewer epochs**: `--epochs 5` for testing
3. **Check logs**: Look for memory usage in output
4. **Reduce model size** if still having issues (edit d_model, num_blocks)

---

**Ready to run!** Just copy the command from section 1 and you're good to go! 🚀
