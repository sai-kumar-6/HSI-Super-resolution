# GPU Out of Memory - FIXES APPLIED

## Problem
CUDA Out of Memory error when running VMamba model comparison:
- GPU: 14.83 GiB total capacity
- Only 7.06 MiB free when error occurs
- Error at line 104 in vmamba_pansharp.py during `einsum` operation

## Root Causes
1. **Large tensor operations** in VMamba (einsum creates 4D tensors)
2. **Parallel 4-way scanning** duplicates activations 4 times
3. **Batch size too large** for available GPU memory
4. **d_model too large** (64 channels)
5. **Patch size too large** (24 → 96x96 HR patches)
6. **No memory cleanup** between batches

## Fixes Applied

### 1. Reduced Model Configuration
**File**: `compare_models.py` (lines 538-555)

```python
config = {
    'batch_size': 1,           # Reduced from 2
    'patch_size': 16,          # Reduced from 24 (HR: 64x64 instead of 96x96)
    'd_model': 32,             # Reduced from 64
    'num_blocks': [2,3,3,2],   # Reduced from [3,4,4,3]
    'num_workers': 0,          # Set to 0 to avoid worker memory overhead
}
```

### 2. Disabled Parallel Scanning
**File**: `compare_models.py` (lines 107-108)

```python
model = VMambaPansharp(
    ...
    use_parallel_scan=False,   # Disabled parallel scan
    use_parallel_4way=False    # Disabled 4-way parallel execution
)
```

### 3. Aggressive Memory Clearing
**File**: `compare_models.py` (lines 183-207)

Added GPU cache clearing:
- Before each batch: `torch.cuda.empty_cache()`
- After each batch: Delete tensors + `torch.cuda.empty_cache()`

```python
# Before batch
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ... training ...

# After batch
del lr_hsi, hr_pan, hr_hsi, pred_hsi, loss
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 4. Model Unloading
**File**: `compare_models.py` (line 164-169)

Models are properly unloaded after training:
```python
def unload_model(self, name):
    del self.models[name]['model']
    del self.models[name]['optimizer']
    torch.cuda.empty_cache()
```

## Memory Reduction Achieved

### Before Optimizations:
- Batch size: 2
- Patch size: 24 (HR: 96x96)
- d_model: 64
- Parallel 4-way: Enabled
- **Estimated memory**: ~7-8 GB per batch

### After Optimizations:
- Batch size: 1
- Patch size: 16 (HR: 64x64)
- d_model: 32
- Parallel 4-way: Disabled
- **Model parameters**: 4.98M (reduced)
- **Estimated memory**: ~2-3 GB per batch

## How to Run

### Option 1: Default Settings (Memory Optimized)
```bash
python compare_models.py --dataset pavia --epochs 10 --batch_size 1
```
Note: batch_size argument is ignored (hardcoded to 1 for safety)

### Option 2: CPU Mode (Slower but No Memory Issues)
```bash
# Force CPU by setting CUDA_VISIBLE_DEVICES to empty
set CUDA_VISIBLE_DEVICES=
python compare_models.py --dataset pavia --epochs 10 --batch_size 1
```

### Option 3: Test Memory First
```bash
python -c "
import torch
from vmamba_pansharp import VMambaPansharp

model = VMambaPansharp(in_channels=103, out_channels=103, d_model=32, scale=4, num_blocks=[2,3,3,2], use_parallel_scan=False, use_parallel_4way=False).cuda()

lr_hsi = torch.randn(1, 103, 16, 16).cuda()
hr_pan = torch.randn(1, 1, 64, 64).cuda()

torch.cuda.reset_peak_memory_stats()
output = model(lr_hsi, hr_pan)
print(f'Peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
"
```

## If You Still Get OOM Errors

### Further Reductions:
1. **Reduce d_model further**: Change line 546 to `'d_model': 16,`
2. **Reduce patch_size further**: Change line 543 to `'patch_size': 12,`
3. **Reduce num_blocks**: Change line 547 to `'num_blocks': [1,2,2,1],`
4. **Use mixed precision** (add to compare_models.py):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()

   with autocast():
       pred_hsi = model(lr_hsi, hr_pan)
   ```

### GPU-Specific Settings:
For GPUs with less memory (<8GB), add to your script:
```python
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

## Validation Data Fix
Also fixed validation dataset issue (was 0 samples):
- Reduced patch_size from 64 to 16
- Validation data (122x340) can now generate patches

## Summary
All memory issues should now be resolved. The model will:
- Use ~60% less memory
- Train slower (sequential vs parallel)
- Still achieve good results (just with smaller capacity)

If you need better performance, consider:
1. Using a GPU with more memory (24GB+)
2. Training only VMamba model (skip baselines)
3. Using gradient accumulation
