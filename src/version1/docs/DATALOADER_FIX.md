# DataLoader Parameter Fix

## Issue
The `create_dataloaders()` function was being called with incorrect parameter names across multiple files.

### Error Message
```
TypeError: create_dataloaders() got an unexpected keyword argument 'dataset_name'
```

## Root Cause
The function signature in [dataset_loader.py](dataset_loader.py) is:
```python
def create_dataloaders(
    dataset="pavia",      # ← Correct parameter name
    mat_path=None,
    batch_size=8,
    patch_size=64,
    scale=4,
    num_workers=4
)
```

But files were calling it with:
```python
create_dataloaders(
    dataset_name=config['dataset'],  # ← Wrong parameter name
    ...
)
```

Additionally, the **required** `mat_path` parameter was missing.

## Files Fixed

### 1. [train_vmamba_pansharp.py](train_vmamba_pansharp.py) (Line 90)
**Before:**
```python
self.train_loader, self.val_loader = create_dataloaders(
    dataset_name=config['dataset'],  # ❌ Wrong
    batch_size=config['batch_size'],
    patch_size=config['patch_size'],
    scale=config['scale'],
    num_workers=config['num_workers']
)
```

**After:**
```python
# Get dataset path
if config['dataset'] == 'pavia':
    mat_path = 'pavia/PaviaU.mat'
elif config['dataset'] == 'chikusei':
    mat_path = 'chikusei/chikusei.mat'
else:
    raise ValueError(f"Unknown dataset: {config['dataset']}")

self.train_loader, self.val_loader = create_dataloaders(
    dataset=config['dataset'],       # ✓ Correct
    mat_path=mat_path,               # ✓ Added
    batch_size=config['batch_size'],
    patch_size=config['patch_size'],
    scale=config['scale'],
    num_workers=config['num_workers']
)
```

### 2. [evaluate_vmamba_pansharp.py](evaluate_vmamba_pansharp.py) (Line 68)
Same fix applied.

### 3. [visualize_model_output.py](visualize_model_output.py) (Line 407)
Same fix applied.

### 4. [compare_models.py](compare_models.py) (Line 51)
Same fix applied.

## Changes Made

1. **Changed parameter name**: `dataset_name` → `dataset`
2. **Added mat_path parameter**: Automatically determined based on dataset name
   - `pavia` → `'pavia/PaviaU.mat'`
   - `chikusei` → `'chikusei/chikusei.mat'`
3. **Added error handling**: Raises `ValueError` for unknown datasets

## Testing

Run the fixed training script:
```bash
python train_vmamba_pansharp.py --dataset pavia --epochs 100
```

All dependent scripts should now work correctly:
- ✓ [train_vmamba_pansharp.py](train_vmamba_pansharp.py)
- ✓ [evaluate_vmamba_pansharp.py](evaluate_vmamba_pansharp.py)
- ✓ [visualize_model_output.py](visualize_model_output.py)
- ✓ [compare_models.py](compare_models.py)

## Note
Make sure your dataset files are in the correct locations:
- `pavia/PaviaU.mat` for Pavia University dataset
- `chikusei/chikusei.mat` for Chikusei dataset

---

**Status**: ✓ Fixed
**Date**: December 21, 2024
