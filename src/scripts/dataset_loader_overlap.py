"""
Dataset Loader with Overlapping Patches (NO DATA LEAKAGE)
Implements Wald's Protocol + spatial train/val/test separation.

patch_size = LR INPUT size (32 or 64).
HR patch   = patch_size × scale_factor.

Wald's Protocol (same as V1–V4):
  Step 1 — Gaussian blur  : gaussian_filter(sigma=[1,1,0]) — spatial PSF simulation
  Step 2 — Bicubic downsample by scale_factor → LR-HSI
  Step 3 — Spectral mean of HR-HSI → HR-PAN (1 channel, HR size)

Scale options: 2, 4, 8
  patch=32, scale=2 → LR 32×32  HR  64×64  PAN  64×64
  patch=32, scale=4 → LR 32×32  HR 128×128 PAN 128×128
  patch=32, scale=8 → LR 32×32  HR 256×256 PAN 256×256
  patch=64, scale=2 → LR 64×64  HR 128×128 PAN 128×128
  patch=64, scale=4 → LR 64×64  HR 256×256 PAN 256×256
  patch=64, scale=8 → LR 64×64  HR 512×512 PAN 512×512

Each batch returns:
  'lr_hsi' : (B, C, patch_size,            patch_size)           LR input
  'hr_pan' : (B, 1, patch_size*scale,       patch_size*scale)     HR PAN input
  'hr_hsi' : (B, C, patch_size*scale,       patch_size*scale)     HR ground truth
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
import h5py
from scipy.ndimage import gaussian_filter


# ============================================================================
# Dataset class
# ============================================================================

class HSIPatchDataset(Dataset):
    """
    Stores LR/HR patches as float16 (RAM efficient).
    Converts to float32 in __getitem__ so training always gets float32.
    """

    def __init__(self, lr_patches, hr_patches, augment=False):
        self.lr      = lr_patches   # (N, lr_h, lr_w, C) float16
        self.hr      = hr_patches   # (N, hr_h, hr_w, C) float16
        self.augment = augment

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        # Cast float16 → float32 for model compatibility
        lr = torch.from_numpy(self.lr[idx].astype(np.float32)).permute(2, 0, 1)  # (C, lr_h, lr_w)
        hr = torch.from_numpy(self.hr[idx].astype(np.float32)).permute(2, 0, 1)  # (C, hr_h, hr_w)

        if self.augment:
            if np.random.rand() < 0.5:
                lr = torch.flip(lr, dims=[2])
                hr = torch.flip(hr, dims=[2])
            if np.random.rand() < 0.5:
                lr = torch.flip(lr, dims=[1])
                hr = torch.flip(hr, dims=[1])
            k = np.random.randint(0, 4)
            if k > 0:
                lr = torch.rot90(lr, k, dims=[1, 2])
                hr = torch.rot90(hr, k, dims=[1, 2])

        # PAN = spectral mean of HR   shape: (1, hr_h, hr_w)
        hr_pan = hr.mean(dim=0, keepdim=True)

        return {'lr_hsi': lr, 'hr_pan': hr_pan, 'hr_hsi': hr}


# ============================================================================
# Dataset loaders
# ============================================================================

def load_pavia_dataset(mat_path):
    print(f"Loading Pavia dataset from: {mat_path}")
    data = sio.loadmat(mat_path)
    img  = data["paviaU"].astype(np.float32)
    print(f"  Shape: {img.shape}  dtype: {img.dtype}")
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.astype(np.float16)


def load_chikusei_dataset(mat_path):
    """
    Load Chikusei (2517×2335×128).
    Supports both MATLAB < v7.3 (scipy) and v7.3 HDF5 (h5py).
    Stores as float16 (~1.5 GB) to allow loading the FULL image.
    """
    print(f"Loading Chikusei dataset from: {mat_path}")

    try:
        data = sio.loadmat(mat_path)
        print("  Loaded via scipy (MATLAB < v7.3)")
        keys = [k for k in data.keys() if not k.startswith('__')]
        for key_name in ['chikusei', 'data', 'img', 'hsi']:
            if key_name in data:
                img = np.array(data[key_name], dtype=np.float32)
                break
        else:
            img = np.array(data[keys[0]], dtype=np.float32)

    except NotImplementedError:
        print("  Loaded via h5py (MATLAB v7.3 HDF5)")
        with h5py.File(mat_path, 'r') as f:
            keys     = [k for k in f.keys() if not k.startswith('__')]
            key_name = next((k for k in ['chikusei', 'data', 'img', 'hsi'] if k in f), keys[0])
            dset     = f[key_name]
            full_shape = dset.shape
            print(f"  Full HDF5 shape: {full_shape}")

            if len(full_shape) == 3:
                # HDF5/MATLAB v7.3 → (C=128, W=2335, H=2517)
                C, W, H = full_shape
                print(f"  Full image: H={H}  W={W}  C={C}")
                print(f"  Loading as float16 (~{H*W*C*2/1024**2:.0f} MB) ...")
                raw = dset[:].astype(np.float16)          # (C, W, H)
                img = np.transpose(raw, (2, 1, 0))        # (H, W, C)
                del raw
            else:
                img = np.array(dset, dtype=np.float32)
                if img.ndim == 3:
                    img = np.transpose(img, (2, 1, 0))

    print(f"  Shape: {img.shape}  dtype: {img.dtype}")

    # Normalise [0, 1] safely in float32, store back as float16
    vmin = float(img.min())
    vmax = float(img.max())
    img  = img.astype(np.float32)
    img -= vmin
    img /= (vmax - vmin + 1e-8)
    img  = img.astype(np.float16)
    print(f"  Normalised to [0,1]  RAM: {img.nbytes/1024**2:.0f} MB (float16)")
    return img


# ============================================================================
# Patch extraction
# ============================================================================

def extract_patches_with_spatial_separation(img, patch_size=32, overlap=16, scale_factor=4):
    """
    Extract patches with 80/10/10 spatial train/val/test split.

    patch_size = LR INPUT size (32 or 64).
    HR patch   = patch_size * scale_factor.

    Args:
        img         : (H, W, C) float16 numpy array, values in [0, 1]
        patch_size  : LR patch spatial size — 32 or 64
        overlap     : overlap between adjacent LR patches (applied in LR space)
        scale_factor: 2, 4, or 8

    Returns:
        (train_lr, train_hr), (val_lr, val_hr), (test_lr, test_hr)
        lr : float16 (N, patch_size,              patch_size,              C)
        hr : float16 (N, patch_size*scale_factor, patch_size*scale_factor, C)

    Example shapes (C=128):
        patch=32, scale=4 → LR (N,32,32,128)  HR (N,128,128,128)  PAN (N,1,128,128)
        patch=64, scale=4 → LR (N,64,64,128)  HR (N,256,256,128)  PAN (N,1,256,256)
    """
    assert scale_factor in [2, 4, 8], \
        f"scale_factor must be 2, 4, or 8 — got {scale_factor}"
    assert patch_size in [32, 64], \
        f"patch_size (LR size) must be 32 or 64 — got {patch_size}"
    assert 0 <= overlap < patch_size, \
        f"overlap must be in [0, patch_size) — got {overlap}"

    hr_patch = patch_size * scale_factor   # HR patch size
    H, W, C  = img.shape

    print(f"\n{'='*70}")
    print(f"PATCH EXTRACTION   scale={scale_factor}x   LR_patch={patch_size}   HR_patch={hr_patch}")
    print(f"  Full image : {H} x {W} x {C}")
    print(f"  LR input   : {patch_size} x {patch_size} x {C}  (model input)")
    print(f"  HR output  : {hr_patch} x {hr_patch} x {C}  (ground truth)")
    print(f"  PAN size   : {hr_patch} x {hr_patch} x 1   (model input)")
    print(f"{'='*70}")

    # Crop so both dims are divisible by hr_patch
    H = (H // hr_patch) * hr_patch
    W = (W // hr_patch) * hr_patch
    img = img[:H, :W, :]

    # stride is in HR space (overlap is LR space → convert to HR)
    hr_stride = (patch_size - overlap) * scale_factor

    # ── Spatial split (column-wise, no leakage) ───────────────────────────────
    train_end = int(0.8 * W)
    val_end   = int(0.9 * W)
    # align boundaries to hr_patch grid
    train_end = (train_end // hr_patch) * hr_patch
    val_end   = (val_end   // hr_patch) * hr_patch
    regions   = {
        'Train': img[:, :train_end,        :],
        'Val'  : img[:, train_end:val_end, :],
        'Test' : img[:, val_end:,          :],
    }
    print(f"\nSpatial regions (NO LEAKAGE):")
    for name, r in regions.items():
        print(f"  {name:5s}: {r.shape[0]}x{r.shape[1]}x{r.shape[2]}")

    # ── Extract HR patches from image ─────────────────────────────────────────
    def extract_hr_patches(region, name):
        h, w, c = region.shape
        patches = [
            region[i:i+hr_patch, j:j+hr_patch, :]
            for i in range(0, h - hr_patch + 1, hr_stride)
            for j in range(0, w - hr_patch + 1, hr_stride)
        ]
        arr = np.array(patches, dtype=np.float16)
        print(f"  {name:5s}: {len(patches)} patches")
        return arr

    print(f"\nExtracting HR patches (HR stride={hr_stride}):")
    train_hr_patches = extract_hr_patches(regions['Train'], 'Train')
    val_hr_patches   = extract_hr_patches(regions['Val'],   'Val')
    test_hr_patches  = extract_hr_patches(regions['Test'],  'Test')

    # ── Wald's Protocol: HR → LR ──────────────────────────────────────────────
    # Step 1: Gaussian blur (spatial PSF simulation, spectral untouched)
    # Step 2: Bicubic downsample by scale_factor → LR patch (patch_size × patch_size)
    def make_lr(hr_patches):
        lr_list = []
        for p in hr_patches:
            p_f32   = p.astype(np.float32)                          # (hr_h, hr_w, C)
            blurred = gaussian_filter(p_f32, sigma=[1.0, 1.0, 0])   # spatial blur only
            t    = torch.from_numpy(blurred).permute(2, 0, 1).unsqueeze(0)  # (1,C,HR,HR)
            lr_t = F.interpolate(t, size=(patch_size, patch_size),
                                 mode='bicubic', align_corners=False)        # (1,C,LR,LR)
            lr_list.append(lr_t.squeeze(0).permute(1, 2, 0).numpy().astype(np.float16))
        return np.array(lr_list, dtype=np.float16)

    print(f"\nWald's Protocol: Gaussian blur + bicubic downsample x{scale_factor}")
    print(f"  HR {hr_patch}x{hr_patch}  ->  LR {patch_size}x{patch_size}")
    train_lr = make_lr(train_hr_patches)
    val_lr   = make_lr(val_hr_patches)
    test_lr  = make_lr(test_hr_patches)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nFinal dataset shapes:")
    print(f"  {'Split':6s}  {'N':>6}  {'lr_hsi (input)':>22}  {'hr_hsi (GT)':>24}  {'hr_pan':>22}")
    print(f"  {'-'*82}")
    for split, lr, hr in [
        ('Train', train_lr, train_hr_patches),
        ('Val',   val_lr,   val_hr_patches),
        ('Test',  test_lr,  test_hr_patches),
    ]:
        lr_s  = f"({len(lr)},{C},{patch_size},{patch_size})"
        hr_s  = f"({len(hr)},{C},{hr_patch},{hr_patch})"
        pan_s = f"({len(hr)},1,{hr_patch},{hr_patch})"
        print(f"  {split:6s}  {len(lr):>6}  {lr_s:>22}  {hr_s:>24}  {pan_s:>22}")
    print(f"{'='*70}\n")

    return (train_lr, train_hr_patches), (val_lr, val_hr_patches), (test_lr, test_hr_patches)


# ============================================================================
# Public API
# ============================================================================

def create_dataloaders_overlap(
    dataset    = 'chikusei',
    mat_path   = None,
    batch_size = 4,
    patch_size = 32,
    overlap    = 16,
    scale      = 4,           # 2, 4, or 8
    num_workers = 0,
):
    """
    Create train/val/test DataLoaders with spatial separation (no data leakage).

    Args:
        dataset    : 'chikusei' or 'pavia'
        mat_path   : path to .mat file
        batch_size : training batch size
        patch_size : LR INPUT patch size — 32 or 64
        overlap    : overlap between adjacent LR patches
        scale      : upsampling factor — 2, 4, or 8
        num_workers: DataLoader workers

    Batch contents:
        'lr_hsi' : (B, C, patch_size,       patch_size)        LR input   float32
        'hr_pan' : (B, 1, patch_size*scale, patch_size*scale)  HR PAN     float32
        'hr_hsi' : (B, C, patch_size*scale, patch_size*scale)  HR GT      float32

    Example shapes (C=128):
        patch=32, scale=2 → lr=(B,128, 32, 32)  pan=(B,1, 64, 64)  hr=(B,128, 64, 64)
        patch=32, scale=4 → lr=(B,128, 32, 32)  pan=(B,1,128,128)  hr=(B,128,128,128)
        patch=32, scale=8 → lr=(B,128, 32, 32)  pan=(B,1,256,256)  hr=(B,128,256,256)
        patch=64, scale=2 → lr=(B,128, 64, 64)  pan=(B,1,128,128)  hr=(B,128,128,128)
        patch=64, scale=4 → lr=(B,128, 64, 64)  pan=(B,1,256,256)  hr=(B,128,256,256)
        patch=64, scale=8 → lr=(B,128, 64, 64)  pan=(B,1,512,512)  hr=(B,128,512,512)
    """
    assert scale in [2, 4, 8], f"scale must be 2, 4, or 8 — got {scale}"
    assert patch_size in [32, 64], f"patch_size (LR size) must be 32 or 64 — got {patch_size}"

    # Load image
    if dataset == 'chikusei':
        img = load_chikusei_dataset(mat_path)
    elif dataset == 'pavia':
        img = load_pavia_dataset(mat_path)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'chikusei' or 'pavia'.")

    # Extract patches
    (train_lr, train_hr), (val_lr, val_hr), (test_lr, test_hr) = \
        extract_patches_with_spatial_separation(img, patch_size, overlap, scale)
    del img   # free the full image from RAM now that patches are ready

    # Build Dataset objects
    train_ds = HSIPatchDataset(train_lr, train_hr, augment=True)
    val_ds   = HSIPatchDataset(val_lr,   val_hr,   augment=False)
    test_ds  = HSIPatchDataset(test_lr,  test_hr,  augment=False)

    # Build DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"DataLoaders ready  (scale={scale}x):")
    print(f"  Train : {len(train_ds):>6} patches  ->  {len(train_loader):>5} batches")
    print(f"  Val   : {len(val_ds):>6} patches  ->  {len(val_loader):>5} batches")
    print(f"  Test  : {len(test_ds):>6} patches  ->  {len(test_loader):>5} batches")

    return train_loader, val_loader, test_loader


# ============================================================================
# Quick self-test
# ============================================================================

if __name__ == '__main__':
    import os, argparse

    parser = argparse.ArgumentParser(description='Dataset loader self-test')
    parser.add_argument('--dataset',    default='chikusei', choices=['chikusei', 'pavia'])
    parser.add_argument('--mat_path',   default='chikusei/chikusei.mat')
    parser.add_argument('--scale',      default=4, type=int, choices=[2, 4, 8])
    parser.add_argument('--patch_size', default=32, type=int, choices=[32, 64])
    parser.add_argument('--overlap',    default=16, type=int)
    parser.add_argument('--batch_size', default=2,  type=int)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Self-test: {args.dataset}  scale={args.scale}x  patch={args.patch_size}")
    print(f"{'='*60}\n")

    train_l, val_l, test_l = create_dataloaders_overlap(
        dataset    = args.dataset,
        mat_path   = args.mat_path,
        batch_size = args.batch_size,
        patch_size = args.patch_size,
        overlap    = args.overlap,
        scale      = args.scale,
    )

    print(f"\nSample batch shapes (scale={args.scale}x, patch={args.patch_size}):")
    for name, loader in [('Train', train_l), ('Val', val_l), ('Test', test_l)]:
        if len(loader) == 0:
            print(f"  {name}: empty"); continue
        b = next(iter(loader))
        lr  = b['lr_hsi']
        pan = b['hr_pan']
        hr  = b['hr_hsi']
        print(f"  {name:5s}  lr_hsi={tuple(lr.shape)}  hr_pan={tuple(pan.shape)}  hr_hsi={tuple(hr.shape)}")
        assert lr.dtype  == torch.float32, "LR must be float32"
        assert pan.dtype == torch.float32, "PAN must be float32"
        assert hr.dtype  == torch.float32, "HR must be float32"
        assert pan.shape[1] == 1,          "PAN must have 1 channel"
        assert lr.shape[2] == args.patch_size,                   "LR must equal patch_size"
        assert hr.shape[2] == args.patch_size * args.scale,       "HR must equal patch_size*scale"
        assert pan.shape[2] == args.patch_size * args.scale,      "PAN must equal patch_size*scale"

    print(f"\n[OK] All assertions passed for scale={args.scale}x")
