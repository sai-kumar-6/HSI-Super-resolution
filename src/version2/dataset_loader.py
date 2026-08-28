"""
dataset_loader.py  —  Version 2
Self-contained dataset loader for Chikusei hyperspectral pansharpening.

Key change vs Version 1:
  PER-BAND NORMALIZATION: each spectral band is normalised independently
      X_norm(i) = (X(i) - min_i) / (max_i - min_i + eps)
  Applied BEFORE PAN generation and BEFORE splitting into train/val/test.
  This prevents bright bands dominating training and improves SAM & ERGAS.

Pipeline:
  load HSI cube (H,W,C) →
  per-band normalise (C bands independently) →
  spatial split 80/10/10 →
  extract overlapping patches →
  create LR-HR pairs (bicubic downsample) →
  generate PAN (mean over bands from HR patch) →
  return DataLoaders
"""
from __future__ import annotations
import numpy as np
import scipy.io as sio
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── Per-band normalisation ────────────────────────────────────────────────────

def per_band_normalise(img: np.ndarray) -> np.ndarray:
    """
    Normalise each spectral band independently to [0, 1].

    Args:
        img : (H, W, C) float32 array
    Returns:
        img : (H, W, C) float32 array, each band in [0, 1]

    Formula:  X_norm(i) = (X(i) - min_i) / (max_i - min_i + eps)
    """
    H, W, C = img.shape
    eps = 1e-8
    for b in range(C):
        band = img[:, :, b]
        bmin = float(band.min())
        bmax = float(band.max())
        img[:, :, b] = (band - bmin) / (bmax - bmin + eps)
    return img


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_chikusei(mat_path: str, max_rows: int = 512) -> np.ndarray:
    """
    Load Chikusei hyperspectral dataset → (H, W, C) float32, per-band normalised.

    Args:
        mat_path : path to chikusei.mat
        max_rows : max spatial rows to load (512 rows × 2517 cols × 128 bands ≈ 310 MB)
    """
    print(f"[DataLoader] Loading Chikusei: {mat_path}")
    try:
        data = sio.loadmat(mat_path)
        keys = [k for k in data.keys() if not k.startswith('_')]
        key  = next((k for k in ['chikusei','data','img','hsi'] if k in data), keys[0])
        img  = np.array(data[key]).astype(np.float32)
    except NotImplementedError:
        with h5py.File(mat_path, 'r') as f:
            keys = [k for k in f.keys() if not k.startswith('_')]
            key  = next((k for k in ['chikusei','data','img','hsi'] if k in f), keys[0])
            dset = f[key]
            full = dset.shape                         # (C, W, H) for HDF5/v7.3
            h_load = min(max_rows, full[2])
            raw  = dset[:, :, :h_load]               # (C, W, h_load)
            img  = np.transpose(raw, (2, 1, 0)).astype(np.float32)  # (h_load, W, C)
            del raw

    # Ensure (H, W, C) layout
    if img.ndim == 3 and img.shape[0] < img.shape[2]:
        # (C, H, W) → (H, W, C)
        img = np.transpose(img, (1, 2, 0))

    print(f"  Shape: {img.shape}  range [{img.min():.1f}, {img.max():.1f}]")

    # ── Per-band normalisation (BEFORE any other processing) ──────────────────
    print("  Applying per-band normalisation …")
    img = per_band_normalise(img)
    print(f"  After normalisation: range [{img.min():.4f}, {img.max():.4f}]")

    return img


# ── Patch extraction with spatial separation ─────────────────────────────────

def extract_patches(
    img: np.ndarray,
    patch_size: int = 32,
    overlap: int = 16,
    scale: int = 4,
) -> tuple:
    """
    Extract non-leaking patches with 80/10/10 spatial split.

    Args:
        img        : (H, W, C) float32 — per-band normalised
        patch_size : HR patch size
        overlap    : overlap in pixels
        scale      : upsampling factor (2 or 4)
    Returns:
        (train_lr, train_hr), (val_lr, val_hr), (test_lr, test_hr)
        each LR: (N, C, H/s, W/s) float32 numpy
        each HR: (N, C, H, W)     float32 numpy
    """
    H, W, C = img.shape
    assert scale in (2, 4), f"scale must be 2 or 4, got {scale}"

    # Crop to be divisible by scale
    H = (H // scale) * scale
    W = (W // scale) * scale
    img = img[:H, :W, :]

    stride = patch_size - overlap

    # Spatial split (along width — columns)
    w_train = int(0.80 * W)
    w_val   = int(0.90 * W)
    regions = {
        'train': img[:, :w_train, :],
        'val':   img[:, w_train:w_val, :],
        'test':  img[:, w_val:, :],
    }
    print(f"\n  Spatial split  train=cols[0:{w_train}]  "
          f"val=[{w_train}:{w_val}]  test=[{w_val}:{W}]")

    def _patches(region, name):
        h, w, c = region.shape
        pts = []
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                p = region[i:i+patch_size, j:j+patch_size, :]
                if p.shape == (patch_size, patch_size, c):
                    pts.append(p)
        arr = np.array(pts, dtype=np.float32)   # (N, ph, pw, C)
        print(f"  {name}: {len(arr)} patches")
        return arr

    splits = {k: _patches(v, k) for k, v in regions.items()}

    def _lr_hr(patches):
        # patches: (N, ph, pw, C)
        # → HR: (N, C, ph, pw), LR: (N, C, ph/s, pw/s)
        hr = torch.from_numpy(patches).permute(0, 3, 1, 2)  # (N,C,H,W)
        lr = F.interpolate(hr, scale_factor=1/scale,
                           mode='bilinear', align_corners=False)
        return lr.numpy(), hr.numpy()

    print(f"\n  Creating LR-HR pairs (scale={scale}) …")
    results = {}
    for k, pts in splits.items():
        lr, hr = _lr_hr(pts)
        results[k] = (lr, hr)
        print(f"  {k}: LR {lr.shape}  HR {hr.shape}")

    return results['train'], results['val'], results['test']


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class HSIPanDataset(Dataset):
    """
    Dataset yielding {'lr_hsi', 'hr_pan', 'hr_hsi'} dictionaries.
    PAN is generated as the mean over HR spectral bands.
    """

    def __init__(self, lr: np.ndarray, hr: np.ndarray, augment: bool = False):
        """
        Args:
            lr      : (N, C, lH, lW) float32
            hr      : (N, C, H,  W)  float32
            augment : random geometric augmentation
        """
        self.lr      = torch.from_numpy(lr)   # (N,C,h,w)
        self.hr      = torch.from_numpy(hr)   # (N,C,H,W)
        self.augment = augment

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        lr = self.lr[idx].clone()   # (C, h, w)
        hr = self.hr[idx].clone()   # (C, H, W)

        if self.augment:
            # Random horizontal flip
            if torch.rand(1) < 0.5:
                lr = torch.flip(lr, dims=[2])
                hr = torch.flip(hr, dims=[2])
            # Random vertical flip
            if torch.rand(1) < 0.5:
                lr = torch.flip(lr, dims=[1])
                hr = torch.flip(hr, dims=[1])
            # Random 90° rotation
            k = torch.randint(0, 4, (1,)).item()
            if k:
                lr = torch.rot90(lr, k, dims=[1, 2])
                hr = torch.rot90(hr, k, dims=[1, 2])

        # PAN generated from HR (spectral mean) — AFTER per-band normalisation
        hr_pan = hr.mean(dim=0, keepdim=True)   # (1, H, W)

        return {'lr_hsi': lr, 'hr_pan': hr_pan, 'hr_hsi': hr}


# ── Public factory ────────────────────────────────────────────────────────────

def create_dataloaders(
    mat_path: str,
    batch_size: int = 4,
    patch_size: int = 32,
    overlap: int    = 16,
    scale: int      = 4,
    num_workers: int = 0,
    max_rows: int   = 512,
) -> tuple:
    """
    Full pipeline: load → per-band normalise → split → patches → DataLoaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    img = load_chikusei(mat_path, max_rows=max_rows)

    (tr_lr, tr_hr), (vl_lr, vl_hr), (ts_lr, ts_hr) = extract_patches(
        img, patch_size=patch_size, overlap=overlap, scale=scale
    )
    del img  # free RAM

    train_ds = HSIPanDataset(tr_lr, tr_hr, augment=True)
    val_ds   = HSIPanDataset(vl_lr, vl_hr, augment=False)
    test_ds  = HSIPanDataset(ts_lr, ts_hr, augment=False)

    kw = dict(num_workers=num_workers, pin_memory=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    test_dl  = DataLoader(test_ds,  batch_size=1,          shuffle=False, **kw)

    print(f"\n  DataLoaders ready  "
          f"train={len(train_dl)}  val={len(val_dl)}  test={len(test_dl)} batches")
    return train_dl, val_dl, test_dl


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    mat = os.path.join(os.path.dirname(__file__), '..', 'chikusei', 'chikusei.mat')
    tr, vl, ts = create_dataloaders(mat, batch_size=2, patch_size=32,
                                    overlap=16, scale=4, max_rows=256)
    b = next(iter(tr))
    print(f"\nBatch:  LR {b['lr_hsi'].shape}  PAN {b['hr_pan'].shape}  HR {b['hr_hsi'].shape}")
    print(f"PAN range:  [{b['hr_pan'].min():.4f}, {b['hr_pan'].max():.4f}]")
    print("[OK] dataset_loader.py self-test passed")
