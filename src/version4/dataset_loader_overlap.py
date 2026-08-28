"""
Dataset Loader with Overlapping Patches (NO DATA LEAKAGE)
Implements spatial separation for train/val/test sets
Scale factor: 2 (configurable)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
import h5py


class HSIPatchDataset(Dataset):
    """Dataset for HSI patches with LR-HR pairs"""
    def __init__(self, lr_patches, hr_patches, augment=False):
        self.lr = lr_patches
        self.hr = hr_patches
        self.augment = augment

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        lr = torch.from_numpy(self.lr[idx]).permute(2, 0, 1).float()  # (C, H, W)
        hr = torch.from_numpy(self.hr[idx]).permute(2, 0, 1).float()

        # Augmentation (geometric only)
        if self.augment:
            # Random flip
            if np.random.rand() < 0.5:
                lr = torch.flip(lr, dims=[2])  # horizontal flip
                hr = torch.flip(hr, dims=[2])

            if np.random.rand() < 0.5:
                lr = torch.flip(lr, dims=[1])  # vertical flip
                hr = torch.flip(hr, dims=[1])

            # Random rotation (90, 180, 270 degrees)
            k = np.random.randint(0, 4)
            if k > 0:
                lr = torch.rot90(lr, k, dims=[1, 2])
                hr = torch.rot90(hr, k, dims=[1, 2])

        # Generate PAN from HR (mean over spectral bands)
        hr_pan = hr.mean(dim=0, keepdim=True)  # (1, H, W)

        # Return as dict for compatibility with training script
        return {
            'lr_hsi': lr,
            'hr_pan': hr_pan,
            'hr_hsi': hr
        }


def load_pavia_dataset(mat_path):
    """Load Pavia University dataset"""
    print(f"Loading Pavia dataset from: {mat_path}")
    data = sio.loadmat(mat_path)
    img = data["paviaU"]

    print(f"Data shape: {img.shape}")
    print(f"Data type: {img.dtype}")
    print(f"Value range: [{img.min():.4f}, {img.max():.4f}]")

    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img.astype(np.float32)


def load_chikusei_dataset(mat_path):
    """Load Chikusei hyperspectral dataset (supports both old and v7.3 MATLAB formats)"""
    print(f"Loading Chikusei dataset from: {mat_path}")

    # Try loading with scipy first (for older MATLAB formats)
    try:
        data = sio.loadmat(mat_path)
        print("Loaded using scipy (MATLAB < v7.3)")

        # Find the data key
        keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"Available keys: {keys}")

        # Try common key names
        for key_name in ['chikusei', 'data', 'img', 'hsi']:
            if key_name in data:
                img = np.array(data[key_name])
                break
        else:
            # Use first non-metadata key
            img = np.array(data[keys[0]])

    except NotImplementedError:
        # File is MATLAB v7.3 format, use h5py
        print("Loading using h5py (MATLAB v7.3 format)")
        with h5py.File(mat_path, 'r') as f:
            keys = [k for k in f.keys() if not k.startswith('__')]
            print(f"Available keys: {keys}")

            key_name = next((k for k in ['chikusei', 'data', 'img', 'hsi'] if k in f), keys[0])
            dset = f[key_name]
            full_shape = dset.shape
            print(f"Full HDF5 shape: {full_shape}")

            # HDF5/MATLAB v7.3 stores as (bands, cols, rows) — lazy-load only
            # first MAX_ROWS rows to keep RAM under ~1 GB.
            MAX_ROWS = 512    # 512×2335×128 float32 ≈ 310 MB; safe for 8 GB RAM
            if len(full_shape) == 3:
                # HDF5/MATLAB v7.3 stores Chikusei as (C=128, W=2335, H=2517)
                h_avail = full_shape[2]
                h_load  = min(MAX_ROWS, h_avail)
                raw = dset[:, :, :h_load]                     # (C, W, h_load)
                img = np.transpose(raw, (2, 1, 0)).astype(np.float32)  # (h_load, W, C) float32
                del raw
                print(f"Loaded sub-region: {img.shape}  ({img.nbytes/1024**2:.0f} MB, float32)")
            else:
                img = np.array(dset).astype(np.float32)
                if img.ndim == 3:
                    img = np.transpose(img, (2, 1, 0))

    print(f"Data shape: {img.shape}")
    print(f"Data type: {img.dtype}")
    print(f"Value range: [{img.min():.4f}, {img.max():.4f}]")

    # Normalize to [0, 1] in-place (avoids allocating a second large array)
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    vmin = float(img.min())
    vmax = float(img.max())
    img -= vmin
    img /= (vmax - vmin + 1e-8)

    return img


def extract_patches_with_spatial_separation(img, patch_size=64, overlap=32, scale_factor=2):
    """
    Extract patches with SPATIAL SEPARATION to prevent data leakage.

    Strategy:
    - Split image into 3 spatial regions (80% train, 10% val, 10% test)
    - Extract overlapping patches WITHIN each region
    - Ensures no test patch overlaps with training patches

    Args:
        img: (H, W, C) numpy array
        patch_size: Size of HR patches
        overlap: Overlap between patches
        scale_factor: Downsampling factor (2 or 4)

    Returns:
        (train_lr, train_hr), (val_lr, val_hr), (test_lr, test_hr)
    """
    H, W, C = img.shape

    print(f"\n{'='*70}")
    print("EXTRACTING PATCHES WITH SPATIAL SEPARATION")
    print(f"{'='*70}")
    print(f"Image: {H}x{W}x{C}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Overlap: {overlap} pixels")
    print(f"Scale factor: {scale_factor}")

    # Validate inputs
    assert scale_factor in [2, 4], f"Scale factor must be 2 or 4, got {scale_factor}"
    assert patch_size > 0, f"Patch size must be positive, got {patch_size}"
    assert 0 <= overlap < patch_size, f"Overlap must be in [0, {patch_size}), got {overlap}"

    # Crop image so both dimensions are divisible by scale_factor (handles odd sizes like Chikusei 2517x2335)
    H_crop = (H // scale_factor) * scale_factor
    W_crop = (W // scale_factor) * scale_factor
    if H_crop != H or W_crop != W:
        img = img[:H_crop, :W_crop, :]
        H, W = H_crop, W_crop
        print(f"Cropped to {H}x{W}x{C} (divisible by scale {scale_factor})")

    stride = patch_size - overlap
    print(f"Stride: {stride}")

    # CRITICAL FIX: Split image spatially FIRST (prevent data leakage)
    train_end = int(0.8 * W)
    val_end = int(0.9 * W)

    train_region = img[:, :train_end, :]
    val_region = img[:, train_end:val_end, :]
    test_region = img[:, val_end:, :]

    print(f"\nSpatial regions (NO LEAKAGE):")
    print(f"  Train region: {train_region.shape} (0 to {train_end})")
    print(f"  Val region: {val_region.shape} ({train_end} to {val_end})")
    print(f"  Test region: {test_region.shape} ({val_end} to {W})")

    def extract_from_region(region, name):
        """Extract overlapping patches from a single region"""
        h, w, c = region.shape
        patches = []

        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = region[i:i+patch_size, j:j+patch_size, :]
                if patch.shape == (patch_size, patch_size, c):
                    patches.append(patch)

        patches = np.array(patches)
        print(f"  {name}: {len(patches)} patches")
        return patches

    # Extract patches from each region separately
    print(f"\nExtracting patches:")
    train_patches = extract_from_region(train_region, "Train")
    val_patches = extract_from_region(val_region, "Val")
    test_patches = extract_from_region(test_region, "Test")

    # Create LR-HR pairs
    def create_lr_hr_pairs(patches, s):
        """Create LR-HR pairs by downsampling"""
        lr_list, hr_list = [], []

        for p in patches:
            # HR patch (original)
            hr = torch.from_numpy(p).permute(2, 0, 1).unsqueeze(0).float()

            # LR patch (downsampled)
            lr = F.interpolate(hr, scale_factor=1/s, mode='bilinear', align_corners=False)

            lr_list.append(lr.squeeze(0).permute(1, 2, 0).numpy())
            hr_list.append(hr.squeeze(0).permute(1, 2, 0).numpy())

        return np.array(lr_list), np.array(hr_list)

    print(f"\nCreating LR-HR pairs (scale={scale_factor}):")
    train_lr, train_hr = create_lr_hr_pairs(train_patches, scale_factor)
    val_lr, val_hr = create_lr_hr_pairs(val_patches, scale_factor)
    test_lr, test_hr = create_lr_hr_pairs(test_patches, scale_factor)

    print(f"\nFinal shapes:")
    print(f"  Train: LR {train_lr.shape}, HR {train_hr.shape}")
    print(f"  Val:   LR {val_lr.shape}, HR {val_hr.shape}")
    print(f"  Test:  LR {test_lr.shape}, HR {test_hr.shape}")
    print(f"{'='*70}\n")

    return (train_lr, train_hr), (val_lr, val_hr), (test_lr, test_hr)


def create_dataloaders_overlap(
    dataset="pavia",
    mat_path=None,
    batch_size=8,
    patch_size=32,
    overlap=16,
    scale=2,
    num_workers=0
):
    """
    Create dataloaders with overlapping patches and spatial separation

    Args:
        dataset: 'pavia' or 'chikusei'
        mat_path: Path to .mat file
        batch_size: Batch size
        patch_size: HR patch size
        overlap: Overlap between patches
        scale: Scale factor (2 or 4)
        num_workers: Number of workers for dataloader

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load dataset
    if dataset == "pavia":
        img = load_pavia_dataset(mat_path)
    elif dataset == "chikusei":
        img = load_chikusei_dataset(mat_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Extract patches with spatial separation
    (train_lr, train_hr), (val_lr, val_hr), (test_lr, test_hr) = \
        extract_patches_with_spatial_separation(img, patch_size, overlap, scale)

    # Create datasets
    train_dataset = HSIPatchDataset(train_lr, train_hr, augment=True)
    val_dataset = HSIPatchDataset(val_lr, val_hr, augment=False)
    test_dataset = HSIPatchDataset(test_lr, test_hr, augment=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"DataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


# Test
if __name__ == "__main__":
    print("Testing overlapping patch dataset loader...")

    # Use smaller patch size for testing
    train_loader, val_loader, test_loader = create_dataloaders_overlap(
        dataset="pavia",
        mat_path="pavia/PaviaU.mat",
        batch_size=4,
        patch_size=32,  # Smaller for testing
        overlap=16,
        scale=2
    )

    # Test train loader
    batch = next(iter(train_loader))
    print(f"\nTrain batch:")
    print(f"  LR-HSI: {batch['lr_hsi'].shape}")
    print(f"  HR-PAN: {batch['hr_pan'].shape}")
    print(f"  HR-HSI: {batch['hr_hsi'].shape}")

    # Test val loader (if not empty)
    if len(val_loader) > 0:
        batch = next(iter(val_loader))
        print(f"\nVal batch:")
        print(f"  LR-HSI: {batch['lr_hsi'].shape}")
        print(f"  HR-PAN: {batch['hr_pan'].shape}")
        print(f"  HR-HSI: {batch['hr_hsi'].shape}")
    else:
        print("\n[WARNING] Validation set is empty (dataset too small)")

    # Test test loader (if not empty)
    if len(test_loader) > 0:
        batch = next(iter(test_loader))
        print(f"\nTest batch:")
        print(f"  LR-HSI: {batch['lr_hsi'].shape}")
        print(f"  HR-PAN: {batch['hr_pan'].shape}")
        print(f"  HR-HSI: {batch['hr_hsi'].shape}")
    else:
        print("[WARNING] Test set is empty (dataset too small)")

    print("\n[SUCCESS] Dataset loader test passed!")
