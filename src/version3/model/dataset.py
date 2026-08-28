"""
Dataset Loader for Hyperspectral Pansharpening
Implements Wald's protocol (CORRECT version)

Datasets:
- Pavia University
- Chikusei

Key properties:
✔ Global min–max normalization only
✔ NO per-patch Z-score normalization
✔ PAN and HSI share same scale
✔ Geometric augmentation only
✔ Radiometrically consistent
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
import cv2
import h5py
from scipy.ndimage import gaussian_filter


# ============================================================================
# Dataset
# ============================================================================

class PansharpeningDataset(Dataset):
    """
    Wald protocol based hyperspectral pansharpening dataset
    """

    def __init__(
        self,
        hsi_data,                 # (H, W, C) in [0,1]
        patch_size=64,            # LR patch size
        scale=4,                  # Upsampling factor
        overlap=0.5,
        augment=True
    ):
        super().__init__()

        self.hsi_data = hsi_data.astype(np.float32)
        self.patch_size = patch_size
        self.scale = scale
        self.hr_patch = patch_size * scale
        self.overlap = overlap
        self.augment = augment

        self.patches = self._extract_hr_patches()

    # ------------------------------------------------------------------
    def _extract_hr_patches(self):
        H, W, _ = self.hsi_data.shape
        stride = int(self.hr_patch * (1 - self.overlap))

        patches = []
        for i in range(0, H - self.hr_patch + 1, stride):
            for j in range(0, W - self.hr_patch + 1, stride):
                patch = self.hsi_data[i:i+self.hr_patch, j:j+self.hr_patch]
                patches.append(patch)

        return patches

    # ------------------------------------------------------------------
    def _generate_lr_hsi(self, hr_hsi):
        """
        Wald protocol LR-HSI generation
        """
        # Gaussian blur (spatial only)
        lr = gaussian_filter(hr_hsi, sigma=[1.0, 1.0, 0])

        H, W, C = lr.shape
        lr = cv2.resize(
            lr,
            (W // self.scale, H // self.scale),
            interpolation=cv2.INTER_CUBIC
        )
        return lr

    # ------------------------------------------------------------------
    def _generate_hr_pan(self, hr_hsi):
        """
        PAN = mean over spectral bands
        """
        return np.mean(hr_hsi, axis=2, keepdims=True)

    # ------------------------------------------------------------------
    def _augment(self, lr, pan, hr):
        """
        Geometric augmentation ONLY
        """
        if np.random.rand() < 0.5:
            lr = np.fliplr(lr)
            pan = np.fliplr(pan)
            hr = np.fliplr(hr)

        if np.random.rand() < 0.5:
            lr = np.flipud(lr)
            pan = np.flipud(pan)
            hr = np.flipud(hr)

        k = np.random.randint(0, 4)
        if k > 0:
            lr = np.rot90(lr, k)
            pan = np.rot90(pan, k)
            hr = np.rot90(hr, k)

        return lr.copy(), pan.copy(), hr.copy()

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.patches)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        hr_hsi = self.patches[idx]

        lr_hsi = self._generate_lr_hsi(hr_hsi)
        hr_pan = self._generate_hr_pan(hr_hsi)

        if self.augment:
            lr_hsi, hr_pan, hr_hsi = self._augment(lr_hsi, hr_pan, hr_hsi)

        # Safety clip
        lr_hsi = np.clip(lr_hsi, 0.0, 1.0)
        hr_pan = np.clip(hr_pan, 0.0, 1.0)
        hr_hsi = np.clip(hr_hsi, 0.0, 1.0)

        # Convert to torch (C, H, W)
        lr_hsi = torch.from_numpy(lr_hsi.transpose(2, 0, 1)).float()
        hr_pan = torch.from_numpy(hr_pan.transpose(2, 0, 1)).float()
        hr_hsi = torch.from_numpy(hr_hsi.transpose(2, 0, 1)).float()

        return {
            "lr_hsi": lr_hsi,
            "hr_pan": hr_pan,
            "hr_hsi": hr_hsi
        }


# ============================================================================
# Dataset loaders
# ============================================================================

def load_pavia_dataset(mat_path, train_ratio=0.8):
    data = sio.loadmat(mat_path)
    hsi = data["paviaU"]

    # Global min–max normalization
    hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min())

    H = hsi.shape[0]
    split = int(H * train_ratio)

    return hsi[:split], hsi[split:]


def load_chikusei_dataset(mat_path, train_ratio=0.8):
    try:
        data = sio.loadmat(mat_path)
        hsi = data[list(data.keys())[-1]]
    except NotImplementedError:
        with h5py.File(mat_path, "r") as f:
            key = [k for k in f.keys() if not k.startswith("__")][0]
            hsi = np.array(f[key]).T

    hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min())

    H = hsi.shape[0]
    split = int(H * train_ratio)

    return hsi[:split], hsi[split:]


# ============================================================================
# Dataloaders
# ============================================================================

def create_dataloaders(
    dataset="pavia",
    mat_path=None,
    batch_size=8,
    patch_size=64,
    scale=4,
    num_workers=4
):
    if dataset == "pavia":
        train_hsi, val_hsi = load_pavia_dataset(mat_path)
    elif dataset == "chikusei":
        train_hsi, val_hsi = load_chikusei_dataset(mat_path)
    else:
        raise ValueError("Unknown dataset")

    train_set = PansharpeningDataset(
        train_hsi,
        patch_size=patch_size,
        scale=scale,
        overlap=0.5,
        augment=True
    )

    # For validation, use some overlap if dataset is small to ensure we get patches
    # Check validation image size
    val_h, val_w = val_hsi.shape[0], val_hsi.shape[1]
    hr_patch_size = patch_size * scale

    # If validation image is barely larger than patch, use overlap
    if val_h < hr_patch_size * 2 or val_w < hr_patch_size * 2:
        val_overlap = 0.5  # Use 50% overlap for small validation sets
        print(f"[INFO] Small validation set detected ({val_h}x{val_w}), using overlap={val_overlap}")
    else:
        val_overlap = 0.0

    val_set = PansharpeningDataset(
        val_hsi,
        patch_size=patch_size,
        scale=scale,
        overlap=val_overlap,
        augment=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders(
        dataset="pavia",
        mat_path="pavia/PaviaU.mat",
        batch_size=4
    )

    batch = next(iter(train_loader))
    print(batch["lr_hsi"].shape)
    print(batch["hr_pan"].shape)
    print(batch["hr_hsi"].shape)
    print("Dataset loader test successful!")
    
    batch = next(iter(val_loader))
    print(batch["lr_hsi"].shape)
    print(batch["hr_pan"].shape)
    print(batch["hr_hsi"].shape)
    print("Dataset loader test successful!")



    