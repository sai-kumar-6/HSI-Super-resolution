"""
Quick Start Script for VMamba-Pansharp
Tests the complete pipeline on a small sample
"""

import torch
import numpy as np
from vmamba_pansharp import VMambaPansharp
from loss_functions import CompositeLoss, compute_psnr, compute_sam_metric, compute_ergas


def test_model():
    """Test model with dummy data"""
    print("="*60)
    print("VMamba-Pansharp Quick Start Test")
    print("="*60)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n1. Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Model parameters
    num_channels = 102  # Pavia-like
    d_model = 64
    scale = 4
    batch_size = 1
    lr_size = 32
    hr_size = lr_size * scale

    print(f"\n2. Model Configuration:")
    print(f"   Input channels: {num_channels}")
    print(f"   Feature dimension: {d_model}")
    print(f"   Scale factor: {scale}")
    print(f"   LR size: {lr_size}×{lr_size}")
    print(f"   HR size: {hr_size}×{hr_size}")

    # Create model
    print(f"\n3. Creating VMamba-Pansharp model...")
    model = VMambaPansharp(
        in_channels=num_channels,
        out_channels=num_channels,
        d_model=d_model,
        scale=scale,
        num_blocks=[3, 4, 4, 3]
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1e6:.2f} MB (float32)")

    # Create dummy data
    print(f"\n4. Creating test data...")
    lr_hsi = torch.randn(batch_size, num_channels, lr_size, lr_size).to(device)
    hr_pan = torch.randn(batch_size, 1, hr_size, hr_size).to(device)
    hr_hsi_gt = torch.randn(batch_size, num_channels, hr_size, hr_size).to(device)

    print(f"   LR-HSI shape: {lr_hsi.shape}")
    print(f"   HR-PAN shape: {hr_pan.shape}")
    print(f"   HR-HSI (GT) shape: {hr_hsi_gt.shape}")

    # Forward pass
    print(f"\n5. Testing forward pass...")
    model.eval()
    with torch.no_grad():
        hr_hsi_pred = model(lr_hsi, hr_pan)

    print(f"   Output shape: {hr_hsi_pred.shape}")
    print(f"   [OK] Forward pass successful!")

    # Test loss functions
    print(f"\n6. Testing loss functions...")
    criterion = CompositeLoss()
    loss, loss_dict = criterion(hr_hsi_pred, hr_hsi_gt, return_components=True)

    print(f"   Total Loss: {loss_dict['total']:.4f}")
    print(f"   - L1:   {loss_dict['l1']:.4f}")
    print(f"   - SAM:  {loss_dict['sam']:.4f}")
    print(f"   - Edge: {loss_dict['edge']:.4f}")
    print(f"   - SSIM: {loss_dict['ssim']:.4f}")
    print(f"   [OK] Loss computation successful!")

    # Test metrics
    print(f"\n7. Testing evaluation metrics...")
    psnr = compute_psnr(hr_hsi_pred, hr_hsi_gt)
    sam = compute_sam_metric(hr_hsi_pred, hr_hsi_gt)
    ergas = compute_ergas(hr_hsi_pred, hr_hsi_gt, scale=scale)

    print(f"   PSNR:  {psnr:.2f} dB")
    print(f"   SAM:   {sam:.2f} degrees")
    print(f"   ERGAS: {ergas:.4f}")
    print(f"   [OK] Metrics computation successful!")

    # Test backward pass
    print(f"\n8. Testing backward pass...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Forward
    hr_hsi_pred = model(lr_hsi, hr_pan)
    loss = criterion(hr_hsi_pred, hr_hsi_gt)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"   All parameters have gradients: {has_grad}")

    # Optimizer step
    optimizer.step()
    print(f"   [OK] Backward pass and optimization successful!")

    # Memory usage
    if device.type == 'cuda':
        print(f"\n9. GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"   Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60)
    print("\nYou can now:")
    print("1. Train on Pavia: python train_vmamba_pansharp.py --dataset pavia")
    print("2. Train on Chikusei: python train_vmamba_pansharp.py --dataset chikusei")
    print("3. See VMAMBA_PANSHARP_README.md for full documentation")
    print("="*60)


if __name__ == '__main__':
    test_model()
