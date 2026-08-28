import torch
from vmamba_pansharp import VMambaPansharp
from dataset_loader_overlap import create_dataloaders_overlap
from loss_functions import CompositeLoss, compute_psnr, compute_sam_metric, compute_ergas

print("Loading data...")
train_loader, _, _ = create_dataloaders_overlap('pavia', 'pavia/PaviaU.mat', 1, 16, 8, 2)
batch = next(iter(train_loader))

print("\nCreating model...")
model = VMambaPansharp(103, 103, 32, 2, [2,3,3,2])
model.train()  # Training mode

print("\nCreating loss...")
criterion = CompositeLoss(lambda_l1=1.0, lambda_sam=0.1, lambda_edge=0.05, lambda_ssim=0.1)

print("\nTesting forward + loss...")
lr_hsi = batch['lr_hsi']
hr_pan = batch['hr_pan']
hr_hsi = batch['hr_hsi']

# Forward pass
output = model(lr_hsi, hr_pan)

print(f"\n[CHECK] Model output:")
print(f"  Shape: {output.shape}")
print(f"  Min: {output.min():.6f}, Max: {output.max():.6f}")
print(f"  Has NaN: {torch.isnan(output).any()}")

# Compute loss
print(f"\n[CHECK] Computing loss...")
loss = criterion(output, hr_hsi)

print(f"\n[CHECK] Loss:")
print(f"  Value: {loss.item():.6f}")
print(f"  Is NaN: {torch.isnan(loss).any()}")

# Test individual metrics
print(f"\n[CHECK] Individual metrics:")
psnr = compute_psnr(output, hr_hsi)
print(f"  PSNR: {psnr:.4f}")

sam = compute_sam_metric(output, hr_hsi)
print(f"  SAM: {sam:.4f} (is_nan={torch.isnan(torch.tensor(sam))})")

ergas = compute_ergas(output, hr_hsi, scale=2)
print(f"  ERGAS: {ergas:.4f}")

if torch.isnan(loss):
    print("\n[ERROR] Loss is NaN!")
else:
    print("\n[OK] Loss computation successful!")

    # Test backward pass
    print("\n[CHECK] Testing backward pass...")
    loss.backward()

    # Check gradients
    has_nan_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"  [ERROR] NaN gradient in {name}")
            has_nan_grad = True

    if not has_nan_grad:
        print("  [OK] No NaN in gradients!")
    else:
        print("  [ERROR] Found NaN in gradients!")
