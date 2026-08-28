import torch
import torch.optim as optim
from vmamba_pansharp import VMambaPansharp
from dataset_loader_overlap import create_dataloaders_overlap
from loss_functions import CompositeLoss, compute_psnr, compute_sam_metric, compute_ergas

print("Loading data...")
train_loader, _, _ = create_dataloaders_overlap('pavia', 'pavia/PaviaU.mat', 1, 16, 8, 2)

print("\nCreating model...")
model = VMambaPansharp(103, 103, 32, 2, [1, 2, 2, 2])
model.train()

print("\nCreating optimizer and loss...")
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = CompositeLoss(lambda_l1=1.0, lambda_sam=0.1, lambda_edge=0.05, lambda_ssim=0.1)

print("\nTraining for 10 iterations...")
for i, batch in enumerate(train_loader):
    if i >= 10:  # Only 10 iterations
        break

    lr_hsi = batch['lr_hsi']
    hr_pan = batch['hr_pan']
    hr_hsi = batch['hr_hsi']

    # Forward
    optimizer.zero_grad()
    output = model(lr_hsi, hr_pan)

    # Loss
    loss = criterion(output, hr_hsi)

    # Metrics
    psnr = compute_psnr(output, hr_hsi)
    sam = compute_sam_metric(output, hr_hsi)
    ergas = compute_ergas(output, hr_hsi, scale=2)

    # Check for NaN
    if torch.isnan(loss).any():
        print(f"\n[ERROR] Iteration {i}: Loss is NaN!")
        break

    # Backward
    loss.backward()

    # Check gradients for NaN
    has_nan_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"[ERROR] Iteration {i}: NaN gradient in {name}")
            has_nan_grad = True
            break

    if has_nan_grad:
        break

    # Update
    optimizer.step()

    # Log
    print(f"Iter {i}: Loss={loss.item():.4f}, PSNR={psnr:.2f}dB, SAM={sam:.2f}°, ERGAS={ergas:.2f}")

print("\n[SUCCESS] Training completed 10 iterations without NaN!")
