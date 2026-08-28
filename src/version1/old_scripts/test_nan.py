import torch
from vmamba_pansharp import VMambaPansharp
from dataset_loader_overlap import create_dataloaders_overlap

print("Loading data...")
train_loader, _, _ = create_dataloaders_overlap('pavia', 'pavia/PaviaU.mat', 1, 16, 8, 2)
batch = next(iter(train_loader))

print("\nCreating model...")
model = VMambaPansharp(103, 103, 32, 2, [2,3,3,2])
model.eval()

print("\nTesting forward pass...")
with torch.no_grad():
    output = model(batch['lr_hsi'], batch['hr_pan'])

print(f"\n[CHECK] Output:")
print(f"  Shape: {output.shape}")
print(f"  Min: {output.min():.6f}")
print(f"  Max: {output.max():.6f}")
print(f"  Mean: {output.mean():.6f}")
print(f"  Has NaN: {torch.isnan(output).any()}")
print(f"  Has Inf: {torch.isinf(output).any()}")

if torch.isnan(output).any():
    print(f"\n[ERROR] Found {torch.isnan(output).sum()} NaN values in output!")
else:
    print("\n[OK] No NaN in forward pass!")
