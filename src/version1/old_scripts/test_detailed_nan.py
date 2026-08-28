import torch
import torch.nn as nn
from vmamba_pansharp import VMambaPansharp
from dataset_loader_overlap import create_dataloaders_overlap

print("Loading data...")
train_loader, val_loader, test_loader = create_dataloaders_overlap(
    dataset='pavia',
    mat_path='pavia/PaviaU.mat',
    batch_size=1,
    patch_size=16,
    overlap=8,
    scale=2,
    num_workers=0
)

print("\nCreating model...")
model = VMambaPansharp(103, 103, 32, 2, [1, 2, 2, 2])
model.train()

print("\nGetting a batch...")
batch = next(iter(train_loader))
lr_hsi = batch['lr_hsi']
hr_pan = batch['hr_pan']
hr_hsi = batch['hr_hsi']

print(f"\n[CHECK] Input data:")
print(f"  LR-HSI: shape={lr_hsi.shape}, min={lr_hsi.min():.6f}, max={lr_hsi.max():.6f}, has_nan={torch.isnan(lr_hsi).any()}")
print(f"  HR-PAN: shape={hr_pan.shape}, min={hr_pan.min():.6f}, max={hr_pan.max():.6f}, has_nan={torch.isnan(hr_pan).any()}")
print(f"  HR-HSI: shape={hr_hsi.shape}, min={hr_hsi.min():.6f}, max={hr_hsi.max():.6f}, has_nan={torch.isnan(hr_hsi).any()}")

# Test forward pass with hooks to track intermediate values
intermediate_outputs = {}

def hook_fn(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            intermediate_outputs[name] = {
                'shape': output.shape,
                'min': output.min().item(),
                'max': output.max().item(),
                'mean': output.mean().item(),
                'std': output.std().item(),
                'has_nan': torch.isnan(output).any().item(),
                'has_inf': torch.isinf(output).any().item()
            }
    return hook

# Register hooks
model.hsi_encoder.register_forward_hook(hook_fn('hsi_encoder'))
model.pan_encoder.register_forward_hook(hook_fn('pan_encoder'))
model.fusion.register_forward_hook(hook_fn('fusion'))
model.backbone.register_forward_hook(hook_fn('backbone'))
model.reconstruction.register_forward_hook(hook_fn('reconstruction'))

print("\n[CHECK] Running forward pass...")
output = model(lr_hsi, hr_pan)

print(f"\n[CHECK] Output:")
print(f"  Shape: {output.shape}")
print(f"  Min: {output.min():.6f}, Max: {output.max():.6f}")
print(f"  Mean: {output.mean():.6f}, Std: {output.std():.6f}")
print(f"  Has NaN: {torch.isnan(output).any()}")
print(f"  Has Inf: {torch.isinf(output).any()}")

print("\n[CHECK] Intermediate outputs:")
for name, stats in intermediate_outputs.items():
    print(f"\n  {name}:")
    for key, val in stats.items():
        print(f"    {key}: {val}")

# Test loss computation
print("\n[CHECK] Computing loss...")
mse_loss = nn.MSELoss()(output, hr_hsi)
print(f"  MSE Loss: {mse_loss.item():.6f}")
print(f"  Has NaN: {torch.isnan(mse_loss).any()}")

# Check if loss is causing issues
print("\n[CHECK] Testing backward pass...")
mse_loss.backward()

print("\n[CHECK] Checking gradients in PANEncoder...")
for name, param in model.pan_encoder.named_parameters():
    if param.grad is not None:
        has_nan = torch.isnan(param.grad).any()
        has_inf = torch.isinf(param.grad).any()
        grad_min = param.grad.min().item() if not has_nan else float('nan')
        grad_max = param.grad.max().item() if not has_nan else float('nan')
        grad_mean = param.grad.mean().item() if not has_nan else float('nan')

        print(f"  {name}:")
        print(f"    grad min={grad_min:.6f}, max={grad_max:.6f}, mean={grad_mean:.6f}")
        print(f"    has_nan={has_nan}, has_inf={has_inf}")

        if has_nan or has_inf:
            print(f"    [ERROR] Gradient has NaN or Inf!")
