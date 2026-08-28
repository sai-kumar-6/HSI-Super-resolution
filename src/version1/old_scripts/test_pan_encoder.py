import torch
import torch.nn as nn
from vmamba_pansharp import PANEncoder

print("Creating PANEncoder...")
encoder = PANEncoder(d_model=32)
encoder.train()

print("\nCreating test input...")
# Simulate a PAN image patch
hr_pan = torch.randn(1, 1, 16, 16) * 0.5 + 0.5  # Values between 0-1
print(f"Input: min={hr_pan.min():.6f}, max={hr_pan.max():.6f}")

print("\nForward pass...")
output = encoder(hr_pan)
print(f"Output: shape={output.shape}, min={output.min():.6f}, max={output.max():.6f}")
print(f"Has NaN: {torch.isnan(output).any()}")
print(f"Has Inf: {torch.isinf(output).any()}")

print("\nComputing loss...")
target = torch.randn_like(output)
loss = nn.MSELoss()(output, target)
print(f"Loss: {loss.item():.6f}, has_nan={torch.isnan(loss).any()}")

print("\nBackward pass...")
loss.backward()

print("\nChecking gradients...")
has_nan = False
for name, param in encoder.named_parameters():
    if param.grad is not None:
        is_nan = torch.isnan(param.grad).any()
        is_inf = torch.isinf(param.grad).any()

        if is_nan or is_inf:
            print(f"  [ERROR] {name}: has_nan={is_nan}, has_inf={is_inf}")
            has_nan = True
        else:
            grad_min = param.grad.min().item()
            grad_max = param.grad.max().item()
            print(f"  [OK] {name}: min={grad_min:.6f}, max={grad_max:.6f}")

if not has_nan:
    print("\n[SUCCESS] No NaN gradients in isolated PANEncoder!")
else:
    print("\n[ERROR] PANEncoder alone produces NaN gradients!")
