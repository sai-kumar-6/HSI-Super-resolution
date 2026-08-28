"""
baseline_models.py
M.Tech Thesis – Hyperspectral Pansharpening

NOTE: CNN, UNet, and Transformer baselines have been REMOVED.
      Only the two VMamba variants (original and improved) are kept here
      as the main models for the thesis comparison.

Factory function `create_vmamba_model` instantiates either variant by name.
"""

import torch
import torch.nn as nn


# ============================================================================
# Utility
# ============================================================================

def count_parameters(model: nn.Module):
    """
    Count total and trainable parameters of a PyTorch model.

    Args:
        model : nn.Module

    Returns:
        total     : int – total number of parameters
        trainable : int – number of parameters with requires_grad=True
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
# Model Factory
# ============================================================================

def create_vmamba_model(
    variant:  str,
    in_ch:    int,
    out_ch:   int,
    scale:    int  = 4,
    d_model:  int  = 64,
    d_state:  int  = 16,
    num_blocks: list = None,
) -> nn.Module:
    """
    Factory function that returns a VMamba pansharpening model.

    Args:
        variant    : 'old'      → VMambaPansharp (baseline with CrossAttention)
                     'improved' → ImprovedVMambaPansharp (Cross-Mamba + Learnable HF + Residual Upsample)
        in_ch      : number of input spectral bands (e.g. 128 for Chikusei)
        out_ch     : number of output spectral bands (usually == in_ch)
        scale      : spatial upsampling factor (default 4)
        d_model    : internal feature dimension (default 64)
        d_state    : SSM state dimension (default 16)
        num_blocks : list of 4 ints, per-stage block counts for VMambaBackbone
                     (default [2, 3, 3, 2])

    Returns:
        model : nn.Module

    Raises:
        ValueError : if variant is not 'old' or 'improved'
    """
    if num_blocks is None:
        num_blocks = [2, 3, 3, 2]

    if variant == 'old':
        from vmamba_pansharp import VMambaPansharp
        model = VMambaPansharp(
            in_channels  = in_ch,
            out_channels = out_ch,
            d_model      = d_model,
            scale        = scale,
            num_blocks   = num_blocks,
        )
        return model

    elif variant == 'improved':
        from vmamba_pansharp_improved import ImprovedVMambaPansharp
        model = ImprovedVMambaPansharp(
            in_channels  = in_ch,
            out_channels = out_ch,
            d_model      = d_model,
            d_state      = d_state,
            scale        = scale,
            num_blocks   = num_blocks,
        )
        return model

    else:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Choose 'old' (VMambaPansharp) or 'improved' (ImprovedVMambaPansharp)."
        )


# ============================================================================
# Quick self-test
# ============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing baseline_models.py on {device}\n")

    in_ch  = 128   # Chikusei bands
    out_ch = 128
    scale  = 4
    B      = 2
    H_lr   = 16
    W_lr   = 16

    lr_hsi = torch.randn(B, in_ch, H_lr, W_lr).to(device)
    hr_pan = torch.randn(B, 1, H_lr * scale, W_lr * scale).to(device)

    for variant in ('old', 'improved'):
        print(f"  Variant: {variant}")
        model = create_vmamba_model(variant, in_ch, out_ch, scale=scale).to(device)
        total, trainable = count_parameters(model)
        print(f"    Parameters: {total:,}  (trainable: {trainable:,})")
        with torch.no_grad():
            out = model(lr_hsi, hr_pan)
        print(f"    Output shape: {tuple(out.shape)}")
        assert out.shape == (B, out_ch, H_lr * scale, W_lr * scale)
        print(f"    [OK]\n")

    print("All checks passed.")
