"""
baseline_models.py  —  Version 3
Factory function for all three model variants.

Variants
────────
  'old'      →  VMambaPansharp        (V1: CrossAttention + Sobel + PixelShuffle)
  'improved' →  ImprovedVMambaPansharp (V2: CrossMamba + LearnableHF + ResidualUps)
  'v3'       →  V3VMambaPansharp      (V3: SDIM + MultiScaleSpectral + SpectralSSM)
"""

import os
import sys
import torch
import torch.nn as nn

_HERE    = os.path.dirname(os.path.abspath(__file__))   # version3/model/
_V3      = os.path.dirname(_HERE)                        # version3/
_PROJECT = os.path.dirname(_V3)                           # src/
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, 'scripts'))
sys.path.insert(0, _HERE)

# VMambaPansharp (V1) is duplicated into this version's own model.py, so this
# is a plain same-version import, not a cross-version one.
from model                    import VMambaPansharp, V3VMambaPansharp
from vmamba_pansharp_improved import ImprovedVMambaPansharp


def count_parameters(model: nn.Module):
    """Return (total_params, trainable_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_vmamba_model(
    variant:    str,
    in_ch:      int,
    out_ch:     int,
    scale:      int  = 4,
    d_model:    int  = 32,
    d_state:    int  = 8,
    num_blocks: list = None,
) -> nn.Module:
    """
    Instantiate a VMamba pansharpening model.

    Parameters
    ----------
    variant    : 'old' | 'improved' | 'v3'
    in_ch      : number of HSI spectral bands  (128 for Chikusei)
    out_ch     : number of output bands        (same as in_ch)
    scale      : spatial upsampling factor     (4 for Chikusei)
    d_model    : feature dimension             (32 = OOM-safe on CPU)
    d_state    : SSM state size                (8 default)
    num_blocks : list of ints per stage        ([1,1,1,1] = lightweight)
    """
    if num_blocks is None:
        num_blocks = [1, 1, 1, 1] if d_model <= 32 else [2, 2, 2, 2]

    if variant == 'old':
        return VMambaPansharp(
            in_channels  = in_ch,
            out_channels = out_ch,
            d_model      = d_model,
            scale        = scale,
            num_blocks   = num_blocks,
        )

    elif variant == 'improved':
        return ImprovedVMambaPansharp(
            in_channels  = in_ch,
            out_channels = out_ch,
            d_model      = d_model,
            d_state      = d_state,
            scale        = scale,
            num_blocks   = num_blocks,
        )

    elif variant == 'v3':
        return V3VMambaPansharp(
            in_channels  = in_ch,
            out_channels = out_ch,
            d_model      = d_model,
            d_state      = d_state,
            scale        = scale,
            num_blocks   = num_blocks,
        )

    else:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose: 'old', 'improved', 'v3'"
        )


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\nBaseline Models — V3 Edition")
    print("=" * 55)

    for variant in ['old', 'improved', 'v3']:
        model = create_vmamba_model(
            variant    = variant,
            in_ch      = 128,
            out_ch     = 128,
            scale      = 4,
            d_model    = 32,
            d_state    = 4,
            num_blocks = [1, 1, 1, 1],
        ).eval()

        lr  = torch.randn(1, 128, 8, 8)
        pan = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            out = model(lr, pan)

        total, _ = count_parameters(model)
        status   = 'OK' if out.shape == (1, 128, 32, 32) else 'FAIL'
        print(f"  Variant: {variant:<10}  Params: {total:>10,}  "
              f"Output: {tuple(out.shape)}  [{status}]")

    print("=" * 55)
