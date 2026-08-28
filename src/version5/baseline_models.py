"""
baseline_models.py  —  V5 Edition
Factory that creates any of the 5 model variants:
  'old'      → V1 VMamba (vmamba_pansharp.py)
  'improved' → V2 Improved VMamba (vmamba_pansharp_improved.py)
  'v3'       → V3 VMamba (vmamba_pansharp_v3.py  — version3 folder)
  'v4'       → V4 VMamba with spectral fixes (vmamba_pansharp_v4.py)
  'v5'       → V5 VMamba with true parallel scan (vmamba_pansharp_v5.py)
"""
from __future__ import annotations
import os, sys

_HERE    = os.path.dirname(os.path.abspath(__file__))   # version5/
_PROJECT = os.path.dirname(_HERE)

# Path priority: version5 wins (inserted last → index 0)
sys.path.insert(0, os.path.join(_PROJECT, 'version3'))  # lowest priority
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, 'scripts'))
sys.path.insert(0, os.path.join(_PROJECT, 'version4'))
sys.path.insert(0, _HERE)                               # highest priority


def create_vmamba_model(
    variant: str,
    in_ch: int   = 128,
    out_ch: int  = 128,
    scale: int   = 4,
    d_model: int = 32,
    d_state: int = 8,
    num_blocks: list[int] | None = None,
):
    if num_blocks is None:
        num_blocks = [1, 1, 1, 1]

    variant = variant.lower()

    if variant == 'old':
        from vmamba_pansharp import VMambaPansharp
        return VMambaPansharp(
            in_channels=in_ch, out_channels=out_ch,
            scale=scale, d_model=d_model,
            num_blocks=num_blocks,
        )
    elif variant == 'improved':
        from vmamba_pansharp_improved import ImprovedVMambaPansharp
        return ImprovedVMambaPansharp(
            in_channels=in_ch, out_channels=out_ch,
            scale=scale, d_model=d_model, d_state=d_state,
            num_blocks=num_blocks,
        )
    elif variant == 'v3':
        from vmamba_pansharp_v3 import V3VMambaPansharp
        return V3VMambaPansharp(
            in_channels=in_ch, out_channels=out_ch,
            scale=scale, d_model=d_model, d_state=d_state,
            num_blocks=num_blocks,
        )
    elif variant == 'v4':
        from vmamba_pansharp_v4 import V4VMambaPansharp
        return V4VMambaPansharp(
            in_channels=in_ch, out_channels=out_ch,
            scale=scale, d_model=d_model, d_state=d_state,
            num_blocks=num_blocks,
            alpha_init=0.1, beta_init=0.5,
        )
    elif variant == 'v5':
        from vmamba_pansharp_v5 import V5VMambaPansharp
        return V5VMambaPansharp(
            in_channels=in_ch, out_channels=out_ch,
            scale=scale, d_model=d_model, d_state=d_state,
            num_blocks=num_blocks,
            alpha_init=0.1, beta_init=0.5,
        )
    else:
        raise ValueError(
            f"Unknown variant {variant!r}. "
            "Choose: 'old', 'improved', 'v3', 'v4', 'v5'"
        )


def count_parameters(model):
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
