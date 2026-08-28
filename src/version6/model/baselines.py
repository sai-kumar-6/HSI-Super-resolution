"""
baseline_models.py  —  V6 Edition
Factory for all model variants v1-v6.

VMambaPansharp (V1), V3VMambaPansharp (V3), V4VMambaPansharp (V4), and
V5VMambaPansharp (V5) are all duplicated into this version's own model.py,
so every variant here is a plain same-version import.
"""
from __future__ import annotations
import os, sys

_HERE    = os.path.dirname(os.path.abspath(__file__))   # version6/model/
_V6      = os.path.dirname(_HERE)                         # version6/
_PROJECT = os.path.dirname(_V6)                            # src/

for p in [
    _PROJECT,
    os.path.join(_PROJECT, 'scripts'),
    _HERE,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model import (
    VMambaPansharp,
    V3VMambaPansharp,
    V4VMambaPansharp,
    V5VMambaPansharp,
    V6VMambaPansharp,
)


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
        return VMambaPansharp(
            in_channels=in_ch, out_channels=out_ch,
            scale=scale, d_model=d_model, num_blocks=num_blocks,
        )
    elif variant == 'improved':
        from vmamba_pansharp_improved import ImprovedVMambaPansharp
        return ImprovedVMambaPansharp(
            in_channels=in_ch, out_channels=out_ch,
            scale=scale, d_model=d_model, d_state=d_state, num_blocks=num_blocks,
        )
    elif variant == 'v3':
        return V3VMambaPansharp(
            in_channels=in_ch, out_channels=out_ch,
            scale=scale, d_model=d_model, d_state=d_state, num_blocks=num_blocks,
        )
    elif variant == 'v4':
        return V4VMambaPansharp(
            in_channels=in_ch, out_channels=out_ch,
            scale=scale, d_model=d_model, d_state=d_state,
            num_blocks=num_blocks, alpha_init=0.1, beta_init=0.5,
        )
    elif variant == 'v5':
        return V5VMambaPansharp(
            in_channels=in_ch, out_channels=out_ch,
            scale=scale, d_model=d_model, d_state=d_state,
            num_blocks=num_blocks, alpha_init=0.1, beta_init=0.5,
        )
    elif variant == 'v6':
        return V6VMambaPansharp(
            in_channels=in_ch, out_channels=out_ch,
            scale=scale, d_model=d_model, d_state=d_state,
            num_blocks=num_blocks,
            alpha_init=0.05,   # reduced PAN injection
            beta_init=0.3,     # reduced residual gate
        )
    else:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            "Choose: 'old', 'improved', 'v3', 'v4', 'v5', 'v6'"
        )


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
