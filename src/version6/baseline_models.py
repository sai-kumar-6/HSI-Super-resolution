"""
baseline_models.py  —  V6 Edition
Factory for all model variants v1-v6.
Uses explicit file-path loading to avoid sys.path shadowing.
"""
from __future__ import annotations
import os, sys, importlib.util

_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_V5      = os.path.join(_PROJECT, 'version5')

for p in [
    os.path.join(_PROJECT, 'version1'),
    os.path.join(_PROJECT, 'version3'),
    _PROJECT,
    os.path.join(_PROJECT, 'scripts'),
    os.path.join(_PROJECT, 'version4'),
    _V5,
    _HERE,
]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_module(name, filepath):
    """Load a Python module by absolute file path, bypassing sys.path shadowing."""
    if name in sys.modules:
        return sys.modules[name]
    spec   = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
            scale=scale, d_model=d_model, num_blocks=num_blocks,
        )
    elif variant == 'improved':
        from vmamba_pansharp_improved import ImprovedVMambaPansharp
        return ImprovedVMambaPansharp(
            in_channels=in_ch, out_channels=out_ch,
            scale=scale, d_model=d_model, d_state=d_state, num_blocks=num_blocks,
        )
    elif variant == 'v3':
        from vmamba_pansharp_v3 import V3VMambaPansharp
        return V3VMambaPansharp(
            in_channels=in_ch, out_channels=out_ch,
            scale=scale, d_model=d_model, d_state=d_state, num_blocks=num_blocks,
        )
    elif variant == 'v4':
        from vmamba_pansharp_v4 import V4VMambaPansharp
        return V4VMambaPansharp(
            in_channels=in_ch, out_channels=out_ch,
            scale=scale, d_model=d_model, d_state=d_state,
            num_blocks=num_blocks, alpha_init=0.1, beta_init=0.5,
        )
    elif variant == 'v5':
        from vmamba_pansharp_v5 import V5VMambaPansharp
        return V5VMambaPansharp(
            in_channels=in_ch, out_channels=out_ch,
            scale=scale, d_model=d_model, d_state=d_state,
            num_blocks=num_blocks, alpha_init=0.1, beta_init=0.5,
        )
    elif variant == 'v6':
        # Explicit file load — guarantees version6/vmamba_pansharp_v6.py is used
        _m = _load_module('vmamba_pansharp_v6',
                          os.path.join(_HERE, 'vmamba_pansharp_v6.py'))
        return _m.V6VMambaPansharp(
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
