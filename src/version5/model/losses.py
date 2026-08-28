"""
losses.py  --  V5 Edition
Imports base losses from project root and V4's SpectralGradientLoss + CompositeLossV4.
CompositeLossV5 is an alias for CompositeLossV4 (same objectives, improved scan only).
"""
from __future__ import annotations
import os
import sys
import importlib.util

_HERE    = os.path.dirname(os.path.abspath(__file__))   # version5/model/
_PROJECT = os.path.dirname(os.path.dirname(_HERE))         # src/
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, 'scripts'))


def _load_module(name, filepath):
    """Load a module by absolute file path under a unique sys.modules key
    (every version's loss file is now named losses.py, so a plain
    `import losses` would collide across versions)."""
    if name in sys.modules:
        return sys.modules[name]
    spec   = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


from loss_functions import (          # noqa: F401
    L1ReconstructionLoss,
    SpectralAngleMapper,
    EdgePreservationLoss,
    SSIMLoss,
    compute_psnr,
    compute_sam_metric,
    compute_ergas,
    compute_ssim,
)

_v4_losses = _load_module('v4_losses', os.path.join(_PROJECT, 'version4', 'model', 'losses.py'))
SpectralGradientLoss = _v4_losses.SpectralGradientLoss
CompositeLossV4      = _v4_losses.CompositeLossV4

# Alias
CompositeLossV5 = CompositeLossV4

# Legacy name aliases used by some callers
L1Loss   = L1ReconstructionLoss
SAMLoss  = SpectralAngleMapper
EdgeLoss = EdgePreservationLoss

__all__ = [
    'L1ReconstructionLoss', 'L1Loss',
    'SpectralAngleMapper',  'SAMLoss',
    'EdgePreservationLoss', 'EdgeLoss',
    'SSIMLoss',
    'SpectralGradientLoss',
    'CompositeLossV4', 'CompositeLossV5',
    'compute_psnr', 'compute_sam_metric', 'compute_ergas', 'compute_ssim',
]
