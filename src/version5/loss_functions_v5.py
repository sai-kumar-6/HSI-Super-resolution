"""
loss_functions_v5.py  --  V5 Edition
Imports base losses from project root and V4's SpectralGradientLoss + CompositeLossV4.
CompositeLossV5 is an alias for CompositeLossV4 (same objectives, improved scan only).
"""
from __future__ import annotations
import os
import sys

_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_V4      = os.path.join(_PROJECT, 'version4')
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _V4)

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
from loss_functions_v4 import (       # noqa: F401
    SpectralGradientLoss,
    CompositeLossV4,
)

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
