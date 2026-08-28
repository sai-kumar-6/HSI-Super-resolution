"""
losses.py  —  Version 6 Edition
M.Tech Thesis: Hyperspectral Image Pansharpening

V6 Change: Spectral-Focused Loss Weights
─────────────────────────────────────────
Root cause of high SAM in V5: λ_sam was too low (0.10) relative to L1 (1.0).
The model learned to minimise pixel error but largely ignored spectral angle.

V6 corrects this by increasing SAM and SpectralGradient weights:

    CompositeLossV6:
        L = 1.0 × L1
          + 0.40 × SAM          ← raised from 0.10  (+300%)
          + 0.05 × Edge
          + 0.01 × SSIM
          + 0.10 × SpectralGrad ← raised from 0.05  (+100%)

Why these values:
  • λ_sam = 0.40  → SAM loss now ~40% of L1 signal strength (was ~10%)
                    Directly optimises the metric we evaluate on.
  • λ_spec = 0.10 → SpectralGradientLoss penalises wrong band-to-band slopes.
                    Doubles enforcement of spectral curve shape.
  • λ_l1 = 1.0    → Kept at 1.0 for radiometric accuracy (PSNR).
  • λ_edge = 0.05 → Unchanged; PAN spatial edges still needed.
  • λ_ssim = 0.01 → Kept low; structural similarity is secondary.

Expected effect:
  SAM:   6.2° → ~4–5°   (spectral distortion reduced)
  PSNR:  small drop acceptable (< 0.5 dB trade-off)
  ERGAS: improvement expected (ERGAS is band-MSE weighted, SAM improvement helps)
"""
from __future__ import annotations
import os
import sys
import importlib.util

_HERE    = os.path.dirname(os.path.abspath(__file__))   # version6/model/
_PROJECT = os.path.dirname(os.path.dirname(_HERE))         # src/
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, 'scripts'))

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ============================================================================
# V6 Composite Loss — Spectral-Focused Weights
# ============================================================================

class CompositeLossV6(nn.Module):
    """
    V6 composite loss with spectral-focused weights to reduce SAM.

    L = λ_l1   × L1            (pixel accuracy)
      + λ_sam   × SAM          (spectral angle — RAISED to 0.40)
      + λ_edge  × Edge         (spatial edge preservation)
      + λ_ssim  × SSIM         (structural similarity)
      + λ_spec  × SpectralGrad (band-to-band slope — RAISED to 0.10)

    Default weights:
        λ_l1=1.0, λ_sam=0.40, λ_edge=0.05, λ_ssim=0.01, λ_spec=0.10
    """

    def __init__(
        self,
        lambda_l1:   float = 1.0,
        lambda_sam:  float = 0.40,   # V5: 0.10 → V6: 0.40
        lambda_edge: float = 0.05,
        lambda_ssim: float = 0.01,
        lambda_spec: float = 0.10,   # V5: 0.05 → V6: 0.10
    ):
        super().__init__()
        self.l1   = L1ReconstructionLoss()
        self.sam  = SpectralAngleMapper()
        self.edge = EdgePreservationLoss()
        self.ssim = SSIMLoss()
        self.spec = SpectralGradientLoss()

        self.w_l1   = lambda_l1
        self.w_sam  = lambda_sam
        self.w_edge = lambda_edge
        self.w_ssim = lambda_ssim
        self.w_spec = lambda_spec

    def forward(self, pred, target, return_components=False):
        l1   = self.l1(pred, target)
        sam  = self.sam(pred, target)
        edge = self.edge(pred, target)
        ssim = self.ssim(pred, target)
        spec = self.spec(pred, target)

        total = (
            self.w_l1   * l1   +
            self.w_sam  * sam  +
            self.w_edge * edge +
            self.w_ssim * ssim +
            self.w_spec * spec
        )

        if return_components:
            return total, {
                'total': total.item(),
                'l1':    l1.item(),
                'sam':   sam.item(),
                'edge':  edge.item(),
                'ssim':  ssim.item(),
                'spec':  spec.item(),
            }
        return total


# Legacy aliases
L1Loss   = L1ReconstructionLoss
SAMLoss  = SpectralAngleMapper
EdgeLoss = EdgePreservationLoss

__all__ = [
    'CompositeLossV6',
    'L1ReconstructionLoss', 'L1Loss',
    'SpectralAngleMapper',  'SAMLoss',
    'EdgePreservationLoss', 'EdgeLoss',
    'SSIMLoss',
    'SpectralGradientLoss',
    'compute_psnr', 'compute_sam_metric', 'compute_ergas', 'compute_ssim',
]
