"""
losses.py  --  V5 Edition
Imports base losses from the shared scripts/loss_functions.py module.
SpectralGradientLoss and CompositeLossV4 are duplicated from V4 (see below)
so this version's model/ folder is fully self-contained.
CompositeLossV5 is an alias for CompositeLossV4 (same objectives, improved scan only).
"""
from __future__ import annotations
import os
import sys

_HERE    = os.path.dirname(os.path.abspath(__file__))   # version5/model/
_PROJECT = os.path.dirname(os.path.dirname(_HERE))         # src/
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, 'scripts'))

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
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Reused from Version 4 (src/version4/model/losses.py)
# Duplicated here so this version's model/ folder is fully self-contained.
# ============================================================================

# ============================================================================
# V4 NEW — Spectral Gradient Loss
# ============================================================================

class SpectralGradientLoss(nn.Module):
    """
    Spectral Gradient Loss (V4 Fix 3).

    Preserves the *shape* of spectral curves by matching band-to-band differences.

    Formula:
        ∇_c(H) = H[:, 1:, :, :] - H[:, :-1, :, :]   (adjacent-band differences)
        L_spec  = || ∇_c(H_pred) - ∇_c(H_gt) ||_1

    Why it helps:
    ─────────────
    L1 / PSNR measure absolute pixel values — they do not distinguish:
        H_pred = [0.5, 0.7, 0.9]   (correct spectral slope)
        H_bad  = [0.5, 0.9, 0.7]   (same L1 error vs gt if signs flip)

    SpectralGradientLoss directly penalises wrong slope direction:
        ∇H_pred = [0.2, 0.2]
        ∇H_bad  = [0.4, -0.2]  ← penalised by L_spec
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, C, H, W)
        # Band-adjacent differences: shape (B, C-1, H, W)
        pred_grad   = pred[:, 1:, :, :]   - pred[:, :-1, :, :]
        target_grad = target[:, 1:, :, :] - target[:, :-1, :, :]
        return F.l1_loss(pred_grad, target_grad)


# ============================================================================
# V4 Composite Loss (corrected weights)
# ============================================================================

class CompositeLossV4(nn.Module):
    """
    V4 composite loss with corrected weights for spectral fidelity.

    L = λ_l1   × L1
      + λ_sam   × SAM         (↑ 0.05 → 0.10: spectral angle is now more important)
      + λ_edge  × Edge
      + λ_ssim  × SSIM        (↓ 0.05 → 0.01: less pure spatial focus)
      + λ_spec  × L_spec      (NEW: spectral gradient preservation)

    Default weights (recommended):
        λ_l1=1.0, λ_sam=0.10, λ_edge=0.05, λ_ssim=0.01, λ_spec=0.05
    """

    def __init__(
        self,
        lambda_l1:   float = 1.0,
        lambda_sam:  float = 0.10,
        lambda_edge: float = 0.05,
        lambda_ssim: float = 0.01,
        lambda_spec: float = 0.05,
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
