"""
vmamba_pansharp_v4.py  —  Version 4 (V4)
M.Tech Thesis: Hyperspectral Image Pansharpening

Four NEW fixes over V3 (addressing spectral distortion issues)
═══════════════════════════════════════════════════════════════════════════════

Fix 1  ─  Scaled SpatialDetailInjection
    V3 SpatialDetailInjection:
        F_out = F_hsi + Gate × Edges
    Problem: if Gate is too large, PAN spatial details corrupt spectral signatures.

    V4 Fix: Add learnable scalar α (init=0.1, small by default):
        F_out = F_hsi + α × Gate × Edges
    α starts small → model must learn how much PAN detail to inject per pixel.

Fix 2  ─  Spectral Residual Reconstruction Head (Strong Bicubic)
    V3 ReconstructionHead:  HR = Conv_residual + Bicubic(LR-HSI)
    Problem: the conv residual can dominate and overwrite spectral content.

    V4 Fix: 3×Conv3×3 head with strongly-weighted Bicubic as primary:
        Residual = Conv3→Conv3→Conv3→Conv1×1   (3 layers for better capture)
        HR = Bicubic(LR-HSI) + Residual        (bicubic = spectral anchor)
    Plus learnable global scale β (init=0.5) on the residual.

Fix 3  ─  Spectral Gradient Loss (in loss_functions_v4.py)
    L_spec = || ∇_c(H_pred) - ∇_c(H_gt) ||_1
    where ∇_c = difference between adjacent spectral bands (pred[:,1:] - pred[:,:-1])
    This explicitly preserves the *shape* of spectral curves.

Fix 4  ─  Corrected Loss Weights
    Old (V1–V3):  L = 1.0*L1 + 0.05*SAM + 0.05*Edge + 0.05*SSIM
    New (V4):     L = 1.0*L1 + 0.10*SAM + 0.05*Edge + 0.01*SSIM + 0.05*L_spec
    SAM weight ↑ 0.05→0.10 forces network to prioritise spectral fidelity.
    SSIM weight ↓ 0.05→0.01 reduces spatial-only focus.

Reused from V3 (unchanged):
    ImprovedHSIEncoder       — V2 encoder
    ImprovedPANEncoder       — V2 encoder
    SpectralSSM1D            — V3 spectral scan
    SpectralSpatialMambaBlock — V3 combined block
    MultiScaleSpectralBackbone — V3 4-stage U-Net
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, 'version3'))

from vmamba_pansharp_improved import (
    ImprovedHSIEncoder,
    ImprovedPANEncoder,
)
from vmamba_pansharp_v3 import (
    SpectralSSM1D,
    SpectralSpatialMambaBlock,
    MultiScaleSpectralBackbone,
)


# ============================================================================
# V4 Fix 1 — ScaledSpatialDetailInjection
# ============================================================================

class ScaledSpatialDetailInjection(nn.Module):
    """
    Spatial Detail Injection with learnable injection scale α.

    V3:  F_out = F_hsi + Gate × Edges
    V4:  F_out = F_hsi + α × Gate × Edges      α ∈ R (learnable scalar)

    α is initialised to 0.1 so PAN injection starts very conservative.
    The network learns how much spatial detail to inject per training step.

    This prevents over-injection of PAN edges into spectrally sensitive regions,
    keeping SAM and ERGAS low while still improving PSNR.

    Difference summary
    ──────────────────
    V2 (LearnableHFInjection) : Gate = sigmoid(Conv1×1(F_pan))          ← PAN only
    V3 (SpatialDetailInjection): Gate = sigmoid(Conv1×1([F_hsi, F_pan])) ← dual-stream
    V4 (this)                  : same dual gate + global α scalar        ← dual + scale
    """

    def __init__(self, d_model: int, alpha_init: float = 0.1):
        super().__init__()
        gn = lambda c: min(8, c)

        # PAN edge extractor
        self.edge_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, bias=False),
            nn.GroupNorm(gn(d_model), d_model),
            nn.ReLU(inplace=True),
        )

        # Dual-stream gate (HSI + PAN context)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, 1, bias=False),
            nn.GroupNorm(gn(d_model), d_model),
            nn.Sigmoid(),
        )

        # Learnable global injection scale
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        self.out_norm = nn.GroupNorm(gn(d_model), d_model)

    def forward(self, F_hsi: torch.Tensor, F_pan: torch.Tensor) -> torch.Tensor:
        edges = self.edge_conv(F_pan)
        gate  = self.gate_conv(torch.cat([F_hsi, F_pan], dim=1))
        # α clipped to [0, 1] to prevent negative injection
        alpha = self.alpha.clamp(0.0, 1.0)
        F_out = F_hsi + alpha * gate * edges
        return self.out_norm(F_out)


# ============================================================================
# V4 Fix 2 — StrongReconstructionHead (3-layer + β-scaled residual)
# ============================================================================

class StrongReconstructionHead(nn.Module):
    """
    Reconstruction head with 3× Conv3×3 layers and learnable residual scale β.

    Old (V1-V3):
        residual = Conv3→Conv3→Conv1×1(features)
        HR = residual + Bicubic(LR-HSI)      ← both terms equal weight

    V4:
        residual = Conv3→Conv3→Conv3→Conv1×1(features)   ← one more conv layer
        HR = Bicubic(LR-HSI) + β × residual               ← bicubic is spectral anchor
        β starts at 0.5 so residual contributes 50% initially, decreases if SAM loss rises.

    The Bicubic term preserves all 128 spectral bands perfectly.
    The residual adds only the spatial detail that the network has learned is safe.
    """

    def __init__(self, d_model: int = 32, out_channels: int = 128,
                 scale: int = 4, beta_init: float = 0.5):
        super().__init__()
        self.scale = scale
        gn = lambda c: min(8, c)

        # 3-layer feature refiner
        self.conv1 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, bias=False),
            nn.GroupNorm(gn(d_model), d_model),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, bias=False),
            nn.GroupNorm(gn(d_model), d_model),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, bias=False),
            nn.GroupNorm(gn(d_model), d_model),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Conv2d(d_model, out_channels, 1)

        # Learnable residual scale (β): controls how strongly the residual is added
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x: torch.Tensor, lr_hsi: torch.Tensor) -> torch.Tensor:
        # x: (B, d_model, rH, rW)
        # lr_hsi: (B, C, H, W)

        # 3-layer residual prediction
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        residual = self.conv_out(feat)             # (B, C, rH, rW)

        # Bicubic upsample: spectral anchor (preserves spectral content)
        bicubic = F.interpolate(
            lr_hsi, scale_factor=self.scale,
            mode='bicubic', align_corners=False,
        )                                          # (B, C, rH, rW)

        # Bicubic primary + scaled residual
        beta = self.beta.clamp(0.0, 1.0)
        return bicubic + beta * residual


# ============================================================================
# V4 Main Model
# ============================================================================

class V4VMambaPansharp(nn.Module):
    """
    Version 4 — VMamba Hyperspectral Pansharpening (Spectral-Consistency Fix)

    Architecture
    ─────────────────────────────────────────────────────────────────────────
    LR-HSI (B, C, H, W)
        │
        ▼  ImprovedHSIEncoder  (V2, unchanged)
        →  F_hsi  (B, d_model, rH, rW)

    HR-PAN (B, 1, rH, rW)
        │
        ▼  ImprovedPANEncoder  (V2, unchanged)
        →  F_pan  (B, d_model, rH, rW)

    F_hsi + F_pan
        │
        ▼  ScaledSpatialDetailInjection  ← V4 Fix 1
           Edges = Conv3×3(F_pan)
           Gate  = sigmoid(Conv1×1([F_hsi ‖ F_pan]))
           F_fused = F_hsi + α × Gate × Edges       (α learnable, init=0.1)
        →  F_fused (B, d_model, rH, rW)

    F_fused
        │
        ▼  MultiScaleSpectralBackbone  (V3, unchanged)
           4-stage U-Net, SpectralSpatialMambaBlock at each stage
        →  F_out  (B, d_model, rH, rW)

    F_out + LR-HSI
        │
        ▼  StrongReconstructionHead  ← V4 Fix 2
           Residual = Conv3→Conv3→Conv3→Conv1×1(F_out)
           HR-HSI = Bicubic(LR-HSI) + β × Residual  (β learnable, init=0.5)
        →  HR-HSI (B, C, rH, rW)

    Loss (V4 Fix 3 + 4):
        L = 1.0*L1 + 0.10*SAM + 0.05*Edge + 0.01*SSIM + 0.05*L_spec
    ─────────────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        in_channels:  int   = 128,
        out_channels: int   = 128,
        d_model:      int   = 32,
        d_state:      int   = 8,
        scale:        int   = 4,
        num_blocks:   list  = None,
        alpha_init:   float = 0.1,
        beta_init:    float = 0.5,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 1, 1, 1]
        self.scale = scale

        # ── Encoders (V2, unchanged) ─────────────────────────────────────────
        self.hsi_encoder = ImprovedHSIEncoder(in_channels, d_model, scale)
        self.pan_encoder = ImprovedPANEncoder(d_model)

        # ── Scaled SDIM (V4 Fix 1) ───────────────────────────────────────────
        self.sdim = ScaledSpatialDetailInjection(d_model, alpha_init=alpha_init)

        # ── Multi-Scale Spectral Backbone (V3, unchanged) ────────────────────
        self.backbone = MultiScaleSpectralBackbone(
            d_model    = d_model,
            num_blocks = num_blocks,
            d_state    = d_state,
        )

        # ── Strong Reconstruction Head (V4 Fix 2) ────────────────────────────
        self.recon = StrongReconstructionHead(
            d_model      = d_model,
            out_channels = out_channels,
            scale        = scale,
            beta_init    = beta_init,
        )

    def forward(
        self,
        lr_hsi: torch.Tensor,   # (B, C, H, W)
        hr_pan: torch.Tensor,   # (B, 1, rH, rW)
    ) -> torch.Tensor:

        F_hsi   = self.hsi_encoder(lr_hsi)    # (B, d_model, rH, rW)
        F_pan   = self.pan_encoder(hr_pan)    # (B, d_model, rH, rW)
        F_fused = self.sdim(F_hsi, F_pan)     # (B, d_model, rH, rW)
        F_out   = self.backbone(F_fused)      # (B, d_model, rH, rW)
        hr_hsi  = self.recon(F_out, lr_hsi)  # (B, C, rH, rW)

        return hr_hsi


# ============================================================================
# Quick self-test
# ============================================================================
if __name__ == '__main__':
    device = torch.device('cpu')
    print("Testing V4VMambaPansharp …")

    model = V4VMambaPansharp(
        in_channels  = 128,
        out_channels = 128,
        d_model      = 32,
        d_state      = 4,
        scale        = 4,
        num_blocks   = [1, 1, 1, 1],
        alpha_init   = 0.1,
        beta_init    = 0.5,
    ).to(device).eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total:,}  ({total/1e6:.3f} M)")

    lr  = torch.randn(1, 128, 8, 8)
    pan = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        out = model(lr, pan)

    print(f"  Input  LR-HSI : {tuple(lr.shape)}")
    print(f"  Input  HR-PAN : {tuple(pan.shape)}")
    print(f"  Output HR-HSI : {tuple(out.shape)}")
    assert out.shape == (1, 128, 32, 32), f"Wrong shape: {out.shape}"
    print(f"  α (injection scale) : {model.sdim.alpha.item():.4f}")
    print(f"  β (residual scale)  : {model.recon.beta.item():.4f}")
    print("  [OK]")
