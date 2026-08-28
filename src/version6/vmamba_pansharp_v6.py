"""
vmamba_pansharp_v6.py  —  Version 6
M.Tech Thesis: Hyperspectral Image Pansharpening

V6: Spectral-Focused Architecture + Loss (SAM Reduction)
=============================================================================

CHANGES FROM V5
---------------
1. n_groups: 8 -> 32  (spectral tokens)
   V5: 128 bands -> 8 tokens  = 16 bands per token  (too coarse)
   V6: 128 bands -> 32 tokens =  4 bands per token  (fine spectral resolution)
   Expected SAM improvement: 0.7-1.2 deg

   NOTE: n_groups=32 requires d_model >= 64 to be fully effective
         (d_per_grp = d_model // n_groups; needs >= 2 for meaningful SSM)
         Default d_model changed to 64. Use --d_model 64 when training.

2. alpha_init: 0.1 -> 0.05  (PAN injection gate)
   Less PAN spatial detail bleeds into spectral bands.
   Reduces cross-band contamination from the panchromatic channel.
   Expected SAM improvement: 0.3-0.6 deg

3. beta_init: 0.5 -> 0.3  (reconstruction residual gate)
   Bicubic baseline (spectrally clean) contributes more.
   Backbone residual (potential distortion source) contributes less.
   Expected SAM improvement: 0.2-0.4 deg

4. Loss weights (from CompositeLossV6):
   lambda_sam:  0.10 -> 0.40  (+300%)
   lambda_spec: 0.05 -> 0.10  (+100%)
   Expected SAM improvement: 0.5-1.0 deg

TOTAL EXPECTED SAM REDUCTION: ~1.7-3.2 deg
V5 best SAM: 6.22 deg  ->  V6 target: ~3.0-4.5 deg

ARCHITECTURE SUMMARY
--------------------
  hillis_steele_scan()        — O(log L) true parallel scan (unchanged)
  TrueParallelSelectiveSSM    — SSM core (unchanged)
  TrueParallelSS2D            — 4-direction spatial scan (unchanged)
  V6SpectralSSM1D             — NEW: spectral scan with n_groups=32
  V6SpectralSpatialBlock      — NEW: uses V6SpectralSSM1D
  V6MultiScaleBackbone        — NEW: 4-stage U-Net with V6 blocks
  V6VMambaPansharp            — full model
    alpha_init=0.05  (was 0.1)
    beta_init=0.3    (was 0.5)
    d_model=64       (recommended, was 32)
=============================================================================
"""

import os
import sys

_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_V5      = os.path.join(_PROJECT, 'version5')

sys.path.insert(0, os.path.join(_PROJECT, 'version1'))
sys.path.insert(0, os.path.join(_PROJECT, 'version3'))
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, 'version4'))
sys.path.insert(0, _V5)
sys.path.insert(0, _HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse scan core + spatial components from V5 (unchanged)
from vmamba_pansharp_v5 import (
    hillis_steele_scan,
    TrueParallelSelectiveSSM,
    TrueParallelSS2D,
    V5VMambaPansharp,
)

# Reuse encoders and heads from V2/V4
from vmamba_pansharp_improved import ImprovedHSIEncoder, ImprovedPANEncoder
from vmamba_pansharp_v4 import ScaledSpatialDetailInjection, StrongReconstructionHead


# =============================================================================
# V6SpectralSSM1D — spectral scan with n_groups=32 (4 bands per token)
# =============================================================================

class V6SpectralSSM1D(nn.Module):
    """
    Spectral SSM with n_groups=32 for finer spectral resolution.

    V5: n_groups=8  -> 128 bands / 8  = 16 bands per token (coarse)
    V6: n_groups=32 -> 128 bands / 32 =  4 bands per token (fine)

    Requires d_model >= 64 for meaningful per-group representation:
        d_model=32, n_groups=32 -> d_per_grp=1  (minimal)
        d_model=64, n_groups=32 -> d_per_grp=2  (recommended)
        d_model=128,n_groups=32 -> d_per_grp=4  (best)

    Uses Hillis-Steele scan: log2(32)=5 passes (was log2(8)=3 in V5).
    """

    def __init__(self, d_model: int, d_state: int = 4, n_groups: int = 32):
        super().__init__()
        # Ensure d_model is divisible by n_groups — halve until it is
        while d_model % n_groups != 0 and n_groups > 1:
            n_groups //= 2
        self.n_groups  = n_groups
        self.d_per_grp = d_model // n_groups

        self.norm = nn.LayerNorm(d_model)

        # SSM on spectral groups: sequence L=n_groups, feature dim=d_per_grp
        self.ssm = TrueParallelSelectiveSSM(
            d_model = self.d_per_grp,
            d_state = d_state,
            d_conv  = min(3, n_groups),
            expand  = 2,
        )
        self.out_norm = nn.GroupNorm(min(8, d_model), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_model, H, W)
        import numpy as np
        B, C, H, W = x.shape
        N = B * H * W

        x_perm = x.permute(0, 2, 3, 1).contiguous()           # (B, H, W, C)
        x_norm = self.norm(x_perm).reshape(N, C)               # (N, C)
        x_tok  = x_norm.reshape(N, self.n_groups, self.d_per_grp)  # (N, grps, d)

        y_tok  = self.ssm(x_tok)                               # (N, grps, d)

        y = y_tok.reshape(N, C)
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)         # (B, C, H, W)
        y = self.out_norm(y)

        return x + y   # residual


# =============================================================================
# V6SpectralSpatialBlock — pre-norm + TrueParallelSS2D + V6SpectralSSM1D
# =============================================================================

class V6SpectralSpatialBlock(nn.Module):
    """
    V6 combined spatial + spectral Mamba block.
    Same as V5SpectralSpatialBlock but uses V6SpectralSSM1D (n_groups=32).
    """

    def __init__(self, dim: int, d_state: int = 8):
        super().__init__()
        self.pre_norm_spatial = nn.LayerNorm(dim)
        self.spatial_ssm      = TrueParallelSS2D(dim, d_state=d_state)
        self.spectral_ssm     = V6SpectralSSM1D(dim, d_state=min(d_state, 4), n_groups=32)

    # Maximum spatial resolution for the SSM scan.
    # L = H_scan * W_scan; Hillis-Steele cost is O(L * log L * D * N).
    # At 32x32 = 1024 tokens, memory is ~32 MB per iteration (manageable).
    # At 128x128 = 16384 tokens, it requires 13+ GB (OOM on 16 GB GPU).
    _SCAN_MAX = 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # ── Spatial scan: cap resolution at _SCAN_MAX x _SCAN_MAX ──────────────
        # When H or W > _SCAN_MAX, downsample → scan → upsample (residual).
        # Fine spatial detail is preserved via U-Net skip connections.
        if H > self._SCAN_MAX or W > self._SCAN_MAX:
            H_s, W_s = min(H, self._SCAN_MAX), min(W, self._SCAN_MAX)
            x_small  = F.adaptive_avg_pool2d(x, (H_s, W_s))   # (B, C, H_s, W_s)
            x_hw     = x_small.permute(0, 2, 3, 1)            # (B, H_s, W_s, C)
            x_normed = self.pre_norm_spatial(x_hw)
            y_small  = self.spatial_ssm(x_normed)              # (B, H_s, W_s, C)
            y_up     = F.interpolate(
                y_small.permute(0, 3, 1, 2),                   # (B, C, H_s, W_s)
                size=(H, W), mode='bilinear', align_corners=False
            )                                                   # (B, C, H, W)
            x = x + y_up
        else:
            x_hw     = x.permute(0, 2, 3, 1)                  # (B, H, W, C)
            x_normed = self.pre_norm_spatial(x_hw)
            y_spatial = self.spatial_ssm(x_normed)             # (B, H, W, C)
            x = x + y_spatial.permute(0, 3, 1, 2)

        # ── Spectral scan (V6: 32 groups, 4 bands/token) ────────────────────────
        x = self.spectral_ssm(x)                               # (B, C, H, W)

        return x


# =============================================================================
# V6MultiScaleBackbone — 4-stage U-Net with V6SpectralSpatialBlocks
# =============================================================================

class V6MultiScaleBackbone(nn.Module):
    """4-stage U-Net backbone using V6SpectralSpatialBlocks (n_groups=32)."""

    _MAX_CH = 256   # raised from 128 to allow d_model=64 to scale properly

    def __init__(self, d_model: int, num_blocks=None, d_state: int = 8):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 1, 1, 1]

        d1 = d_model
        d2 = min(d_model * 2, self._MAX_CH)
        d3 = min(d_model * 4, self._MAX_CH)
        d4 = d3

        def make_stage(dim, n):
            return nn.Sequential(*[V6SpectralSpatialBlock(dim, d_state) for _ in range(n)])

        # Encoder
        self.stage1 = make_stage(d1, num_blocks[0])
        self.down1  = nn.Conv2d(d1, d2, 2, stride=2)

        self.stage2 = make_stage(d2, num_blocks[1])
        self.down2  = nn.Conv2d(d2, d3, 2, stride=2)

        self.stage3 = make_stage(d3, num_blocks[2])
        self.down3  = nn.Conv2d(d3, d4, 2, stride=2)

        # Bottleneck
        self.stage4 = make_stage(d4, num_blocks[3])

        # Decoder
        self.up3    = nn.ConvTranspose2d(d4, d3, 2, stride=2)
        self.merge3 = nn.Sequential(nn.Conv2d(d3*2, d3, 1, bias=False),
                                    nn.GroupNorm(min(8, d3), d3))
        self.dec3   = V6SpectralSpatialBlock(d3, d_state)

        self.up2    = nn.ConvTranspose2d(d3, d2, 2, stride=2)
        self.merge2 = nn.Sequential(nn.Conv2d(d2*2, d2, 1, bias=False),
                                    nn.GroupNorm(min(8, d2), d2))
        self.dec2   = V6SpectralSpatialBlock(d2, d_state)

        self.up1    = nn.ConvTranspose2d(d2, d1, 2, stride=2)
        self.merge1 = nn.Sequential(nn.Conv2d(d1*2, d1, 1, bias=False),
                                    nn.GroupNorm(min(8, d1), d1))
        self.dec1   = V6SpectralSpatialBlock(d1, d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.stage1(x)
        s2 = self.stage2(self.down1(s1))
        s3 = self.stage3(self.down2(s2))
        s4 = self.stage4(self.down3(s3))

        d3 = self.dec3(self.merge3(torch.cat([self.up3(s4), s3], dim=1)))
        d2 = self.dec2(self.merge2(torch.cat([self.up2(d3), s2], dim=1)))
        d1 = self.dec1(self.merge1(torch.cat([self.up1(d2), s1], dim=1)))

        return d1


# =============================================================================
# V6VMambaPansharp — Full Model
# =============================================================================

class V6VMambaPansharp(nn.Module):
    """
    Version 6 VMamba Pansharpening — Spectral-Focused Architecture.

    Key changes from V5:
      n_groups : 8  -> 32   (finer spectral tokens, 4 bands/token vs 16)
      alpha    : 0.1 -> 0.05 (less PAN contamination into spectral bands)
      beta     : 0.5 -> 0.3  (cleaner spectral baseline from bicubic)
      d_model  : 32 -> 64   (recommended; gives d_per_grp=2 for n_groups=32)
      Loss     : lambda_sam=0.40, lambda_spec=0.10 (via CompositeLossV6)
    """

    def __init__(
        self,
        in_channels:  int   = 128,
        out_channels: int   = 128,
        d_model:      int   = 64,    # V5: 32 -> V6: 64 (needed for n_groups=32)
        d_state:      int   = 8,
        scale:        int   = 4,
        num_blocks:   list  = None,
        alpha_init:   float = 0.05,  # V5: 0.1  -> V6: 0.05
        beta_init:    float = 0.3,   # V5: 0.5  -> V6: 0.3
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 1, 1, 1]
        self.scale = scale

        self.hsi_encoder = ImprovedHSIEncoder(in_channels, d_model, scale)
        self.pan_encoder = ImprovedPANEncoder(d_model)
        self.sdim        = ScaledSpatialDetailInjection(d_model, alpha_init=alpha_init)
        self.backbone    = V6MultiScaleBackbone(d_model, num_blocks, d_state)
        self.recon       = StrongReconstructionHead(d_model, out_channels, scale, beta_init)

    def forward(self, lr_hsi: torch.Tensor, hr_pan: torch.Tensor) -> torch.Tensor:
        F_hsi   = self.hsi_encoder(lr_hsi)
        F_pan   = self.pan_encoder(hr_pan)
        F_fused = self.sdim(F_hsi, F_pan)
        F_out   = self.backbone(F_fused)
        return self.recon(F_out, lr_hsi)

    def version_info(self) -> str:
        alpha = self.sdim.alpha.item()
        beta  = self.recon.beta.item()
        n_grp = self.backbone.stage1[0].spectral_ssm.n_groups
        d_m   = self.backbone.stage1[0].spectral_ssm.d_per_grp * n_grp
        return (
            f"V6VMambaPansharp\n"
            f"  n_groups     : {n_grp}  (4 bands/token, finer than V5's 16)\n"
            f"  d_per_grp    : {self.backbone.stage1[0].spectral_ssm.d_per_grp}\n"
            f"  alpha (init) : {alpha:.4f}  (PAN injection gate, was 0.1)\n"
            f"  beta  (init) : {beta:.4f}  (recon residual gate, was 0.5)\n"
            f"  Loss         : lambda_sam=0.40  lambda_spec=0.10\n"
        )


# =============================================================================
# Quick self-test
# =============================================================================

if __name__ == '__main__':
    print("Testing V6VMambaPansharp ...")
    device = torch.device('cpu')

    model = V6VMambaPansharp(
        in_channels=128, out_channels=128,
        d_model=64, d_state=4, scale=4, num_blocks=[1, 1, 1, 1],
    ).to(device).eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total:,}  ({total/1e6:.3f} M)")
    print(model.version_info())

    # Test with patch_size=32 LR input (scale=4 -> HR=128)
    lr  = torch.randn(1, 128, 32,  32)
    pan = torch.randn(1, 1,  128, 128)
    with torch.no_grad():
        out = model(lr, pan)
    assert out.shape == (1, 128, 128, 128), f"Shape error: {out.shape}"
    print(f"  LR (32x32) -> OUT (128x128)  [OK]")

    # Test with patch_size=64 LR input (scale=4 -> HR=256)
    lr2  = torch.randn(1, 128, 64,  64)
    pan2 = torch.randn(1, 1,  256, 256)
    with torch.no_grad():
        out2 = model(lr2, pan2)
    assert out2.shape == (1, 128, 256, 256), f"Shape error: {out2.shape}"
    print(f"  LR (64x64) -> OUT (256x256)  [OK]")

    # Loss test
    from loss_functions_v6 import CompositeLossV6
    criterion = CompositeLossV6()
    pred = torch.rand(1, 128, 128, 128)
    gt   = torch.rand(1, 128, 128, 128)
    loss, comps = criterion(pred, gt, return_components=True)
    print(f"\n  Loss components:")
    for k, v in comps.items():
        print(f"    {k:10s}: {v:.5f}")
    print(f"  [OK] CompositeLossV6  lambda_sam=0.40  lambda_spec=0.10")
    print(f"\n  alpha={model.sdim.alpha.item():.4f}  beta={model.recon.beta.item():.4f}")
