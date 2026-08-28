"""
vmamba_pansharp_v3.py  —  Version 3 (V3)
M.Tech Thesis: Hyperspectral Image Pansharpening

Three NEW improvements over V2
═══════════════════════════════════════════════════════════════════════════════
Improvement 1  ─  SpectralMambaBlock
    V2 backbone only modeled SPATIAL (H, W) dependencies via SS2D.
    V3 adds a SPECTRAL SSM that scans along the feature-channel dimension C
    at each spatial position, capturing inter-band correlations.

    Each backbone block now runs:
        x → SpatialSS2D (models H × W)
          → SpectralSSM1D (models C at every pixel)
          → output

    Combined modeling: H, W, C  ← three dimensions, not just two.

Improvement 2  ─  MultiScaleSpectralBackbone  (4-stage U-Net)
    V2 used a 3-stage U-Net (½, ¼, ¼ bottleneck).
    V3 uses a DEEPER 4-stage U-Net: full → ½ → ¼ → ⅛ resolution.
        Stage 1  (d,     H,   W  )  full resolution
        Stage 2  (2d,    H/2, W/2)  1/2
        Stage 3  (4d_c,  H/4, W/4)  1/4
        Stage 4  (4d_c,  H/8, W/8)  1/8  ← bottleneck
        + symmetric decoder with skip connections
    Each stage uses the SpectralSpatialMambaBlock.
    Deeper multi-scale hierarchy → better context modeling.

Improvement 3  ─  SpatialDetailInjection (SDIM)
    V2 used CrossMambaFusion (replaces V1 CrossAttention).
    V3 adds an explicit DYNAMIC SPATIAL DETAIL INJECTION module:

        Edges = Conv3×3(F_pan)                          ← extract PAN detail
        Gate  = sigmoid(Conv1×1([F_hsi ‖ F_pan]))       ← BOTH streams gating
        F_out = F_hsi + Gate × Edges                    ← gated injection

    Key difference from V2 LearnableHFInjection:
        V2 gate uses PAN only:      Gate = sigmoid(Conv1×1(F_pan))
        V3 gate uses HSI + PAN:     Gate = sigmoid(Conv1×1([F_hsi, F_pan]))
    → context-aware gating avoids spectral distortion.

Reused from V2 (unchanged):
    ImprovedHSIEncoder  ─ 3D Conv + 2× ResidualUpsample2×
    ImprovedPANEncoder  ─ 3-layer Conv
    ReconstructionHead  ─ residual learning + bicubic upsampled LR-HSI
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

from vmamba_pansharp import (
    SelectiveSSMOptimized,
    MambaVisionBlockOptimized,
    ReconstructionHead,
)
from vmamba_pansharp_improved import (
    ImprovedHSIEncoder,
    ImprovedPANEncoder,
)


# ============================================================================
# V3 Improvement 1a — SpectralSSM1D
# ============================================================================

class SpectralSSM1D(nn.Module):
    """
    Selective SSM scanning along the SPECTRAL (channel) dimension.

    The d_model feature channels are grouped into n_groups spectral tokens.
    The SelectiveSSMOptimized scans across these n_groups tokens, modelling
    how information flows from one band-group to the next.

    Grouping strategy:
        d_model=32,  n_groups=8  →  d_per_group=4
        d_model=64,  n_groups=8  →  d_per_group=8
        d_model=128, n_groups=8  →  d_per_group=16

    Input:  (B, d_model, H, W)
    Output: (B, d_model, H, W)  with spectral context added
    """

    def __init__(self, d_model: int, d_state: int = 4, n_groups: int = 8):
        super().__init__()
        # Resolve n_groups so that d_model % n_groups == 0
        while d_model % n_groups != 0 and n_groups > 1:
            n_groups //= 2
        self.d_model    = d_model
        self.n_groups   = n_groups
        self.d_per_grp  = d_model // n_groups   # feature dim per spectral token

        self.norm = nn.LayerNorm(d_model)

        # Core SSM: sequence_length = n_groups, feature_dim = d_per_grp
        self.ssm = SelectiveSSMOptimized(
            d_model = self.d_per_grp,
            d_state = d_state,
            d_conv  = min(3, n_groups),   # kernel ≤ sequence length
            expand  = 2,
        )

        self.out_norm = nn.GroupNorm(min(8, d_model), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = B * H * W  # each pixel is an independent spectral sequence

        # LayerNorm over channel dimension
        x_perm = x.permute(0, 2, 3, 1).contiguous()   # (B, H, W, C)
        x_norm = self.norm(x_perm)                     # (B, H, W, C)
        x_flat = x_norm.reshape(N, C)                  # (N, C)

        # Group channels → spectral tokens
        # (N, C) → (N, n_groups, d_per_grp)
        x_tokens = x_flat.reshape(N, self.n_groups, self.d_per_grp)

        # SelectiveSSMOptimized expects (B, L, d_model)
        # Here: batch=N, L=n_groups, d_model=d_per_grp
        y_tokens = self.ssm(x_tokens)                  # (N, n_groups, d_per_grp)

        # Ungroup → (N, C)
        y = y_tokens.reshape(N, C)

        # Reshape to (B, C, H, W) and normalise
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        y = self.out_norm(y)

        # Residual connection
        return x + y


# ============================================================================
# V3 Improvement 1b — SpectralSpatialMambaBlock
# ============================================================================

class SpectralSpatialMambaBlock(nn.Module):
    """
    Joint Spatial + Spectral Mamba block (Improvement 1).

    Pipeline:
        x  →  SpatialSS2D   (scans H × W  per channel)
           →  SpectralSSM1D (scans C groups per pixel)
           →  output

    Models H, W, C jointly — not just H and W.
    """

    def __init__(self, dim: int, d_state: int = 8):
        super().__init__()
        self.spatial_block = MambaVisionBlockOptimized(dim, d_state=d_state)
        # Smaller d_state for spectral (4 groups of bands usually enough)
        self.spectral_ssm  = SpectralSSM1D(dim, d_state=min(d_state, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_block(x)   # spatial: scan (H, W)
        x = self.spectral_ssm(x)    # spectral: scan channel groups
        return x


# ============================================================================
# V3 Improvement 3 — SpatialDetailInjection (SDIM)
# ============================================================================

class SpatialDetailInjection(nn.Module):
    """
    Spatial Detail Injection Module — Improvement 3.

    Injects PAN spatial edges into HSI features with a DYNAMIC GATE that
    depends on BOTH streams, giving context-aware injection strength.

        Edges  = Conv3×3(F_pan)                         ← PAN spatial detail
        Gate   = sigmoid(Conv1×1([F_hsi ‖ F_pan]))      ← both-stream gate
        F_out  = F_hsi + Gate × Edges                   ← gated injection
        F_out  = GroupNorm(F_out)

    Difference from V2 LearnableHFInjection:
        V2 gate is computed from F_pan only → may not respect current HSI state
        V3 gate uses [F_hsi, F_pan] → adapts to spectral content of HSI too
    """

    def __init__(self, d_model: int):
        super().__init__()
        gn = lambda c: min(8, c)

        # Extract spatial detail (edges / texture) from PAN stream
        self.edge_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, bias=False),
            nn.GroupNorm(gn(d_model), d_model),
            nn.ReLU(inplace=True),
        )

        # Dynamic gate: context from BOTH HSI and PAN
        self.gate_conv = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, 1, bias=False),
            nn.GroupNorm(gn(d_model), d_model),
            nn.Sigmoid(),
        )

        self.out_norm = nn.GroupNorm(gn(d_model), d_model)

    def forward(self, F_hsi: torch.Tensor, F_pan: torch.Tensor) -> torch.Tensor:
        # F_hsi, F_pan: (B, d_model, H, W)

        # 1. Extract PAN spatial detail
        edges = self.edge_conv(F_pan)                              # (B, d_model, H, W)

        # 2. Dynamic gate conditioned on both HSI and PAN
        gate  = self.gate_conv(torch.cat([F_hsi, F_pan], dim=1))  # (B, d_model, H, W)

        # 3. Gated injection — avoids spectral distortion
        F_out = F_hsi + gate * edges

        return self.out_norm(F_out)


# ============================================================================
# V3 Improvement 2 — MultiScaleSpectralBackbone (4-stage U-Net)
# ============================================================================

class MultiScaleSpectralBackbone(nn.Module):
    """
    4-stage hierarchical U-Net backbone — Improvement 2.

    Each stage uses SpectralSpatialMambaBlock (spatial SS2D + spectral SSM).

    Channel schedule (capped at 128 for OOM safety):
        d_model=32 :  32 → 64 → 128 → 128
        d_model=64 :  64 → 128 → 128 → 128

    Spatial schedule (example: input 32×32):
        Stage 1 : 32×32   (full res)
        Stage 2 : 16×16   (½)
        Stage 3 : 8×8     (¼)
        Stage 4 : 4×4     (⅛) ← bottleneck
        Decoder : 8 → 16 → 32  (symmetric, with skip connections)

    Minimum input spatial size: 8×8  (so bottleneck ≥ 1×1)
    Recommended:               32×32 (bottleneck = 4×4)
    """

    _MAX_CH = 128   # cap channel growth for OOM safety

    def __init__(self, d_model: int, num_blocks=None, d_state: int = 8):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 1, 1, 1]

        # Channel schedule
        d1 = d_model
        d2 = min(d_model * 2, self._MAX_CH)
        d3 = min(d_model * 4, self._MAX_CH)
        d4 = d3   # bottleneck same as stage-3 (no further doubling)
        self.d1, self.d2, self.d3, self.d4 = d1, d2, d3, d4

        def make_stage(dim, n):
            return nn.Sequential(*[
                SpectralSpatialMambaBlock(dim, d_state) for _ in range(n)
            ])

        # ── Encoder ──────────────────────────────────────────────────────────
        self.stage1 = make_stage(d1, num_blocks[0])
        self.down1  = nn.Conv2d(d1, d2, kernel_size=2, stride=2)

        self.stage2 = make_stage(d2, num_blocks[1])
        self.down2  = nn.Conv2d(d2, d3, kernel_size=2, stride=2)

        self.stage3 = make_stage(d3, num_blocks[2])
        self.down3  = nn.Conv2d(d3, d4, kernel_size=2, stride=2)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.stage4 = make_stage(d4, num_blocks[3])

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up3    = nn.ConvTranspose2d(d4, d3, kernel_size=2, stride=2)
        self.merge3 = nn.Sequential(
            nn.Conv2d(d3 * 2, d3, 1, bias=False),
            nn.GroupNorm(min(8, d3), d3),
        )
        self.dec3   = SpectralSpatialMambaBlock(d3, d_state)

        self.up2    = nn.ConvTranspose2d(d3, d2, kernel_size=2, stride=2)
        self.merge2 = nn.Sequential(
            nn.Conv2d(d2 * 2, d2, 1, bias=False),
            nn.GroupNorm(min(8, d2), d2),
        )
        self.dec2   = SpectralSpatialMambaBlock(d2, d_state)

        self.up1    = nn.ConvTranspose2d(d2, d1, kernel_size=2, stride=2)
        self.merge1 = nn.Sequential(
            nn.Conv2d(d1 * 2, d1, 1, bias=False),
            nn.GroupNorm(min(8, d1), d1),
        )
        self.dec1   = SpectralSpatialMambaBlock(d1, d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d1, H, W)

        # ── Encoder ──────────────────────────────────────────────────────────
        s1 = self.stage1(x)                 # (B, d1, H,   W  )
        s2 = self.stage2(self.down1(s1))    # (B, d2, H/2, W/2)
        s3 = self.stage3(self.down2(s2))    # (B, d3, H/4, W/4)
        s4 = self.stage4(self.down3(s3))    # (B, d4, H/8, W/8) ← bottleneck

        # ── Decoder ──────────────────────────────────────────────────────────
        d3 = self.dec3(self.merge3(torch.cat([self.up3(s4), s3], dim=1)))
        d2 = self.dec2(self.merge2(torch.cat([self.up2(d3), s2], dim=1)))
        d1 = self.dec1(self.merge1(torch.cat([self.up1(d2), s1], dim=1)))

        return d1   # (B, d1, H, W)


# ============================================================================
# V3 Main Model
# ============================================================================

class V3VMambaPansharp(nn.Module):
    """
    Version 3 — VMamba Hyperspectral Pansharpening

    Architecture
    ─────────────────────────────────────────────────────────────────────────
    LR-HSI (B, C, H, W)
        │
        ▼  ImprovedHSIEncoder  (V2, unchanged)
           3D Conv + spectral aggregation + 2× ResidualUpsample2×
        →  F_hsi  (B, d_model, rH, rW)

    HR-PAN (B, 1, rH, rW)
        │
        ▼  ImprovedPANEncoder  (V2, unchanged)
           3× Conv2d + GroupNorm + ReLU
        →  F_pan  (B, d_model, rH, rW)

    F_hsi + F_pan
        │
        ▼  SpatialDetailInjection  (V3 Improvement 3)
           Edges = Conv3×3(F_pan)
           Gate  = sigmoid(Conv1×1([F_hsi ‖ F_pan]))
           F_fused = F_hsi + Gate × Edges
        →  F_fused (B, d_model, rH, rW)

    F_fused
        │
        ▼  MultiScaleSpectralBackbone  (V3 Improvements 1 + 2)
           4-stage U-Net with SpectralSpatialMambaBlock at each stage
           (spatial SS2D + spectral SSM)
        →  F_out  (B, d_model, rH, rW)

    F_out + LR-HSI
        │
        ▼  ReconstructionHead  (reused from V1/V2)
           Conv → residual + bicubic(LR-HSI)
        →  HR-HSI (B, C, rH, rW)
    ─────────────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        in_channels:  int  = 128,
        out_channels: int  = 128,
        d_model:      int  = 32,
        d_state:      int  = 8,
        scale:        int  = 4,
        num_blocks:   list = None,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 1, 1, 1]
        self.scale = scale

        # ── Encoders (V2, unchanged) ─────────────────────────────────────────
        self.hsi_encoder = ImprovedHSIEncoder(in_channels, d_model, scale)
        self.pan_encoder = ImprovedPANEncoder(d_model)

        # ── Spatial Detail Injection (V3 Improvement 3) ─────────────────────
        self.sdim = SpatialDetailInjection(d_model)

        # ── Multi-Scale Spectral Backbone (V3 Improvements 1 + 2) ───────────
        self.backbone = MultiScaleSpectralBackbone(
            d_model    = d_model,
            num_blocks = num_blocks,
            d_state    = d_state,
        )

        # ── Reconstruction Head (reused from V1/V2) ──────────────────────────
        self.recon = ReconstructionHead(d_model, out_channels, scale)

    def forward(
        self,
        lr_hsi: torch.Tensor,   # (B, C, H, W)
        hr_pan: torch.Tensor,   # (B, 1, rH, rW)
    ) -> torch.Tensor:

        # Encode both modalities
        F_hsi = self.hsi_encoder(lr_hsi)   # (B, d_model, rH, rW)
        F_pan = self.pan_encoder(hr_pan)   # (B, d_model, rH, rW)

        # Dynamic spatial detail injection (V3)
        F_fused = self.sdim(F_hsi, F_pan)  # (B, d_model, rH, rW)

        # Multi-scale spectral backbone (V3)
        F_out = self.backbone(F_fused)     # (B, d_model, rH, rW)

        # Residual reconstruction
        hr_hsi = self.recon(F_out, lr_hsi) # (B, C, rH, rW)

        return hr_hsi


# ============================================================================
# Quick test
# ============================================================================

if __name__ == '__main__':
    import sys
    device = torch.device('cpu')
    print("Testing V3VMambaPansharp …")

    model = V3VMambaPansharp(
        in_channels  = 128,
        out_channels = 128,
        d_model      = 32,
        d_state      = 4,
        scale        = 4,
        num_blocks   = [1, 1, 1, 1],
    ).to(device).eval()

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total:,}  ({total/1e6:.3f} M)")

    # Forward pass
    lr  = torch.randn(1, 128, 8, 8)
    pan = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        out = model(lr, pan)
    print(f"  Input  LR-HSI : {tuple(lr.shape)}")
    print(f"  Input  HR-PAN : {tuple(pan.shape)}")
    print(f"  Output HR-HSI : {tuple(out.shape)}")
    assert out.shape == (1, 128, 32, 32), f"Wrong shape: {out.shape}"
    print("  [OK]")
