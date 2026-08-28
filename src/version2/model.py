"""
model.py  —  Version 2  (self-contained)
Improved VMamba Pansharpening Model with True Hillis-Steele Parallel Scan.

Architecture overview:
  LR-HSI → ImprovedHSIEncoder → F_hsi  (residual 2× upsampling, no PixelShuffle)
  HR-PAN → ImprovedPANEncoder → F_pan  (3-layer Conv+GN+ReLU, Kaiming init)
           CrossMambaFusion(F_hsi, F_pan)  ← O(L) cross-modal, replaces O(N²) attn
           VMambaBackbone  (4-stage U-Net with Mamba blocks)
           ReconstructionHead  (residual + bicubic anchor)

Key difference from V1:
  ALL SSM scans use Hillis-Steele true parallel scan (O(log L) depth).
  CrossMambaSSM1D also uses H-S scan (both row and col directions).
  MambaBlock uses pre-norm (y = SSM(Norm(x)) + x) for stability.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ── 1. Hillis-Steele Parallel Scan ───────────────────────────────────────────

def hillis_steele_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    True Hillis-Steele inclusive prefix scan.
    Binary operator: (a2,b2) ⊕ (a1,b1) = (a2·a1, a2·b1 + b2)
    Computes h_t = a[t]*a[t-1]*…*a[0]*h_init + … in O(log L) passes.

    Args:
        a : (B, L, D, N) — diagonal A_bar (state-transition coefficients)
        b : (B, L, D, N) — B_bar * x   (input contribution)
    Returns:
        h : (B, L, D, N) — hidden states h[0..L-1]
    """
    B, L, D, N = a.shape
    L_orig = L

    # Pad to next power of 2 if needed
    if L & (L - 1):
        L_pad = 1 << math.ceil(math.log2(max(L, 2)))
        pad   = L_pad - L
        a = torch.cat([a, a.new_ones(B, pad, D, N)],  dim=1)
        b = torch.cat([b, b.new_zeros(B, pad, D, N)], dim=1)
        L = L_pad

    log2_L = int(round(math.log2(L)))
    for d in range(log2_L):
        stride  = 1 << d
        prev_a  = torch.cat([a.new_ones(B, stride, D, N),  a[:, :L-stride]], dim=1)
        prev_b  = torch.cat([b.new_zeros(B, stride, D, N), b[:, :L-stride]], dim=1)
        new_b   = a * prev_b + b   # use OLD a
        new_a   = a * prev_a
        a, b    = new_a, new_b

    return b[:, :L_orig]


# ── 2. Selective SSM with H-S Scan ───────────────────────────────────────────

class SelectiveSSMV2(nn.Module):
    """
    Full Mamba-style Selective SSM using Hillis-Steele parallel scan.
    Works on 1-D token sequences of shape (B, L, d_model).
    """

    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # Input projection (fused x and z branches)
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Short-range depthwise conv
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                  padding=d_conv-1, groups=self.d_inner, bias=True)

        # SSM parameter projections
        self.x_proj   = nn.Linear(self.d_inner, d_state*2 + self.d_inner, bias=False)
        self.dt_proj  = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Learnable A (log-parameterised, negative → stable)
        A = repeat(torch.arange(1, d_state+1, dtype=torch.float32),
                   'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection (zero-init → identity residual at start)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        B, L, _ = x.shape

        # Fused in-proj + split
        xz       = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)           # each (B, L, d_inner)

        # Depthwise conv
        x_inner  = rearrange(x_inner, 'b l d -> b d l')
        x_inner  = self.conv1d(x_inner)[:, :, :L]
        x_inner  = rearrange(x_inner, 'b d l -> b l d')
        x_inner  = self.act(x_inner)

        # SSM parameters
        x_proj   = self.x_proj(x_inner)
        B_param, C_param, dt_raw = torch.split(
            x_proj, [self.d_state, self.d_state, self.d_inner], dim=-1)
        delta    = F.softplus(self.dt_proj(dt_raw))    # (B, L, d_inner) ≥ 0
        A        = -torch.exp(self.A_log.float())       # (d_inner, d_state) negative

        # ZOH discretisation
        a_bar    = torch.exp(torch.einsum('bld,dn->bldn', delta, A))  # (B,L,D,N)
        b_bar_x  = torch.einsum('bld,bln,bld->bldn', delta, B_param, x_inner)

        # True parallel scan
        h_all    = hillis_steele_scan(a_bar, b_bar_x)  # (B,L,D,N)

        # Output
        y        = torch.einsum('bldn,bln->bld', h_all, C_param)  # (B,L,D)
        y        = y + self.D * x_inner                            # skip
        y        = y * self.act(z)                                 # gate

        return self.out_proj(y)                                    # (B,L,d_model)


# ── 3. 2D Selective Scan (4 directions) ──────────────────────────────────────

class SS2DV2(nn.Module):
    """
    2D Selective Scan: row-fwd, row-bwd, col-fwd, col-bwd.
    Each direction uses SelectiveSSMV2 (Hillis-Steele scan).
    Training: sequential per-direction (memory-safe).
    Inference: batched all-4 (fast).
    """

    def __init__(self, d_model: int, d_state: int = 8):
        super().__init__()
        self.ssm  = SelectiveSSMV2(d_model, d_state)
        self.proj = nn.Linear(d_model * 4, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C) → (B, H, W, C)"""
        B, H, W, C = x.shape

        if self.training:
            x1 = rearrange(x, 'b h w c -> (b h) w c')
            y1 = self.ssm(x1)
            y1 = rearrange(y1, '(b h) w c -> b h w c', b=B, h=H)

            x2 = torch.flip(x1, dims=[1])
            y2 = self.ssm(x2)
            y2 = torch.flip(y2, dims=[1])
            y2 = rearrange(y2, '(b h) w c -> b h w c', b=B, h=H)

            x3 = rearrange(x, 'b h w c -> (b w) h c')
            y3 = self.ssm(x3)
            y3 = rearrange(y3, '(b w) h c -> b h w c', b=B, w=W)

            x4 = torch.flip(x3, dims=[1])
            y4 = self.ssm(x4)
            y4 = torch.flip(y4, dims=[1])
            y4 = rearrange(y4, '(b w) h c -> b h w c', b=B, w=W)
        else:
            x1 = rearrange(x, 'b h w c -> (b h) w c')
            x2 = torch.flip(x1, dims=[1])
            x3 = rearrange(x, 'b h w c -> (b w) h c')
            x4 = torch.flip(x3, dims=[1])

            x_all = torch.cat([x1, x2, x3, x4], dim=0)
            y_all = self.ssm(x_all)

            y1, y2, y3, y4 = torch.split(y_all, [B*H, B*H, B*W, B*W], dim=0)
            y2 = torch.flip(y2, dims=[1])
            y4 = torch.flip(y4, dims=[1])

            y1 = rearrange(y1, '(b h) w c -> b h w c', b=B, h=H)
            y2 = rearrange(y2, '(b h) w c -> b h w c', b=B, h=H)
            y3 = rearrange(y3, '(b w) h c -> b h w c', b=B, w=W)
            y4 = rearrange(y4, '(b w) h c -> b h w c', b=B, w=W)

        y = torch.cat([y1, y2, y3, y4], dim=-1)  # (B,H,W,4C)
        return self.proj(y)                        # (B,H,W,C)


# ── 4. Mamba Vision Block (pre-norm) ─────────────────────────────────────────

class MambaBlockV2(nn.Module):
    """
    Pre-norm Mamba block: y = SS2D(LayerNorm(x)) + x
    More stable than post-norm used in V1.
    """

    def __init__(self, dim: int, d_state: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2DV2(dim, d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W)"""
        B, C, H, W = x.shape
        h = rearrange(x, 'b c h w -> b h w c')
        h = self.ss2d(self.norm(h))
        return rearrange(h, 'b h w c -> b c h w') + x


# ── 5. VMamba Backbone (4-stage U-Net) ────────────────────────────────────────

class VMambaBackboneV2(nn.Module):
    """4-stage hierarchical U-Net using MambaBlockV2 (parallel scan)."""

    def __init__(self, d_model: int = 32, d_state: int = 8,
                 num_blocks: list[int] | None = None):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 1, 1, 1]
        D = d_model

        def _stage(dim, n):
            return nn.Sequential(*[MambaBlockV2(dim, d_state) for _ in range(n)])

        self.stage1  = _stage(D,   num_blocks[0])
        self.down1   = nn.Conv2d(D,   D*2, 2, stride=2)
        self.stage2  = _stage(D*2, num_blocks[1])
        self.down2   = nn.Conv2d(D*2, D*4, 2, stride=2)
        self.stage3  = _stage(D*4, num_blocks[2])          # bottleneck
        self.up1     = nn.ConvTranspose2d(D*4, D*2, 2, stride=2)
        self.skip2   = nn.Conv2d(D*4, D*2, 1)
        self.stage4a = _stage(D*2, max(1, num_blocks[3]//2))
        self.up2     = nn.ConvTranspose2d(D*2, D, 2, stride=2)
        self.stage4b = _stage(D,   max(1, num_blocks[3] - num_blocks[3]//2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stage1(x)
        x2 = self.stage2(self.down1(x1))
        x3 = self.stage3(self.down2(x2))
        x4 = self.up1(x3)
        x4 = self.stage4a(self.skip2(torch.cat([x4, x2], 1)))
        x5 = self.up2(x4) + x1
        return self.stage4b(x5)


# ── 6. Encoder components ─────────────────────────────────────────────────────

class ResidualUpsample2x(nn.Module):
    """2× residual upsampling (no PixelShuffle artefacts)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv_m = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn_m   = nn.GroupNorm(max(1, out_ch//8), out_ch)
        self.conv_r = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.relu   = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv_m.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_r.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        main = self.relu(self.gn_m(self.conv_m(x_up)))
        res  = F.interpolate(self.conv_r(x), scale_factor=2, mode='bilinear', align_corners=False)
        return main + res


class ImprovedHSIEncoder(nn.Module):
    """3D Conv → spectral aggregation → ResidualUpsample × (1 or 2)."""

    def __init__(self, in_channels: int, d_model: int = 32, scale: int = 4):
        super().__init__()
        self.scale = scale
        feat = 8
        self.conv3d   = nn.Conv3d(1, feat, 3, padding=1, bias=False)
        self.gn3d     = nn.GroupNorm(max(1, feat//2), feat)
        self.relu     = nn.ReLU(inplace=True)
        self.spec_agg = nn.Conv2d(feat * in_channels, d_model, 1, bias=False)
        self.up1      = ResidualUpsample2x(d_model, d_model)
        self.up2      = ResidualUpsample2x(d_model, d_model) if scale == 4 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x3d = self.relu(self.gn3d(self.conv3d(x.unsqueeze(1))))  # (B,8,C,H,W)
        x2d = self.relu(self.spec_agg(x3d.reshape(B, -1, H, W))) # (B,d,H,W)
        x2d = self.up1(x2d)
        if self.up2 is not None:
            x2d = self.up2(x2d)
        return x2d


class ImprovedPANEncoder(nn.Module):
    """3-layer Conv+GN+ReLU with Kaiming init."""

    def __init__(self, d_model: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,  32,      3, padding=1, bias=False), nn.GroupNorm(8, 32),     nn.ReLU(True),
            nn.Conv2d(32, 64,      3, padding=1, bias=False), nn.GroupNorm(8, 64),     nn.ReLU(True),
            nn.Conv2d(64, d_model, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, d_model//8), d_model), nn.ReLU(True),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── 7. Cross-Mamba Fusion ─────────────────────────────────────────────────────

class LearnableHFInjection(nn.Module):
    """Learnable high-frequency injection: hsi + sigmoid(Conv1×1(pan)) × Conv3×3(pan)."""

    def __init__(self, d_model: int):
        super().__init__()
        ng = max(1, d_model//8)
        self.edge_conv = nn.Conv2d(d_model, d_model, 3, padding=1)
        self.gate_conv = nn.Conv2d(d_model, d_model, 1)
        self.norm      = nn.GroupNorm(ng, d_model)

    def forward(self, hsi, pan):
        if hsi.shape[2:] != pan.shape[2:]:
            pan = F.interpolate(pan, size=hsi.shape[2:], mode='bilinear', align_corners=False)
        return self.norm(hsi + torch.sigmoid(self.gate_conv(pan)) * self.edge_conv(pan))


class CrossMambaSSM1D(nn.Module):
    """
    PAN-guided 1-D SSM scan for cross-modal fusion.
    PAN projects B, C, Δ parameters; HSI is the input signal.
    Uses Hillis-Steele parallel scan (O(log L) depth).
    """

    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.hsi_dw  = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv-1,
                                  groups=d_model, bias=True)
        self.hsi_act = nn.SiLU()
        self.pan_proj= nn.Linear(d_model, d_state*2 + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        A = repeat(torch.arange(1, d_state+1, dtype=torch.float32),
                   'n -> d n', d=d_model)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(d_model))
        self.out   = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_hsi: torch.Tensor, x_pan: torch.Tensor) -> torch.Tensor:
        """x_hsi, x_pan: (B, L, d_model) → (B, L, d_model)"""
        B, L, D = x_hsi.shape

        # HSI depthwise conv
        h = rearrange(x_hsi, 'b l d -> b d l')
        h = self.hsi_dw(h)[:, :, :L]
        h = rearrange(h, 'b d l -> b l d')
        h = self.hsi_act(h)

        # PAN projects SSM parameters
        pan_p = self.pan_proj(x_pan)
        B_param, C_param, dt_raw = torch.split(pan_p,
            [self.d_state, self.d_state, D], dim=-1)
        delta = F.softplus(self.dt_proj(dt_raw))
        A     = -torch.exp(self.A_log.float())

        # ZOH discretise
        a_bar   = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        b_bar_x = torch.einsum('bld,bln,bld->bldn', delta, B_param, h)

        # Hillis-Steele parallel scan
        h_all = hillis_steele_scan(a_bar, b_bar_x)
        y     = torch.einsum('bldn,bln->bld', h_all, C_param)
        y     = y + self.D * h

        return self.out(y)


class CrossMambaFusion(nn.Module):
    """
    Cross-Mamba Fusion replacing O(N²) cross-attention.
    Step 1: LearnableHFInjection
    Step 2: Row scan  (B×H sequences of length W)
    Step 3: Col scan  (B×W sequences of length H)
    Step 4: Concat + Conv1×1 + GN
    Step 5: Residual
    """

    def __init__(self, d_model: int, d_state: int = 8):
        super().__init__()
        ng = max(1, d_model//8)
        self.hf     = LearnableHFInjection(d_model)
        self.ssm_r  = CrossMambaSSM1D(d_model, d_state)
        self.ssm_c  = CrossMambaSSM1D(d_model, d_state)
        self.merge  = nn.Conv2d(d_model*2, d_model, 1, bias=False)
        self.norm   = nn.GroupNorm(ng, d_model)

    def forward(self, F_hsi, F_pan):
        if F_hsi.shape[2:] != F_pan.shape[2:]:
            F_hsi = F.interpolate(F_hsi, size=F_pan.shape[2:], mode='bilinear', align_corners=False)
        B, C, H, W = F_hsi.shape

        enh  = self.hf(F_hsi, F_pan)

        # Row scan
        yr = self.ssm_r(rearrange(enh, 'b c h w -> (b h) w c'),
                        rearrange(F_pan,'b c h w -> (b h) w c'))
        yr = rearrange(yr, '(b h) w c -> b c h w', b=B, h=H)

        # Col scan
        yc = self.ssm_c(rearrange(enh, 'b c h w -> (b w) h c'),
                        rearrange(F_pan,'b c h w -> (b w) h c'))
        yc = rearrange(yc, '(b w) h c -> b c h w', b=B, w=W)

        merged = self.norm(self.merge(torch.cat([yr, yc], dim=1)))
        return merged + enh


# ── 8. Reconstruction Head ────────────────────────────────────────────────────

class ReconstructionHead(nn.Module):
    """Residual head: HR = bicubic(LR) + Conv(features)."""

    def __init__(self, d_model: int, out_channels: int, scale: int):
        super().__init__()
        self.scale = scale
        ng = max(1, d_model//8)
        self.conv1   = nn.Conv2d(d_model, d_model, 3, padding=1)
        self.gn1     = nn.GroupNorm(ng, d_model)
        self.conv2   = nn.Conv2d(d_model, d_model, 3, padding=1)
        self.gn2     = nn.GroupNorm(ng, d_model)
        self.conv_out= nn.Conv2d(d_model, out_channels, 1)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x, lr_hsi):
        res   = self.relu(self.gn1(self.conv1(x)))
        res   = self.relu(self.gn2(self.conv2(res)))
        res   = self.conv_out(res)
        bicub = F.interpolate(lr_hsi, scale_factor=self.scale,
                              mode='bicubic', align_corners=False)
        return res + bicub


# ── 9. Full V2 Model ──────────────────────────────────────────────────────────

class V2VMambaPansharp(nn.Module):
    """
    Version 2 — Improved VMamba Pansharpening with True Parallel Scan.

    Improvements over V1:
      - ResidualUpsample (no PixelShuffle checkerboard)
      - CrossMambaFusion (O(L) vs O(N²))
      - LearnableHFInjection (adaptive vs fixed Sobel)
      - Hillis-Steele parallel scan everywhere (O(log L) depth)
      - Pre-norm Mamba blocks (more stable training)
    """

    def __init__(self,
                 in_channels:  int  = 128,
                 out_channels: int  = 128,
                 d_model:      int  = 32,
                 d_state:      int  = 8,
                 scale:        int  = 4,
                 num_blocks:   list[int] | None = None):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 1, 1, 1]

        self.scale = scale
        self.in_channels = in_channels

        self.hsi_encoder  = ImprovedHSIEncoder(in_channels, d_model, scale)
        self.pan_encoder  = ImprovedPANEncoder(d_model)
        self.fusion       = CrossMambaFusion(d_model, d_state)
        self.backbone     = VMambaBackboneV2(d_model, d_state, num_blocks)
        self.recon        = ReconstructionHead(d_model, out_channels, scale)

    def forward(self, lr_hsi: torch.Tensor, hr_pan: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lr_hsi : (B, C, H, W)       — per-band normalised LR-HSI
            hr_pan : (B, 1, sH, sW)     — HR-PAN (s = scale)
        Returns:
            hr_hsi : (B, C, sH, sW)     — reconstructed HR-HSI
        """
        F_hsi   = self.hsi_encoder(lr_hsi)              # (B,d,sH,sW)
        F_pan   = self.pan_encoder(hr_pan)               # (B,d,sH,sW)
        F_fused = self.fusion(F_hsi, F_pan)              # (B,d,sH,sW)

        # Memory-efficient backbone: halve spatial → process → restore
        F_lr  = F.interpolate(F_fused, scale_factor=0.5, mode='bilinear', align_corners=False)
        F_bb  = self.backbone(F_lr)
        F_hr  = F.interpolate(F_bb, scale_factor=2.0, mode='bilinear', align_corners=False)

        return self.recon(F_hr, lr_hsi)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total    = sum(p.numel() for p in model.parameters())
    trainable= sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    for scale in [2, 4]:
        m = V2VMambaPansharp(128, 128, d_model=32, d_state=8,
                             scale=scale, num_blocks=[1,1,1,1]).eval()
        lr  = torch.randn(1, 128, 8, 8)
        pan = torch.randn(1, 1, 8*scale, 8*scale)
        with torch.no_grad():
            out = m(lr, pan)
        total, _ = count_parameters(m)
        ok = 'OK' if out.shape == (1, 128, 8*scale, 8*scale) else 'FAIL'
        print(f"scale={scale}  params={total:,}  output={tuple(out.shape)}  [{ok}]")
