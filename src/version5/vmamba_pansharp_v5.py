"""
vmamba_pansharp_v5.py  —  Version 5 (V5)
M.Tech Thesis: Hyperspectral Image Pansharpening

Core Innovation: TRUE Parallel Scan (Hillis-Steele Algorithm)
═══════════════════════════════════════════════════════════════════════════════

WHY THE PREVIOUS SCAN WAS NOT TRULY PARALLEL
─────────────────────────────────────────────
V1–V4 "parallel_scan_optimized" is NOT a true parallel scan.
It is a CHUNKED SEQUENTIAL scan: it splits L tokens into chunks of 64 and
runs a Python for-loop of 64 iterations inside each chunk.

    True parallel scan:  O(log L) depth — log₂(32) = 5 steps for L=32
    Chunked sequential:  O(L) depth    — still 32 steps for L=32

The difference matters for:
  • Training speed  : chunked has L Python loop iters, H-S has log₂(L)
  • Gradient flow   : H-S tree = log₂(L) levels vs L levels (numerically stable)
  • GPU utilisation : each H-S step is FULLY vectorised over L/2 pairs at once

═══════════════════════════════════════════════════════════════════════════════

THE HILLIS-STEELE PARALLEL SCAN
─────────────────────────────────
For the linear recurrence  h_t = a_t · h_{t-1} + b_t,
define the binary associative operator ⊕:

    (a₂, b₂) ⊕ (a₁, b₁)  =  (a₂·a₁,  a₂·b₁ + b₂)

Identity element: (1, 0).

The Hillis-Steele algorithm runs log₂(L) passes, each pass d with stride=2^d:

    a_new[t] = a[t] · a[t − stride]          (prefix product so far)
    b_new[t] = a[t] · b[t − stride] + b[t]   (accumulated state so far)

After log₂(L) passes, b[t] = h_t for ALL t simultaneously.

Proof of correctness (L=4 example):
  Init:    a=[a0,a1,a2,a3]   b=[b0,b1,b2,b3]
  Pass 0:  b=[h0, h1, a2·b1+b2, a3·b2+b3]      (b1 still original)
  Pass 1:  b=[h0, h1,     h2,       h3 ]        ✓ all correct

Benefits over chunked sequential:
  L=32  → 32 Python iters → 5 Python iters  (6.4× reduction)
  L=64  → 64 Python iters → 6 Python iters  (10.7× reduction)
  L=1024→ 1024 iters      → 10 iters        (100× reduction)

═══════════════════════════════════════════════════════════════════════════════

V5 COMPONENTS (new or changed)
───────────────────────────────
  hillis_steele_scan()             — true parallel scan, O(log L) depth
  TrueParallelSelectiveSSM         — SSM using H-S scan (training + inference)
  TrueParallelSS2D                 — 4-direction 2D scan with direction adapters
  TrueParallelSpectralSSM1D        — spectral scan using H-S scan
  V5SpectralSpatialBlock           — pre-norm + TrueParallelSS2D + spectral SSM
  V5MultiScaleBackbone             — 4-stage U-Net with V5 blocks
  V5VMambaPansharp                 — full model

REUSED FROM V4 (unchanged):
  ImprovedHSIEncoder               — V2 encoder
  ImprovedPANEncoder               — V2 encoder
  ScaledSpatialDetailInjection     — V4 fusion (α learnable)
  StrongReconstructionHead         — V4 head (β learnable, 3-layer)
  CompositeLossV4                  — V4 loss (SpectralGradientLoss included)
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PROJECT, 'version3'))
sys.path.insert(0, os.path.join(_PROJECT, 'version4'))
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, 'scripts'))
sys.path.insert(0, _HERE)

from vmamba_pansharp_improved import ImprovedHSIEncoder, ImprovedPANEncoder
from vmamba_pansharp_v4 import ScaledSpatialDetailInjection, StrongReconstructionHead


# ============================================================================
# CORE: True Hillis-Steele Parallel Scan
# ============================================================================

def hillis_steele_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    True Hillis-Steele parallel prefix scan for linear recurrence:
        h_t = a_t · h_{t-1} + b_t,   h_{-1} = 0

    Algorithm — log₂(L) passes, each fully vectorised:
        Pass d (stride = 2^d):
            prev_a[t] = a[t − stride]  if t >= stride  else  1
            prev_b[t] = b[t − stride]  if t >= stride  else  0
            a[t] ← a[t] · prev_a[t]
            b[t] ← a[t] · prev_b[t] + b[t]          (uses OLD a[t])

    After log₂(L) passes, b[t] == h_t for all t.

    Complexity:
        Depth (sequential steps)  : O(log L)   — vs O(L) for chunked scan
        Work  (total operations)  : O(L log L)  — vs O(L) for sequential
        Python loop iterations    : log₂(L)     — vs L for sequential

    Args:
        a : (B, L, D, N)  decay coefficients in (0, 1]
        b : (B, L, D, N)  input contributions b_t = Δ_t · B_t · x_t

    Returns:
        h : (B, L, D, N)  all hidden states  h[t] = a[t]·h[t-1] + b[t]
    """
    B, L, D, N = a.shape
    L_orig = L

    # ── Pad to next power of 2 ────────────────────────────────────────────────
    if L & (L - 1):   # not a power of 2
        L_pad = 1 << math.ceil(math.log2(max(L, 2)))
        pad   = L_pad - L
        # Identity element: a=1 (no decay), b=0 (no input contribution)
        a = torch.cat([a, a.new_ones(B, pad, D, N)],  dim=1)
        b = torch.cat([b, b.new_zeros(B, pad, D, N)], dim=1)
        L = L_pad

    log2_L = int(round(math.log2(L)))

    # ── Hillis-Steele passes ───────────────────────────────────────────────────
    for d in range(log2_L):
        stride = 1 << d   # 1, 2, 4, 8, ...

        # Shift a and b right by stride (fill left with identity)
        prev_a = torch.cat([a.new_ones(B, stride, D, N),  a[:, :L - stride]], dim=1)
        prev_b = torch.cat([b.new_zeros(B, stride, D, N), b[:, :L - stride]], dim=1)

        # Both updates computed from OLD a and b before reassignment
        new_b = a * prev_b + b    # uses OLD a[t]
        new_a = a * prev_a        # product of 2^(d+1) consecutive terms

        a = new_a
        b = new_b                 # after this pass, b[t] = h_t up to depth 2^(d+1)

    return b[:, :L_orig]          # trim padding, return hidden states


# ============================================================================
# TrueParallelSelectiveSSM — Mamba SSM with Hillis-Steele scan
# ============================================================================

class TrueParallelSelectiveSSM(nn.Module):
    """
    Selective SSM (Mamba S6) using the Hillis-Steele true parallel scan.

    Key difference from V1–V4 SelectiveSSMOptimized:
    ──────────────────────────────────────────────────
    V1–V4:  sequential for-loop at training (L Python iters)
            chunked loop at inference   (L/64 outer iters × 64 inner iters)
    V5:     Hillis-Steele scan at BOTH training AND inference (log₂(L) iters)

    Same selective property: B, C, Δ are input-dependent (data-gated SSM).
    Same diagonal A matrix (efficient, stable).
    Same fused projections (in_proj, x_proj, dt_proj).
    Same SiLU gating and D skip connection.
    """

    def __init__(self, d_model: int, d_state: int = 8,
                 d_conv: int = 3, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # Fused input projection (x + gating residual in one matmul)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise Conv1D (groups=d_inner → channel-separable, cheap)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner,
                                kernel_size=d_conv, padding=d_conv - 1,
                                groups=self.d_inner, bias=True)

        # Fused SSM parameter projection: B, C, Δ from one matmul
        self.x_proj  = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Diagonal A matrix (negative → stable exponential decay)
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32),
                   'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip connection (D in the SSM equations)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act      = nn.SiLU()

        # Zero-init out_proj → identity residual at start of training
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        Returns: (B, L, d_model)

        Uses Hillis-Steele parallel scan for BOTH training and inference.
        Python loop depth: log₂(L) = 5 for L=32, vs L=32 for sequential.
        """
        B, L, D = x.shape

        # ── Fused input projection ────────────────────────────────────────────
        x_inner, res = self.in_proj(x).chunk(2, dim=-1)   # each (B,L,d_inner)

        # ── Depthwise Conv1D (causal short-range context) ─────────────────────
        x_inner = rearrange(x_inner, 'b l d -> b d l')
        x_inner = self.conv1d(x_inner)[:, :, :L]
        x_inner = rearrange(x_inner, 'b d l -> b l d')
        x_inner = self.act(x_inner)

        # ── Fused SSM parameter computation ───────────────────────────────────
        x_proj  = self.x_proj(x_inner)                    # (B,L, 2N+d_inner)
        B_param, C_param, delta_raw = torch.split(
            x_proj, [self.d_state, self.d_state, self.d_inner], dim=-1)
        delta = F.softplus(self.dt_proj(delta_raw))        # (B,L,d_inner) ≥ 0
        delta = delta.clamp(max=10.0)                      # prevent overflow in b_disc

        # ── Diagonal A (negative, stable decay) ───────────────────────────────
        A = -torch.exp(self.A_log.float().clamp(min=-8.0)) # (d_inner, d_state); clamp keeps a_disc > 0
        A = A.clamp(max=-1e-3)                             # ensure at least minimal decay

        # ── Discretise: ZOH ────────────────────────────────────────────────────
        # a_t = exp(Δ_t · A)    shape: (B, L, d_inner, d_state)
        # b_t = Δ_t · B_t · x_t (merged input contribution)
        a_disc = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        b_disc = torch.einsum('bld,bln,bld->bldn', delta, B_param, x_inner)

        # ── TRUE PARALLEL SCAN (Hillis-Steele) ────────────────────────────────
        # O(log L) depth — replaces O(L) sequential scan in V1–V4
        h_seq = hillis_steele_scan(a_disc, b_disc)        # (B,L,d_inner,d_state)

        # ── Output: y_t = C_t · h_t ───────────────────────────────────────────
        y = torch.einsum('bldn,bln->bld', h_seq, C_param) # (B,L,d_inner)

        # ── D skip + SiLU gate ─────────────────────────────────────────────────
        y = y + self.D[None, None, :] * x_inner
        y = y * self.act(res)

        return self.out_proj(y)                            # (B,L,d_model)


# ============================================================================
# TrueParallelSS2D — 4-direction 2D scan with direction-specific adapters
# ============================================================================

class TrueParallelSS2D(nn.Module):
    """
    4-direction 2D Selective Scan using TrueParallelSelectiveSSM.

    Scan directions (same as V1–V4 SS2DOptimized):
        Dir 0: row-forward    → scan each row left→right
        Dir 1: row-backward   → scan each row right→left
        Dir 2: col-forward    → scan each column top→bottom
        Dir 3: col-backward   → scan each column bottom→top

    V5 Enhancement — Direction-Specific Adapters:
        After each directional scan, a lightweight Conv1×1 re-weights the
        output so the model can learn that row-scans vs col-scans carry
        different information. Cost: 4 × d_model² parameters (≈4K for d=32).

    All 4 directions use the SAME TrueParallelSelectiveSSM parameters
    (weight-sharing as in V1–V4) to keep parameter count manageable.

    Training + Inference: ALWAYS uses Hillis-Steele scan.
    No training/inference mode switch needed.
    """

    def __init__(self, d_model: int, d_state: int = 8,
                 d_conv: int = 3, expand: int = 2):
        super().__init__()
        self.d_model = d_model

        # Shared SSM across all 4 directions
        self.ssm = TrueParallelSelectiveSSM(d_model, d_state, d_conv, expand)

        # Direction-specific 1×1 projections (lightweight post-processing)
        # Each learns to re-weight features based on scan orientation
        self.dir_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model, bias=False),
                nn.GroupNorm(min(8, d_model), d_model),
            ) for _ in range(4)
        ])

        # Merge all 4 directions into d_model
        self.merge = nn.Linear(d_model * 4, d_model, bias=False)
        self.norm  = nn.GroupNorm(min(8, d_model), d_model)

    def _run_dir(self, x_seq: torch.Tensor, dir_idx: int,
                 B: int, H: int, W: int,
                 is_row: bool, reverse: bool) -> torch.Tensor:
        """Run SSM in one scan direction and apply direction adapter."""
        out = self.ssm(x_seq)   # (batch, L, d_model)

        # Apply direction-specific projection (channel dim = last)
        # dir_proj expects (batch, L, d_model) → GroupNorm needs channel in dim1
        batch, L, C = out.shape
        out_2d = out.permute(0, 2, 1)                        # (batch, C, L)
        gn_in  = out_2d                                      # GroupNorm on C dim
        # Use LayerNorm-style: convert GN to work on (batch, C, L)
        out_proj = self.dir_proj[dir_idx][0](out)            # Linear: (batch,L,C)
        out_proj_t = out_proj.permute(0, 2, 1)               # (batch, C, L)
        out_normed  = self.dir_proj[dir_idx][1](out_proj_t)  # GN on C
        out = out_normed.permute(0, 2, 1)                    # (batch, L, C)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, C)  ← channel-last layout
        Returns: (B, H, W, C)
        """
        B, H, W, C = x.shape

        # ── Dir 0: Row forward ────────────────────────────────────────────────
        x0 = rearrange(x, 'b h w c -> (b h) w c')           # (B·H, W, C)
        y0 = self._run_dir(x0, 0, B, H, W, is_row=True, reverse=False)
        y0 = rearrange(y0, '(b h) w c -> b h w c', b=B, h=H)

        # ── Dir 1: Row backward ────────────────────────────────────────────────
        x1 = torch.flip(x0, dims=[1])
        y1 = self._run_dir(x1, 1, B, H, W, is_row=True, reverse=True)
        y1 = torch.flip(y1, dims=[1])
        y1 = rearrange(y1, '(b h) w c -> b h w c', b=B, h=H)

        # ── Dir 2: Col forward ────────────────────────────────────────────────
        x2 = rearrange(x, 'b h w c -> (b w) h c')           # (B·W, H, C)
        y2 = self._run_dir(x2, 2, B, H, W, is_row=False, reverse=False)
        y2 = rearrange(y2, '(b w) h c -> b h w c', b=B, w=W)

        # ── Dir 3: Col backward ───────────────────────────────────────────────
        x3 = torch.flip(x2, dims=[1])
        y3 = self._run_dir(x3, 3, B, H, W, is_row=False, reverse=True)
        y3 = torch.flip(y3, dims=[1])
        y3 = rearrange(y3, '(b w) h c -> b h w c', b=B, w=W)

        # ── Merge: concat all 4 directions → project ─────────────────────────
        y_cat = torch.cat([y0, y1, y2, y3], dim=-1)         # (B, H, W, 4C)
        y_flat = y_cat.reshape(B * H * W, 4 * C)
        y_out  = self.merge(y_flat).reshape(B, H, W, C)

        # Final GroupNorm (channel-last → permute)
        y_out = y_out.permute(0, 3, 1, 2)                   # (B, C, H, W)
        y_out = self.norm(y_out)
        y_out = y_out.permute(0, 2, 3, 1)                   # (B, H, W, C)

        return y_out


# ============================================================================
# TrueParallelSpectralSSM1D — spectral scan with Hillis-Steele
# ============================================================================

class TrueParallelSpectralSSM1D(nn.Module):
    """
    Spectral SSM scanning along the channel (band) dimension using Hillis-Steele.

    Groups d_model channels into n_groups spectral tokens.
    Hillis-Steele scan across n_groups tokens (L=8) — log₂(8)=3 passes only.

    Same logic as V3/V4 SpectralSSM1D but uses true parallel scan.
    For L=n_groups=8: scan depth = 3 instead of 8 (2.67× fewer loop iters).
    """

    def __init__(self, d_model: int, d_state: int = 4, n_groups: int = 8):
        super().__init__()
        # Adjust n_groups so d_model is divisible
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
        B, C, H, W = x.shape
        N = B * H * W   # each pixel is an independent spectral sequence

        # LayerNorm over channels
        x_perm = x.permute(0, 2, 3, 1).contiguous()          # (B, H, W, C)
        x_norm = self.norm(x_perm).reshape(N, C)              # (N, C)

        # Group channels → spectral tokens
        x_tok = x_norm.reshape(N, self.n_groups, self.d_per_grp)  # (N, grps, d)

        # True parallel scan along spectral dimension
        y_tok = self.ssm(x_tok)                                # (N, grps, d)

        # Reshape back
        y = y_tok.reshape(N, C)
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)         # (B, C, H, W)
        y = self.out_norm(y)

        return x + y   # residual connection


# ============================================================================
# V5SpectralSpatialBlock — Pre-norm + TrueParallelSS2D + SpectralSSM
# ============================================================================

class V5SpectralSpatialBlock(nn.Module):
    """
    V5 combined spatial + spectral Mamba block with pre-norm architecture.

    Pipeline (pre-norm is more stable than post-norm for deep networks):
        x  →  LayerNorm  →  TrueParallelSS2D   → + x   (spatial residual)
           →  TrueParallelSpectralSSM1D          → + x   (spectral residual)

    Pre-norm vs post-norm:
        Post-norm (V1–V4): y = Norm(SSM(x) + x)  → norm sees summed values
        Pre-norm (V5):     y = SSM(Norm(x)) + x   → SSM sees normalised input
        Pre-norm converges faster and is more robust to learning rate choice.
    """

    def __init__(self, dim: int, d_state: int = 8):
        super().__init__()
        self.pre_norm_spatial  = nn.LayerNorm(dim)
        self.spatial_ssm       = TrueParallelSS2D(dim, d_state=d_state)
        self.spectral_ssm      = TrueParallelSpectralSSM1D(dim, d_state=min(d_state, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # ── Spatial scan (pre-norm) ───────────────────────────────────────────
        x_hw = x.permute(0, 2, 3, 1)                  # (B, H, W, C)
        x_hw_normed = self.pre_norm_spatial(x_hw)
        y_spatial = self.spatial_ssm(x_hw_normed)     # (B, H, W, C)
        x = x + y_spatial.permute(0, 3, 1, 2)         # residual in channel-first

        # ── Spectral scan (with its own LayerNorm inside) ─────────────────────
        x = self.spectral_ssm(x)                      # (B, C, H, W) — has residual

        return x


# ============================================================================
# V5MultiScaleBackbone — 4-stage U-Net with V5 blocks
# ============================================================================

class V5MultiScaleBackbone(nn.Module):
    """
    4-stage hierarchical U-Net backbone using V5SpectralSpatialBlocks.

    Same structure as V3/V4 MultiScaleSpectralBackbone but all blocks
    now use:  TrueParallelSS2D (Hillis-Steele scan, O(log L) depth)
              + TrueParallelSpectralSSM1D (same, spectral dimension)
              + Pre-norm architecture (more stable training)

    Channel schedule: d→2d→4d_cap→4d_cap (cap=128 for OOM safety)
    Spatial schedule: H→H/2→H/4→H/8 (bottleneck = H/8)
    """

    _MAX_CH = 128

    def __init__(self, d_model: int, num_blocks=None, d_state: int = 8):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 1, 1, 1]

        d1 = d_model
        d2 = min(d_model * 2, self._MAX_CH)
        d3 = min(d_model * 4, self._MAX_CH)
        d4 = d3

        def make_stage(dim, n):
            return nn.Sequential(*[V5SpectralSpatialBlock(dim, d_state) for _ in range(n)])

        # ── Encoder ──────────────────────────────────────────────────────────
        self.stage1 = make_stage(d1, num_blocks[0])
        self.down1  = nn.Conv2d(d1, d2, 2, stride=2)

        self.stage2 = make_stage(d2, num_blocks[1])
        self.down2  = nn.Conv2d(d2, d3, 2, stride=2)

        self.stage3 = make_stage(d3, num_blocks[2])
        self.down3  = nn.Conv2d(d3, d4, 2, stride=2)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.stage4 = make_stage(d4, num_blocks[3])

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up3    = nn.ConvTranspose2d(d4, d3, 2, stride=2)
        self.merge3 = nn.Sequential(nn.Conv2d(d3 * 2, d3, 1, bias=False),
                                    nn.GroupNorm(min(8, d3), d3))
        self.dec3   = V5SpectralSpatialBlock(d3, d_state)

        self.up2    = nn.ConvTranspose2d(d3, d2, 2, stride=2)
        self.merge2 = nn.Sequential(nn.Conv2d(d2 * 2, d2, 1, bias=False),
                                    nn.GroupNorm(min(8, d2), d2))
        self.dec2   = V5SpectralSpatialBlock(d2, d_state)

        self.up1    = nn.ConvTranspose2d(d2, d1, 2, stride=2)
        self.merge1 = nn.Sequential(nn.Conv2d(d1 * 2, d1, 1, bias=False),
                                    nn.GroupNorm(min(8, d1), d1))
        self.dec1   = V5SpectralSpatialBlock(d1, d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.stage1(x)
        s2 = self.stage2(self.down1(s1))
        s3 = self.stage3(self.down2(s2))
        s4 = self.stage4(self.down3(s3))

        d3 = self.dec3(self.merge3(torch.cat([self.up3(s4), s3], dim=1)))
        d2 = self.dec2(self.merge2(torch.cat([self.up2(d3), s2], dim=1)))
        d1 = self.dec1(self.merge1(torch.cat([self.up1(d2), s1], dim=1)))

        return d1


# ============================================================================
# V5VMambaPansharp — Full Model
# ============================================================================

class V5VMambaPansharp(nn.Module):
    """
    Version 5 — VMamba Pansharpening with True Parallel Scan

    Architecture
    ─────────────────────────────────────────────────────────────────────────
    LR-HSI (B, C, H, W)
        │
        ▼  ImprovedHSIEncoder  (V2, unchanged)
        →  F_hsi  (B, d, rH, rW)

    HR-PAN (B, 1, rH, rW)
        │
        ▼  ImprovedPANEncoder  (V2, unchanged)
        →  F_pan  (B, d, rH, rW)

    F_hsi + F_pan
        │
        ▼  ScaledSpatialDetailInjection  (V4 — α learnable, init=0.1)
        →  F_fused (B, d, rH, rW)

    F_fused
        │
        ▼  V5MultiScaleBackbone  ← V5 NEW (Hillis-Steele parallel scan)
           4-stage U-Net, V5SpectralSpatialBlock at each stage
           V5SpectralSpatialBlock = pre-norm + TrueParallelSS2D + SpectralSSM
        →  F_out  (B, d, rH, rW)

    F_out + LR-HSI
        │
        ▼  StrongReconstructionHead  (V4 — β learnable, init=0.5)
           Bicubic(LR-HSI) + β × Residual(3×Conv3×3)
        →  HR-HSI (B, C, rH, rW)

    Scan complexity per block:
        SS2D (L=32 per row):       log₂(32) = 5 H-S passes  (was 32 sequential)
        SpectralSSM (L=n_grp=8):   log₂(8)  = 3 H-S passes  (was 8 sequential)
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

        # Encoders (V2, unchanged)
        self.hsi_encoder = ImprovedHSIEncoder(in_channels, d_model, scale)
        self.pan_encoder = ImprovedPANEncoder(d_model)

        # V4 scaled injection (α learnable)
        self.sdim = ScaledSpatialDetailInjection(d_model, alpha_init=alpha_init)

        # V5 backbone (true parallel scan)
        self.backbone = V5MultiScaleBackbone(d_model, num_blocks, d_state)

        # V4 strong reconstruction head (β learnable, 3-layer)
        self.recon = StrongReconstructionHead(d_model, out_channels, scale, beta_init)

    def forward(self, lr_hsi: torch.Tensor, hr_pan: torch.Tensor) -> torch.Tensor:
        F_hsi   = self.hsi_encoder(lr_hsi)
        F_pan   = self.pan_encoder(hr_pan)
        F_fused = self.sdim(F_hsi, F_pan)
        F_out   = self.backbone(F_fused)
        return self.recon(F_out, lr_hsi)


# ============================================================================
# Quick self-test
# ============================================================================

if __name__ == '__main__':
    print("Testing V5VMambaPansharp (True Parallel Scan) …")
    device = torch.device('cpu')

    model = V5VMambaPansharp(
        in_channels=128, out_channels=128,
        d_model=32, d_state=4, scale=4, num_blocks=[1, 1, 1, 1],
    ).to(device).eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total:,}  ({total/1e6:.3f} M)")

    lr  = torch.randn(1, 128, 8, 8)
    pan = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        out = model(lr, pan)

    assert out.shape == (1, 128, 32, 32), f"Shape error: {out.shape}"
    print(f"  Output : {tuple(out.shape)}  [OK]")
    print(f"  alpha={model.sdim.alpha.item():.4f}  beta={model.recon.beta.item():.4f}")

    # Verify parallel scan correctness vs sequential
    print("\n  Verifying Hillis-Steele == sequential scan …")
    torch.manual_seed(0)
    L, D, N = 32, 16, 4
    a = torch.rand(1, L, D, N) * 0.9 + 0.05  # decay in (0.05, 0.95)
    b = torch.randn(1, L, D, N)

    # Sequential reference
    h_seq = torch.zeros(1, L, D, N)
    h = torch.zeros(1, D, N)
    for t in range(L):
        h = a[:, t] * h + b[:, t]
        h_seq[:, t] = h

    # Hillis-Steele
    h_hs = hillis_steele_scan(a, b)

    max_err = (h_seq - h_hs).abs().max().item()
    print(f"  Max absolute error vs sequential: {max_err:.2e}")
    assert max_err < 1e-4, f"Scan mismatch: {max_err}"
    print("  [OK] Scan is numerically equivalent to sequential.")
