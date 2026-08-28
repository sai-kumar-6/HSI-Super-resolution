"""
Improved VMamba-Pansharp Model
M.Tech Thesis Project – Hyperspectral Pansharpening

Three key improvements over the baseline (vmamba_pansharp.py):

  Improvement 1 – Cross-Mamba Fusion (replaces O(N²) CrossAttention):
      h_t = A(x_pan)*h_{t-1} + B(x_pan)*x_hsi
      y_t = C(x_pan)*h_t

  Improvement 2 – Learnable High-Frequency Injection (replaces fixed Sobel):
      Edge  = Conv3x3(PAN)
      Gate  = sigmoid(Conv1x1(PAN))
      Output = HSI + Gate * Edge

  Improvement 3 – Residual Upsampling (replaces PixelShuffle):
      Main     = BilinearUpsample → Conv3x3 → GN → ReLU
      Residual = Conv3x3 → BilinearUpsample
      Output   = Main + Residual
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# Import shared building blocks from the baseline
# ---------------------------------------------------------------------------
from vmamba_pansharp import (
    SelectiveSSMOptimized,
    SS2DOptimized,
    MambaVisionBlockOptimized,
    VMambaBackbone,
    ReconstructionHead,
)


# ============================================================================
# Improvement 3 – Residual Upsampling Block (replaces PixelShuffle)
# ============================================================================

class ResidualUpsample2x(nn.Module):
    """
    2× residual upsampling block.

    Main path   : BilinearUpsample → Conv3x3 → GroupNorm → ReLU
    Residual    : Conv3x3 → BilinearUpsample
    Output      : Main + Residual

    This prevents checkerboard artefacts (PixelShuffle) while keeping a
    learned residual shortcut for gradient flow.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        # ---------- main path ----------
        self.conv_main = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn_main   = nn.GroupNorm(max(1, out_ch // 8), out_ch)
        self.relu      = nn.ReLU(inplace=True)

        # ---------- residual path ----------
        self.conv_res = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

        # ---------- weight init ----------
        nn.init.kaiming_normal_(self.conv_main.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_res.weight,  mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, in_ch, H, W)
        Returns:
            out : (B, out_ch, 2H, 2W)
        """
        # --- main path ---
        x_up   = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        main   = self.relu(self.gn_main(self.conv_main(x_up)))

        # --- residual path ---
        res_c  = self.conv_res(x)
        res    = F.interpolate(res_c, scale_factor=2, mode='bilinear', align_corners=False)

        return main + res


# ============================================================================
# Improvement 2 – Learnable High-Frequency Injection (replaces fixed Sobel)
# ============================================================================

class LearnableHFInjection(nn.Module):
    """
    Learnable High-Frequency Injection module.

    Given aligned HSI and PAN feature maps of the same spatial size:
        edge   = Conv3x3(pan)
        gate   = sigmoid(Conv1x1(pan))
        output = GroupNorm(hsi + gate * edge)

    The network decides *which* high-frequency details to inject and
    *how strongly*, adapting to the scene content.
    """

    def __init__(self, d_model: int):
        super().__init__()
        num_groups = max(1, d_model // 8)

        self.edge_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=True)
        self.gate_conv = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.norm      = nn.GroupNorm(num_groups, d_model)

        nn.init.kaiming_normal_(self.edge_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.edge_conv.bias)
        nn.init.kaiming_normal_(self.gate_conv.weight, mode='fan_out', nonlinearity='sigmoid')
        nn.init.zeros_(self.gate_conv.bias)

    def forward(self, hsi: torch.Tensor, pan: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hsi : (B, d_model, H, W) – HSI feature map
            pan : (B, d_model, H, W) – PAN feature map (same resolution)
        Returns:
            out : (B, d_model, H, W) – enhanced HSI features
        """
        # Ensure spatial alignment
        if hsi.shape[2:] != pan.shape[2:]:
            pan = F.interpolate(pan, size=hsi.shape[2:], mode='bilinear', align_corners=False)

        edge = self.edge_conv(pan)                        # (B, d_model, H, W)
        gate = torch.sigmoid(self.gate_conv(pan))         # (B, d_model, H, W)
        injected = hsi + gate * edge                      # element-wise injection
        return self.norm(injected)


# ============================================================================
# Improvement 1 – Cross-Mamba SSM (1-D scan, PAN-guided parameters)
# ============================================================================

class CrossMambaSSM1D(nn.Module):
    """
    Cross-Mamba selective SSM scan (1-D along a sequence dimension).

    PAN features project the SSM parameters B, C, and Δ (dt), making the
    state-transition data-dependent on PAN.  HSI features are the *input*
    signal that is filtered by the PAN-guided SSM:

        h_t = exp(Δ·A) * h_{t-1}  +  Δ·B(x_pan) * x_hsi_t
        y_t = C(x_pan) * h_t

    This achieves O(L) cross-modal fusion instead of O(L²) cross-attention.

    Args:
        d_model  : feature dimension
        d_state  : SSM state dimension  (N in the equations above)
        d_conv   : depthwise conv kernel size on HSI input
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv  = d_conv

        # --- HSI branch: depthwise temporal conv ---
        self.hsi_dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_model, bias=True
        )
        self.hsi_act = nn.SiLU()

        # --- PAN branch: projects to SSM parameters ---
        # B_param: (L, d_state), C_param: (L, d_state), dt: (L, d_model)
        self.pan_proj = nn.Linear(d_model, d_state * 2 + d_model, bias=False)
        self.dt_proj  = nn.Linear(d_model, d_model, bias=True)

        # Learnable A matrix (log-parameterised, negative → stable decay)
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            'n -> d n', d=d_model
        )
        self.A_log = nn.Parameter(torch.log(A))

        # Skip connection D
        self.D = nn.Parameter(torch.ones(d_model))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    # ------------------------------------------------------------------
    def _sequential_scan(
        self,
        x_hsi: torch.Tensor,    # (B, L, d_model) – filtered HSI input
        delta:  torch.Tensor,   # (B, L, d_model) – Δ (softplus)
        A:      torch.Tensor,   # (d_model, d_state)
        B_param: torch.Tensor,  # (B, L, d_state)
        C_param: torch.Tensor,  # (B, L, d_state)
    ) -> torch.Tensor:
        """Memory-safe sequential scan used during training."""
        B_batch, L, D = x_hsi.shape
        N = A.shape[1]

        # Discretise A and B×x once for all time steps
        deltaA   = torch.exp(torch.einsum('bld,dn->bldn', delta, A))    # (B,L,D,N)
        deltaB_x = torch.einsum('bld,bln,bld->bldn', delta, B_param, x_hsi)  # (B,L,D,N)

        h = torch.zeros(B_batch, D, N, dtype=x_hsi.dtype, device=x_hsi.device)
        outputs = []
        for i in range(L):
            h = deltaA[:, i] * h + deltaB_x[:, i]          # (B, D, N)
            y_i = torch.einsum('bdn,bn->bd', h, C_param[:, i])  # (B, D)
            outputs.append(y_i)

        return torch.stack(outputs, dim=1)  # (B, L, D)

    # ------------------------------------------------------------------
    def forward(self, x_hsi: torch.Tensor, x_pan: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_hsi : (B, L, d_model) – HSI sequence (rows or columns)
            x_pan : (B, L, d_model) – PAN sequence (same ordering)
        Returns:
            out   : (B, L, d_model)
        """
        B, L, D = x_hsi.shape

        # --- HSI depthwise conv (short-range context on input signal) ---
        hsi_t = rearrange(x_hsi, 'b l d -> b d l')
        hsi_t = self.hsi_dw_conv(hsi_t)[:, :, :L]          # causal trim
        hsi_t = rearrange(hsi_t, 'b d l -> b l d')
        hsi_t = self.hsi_act(hsi_t)

        # --- PAN projects SSM parameters ---
        pan_proj = self.pan_proj(x_pan)                     # (B, L, 2N+D)
        B_param, C_param, dt_raw = torch.split(
            pan_proj, [self.d_state, self.d_state, self.d_model], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_raw))            # (B, L, D) ≥ 0

        A = -torch.exp(self.A_log.float())                  # (D, N) negative

        # --- SSM scan ---
        y = self._sequential_scan(hsi_t, delta, A, B_param, C_param)

        # Skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * hsi_t   # (B, L, D)

        return self.out_proj(y)                             # (B, L, D)


# ============================================================================
# Cross-Mamba Fusion Module (row + column scans)
# ============================================================================

class CrossMambaFusion(nn.Module):
    """
    Cross-Mamba Fusion: replaces O(N²) CrossAttentionFusion.

    Pipeline:
      1. LearnableHFInjection(F_hsi, F_pan)   → F_hsi_enhanced
      2. Row scan    : B*H sequences of length W
      3. Column scan : B*W sequences of length H
      4. Concat row + col outputs → 1×1 Conv → GroupNorm
      5. Add residual F_hsi_enhanced

    Args:
        d_model : feature dimension
        d_state : SSM state dimension
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model

        self.hf_inject  = LearnableHFInjection(d_model)
        self.ssm_row    = CrossMambaSSM1D(d_model, d_state)
        self.ssm_col    = CrossMambaSSM1D(d_model, d_state)

        num_groups      = max(1, d_model // 8)
        self.merge_conv = nn.Conv2d(d_model * 2, d_model, kernel_size=1, bias=False)
        self.norm       = nn.GroupNorm(num_groups, d_model)

    def forward(self, F_hsi: torch.Tensor, F_pan: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F_hsi : (B, d_model, H, W)
            F_pan : (B, d_model, H, W)
        Returns:
            fused : (B, d_model, H, W)
        """
        # Align spatial sizes
        if F_hsi.shape[2:] != F_pan.shape[2:]:
            F_hsi = F.interpolate(F_hsi, size=F_pan.shape[2:], mode='bilinear', align_corners=False)

        B, C, H, W = F_hsi.shape

        # Step 1 – Learnable HF injection
        F_hsi_enh = self.hf_inject(F_hsi, F_pan)           # (B, C, H, W)

        # Step 2 – Row scan  (B*H sequences of length W)
        hsi_row = rearrange(F_hsi_enh, 'b c h w -> (b h) w c')
        pan_row = rearrange(F_pan,     'b c h w -> (b h) w c')
        y_row   = self.ssm_row(hsi_row, pan_row)            # (B*H, W, C)
        y_row   = rearrange(y_row, '(b h) w c -> b c h w', b=B, h=H)

        # Step 3 – Column scan  (B*W sequences of length H)
        hsi_col = rearrange(F_hsi_enh, 'b c h w -> (b w) h c')
        pan_col = rearrange(F_pan,     'b c h w -> (b w) h c')
        y_col   = self.ssm_col(hsi_col, pan_col)            # (B*W, H, C)
        y_col   = rearrange(y_col, '(b w) h c -> b c h w', b=B, w=W)

        # Step 4 – Merge row + col
        merged  = torch.cat([y_row, y_col], dim=1)          # (B, 2C, H, W)
        merged  = self.norm(self.merge_conv(merged))        # (B, C, H, W)

        # Step 5 – Residual
        return merged + F_hsi_enh


# ============================================================================
# Improved HSI Encoder (Improvement 3 – Residual Upsampling)
# ============================================================================

class ImprovedHSIEncoder(nn.Module):
    """
    HSI Encoder with 3D Conv + ResidualUpsample (replaces PixelShuffle).

    Architecture:
        Conv3d(1→8, k=3, pad=1) → GroupNorm → ReLU
        reshape: (B, 8*C, H, W)
        Conv2d(8*C → d_model, k=1) + ReLU   (spectral aggregation)
        ResidualUpsample2x × (2 for scale=4, 1 for scale=2)

    Args:
        in_channels : spectral bands of LR-HSI  (128 for Chikusei)
        d_model     : feature dimension
        scale       : spatial upsampling factor (2 or 4)
    """

    def __init__(self, in_channels: int, d_model: int = 64, scale: int = 4):
        super().__init__()
        self.scale = scale
        feat_3d = 8           # 3D conv output channels

        # 3D convolution: joint spectral-spatial feature extraction
        self.conv3d = nn.Conv3d(1, feat_3d, kernel_size=3, padding=1, bias=False)
        self.gn3d   = nn.GroupNorm(max(1, feat_3d // 2), feat_3d)
        self.relu   = nn.ReLU(inplace=True)

        # Spectral aggregation: collapse spectral dimension → d_model channels
        self.spec_agg = nn.Conv2d(feat_3d * in_channels, d_model, kernel_size=1, bias=False)

        # Residual upsampling stages
        if scale == 4:
            self.up1 = ResidualUpsample2x(d_model, d_model)
            self.up2 = ResidualUpsample2x(d_model, d_model)
        elif scale == 2:
            self.up1 = ResidualUpsample2x(d_model, d_model)
            self.up2 = None
        else:
            raise ValueError(f"Unsupported scale={scale}. Use 2 or 4.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (B, C, H, W)  LR-HSI
        Returns:
            out : (B, d_model, rH, rW)  where r = scale
        """
        B, C, H, W = x.shape

        # 3D conv: add channel dim → (B, 1, C, H, W)
        x3d = x.unsqueeze(1)
        x3d = self.relu(self.gn3d(self.conv3d(x3d)))   # (B, 8, C, H, W)

        # Reshape to 2D: (B, 8*C, H, W)
        x2d = x3d.reshape(B, -1, H, W)

        # Spectral aggregation
        x2d = self.relu(self.spec_agg(x2d))            # (B, d_model, H, W)

        # Residual upsampling
        x2d = self.up1(x2d)                            # (B, d_model, 2H, 2W)
        if self.up2 is not None:
            x2d = self.up2(x2d)                        # (B, d_model, 4H, 4W)

        return x2d


# ============================================================================
# Improved PAN Encoder (Kaiming init, clean 3-layer conv)
# ============================================================================

class ImprovedPANEncoder(nn.Module):
    """
    PAN Encoder: three Conv2d + GroupNorm + ReLU layers with Kaiming init.

    Deliberately *no* hard-coded Sobel: high-frequency detail is handled
    by LearnableHFInjection inside CrossMambaFusion.

    Args:
        d_model : output feature dimension
    """

    def __init__(self, d_model: int = 64):
        super().__init__()

        self.conv1 = nn.Conv2d(1,  32,      kernel_size=3, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(8, 32)

        self.conv2 = nn.Conv2d(32, 64,      kernel_size=3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(8, 64)

        self.conv3 = nn.Conv2d(64, d_model, kernel_size=3, padding=1, bias=False)
        self.gn3   = nn.GroupNorm(max(1, d_model // 8), d_model)

        self.relu  = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (B, 1, rH, rW)  HR-PAN
        Returns:
            out : (B, d_model, rH, rW)
        """
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.relu(self.gn3(self.conv3(x)))
        return x


# ============================================================================
# Complete Improved VMamba-Pansharp Model
# ============================================================================

class ImprovedVMambaPansharp(nn.Module):
    """
    Improved VMamba-Pansharp model combining all three improvements:

      - ImprovedHSIEncoder   (Improvement 3 – Residual Upsampling)
      - ImprovedPANEncoder   (clean, Kaiming-initialised)
      - CrossMambaFusion     (Improvements 1 + 2 – Cross-Mamba + Learnable HF)
      - VMambaBackbone       (shared with baseline, hierarchical U-Net)
      - ReconstructionHead   (shared with baseline, residual learning)

    Forward pipeline:
        F_hsi  = ImprovedHSIEncoder(lr_hsi)          # (B, d, rH, rW)
        F_pan  = ImprovedPANEncoder(hr_pan)           # (B, d, rH, rW)
        F_fused = CrossMambaFusion(F_hsi, F_pan)      # (B, d, rH, rW)
        F_lr    = ↓0.5× F_fused   (memory-efficient backbone input)
        F_bb    = VMambaBackbone(F_lr)                # (B, d, rH/2, rW/2)
        F_hr    = ↑2×  F_bb                          # (B, d, rH, rW)
        hr_hsi  = ReconstructionHead(F_hr, lr_hsi)   # (B, C, rH, rW)

    Args:
        in_channels  : spectral bands of input LR-HSI  (128 for Chikusei)
        out_channels : spectral bands of output HR-HSI (128 for Chikusei)
        d_model      : internal feature dimension
        d_state      : SSM state size
        scale        : spatial upsampling factor
        num_blocks   : per-stage block counts for VMambaBackbone
    """

    def __init__(
        self,
        in_channels:  int = 128,
        out_channels: int = 128,
        d_model:      int = 64,
        d_state:      int = 16,
        scale:        int = 4,
        num_blocks:   list = None,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 3, 3, 2]

        self.scale       = scale
        self.in_channels = in_channels

        # Encoders
        self.hsi_encoder = ImprovedHSIEncoder(in_channels, d_model, scale)
        self.pan_encoder = ImprovedPANEncoder(d_model)

        # Cross-Mamba Fusion
        self.fusion = CrossMambaFusion(d_model, d_state)

        # VMamba backbone (reused from baseline; expects d_model input)
        self.backbone = VMambaBackbone(d_model=d_model, num_blocks=num_blocks)

        # Reconstruction head (reused from baseline)
        self.reconstruction = ReconstructionHead(d_model, out_channels, scale)

    def forward(self, lr_hsi: torch.Tensor, hr_pan: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lr_hsi : (B, C, H, W)     Low-resolution HSI
            hr_pan : (B, 1, rH, rW)   High-resolution PAN  (rH = scale*H)
        Returns:
            hr_hsi : (B, C, rH, rW)   Reconstructed HR-HSI
        """
        # ---- Encode ----
        F_hsi = self.hsi_encoder(lr_hsi)   # (B, d, rH, rW)
        F_pan = self.pan_encoder(hr_pan)   # (B, d, rH, rW)

        # ---- Cross-Mamba fusion at HR scale ----
        F_fused = self.fusion(F_hsi, F_pan)  # (B, d, rH, rW)

        # ---- Downsample for memory-efficient backbone processing ----
        # 0.5× keeps spatial size manageable while retaining rich features
        F_lr = F.interpolate(F_fused, scale_factor=0.5, mode='bilinear', align_corners=False)

        # ---- Hierarchical VMamba backbone ----
        F_bb = self.backbone(F_lr)         # (B, d, rH/2, rW/2)

        # ---- Upsample back to HR scale ----
        F_hr = F.interpolate(F_bb, scale_factor=2.0, mode='bilinear', align_corners=False)

        # ---- Reconstruct HR-HSI ----
        hr_hsi = self.reconstruction(F_hr, lr_hsi)  # (B, C, rH, rW)

        return hr_hsi


# ============================================================================
# Quick integration test
# ============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("  Improved VMamba-Pansharp – Integration Test")
    print(f"{'='*70}")
    print(f"Device : {device}")

    # Chikusei-like dimensions: 128 bands, LR=16×16, HR=64×64, scale=4
    B      = 2
    C      = 128
    H_lr   = 16
    W_lr   = 16
    scale  = 4
    H_hr   = H_lr * scale
    W_hr   = W_lr * scale

    model = ImprovedVMambaPansharp(
        in_channels  = C,
        out_channels = C,
        d_model      = 64,
        d_state      = 16,
        scale        = scale,
        num_blocks   = [2, 3, 3, 2],
    ).to(device)

    lr_hsi = torch.randn(B, C, H_lr, W_lr).to(device)
    hr_pan = torch.randn(B, 1, H_hr, W_hr).to(device)

    model.train()
    with torch.no_grad():
        hr_hsi = model(lr_hsi, hr_pan)

    print(f"\nInput  LR-HSI : {tuple(lr_hsi.shape)}")
    print(f"Input  HR-PAN : {tuple(hr_pan.shape)}")
    print(f"Output HR-HSI : {tuple(hr_hsi.shape)}")

    assert hr_hsi.shape == (B, C, H_hr, W_hr), \
        f"Shape mismatch! Expected {(B, C, H_hr, W_hr)}, got {tuple(hr_hsi.shape)}"

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")

    # Component-level check
    print("\n--- Component shapes ---")
    with torch.no_grad():
        F_hsi   = model.hsi_encoder(lr_hsi)
        F_pan   = model.pan_encoder(hr_pan)
        F_fused = model.fusion(F_hsi, F_pan)
    print(f"  HSI encoder output : {tuple(F_hsi.shape)}")
    print(f"  PAN encoder output : {tuple(F_pan.shape)}")
    print(f"  CrossMamba fused   : {tuple(F_fused.shape)}")

    print(f"\n{'='*70}")
    print("  [PASSED] All checks completed successfully.")
    print(f"{'='*70}\n")
