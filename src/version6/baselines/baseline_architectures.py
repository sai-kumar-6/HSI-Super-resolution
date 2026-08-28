"""
baseline_architectures.py
=========================
Classical & deep-learning pansharpening baselines for comparison with V6.

Methods implemented:
  1. Bicubic         — no parameters, pure interpolation
  2. PanNet          — Yang et al., 2017  (ICCV)
  3. PanGAN          — Ma et al., 2020   (GAN-based)
  4. PSGAN           — Dong et al., 2021 (spectral attention GAN)
  5. Panformer       — He et al., 2022   (cross-attention transformer)

All models share the same I/O contract as V6:
    forward(lr_hsi, hr_pan) -> hr_hsi_pred
    lr_hsi : (B, C, H_lr, W_lr)
    hr_pan : (B, 1, H_hr, W_hr)   H_hr = H_lr * scale
    output : (B, C, H_hr, W_hr)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter as scipy_gaussian
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bicubic baseline (no parameters)
# ─────────────────────────────────────────────────────────────────────────────
class BicubicBaseline(nn.Module):
    """Spectral-lossless bicubic upsample of LR-HSI. No learning."""
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale

    def forward(self, lr_hsi, hr_pan):
        return F.interpolate(lr_hsi, scale_factor=self.scale,
                             mode='bicubic', align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PanNet  (Yang et al., ICCV 2017)
# ─────────────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
    def forward(self, x): return x + self.net(x)


class PanNet(nn.Module):
    """
    PanNet: Learns residual over bicubic baseline.
    Input: concat(bicubic_LR_HSI, HPF_PAN)  [C+1 channels]
    HPF = PAN - low_pass(PAN)  (high-pass filter injects spatial detail)
    """
    def __init__(self, in_channels=128, scale=4, num_res=8, feat=32):
        super().__init__()
        self.scale = scale
        self.head  = nn.Conv2d(in_channels + 1, feat, 3, padding=1, bias=False)
        self.body  = nn.Sequential(*[ResBlock(feat) for _ in range(num_res)])
        self.tail  = nn.Conv2d(feat, in_channels, 3, padding=1, bias=False)

    @staticmethod
    def high_pass(pan):
        """High-pass filter: PAN - Gaussian-smoothed PAN."""
        # pan: (B,1,H,W) tensor
        B, _, H, W = pan.shape
        out = []
        for b in range(B):
            p = pan[b, 0].cpu().numpy()
            lp = scipy_gaussian(p, sigma=1.5)
            hp = p - lp
            out.append(torch.from_numpy(hp).float())
        hp_t = torch.stack(out, 0).unsqueeze(1).to(pan.device)
        return hp_t

    def forward(self, lr_hsi, hr_pan):
        bicubic = F.interpolate(lr_hsi, scale_factor=self.scale,
                                mode='bicubic', align_corners=False)
        hp = self.high_pass(hr_pan)                          # (B,1,H_hr,W_hr)
        x  = torch.cat([bicubic, hp], dim=1)                # (B,C+1,H,W)
        x  = self.head(x)
        x  = self.body(x)
        x  = self.tail(x)
        return bicubic + x                                   # residual learning


# ─────────────────────────────────────────────────────────────────────────────
# 3. PanGAN  (Ma et al., 2020)
# ─────────────────────────────────────────────────────────────────────────────
class PanGAN_Generator(nn.Module):
    """
    PanGAN generator: U-Net that fuses bicubic HSI + PAN.
    """
    def __init__(self, in_channels=128, scale=4, feat=64):
        super().__init__()
        self.scale = scale
        inp = in_channels + 1   # bicubic HSI + PAN

        def _down(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(oc), nn.LeakyReLU(0.2, inplace=True))

        def _up(ic, oc):
            return nn.Sequential(
                nn.ConvTranspose2d(ic, oc, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(oc), nn.ReLU(inplace=True))

        self.e1 = nn.Conv2d(inp, feat,    3, padding=1, bias=False)
        self.e2 = _down(feat,    feat*2)
        self.e3 = _down(feat*2,  feat*4)
        self.bn = _down(feat*4,  feat*4)

        self.d3 = _up(feat*4,   feat*4)
        self.d2 = _up(feat*8,   feat*2)
        self.d1 = _up(feat*4,   feat)
        self.out = nn.Sequential(
            nn.Conv2d(feat*2, in_channels, 3, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, lr_hsi, hr_pan):
        bicubic = F.interpolate(lr_hsi, scale_factor=self.scale,
                                mode='bicubic', align_corners=False)
        x  = torch.cat([bicubic, hr_pan], dim=1)
        e1 = F.relu(self.e1(x),  inplace=True)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        bn = self.bn(e3)
        d3 = self.d3(bn)
        d2 = self.d2(torch.cat([d3, e3], 1))
        d1 = self.d1(torch.cat([d2, e2], 1))
        res = self.out(torch.cat([d1, e1], 1))
        # res is in [-1,1]; scale to residual and clip
        return (bicubic + res * 0.1).clamp(0, 1)


class PanGAN_Discriminator(nn.Module):
    """PatchGAN discriminator."""
    def __init__(self, in_channels=128):
        super().__init__()
        def blk(ic, oc, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, stride=2, padding=1, bias=not norm)]
            if norm: layers.append(nn.InstanceNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        self.net = nn.Sequential(
            blk(in_channels, 64, norm=False),
            blk(64, 128), blk(128, 256),
            nn.Conv2d(256, 1, 4, padding=1),
        )
    def forward(self, x): return self.net(x)


class PanGAN(nn.Module):
    """Wrapper exposing generator forward only (for inference)."""
    def __init__(self, in_channels=128, scale=4):
        super().__init__()
        self.generator     = PanGAN_Generator(in_channels, scale)
        self.discriminator = PanGAN_Discriminator(in_channels)

    def forward(self, lr_hsi, hr_pan):
        return self.generator(lr_hsi, hr_pan)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PSGAN  (Dong et al., 2021) — Spectral-attention GAN
# ─────────────────────────────────────────────────────────────────────────────
class SpectralAttention(nn.Module):
    """Channel-wise spectral attention (SE-style)."""
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, max(1, ch//r)), nn.ReLU(inplace=True),
            nn.Linear(max(1, ch//r), ch), nn.Sigmoid(),
        )
    def forward(self, x):
        w = self.fc(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


class PSGANResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(ch),
        )
        self.sa = SpectralAttention(ch)

    def forward(self, x): return x + self.sa(self.conv(x))


class PSGAN_Generator(nn.Module):
    """
    PSGAN generator with spectral attention blocks.
    """
    def __init__(self, in_channels=128, scale=4, feat=64, n_blocks=6):
        super().__init__()
        self.scale = scale
        inp = in_channels + 1

        self.head = nn.Sequential(
            nn.Conv2d(inp, feat, 7, padding=3, bias=False),
            nn.InstanceNorm2d(feat), nn.ReLU(inplace=True),
        )
        # Downsampling
        self.down = nn.Sequential(
            nn.Conv2d(feat,   feat*2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feat*2), nn.ReLU(inplace=True),
            nn.Conv2d(feat*2, feat*4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feat*4), nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[PSGANResBlock(feat*4) for _ in range(n_blocks)])
        # Upsampling
        self.up = nn.Sequential(
            nn.ConvTranspose2d(feat*4, feat*2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feat*2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat*2, feat,   4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feat),   nn.ReLU(inplace=True),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(feat, in_channels, 7, padding=3, bias=False),
            nn.Tanh(),
        )

    def forward(self, lr_hsi, hr_pan):
        bicubic = F.interpolate(lr_hsi, scale_factor=self.scale,
                                mode='bicubic', align_corners=False)
        x = torch.cat([bicubic, hr_pan], dim=1)
        x = self.head(x)
        x = self.down(x)
        x = self.body(x)
        x = self.up(x)
        res = self.tail(x)
        return (bicubic + res * 0.1).clamp(0, 1)


class PSGAN_Discriminator(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        def blk(ic, oc, norm=True):
            l = [nn.Conv2d(ic, oc, 4, stride=2, padding=1, bias=not norm)]
            if norm: l.append(nn.InstanceNorm2d(oc))
            l.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*l)
        self.net = nn.Sequential(
            blk(in_channels+1, 64, False),
            blk(64, 128), blk(128, 256),
            nn.Conv2d(256, 1, 4, padding=1),
        )
    def forward(self, hsi, pan): return self.net(torch.cat([hsi, pan], 1))


class PSGAN(nn.Module):
    def __init__(self, in_channels=128, scale=4):
        super().__init__()
        self.generator     = PSGAN_Generator(in_channels, scale)
        self.discriminator = PSGAN_Discriminator(in_channels)

    def forward(self, lr_hsi, hr_pan):
        return self.generator(lr_hsi, hr_pan)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Panformer  (He et al., 2022)
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttention(nn.Module):
    """
    Cross-attention: HSI features (query) attend to PAN features (key/value).
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = max(1, dim // num_heads)
        inner = self.head_dim * num_heads
        self.q = nn.Linear(dim,   inner, bias=False)
        self.k = nn.Linear(dim,   inner, bias=False)
        self.v = nn.Linear(dim,   inner, bias=False)
        self.proj = nn.Linear(inner, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, hsi_feat, pan_feat):
        # hsi_feat, pan_feat: (B, L, dim)
        B, L, _ = hsi_feat.shape
        H = self.num_heads
        D = self.head_dim

        Q = self.q(hsi_feat).reshape(B, L, H, D).transpose(1, 2)  # (B,H,L,D)
        K = self.k(pan_feat).reshape(B, L, H, D).transpose(1, 2)
        V = self.v(pan_feat).reshape(B, L, H, D).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / self.scale              # (B,H,L,L)
        attn = attn.softmax(dim=-1)
        out  = (attn @ V).transpose(1, 2).reshape(B, L, H*D)       # (B,L,H*D)
        out  = self.proj(out)
        return self.norm(hsi_feat + out)


class PanformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ff_mult=2):
        super().__init__()
        self.cross_attn = CrossAttention(dim, num_heads)
        self.self_norm  = nn.LayerNorm(dim)
        self.ff         = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, hsi_feat, pan_feat):
        x = self.cross_attn(hsi_feat, pan_feat)
        x = self.ff_norm(x + self.ff(self.self_norm(x)))
        return x


class Panformer(nn.Module):
    """
    Panformer: cross-attention transformer for HSI pansharpening.
    Architecture:
      1. Encode LR-HSI (bicubic to HR) and PAN into shared feature dim
      2. Stack N PanformerBlocks (HSI attends to PAN)
      3. Decode back to HR-HSI channels
    """
    def __init__(self, in_channels=128, scale=4, dim=64, depth=4, num_heads=4,
                 patch_size=8):
        super().__init__()
        self.scale      = scale
        self.patch_size = patch_size

        # Patch embeddings
        self.hsi_embed = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
        self.pan_embed = nn.Conv2d(1,           dim, patch_size, stride=patch_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PanformerBlock(dim, num_heads) for _ in range(depth)
        ])

        # Reconstruction
        self.recon = nn.Sequential(
            nn.Conv2d(dim, in_channels * (patch_size**2), 1),
            nn.PixelShuffle(patch_size),
        )

    def forward(self, lr_hsi, hr_pan):
        bicubic = F.interpolate(lr_hsi, scale_factor=self.scale,
                                mode='bicubic', align_corners=False)
        B, C, H, W = bicubic.shape

        # Tokenise
        hsi_tok = self.hsi_embed(bicubic)                      # (B, dim, H/p, W/p)
        pan_tok = self.pan_embed(hr_pan)                       # (B, dim, H/p, W/p)
        Hp, Wp  = hsi_tok.shape[2], hsi_tok.shape[3]
        L       = Hp * Wp

        hsi_seq = hsi_tok.flatten(2).transpose(1, 2)          # (B, L, dim)
        pan_seq = pan_tok.flatten(2).transpose(1, 2)          # (B, L, dim)

        # Transformer
        x = hsi_seq
        for blk in self.blocks:
            x = blk(x, pan_seq)

        # Reconstruct
        x   = x.transpose(1, 2).reshape(B, -1, Hp, Wp)       # (B, dim, Hp, Wp)
        res = self.recon(x)                                    # (B, C, H, W)

        # Ensure spatial size matches
        if res.shape[-2:] != (H, W):
            res = F.interpolate(res, size=(H, W), mode='bilinear', align_corners=False)

        return (bicubic + res * 0.1).clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
def create_baseline(name: str, in_channels=128, scale=4):
    name = name.lower()
    if name == 'bicubic':
        return BicubicBaseline(scale)
    elif name == 'pannet':
        return PanNet(in_channels, scale)
    elif name == 'pangan':
        return PanGAN(in_channels, scale)
    elif name == 'psgan':
        return PSGAN(in_channels, scale)
    elif name == 'panformer':
        return Panformer(in_channels, scale)
    else:
        raise ValueError(f"Unknown baseline '{name}'. "
                         "Choose: bicubic, pannet, pangan, psgan, panformer")


ALL_BASELINES = ['bicubic', 'pannet', 'pangan', 'psgan', 'panformer']
