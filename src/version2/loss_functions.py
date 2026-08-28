"""
loss_functions.py  —  Version 2  (self-contained)
All losses and metrics for hyperspectral pansharpening.

Loss used in training:
  L = 1.0·L1 + 0.10·SAM + 0.05·Edge + 0.01·SSIM + 0.05·SpectralGradient

Metrics for evaluation:
  PSNR  (higher is better, dB)
  SAM   (lower is better, degrees)
  ERGAS (lower is better)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Individual losses ─────────────────────────────────────────────────────────

class L1Loss(nn.Module):
    """Pixel-wise L1 (radiometric accuracy)."""
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class SAMLoss(nn.Module):
    """Spectral Angle Mapper loss (shape similarity)."""
    def forward(self, pred, target):
        B, C, H, W = pred.shape
        p = pred.reshape(B, C, -1)
        t = target.reshape(B, C, -1)
        dot  = (p * t).sum(1)
        np_  = p.norm(dim=1)
        nt   = t.norm(dim=1)
        cos  = (dot / (np_ * nt + 1e-8)).clamp(-1, 1)
        return torch.acos(cos).mean()


class EdgeLoss(nn.Module):
    """Sobel-gradient edge preservation loss."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        self.register_buffer('kx', kx.view(1,1,3,3))
        self.register_buffer('ky', ky.view(1,1,3,3))

    def _edge(self, x):
        x = x.mean(1, keepdim=True)
        gx = F.conv2d(x, self.kx.to(x), padding=1)
        gy = F.conv2d(x, self.ky.to(x), padding=1)
        return (gx**2 + gy**2 + 1e-8).sqrt()

    def forward(self, pred, target):
        return F.l1_loss(self._edge(pred), self._edge(target))


class SSIMLoss(nn.Module):
    """SSIM on spectral mean (avoids per-band OOM)."""
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.ws = window_size
        g  = torch.tensor([math.exp(-(x - window_size//2)**2 / (2*1.5**2))
                            for x in range(window_size)])
        g /= g.sum()
        w  = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('w', w)

    def _ssim1(self, x, y):
        C1, C2 = 0.01**2, 0.03**2
        w = self.w.to(x)
        p = self.ws // 2
        mu_x = F.conv2d(x, w, padding=p)
        mu_y = F.conv2d(y, w, padding=p)
        s_x  = F.conv2d(x*x, w, padding=p) - mu_x**2
        s_y  = F.conv2d(y*y, w, padding=p) - mu_y**2
        s_xy = F.conv2d(x*y, w, padding=p) - mu_x*mu_y
        num  = (2*mu_x*mu_y + C1) * (2*s_xy + C2)
        den  = (mu_x**2 + mu_y**2 + C1) * (s_x + s_y + C2)
        return (num/den).mean()

    def forward(self, pred, target):
        pg = pred.mean(1, keepdim=True)
        tg = target.mean(1, keepdim=True)
        return 1.0 - self._ssim1(pg, tg)


class SpectralGradientLoss(nn.Module):
    """
    Spectral gradient loss: penalises band-to-band continuity errors.
    L_spec = ||∇_c(pred) - ∇_c(target)||_1
    ∇_c(H) = H[:,1:] - H[:,:-1]
    """
    def forward(self, pred, target):
        dp = pred[:, 1:] - pred[:, :-1]
        dt = target[:, 1:] - target[:, :-1]
        return F.l1_loss(dp, dt)


class CompositeLoss(nn.Module):
    """
    Combined loss for V2 training.
    L = 1.0·L1 + 0.10·SAM + 0.05·Edge + 0.01·SSIM + 0.05·SpectralGradient
    """
    def __init__(self,
                 w_l1=1.0, w_sam=0.10, w_edge=0.05,
                 w_ssim=0.01, w_spec=0.05):
        super().__init__()
        self.l1   = L1Loss()
        self.sam  = SAMLoss()
        self.edge = EdgeLoss()
        self.ssim = SSIMLoss()
        self.spec = SpectralGradientLoss()
        self.w = dict(l1=w_l1, sam=w_sam, edge=w_edge, ssim=w_ssim, spec=w_spec)

    def forward(self, pred, target, return_components=False):
        l1   = self.l1(pred, target)
        sam  = self.sam(pred, target)
        edge = self.edge(pred, target)
        ssim = self.ssim(pred, target)
        spec = self.spec(pred, target)

        total = (self.w['l1']   * l1   +
                 self.w['sam']  * sam  +
                 self.w['edge'] * edge +
                 self.w['ssim'] * ssim +
                 self.w['spec'] * spec)

        if return_components:
            return total, {
                'total': total.item(), 'l1': l1.item(), 'sam': sam.item(),
                'edge': edge.item(), 'ssim': ssim.item(), 'spec': spec.item(),
            }
        return total


# ── Evaluation metrics ────────────────────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Band-averaged PSNR (dB). Higher is better."""
    mse  = ((pred - target)**2).mean(dim=(2,3))
    psnr = 20 * torch.log10(max_val / (mse + 1e-8).sqrt())
    return psnr.mean().item()


def compute_sam(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Average SAM in degrees. Lower is better."""
    B, C, H, W = pred.shape
    p   = pred.reshape(B, C, -1)
    t   = target.reshape(B, C, -1)
    cos = ((p*t).sum(1) / (p.norm(dim=1)*t.norm(dim=1) + 1e-8)).clamp(-1,1)
    return (torch.acos(cos) * 180 / math.pi).mean().item()


def compute_ergas(pred: torch.Tensor, target: torch.Tensor, scale: int = 4) -> float:
    """ERGAS metric. Lower is better."""
    mse    = ((pred - target)**2).mean(dim=(2,3))
    mean_t = target.mean(dim=(2,3))
    ergas  = 100 / scale * (mse / (mean_t**2 + 1e-8)).mean().sqrt()
    return ergas.item()


_ssim_metric_fn = None

def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean SSIM on spectral mean. Higher is better (range 0-1)."""
    global _ssim_metric_fn
    if _ssim_metric_fn is None:
        _ssim_metric_fn = SSIMLoss()
    with torch.no_grad():
        loss = _ssim_metric_fn(pred.cpu(), target.cpu())
    return round(1.0 - loss.item(), 6)
