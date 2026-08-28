"""
Loss Functions and Metrics for Hyperspectral Pansharpening
FINAL – Correct, Stable, and Research-Grade
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Basic Losses
# ============================================================================

class L1ReconstructionLoss(nn.Module):
    """Pixel-wise L1 loss (radiometric accuracy)"""
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class SpectralAngleMapper(nn.Module):
    """
    Spectral Angle Mapper (SAM) loss
    Measures spectral shape similarity
    """
    def forward(self, pred, target):
        B, C, H, W = pred.shape
        pred = pred.view(B, C, -1)
        target = target.view(B, C, -1)

        dot = torch.sum(pred * target, dim=1)
        norm_p = torch.norm(pred, dim=1)
        norm_t = torch.norm(target, dim=1)

        cos_sim = dot / (norm_p * norm_t + 1e-8)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

        sam = torch.acos(cos_sim)
        return sam.mean()


class EdgePreservationLoss(nn.Module):
    """
    Edge loss using Sobel gradients on mean spectrum
    (PAN-like spatial structure)
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32
        )

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _edge(self, x):
        x = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(x, self.sobel_x.to(x.device), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(x.device), padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def forward(self, pred, target):
        return F.l1_loss(self._edge(pred), self._edge(target))


# ============================================================================
# SSIM (Spatial-only, SAFE for HSI)
# ============================================================================

class SSIMLoss(nn.Module):
    """
    SSIM computed ONLY on mean spectrum (grayscale)
    This avoids per-band SSIM explosion and OOM
    """
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", self._create_window(window_size))

    def _gaussian(self, size, sigma=1.5):
        gauss = torch.tensor([
            math.exp(-(x - size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, size):
        g = self._gaussian(size).unsqueeze(1)
        w = g @ g.t()
        return w.unsqueeze(0).unsqueeze(0)

    def _ssim(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        w = self.window.to(x.device)

        mu_x = F.conv2d(x, w, padding=self.window_size // 2)
        mu_y = F.conv2d(y, w, padding=self.window_size // 2)

        sigma_x = F.conv2d(x * x, w, padding=self.window_size // 2) - mu_x**2
        sigma_y = F.conv2d(y * y, w, padding=self.window_size // 2) - mu_y**2
        sigma_xy = F.conv2d(x * y, w, padding=self.window_size // 2) - mu_x * mu_y

        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))

        return ssim.mean()

    def forward(self, pred, target):
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        return 1.0 - self._ssim(pred_gray, target_gray)


# ============================================================================
# Composite Loss (FINAL weights)
# ============================================================================

class CompositeLoss(nn.Module):
    """
    Final recommended composite loss for pansharpening

    L = 1.0*L1 + 0.05*SAM + 0.05*Edge + 0.05*SSIM
    """
    def __init__(
        self,
        lambda_l1=1.0,
        lambda_sam=0.05,
        lambda_edge=0.05,
        lambda_ssim=0.05
    ):
        super().__init__()
        self.l1 = L1ReconstructionLoss()
        self.sam = SpectralAngleMapper()
        self.edge = EdgePreservationLoss()
        self.ssim = SSIMLoss()

        self.w1 = lambda_l1
        self.w2 = lambda_sam
        self.w3 = lambda_edge
        self.w4 = lambda_ssim

    def forward(self, pred, target, return_components=False):
        l1 = self.l1(pred, target)
        sam = self.sam(pred, target)
        edge = self.edge(pred, target)
        ssim = self.ssim(pred, target)

        total = (
            self.w1 * l1 +
            self.w2 * sam +
            self.w3 * edge +
            self.w4 * ssim
        )

        if return_components:
            return total, {
                "total": total.item(),
                "l1": l1.item(),
                "sam": sam.item(),
                "edge": edge.item(),
                "ssim": ssim.item()
            }
        return total


# ============================================================================
# Evaluation Metrics (CORRECT)
# ============================================================================

def compute_psnr(pred, target, max_val=1.0):
    """Band-averaged PSNR (recommended for HSI)"""
    mse = torch.mean((pred - target) ** 2, dim=(2, 3))
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))
    return psnr.mean().item()


def compute_sam_metric(pred, target):
    """Average SAM in degrees"""
    B, C, H, W = pred.shape
    pred = pred.view(B, C, -1)
    target = target.view(B, C, -1)

    dot = torch.sum(pred * target, dim=1)
    norm_p = torch.norm(pred, dim=1)
    norm_t = torch.norm(target, dim=1)

    cos = dot / (norm_p * norm_t + 1e-8)
    cos = torch.clamp(cos, -1.0, 1.0)

    sam = torch.acos(cos) * 180.0 / math.pi
    return sam.mean().item()


def compute_ergas(pred, target, scale=2):
    """ERGAS metric (correct for [0,1] data)"""
    mse = torch.mean((pred - target) ** 2, dim=(2, 3))
    mean_t = torch.mean(target, dim=(2, 3))
    ergas = 100 / scale * torch.sqrt(torch.mean(mse / (mean_t**2 + 1e-8)))
    return ergas.item()


_ssim_metric_fn = None

def compute_ssim(pred, target):
    """
    Mean SSIM over the image, applied on spectral mean (grayscale).
    Higher is better (range 0-1).
    Reuses SSIMLoss internally: SSIM = 1 - SSIMLoss.
    """
    global _ssim_metric_fn
    if _ssim_metric_fn is None:
        _ssim_metric_fn = SSIMLoss()
    with torch.no_grad():
        loss = _ssim_metric_fn(pred.cpu(), target.cpu())
    return round(1.0 - loss.item(), 6)
