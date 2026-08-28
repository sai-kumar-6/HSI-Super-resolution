"""
generate_architecture_diagram.py
Generates a clean block-flow architecture diagram for V5 VMamba Pansharpening.
Output: architecture_diagram.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis('off')
fig.patch.set_facecolor('#F8F9FA')

# ── Color palette ─────────────────────────────────────────────────────────────
C_INPUT_HSI  = '#4CAF50'   # green
C_INPUT_PAN  = '#FFC107'   # amber
C_ENCODER    = '#1565C0'   # dark blue
C_SDIM       = '#6A1B9A'   # purple
C_BACKBONE   = '#6A1B9A'   # purple
C_RECON      = '#6A1B9A'   # purple
C_OUTPUT     = '#E64A19'   # deep orange
C_TEXT_LIGHT = 'white'
C_TEXT_DARK  = '#212121'
C_ARROW      = '#37474F'

def box(ax, x, y, w, h, color, label, sublabel=None, text_color='white', fontsize=11):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.12",
                          linewidth=1.8,
                          edgecolor='white',
                          facecolor=color,
                          zorder=3)
    ax.add_patch(rect)
    cy = y + h / 2 + (0.18 if sublabel else 0)
    ax.text(x + w / 2, cy, label,
            ha='center', va='center',
            fontsize=fontsize, fontweight='bold',
            color=text_color, zorder=4)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.30, sublabel,
                ha='center', va='center',
                fontsize=8.5, color=text_color, zorder=4, style='italic')

def arrow(ax, x1, y1, x2, y2, color='#37474F'):
    ax.annotate('',
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=2.2, mutation_scale=18),
                zorder=2)

def merge_arrow(ax, x1a, y1a, x1b, y1b, xm, ym, color='#37474F'):
    """Two lines converge to a midpoint then one arrow out."""
    ax.annotate('', xy=(xm, ym), xytext=(x1a, y1a),
                arrowprops=dict(arrowstyle='-', color=color, lw=2.2), zorder=2)
    ax.annotate('', xy=(xm, ym), xytext=(x1b, y1b),
                arrowprops=dict(arrowstyle='-', color=color, lw=2.2), zorder=2)

# ── Layout constants ──────────────────────────────────────────────────────────
BW, BH = 1.9, 0.85          # standard box width / height
TOP_Y   = 4.5                # top row  (LR-HSI, HSI Encoder)
BOT_Y   = 2.5                # bottom row (HR-PAN, PAN Encoder)
MID_Y   = (TOP_Y + BOT_Y) / 2 + BH / 2 - BH / 2   # vertical center for merged path

# X positions (left edge of each box)
X_IN1   = 0.4
X_ENC1  = 2.7
X_IN2   = 0.4
X_ENC2  = 2.7
X_SDIM  = 5.6
X_BACK  = 8.0
X_RECON = 10.7
X_OUT   = 13.4

ROW_MID = (TOP_Y + BOT_Y) / 2 + BH / 4          # ~3.6

# ── Input boxes ───────────────────────────────────────────────────────────────
box(ax, X_IN1, TOP_Y,  BW, BH, C_INPUT_HSI,  'LR-HSI',  '(Low-Res HSI)',   fontsize=11)
box(ax, X_IN2, BOT_Y,  BW, BH, C_INPUT_PAN,  'HR-PAN',  '(Hi-Res PAN)',    fontsize=11)

# ── Encoder boxes ─────────────────────────────────────────────────────────────
box(ax, X_ENC1, TOP_Y, BW, BH, C_ENCODER, 'HSI Encoder', fontsize=10)
box(ax, X_ENC2, BOT_Y, BW, BH, C_ENCODER, 'PAN Encoder', fontsize=10)

# ── Arrows: inputs → encoders ─────────────────────────────────────────────────
arrow(ax, X_IN1+BW,       TOP_Y+BH/2, X_ENC1,       TOP_Y+BH/2)
arrow(ax, X_IN2+BW,       BOT_Y+BH/2, X_ENC2,       BOT_Y+BH/2)

# ── Merge point ───────────────────────────────────────────────────────────────
MX = X_SDIM - 0.35          # horizontal merge node x
MY = ROW_MID                 # vertical merge node y

merge_arrow(ax,
            X_ENC1+BW, TOP_Y+BH/2,
            X_ENC2+BW, BOT_Y+BH/2,
            MX, MY)

# ── ScaledSDIM ────────────────────────────────────────────────────────────────
box(ax, X_SDIM, MY-BH/2, BW, BH, C_SDIM,
    'Scaled SDIM', '(α learnable)', fontsize=10)
arrow(ax, MX, MY, X_SDIM, MY)

# ── V5 U-Net Backbone ─────────────────────────────────────────────────────────
box(ax, X_BACK, MY-BH/2, 2.3, BH, C_BACKBONE,
    'V5 U-Net Backbone', '(4-Stage TrueParallelSSM)', fontsize=9)
arrow(ax, X_SDIM+BW, MY, X_BACK, MY)

# ── StrongReconHead ───────────────────────────────────────────────────────────
box(ax, X_RECON, MY-BH/2, BW+0.15, BH, C_RECON,
    'Recon Head', '(β learnable)', fontsize=10)
arrow(ax, X_BACK+2.3, MY, X_RECON, MY)

# ── HR-HSI output ─────────────────────────────────────────────────────────────
box(ax, X_OUT, MY-BH/2, BW, BH, C_OUTPUT,
    'HR-HSI', '(4× upsampled)', fontsize=11)
arrow(ax, X_RECON+BW+0.15, MY, X_OUT, MY)

# ── Scale label ───────────────────────────────────────────────────────────────
ax.text(8.0, MY - BH, '4× Spatial Upsampling  |  128 spectral bands preserved',
        ha='center', va='center', fontsize=9.5, color='#455A64', style='italic')

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(8.0, 6.55,
        'V5 VMamba Architecture for Hyperspectral Image Pansharpening',
        ha='center', va='center', fontsize=14, fontweight='bold', color='#1A237E')

# ── Legend (loss components) ──────────────────────────────────────────────────
legend_x, legend_y = 0.4, 1.6
ax.text(legend_x, legend_y + 0.3, 'Loss = L₁ + λ₁·SAM + λ₂·Edge + λ₃·SSIM + λ₄·SpectralGrad',
        fontsize=9, color='#37474F',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8EAF6',
                  edgecolor='#3949AB', linewidth=1.2))

# ── SSM note ──────────────────────────────────────────────────────────────────
ax.text(15.6, 1.6,
        'Scan depth:\nO(log L)\n(Hillis-Steele)',
        fontsize=8.5, color='#37474F', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#F3E5F5',
                  edgecolor='#7B1FA2', linewidth=1.2))

plt.tight_layout(pad=0.3)
out = 'architecture_diagram.png'
fig.savefig(out, dpi=160, bbox_inches='tight', facecolor='#F8F9FA')
plt.close(fig)
print(f'Saved: {out}')
