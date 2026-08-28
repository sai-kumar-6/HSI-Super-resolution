"""
generate_poster.py
──────────────────
Generates a professional A0 academic conference poster as a PDF for the
M.Tech thesis: "VMamba-Based Hyperspectral Pansharpening"

Run:
    cd "Mtech project"
    python generate_poster.py

Output:
    poster_vmamba_pansharpening.pdf   (A0 portrait, print-ready)
    poster_vmamba_pansharpening.png   (screen preview, 150 dpi)
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Resolve figure paths ────────────────────────────────────────────────────
FIG_MASTER_CHART   = os.path.join(_HERE, 'results', 'master_comparison_chart.png')
FIG_V4_COMPARISON  = os.path.join(_HERE, 'version4', 'comparison', 'plots', 'comparison_all.png')
FIG_V3_COMPARISON  = os.path.join(_HERE, 'version3', 'comparison', 'plots', 'comparison_all.png')
FIG_V5_CURVE       = os.path.join(_HERE, 'version5', 'comparison', 'plots', 'v5_training_curve.png')
FIG_V5_COMPARISON  = os.path.join(_HERE, 'version5', 'comparison', 'plots', 'comparison_summary.png')

# ── Colour palette ──────────────────────────────────────────────────────────
DARK_BLUE   = '#0D2E5A'   # header background
MID_BLUE    = '#1565C0'   # section title bars
LIGHT_BLUE  = '#E3F2FD'   # section body background
TEAL        = '#00796B'   # accent / highlights
AMBER       = '#FFF8E1'   # alternate section background
WHITE       = '#FFFFFF'
LIGHT_GREY  = '#F5F5F5'
DARK_TEXT   = '#1A1A2E'
RED_ACCENT  = '#C62828'

# ── Data ────────────────────────────────────────────────────────────────────
VERSIONS = ['V1\n(Baseline)', 'V2\n(Improved)', 'V3\n(Spectral)', 'V4\n(Best)', 'V5\n(Parallel)']
PSNR     = [44.65, 44.99, 47.35, 49.37, 43.34]
SAM      = [6.74,  6.77,  6.61,  6.32,  6.74]
ERGAS    = [71.72, 113.05, 61.20, 4.49,  5.61]
SSIM     = [None,  None,   None,  0.9997, 0.9622]
PARAMS   = [1.84,  0.67,   1.52,  1.53,  1.77]
BAR_COLORS = ['#90CAF9','#4FC3F7','#4DB6AC','#2E7D32','#7B1FA2']

VERSION_INNOVATIONS = [
    ('V1', 'CrossAttention (O(N²)) + PixelShuffle upsampling', 'Baseline reference'),
    ('V2', 'CrossMamba (O(N)) + Learnable edge injection', '−62% params, no checkerboard'),
    ('V3', 'SpectralMambaBlock + 4-stage U-Net backbone',   '+2.36 dB PSNR over V2'),
    ('V4', 'Scaled SDIM (α) + StrongReconHead (β) + SpectralGradLoss', 'Best: PSNR 49.37, ERGAS 4.49'),
    ('V5', 'Hillis-Steele O(log L) true parallel scan',     '6.4× fewer scan iterations'),
]


# ════════════════════════════════════════════════════════════════════════════
# Helper drawing functions
# ════════════════════════════════════════════════════════════════════════════

def section_box(ax, title, facecolor=LIGHT_BLUE, title_color=WHITE,
                title_bg=MID_BLUE, fontsize_title=16, pad=0.02):
    """Draw a titled section box on a given axes."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    # Outer box
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                boxstyle='round,pad=0.01',
                                linewidth=1.5, edgecolor=MID_BLUE,
                                facecolor=facecolor, zorder=1,
                                transform=ax.transAxes))
    # Title bar
    ax.add_patch(FancyBboxPatch((0, 0.92), 1, 0.08,
                                boxstyle='round,pad=0.005',
                                linewidth=0, edgecolor='none',
                                facecolor=title_bg, zorder=2,
                                transform=ax.transAxes))
    ax.text(0.5, 0.96, title, ha='center', va='center',
            fontsize=fontsize_title, fontweight='bold',
            color=title_color, zorder=3, transform=ax.transAxes)


def bullet_text(ax, lines, x=0.04, y_start=0.88, dy=0.085,
                fontsize=11.5, color=DARK_TEXT, bold_first=False):
    """Write a list of bullet-point lines on the given axes."""
    for i, line in enumerate(lines):
        fw = 'bold' if (bold_first and i == 0) else 'normal'
        ax.text(x, y_start - i * dy, line,
                ha='left', va='top', fontsize=fontsize,
                color=color, fontweight=fw,
                transform=ax.transAxes, wrap=True,
                bbox=dict(boxstyle='square,pad=0', facecolor='none', edgecolor='none'))


# ════════════════════════════════════════════════════════════════════════════
# Build poster
# ════════════════════════════════════════════════════════════════════════════

def build_poster():
    # A0 in inches at 100 dpi  (841 x 1189 mm → 33.1 x 46.8 in)
    fig = plt.figure(figsize=(33.1, 46.8), dpi=100)
    fig.patch.set_facecolor(WHITE)

    # ── Master grid ──────────────────────────────────────────────────────────
    #  Row 0: header         (height ratio 6)
    #  Row 1: intro + problem (8)
    #  Row 2: architecture evolution (10)
    #  Row 3: innovations + results table (10)
    #  Row 4: figures row (10)
    #  Row 5: conclusion + references (7)
    outer = gridspec.GridSpec(6, 1,
                              figure=fig,
                              height_ratios=[6, 8, 10, 10, 10, 7],
                              hspace=0.025,
                              left=0.02, right=0.98,
                              top=0.98, bottom=0.01)

    # ────────────────────────────────────────────────────────────────────────
    # ROW 0 — HEADER
    # ────────────────────────────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(outer[0])
    ax_hdr.set_xlim(0, 1)
    ax_hdr.set_ylim(0, 1)
    ax_hdr.axis('off')
    ax_hdr.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                    boxstyle='round,pad=0.005',
                                    linewidth=0, facecolor=DARK_BLUE,
                                    transform=ax_hdr.transAxes))
    # Conference tag
    ax_hdr.text(0.5, 0.97,
                'M.TECH THESIS — HYPERSPECTRAL IMAGE PROCESSING',
                ha='center', va='top', fontsize=14, color='#90CAF9',
                fontweight='bold', transform=ax_hdr.transAxes)
    # Main title
    ax_hdr.text(0.5, 0.72,
                'VMamba: Progressive State-Space Model Architecture\n'
                'for Hyperspectral Image Pansharpening',
                ha='center', va='center', fontsize=34,
                color=WHITE, fontweight='bold', transform=ax_hdr.transAxes,
                linespacing=1.35)
    # Subtitle
    ax_hdr.text(0.5, 0.36,
                'A Five-Version Evolutionary Study: Spatial Efficiency · '
                'Spectral Fidelity · Parallel Scan Algorithms',
                ha='center', va='center', fontsize=16,
                color='#B3E5FC', transform=ax_hdr.transAxes)
    # Authors / Institution  ← fill in your details
    ax_hdr.text(0.5, 0.14,
                '[Author Name]   ·   Supervisor: [Supervisor Name]   ·   '
                'Department of [Dept], [University Name]   ·   [email@institution.ac.in]',
                ha='center', va='center', fontsize=13,
                color='#E0E0E0', transform=ax_hdr.transAxes)

    # ────────────────────────────────────────────────────────────────────────
    # ROW 1 — INTRODUCTION  |  PROBLEM FORMULATION
    # ────────────────────────────────────────────────────────────────────────
    row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                            wspace=0.025)

    # ── Introduction ────────────────────────────────────────────────────────
    ax_intro = fig.add_subplot(row1[0])
    section_box(ax_intro, 'Introduction', facecolor=LIGHT_BLUE, title_bg=MID_BLUE)
    intro_lines = [
        '• Hyperspectral pansharpening fuses two complementary sensors:',
        '     LR-HSI : low spatial res, 128 spectral bands (8×8 pixels)',
        '     HR-PAN  : high spatial res, single panchromatic band (32×32)',
        '     Goal    : produce HR-HSI (32×32, 128 bands) at 4× scale',
        '',
        '• Challenge: Spatial detail injection must NOT distort spectral\n'
        '   signatures — measured by SAM (lower = better).',
        '',
        '• Mamba / VMamba State-Space Models offer:',
        '     ◦  O(N) sequence modeling vs O(N²) cross-attention',
        '     ◦  Selective state propagation (data-gated SSM)',
        '     ◦  Efficient 2D spatial scanning (row + column scans)',
        '',
        '• This work evolves VMamba through 5 versions, each targeting\n'
        '   a specific bottleneck in quality or efficiency.',
    ]
    bullet_text(ax_intro, intro_lines, dy=0.069, fontsize=12)

    # ── Problem Formulation ─────────────────────────────────────────────────
    ax_prob = fig.add_subplot(row1[1])
    section_box(ax_prob, 'Problem Formulation & Dataset', facecolor=AMBER, title_bg=TEAL)

    # Draw a small diagram: LR-HSI + HR-PAN → Model → HR-HSI
    diag_ax = ax_prob
    arrow_kw = dict(arrowstyle='->', color=MID_BLUE,
                    lw=2.5, mutation_scale=20)

    def rbox(ax, x, y, w, h, txt, fc='#BBDEFB', tc=DARK_TEXT, fs=11.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01',
                                    linewidth=1.5, edgecolor=MID_BLUE,
                                    facecolor=fc, zorder=4, transform=ax.transAxes))
        ax.text(x + w/2, y + h/2, txt, ha='center', va='center',
                fontsize=fs, color=tc, fontweight='bold',
                zorder=5, transform=ax.transAxes)

    # Boxes
    rbox(diag_ax, 0.05, 0.62, 0.27, 0.12,
         'LR-HSI\n(B, 128, H, W)', fc='#C8E6C9')
    rbox(diag_ax, 0.05, 0.44, 0.27, 0.12,
         'HR-PAN\n(B, 1, 4H, 4W)', fc='#FFF9C4')
    rbox(diag_ax, 0.39, 0.50, 0.24, 0.14,
         'VMamba\nModel', fc='#CE93D8', tc=WHITE)
    rbox(diag_ax, 0.72, 0.50, 0.24, 0.14,
         'HR-HSI\n(B, 128, 4H, 4W)', fc='#FFCCBC')

    # Arrows
    diag_ax.annotate('', xy=(0.39, 0.60), xytext=(0.32, 0.68),
                     xycoords='axes fraction', textcoords='axes fraction',
                     arrowprops=arrow_kw)
    diag_ax.annotate('', xy=(0.39, 0.56), xytext=(0.32, 0.50),
                     xycoords='axes fraction', textcoords='axes fraction',
                     arrowprops=arrow_kw)
    diag_ax.annotate('', xy=(0.72, 0.57), xytext=(0.63, 0.57),
                     xycoords='axes fraction', textcoords='axes fraction',
                     arrowprops=arrow_kw)
    diag_ax.text(0.50, 0.42, '4× upsampling', ha='center', va='top',
                 fontsize=10.5, color='#555', transform=diag_ax.transAxes)

    prob_lines = [
        '• Dataset: Chikusei (Japanese airborne VNIR)',
        '     2517 × 2335 pixels · 128 spectral bands',
        '     Wavelength: ~363–1018 nm',
        '',
        '• Patch training: 32×32 HR-HSI patches,',
        '   downsampled 4× to 8×8 LR-HSI inputs',
        '',
        '• Evaluation Metrics:',
        '     PSNR (dB) ↑   — reconstruction fidelity',
        '     SAM   (°)  ↓   — spectral angle per pixel',
        '     ERGAS       ↓   — band-wise energy error',
        '     SSIM        ↑   — structural similarity',
    ]
    bullet_text(ax_prob, prob_lines, y_start=0.38, dy=0.062, fontsize=12)

    # ────────────────────────────────────────────────────────────────────────
    # ROW 2 — ARCHITECTURE EVOLUTION (full width)
    # ────────────────────────────────────────────────────────────────────────
    ax_arch = fig.add_subplot(outer[2])
    section_box(ax_arch, 'Architecture Evolution: V1 → V5', title_bg=DARK_BLUE, fontsize_title=18)

    # --- Evolution timeline (horizontal boxes + arrows) ---
    evo_y = 0.74
    evo_h = 0.14
    evo_x = [0.03, 0.21, 0.39, 0.57, 0.75]
    evo_w = 0.16
    evo_labels = ['V1\nBaseline\n1.84M', 'V2\nImproved\n0.67M',
                  'V3\nSpectral\n1.52M', 'V4\nBest\n1.53M', 'V5\nParallel\n1.77M']
    evo_colors = ['#90CAF9','#4FC3F7','#4DB6AC','#2E7D32','#7B1FA2']
    evo_psnr = [44.65, 44.99, 47.35, 49.37, 43.34]

    for i, (x, lbl, fc, psnr) in enumerate(zip(evo_x, evo_labels, evo_colors, evo_psnr)):
        ax_arch.add_patch(FancyBboxPatch((x, evo_y), evo_w, evo_h,
                                         boxstyle='round,pad=0.01',
                                         linewidth=2, edgecolor='white',
                                         facecolor=fc, zorder=4,
                                         transform=ax_arch.transAxes))
        ax_arch.text(x + evo_w/2, evo_y + evo_h/2 + 0.01, lbl,
                     ha='center', va='center', fontsize=13,
                     color=WHITE, fontweight='bold',
                     zorder=5, transform=ax_arch.transAxes)
        ax_arch.text(x + evo_w/2, evo_y - 0.025, f'PSNR {psnr:.2f} dB',
                     ha='center', va='top', fontsize=11,
                     color=DARK_TEXT, zorder=5, transform=ax_arch.transAxes)
        # Arrow between boxes
        if i < 4:
            ax_arch.annotate('', xy=(evo_x[i+1], evo_y + evo_h/2),
                             xytext=(x + evo_w, evo_y + evo_h/2),
                             xycoords='axes fraction', textcoords='axes fraction',
                             arrowprops=dict(arrowstyle='->', color='#555',
                                             lw=2, mutation_scale=18))

    # --- Innovation table ---
    table_y = 0.615
    table_dy = 0.097
    col_x = [0.02, 0.08, 0.42, 0.77]
    col_headers = ['Ver.', 'Key Innovation', 'Problem Solved / Outcome']

    # Header row
    for cx, ch in zip(col_x[1:], col_headers):
        ax_arch.text(cx, table_y, ch, ha='left', va='top', fontsize=13,
                     fontweight='bold', color=WHITE,
                     transform=ax_arch.transAxes)
    ax_arch.plot([0.02, 0.98], [table_y - 0.015, table_y - 0.015],
                 color='#90CAF9', linewidth=0.8, transform=ax_arch.transAxes, zorder=4)

    row_bg_colors = ['#F3E5F5','#E8F5E9','#E1F5FE','#E8F5E9','#FFF3E0']
    for i, (ver, innov, outcome) in enumerate(VERSION_INNOVATIONS):
        y = table_y - 0.035 - i * table_dy
        ax_arch.add_patch(FancyBboxPatch((0.02, y - 0.015), 0.96, table_dy * 0.9,
                                          boxstyle='round,pad=0.005',
                                          linewidth=0, facecolor=row_bg_colors[i],
                                          zorder=3, transform=ax_arch.transAxes))
        ax_arch.text(col_x[1], y + 0.015, ver, ha='left', va='center',
                     fontsize=13, fontweight='bold', color=evo_colors[i],
                     zorder=4, transform=ax_arch.transAxes)
        ax_arch.text(col_x[2], y + 0.015, innov, ha='left', va='center',
                     fontsize=11.5, color=DARK_TEXT, zorder=4,
                     transform=ax_arch.transAxes)
        ax_arch.text(col_x[3], y + 0.015, outcome, ha='left', va='center',
                     fontsize=11.5, color=DARK_TEXT, zorder=4,
                     transform=ax_arch.transAxes)

    # ────────────────────────────────────────────────────────────────────────
    # ROW 3 — KEY INNOVATIONS  |  RESULTS TABLE
    # ────────────────────────────────────────────────────────────────────────
    row3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3],
                                            wspace=0.025, width_ratios=[1, 1.05])

    # ── Key Innovations ─────────────────────────────────────────────────────
    ax_kv = fig.add_subplot(row3[0])
    section_box(ax_kv, 'Key Technical Innovations', facecolor=LIGHT_BLUE,
                title_bg=MID_BLUE)

    # Sub-box A: Scaled SDIM
    ax_kv.add_patch(FancyBboxPatch((0.03, 0.52), 0.94, 0.38,
                                    boxstyle='round,pad=0.01',
                                    linewidth=1.5, edgecolor=TEAL,
                                    facecolor='#E0F2F1', zorder=3,
                                    transform=ax_kv.transAxes))
    ax_kv.text(0.50, 0.88, 'V4 — Scaled Spatial Detail Injection (SDIM)',
               ha='center', va='center', fontsize=13, fontweight='bold',
               color=TEAL, zorder=4, transform=ax_kv.transAxes)
    sdim_lines = [
        '  F_fused = F_hsi + α × (F_pan − F_hsi)',
        '  α: learnable scalar, initialised at 0.1',
        '  • Controls PAN injection strength adaptively',
        '  • Prevents spectral distortion from over-injection',
        '  • β scalar in Recon Head anchors bicubic baseline',
    ]
    for j, l in enumerate(sdim_lines):
        ax_kv.text(0.05, 0.83 - j * 0.062, l, ha='left', va='top',
                   fontsize=12, color=DARK_TEXT,
                   fontfamily='monospace' if '=' in l else 'sans-serif',
                   transform=ax_kv.transAxes, zorder=4)

    # Sub-box B: Hillis-Steele
    ax_kv.add_patch(FancyBboxPatch((0.03, 0.06), 0.94, 0.43,
                                    boxstyle='round,pad=0.01',
                                    linewidth=1.5, edgecolor='#7B1FA2',
                                    facecolor='#F3E5F5', zorder=3,
                                    transform=ax_kv.transAxes))
    ax_kv.text(0.50, 0.46, 'V5 — Hillis-Steele True Parallel Scan',
               ha='center', va='center', fontsize=13, fontweight='bold',
               color='#7B1FA2', zorder=4, transform=ax_kv.transAxes)
    hs_lines = [
        '  Classic SSM scan:  O(L) depth — 32 loops for L=32',
        '  Hillis-Steele:      O(log L) depth — 5 loops for L=32',
        '',
        '  Each pass d with stride = 2^d:',
        '    a_new[t] = a[t] · a[t − stride]',
        '    b_new[t] = a[t] · b[t − stride] + b[t]',
        '',
        '  After log₂(L) passes → all h_t computed in parallel',
        '  Result: 6.4× fewer Python iterations vs sequential',
    ]
    for j, l in enumerate(hs_lines):
        ax_kv.text(0.05, 0.41 - j * 0.052, l, ha='left', va='top',
                   fontsize=11.5, color=DARK_TEXT,
                   fontfamily='monospace' if ('=' in l or '·' in l) else 'sans-serif',
                   transform=ax_kv.transAxes, zorder=4)

    # ── Results Table ───────────────────────────────────────────────────────
    ax_res = fig.add_subplot(row3[1])
    section_box(ax_res, 'Quantitative Results — Chikusei Dataset (Scale 4×)',
                facecolor=AMBER, title_bg=TEAL)

    # Draw table manually
    headers = ['Version', 'PSNR (dB) ↑', 'SAM (°) ↓', 'ERGAS ↓', 'SSIM ↑', 'Params']
    col_xs = [0.03, 0.22, 0.37, 0.52, 0.65, 0.78]
    th_y = 0.87
    ax_res.add_patch(FancyBboxPatch((0.02, th_y - 0.015), 0.96, 0.055,
                                     boxstyle='round,pad=0.005',
                                     linewidth=0, facecolor=MID_BLUE,
                                     zorder=3, transform=ax_res.transAxes))
    for cx, h in zip(col_xs, headers):
        ax_res.text(cx, th_y + 0.012, h, ha='left', va='center',
                    fontsize=12, fontweight='bold', color=WHITE,
                    zorder=4, transform=ax_res.transAxes)

    table_data = [
        ('V1 (Baseline)', '44.65', '6.74', '71.72', '—',     '1.84M'),
        ('V2 (Improved)', '44.99', '6.77', '113.05','—',     '0.67M'),
        ('V3 (Spectral)', '47.35', '6.61', '61.20', '—',     '1.52M'),
        ('V4 ★ Best',     '49.37', '6.32', '4.49',  '0.9997','1.53M'),
        ('V5 (Parallel)', '43.34', '6.74', '5.61',  '0.9622','1.77M'),
    ]
    row_bg = ['#FFFFFF','#F5F5F5','#FFFFFF','#E8F5E9','#F3E5F5']

    for i, (row, rbg) in enumerate(zip(table_data, row_bg)):
        ry = th_y - 0.065 - i * 0.10
        ax_res.add_patch(FancyBboxPatch((0.02, ry - 0.025), 0.96, 0.08,
                                         boxstyle='round,pad=0.005',
                                         linewidth=0.5, edgecolor='#CCCCCC',
                                         facecolor=rbg, zorder=3,
                                         transform=ax_res.transAxes))
        fw = 'bold' if '★' in row[0] else 'normal'
        fc = '#2E7D32' if '★' in row[0] else DARK_TEXT
        for cx, cell in zip(col_xs, row):
            ax_res.text(cx, ry + 0.015, cell, ha='left', va='center',
                        fontsize=12.5, fontweight=fw, color=fc,
                        zorder=4, transform=ax_res.transAxes)

    # Best row label
    ax_res.text(0.97, th_y - 0.065 - 3 * 0.10 + 0.015,
                '← BEST', ha='right', va='center', fontsize=11,
                color='#2E7D32', fontweight='bold', transform=ax_res.transAxes)

    # Summary note
    ax_res.text(0.50, 0.055,
                'V4 achieves 13× lower ERGAS than V1 (71.72 → 4.49) with near-perfect SSIM 0.9997\n'
                'V5 maintains low ERGAS (5.61) while enabling O(log L) parallel training',
                ha='center', va='center', fontsize=11.5,
                color='#37474F', style='italic',
                transform=ax_res.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', edgecolor=TEAL, lw=1.5))

    # ────────────────────────────────────────────────────────────────────────
    # ROW 4 — FIGURES
    # ────────────────────────────────────────────────────────────────────────
    row4 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[4],
                                            wspace=0.03)

    fig_data = [
        (FIG_V4_COMPARISON,
         'Fig. 1 — V4 Model Comparison\n(PSNR · SAM · ERGAS · Parameters)',
         'V1 vs V2 vs V3 vs V4 bar chart'),
        (FIG_V5_COMPARISON,
         'Fig. 2 — V5 Model Comparison\n(All 5 Variants)',
         'V5 parallel scan vs all variants'),
        (FIG_V5_CURVE,
         'Fig. 3 — V5 Training Curve\n(PSNR & SAM vs Epoch)',
         'Training convergence with H-S scan'),
    ]

    for i, (fpath, caption, placeholder_txt) in enumerate(fig_data):
        ax_fig = fig.add_subplot(row4[i])
        ax_fig.set_xlim(0, 1); ax_fig.set_ylim(0, 1); ax_fig.axis('off')
        ax_fig.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                        boxstyle='round,pad=0.01',
                                        linewidth=1.5, edgecolor=MID_BLUE,
                                        facecolor=LIGHT_BLUE, zorder=1,
                                        transform=ax_fig.transAxes))

        if os.path.isfile(fpath):
            try:
                img = plt.imread(fpath)
                inner = ax_fig.inset_axes([0.02, 0.12, 0.96, 0.80])
                inner.imshow(img, aspect='auto')
                inner.axis('off')
            except Exception:
                ax_fig.text(0.5, 0.5, f'[Figure]\n{placeholder_txt}',
                            ha='center', va='center', fontsize=13,
                            color='#888', transform=ax_fig.transAxes)
        else:
            ax_fig.text(0.5, 0.50, f'[Figure]\n{placeholder_txt}',
                        ha='center', va='center', fontsize=13,
                        color='#888', transform=ax_fig.transAxes)

        ax_fig.text(0.5, 0.04, caption,
                    ha='center', va='center', fontsize=11,
                    color=DARK_TEXT, fontweight='bold',
                    transform=ax_fig.transAxes, style='italic')

    # ────────────────────────────────────────────────────────────────────────
    # ROW 5 — CONCLUSION  |  REFERENCES
    # ────────────────────────────────────────────────────────────────────────
    row5 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[5],
                                            wspace=0.025, width_ratios=[1.2, 0.8])

    # ── Conclusion ───────────────────────────────────────────────────────────
    ax_conc = fig.add_subplot(row5[0])
    section_box(ax_conc, 'Conclusions', facecolor='#E8F5E9', title_bg='#2E7D32')
    conc_lines = [
        '✓  Progressive architecture evolution provides a clean ablation study across 5 versions',
        '✓  V4 achieves the best spectral fidelity with PSNR 49.37 dB and ERGAS 4.49',
        '     — a 13× improvement in ERGAS over the V1 baseline (71.72 → 4.49)',
        '✓  Learnable α (SDIM) and β (Recon Head) scalars are critical for spectral control',
        '✓  SpectralGradientLoss penalises band-to-band gradient mismatches, reducing SAM',
        '✓  V5 Hillis-Steele scan achieves O(log L) depth: 6.4× fewer loop iterations (L=32)',
        '✓  Pre-norm architecture in V5 improves gradient flow and training stability',
        '✓  All versions validated on the Chikusei VNIR dataset at 4× upsampling scale',
    ]
    bullet_text(ax_conc, conc_lines, dy=0.097, fontsize=12.5)

    # ── References ───────────────────────────────────────────────────────────
    ax_ref = fig.add_subplot(row5[1])
    section_box(ax_ref, 'References', facecolor=LIGHT_GREY, title_bg='#546E7A')
    refs = [
        '[1] Gu et al., "Mamba: Linear-Time Sequence Modeling with Selective\n'
        '     State Spaces," arXiv:2312.00752, 2023.',
        '[2] Liu et al., "VMamba: Visual State Space Model," arXiv:2401.13260, 2024.',
        '[3] Yokoya et al., "Airborne Hyperspectral Data over Chikusei," Space\n'
        '     Appl. Lab., Univ. Tokyo, Japan, SAL-2016-05-27, 2016.',
        '[4] Vivone et al., "A Critical Comparison Among Pansharpening\n'
        '     Algorithms," IEEE TGRS, 2015.',
        '[5] Hillis & Steele, "Data Parallel Algorithms," Comm. ACM, 1986.',
    ]
    for j, r in enumerate(refs):
        ax_ref.text(0.04, 0.84 - j * 0.155, r, ha='left', va='top',
                    fontsize=10.5, color=DARK_TEXT,
                    transform=ax_ref.transAxes)

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Building poster …')
    fig = build_poster()

    # PDF (print quality)
    pdf_path = os.path.join(_HERE, 'poster_vmamba_pansharpening.pdf')
    fig.savefig(pdf_path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    print(f'  Saved PDF : {pdf_path}')

    # PNG preview
    png_path = os.path.join(_HERE, 'poster_vmamba_pansharpening.png')
    fig.savefig(png_path, dpi=100, bbox_inches='tight', facecolor=WHITE)
    print(f'  Saved PNG : {png_path}')

    plt.close(fig)
    print('Done.')
