# VMamba Pansharpening — Architecture Diagram
### Tensor shapes at every step | d_model=32, scale=4, C=128

> **Example inputs used throughout:**
> - `LR-HSI : (B=1, C=128, H=8,  W=8)`
> - `HR-PAN : (B=1,   1,  H=32, W=32)`

---

## V2 — Improved VMamba Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INPUTS                                             │
│                                                                             │
│   LR-HSI  (1, 128,  8,  8)          HR-PAN  (1, 1, 32, 32)                │
│      │                                    │                                 │
│      ▼                                    ▼                                 │
│ ┌────────────────────────┐     ┌──────────────────────────┐                │
│ │   ImprovedHSIEncoder   │     │   ImprovedPANEncoder     │                │
│ └────────────────────────┘     └──────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Block 1 — Improved HSI Encoder

```
LR-HSI
(1, 128, 8, 8)
     │
     │  unsqueeze(1)
     ▼
(1, 1, 128, 8, 8)           ← add channel dim for 3D conv
     │
     │  Conv3d(1→8, kernel=(1,3,3), padding=(0,1,1))
     │  GroupNorm(8 groups) + ReLU
     ▼
(1, 8, 128, 8, 8)           ← 8 spectral feature maps
     │
     │  reshape: merge bands×features
     ▼
(1, 1024, 8, 8)             ← 8×128 = 1024 channels
     │
     │  Conv1×1  (spectral aggregation: 1024→32)
     │  GroupNorm + ReLU
     ▼
(1, 32, 8, 8)               ← compressed to d_model=32
     │
     │  ResidualUpsample2×   [Main: Bilinear→Conv3×3→GN→ReLU]
     │                        [Res:  Conv3×3→Bilinear       ]
     │                        [Out:  Main + Res             ]
     ▼
(1, 32, 16, 16)             ← spatial ×2
     │
     │  ResidualUpsample2×   (same structure)
     ▼
(1, 32, 32, 32)             ← F_hsi  ✓ matches PAN spatial size
```

---

### Block 2 — Improved PAN Encoder

```
HR-PAN
(1, 1, 32, 32)
     │
     │  Conv2d(1→32, 3×3) + GroupNorm(4) + ReLU
     ▼
(1, 32, 32, 32)
     │
     │  Conv2d(32→64, 3×3) + GroupNorm(8) + ReLU
     ▼
(1, 64, 32, 32)
     │
     │  Conv2d(64→32, 3×3) + GroupNorm(8) + ReLU
     ▼
(1, 32, 32, 32)             ← F_pan  ✓ same shape as F_hsi
```

---

### Block 3 — Cross-Mamba Fusion

```
F_hsi (1, 32, 32, 32)          F_pan (1, 32, 32, 32)
       │                               │
       └───────────┬───────────────────┘
                   ▼
     ┌─────────────────────────────────┐
     │     LearnableHFInjection        │
     │                                 │
     │  Edge = Conv3×3(F_pan)          │   (1, 32, 32, 32)
     │  Gate = sigmoid(Conv1×1(F_pan)) │   (1, 32, 32, 32)
     │  F_hsi_enh = F_hsi + Gate×Edge │   (1, 32, 32, 32)
     │  GroupNorm                      │
     └──────────────┬──────────────────┘
                    │  F_hsi_enhanced  (1, 32, 32, 32)
                    ▼
     ┌─────────────────────────────────┐
     │           Row Scan              │
     │                                 │
     │  reshape: (1×32, 32, 32)        │  ← B*H sequences, len=W
     │  CrossMambaSSM1D                │
     │    x_hsi: (32, 32, 32)          │
     │    x_pan: (32, 32, 32)          │
     │    h_t = exp(Δ·A)·h_{t-1}       │
     │         + Δ·B(pan)·hsi_t        │
     │    y_t = C(pan)·h_t             │
     │  reshape back: (1, 32, 32, 32)  │
     └──────────────┬──────────────────┘
                    │ row_out  (1, 32, 32, 32)
                    ▼
     ┌─────────────────────────────────┐
     │          Column Scan            │
     │                                 │
     │  reshape: (1×32, 32, 32)        │  ← B*W sequences, len=H
     │  CrossMambaSSM1D                │
     │  reshape back: (1, 32, 32, 32)  │
     └──────────────┬──────────────────┘
                    │ col_out  (1, 32, 32, 32)
                    ▼
     ┌─────────────────────────────────┐
     │            Merge                │
     │                                 │
     │  concat([row, col], dim=1)      │   (1, 64, 32, 32)
     │  Conv1×1 (64→32)                │   (1, 32, 32, 32)
     │  GroupNorm                      │
     │  + F_hsi_enhanced  (residual)   │
     └──────────────┬──────────────────┘
                    ▼
            (1, 32, 32, 32)            ← fused features F_fused
```

---

### Block 4 — VMamba Backbone (4-stage U-Net)

```
F_fused
(1, 32, 32, 32)
     │
     │  Conv2d stride=2  (downsample 0.5×)
     ▼
(1, 32, 16, 16)
     │
     ├─────────────────────────────────────────────── skip_1
     │  Stage 1: MambaVisionBlock × num_blocks[0]
     │  [LayerNorm → SS2D (4-dir scan) → residual]
     ▼
(1, 32, 16, 16)
     │
     │  Conv2d(32→64, stride=2)   Down-1
     ▼
(1, 64, 8, 8)
     │
     ├─────────────────────────────────────────────── skip_2
     │  Stage 2: MambaVisionBlock × num_blocks[1]
     ▼
(1, 64, 8, 8)
     │
     │  Conv2d(64→128, stride=2)  Down-2
     ▼
(1, 128, 4, 4)                   ← bottleneck (minimum spatial)
     │
     │  Stage 3: MambaVisionBlock × num_blocks[2]
     ▼
(1, 128, 4, 4)
     │
     │  ConvTranspose2d(128→64, stride=2)  Up-1
     ▼
(1, 64, 8, 8)
     │
     │  concat with skip_2 → (1, 128, 8, 8)
     │  Conv1×1 (128→64)
     │  Stage 4a: MambaVisionBlock × (num_blocks[3]//2)
     ▼
(1, 64, 8, 8)
     │
     │  ConvTranspose2d(64→32, stride=2)   Up-2
     ▼
(1, 32, 16, 16)
     │  Stage 4b: MambaVisionBlock × remaining
     ▼
(1, 32, 16, 16)
     │
     │  ConvTranspose2d(32→32, stride=2)  Upsample back ×2
     ▼
(1, 32, 32, 32)                  ← backbone output
```

---

### Block 5 — Reconstruction Head

```
backbone_out              LR-HSI (original)
(1, 32, 32, 32)           (1, 128, 8, 8)
      │                         │
      │  Conv2d + GN + ReLU     │  F.interpolate (bicubic, ×4)
      ▼                         ▼
(1, 32, 32, 32)           (1, 128, 32, 32)
      │                         │
      │  Conv2d + GN + ReLU     │
      ▼                         │
(1, 32, 32, 32)                 │
      │                         │
      │  Conv1×1 (32→128)       │
      ▼                         │
(1, 128, 32, 32) ──── + ────────┘
      │        residual  bicubic
      ▼
(1, 128, 32, 32)                ← HR-HSI output ✓
```

---

## Complete Shape Summary — V2 (Improved)

```
Step                          Tensor Shape           Operation
─────────────────────────────────────────────────────────────────────────────
INPUT  LR-HSI                 (1, 128,   8,   8)
INPUT  HR-PAN                 (1,   1,  32,  32)
─────────────────────────────────────────────────────────────────────────────
HSI Encoder
  Reshape (add dim)           (1,   1, 128,   8,   8)   unsqueeze
  Conv3d(1→8)                 (1,   8, 128,   8,   8)   spectral features
  Reshape (flatten)           (1, 1024,   8,   8)        8×128=1024
  Conv1×1 (1024→32)           (1,  32,   8,   8)         spectral compress
  ResidualUpsample2×          (1,  32,  16,  16)         ×2 spatial
  ResidualUpsample2×          (1,  32,  32,  32)  F_hsi  ×2 spatial
─────────────────────────────────────────────────────────────────────────────
PAN Encoder
  Conv2d(1→32)+GN+ReLU        (1,  32,  32,  32)
  Conv2d(32→64)+GN+ReLU       (1,  64,  32,  32)
  Conv2d(64→32)+GN+ReLU       (1,  32,  32,  32)  F_pan
─────────────────────────────────────────────────────────────────────────────
Cross-Mamba Fusion
  LearnableHFInjection        (1,  32,  32,  32)   HF enhanced
  Row scan (SSM1D)            (1,  32,  32,  32)   O(N) along W
  Col scan (SSM1D)            (1,  32,  32,  32)   O(N) along H
  Concat + Conv1×1 + residual (1,  32,  32,  32)  F_fused
─────────────────────────────────────────────────────────────────────────────
VMamba Backbone
  Downsample (stride=2)       (1,  32,  16,  16)
  Stage 1 Mamba               (1,  32,  16,  16)   ← skip_1 saved
  Down1 (stride=2)            (1,  64,   8,   8)
  Stage 2 Mamba               (1,  64,   8,   8)   ← skip_2 saved
  Down2 (stride=2)            (1, 128,   4,   4)
  Stage 3 Mamba (bottleneck)  (1, 128,   4,   4)
  Up1 (ConvTranspose×2)       (1,  64,   8,   8)
  + skip_2, Conv1×1           (1,  64,   8,   8)
  Stage 4a Mamba              (1,  64,   8,   8)
  Up2 (ConvTranspose×2)       (1,  32,  16,  16)
  Stage 4b Mamba              (1,  32,  16,  16)
  Upsample ×2                 (1,  32,  32,  32)
─────────────────────────────────────────────────────────────────────────────
Reconstruction Head
  Conv+GN+ReLU ×2             (1,  32,  32,  32)
  Conv1×1 (32→128)            (1, 128,  32,  32)   residual
  Bicubic(LR-HSI, ×4)         (1, 128,  32,  32)   coarse estimate
  Add residual + bicubic      (1, 128,  32,  32)
─────────────────────────────────────────────────────────────────────────────
OUTPUT HR-HSI                 (1, 128,  32,  32)   ✓  4× upsampled
─────────────────────────────────────────────────────────────────────────────
```

---

## V1 — Old VMamba Pipeline (for comparison)

```
Step                          Tensor Shape           Operation
─────────────────────────────────────────────────────────────────────────────
INPUT  LR-HSI                 (1, 128,   8,   8)
INPUT  HR-PAN                 (1,   1,  32,  32)
─────────────────────────────────────────────────────────────────────────────
HSI Encoder (OLD)
  Reshape                     (1,   1, 128,   8,   8)   unsqueeze
  Conv3d(1→64)                (1,  64, 128,   8,   8)   heavier 3D conv
  Reshape                     (1, 8192,   8,   8)        64×128=8192
  PixelShuffle2×              (1, 2048,  16,  16)        ← checkerboard risk
  Conv2d + PixelShuffle2×     (1,  512,  32,  32)
  Conv1×1 (→d_model)         (1,  32,  32,  32)  F_hsi
─────────────────────────────────────────────────────────────────────────────
PAN Encoder (OLD — Sobel included)
  Conv2d(1→32)+GN+ReLU        (1,  32,  32,  32)
  Conv2d(32→64)+GN+ReLU       (1,  64,  32,  32)
  ResidualBlock               (1,  64,  32,  32)
  Fixed Sobel edges           (1,  64,  32,  32)   + alpha × |∇PAN|
  Conv1×1 (→d_model)         (1,  32,  32,  32)  F_pan
─────────────────────────────────────────────────────────────────────────────
CrossAttention Fusion (OLD — O(N²))
  Q, K, V projections         (1,  32,  32,  32)   each
  Reshape for multi-head      (1, 4, 8, 1024)      4 heads, d_k=8
  Scaled dot-product attn     (1, 4, 1024, 1024)   ← 1024×1024 matrix!
  Output projection           (1,  32,  32,  32)  F_fused
─────────────────────────────────────────────────────────────────────────────
VMamba Backbone (SAME as V2)
  Downsample (stride=2)       (1,  32,  16,  16)
  ... (identical U-Net stages) ...
  Upsample ×4                 (1,  32,  32,  32)
─────────────────────────────────────────────────────────────────────────────
Reconstruction Head (SAME as V2)
  Conv+GN+ReLU ×2             (1,  32,  32,  32)
  Conv1×1 (32→128)            (1, 128,  32,  32)
  + Bicubic(LR-HSI)           (1, 128,  32,  32)
─────────────────────────────────────────────────────────────────────────────
OUTPUT HR-HSI                 (1, 128,  32,  32)   ✓
─────────────────────────────────────────────────────────────────────────────
```

---

## V1 vs V2 — Component Differences

```
┌────────────────────┬──────────────────────────────┬──────────────────────────────────┐
│ Component          │ V1  Old VMamba               │ V2  Improved VMamba              │
├────────────────────┼──────────────────────────────┼──────────────────────────────────┤
│ HSI Encoder        │ Conv3d(1→64) → PixelShuffle  │ Conv3d(1→8) → Conv1×1 →         │
│                    │ 8192 intermediate channels   │ ResidualUpsample2× × 2           │
│                    │ Checkerboard artefacts risk  │ No artefacts, stable gradients   │
├────────────────────┼──────────────────────────────┼──────────────────────────────────┤
│ PAN Edge Extract   │ Fixed Sobel filter           │ Learnable Conv3×3 + sigmoid gate │
│                    │ Non-adaptive to scene        │ Adapts during training           │
├────────────────────┼──────────────────────────────┼──────────────────────────────────┤
│ Fusion             │ CrossAttention               │ CrossMambaFusion                 │
│                    │ O(N²) — 1024×1024 matrix     │ O(N) — SSM sequential scan       │
│                    │ OOM at large spatial sizes   │ Memory-efficient at any size     │
├────────────────────┼──────────────────────────────┼──────────────────────────────────┤
│ Backbone           │ Downsample 0.25×             │ Downsample 0.5×                  │
│ Reduction          │ 32×32 → 8×8                  │ 32×32 → 16×16                    │
│                    │ (fused at 8×8 spatial)       │ (fused at 16×16, more detail)    │
├────────────────────┼──────────────────────────────┼──────────────────────────────────┤
│ Parameters         │ ~1,840,352  (d_model=32)     │ ~677,288  (d_model=32)           │
│                    │ ~2,200,000  (d_model=64)     │ ~1,000,000 (d_model=64)          │
│                    │                              │ −62% fewer parameters            │
└────────────────────┴──────────────────────────────┴──────────────────────────────────┘
```

---

## CrossMamba SSM — Internal Data Flow

```
x_hsi  (B*H, W, d_model)        x_pan  (B*H, W, d_model)
    │                                │
    │  depthwise Conv1d             │  Linear(d_model → 2·d_state + d_model)
    ▼                                ▼
x_hsi_conv (B*H, W, d_model)   [B_param | C_param | Δ_raw]
                                 (B*H, W, d_state) (d_state) (d_model)
                │                        │           │
                │                        ▼           ▼
                │              Δ = softplus(Δ_raw)   (discretization)
                │
                │  Selective SSM scan (timestep t=0..W-1):
                │    h_t = exp(Δ_t · A) · h_{t-1}        A: (d_model, d_state)
                │         + Δ_t · B_t(pan) · x_hsi_t
                │    y_t = C_t(pan) · h_t
                │
                ▼
         y  (B*H, W, d_model)
                │
                │  skip: y += D · x_hsi_conv        D: learnable scalar
                │  Linear (d_model → d_model)
                ▼
        output (B*H, W, d_model)
```

---

## ResidualUpsample2× — Internal Flow

```
input  (B, C_in, H, W)
    │
    ├──── Main path ──────────────────────────────────┐
    │     F.interpolate (bilinear, ×2)                │
    │     (B, C_in, 2H, 2W)                           │
    │     Conv3×3 (C_in → C_out)                      │
    │     GroupNorm + ReLU                            │
    │     (B, C_out, 2H, 2W)                          │
    │                                                 │
    ├──── Residual path ──────────────────────────────┤
    │     Conv3×3 (C_in → C_out)                      │
    │     F.interpolate (bilinear, ×2)                │
    │     (B, C_out, 2H, 2W)                          │
    │                                                 │
    └──── Add (Main + Residual) ──────────────────────┘
                   │
                   ▼
           (B, C_out, 2H, 2W)     ← no checkerboard ✓
```

---

*Generated for M.Tech Thesis — Hyperspectral Pansharpening.*
*See `README_V2.md` for full architecture description and training commands.*
