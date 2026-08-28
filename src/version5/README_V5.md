# Version 5 — True Parallel Scan VMamba for Hyperspectral Pansharpening

## Overview

V5 replaces the sequential chunked scan (V1–V4) with a **true Hillis-Steele parallel
prefix scan** that runs in O(log L) depth instead of O(L) sequential steps. All other
architectural improvements from V4 are preserved.

---

## Key Improvement: Hillis-Steele Parallel Scan

### V1–V4 Sequential Scan (What We Had)

```python
# O(L) sequential — L Python loop iterations
h = zeros(B, D, N)
for t in range(L):          # 32 iterations for L=32
    h = A_bar[t] * h + B_bar[t] * x[t]
    y[t] = C[t] @ h
```

For L = 32 (32×32 spatial grid), this runs **32 sequential steps** — each step depends
on the previous one, so it cannot be parallelised.

### V5 Hillis-Steele Parallel Scan (What We Have Now)

```python
# O(log L) depth — log2(L) Python iterations
# Binary operator: (a2, b2) ⊕ (a1, b1) = (a2·a1, a2·b1 + b2)
for d in range(log2(L)):    # 5 iterations for L=32
    stride = 1 << d
    prev_a = shift_right(a, stride, fill=1.0)
    prev_b = shift_right(b, stride, fill=0.0)
    new_b = a * prev_b + b   # parallel across all L positions
    new_a = a * prev_a
    a, b = new_a, new_b

# After log2(32)=5 iterations: b[t] = h_t for all t simultaneously
```

| Metric          | Sequential (V1–V4) | Parallel (V5)   |
|-----------------|--------------------|-----------------|
| Python loop iters | L = 32           | log₂(32) = 5   |
| Depth (algorithmic) | O(L) = O(32)   | O(log L) = O(5) |
| Parallelism     | None               | Full (all t at once) |
| GPU utilisation | Low (serial)       | High (tensor ops) |
| Numerical result | Identical         | Identical (verified) |

---

## V5 Architectural Components

### 1. `hillis_steele_scan(a, b)` — Core Algorithm

```
Input:  a ∈ R^(B,L,D,N)  — diagonal A_bar values
        b ∈ R^(B,L,D,N)  — B_bar * x values
Output: h ∈ R^(B,L,D,N)  — hidden states for all positions

Binary operator: (a_right, b_right) ⊕ (a_left, b_left)
               = (a_right * a_left, a_right * b_left + b_right)

Pad L to next power of 2 → apply log2(L_pad) passes → trim to L_orig
```

Correctness verified: max error vs sequential reference < 1e-4.

### 2. `TrueParallelSelectiveSSM` — Full Mamba SSM with H-S scan

- Input projections (x, z branches) via Linear
- Compute input-selective Δ, B, C from x
- Discretise: A_bar = exp(Δ·A), B_bar = Δ·B (Zero-Order Hold)
- Call `hillis_steele_scan` → get all hidden states in parallel
- Output = C @ h + D·x, gated by sigmoid(z)
- **Zero-init out_proj**: starts at identity residual, stabilises early training

### 3. `TrueParallelSS2D` — 4-Direction 2D Spatial Scan

Each of the 4 scan directions gets its own **direction-specific adapter**:
```
Direction 0: row forward  (left → right)
Direction 1: row backward (right → left)
Direction 2: col forward  (top → bottom)
Direction 3: col backward (bottom → top)
```

Per-direction adapter = `Linear(d_model → d_model) + GroupNorm` allows the model
to learn that row scans and column scans carry different types of spatial information.

### 4. `TrueParallelSpectralSSM1D` — Spectral Scan

Groups C=128 channels into n_groups=8 spectral tokens of size d_model=32.
Applies H-S scan across the 8 spectral tokens. Captures inter-band correlations.

### 5. `V5SpectralSpatialBlock` — Pre-norm Architecture

```
V4 (post-norm):  y = Norm(SSM(x) + x)
V5 (pre-norm):   y = SSM(Norm(x)) + x
```

Pre-norm is more numerically stable and converges faster in SSMs with long sequences.
Used in modern transformers (LLaMA, GPT-NeoX) and Mamba variants.

### 6. `V5MultiScaleBackbone` — 4-Stage U-Net with V5 Blocks

```
(B,32,32,32) → Downsample → (B,32,16,16)
Stage 1: V5Block × num_blocks[0] → (B,32,16,16)
Down(2×): → (B,64,8,8)
Stage 2: V5Block × num_blocks[1] → (B,64,8,8)
Down(2×): → (B,128,4,4)
Stage 3: V5Block × num_blocks[2] (bottleneck) → (B,128,4,4)
Up+skip: → (B,64,8,8)
Stage 4a: V5Block × num_blocks[3] → (B,64,8,8)
Up: → (B,32,16,16)
Stage 4b: V5Block × num_blocks[3] → (B,32,16,16)
Upsample(2×): → (B,32,32,32)
```

---

## Inherited V4 Components

| Component | Description |
|-----------|-------------|
| V2 HSI Encoder | 3D Conv → 2D Conv → ResidualUpsample × 2 |
| V2 PAN Encoder | 3-layer Conv with GroupNorm |
| ScaledSpatialDetailInjection | F_hsi + α × Gate × Edges (α=learnable, init=0.1) |
| StrongReconstructionHead | 3×Conv3×3 + β·Residual + Bicubic anchor (β=learnable, init=0.5) |
| CompositeLossV5 | L1 + 0.10·SAM + 0.05·Edge + 0.01·SSIM + 0.05·SpectralGradient |

---

## Running V5

```bash
cd version5
pip install -r requirements.txt

# Verify all 5 models load correctly
python run_experiment.py verify

# Quick test (random weights)
python run_experiment.py test

# Train only V5 (fast)
python run_experiment.py compare \
    --skip_old --skip_improved --skip_v3 --skip_v4 \
    --epochs 30 --batch_size 1 --patch_size 32

# Train all 5 variants and compare
python run_experiment.py compare --epochs 30

# Visualize V5 results
python run_experiment.py visualize \
    --checkpoint_v5 comparison/checkpoints/v5_best.pth
```

---

## Results

Two recorded runs give different numbers and haven't been reconciled yet:

| Source | PSNR | SAM | ERGAS | SSIM |
|---|---|---|---|---|
| `comparison/results/comparison_results.json` | 48.63 | 6.38° | 5.41 | 0.9992 |
| Used in `version6/README.md`'s evolution table | 43.34 | 6.74° | 5.61 | — |

Before citing either number, check which `comparison/checkpoints/v5_best.pth` run they
came from — see `src/results/master_comparison.txt` for the same flag at the project level.

---

## Expected Benefits of True Parallel Scan

1. **GPU efficiency**: On CUDA, the 5 tensor passes in H-S scan can overlap in
   compute streams, unlike 32 sequential Python calls
2. **Gradient flow**: All-parallel computation gives cleaner gradients vs serial
   truncated BPTT through L steps
3. **Pre-norm stability**: Faster convergence, especially important with the
   spectral gradient loss that penalises band-to-band smoothness
4. **Direction specialisation**: Per-direction adapters in SS2D allow the model
   to differentiate row-scan (horizontal edges) from col-scan (vertical edges)

---

## 3D Scan Analysis

See [3D_SCAN_ANALYSIS.md](3D_SCAN_ANALYSIS.md) for a detailed explanation of:
- Why a true 3D volumetric scan is theoretically possible
- Why it would be 192× more expensive and OOM in practice
- How SpectralSSM1D + SS2D achieves the same coverage efficiently
- When 3D scans would be appropriate (small volumes, custom CUDA)

---

## File Structure

```
version5/
├── requirements.txt               ← This version's own dependencies
├── model/
│   ├── model.py                   ← V5VMambaPansharp (true parallel scan), plus
│   │                                 V1/V3/V4 classes it builds on or offers as
│   │                                 comparison baselines (duplicated here)
│   ├── losses.py                  ← CompositeLossV5; re-exports + duplicates V4 losses
│   └── baselines.py               ← Factory for all 5 variants
├── run_experiment.py              ← Main runner
├── 3D_SCAN_ANALYSIS.md            ← 3D scan explanation
├── README_V5.md                   ← This file
├── comparison/
│   ├── compare_all.py             ← Train & compare all 5 variants
│   ├── checkpoints/               ← Saved best checkpoints
│   ├── plots/                     ← Training curves + comparison bar chart
│   ├── results/                   ← JSON + Excel summary
│   └── saved_images/              ← Per-epoch false-colour images
├── testing/
│   ├── test_chikusei.py           ← Evaluate on test set
│   └── results/                   ← Test metrics JSON
└── visualization/
    ├── visualize_results.py       ← Publication figures
    └── outputs/
        ├── false_colour_comparison.png
        ├── error_map_comparison.png
        ├── spectral_curves.png
        └── individual/            ← Per-variant prediction images
```

**Self-containment note:** `model/model.py` and `model/losses.py` physically contain
V1's, V3's, and V4's classes they build on (not just V5's own new ones) — duplicated in
rather than imported from those version folders, so this `version5/` directory is
independently runnable on its own. Only the shared, version-agnostic building blocks
(`vmamba_pansharp_improved.py`, `loss_functions.py`, `dataset_loader_overlap.py` under
`src/scripts/`) are still imported from outside this folder, since they aren't owned by
any one version.
