# Version 2 — Improved VMamba Pansharpening

M.Tech Thesis | Hyperspectral Image Pansharpening using State-Space Models

---

## What V2 Improves Over V1

| Issue in V1 | V2 Fix |
|---|---|
| PixelShuffle checkerboard artefacts | `ResidualUpsample2x` (bilinear + conv residual) |
| Fixed Sobel edge filter | `LearnableHFInjection` (learned, content-adaptive gate) |
| `CrossAttentionFusion` — O(N²) dense attention | `CrossMambaFusion` — O(N) cross-modal SSM scan |
| Sequential SSM scan | Hillis-Steele **true parallel scan** (O(log L) depth) throughout |

## Architecture

```
LR-HSI ─→ ImprovedHSIEncoder ─┐
                               ├─→ CrossMambaFusion ─→ VMambaBackboneV2 (4-stage U-Net)
HR-PAN ─→ ImprovedPANEncoder ─┘         (O(N) fusion)      (Hillis-Steele scan blocks)
                                                                    │
                                                                    ▼
                                                          ReconstructionHead ─→ HR-HSI
```

`model/model.py` is fully self-contained: it defines its own `SelectiveSSMV2`,
`SS2DV2`, `MambaBlockV2`, `VMambaBackboneV2`, `ResidualUpsample2x`,
`ImprovedHSIEncoder`, `ImprovedPANEncoder`, `LearnableHFInjection`, `CrossMambaSSM1D`,
`CrossMambaFusion`, `ReconstructionHead`, and `V2VMambaPansharp` — none of it is
imported from `version1/` or anywhere else. (A separate reimplementation of these same
ideas, `src/scripts/vmamba_pansharp_improved.py`, is what `version3`-`version6` actually
import as their "improved" baseline for comparison tables — it is a different file with
different numbers; see the Results section below.)

---

## Folder Structure

```
version2/
├── requirements.txt          ← This version's own dependencies
├── model/
│   ├── model.py               ← V2VMambaPansharp and all its building blocks
│   ├── dataset.py              ← Dataset loader (Wald's protocol)
│   ├── losses.py                ← CompositeLoss + metrics (PSNR/SAM/ERGAS/SSIM)
│   └── logger.py                ← ExperimentLogger (CSV/JSON logging, optional FLOPs via thop)
├── train.py                   ← Train for one scale factor
├── run_experiment.py          ← Main entry point (verify / train / compare_scales / visualize)
└── runs/
    ├── scale4/                ← Checkpoints, logs, plots, metrics.csv for scale=4
    └── scale2/                ← Same, for scale=2
```

---

## Setup

```bash
cd version2
pip install -r requirements.txt
```

## Quick Commands

```bash
# Verify the model builds and runs for both scale 2 and 4
python run_experiment.py verify

# Train scale 4
python run_experiment.py train --scale 4 --epochs 30

# Train scale 2
python run_experiment.py train --scale 2 --epochs 30

# Train both scales and produce a comparison table
python run_experiment.py compare_scales --epochs 30

# Generate test visuals from saved checkpoints
python run_experiment.py visualize \
    --checkpoint_s4 runs/scale4/checkpoints/best.pth \
    --checkpoint_s2 runs/scale2/checkpoints/best.pth
```

`train.py` can also be run directly for a single scale:

```bash
python train.py --scale 4 --epochs 30 --batch_size 2 --d_model 32
```

All logs, checkpoints, and images are saved under `runs/scale<N>/`.

---

## Results (scale 4, epoch 50/50)

| Metric | Value |
|---|---|
| PSNR (dB) | 44.32 |
| SAM (°) | 6.97 |
| ERGAS | 4.53 |
| SSIM | 0.9969 |
| Params | 741,352 |

Source: `runs/scale4/log.json` / `metrics.csv` (final epoch). See
`src/results/master_comparison.txt` for how this compares to the other five versions,
and to the separate `vmamba_pansharp_improved.py` reimplementation that V3-V6 use as
their "improved" baseline (44.98 dB / 6.77° SAM — a different run of similar ideas).
