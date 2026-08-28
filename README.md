# VMamba Hyperspectral Pansharpening

Hyperspectral image pansharpening using Mamba-based state-space models.
The model architecture and training pipeline evolved across six iterations, each kept as a
self-contained snapshot under `src/`.

## Repository layout

```
Mtech-project/
├── README.md
├── requirements.txt                  Root-level, covers the whole repo
├── .gitignore
│
├── docs/                             Thesis-level documentation
│   ├── ARCHITECTURE.md
│   ├── ARCHITECTURE_COMPARISON.md
│   ├── FLOPS_ANALYSIS.md
│   └── QUICKSTART.md
│       (SAI_SC24M175.pdf, the thesis report, is kept locally only — git-ignored)
│
└── src/
    ├── chikusei/                     Dataset folder (*.mat files git-ignored)
    │   ├── chikusei.mat
    │   ├── HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat
    │   └── *.png                     Dataset preview/visualization images
    │
    ├── results/                      Cross-version comparison summaries
    │   ├── master_comparison.json      Structured, with per-row source/notes
    │   ├── master_comparison.txt        Human-readable table + notes
    │   └── master_comparison_chart.png
    │
    ├── training_plots/               Training curve plots (mainly v5)
    │
    ├── scripts/                      Shared modules + cross-version orchestration
    │   ├── dataset_loader_overlap.py   Shared dataset loader (used by v5/v6)
    │   ├── loss_functions.py            Shared base losses (used by v3-v6)
    │   ├── vmamba_pansharp_improved.py  Shared "improved" (V2-style) baseline model
    │   ├── train_all_versions.py        Train/test all versions from one entrypoint
    │   ├── generate_poster.py, generate_architecture_diagram.py, json_to_excel.py
    │   └── plot_training_metrics.py, plot_v5_metrics.py, plot_v6_metrics.py,
    │       regenerate_metric_plots.py
    │
    ├── v1v2_comparison/               Standalone head-to-head harness for v1 vs v2
    │   ├── baseline_models.py, dataset_loader.py, loss_functions.py, run_experiment.py
    │   ├── comparison/compare_vmamba.py
    │   ├── testing/test_chikusei.py
    │   └── visualization/visualize_results.py
    │
    ├── version1/                     V1 — CrossAttention + Sobel + PixelShuffle
    │   ├── README_V1.md
    │   ├── requirements.txt
    │   ├── model/
    │   │   └── model.py                VMambaPansharp + all its building blocks
    │   ├── docs/                       19 legacy design docs
    │   ├── old_scripts/                26 archived scripts + notebooks (not maintained)
    │   ├── old_folders/                 Archived unused folders (git-ignored)
    │   ├── experiments/                 15 timestamped training runs (git-ignored)
    │   ├── comparison_results/          18 timestamped baseline-comparison runs (git-ignored)
    │   └── visualizations/              Dataset overview figures
    │
    ├── version2/                     V2 — CrossMamba + LearnableHF + ResidualUpsample
    │   ├── README.md
    │   ├── requirements.txt
    │   ├── model/
    │   │   ├── model.py, dataset.py, losses.py, logger.py
    │   ├── train.py, run_experiment.py
    │   └── runs/scale4/, runs/scale2/   Checkpoints, logs, plots (git-ignored)
    │
    ├── version3/                     V3 — SpectralMambaBlock + SDIM + 4-stage U-Net
    │   ├── README_V3.md, FLOPS_ANALYSIS.md
    │   ├── requirements.txt
    │   ├── model/
    │   │   ├── model.py                V1's classes (duplicated) + V3's own
    │   │   ├── baselines.py, dataset.py, dataset_overlap.py, losses.py
    │   ├── run_experiment.py
    │   ├── comparison/compare_all.py + checkpoints/, plots/, results/ (git-ignored)
    │   ├── testing/test_chikusei.py
    │   └── visualization/               (scaffold only, no implementation yet)
    │
    ├── version4/                     V4 — ScaledSDIM + StrongReconHead + SpectralGradLoss
    │   ├── README_V4.md
    │   ├── requirements.txt
    │   ├── model/
    │   │   ├── model.py                V1+V3's classes (duplicated) + V4's own
    │   │   ├── baselines.py, dataset.py, dataset_overlap.py, losses.py
    │   ├── run_experiment.py
    │   ├── comparison/compare_all.py + checkpoints/, plots/, results/, saved_images/ (git-ignored)
    │   ├── testing/test_chikusei.py
    │   └── visualization/visualize_results.py + outputs/
    │
    ├── version5/                     V5 — Hillis-Steele true parallel scan
    │   ├── README_V5.md, 3D_SCAN_ANALYSIS.md
    │   ├── requirements.txt
    │   ├── model/
    │   │   ├── model.py                V1+V3+V4's classes (duplicated) + V5's own
    │   │   ├── baselines.py, losses.py
    │   ├── run_experiment.py
    │   ├── comparison/compare_all.py + checkpoints/, plots/, metric_plots/, results/, saved_images/ (git-ignored)
    │   ├── testing/test_chikusei.py + results/
    │   └── visualization/visualize_results.py + outputs/
    │
    └── version6/                     V6 — n_groups=32, spectral-focused loss weights
        ├── README.md
        ├── requirements.txt
        ├── model/
        │   ├── model.py                V1+V3+V4+V5's classes (duplicated) + V6's own
        │   ├── baselines.py, losses.py
        ├── baselines/                  Classic (non-VMamba) baselines: PanNet/PanGAN/PSGAN/Panformer
        │   ├── baseline_architectures.py, train_baselines.py, compare_inference.py
        ├── run_experiment.py, inference_visual.py
        ├── comparison/compare_all.py + checkpoints/, plots/, metric_plots/, results/, saved_images/ (git-ignored)
        ├── testing/test_chikusei.py
        └── v6_metric_plots/, inference_results/
```

Every `versionN/` folder above follows the same shape (`model/`, `run_experiment.py`,
`comparison/`, `testing/`, `visualization/`, its own `README*.md` and `requirements.txt`)
once you get to V3 — V1 and V2 predate that convention and are simpler (V1 is just its
model file plus an archived legacy folder; V2 has no `comparison/`/`testing/` split since
it isn't part of the multi-variant comparison harness).

Each `version*` script resolves its project root dynamically as its own parent directory,
so the whole `src/` cluster must stay together as-is — that's why the code and data live
in one nested folder while `docs/` stays at the top level. `scripts/` sits as a sibling of
`version1`..`version6` for the same reason: each version's `sys.path` setup adds
`src/scripts` alongside its own folder so it can still resolve the shared modules.

Every version's `model/` folder uses the same file names (`model.py`, `dataset.py`,
`losses.py`, `baselines.py`) for consistency, and **each version folder is independently
self-contained**: even though v4 architecturally builds on v3's classes, v5 on v3/v4's,
and v6 on v1/v4/v5's, those reused classes are physically duplicated into the later
version's own `model/model.py` (and `model/losses.py` where relevant) rather than
imported across folders. This means every `versionN/` can be copied out and run on its
own, at the cost of some duplicated code — see the "Self-containment note" in each
version's own README for exactly which classes are its own vs. duplicated from an
earlier version. The only things still imported from outside a version folder are the
shared, version-agnostic building blocks under `src/scripts/`
(`vmamba_pansharp_improved.py`, `loss_functions.py`, `dataset_loader_overlap.py`), since
those aren't owned by any one version.

Each version also has its own `requirements.txt` (trimmed to what that version actually
needs — e.g. only v3/v4 need `opencv-python` for their own dataset loader) in addition to
the root `requirements.txt` covering the whole repo.

## Version history

| Version | Focus |
|---|---|
| v1 | Baseline VMamba pansharpening architecture |
| v2 | Hillis-Steele parallel scan, per-band normalization |
| v3 | SpectralMamba + SpatialDetailInjection |
| v4 | ScaledSDIM + StrongHead + SpectralGradLoss |
| v5 | True parallel scan throughout the network |
| v6 | Spectral-focused architecture, reduced SAM |

See `src/versionN/README*.md` for full architectural details and per-version run instructions.

## Results

| Version | PSNR (dB) | SAM (°) | ERGAS | Params |
|---|---|---|---|---|
| V1 | 44.65 | 6.74 | 71.72 | 1.84M |
| V2 | 44.32 | 6.97 | 4.53 | 0.74M |
| V3 | 47.35 | 6.61 | 61.20 | 1.52M |
| V4 | 49.37 | 6.32 | 4.49 | 1.53M |
| V5 | 43.34 | 6.74 | 5.61 | 1.77M |
| V6 | 47.69 | 2.64 | 3.33 | 6.78M |

V4 has the best PSNR/SAM combination overall; V6 has by far the best SAM (spectral
fidelity), beating its own 3.0–4.5° target. See `src/results/master_comparison.txt` /
`master_comparison.json` for exact sources per number and two flagged discrepancies
worth checking before citing in the thesis (a newer, better V5 run not yet reconciled,
and two different "V2" numbers from two different implementations/runs).

## Setup

```bash
pip install -r requirements.txt
```

## Data

This repo does not track raw datasets or trained weights (see `.gitignore`) — `.mat`, `.npy`,
`.pth`/`.pt`/`.ckpt` files, and experiment output folders (`checkpoints/`, `experiments/`,
`comparison_results/`, `runs/`) are excluded. Place datasets under `src/chikusei/` (expects
`chikusei.mat`) or the path each version's dataset loader expects before training.

## Running a version

```bash
cd src/version6
python run_experiment.py --epochs 30
```

Refer to each version's own README for its exact CLI and expected outputs. To train/compare
across versions from one entry point, see `src/scripts/train_all_versions.py`.
