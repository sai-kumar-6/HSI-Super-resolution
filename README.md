# VMamba Hyperspectral Pansharpening

Hyperspectral image pansharpening using Mamba-based state-space models.
The model architecture and training pipeline evolved across six iterations, each kept as a
self-contained snapshot under `src/`.

## Repository layout

```
.
├── docs/                      Thesis-level documentation
│   ├── ARCHITECTURE.md
│   ├── ARCHITECTURE_COMPARISON.md
│   ├── FLOPS_ANALYSIS.md
│   ├── QUICKSTART.md
│   └── SAI_SC24M175.pdf       Thesis report
├── requirements.txt
└── src/
    ├── version1 .. version6/  One folder per architecture iteration
    │   (model, dataset loader, loss functions, training/comparison/testing scripts;
    │    see each version's own README for what changed and how to run it)
    ├── v1v2_comparison/        Head-to-head comparison harness for v1 vs v2
    ├── chikusei/               Dataset folder (data files are git-ignored, see below)
    ├── results/                Cross-version comparison summaries
    ├── training_plots/         Training curve plots
    └── scripts/                Shared modules and cross-version orchestration scripts
        ├── dataset_loader_overlap.py, loss_functions.py, vmamba_pansharp_improved.py
        │                       Shared modules imported by version3-6
        └── train_all_versions.py, generate_poster.py, generate_architecture_diagram.py,
            json_to_excel.py, plot_*.py, regenerate_metric_plots.py
                                Cross-version orchestration/reporting scripts
```

Each `version*` script resolves its project root dynamically as its own parent directory,
so the whole `src/` cluster must stay together as-is — that's why the code and data live
in one nested folder while `docs/` stays at the top level. `scripts/` sits as a sibling of
`version1`..`version6` for the same reason: each version's `sys.path` setup adds
`src/scripts` alongside its own folder so it can still resolve the shared modules.

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
across versions from one entry point, see `src/train_all_versions.py`.
