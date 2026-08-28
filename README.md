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
    │   ├── model/              model.py, dataset.py, losses.py, baselines.py
    │   │                       (naming is consistent across versions; see below
    │   │                        for how identically-named files coexist)
    │   ├── run_experiment.py   Training/testing entrypoint
    │   ├── comparison/         compare_all.py + checkpoints/, plots/, results/
    │   ├── testing/            test_chikusei.py + results/
    │   └── visualization/      visualize_results.py + outputs/
    │   (see each version's own README for what changed and how to run it)
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
