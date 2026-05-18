# COALITION-4 Nowcasting System — Romanian Adaptation

Adaptation of the [COALITION-4](https://doi.org/10.1175/MWR-D-22-0084.1) deep learning nowcasting system (Leinonen et al., 2022, MeteoSwiss) for Romanian meteorological conditions. Developed at Romania's National Meteorological Administration (ANM) as part of the EUMETSAT Training Placement Scheme.

The system uses a recurrent-convolutional encoder-decoder architecture operating on the EPSG:31700 Stereo70 projection of Romania to produce 45-minute precipitation and lightning nowcasts from multi-source meteorological inputs.

## Project Overview

Three training configurations are supported. The first two compare MSG vs MTG with the legacy ANM-radar precipitation target; the third replaces the legacy radar with the pan-European **OPERA** composite.

- **MSG experiment**: ANM radar + LINET lightning + MSG SEVIRI (5 channels, 3 km)
- **MTG experiment**: ANM radar + LINET lightning + MTG FCI (5 channels, 1–2 km)
- **MTG + OPERA experiment**: **OPERA** precipitation composite (reflectivity + instantaneous rainfall rate, 2 km, 15 min) + MTG FCI. The label is `opera_rainfall_rate` multi-class (5 bins, same thresholds as the radar configuration). No lightning.

The first two share the same radar targets and lightning labels; only the satellite input branch differs. The OPERA configuration switches the radar source: same downstream pipeline (reprojection to EPSG:31700, DBSCAN-driven patch index, sequence extraction, normalization, dataset creation, training, evaluation), but driven by OPERA's pre-reprojected HDF5 instead of the legacy ANM-radar NetCDF.

### Two sample-selection tracks (`--source`)

Independently of which training mode you pick, sample selection now has two parallel tracks that can coexist on disk:

| Track | What drives sample selection | Per-patch index produced by | `extract_patch_seq` / `create_datasets` / `train_models` flag |
|---|---|---|---|
| **DBSCAN track** | DBSCAN clusters in OPERA `rainfall_rate` (or RZC) — broad convective coverage | `identify_patches.py --source {radar, opera}` | `--source dbscan` |
| **Lightning track** | Per-map occurrence-fraction threshold (≥ 0.30 × mean over active maps) — lightning-active windows only | `identify_lightning_periods.py` | `--source lightning` |

> Note the two `--source` vocabularies: `identify_patches.py --source {radar, opera}` picks the **sensor** whose data is clustered (both writes go to the same `patch_index.csv`); downstream `extract_patch_seq / create_datasets / train_models --source {dbscan, lightning}` picks the **index file** to consume (`patch_index.csv` vs `lightning_patches.csv`).

Both tracks share the same model architecture and downstream training script. Every artefact emitted by `extract_patch_seq` / `create_datasets` / `train_models` is suffixed with `_<source>` so the two tracks never overwrite each other. Run `python train_models.py --source dbscan --stage both` on one machine and `... --source lightning --stage both` on another to train them in parallel.

For end-to-end commands per track, see the runbooks at the project root:

- [`run_opera.config`](run_opera.config) — full pipeline for the OPERA-driven track (comments-only).
- [`run_lightning.config`](run_lightning.config) — full pipeline for the lightning-driven track (comments-only).

### Optional domain-adaptation fine-tune (Swin transformer head)

`train_models.py --stage both` adds an optional second training stage after the base model finishes: load the saved base, **freeze the encoder-forecaster**, graft a 2-block Swin transformer head with 8×8 windowed attention on top, and fine-tune that head with AdamW. The Swin head shares features once across the 3 lead-time predictions and projects each lead-time output through an independent lightweight Conv 1×1 head. All hyperparameters live in the `[finetune]` section of [`training.config`](training.config). For a simpler first run, use `--stage base` instead — that's the standard COALITION-4 model with no transformer head.

### Optional class-weighted focal loss for the radar / OPERA multiclass head

The OPERA rainfall target is severely imbalanced: class 0 (`R<10` mm/h) is ~98% of pixels, class 4 (`R≥40`) is sparser by orders of magnitude. The `[radar_loss]` section of [`training.config`](training.config) lets you swap the historical `CategoricalCrossentropy(label_smoothing=0.01)` for `WeightedFocalCategoricalCrossentropy` — focal modulation `(1 − p_t)^gamma` plus per-class weights derived from the training-distribution pixel fractions (read from `our_data/opera_rainfall_fraction_<source>.json`, produced by `opera_rainfall_fraction.py --source <source>`). `weighting` accepts `inverse`, `median`, or `none`; setting `weighting = none` and `gamma = 0` reproduces the plain-CCE baseline exactly. Per-source like the lightning prior, so the dbscan / lightning tracks each compute their own.

## Environment Setup

### Prerequisites

- **Conda** (Miniconda or Anaconda)
- **NVIDIA GPU** with CUDA support
- **Windows** (the pipeline was developed and tested on Windows; TensorFlow GPU support on Windows requires specific version pinning)

### Installation

1. Create a conda environment with Python 3.10 (one of the last versions with native Windows GPU support for TensorFlow):

```bash
conda create -n tfenv python=3.10
conda activate tfenv
```

2. Install CUDA toolkit and cuDNN via conda-forge:

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Verify GPU access:

```python
import tensorflow as tf
tf.config.list_physical_devices('GPU')
tf.test.is_gpu_available()
```

## Directory Structure

```
coalition4-rcnn/
│
├── c4dl/                              # Core library (projection, datasets, features)
│   ├── projection.py                  # GridProjection, romania_grid_area (EPSG:31700)
│   ├── datasets/                      # Data readers (mchradar, mchlightning, msgccs4, etc.)
│   └── features/
│       └── regions.py                 # Patch saving functions (save_patches_radar, etc.)
│
├── our_data/                          # All data (not tracked in git)
│   ├── radar_data/
│   │   ├── RZC/nc4_{date}-Romania_RZC/*.nc
│   │   ├── BZC/nc4_{date}-Romania_BZC/*.nc
│   │   ├── CZC/...
│   │   ├── EZC-20/...
│   │   ├── LZC/...
│   │   └── CPCH/...
│   ├── satellite_data/
│   │   ├── MSG/
│   │   │   ├── VIS006/nc4_{date}-Romania_VIS006/*.nc
│   │   │   ├── IR_039/...
│   │   │   ├── IR_108/...
│   │   │   ├── WV_062/...
│   │   │   └── WV_073/...
│   │   ├── MTG/
│   │   │   ├── mtg_constants.json     # Grid constants (geos projection + x/y scan angles)
│   │   │   │                          #   written by pipeline_msg_mtg.py on first run
│   │   │   ├── vis_06/nc4_{date}-Romania_vis_06/*.npy  # per-channel arrays on geos grid
│   │   │   ├── ir_38/...
│   │   │   ├── ir_105/...
│   │   │   ├── wv_63/...
│   │   │   └── wv_73/...
│   │   ├── pipeline_msg_mtg.py        # MTG FCI L1C SFTP + Satpy processing pipeline
│   │   ├── inspect_mtg.py             # Reconstruct .nc and plot raw / reprojected MTG data
│   │   ├── summarize_mtg.py    # CSV report of downloaded FCI chunks per date
│   │   ├── check_chunk_names.py       # Diagnostic: inspect FCI chunk NetCDF structure
│   │   ├── check_chunk_contents.py    # Diagnostic: read FCI chunk radiance data
│   │   └── _raw_chunks/               # Cache of downloaded FCI chunk files (gitignored)
│   ├── lightning_data/
│   │   ├── kml_data/{date}/{date}.kml
│   │   ├── density/nc4_{date}-Romania_density/*.npy
│   │   ├── current/nc4_{date}-Romania_current/*.npy
│   │   ├── occurrence/nc4_{date}-Romania_occurrence/*.npy
│   │   ├── read_kml_version2.py       # KML → per-cadence .npy lightning maps (filter-aligned, variable windows)
│   │   ├── summarize_lightning_data.py # Scan .npy → lightning_summary.csv + lightning_active_steps.csv at project root
│   │   └── visualize_lightning_stats.py  # Lightning activity bar plots (reads lightning_active_steps.csv, plots-only)
│   ├── opera_data/
│   │   ├── reflectivity/{YYYY}/{MM}/{DD}/*.h5      # OPERA max-reflectivity HDF5 (2 km, 15 min)
│   │   ├── rainfall_rate/{YYYY}/{MM}/{DD}/*.h5     # OPERA rain-rate HDF5 (2 km, 15 min)
│   │   ├── pipeline_opera.py          # SFTP/SCP download from EWC VM with cadence filtering
│   │   └── summarize_opera_data.py    # CSV + missing-timesteps report
│   ├── raw_data/
│   │   ├── radar_arrange.py           # Arrange raw radar files to COALITION-4 structure
│   │   └── lightning_arrange.py       # Arrange raw KML files (date-based or sequential)
│   ├── reprojected_data/                # Cached reprojected data (generated by reproject.py)
│   │   ├── romania_grid_lats.npy           # Shared 768×1536 EPSG:31700 lat array
│   │   ├── romania_grid_lons.npy           # Shared 768×1536 EPSG:31700 lon array
│   │   ├── radar_data/{product}/nc4_{date}-Romania_{product}/*.npy
│   │   ├── satellite_data/MSG/{channel}/nc4_{date}-Romania_{channel}/*.npy
│   │   ├── satellite_data/MTG/{channel}/nc4_{date}-Romania_{channel}/*.npy
│   │   ├── lightning_data/{product}/nc4_{date}-Romania_{product}/*.npy
│   │   ├── opera_data/{product}/nc4_{date}-Romania_{product}/*.npy
│   │   └── opera_data/opera_constants.json                    # source projection (/where attrs)
│   ├── patch_index/                   # DBSCAN patch identification output (identify_patches --source {radar, opera})
│   │   ├── patch_index.csv
│   │   ├── patch_index.json
│   │   └── plots/                     # Optional diagnostic plots
│   ├── lightning_periods/             # Occurrence-fraction filter output (--source lightning)
│   │   ├── lightning_periods_config.json   # Parameters used + threshold metadata (reproducibility)
│   │   └── lightning_patches.csv      # Per-map per-patch index keyed by master HHMM (schema mirrors patch_index.csv)
│   ├── patches/                       # Extracted 256×256 patches (generated by extract_patches.py)
│   │   └── {date}/{variable}_{HHMM}_{HR|LR}.npy
│   ├── datasets/                      # Saved TF datasets (generated by create_datasets.py)
│   │   └── {mode}_{source}/train|validation|test/
│   │       └── metadata.json          # Input shapes, label type (used by train_models.py)
│   ├── data_statistics/               # Diagnostic plots (generated by data_statistics.py)
│   ├── timestep_config.json           # Cadence config (generated by validate_timestep.py)
│   ├── sequence_meta_{source}.json    # Per-sample window (generated in Step 4.1, per source)
│   ├── timestep_manifest.csv          # Surviving (date, HHMM) timesteps (generated in Step 4.2)
│   ├── intersect_summary.png          # Per-date stacked bar of kept vs dropped (Step 4.2)
│   ├── normalization_stats_{source}.json  # Per-variable mean/std per source (generated in Step 4.3)
│   ├── train_data_{source}.csv        # Training sequences per source (80% per temporal block)
│   ├── validation_data_{source}.csv   # Validation sequences per source (10% per temporal block)
│   ├── test_data_{source}.csv         # Test sequences per source (10% per temporal block)
│   ├── extract_patch_seq_drops_{source}.csv  # Audit: (date, HHMM) dropped by the manifest gate
│   └── lightning_fraction_{source}.json  # Per-source training-scope non-zero pixel fraction for focal loss
│
├── models/                            # Saved trained models (not tracked in git)
│   └── {mode}/
├── evaluation/                        # Evaluation outputs (not tracked in git)
│   └── eval_{mode}/
│
├── product_cadences.config              # Native cadence per data product (input to Step 0)
├── training.config                     # train_models.py hyperparameters + [finetune] Swin head + [radar_loss] focal/class-weighted CCE config
├── run_lightning.config                # Comments-only runbook for the lightning-driven track
├── run_opera.config                    # Comments-only runbook for the OPERA-driven track
│
├── validate_timestep.py               # Step 0: Validate cadence → timestep_config.json
├── identify_patches.py                # Step 1 (--source {radar, opera}): DBSCAN → patch_index.csv
├── identify_lightning_periods.py      # Step 1 (--source lightning): occurrence-fraction filter → lightning_periods/lightning_patches.csv
├── reproject.py                          # Step 2: Reproject all products to Romania grid; also aggregates per-category logs into errors.txt
├── extract_patches.py                 # Step 3: Slice 256×256 patches from reprojected data (driven by Step 4.1's per-source split CSVs; --source dbscan / lightning)
├── extract_patch_seq_for_datasets.py  # Step 4.1: Continuous sequences + Czibula split + manifest gate (per-source CSVs)
├── intersect_product_coverage.py      # Step 4.2: Per-timestep manifest + plot (--summary / --missing / --active per product)
├── compute_normalization_stats.py     # Step 4.3: Per-variable mean/std → normalization_stats_<source>.json (--source dbscan / lightning)
├── create_datasets.py                 # Step 5: Build TF datasets + metadata.json (--source aware)
├── train_models.py                    # Step 6: Train (--stage base / finetune / both, --source dbscan / lightning)
├── evaluate_coalition.py              # Step 7: Evaluate, generate metrics + plots (--source / --finetuned)
├── bundle_eval_scores.py              # Bundle per-mode eval results into Shapley-ready per-leadtime CSVs (--source / --finetuned)
├── visualize_full_domain_predictions.py  # Top-N reference timesteps -> full-domain GT vs Pred (Romania-centred, country borders, zoom-in)
│
├── feature_importance_analysis.py     # Grad-CAM + Xi, SHAP, classical Shapley analysis
├── lightning_fraction.py              # Per-source lightning-occurrence prior for the binary focal loss (--source dbscan / lightning)
├── opera_rainfall_fraction.py         # Per-source OPERA 5-class pixel fractions; prior for WeightedFocalCategoricalCrossentropy
├── data_statistics.py                 # Generate diagnostic plots from patch/sequence data
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Data Products

### Resolution Categories

| Category | Grid Resolution | Patch Size | Pooling | Products |
|----------|----------------|------------|---------|----------|
| HR (1 km) | 1536×768 | 256×256 | None | Radar (RZC, BZC, CZC, EZC-20, LZC, CPCH), Lightning (density, current, occurrence), MTG vis_06, OPERA rainfall_rate (HR alias for the label head) |
| LR (2 km) | 1536×768 | 128×128 | 2×2 avg | MTG IR/WV (ir_38, ir_105, wv_63, wv_73), OPERA (reflectivity, rainfall_rate) |
| LR (3 km) | 1536×768 | 64×64 | 4×4 avg | MSG (VIS006, IR_039, IR_108, WV_062, WV_073) |

### Handling multi-resolution inputs

The pipeline mixes radar (~1 km), lightning (~1 km), MTG vis (1 km), MTG IR/WV (2 km), OPERA (2 km), and MSG (3 km). Two design choices reconcile them without lying about each product's information content:

**1. Reproject every product to the same 1 km Romania grid first.** `reproject.py` runs a precomputed pyresample KD-tree mapping from each product's native CRS into the shared 1536×768 EPSG:31700 (Stereo70) canvas. After this step every `.npy` is on the *same grid* and the same patch-number maps to the same geographic tile across all products. This makes spatial alignment trivial downstream — the cost of cross-scale comparison is paid once per cadence step, not at training time.

**2. Pool *down* to native resolution before the model sees the data, not up.** `extract_patches.py` slices each reprojected product into 256×256 1 km HR tiles, then average-pools to:
- **128×128 (MR)** for products natively at ~2 km (MTG IR/WV, OPERA)
- **64×64 (LR)** for products natively at ~3 km (MSG)

The HR (1 km) products keep their 256×256 tiles unchanged.

Why downsample instead of bilinearly upsampling the 2 km / 3 km inputs to 1 km? Upsampling fabricates pixels — the model would see "1 km MSG IR" with no genuine 1 km information and would inevitably overfit interpolation artifacts. Downsampling does the opposite: it preserves the *real* spatial resolution each instrument actually measured, so the encoder is forced to cross-scale honestly rather than hallucinating detail.

**3. Multi-branch encoder, merges at matching scales.** [`train_models.build_coalition_model`](train_models.py) wires one input per resolution bucket (`past_hr` 256×256, `past_mr` 128×128, optionally `past_lr` 64×64) and walks each through ResBlock + ConvGRU stages of [32, 64, 128] channels. After the HR branch's first stride-2 ResBlock it lands at 128×128 and gets concatenated with the MR branch; after the second it lands at 64×64 and merges with the LR branch (if present). The decoder then upsamples a single fused state back to 256×256 over three lead times.

```
INPUT  past_hr (256×256, 1 km) ─┐
                                ├─ ResBlock + ConvGRU ─stride 2─┐
                                │                                ▼
       past_mr (128×128, 2 km) ─┴────── concat ─────────────────[merge @128]
                                                                 │
                                                                 ▼
                                          ResBlock + ConvGRU ─stride 2─┐
                                                                       ▼
       past_lr (64×64, 3 km) ────────── concat ─────────────────[merge @64]
                                                                       │
                                                                       ▼
                                                                  DECODER → T+15/30/45
```

**4. The model rebuilds itself from `metadata.json`.** Each dataset's `metadata.json` records the input group names + their shapes (`[T, H, W, C]`). `build_coalition_model` reads the metadata, computes each branch's downsample factor as `max_res / branch_res` (so 1 for HR, 2 for MR, 4 for LR), and wires the merges automatically. Adding a new resolution bucket needs no code change in the model — only `create_datasets.get_mode_config()` learns about it.

### Data Sources

| Source | Products | Native cadence | Native resolution | Role | Pipeline entry point |
|---|---|---|---|---|---|
| **ANM radar** (legacy) | RZC, BZC, CZC, EZC-20, LZC, CPCH | 10 min | ~1 km | Precipitation target + features (MSG/MTG experiments) | `our_data/raw_data/radar_arrange.py` |
| **LINET lightning** | density, current, occurrence | 10 min native; filter-aligned to `products.lightning.filter` | Native KML → 1 km grid (variable-width windows that cover every minute of the day) | Lightning target + features | `our_data/raw_data/lightning_arrange.py`, `our_data/lightning_data/read_kml_version2.py` |
| **MSG SEVIRI** *(disabled in active build)* | VIS006, IR_039, IR_108, WV_062, WV_073 | 15 min | 3 km | Satellite features (LR branch) | `our_data/satellite_data/pipeline_msg_mtg.py` |
| **MTG FCI L1C** | vis_06, ir_38, ir_105, wv_63, wv_73 | 10 min | 1 km (vis_06) / 2 km (IR/WV) | Satellite features (HR + MR branches) | `our_data/satellite_data/pipeline_msg_mtg.py` |
| **OPERA composite** | reflectivity (dBZ), rainfall_rate (mm/h) | 15 min | 2 km | Precipitation target + features for the **MTG+OPERA experiment** (replaces ANM radar) | `our_data/opera_data/pipeline_opera.py` |

### Satellite Channel Selection

5 channels per instrument, chosen as spectral equivalents for fair MSG vs MTG comparison:

| Physical property | MSG SEVIRI | MTG FCI |
|---|---|---|
| Cloud optical thickness (VIS) | VIS006 (0.6 µm) | vis_06 |
| Cloud phase discrimination (SWIR) | IR_039 (3.9 µm) | ir_38 |
| Cloud top temperature (TIR) | IR_108 (10.8 µm) | ir_105 |
| Upper-tropospheric moisture (WV) | WV_062 (6.2 µm) | wv_63 |
| Mid-tropospheric moisture (WV) | WV_073 (7.3 µm) | wv_73 |

### Grid Definition

Romania grid on EPSG:31700 (Stereo70):
- **Dimensions**: 1536×768 pixels (~1 km resolution)
- **Patch layout**: 6 columns × 3 rows = 18 patches of 256×256
- **Area extent**: (-177324, 77148, 1331353, 723370) meters

## Pipeline

### Full Pipeline (run in order)

All scripts are run from the project root (`coalition4-rcnn/`):

```bash
cd F:\nowcasting\coalition4-rcnn
conda activate tfenv
```

#### Step 0 — Set the training cadence

The pipeline supports any training step that is at least as coarse as the
highest native data cadence. Native cadences are declared in
[`product_cadences.config`](product_cadences.config) at the project root and
cover every product family the pipeline knows about (radar, MTG,
the two OPERA products, and lightning).

**Lightning** is special: it has no native scan cadence — LINET strokes
are individual observed events, not raster scans. The lightning maps that
feed the model are produced by binning strokes into windows whose width
mirrors whichever paired product the experiment uses:

- When **radar** is in the configuration, lightning bins align to the
  radar cadence (`step_minutes` resolved against the radar `filter` set).
- When **OPERA** replaces radar, lightning bins align to the OPERA
  cadence — although the OPERA experiment can be run without lightning
  entirely if the goal is OPERA vs MTG only.
- In any other combination the validator picks the most common cadence of
  the active products and aligns lightning bins to it.

OPERA composites are scanned with the 15-min product family (NIMBUS,
CIRRUS, ODYSSEY) — see the *OPERA composite acquisition window* note
just below for the exact `[NT-X, NT+Y]` data windows each composite type
uses, and why the 10-min products are filtered to the alternating
`{:00, :10, :30, :40}` pattern when paired with a 15-min training step.

If you add a new data source or its native cadence changes, edit
`product_cadences.config` rather than the validator script. The file is
INI-style with a single `[cadences]` section; comments start with `#`
or `;`, and a value of `null` or an empty value marks a continuous /
event-based product (lightning). The validator reads the file at
startup; it does not inspect any data folders. Override the path with
`--cadences_file path/to/other.config`. Comment out a product line
with a leading `#` to drop it from the active set — e.g. comment out
OPERA's two entries to bring the validator floor back to 10 min when
running a radar-only experiment.

Run the validator once before any other pipeline step:

```bash
python validate_timestep.py --step_minutes 15        # 15-min cadence (required when OPERA is in the mix)
python validate_timestep.py --step_minutes 10        # 10-min cadence (radar/MTG only; OPERA must be commented out)
python validate_timestep.py --step_minutes 30        # 30-min cadence (any product set)
python validate_timestep.py --step_minutes 5         # ERROR (below 10-min native)
python validate_timestep.py --print                  # show current config
```

##### Why the alternating 10–20–10–20 filter for 10-min products at a 15-min step?

OPERA composites are produced over fixed temporal windows centred on the
quarter-hour. The data-time-window contract from EUMETNET differs by
composite generation:

- **NIMBUS** (15-min composites): data window is `[NT − 12 min, NT + 7 min]`
  around each nominal time `HH:00, HH:15, HH:30, HH:45`. NIMBUS takes the
  input scan closest to NT.
- **CIRRUS**: temporal coverage `[NT − 10 min, NT]` where NT is the
  composite's Nominal Time. Updated every 5 minutes (288× per day).
- **Old ODYSSEY** (legacy): 15-min scanning interval `[NT − 10 min,
  NT + 5 min]`.

In all three cases the OPERA nominal grid is the strict `{:00, :15, :30,
:45}` set, and the 10-minute products that feed the same training sample
(radar, MTG) need to land within roughly ±5 min of those nominal
times so the composite's temporal window matches the satellite/radar
acquisition. The minute filter `{:00, :10, :30, :40}` is exactly that
choice — at every 15-min step, it picks the 10-min slot whose acquisition
is closest to the OPERA nominal time, giving an alternating 10–20–10–20
spacing between consecutive samples. The `extract_patches.py` cadence
snap (see Step 3) translates between the two grids automatically.

The script writes `our_data/timestep_config.json` with the chosen
`step_minutes`, the per-product `minute_filter` (the native minutes to keep
when arranging or downloading), the `steps_per_day`, and a
`cadences_source` pointer back to the JSON file it loaded. All downstream
scripts (`radar_arrange.py`, `pipeline_msg_mtg.py`,
`pipeline_opera.py`, `read_kml_version2.py`, `identify_patches.py`,
`identify_lightning_periods.py`, `extract_patch_seq_for_datasets.py`,
`create_datasets.py`) **read this file** and refuse to run if it is missing.

For step=15 with 10-min radar/MTG the resulting filter is `{00, 10, 30, 40}`,
giving an alternating 10–20–10–20 spacing between consecutive samples (no optical
flow interpolation is used).

> **MSG**: the MSG SEVIRI ingestion path is currently disabled in
> `pipeline_msg_mtg.py` and `create_datasets.py`. Only MTG FCI L1C is supported.

#### Step 1 — Identify convective patches from RZC radar

Runs DBSCAN on RZC rain rate data at 15-minute resolution. Produces a patch index mapping each timestamp to the active patches (1–18) on the fixed 6×3 grid.

```bash
python identify_patches.py
python identify_patches.py --date 2025-05-15 --plot    # single date with diagnostic plots

# OPERA-driven DBSCAN (uses pre-reprojected opera_rainfall_rate; same 10 mm/h threshold as RZC)
python identify_patches.py --source opera
python identify_patches.py --source opera --start 2026-03-13 --end 2026-05-11
```

Output: `our_data/patch_index/patch_index.csv` and `patch_index.json`. The `--source opera` flag reads from `reprojected_data/opera_data/rainfall_rate/` instead of reprojection RZC on the fly; OPERA-source runs require `reproject.py --opera` to have been run first.

#### Step 2 — Reproject all products to the Romania grid

Regrids radar, satellite (MSG/MTG), lightning, **and OPERA** to the 1536×768 grid. Uses precomputed KD-tree mappings (built once per source geometry) and parallel day-folder processing for speed.

```bash
python reproject.py --all                          # all products
python reproject.py --radar                        # radar only
python reproject.py --satellite MSG                # MSG channels only
python reproject.py --satellite MTG                # MTG channels only
python reproject.py --lightning                    # lightning (cache as .npy)
python reproject.py --opera                        # OPERA radar (HDF5 → .npy)
python reproject.py --all --workers 8              # 8 parallel workers
python reproject.py --radar --date 2025-05-15      # single date
```

**Every product family writes `.npy`** under `our_data/reprojected_data/...`. The shared Romania-grid lat/lon arrays are written once at `our_data/reprojected_data/romania_grid_{lats,lons}.npy`, and each non-trivial source (MTG, OPERA) drops a sidecar `*_constants.json` with the source projection so an `.npy` can be re-attached to its projection at any later step (e.g. by `inspect_mtg.py --reprojected`).

#### Step 3 — Extract 256×256 patches

> **Order of operations:** this step now runs **after Step 4.1** (`extract_patch_seq_for_datasets.py`). The split CSVs that step produces (with the manifest gate baked in) are the exact list of (date, time) pairs extract_patches needs to walk — driving off them avoids the previous patch_index + manifest filtering loop and guarantees the saved patches stay in sync with what training will load.

Reads the per-source patch-activity index (`patch_index.csv` for `--source dbscan`, `lightning_patches.csv` for `--source lightning`) and the cached reprojected data, walks the **union of (date, time) tuples across `train/val/test_data_<source>.csv`**, slices the active patches at each step, applies resolution-dependent pooling (none for HR, 2×2 for MTG LR, 4×4 for MSG LR), and saves stacked `.npy` files.

`timestep_config.json` is still read for the per-product cadence snap (e.g. OPERA's 15-min grid vs MTG's `:00, :10, :30, :40`). No separate manifest read is needed because the split CSVs already incorporate that gate (Step 4.1 enforces it).

```bash
python extract_patches.py --source dbscan
python extract_patches.py --source lightning
python extract_patches.py --source dbscan --date 2025-05-15
python extract_patches.py --source dbscan --products satellite_MTG opera lightning
```

Output: `our_data/patches/{date}/{variable}_{HHMM}_{HR|LR}.npy` (shape `(num_active_patches, H, W)`; the per-patch order at each timestep matches whichever index file `--source` selected, so the `idx_t*` columns in the split CSVs address the right slice).

#### Step 4.1 — Extract temporally continuous sequences and split dataset

Analyzes the patch index to find patches with uninterrupted activity over a 6-step window (2 past + current + 3 future, 90 minutes total). Produces per-timestep npy indices accounting for index shifts when the active patch set changes.

Dataset splitting follows Czibula et al. (2024): each day is divided into equal temporal blocks (default 6h). Within each block, qualifying sequences are ordered chronologically — the first 10% go to test, next 10% to validation, remaining 80% to training. This ensures all three splits sample from the same diurnal distribution, avoiding hour-based bias.

**Manifest gate (final cross-product filter).** Step 4.1 auto-discovers `our_data/timestep_manifest.csv` (from Step 4.2 below) and intersects the patch index with it before searching for sequences. This is the final filter that decides which `(date, time)` tuples can become training samples: a slot only survives if every product in the intersect set had data there. The intersection is reported on stdout with a top-N "dates by drop count" table, and every dropped `(date, time)` is written to `our_data/extract_patch_seq_drops_<source>.csv` for audit. Pass `--manifest none` to disable the gate (debug only).

```bash
python extract_patch_seq_for_datasets.py --source dbscan            # DBSCAN track (reads patch_index.csv)
python extract_patch_seq_for_datasets.py --source lightning         # Lightning track (reads lightning_patches.csv)
python extract_patch_seq_for_datasets.py --source dbscan --block_hours 4           # 4h blocks (finer diurnal balance)
python extract_patch_seq_for_datasets.py --source dbscan --test_frac 0.15 --val_frac 0.15  # 15/15/70 split
python extract_patch_seq_for_datasets.py --source dbscan --manifest none           # Skip the manifest gate
```

Outputs (suffixed by source so the two tracks coexist on disk):

- `our_data/train_data_<source>.csv`
- `our_data/validation_data_<source>.csv`
- `our_data/test_data_<source>.csv`
- `our_data/sequence_meta_<source>.json` — source, effective step, past/future window length
- `our_data/extract_patch_seq_drops_<source>.csv` — every (date, time) the manifest gate removed

##### Lightning-source sequences

For the lightning training pipeline, sample selection is driven by **lightning activity in time** rather than by radar-DBSCAN convective clusters in space. `--source lightning` reads `lightning_patches.csv` produced by [`identify_lightning_periods.py`](#lightning-periods-occurrence-fraction-filter) (see below). The CSV format, Czibula splitting, and step-column naming are identical; only the upstream activity index and the step interval change.

```bash
# 1. Run summarize → emits lightning_active_steps.csv at project root
python our_data/lightning_data/summarize_lightning_data.py

# 2. Run the occurrence-fraction filter (produces our_data/lightning_periods/)
python identify_lightning_periods.py

# 3. Build lightning-driven sequences (uses step_minutes from timestep_config.json)
python extract_patch_seq_for_datasets.py --source lightning
```

#### Lightning periods (occurrence-fraction filter)

`identify_lightning_periods.py` produces the lightning-activity index that drives `extract_patch_seq_for_datasets.py --source lightning`. Lightning is much sparser than radar, so the model is trained on a separate sample list filtered to lightning-active windows only.

**Two upstream files at project root**, both produced by `our_data/lightning_data/summarize_lightning_data.py`:

| File | Purpose |
|---|---|
| `lightning_summary.csv` | Per-date coverage report keyed against `products.lightning.filter` from `timestep_config.json`. Same shape as `opera_summary.csv` — consumed by `intersect_product_coverage.py` via `--summary lightning=...`. |
| `lightning_active_steps.csv` | Per-`(date, HH:MM)` activity flags for density / current / occurrence. Consumed by both `intersect_product_coverage.py --active lightning=...` (as the cross-product gate) and `identify_lightning_periods.py` (as the candidate set for the fraction threshold). |

**Filter logic.** `identify_lightning_periods.py` walks the master grid at `step_minutes`, snaps each master HHMM to the lightning filter to find the matching `.npy` file, computes the per-map occurrence-fraction `nonzero_pixels / total_pixels`, then keeps only `(date, HH:MM)` where `fraction ≥ ratio × mean(fraction over active occurrence maps)`. **No temporal aggregation** — each surviving map is one row in the output CSV. The per-patch step then marks any 256×256 patch with at least one non-zero pixel as active.

| Knob | Default flag | Default value | Effect |
|---|---|---|---|
| **Fraction threshold ratio** | `--fraction_threshold_ratio` | `0.30` | Keep maps whose occurrence-fraction is at least this fraction of the mean fraction over the active set. |
| Active CSV path | `--active_csv` | `lightning_active_steps.csv` at project root | Source of candidate `(date, HH:MM)` pairs. |
| Lightning data root | `--lightning_dir` | `our_data/lightning_data` | Where to find the `occurrence/...npy` files. |
| Output dir | `--output_dir` | `our_data/lightning_periods` | Where to write the index. |

```bash
# All defaults: reads lightning_active_steps.csv, threshold = 0.30 × mean.
python identify_lightning_periods.py

# Stricter threshold (keep only above-average maps).
python identify_lightning_periods.py --fraction_threshold_ratio 1.0
```

Outputs in `our_data/lightning_periods/`:

| File | Purpose |
|---|---|
| `lightning_periods_config.json` | All CLI parameters used + computed mean / threshold / per-stage counts (reproducibility) |
| `lightning_patches.csv` | Per-map per-patch index keyed by **master HHMM** so it lines up with the same grid every other product walks. Schema mirrors `patch_index.csv` — `extract_patch_seq_for_datasets.py --source lightning` consumes it directly. |

##### Diagnostics

```bash
# Per-day + per-timestep bar charts (reads lightning_active_steps.csv)
python our_data/lightning_data/visualize_lightning_stats.py \
    --output_dir our_data/lightning_data

# Per-source training-scope non-zero pixel fraction. Reads
# train_data_<source>.csv and writes lightning_fraction_<source>.json,
# which train_models.py loads for the focal-loss prior on any mode with
# label_type='lightning' (mtg_lightning, mtg_lightning_opera_occurrence).
python lightning_fraction.py --source dbscan
python lightning_fraction.py --source lightning
# Broader scope (skip the train-split filter):
python lightning_fraction.py --source dbscan --scope_csv lightning_active_steps.csv
```

#### Step 4.2 — Intersect per-product timestep coverage

`intersect_product_coverage.py` computes the per-timestep intersection of available data across the chosen product set and writes a manifest that Step 4.1 (`extract_patch_seq_for_datasets.py`, the final filter) consumes directly. `extract_patches.py` does **not** read the manifest itself any more — it walks the train/val/test CSVs produced by Step 4.1, which already incorporate the manifest gate.

- **Active products** are determined by which `--summary KEY=PATH` flags you pass. Lightning is included only when `--summary lightning=...` is supplied.
- **Per-timestep availability** is sourced from one of two gates per product:
    - **Missing-JSON gate** (default for MTG / OPERA): each summarizer's companion `<key>_missing_timesteps.json` (auto-discovered next to the summary CSV; overridable per product with `--missing KEY=PATH`). A slot survives iff its snapped HHMM is **not** in the missing set.
    - **Active-CSV gate** (opt-in via `--active KEY=PATH`): used for **lightning** — `lightning_active_steps.csv` carries per-`(date, HH:MM)` activity flags. A slot survives iff its snapped HHMM **is** in the active set (any of the flag columns == 1). When `--active` is given for a product, the missing-JSON gate is skipped for that product entirely.
- **Per-product cadences** come from `timestep_config.json` (Step 0). For each master-grid HHMM the script snaps to the nearest minute in each product's filter, then checks the per-product gate.
- **Error logs** (`--errors_log PATH`, repeatable): consumed in the same format whether you pass the per-category `reproject_<category>.log` files or the single aggregated `errors.txt` (which `reproject.py` now writes automatically at the end of every category run).
- Train / val / test CSVs are **not** touched — they keep whatever `extract_patch_seq_for_datasets.py` wrote.

```bash
# OPERA / radar track (no lightning gating):
python intersect_product_coverage.py \
    --summary mtg=mtg_summary.csv \
    --summary opera=opera_summary.csv \
    --errors_log our_data/reprojected_data/errors.txt

# Lightning track (lightning as the activity gate):
python intersect_product_coverage.py \
    --summary  mtg=mtg_summary.csv \
    --summary  opera=opera_summary.csv \
    --summary  lightning=lightning_summary.csv \
    --active   lightning=lightning_active_steps.csv \
    --errors_log our_data/reprojected_data/errors.txt
```

| Flag | Purpose |
|---|---|
| `--summary KEY=PATH` (repeatable, **required**) | One per active product. `KEY ∈ {radar, mtg, opera, lightning}`. `PATH` is the per-product summary CSV; the script reads its date list. |
| `--missing KEY=PATH` (repeatable, optional) | Override the auto-discovered missing-timesteps JSON. Mutually exclusive with `--active` for the same product. |
| `--active KEY=PATH` (repeatable, optional) | Replace the missing-JSON gate with an active-steps CSV (`date,time_utc,<flag1>,<flag2>,...`). Slot survives iff snapped HHMM is in the active set. |
| `--errors_log PATH` (repeatable, optional) | A `reproject_<category>.log` or the aggregated `errors.txt` from `reproject.py`. Any `(date, HHMM)` parsed from these logs is removed from the kept set. |
| `--timestep_config PATH` | Override the location of `timestep_config.json`. |
| `--output_csv`, `--output_plot` | Override the default output paths. |

**Outputs (only two):**

| File | Purpose |
|---|---|
| `our_data/timestep_manifest.csv` | One row per surviving `(date, HHMM)`; columns also include the per-product snapped HHMM each product loaded so the manifest doubles as an audit trail. Consumed by `extract_patch_seq_for_datasets.py` (Step 4.1) as the final cross-product gate before the split CSVs are written. |
| `our_data/intersect_summary.png` | Per-date stacked bar: kept timesteps + drops attributed to each product / error log. Shows the quantitative impact of the intersection. |

> **Why this comes before Step 4.3**: the normalization stats are computed on the surviving timesteps, so they never see slots that would later be discarded for missing inputs — fewer file reads and stats that match the eventual training distribution.

#### Step 4.3 — Compute normalization statistics

`compute_normalization_stats.py` derives per-variable mean / std from the reprojected data so the model trains on values centred for the **Romanian** distribution, not the Swiss one. This step is **mandatory** before Step 5: `create_datasets.py` no longer falls back to the Leinonen Table A1 constants — if `normalization_stats_<source>.json` is missing, the run fails with an explicit pointer back to this script.

Stats are now **per-source**. The DBSCAN-driven and lightning-driven tracks have different training distributions (different sets of `(date, time)` survive each filter chain), so each writes its own `normalization_stats_<source>.json` and `create_datasets.py --source <source>` reads the matching one. The inputs (`train_data_<source>.csv`, `sequence_meta_<source>.json`) are also auto-resolved from `--source`.

```bash
# DBSCAN track (reads train_data_dbscan.csv + sequence_meta_dbscan.json,
# writes normalization_stats_dbscan.json)
python compute_normalization_stats.py --source dbscan

# Lightning track (reads train_data_lightning.csv + sequence_meta_lightning.json,
# writes normalization_stats_lightning.json)
python compute_normalization_stats.py --source lightning

# Subset of variables (faster iteration while tuning)
python compute_normalization_stats.py --source dbscan --variables RZC ir_105

# Also surface p01 / p50 / p99 + MAD via reservoir sampling
python compute_normalization_stats.py --source dbscan --with_percentiles

# Disable training-window filter (DIAGNOSTIC ONLY — leaks val/test data)
python compute_normalization_stats.py --source dbscan --no_split_filter
```

**Policy decisions** (recorded inside the JSON for traceability):

| Decision | Choice | Why |
|---|---|---|
| Sample scope | training-set only, expanded across each row's past + current + future window | Stats from val / test would leak distributional info into the model |
| Spatial scope | single scalar mean / std per variable | Per-pixel climatology would overfit to training-domain geography (e.g. permanent radar beam blockage) |
| Source data | `our_data/reprojected_data/` (full 1536 × 768 grids) | The pre-built patches in `our_data/patches/` are filtered by RZC / lightning activity — computing stats on them would bias every variable's distribution toward convective scenes |
| Missing values | NaN and per-variable "missing sentinel" pixels are **dropped** from the Welford accumulator | Replacing them with `fill` (as the inference-time transforms still do) would silently drag the mean toward the fill value |
| Pre-norm for heavy-tailed | `log10` after clipping to a positive floor | RZC, LZC, lightning density / current, `cmic_cot`, `opera_rainfall_rate` are zero-inflated with long right tails — z-scoring them directly would compress the bulk into a tiny range |
| Near-constant flag | `std < 1e-3 · |mean|` → flagged in JSON | Standardising near-constant variables amplifies noise in the rare non-zero pixels; the consumer should consider clipping / robust scaling |

**Per-product normalization table** (also encoded in `NORMALIZATION_SPEC` inside `compute_normalization_stats.py`):

| Product / Variable | Source dir | Transform | Why |
|---|---|---|---|
| **Radar — RZC** (rain rate, mm/h) | `radar_data/` | clip 0.01 → `log10` → z-score | Zero-inflated, log-normal-like in the tail. The `0.01 mm/h` floor avoids `log10(0)` and matches the gauge detection limit. |
| **Radar — LZC** (liquid water content) | `radar_data/` | clip 0.5 → `log10` → z-score | Same family as RZC; the higher floor reflects the noise level of the radar-derived LWC product. |
| **Radar — CZC** (composite reflectivity, dBZ) | `radar_data/` | linear z-score | dBZ is already a logarithmic scale; the distribution is roughly Gaussian when signal is present. |
| **Radar — BZC** (base reflectivity, dBZ) | `radar_data/` | hardcoded `x / 100` | Stored as integer-encoded `dBZ * 100`; this divides back to physical units. No statistical centring needed. |
| **Radar — EZC-20** (echo top height) | `radar_data/` | hardcoded `x / 1.97` | Empirically chosen unit conversion (height/km divided by 1.97 maps to a [0, 1]-ish range). |
| **Radar — CPCH** (1 h precipitation) | `radar_data/` | pure `log10`, no z-score | Used as a sub-threshold mask (values < 0.1 mm/h are set to the fill). Already on a log scale; further centring would obscure the threshold semantics. |
| **MTG — `ir_38`** (3.8 µm, K) | `satellite_data/MTG/` | linear z-score | Brightness temperatures are approximately Gaussian; centring stabilises gradient magnitudes across channels. |
| **MTG — `ir_105`** (10.5 µm, K) | `satellite_data/MTG/` | linear z-score | Same reasoning as `ir_38`. |
| **MTG — `wv_63`** (6.3 µm water vapour, K) | `satellite_data/MTG/` | linear z-score | Water-vapour channels have narrower dynamic ranges than thermal IR; per-channel stats matter. |
| **MTG — `wv_73`** (7.3 µm water vapour, K) | `satellite_data/MTG/` | linear z-score | As above. |
| **MTG — `vis_06`** (0.6 µm reflectance, %) | `satellite_data/MTG/` | hardcoded `x / 100` | Source stores integer `reflectance × 100`; `/100` recovers `[0, 1]` already on a natural scale. No z-score. |
| **Lightning — density** (strokes/km²) | `lightning_data/` | clip 1e-4 → `log10` → z-score | Extremely heavy-tailed (most pixels zero, rare pixels with thousands). The 1e-4 floor preserves zero-class handling while allowing log. |
| **Lightning — current** (kA-weighted) | `lightning_data/` | clip 1e-8 → `log10` → z-score | Same as density, with a smaller floor matching the smallest measurable current. |
| **Lightning — occurrence** (binary) | `lightning_data/` | clip to {0, 1} only | Already on its natural scale; z-scoring would destroy the binary interpretation. |
| **OPERA — `opera_reflectivity`** (max reflectivity, dBZ) | `opera_data/` | linear z-score | Like CZC: dBZ is already logarithmic, Gaussian-ish where signal is present. |
| **OPERA — `opera_rainfall_rate`** (mm/h) | `opera_data/` | clip 0.01 → `log10` → z-score | Like RZC: heavy-tailed, zero-inflated. Same floor and transform family for consistency across both rain-rate sources. |

When `--with_percentiles` is passed, each variable block additionally carries `p01`, `p50`, `p99`, and `mad` (median absolute deviation) — useful for sanity-checking against the mean/std, and as robust alternatives if a variable is flagged `near_constant: true`.

#### Step 5 — Build TF datasets

Transforms patches using the **data-driven** mean / std from
`our_data/normalization_stats_<source>.json` (Step 4.3, matching the chosen `--source`) and saves as TFRecord shards for each `(mode, source)` pair. Each dataset split also saves a `metadata.json` containing `input_shapes`, `label_type`, `past_timesteps`, and `future_timesteps` — this metadata drives dynamic model construction in Step 6.

Active modes: `mtg_lightning`, `mtg_radar`, `mtg_radar_continuous`, the OPERA-driven `mtg_opera_radar_only` / `mtg_opera_mtgmr`, and the dual-target full-input pair `mtg_lightning_opera` (OPERA rainfall label) / `mtg_lightning_opera_occurrence` (lightning binary label). The MSG modes (`msg_lightning`, `msg_radar`, `msg_radar_continuous`) are commented out in `get_mode_config()` — re-enable in source if you need them.

The new `--source {dbscan, lightning}` flag selects which `sequence_meta_<source>.json` + `{train,validation,test}_data_<source>.csv` triplet to read and lands the output dataset at `our_data/datasets/<mode>_<source>/`. Mode and source are independent: a single mode can be built once per source so the lightning- and DBSCAN-driven tracks have separate dataset directories.

```bash
# DBSCAN-driven sample selection (uses patch_index.csv produced by
# identify_patches.py, whether that script was run with --source radar
# or --source opera).
python create_datasets.py --mode mtg_lightning        --source dbscan
python create_datasets.py --mode mtg_radar            --source dbscan
python create_datasets.py --mode mtg_radar_continuous --source dbscan

# OPERA-driven precipitation modes
python create_datasets.py --mode mtg_opera_radar_only --source dbscan
python create_datasets.py --mode mtg_opera_mtgmr      --source dbscan

# Lightning-driven sample selection (same modes can be rebuilt here)
python create_datasets.py --mode mtg_lightning        --source lightning
python create_datasets.py --mode mtg_opera_mtgmr      --source lightning
```

The OPERA modes replace radar (RZC and friends) with OPERA `opera_reflectivity` + `opera_rainfall_rate` in the MR branch (2 km, `pool=2`) and use `opera_rainfall_rate_hr` (HR alias of the same reprojected file) as the 5-class multi-class label (same bin edges as RZC: `<10`, `10–20`, `20–30`, `30–40`, `≥40 mm/h`):

| Mode | HR | MR | LR | Label |
|---|---|---|---|---|
| `mtg_opera_radar_only` | MTG `vis_06` | OPERA | — | OPERA rainfall 5-class |
| `mtg_opera_mtgmr` | MTG `vis_06` | OPERA + MTG IR/WV | — | OPERA rainfall 5-class |
| `mtg_lightning_opera` | lightning + MTG `vis_06` | OPERA + MTG IR/WV | — | OPERA rainfall 5-class |
| `mtg_lightning_opera_occurrence` | lightning + MTG `vis_06` | OPERA + MTG IR/WV | — | lightning binary occurrence |

`mtg_lightning_opera` and `mtg_lightning_opera_occurrence` share the **same input stack** but differ on the label head — same OPERA-driven sample selection (`--source dbscan`), same channels in HR and MR, but one predicts OPERA rainfall while the other predicts whether lightning will fire. They're the natural dual-target pair to train side-by-side and feed into the domain-adaptation Swin head.

Custom modes can be defined by adding a new configuration in `create_datasets.py`. The training script requires no code changes — it reads whatever inputs are in the dataset.

Output: `our_data/datasets/{mode}_{source}/train|validation|test/` (each with `metadata.json`)

##### When `--mode` is required vs. optional

The `--mode` argument behaves differently in each of the three pipeline scripts. **Only `train_models.py` is truly dynamic** — the dataset name there is just a label. The other two need a name that matches a hardcoded recipe.

| Script | `--mode` | Resolved as | If you want a new input combination |
|---|---|---|---|
| `create_datasets.py` | **required**, restricted to the names in `get_mode_config()` | Picks the HR / MR / LR variable recipe + label transform | Add a branch in `get_mode_config()` — there is no CLI alternative (the per-variable transforms exist as a registry, but the group composition is hardcoded) |
| `train_models.py` | **required**, but a free-form string | Used only for the saved-model folder name and the default dataset path `{data_root}/datasets/{mode}` | Pass any string. Combine with `--dataset_dir` to point at a dataset that doesn't follow the `{mode}` naming convention |
| `evaluate_coalition.py` | **required**, restricted to a `choices=[...]` list | Used for the eval output folder name and the dataset path | Add the new name to the `choices=[...]` list in `main()` |

So **the only "skippable" use** of `--mode` is on `train_models.py` when paired with `--dataset_dir`:

```bash
# Re-train any saved dataset under whatever model label you want
python train_models.py --mode my_label \
    --dataset_dir our_data/datasets/mtg_opera_full
```

For `create_datasets.py` and `evaluate_coalition.py`, the mode name **must** exist in code — adding a new product combination still requires a one-line edit in each. `train_models.py` is unaffected: it reads `metadata.json` and adapts to whatever inputs the dataset declares.

#### Step 6 — Train

Builds the COALITION recurrent-convolutional architecture (ResBlock + ConvGRU encoder-decoder) dynamically from the dataset's `metadata.json`. The model architecture adapts automatically to whatever inputs are present — number of input groups, channel counts, and resolutions are all read from metadata rather than hardcoded. This means training with different input configurations (MSG vs MTG, different precipitation targets) requires no code changes; only the dataset needs to change.

Hyperparameters live in [`training.config`](training.config) (epochs, batch size, dropout, cosine-warmup LR schedule, early stopping, checkpointing, the `[finetune]` section for the Swin head, and the `[radar_loss]` section for the OPERA multiclass focal/class-weighted loss). Per-mode overrides go under `[mode.<name>]`.

##### `--stage` — base training vs. domain-adaptation fine-tune

`train_models.py` runs in one of three stages, selected by `--stage`:

| Stage | What it does | Optimizer | Saves |
|---|---|---|---|
| `base` (default) | Standard COALITION-4 training from scratch. | Adam (LR from `[lr_schedule]`) | `coalition_<mode>_<source>.keras` |
| `finetune` | Loads `--base_checkpoint`, freezes the encoder-forecaster, grafts a 2-block Swin transformer head with 8×8 windowed attention + 3 per-lead-time projection heads, fine-tunes only the head. | AdamW (LR + weight_decay from `[finetune]`) | `coalition_<mode>_<source>_finetuned.keras` |
| `both` | Runs `base` then `finetune` back-to-back in the same Python process. The just-saved base model is used as the frozen backbone — no separate `--base_checkpoint` flag needed. | base uses Adam, finetune uses AdamW | both `.keras` files above |

The Swin head sits on the named `backbone_output` layer (the decoder's final feature tensor, shape `(B, F=3, H, W, C_deep)`). It collapses the future-time axis into shared spatial features, runs the Swin blocks, then projects to 3 independent lead-time outputs that stack back onto axis=1 — same `(B, F, H, W, num_outputs)` contract as the base model, so loss and metrics carry over unchanged.

```bash
# A. Simple — base only (the standard model, single run)
python train_models.py \
    --config training.config \
    --mode mtg_opera_mtgmr \
    --source dbscan \
    --stage base

# B. Domain adaptation — base + Swin head fine-tune in one process
python train_models.py \
    --config training.config \
    --mode mtg_opera_mtgmr \
    --source dbscan \
    --stage both

# C. Resume / fine-tune a previously-saved base
python train_models.py \
    --config training.config \
    --mode mtg_opera_mtgmr \
    --source dbscan \
    --stage finetune \
    --base_checkpoint models/coalition_mtg_opera_mtgmr_dbscan.keras

# Train every mode listed in [modes].run, both stages, lightning source
python train_models.py --config training.config --source lightning --stage both

# Override the default dataset path (debug only; --stage base only)
python train_models.py \
    --config training.config \
    --mode mtg_opera_mtgmr --source dbscan --stage base \
    --dataset_dir our_data/datasets/mtg_opera_mtgmr_dbscan
```

Outputs under `./models/` (`run_tag = <mode>_<source>`):

```
checkpoints/<run_tag>_latest.keras                # resumable per-epoch base checkpoint
checkpoints/<run_tag>_latest.json
checkpoints/<run_tag>_finetune_latest.keras       # resumable per-epoch finetune checkpoint
checkpoints/<run_tag>_finetune_latest.json
coalition_<run_tag>.keras                         # base model
coalition_<run_tag>_finetuned.keras               # fine-tuned (Swin head) model
history_<run_tag>.json
history_<run_tag>_finetuned.json
```

The per-epoch checkpoint is the run's safety net for the occasional CUDA crash — at most one epoch of work is lost. Pass `--fresh` to ignore an existing checkpoint and start over. The two tracks (`--source dbscan` vs `--source lightning`) have completely separate checkpoint paths so they can train in parallel on different machines without colliding.

#### Step 7 — Evaluate

Loads the trained model, runs evaluation on the test set, and generates diagnostic plots. Both `--source` (sample-selection track) and `--finetuned` (Swin-head model) are wired through every on-disk path so the four artefact combinations (`{base, finetuned} × {dbscan, lightning}`) never collide.

```bash
# Base model, OPERA-driven sample selection
python evaluate_coalition.py --mode mtg_lightning_opera_occurrence --source dbscan

# Swin-head fine-tuned variant of the same run
python evaluate_coalition.py --mode mtg_lightning_opera_occurrence --source dbscan --finetuned

# Lightning-driven sample selection, base model
python evaluate_coalition.py --mode mtg_lightning --source lightning
```

For OPERA multiclass modes (`mtg_opera_radar_only`, `mtg_opera_mtgmr`, `mtg_lightning_opera`), the radar branch now emits aggregate per-class precision / recall / F1 / CSI plus macro-F1, macro-CSI, balanced accuracy on top of the existing accuracy and confusion-matrix outputs. Two extra plots ride along: `csi_per_class.png` and `macro_summary_per_leadtime.png` so the dominance of class 0 (`R<10`) is visually obvious next to the balanced numbers.

Output: `evaluation/eval_<mode>_<source>[_finetuned]/` (plots + `evaluation_results.json`).

##### Fine-tune evaluation path

`--finetuned` does not call `tf.keras.models.load_model` on the saved `.keras`. Saved fine-tuned models contain the backbone as a sub-Model, and Keras 2.10's deserializer rebuilds the variable list in a different order than `save_model` wrote it, so the assigns fail with shape mismatches. Instead the script rebuilds the architecture from scratch with `train_models.build_finetune_model` and calls `model.load_weights(...)` which matches by name. Swin hyperparameters get recovered from `history_<run_tag>_finetuned.json`, so the rebuild matches what training produced. The base `coalition_<run_tag>.keras` must sit alongside the fine-tuned file in `--model_dir`; `--stage both` keeps them together automatically.

##### Per-leadtime CSV bundle for Shapley

`bundle_eval_scores.py` reads each mode's `evaluation_results.json` and writes the per-leadtime CSVs `feature_importance_analysis.py --methods classical_shapley` expects. Same `--source` / `--finetuned` flags as `evaluate_coalition.py`, plus a `--mode MODE=LETTERS` flag to override the default coalition pairing.

```bash
# Default OPERA coalition (mtg_opera_radar_only = o, mtg_opera_mtgmr = om)
python bundle_eval_scores.py --source dbscan

# Custom coalition for the lightning study
python bundle_eval_scores.py --source dbscan --prefix lightning \
    --mode "mtg_lightning_opera_occurrence=l" \
    --mode "mtg_lightning_opera=or"
```

##### Full-domain GT vs Pred visualisation

`visualize_full_domain_predictions.py` is a richer companion to `evaluate_coalition.py`'s per-patch plots. For every top-N reference timestep (by qualifying-patch count in the chosen CSV), it builds full 768×1536 Romania-canvas GT and prediction maps for all three lead times, in a single batched `model.predict(...)` call, with everything in memory (no disk writes). The plot is centred on Romania with neighbour-country borders, all 18 patch slots are outlined with a dashed grid, and a second figure zooms into the patch with the most GT activity per timestep.

```bash
# OPERA multiclass base model
python visualize_full_domain_predictions.py \
    --csv our_data/test_data_dbscan.csv \
    --mode mtg_lightning_opera \
    --source dbscan --top_n 3

# Lightning occurrence fine-tuned model; threshold from evaluation_results.json
python visualize_full_domain_predictions.py \
    --csv our_data/test_data_dbscan.csv \
    --mode mtg_lightning_opera_occurrence \
    --source dbscan --top_n 3 --finetuned
```

Output: `full_domain_plots/full_domain_<run_tag>[_finetuned]/ts<NN>_<date>_<HHMM>.png` plus a `..._zoom_p<NN>.png` per timestep. With cartopy installed, neighbour-country borders draw from Natural Earth's 10m `admin_0_countries` shapefile; without it the script falls back to a coarse hardcoded Romania polygon and prints `Border src: hardcoded_coarse` in the startup banner.

### Utility Scripts

```bash
# Training-scope non-zero pixel fraction for the LIGHTNING binary head.
# Defaults to train_data_<source>.csv and writes
# lightning_fraction_<source>.json so the prior matches what the model
# sees. Run once per --source you plan to train; train_models.py
# auto-resolves the matching JSON.
python lightning_fraction.py --source dbscan
python lightning_fraction.py --source lightning
# Broader scope: --scope_csv lightning_active_steps.csv. Legacy
# everything-on-disk: --scope_csv none.

# Training-scope per-class pixel fractions for the RADAR / OPERA
# multiclass head. Writes opera_rainfall_fraction_<source>.json which
# WeightedFocalCategoricalCrossentropy reads to build its alpha_k
# class weights. Required when [radar_loss].weighting != none in
# training.config; ignored when the configured radar loss is plain CCE.
python opera_rainfall_fraction.py --source dbscan
python opera_rainfall_fraction.py --source lightning

# Generate dataset diagnostic plots (6 panels: diurnal cycle, spatial
# heatmap, daily timeline, simultaneously-active patches, samples per
# date, patch survival). --source / --split auto-resolve the sequence
# CSV (defaults: dbscan + train). Pass --sequences to override.
python data_statistics.py                                    # train_data_dbscan.csv
python data_statistics.py --source lightning                 # train_data_lightning.csv
python data_statistics.py --source dbscan --split validation # validation_data_dbscan.csv
python data_statistics.py --csv /any/path.csv                # explicit override

# Lightning activity bar plots (reads lightning_active_steps.csv; plots-only).
python our_data/lightning_data/visualize_lightning_stats.py \
    --output_dir our_data/lightning_data

# Re-run the DBSCAN patch-selection diagnostic plot for one date.
# Now styled to match the prediction plotter (Romania-centred view,
# neighbour-country borders, dashed grid + red active highlight).
python identify_patches.py --date 2025-05-16 --plot
python identify_patches.py --date 2025-05-16 --plot --source opera
```

### Data Acquisition and Arrangement

#### MTG FCI L1C Satellite Data Pipeline (SFTP)

`pipeline_msg_mtg.py` was rewritten to pull FCI L1C from the ANM internal storage via SFTP (instead of EUMETSAT Data Store / `eumdac`) and to read chunks directly with `netCDF4` + `hdf5plugin` (instead of going through Satpy).

```bash
# 1. Pick the training cadence first (writes our_data/timestep_config.json)
python validate_timestep.py --step_minutes 15

# 2. Download + process MTG FCI L1C for a time range
python our_data/satellite_data/pipeline_msg_mtg.py \
    --start 2026/02/01-0000 \
    --end   2026/04/01-0000 \
    --password_file password.txt
```

**Required arguments:**

| Flag | Description |
|------|-------------|
| `--start`, `--end` | Date/time range in `yyyy/mm/dd-hhmm` (UTC). Use `2026/04/01-0000` to include all of March 31. |
| `--password_file`, `-pw` | Text file containing the SSH password for `anm@192.168.11.223` on a single line. Keep out of git. |

**Optional flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--products_file`, `-pf` | JSON file listing MTG channels under the `"mtg"` key. | `satellite_products.json` |
| `--output_dir`, `-o` | Output directory for processed channels. | `./MTG` in CWD |
| `--full_disk` | Download all 40 chunks instead of Romania-only. | off |
| `--timesteps` | Override the minute filter (e.g. `00 10 30 40`) or pass `all` for every native :00/.../:50. | read from `timestep_config.json` |
| `--skip_download` | Skip SFTP and process files already in `<output_dir>/_raw_chunks/`. | off |
| `--workers`, `-w` | Parallel workers for repeat-cycle processing. | `10` |

**Key differences from the previous implementation:**

1. **Data source**: EUMETSAT Data Store (`eumdac`) → SFTP from `anm@192.168.11.223:/ShortTermStorage/GEOSTATIONARY/MTG/FCI/`. Date, chunk number, and minute-of-hour filters are applied **server-side** via SSH `ls` with glob patterns before anything is transferred.
2. **Chunk filter**: `{34, 35, 36, 37, 38}` (±1 buffer) → `{35, 36}` based on the Météo-France FCI scan diagram. Halves transfer volume vs. the previous default.
3. **Processing**: manual `netCDF4.Dataset` reading is kept but the custom `fci_scanning_angles_to_latlon()` inverse projection and the `ProcessPoolExecutor` chunk-stitching pipeline are replaced by a single `process_repeat_cycle()` function that:
   - opens each chunk with `netCDF4` (hdf5plugin provides CharLS decompression),
   - reads `data[<channel>]/measured/effective_radiance` with `scale_factor` / `add_offset` auto-applied,
   - concatenates Romania chunks vertically,
   - writes one `.npy` per channel + a sidecar `MTG/mtg_constants.json` describing the geos projection (perspective height, semi-major/minor axes, sub-satellite longitude, sweep axis) and the 1-D scanning angles `x_geos`, `y_geos`.
4. **No more coordinate `.npy` files**: the old `coordinates/lat_{1,2}km.npy` and `lon_{1,2}km.npy` are gone. `reproject.py` rebuilds the source lat/lon arrays on demand from `mtg_constants.json` via `pyproj.Proj(proj='geos', ...)`.
5. **Legacy code removed**: the original `eumdac` Data Store path and the inert MSG SEVIRI block that used to live at the bottom of `pipeline_msg_mtg.py` have been deleted. Only the MTG-via-SFTP path remains. If you need to re-enable MSG ingestion you can recover the historical code from `git log`.

> **Network**: you must be on a network with route to `192.168.11.223` (ANM internal/VPN) and have read access to the FCI storage path. The script fails fast if SFTP can't connect or the password file is missing.

#### MTG Helper Scripts

**`summarize_mtg.py`** — scan `_raw_chunks/` and emit a CSV showing how many repeat cycles and chunk files are present per date. Useful to confirm a download is complete before processing.

```bash
python our_data/satellite_data/summarize_mtg.py
python our_data/satellite_data/summarize_mtg.py --raw_dir path/to/_raw_chunks --output summary.csv
```

**`inspect_mtg.py`** — reconstruct a CF-compliant NetCDF from a pipeline `.npy` (using `mtg_constants.json` for the geos grid) or from a reprojected `.npy` (using `romania_grid_lats/lons.npy`), and optionally plot it with matplotlib. Use this to open the data in Panoply / QGIS or to sanity-check a frame.

```bash
# Plot pipeline output (geostationary grid)
python our_data/satellite_data/inspect_mtg.py --raw \
    --npy MTG/vis_06/nc4_2026-02-13-Romania_vis_06/nc4_2026-02-13-Romania_0930_vis_06.npy \
    --constants MTG/mtg_constants.json

# Plot reprojected output (Romania EPSG:31700 grid)
python our_data/satellite_data/inspect_mtg.py --reprojected \
    --npy reprojected_data/satellite_data/MTG/vis_06/.../nc4_..._0930_vis_06.npy

# Save .nc without plotting
python our_data/satellite_data/inspect_mtg.py --raw --npy <path> --constants <path> --save_nc --no_plot
```

#### `reproject.py` — MTG branch updates

Two changes were needed to align `reproject.py` with the new pipeline output:

1. **Source grid reconstruction**: previously `reproject_satellite_mtg()` loaded the precomputed `coordinates/lat_{1,2}km.npy` / `lon_{1,2}km.npy` files written by the old pipeline. Those files no longer exist. The new code reads `our_data/satellite_data/MTG/mtg_constants.json`, builds a `pyproj.Proj(proj='geos', h=..., a=..., b=..., lon_0=..., sweep=...)` from the embedded projection parameters, and reconstructs 2-D source lat/lon arrays from `x_geos` / `y_geos` once per resolution. The KD-tree (`PrecomputedMapping`) caching strategy is unchanged.
2. **Input format**: MTG and lightning inputs are now `.npy` arrays (MTG written by `pipeline_msg_mtg.py`, lightning written by `read_kml_version2.py`). Radar and MSG (disabled) paths still use `.nc`. To get a CF-compliant `.nc` for GIS inspection from any reprojected `.npy`, see `inspect_mtg.py --reprojected` and `inspect_lightning.py`.

The reproject output for MTG remains `.npy` on the Romania 1536×768 grid. Use `inspect_mtg.py --reprojected` to get a CF NetCDF for any single reprojected sample.

#### OPERA Radar Pipeline (SFTP + reproject)

OPERA is an alternative radar source with two products:

| Product | Native resolution | Cadence | File format |
|---|---|---|---|
| Maximum reflectivity (dBZ) | 2 km | 15 min | HDF5 (`.h5`) |
| Instantaneous rainfall rate (mm/h) | 2 km | 15 min | HDF5 (`.h5`) |

Both products are listed in `product_cadences.config` as `opera_reflectivity = 15` and `opera_rainfall_rate = 15`. They raise the validator floor to 15 min when OPERA is in use — comment them out with a leading `#` in the cadences file if you're not using OPERA and want a finer training step.

##### Step 1 — Download (`pipeline_opera.py`)

Connects to `claudiu@64.225.128.186` via SFTP and pulls files from `/eumetsatdata/opera-reflectivity/` and `/eumetsatdata/opera-rainfall-rate/`. Server-side dirs are already partitioned as `{YYYY}/{MM}/{DD}/`, so the local mirror reuses the same hierarchy under `our_data/opera_data/`. Per-product minute filters are read from `timestep_config.json`; the script skips files already present in the cache for resumability.

```bash
# 1. Pick the training cadence (writes our_data/timestep_config.json)
python validate_timestep.py --step_minutes 15

# 2. Download OPERA for a date range
python our_data/opera_data/pipeline_opera.py \
    --start 2025/06/15-0000 --end 2025/06/15-2359 \
    --password_file password.txt
```

**Required arguments:**

| Flag | Description |
|------|-------------|
| `--start`, `--end` | Date/time range in `yyyy/mm/dd-hhmm` (UTC), end inclusive. |
| `--password_file`, `-pw` *or* `--ssh_key`, `-i` | **Exactly one** of: text file with the SSH password for `claudiu@64.225.128.186`, **or** a path to a private key (e.g. `~/.ssh/id_ed25519` — matching the `scp -i` example). Mutually exclusive. |

**Optional flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--cache_dir`, `-c` | Local OPERA root | `our_data/opera_data/` |
| `--products` | `opera_reflectivity`, `opera_rainfall_rate`, or both | both |
| `--timesteps` | Override the per-product minute filter (one filter applied to all chosen products); `all` keeps every native timestep | per-product filter from `timestep_config.json` |
| `--remote_host` | Override the SSH host | `64.225.128.186` |
| `--remote_user` | Override the SSH user | `claudiu` |
| `--remote_base` | Remote directory holding the per-product subdirs. Switch to `/home/eumetsatdata` (or any other path) if the default isn't where the data lives on the VM. | `/eumetsatdata` |

Only `.h5` files are transferred; OPERA-internal metadata or index files in the same directory are skipped. Per-file `[i/total] Downloading <filename>` progress, identical to the MTG pipeline.

**Supported filename conventions** (the timestamp parser tries both):

- ISO (current EWC dump): `2026-05-11T000500Z-reflectivity-composite-opera.h5`
- Compact (legacy / EUMETSAT): `T_PAAH21_C_LFPW_20250615120000.h5`, `composite_201801011500.h5`

**Troubleshooting**

- `cannot list /eumetsatdata/opera-reflectivity/...: [Errno 2] No such file` → the data is mounted under a different prefix. Try `--remote_base /home/eumetsatdata`.
- `cannot list ...: Permission denied` → password auth is being rejected. Switch to `--ssh_key ~/.ssh/id_ed25519`.
- Per-date `cannot list ...` messages with the right base path mean the remote dirs simply don't exist for those dates (incomplete archive). That's an upstream data gap, not a script issue.

##### Step 2 — Coverage report (`summarize_opera_data.py`)

Walks the per-date subdirs, parses each filename's timestamp, and produces a per-date / per-product completeness table plus a JSON listing the exact missing timestamps relative to the configured cadence.

```bash
# Defaults (read cadence from timestep_config.json, both products)
python our_data/opera_data/summarize_opera_data.py

# Check completeness at native cadence (every :00, :05, :10, ... for reflectivity)
python our_data/opera_data/summarize_opera_data.py --timesteps all

# Only one product, custom paths
python our_data/opera_data/summarize_opera_data.py \
    --products opera_reflectivity \
    --data_dir D:/backup/opera --output opera_summary.csv \
    --missing opera_missing.json
```

The CSV has per-product `_files`, `_on_grid`, `_off_grid`, `_expected`, `_coverage_pct` columns. The JSON has per-date `missing_times` lists (HH:MM) plus per-product overall coverage in `summary`.

##### Step 3 — Reproject to the Romania grid (via `reproject.py --opera`)

OPERA reprojection lives inside the unified `reproject.py` (the old standalone `reproject_opera.py` was removed). Reads each `.h5` file's `/where` metadata, builds `pyproj.Proj(projdef)` source projection, projects via `pyresample` KD-tree onto the EPSG:31700 Stereo70 grid (1536×768), and saves one **`.npy` per file** under `our_data/reprojected_data/opera_data/{product}/nc4_{date}-Romania_{product}/`. The KD-tree mapping is built **once per product** from the first file and reused across the rest. Day folders run in parallel via the shared `ThreadPoolExecutor`, same as the other product families.

```bash
# Both OPERA products
python reproject.py --opera

# All products in a single run (radar + MTG + lightning + OPERA)
python reproject.py --all

# Single date / custom worker count
python reproject.py --opera --date 2025-06-15 --workers 8
```

Output schema (one `.npy` per source `.h5`, plus shared sidecars):

| Path | Contents | Notes |
|---|---|---|
| `reprojected_data/opera_data/{product}/nc4_{date}-Romania_{product}/nc4_{date}-Romania_{HHMM}_{product}.npy` | `float32` array on the 768×1536 Romania grid | `nodata` → NaN, `undetect` → 0 (no precipitation detected) |
| `reprojected_data/opera_data/opera_constants.json` | Source projection per product: `projdef`, `xsize`, `ysize`, `xscale`, `yscale`, LL/UR corner coords | Written once from the first file of each product |
| `reprojected_data/romania_grid_lats.npy`, `reprojected_data/romania_grid_lons.npy` | Target lat/lon arrays on the Romania grid | Shared across every product (radar / MTG / lightning / OPERA) |

To rebuild a CF-compliant NetCDF for inspection in Panoply / QGIS, point `inspect_mtg.py --reprojected` at one of the `.npy` files — it auto-finds the shared `romania_grid_*.npy` via a walk-up.

> **Note**: OPERA files are read with `h5py` (the format is plain HDF5, not NetCDF). Make sure `h5py` is installed (`pip install h5py`).

#### Timestep Selection (no interpolation)

Both radar composites and MTG FCI data are acquired at native 10-minute cadence (:00, :10, :20, :30, :40, :50). The training cadence is chosen via [`validate_timestep.py`](#step-0--set-the-training-cadence) and stored in `our_data/timestep_config.json`; both `radar_arrange.py` and `pipeline_msg_mtg.py` read that config to decide which native minutes to keep. No optical flow interpolation is used.

For step = 15 min with 10-min sources, the script picks `{00, 10, 30, 40}` — the natives that minimise distance to each grid slot, breaking equidistant ties by preferring the minute closer to the hour boundary. This produces an alternating 10-20-10-20 spacing across the day:

| Sample | Native minute | Distance to next sample |
|---|---|---|
| Slot 0 | :00 | 10 min |
| Slot 1 | :10 | 20 min |
| Slot 2 | :30 | 10 min |
| Slot 3 | :40 | 20 min |
| Slot 4 | next hour :00 | 10 min |

Both scripts accept `--timesteps all` (keep every native timestep) or `--timesteps NN [NN ...]` (override the config explicitly).

#### Raw Data Arrangement (one-time setup)

These scripts organize raw data files into the COALITION-4 directory structure:

Raw data is stored under `our_data/raw_data/` with the following layout:

```
our_data/raw_data/
├── radar/netcdf/{rain_rate, reflectivity, cmax, vil, eht, acum1h}/*.nc
└── lightning/*.kml
```

Run all arrange scripts from the project root:

```bash
# Radar: raw NetCDF → our_data/radar_data/{product}/nc4_{date}-Romania_{product}/
# Default: only keeps :00, :10, :30, :50 timesteps (nearest to 15-min grid)
python our_data/raw_data/radar_arrange.py --source_root our_data/raw_data/radar/netcdf --target_root our_data/radar_data
# Keep all timesteps (no filtering)
python our_data/raw_data/radar_arrange.py --source_root our_data/raw_data/radar/netcdf --target_root our_data/radar_data --timesteps all

# Lightning: date-based filenames (dd_mm_yyyy.kml)
python our_data/raw_data/lightning_arrange.py -s our_data/raw_data/lightning -t our_data/lightning_data

# Lightning: sequential filenames (lightning.kml, lightning (N).kml, plus
# the timestamped lightning - YYYY-MM-DDTHHMMSS.fff.kml form). Numbered
# files take indices 0..max_numbered; timestamped files are sorted
# ascending by embedded timestamp and assigned indices max+1, max+2, ...
python our_data/raw_data/lightning_arrange.py -s our_data/raw_data/lightning --start-date 2025-04-01 --end-date 2025-09-30
python our_data/raw_data/lightning_arrange.py -s our_data/raw_data/lightning --start-date 2025-04-01 --end-date 2025-09-30 --dry-run

# KML → per-cadence .npy lightning maps (density, current, occurrence)
# Filter-aligned to products.lightning.filter; variable-width windows
# cover every minute of the day so no strokes are lost. Skips already-
# complete dates by default; pass --force to overwrite.
python our_data/lightning_data/read_kml_version2.py
python our_data/lightning_data/read_kml_version2.py --date 2025-05-15
python our_data/lightning_data/read_kml_version2.py --force
```

The sequential lightning arrangement mode also generates a `lightning_filename_mapping.csv` at the target root mapping each original filename to its assigned date (with `source_kind` and `embedded_ts` columns so timestamped vs. numbered ordering decisions are auditable).

## Architecture Summary

The model uses a multi-branch encoder with resolution-specific input streams:

- **HR branch** (1 km): radar products + lightning + MTG vis_06 → ConvGRU over 3 input timesteps
- **MR branch** (2 km): MTG IR/WV channels and/or OPERA → ConvGRU over 3 input timesteps

Branches merge at matching spatial scales during the encoder's downsampling stages. The decoder generates 3 future frames (T+15, T+30, T+45 min) autoregressively from the encoded state.

- **Lightning target**: weighted focal loss (γ=2) to handle severe class imbalance (~1% positive pixels)
- **Radar / OPERA target**: categorical cross-entropy with 5 precipitation intensity classes

### Dynamic Model Construction

The model architecture is built dynamically from `metadata.json` saved by `create_datasets.py`. Each dataset records its input group names (e.g. `past_hr`, `past_mr`), their shapes `[T, H, W, C]`, and the label type. `train_models.py` reads this metadata and:

1. Creates one input branch per group
2. Computes the spatial downsampling factor from the ratio of each input's resolution to the maximum resolution
3. Concatenates inputs that share the same resolution
4. Builds the encoder-decoder with the correct channel counts

This means a dataset created with any subset of inputs produces a matching model with the corresponding branches — no code changes needed. The Swin head from `--stage finetune` similarly inherits the backbone's shape.

## Key Differences from Original COALITION-4 (MeteoSwiss)

| Aspect | Original (Switzerland) | This adaptation (Romania) |
|---|---|---|
| Projection | EPSG:21781 (Swiss Grid) | EPSG:31700 (Stereo70) |
| Grid size | 710×640 | 1536×768 |
| Patch size | 32×32 (variable position) | 256×256 (fixed 6×3 grid) |
| Temporal resolution | 5 min | 15 min |
| Input timesteps | 12 (60 min) | 3 (30 min) + current |
| Output timesteps | 12 (60 min) | 3 (45 min) |
| Satellite | MSG SEVIRI (12 channels) | MSG (5 ch) or MTG FCI (5 ch) |
| DEM / NWP | Included | Removed (validated via Grad-CAM + ξ analysis) |
| Lightning network | MeteoSwiss | LINET |
| Dataset split | By year | Czibula temporal blocks (10/10/80 within configurable blocks per day) |

## Feature Importance Analysis

After training, the model's learned representations can be analysed with three complementary methods using `feature_importance_analysis.py`:

| Method | What it measures | Requirements |
|--------|-----------------|--------------|
| **Grad-CAM + Xi** | Spatial attention per input, correlated with output via Chatterjee's rank coefficient | 1 trained model |
| **SHAP** | Pixel-level feature importance via DeepExplainer | 1 trained model |

The script also produces a **prediction diagnostics** panel (MAE, RMSE, predicted-vs-target curves, and a per-sample MAE heatmap across lead times).

### Quick Start

```bash
conda activate tfenv

# Grad-CAM + Xi analysis only (fastest)
python feature_importance_analysis.py \
    --model models/coalition_mtg_lightning_lightning.keras \
    --data our_data/datasets/mtg_lightning_lightning/test \
    --methods gradcam_xi

# Grad-CAM + Xi + SHAP
python feature_importance_analysis.py \
    --model models/coalition_mtg_lightning_lightning.keras \
    --data our_data/datasets/mtg_lightning_lightning/test \
    --methods gradcam_xi shap
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Path to `.keras` model file | (required) |
| `--data` | Path to saved test dataset directory | (required) |
| `--output` | Output directory for results | `results/feature_importance` |
| `--methods` | Which analyses to run (`gradcam_xi`, `shap`) | `gradcam_xi shap` |
| `--num-samples` | Number of test samples to average Grad-CAM over | `4` |

### Outputs

All results are saved to `--output` (default `results/feature_importance/`):

| File | Content |
|------|---------|
| `prediction_diagnostics.png` | 4-panel MAE/RMSE/value comparison/heatmap |
| `xi_matrix.csv` | Xi correlations (inputs x timesteps) |
| `xi_heatmap.html` | Interactive Xi heatmap |
| `xi_bar_chart.html` | Top-N inputs by Xi |
| `xi_boxplots.html` | Xi distribution per input |
| `gradcam_rank*_*.png` | 5-panel Grad-CAM comparison plots |
| `shap_spatial_maps.png` | SHAP importance heatmaps |
| `shap_bar_chart.html` | Global SHAP importance |
| `shap_importance.csv` | SHAP values per input |
| `method_comparison.csv` | Grad-CAM + Xi vs SHAP side-by-side |
| `method_correlations.csv` | Spearman + Pearson between methods |
| `method_comparison.html` | Normalised bar chart comparing methods |

## References

- Leinonen, J., et al. (2022). Seamless lightning nowcasting with recurrent-convolutional deep learning. *Monthly Weather Review*, 150(6).
- Czibula, G., et al. (2024). SepConv-based precipitation nowcasting using radar data. *Natural Hazards and Earth System Sciences*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks. *ICCV*.
- Chatterjee, S. (2021). A new coefficient of correlation. *Journal of the American Statistical Association*.

## License

This project is developed at ANM Romania under the EUMETSAT Training Placement Scheme.
