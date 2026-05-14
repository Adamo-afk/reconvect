# COALITION-4 Nowcasting System — Romanian Adaptation

Adaptation of the [COALITION-4](https://doi.org/10.1175/MWR-D-22-0084.1) deep learning nowcasting system (Leinonen et al., 2022, MeteoSwiss) for Romanian meteorological conditions. Developed at Romania's National Meteorological Administration (ANM) as part of the EUMETSAT Training Placement Scheme.

The system uses a recurrent-convolutional encoder-decoder architecture operating on the EPSG:31700 Stereo70 projection of Romania to produce 45-minute precipitation and lightning nowcasts from multi-source meteorological inputs.

## Project Overview

Three training configurations are supported. The first two compare MSG vs MTG with the legacy ANM-radar precipitation target; the third replaces the legacy radar with the pan-European **OPERA** composite and is the basis for the NWCSAF Shapley study (whether NWCSAF adds skill on top of radar + MTG).

- **MSG experiment**: ANM radar + LINET lightning + MSG SEVIRI (5 channels, 3 km) + NWCSAF
- **MTG experiment**: ANM radar + LINET lightning + MTG FCI (5 channels, 1–2 km) + NWCSAF
- **MTG + OPERA experiment** (NWCSAF Shapley study): **OPERA** precipitation composite (reflectivity + instantaneous rainfall rate, 2 km, 15 min) + MTG FCI + NWCSAF. The label is `opera_rainfall_rate` multi-class (5 bins, same thresholds as the radar configuration). No lightning. Four coalition modes (`mtg_opera_radar_only / _mtgmr / _nwcsaf / _full`) are trained to compute classical Shapley values for MTG IR/WV and NWCSAF, with OPERA always present.

The first two share the same radar targets, lightning labels, and NWCSAF cloud products; only the satellite input branch differs. The OPERA configuration switches the radar source: same downstream pipeline (regridding to EPSG:31700, DBSCAN-driven patch index, sequence extraction, normalization, dataset creation, training, evaluation), but driven by OPERA's pre-regridded HDF5 instead of the legacy ANM-radar NetCDF.

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
│   │   ├── inspect_mtg.py             # Reconstruct .nc and plot raw / regridded MTG data
│   │   ├── summarize_raw_chunks.py    # CSV report of downloaded FCI chunks per date
│   │   ├── check_chunk_names.py       # Diagnostic: inspect FCI chunk NetCDF structure
│   │   ├── check_chunk_contents.py    # Diagnostic: read FCI chunk radiance data
│   │   └── _raw_chunks/               # Cache of downloaded FCI chunk files (gitignored)
│   ├── lightning_data/
│   │   ├── kml_data/{date}/{date}.kml
│   │   ├── density/nc4_{date}-Romania_density/*.nc
│   │   ├── current/nc4_{date}-Romania_current/*.nc
│   │   ├── occurrence/nc4_{date}-Romania_occurrence/*.nc
│   │   ├── read_kml_version2.py       # KML → 15-min NetCDF lightning maps
│   │   └── visualize_lightning_stats.py  # Lightning activity bar plots and CSV
│   ├── nwcsaf_data/
│   │   ├── {date}-Romania/S_NWC_{CMIC|CTTH}_*.nc   # arranged per-date dirs
│   │   ├── _raw_data/                  # SFTP cache of downloaded SAFNWC files (gitignored)
│   │   ├── pipeline_nwcsaf.py         # NWCSAF L2 SFTP + filter + per-date arrange
│   │   ├── summarize_raw_data.py      # CSV + missing-timesteps report for _raw_data/
│   │   └── process_nwcsaf.py          # Extract lat/lon from NWCSAF NetCDF files
│   ├── opera_data/
│   │   ├── reflectivity/{YYYY}/{MM}/{DD}/*.h5      # OPERA max-reflectivity HDF5 (2 km, 15 min)
│   │   ├── rainfall_rate/{YYYY}/{MM}/{DD}/*.h5     # OPERA rain-rate HDF5 (2 km, 15 min)
│   │   ├── pipeline_opera.py          # SFTP/SCP download from EWC VM with cadence filtering
│   │   └── summarize_opera_data.py    # CSV + missing-timesteps report
│   ├── raw_data/
│   │   ├── radar_arrange.py           # Arrange raw radar files to COALITION-4 structure
│   │   └── lightning_arrange.py       # Arrange raw KML files (date-based or sequential)
│   ├── regridded_data/                # Cached reprojected data (generated by regrid.py)
│   │   ├── romania_grid_lats.npy           # Shared 768×1536 EPSG:31700 lat array
│   │   ├── romania_grid_lons.npy           # Shared 768×1536 EPSG:31700 lon array
│   │   ├── radar_data/{product}/nc4_{date}-Romania_{product}/*.npy
│   │   ├── satellite_data/MSG/{channel}/nc4_{date}-Romania_{channel}/*.npy
│   │   ├── satellite_data/MTG/{channel}/nc4_{date}-Romania_{channel}/*.npy
│   │   ├── lightning_data/{product}/nc4_{date}-Romania_{product}/*.npy
│   │   ├── nwcsaf_data/{var}/nc4_{date}-Romania_{var}/*.npy   # per-variable .npy
│   │   ├── nwcsaf_data/nwcsaf_constants.json                  # source projection (gdal_projection)
│   │   ├── opera_data/{product}/nc4_{date}-Romania_{product}/*.npy
│   │   └── opera_data/opera_constants.json                    # source projection (/where attrs)
│   ├── patch_index/                   # DBSCAN patch identification output (radar source)
│   │   ├── patch_index.csv
│   │   ├── patch_index.json
│   │   └── plots/                     # Optional diagnostic plots
│   ├── lightning_periods/             # 3-stage lightning cascade output (lightning source)
│   │   ├── lightning_periods_config.json   # Parameters used (reproducibility)
│   │   ├── lightning_bins.csv               # Per-bin lightning_fraction + valid flag
│   │   ├── lightning_periods.csv            # Per-day summary + kept flag
│   │   └── lightning_patches.csv            # Per-bin per-patch index (schema mirrors patch_index.csv)
│   ├── patches/                       # Extracted 256×256 patches (generated by extract_patches.py)
│   │   └── {date}/{variable}_{HHMM}_{HR|LR}.npy
│   ├── datasets/                      # Saved TF datasets (generated by create_datasets.py)
│   │   └── {mode}/train|validation|test/
│   │       └── metadata.json          # Input shapes, label type (used by train_models.py)
│   ├── data_statistics/               # Diagnostic plots (generated by data_statistics.py)
│   ├── timestep_config.json           # Cadence config (generated by validate_timestep.py)
│   ├── sequence_meta.json             # Per-sample window (generated in Step 4.1)
│   ├── consistent_dates.csv           # Cross-product keep flags (generated in Step 4.2)
│   ├── intersect_product_coverage.json # Manifest from Step 4.2
│   ├── normalization_stats.json       # Per-variable mean/std (generated in Step 4.3)
│   ├── train_data.csv                 # Training sequences (80% per temporal block)
│   ├── train_data_consistent.csv      # Step 4.2 filter applied (filtered version of train_data.csv)
│   ├── validation_data.csv            # Validation sequences (10% per temporal block)
│   ├── test_data.csv                  # Test sequences (10% per temporal block)
│   └── lightning_fraction.json        # Positive pixel fraction for focal loss
│
├── models/                            # Saved trained models (not tracked in git)
│   └── {mode}/
├── evaluation/                        # Evaluation outputs (not tracked in git)
│   └── eval_{mode}/
│
├── product_cadences.json              # Native cadence per data product (input to Step 0)
├── validate_timestep.py               # Step 0: Validate cadence → timestep_config.json
├── identify_patches.py                # Step 1: DBSCAN → patch_index.csv/json (radar)
├── identify_lightning_periods.py      # Step 1 (lightning variant): 3-stage cascade → lightning_periods/
├── summarize_lightning_periods.py     # Coverage / missing-timestep report for the cascade
├── regrid.py                          # Step 2: Reproject all products to Romania grid
├── extract_patches.py                 # Step 3: Slice 256×256 patches from regridded data
├── extract_patch_seq_for_datasets.py  # Step 4.1: Continuous sequences + Czibula split
├── intersect_product_coverage.py      # Step 4.2: Cross-product train/val/test filter
├── compute_normalization_stats.py     # Step 4.3: Per-variable mean/std → normalization_stats.json
├── create_datasets.py                 # Step 5: Build TF datasets + metadata.json
├── train_models.py                    # Step 6: Train (dynamic architecture from metadata)
├── evaluate_coalition.py              # Step 7: Evaluate and generate plots
│
├── feature_importance_analysis.py     # Grad-CAM + Xi, SHAP, classical Shapley analysis
├── lightning_fraction.py              # Compute positive pixel fraction for focal loss
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
| LR (3 km) | 1536×768 | 64×64 | 4×4 avg | MSG (VIS006, IR_039, IR_108, WV_062, WV_073), NWCSAF (ctth_alti, ctth_tempe, cmic_phase, cmic_cot) |

### Data Sources

| Source | Products | Native cadence | Native resolution | Role | Pipeline entry point |
|---|---|---|---|---|---|
| **ANM radar** (legacy) | RZC, BZC, CZC, EZC-20, LZC, CPCH | 10 min | ~1 km | Precipitation target + features (MSG/MTG experiments) | `our_data/raw_data/radar_arrange.py` |
| **LINET lightning** | density, current, occurrence | 5 min (aggregated to step) | Native KML → 1 km grid | Lightning target + features | `our_data/raw_data/lightning_arrange.py`, `our_data/lightning_data/read_kml_version2.py` |
| **MSG SEVIRI** *(disabled in active build)* | VIS006, IR_039, IR_108, WV_062, WV_073 | 15 min | 3 km | Satellite features (LR branch) | `our_data/satellite_data/pipeline_msg_mtg.py` |
| **MTG FCI L1C** | vis_06, ir_38, ir_105, wv_63, wv_73 | 10 min | 1 km (vis_06) / 2 km (IR/WV) | Satellite features (HR + MR branches) | `our_data/satellite_data/pipeline_msg_mtg.py` |
| **NWCSAF L2** | ctth_alti, ctth_tempe, cmic_phase, cmic_cot | 10 min | 3 km | Cloud-product features (LR branch) | `our_data/nwcsaf_data/pipeline_nwcsaf.py` |
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
[`product_cadences.json`](product_cadences.json) at the project root and
cover every product family the pipeline knows about (radar, MTG, NWCSAF,
the two OPERA products, and lightning).

**Lightning** is special: it has no native scan cadence — LINET strokes
are individual observed events, not raster scans. The lightning maps that
feed the model are produced by binning strokes into windows whose width
mirrors whichever paired product the experiment uses:

- When **radar** is in the configuration, lightning bins align to the
  radar cadence (`step_minutes` resolved against the radar `filter` set).
- When **OPERA** replaces radar, lightning bins align to the OPERA
  cadence — although the current Shapley study deliberately runs
  **without lightning** so the comparison is OPERA vs MTG vs NWCSAF only.
- In any other combination the validator picks the most common cadence of
  the active products and aligns lightning bins to it.

OPERA composites are scanned with the 15-min product family (NIMBUS,
CIRRUS, ODYSSEY) — see the *OPERA composite acquisition window* note
just below for the exact `[NT-X, NT+Y]` data windows each composite type
uses, and why the 10-min products are filtered to the alternating
`{:00, :10, :30, :40}` pattern when paired with a 15-min training step.

If you add a new data source or its native cadence changes, edit
`product_cadences.json` rather than the validator script. The validator
reads the file at startup; it does not inspect any data folders. Override
with `--cadences_file path/to/other.json`. Comment out products you are
not using (rename with a leading underscore) if they would otherwise
raise the floor beyond what you need — e.g. drop OPERA's 15-min floor
back to 10 min when running a radar-only experiment.

Run the validator once before any other pipeline step:

```bash
python validate_timestep.py --step_minutes 15        # 15-min cadence (required when OPERA is in the mix)
python validate_timestep.py --step_minutes 10        # 10-min cadence (radar/MTG/NWCSAF only; OPERA must be commented out)
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
(radar, MTG, NWCSAF) need to land within roughly ±5 min of those nominal
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
scripts (`radar_arrange.py`, `pipeline_msg_mtg.py`, `pipeline_nwcsaf.py`,
`pipeline_opera.py`, `read_kml_version2.py`, `identify_patches.py`,
`identify_lightning_periods.py`, `extract_patch_seq_for_datasets.py`,
`create_datasets.py`) **read this file** and refuse to run if it is missing.

For step=15 with 10-min radar/MTG/NWCSAF the resulting filter is `{00, 10, 30, 40}`,
giving an alternating 10–20–10–20 spacing between consecutive samples (no optical
flow interpolation is used).

> **MSG**: the MSG SEVIRI ingestion path is currently disabled in
> `pipeline_msg_mtg.py` and `create_datasets.py`. Only MTG FCI L1C is supported.

#### Step 1 — Identify convective patches from RZC radar

Runs DBSCAN on RZC rain rate data at 15-minute resolution. Produces a patch index mapping each timestamp to the active patches (1–18) on the fixed 6×3 grid.

```bash
python identify_patches.py
python identify_patches.py --date 2025-05-15 --plot    # single date with diagnostic plots

# OPERA-driven DBSCAN (uses pre-regridded opera_rainfall_rate; same 10 mm/h threshold as RZC)
python identify_patches.py --source opera
python identify_patches.py --source opera --start 2026-03-13 --end 2026-05-11
```

Output: `our_data/patch_index/patch_index.csv` and `patch_index.json`. The `--source opera` flag reads from `regridded_data/opera_data/rainfall_rate/` instead of regridding RZC on the fly; OPERA-source runs require `regrid.py --opera` to have been run first.

#### Step 2 — Reproject all products to the Romania grid

Regrids radar, satellite (MSG/MTG), lightning, NWCSAF, **and OPERA** to the 1536×768 grid. Uses precomputed KD-tree mappings (built once per source geometry) and parallel day-folder processing for speed.

```bash
python regrid.py --all                          # all products
python regrid.py --radar                        # radar only
python regrid.py --satellite MSG                # MSG channels only
python regrid.py --satellite MTG                # MTG channels only
python regrid.py --lightning                    # lightning (cache as .npy)
python regrid.py --nwcsaf                       # NWCSAF products
python regrid.py --opera                        # OPERA radar (HDF5 → .npy)
python regrid.py --all --workers 8              # 8 parallel workers
python regrid.py --radar --date 2025-05-15      # single date
```

**Every product family writes `.npy`** under `our_data/regridded_data/...`. The shared Romania-grid lat/lon arrays are written once at `our_data/regridded_data/romania_grid_{lats,lons}.npy`, and each non-trivial source (MTG, NWCSAF, OPERA) drops a sidecar `*_constants.json` with the source projection so an `.npy` can be re-attached to its projection at any later step (e.g. by `inspect_mtg.py --regridded`).

#### Step 3 — Extract 256×256 patches

Reads the patch index and regridded data, extracts active patches, applies resolution-dependent pooling (none for HR, 2×2 for MTG LR, 4×4 for MSG/NWCSAF LR).

Two pieces of context are read automatically:

1. **`timestep_config.json`** — provides the per-product minute filter. The patch index uses OPERA's 15-min grid (`:00, :15, :30, :45`), but MTG/NWCSAF files are written at `:00, :10, :30, :40`. The file finder maps the requested HHMM to the nearest minute in each product's filter — OPERA `:15` reads MTG `:10`, OPERA `:45` reads MTG `:40`, OPERA `:00 / :30` use exact match. The mapping is a direct function of the filter (not a ±tolerance search). If the config or a product entry is missing, the finder uses exact match.

2. **`train_data.csv` / `validation_data.csv` / `test_data.csv`** — the filtered CSVs from Step 4.2. Only timesteps referenced by at least one surviving sequence row are processed; the rest are skipped. Pass `--sequence_csvs none` to process every entry in `patch_index.csv` instead.

```bash
python extract_patches.py
python extract_patches.py --date 2025-05-15
python extract_patches.py --products radar lightning
```

Output: `our_data/patches/{date}/{variable}_{HHMM}_{HR|LR}.npy`

#### Step 4.1 — Extract temporally continuous sequences and split dataset

Analyzes the patch index to find patches with uninterrupted activity over a 6-step window (2 past + current + 3 future, 90 minutes total). Produces per-timestep npy indices accounting for index shifts when the active patch set changes.

Dataset splitting follows Czibula et al. (2024): each day is divided into equal temporal blocks (default 6h). Within each block, qualifying sequences are ordered chronologically — the first 10% go to test, next 10% to validation, remaining 80% to training. This ensures all three splits sample from the same diurnal distribution, avoiding hour-based bias.

```bash
python extract_patch_seq_for_datasets.py                            # 6h blocks, 10/10/80 split (radar source)
python extract_patch_seq_for_datasets.py --block_hours 4            # 4h blocks (finer diurnal balance)
python extract_patch_seq_for_datasets.py --block_hours 8            # 8h blocks
python extract_patch_seq_for_datasets.py --test_frac 0.15 --val_frac 0.15  # 15/15/70 split
python extract_patch_seq_for_datasets.py --source lightning         # lightning-driven sequences
```

Output: `our_data/train_data.csv`, `our_data/validation_data.csv`, `our_data/test_data.csv` (all three generated in a single run) plus `our_data/sequence_meta.json` (records the source, effective step, and past/future window length).

##### Lightning-source sequences

For the lightning training pipeline, sample selection must be driven by **lightning activity in time** rather than by radar-DBSCAN convective clusters in space. Add a `--source lightning` switch to `extract_patch_seq_for_datasets.py` and feed it the output of `identify_lightning_periods.py` (see [the lightning cascade section](#lightning-periods-3-stage-cascade) below). The CSV format, Czibula splitting, and step-column naming are identical; only the activity-index CSV and the step interval change.

```bash
# 1. Run the cascade once (produces our_data/lightning_periods/)
python identify_lightning_periods.py

# 2. Build lightning-driven sequences (uses aggregation_minutes as step interval)
python extract_patch_seq_for_datasets.py --source lightning
```

#### Lightning periods (3-stage cascade)

`identify_lightning_periods.py` produces the lightning-activity index that drives `extract_patch_seq_for_datasets.py --source lightning`. Lightning is much sparser than radar, so the model is trained on a separate sample list filtered to lightning-active periods only.

Cascade stages (all thresholds CLI-configurable, defaults from the design note):

| Stage | Default flag | Default value | Effect |
|---|---|---|---|
| **1 — Aggregation** | `--aggregation_minutes` | `60` | Bin native lightning maps into windows of this width. Must be `>=` and a multiple of `step_minutes` from `timestep_config.json`. The number of bins/day = `1440 / aggregation_minutes`. |
| **2 — Map filter** | `--map_fraction_threshold` | `1e-4` | Drop bins whose lightning_fraction (non-zero pixels / total pixels in the bin-aggregated map) is below this. |
| **3 — Day filter** | `--day_min_valid_fraction` | `0.20` | Drop days where fewer than this fraction of the day's bins survived stage 2. |
| **4 — Window filter** | `--window_days`, `--window_min_valid_days` | `10`, `2` | Tile the date range into non-overlapping `window_days`-day windows. Keep an entire window iff it has at least `window_min_valid_days` days that survived stage 3. |
| Source product | `--lightning_product` | `occurrence` | Which of `density` / `current` / `occurrence` feeds the cascade. |

For each (date, bin) tuple that survives all four stages, the script also computes **per-patch activity** on the standard 6×3 grid (any non-zero pixel in the 256×256 patch ⇒ patch is active). This is **Option B** of the activity-source design — lightning fully drives both the temporal selection *and* the spatial patch selection, taking over the role radar plays in the default sequence extractor.

```bash
python identify_lightning_periods.py                            # all defaults
python identify_lightning_periods.py --aggregation_minutes 30   # 30-min bins (must be a multiple of step_minutes)
python identify_lightning_periods.py --map_fraction_threshold 5e-5 \
                                     --day_min_valid_fraction 0.30
python identify_lightning_periods.py --start_date 2026-03-01 \
                                     --end_date 2026-04-30
```

Outputs in `our_data/lightning_periods/`:

| File | Purpose |
|---|---|
| `lightning_periods_config.json` | All CLI parameters used (reproducibility) |
| `lightning_bins.csv` | Per-bin detail: `lightning_fraction`, `valid_bin` |
| `lightning_periods.csv` | Per-day summary: `valid_day`, `window_kept`, `kept` |
| `lightning_patches.csv` | Per-bin per-patch index — schema mirrors `patch_index.csv` so `extract_patch_seq_for_datasets.py --source lightning` consumes it directly |

#### Cascade coverage report

`summarize_lightning_periods.py` reads the cascade outputs and produces a report parallel to `summarize_raw_chunks.py` (MTG) and `summarize_raw_data.py` (NWCSAF). For each date it tells you exactly how many bins survived the cascade and which bins were dropped, **broken down by the stage that filtered them**:

| Bucket | Meaning |
|---|---|
| `active` | bin survived all four stages — feeds into `lightning_patches.csv` |
| `no_data` | bin had zero native lightning maps (data gap) |
| `stage2_filter` | bin had data but `lightning_fraction < map_fraction_threshold` |
| `stage3_filter` | bin would have been valid but its day failed `day_min_valid_fraction` |
| `stage4_filter` | bin's day was valid but its 10-day window failed `window_min_valid_days` |

Bins fall into the **earliest** failing bucket (so a bin with no data on a day that would have failed stage 3 anyway is still reported as `no_data` — the most actionable signal). Run it right after `identify_lightning_periods.py`:

```bash
python summarize_lightning_periods.py                              # defaults
python summarize_lightning_periods.py --output coverage.csv \
                                      --missing missing.json
python summarize_lightning_periods.py --periods_dir other/dir
```

Two output files:

- **`lightning_summary.csv`** — per-date table with columns
  `date, bins_total, bins_active, bins_no_data, bins_stage2, bins_stage3, bins_stage4, valid_day, window_kept, kept, coverage_pct`.
- **`lightning_missing_timesteps.json`** — per-date `missing_breakdown` (HH:MM lists per bucket) plus an `overall_coverage_pct` summary block. Useful for spotting systematic data gaps before training.

#### Step 4.2 — Intersect per-product coverage across train / val / test

`intersect_product_coverage.py` reads per-product summary CSVs (the output of the `summarize_*.py` scripts from each data family), intersects them per date, and writes filtered `*_consistent.csv` versions of the train/val/test CSVs. The model can only train on samples where **every** input product has data, so this step protects the next two from silently working on inconsistent input sets.

Designed to work with **any combination of products** — pass only the `--summary` arguments for what you'll actually feed to the model. With/without NWCSAF, with/without lightning, with/without OPERA, etc.

```bash
# Full configuration — every product required at 100% per date
python intersect_product_coverage.py \
    --summary mtg=mtg_summary.csv \
    --summary nwcsaf=nwcsaf_summary.csv \
    --summary opera=opera_summary.csv \
    --summary lightning=lightning_summary.csv:kept \
    --threshold lightning=1

# Smaller experiment — only radar + MTG, 80% threshold acceptable
python intersect_product_coverage.py \
    --summary mtg=mtg_summary.csv \
    --min_coverage 80
```

**Each `--summary` argument** has the form `KEY=PATH[:COLUMN]`:

| Token | Meaning |
|---|---|
| `KEY` | Free-form label for the product (used in the manifest + decisions CSV) |
| `PATH` | Per-product summary CSV produced by a `summarize_*.py` script |
| `COLUMN` | Column to read (default `coverage_pct` — works for MTG, NWCSAF, and OPERA). Use `kept` for the lightning boolean, or `opera_reflectivity_coverage_pct` / `opera_rainfall_rate_coverage_pct` to gate on a single OPERA product instead of the intersection of both. |

Per-product threshold override via `--threshold KEY=VALUE` — needed when one product uses a 0–100 percentage and another uses a 0/1 flag.

**Outputs (under `--output_dir`, default `our_data/`):**

| File | Purpose |
|---|---|
| `consistent_dates.csv` | One row per date encountered, with per-product `_value` / `_ok` columns and the final `kept` flag |
| `train_data_consistent.csv`, `validation_data_consistent.csv`, `test_data_consistent.csv` | The original CSVs with rows whose `date` is not in the kept set removed. Drop-in replacements for the originals. |
| `intersect_product_coverage.json` | Manifest of every CLI argument, per-product source paths, thresholds, and per-split row counts |

Pass `--in_place` to overwrite `train_data.csv` / `validation_data.csv` / `test_data.csv` directly (backups are renamed to `*_original.csv` automatically).

> **Why this comes before Step 4.3**: the normalization stats are computed on the **filtered** training set, so they never see samples that would later be discarded for missing inputs — fewer file reads and stats that exactly match the eventual training distribution.

#### Step 4.3 — Compute normalization statistics

`compute_normalization_stats.py` derives per-variable mean / std from the regridded data so the model trains on values centred for the **Romanian** distribution, not the Swiss one. This step is **mandatory** before Step 5: `create_datasets.py` no longer falls back to the Leinonen Table A1 constants — if `normalization_stats.json` is missing, the run fails with an explicit pointer back to this script.

```bash
# Default — read regridded_data/, filter to training-eligible timesteps,
# scalar mean/std per variable
python compute_normalization_stats.py

# Subset of variables (faster iteration while tuning)
python compute_normalization_stats.py --variables RZC ir_105 cmic_cot

# Also surface p01 / p50 / p99 + MAD via reservoir sampling
python compute_normalization_stats.py --with_percentiles

# Disable training-window filter (DIAGNOSTIC ONLY — leaks val/test data)
python compute_normalization_stats.py --no_split_filter
```

**Policy decisions** (recorded inside the JSON for traceability):

| Decision | Choice | Why |
|---|---|---|
| Sample scope | training-set only, expanded across each row's past + current + future window | Stats from val / test would leak distributional info into the model |
| Spatial scope | single scalar mean / std per variable | Per-pixel climatology would overfit to training-domain geography (e.g. permanent radar beam blockage) |
| Source data | `our_data/regridded_data/` (full 1536 × 768 grids) | The pre-built patches in `our_data/patches/` are filtered by RZC / lightning activity — computing stats on them would bias every variable's distribution toward convective scenes |
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
| **NWCSAF — `ctth_alti`** (cloud-top height, m) | `nwcsaf_data/` | linear z-score, sentinel 65535 dropped | Roughly Gaussian when valid; the 65535 sentinel must be masked or it skews the mean badly. |
| **NWCSAF — `ctth_tempe`** (cloud-top temperature, K) | `nwcsaf_data/` | linear z-score, sentinel 65535 dropped | As above, with the temperature range. |
| **NWCSAF — `cmic_phase`** (categorical) | `nwcsaf_data/` | one-hot to 5 channels | Categorical variable — no continuous normalisation makes sense. |
| **NWCSAF — `cmic_cot`** (cloud optical thickness) | `nwcsaf_data/` | clip 0.1 → `log10` → z-score, sentinel 65535 dropped | Strictly positive, heavy-tailed; logging tames the distribution. |
| **OPERA — `opera_reflectivity`** (max reflectivity, dBZ) | `opera_data/` | linear z-score | Like CZC: dBZ is already logarithmic, Gaussian-ish where signal is present. |
| **OPERA — `opera_rainfall_rate`** (mm/h) | `opera_data/` | clip 0.01 → `log10` → z-score | Like RZC: heavy-tailed, zero-inflated. Same floor and transform family for consistency across both rain-rate sources. |

When `--with_percentiles` is passed, each variable block additionally carries `p01`, `p50`, `p99`, and `mad` (median absolute deviation) — useful for sanity-checking against the mean/std, and as robust alternatives if a variable is flagged `near_constant: true`.

#### Step 5 — Build TF datasets

Transforms patches using the **data-driven** mean / std from
`our_data/normalization_stats.json` (Step 4.3) and saves as `tf.data.Dataset` for each mode. Each dataset split also saves a `metadata.json` containing `input_shapes`, `label_type`, `past_timesteps`, and `future_timesteps` — this metadata drives dynamic model construction in Step 6.

Active modes: `mtg_lightning`, `mtg_radar`, `mtg_radar_continuous`, and the four OPERA-driven modes for the NWCSAF Shapley study: `mtg_opera_radar_only`, `mtg_opera_mtgmr`, `mtg_opera_nwcsaf`, `mtg_opera_full`. The MSG modes (`msg_lightning`, `msg_radar`, `msg_radar_continuous`) are commented out in `get_mode_config()` — re-enable in source if you need them.

```bash
python create_datasets.py --mode mtg_lightning --data_root ./our_data
python create_datasets.py --mode mtg_radar --data_root ./our_data
python create_datasets.py --mode mtg_radar_continuous --data_root ./our_data

# OPERA Shapley study (4-model coalition, "is NWCSAF useful?")
python create_datasets.py --mode mtg_opera_radar_only --data_root ./our_data
python create_datasets.py --mode mtg_opera_mtgmr      --data_root ./our_data
python create_datasets.py --mode mtg_opera_nwcsaf     --data_root ./our_data
python create_datasets.py --mode mtg_opera_full       --data_root ./our_data
```

The OPERA modes replace radar (RZC and friends) with OPERA `opera_reflectivity` + `opera_rainfall_rate` in the MR branch (2 km, `pool=2`), drop the lightning HR channels, and use `opera_rainfall_rate_hr` (HR alias of the same regridded file) as the 5-class multi-class label (same bin edges as RZC: `<10`, `10–20`, `20–30`, `30–40`, `≥40 mm/h`). The four modes form a Shapley coalition over MTG IR/WV and NWCSAF:

| Mode | HR | MR | LR |
|---|---|---|---|
| `mtg_opera_radar_only` | MTG `vis_06` | OPERA | — |
| `mtg_opera_mtgmr`      | MTG `vis_06` | OPERA + MTG IR/WV | — |
| `mtg_opera_nwcsaf`     | MTG `vis_06` | OPERA | NWCSAF |
| `mtg_opera_full`       | MTG `vis_06` | OPERA + MTG IR/WV | NWCSAF |

Custom modes (e.g. without NWCSAF) can be defined by adding a new configuration in `create_datasets.py` that omits `past_lr` from the input groups. The training script requires no code changes — it reads whatever inputs are in the dataset.

Output: `our_data/datasets/{mode}/train|validation|test/` (each with `metadata.json`)

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

Builds the COALITION recurrent-convolutional architecture (ResBlock + ConvGRU encoder-decoder) dynamically from the dataset's `metadata.json`. The model architecture adapts automatically to whatever inputs are present — number of input groups, channel counts, and resolutions are all read from metadata rather than hardcoded. This means training with different input configurations (e.g., with/without NWCSAF, MSG vs MTG) requires no code changes; only the dataset needs to change.

```bash
python train_models.py --mode mtg_lightning --epochs 10 --batch_size 32
python train_models.py --mode mtg_radar --epochs 10 --batch_size 32
python train_models.py --mode mtg_radar_continuous --epochs 10
python train_models.py --mode mtg_lightning --epochs 1 --batch_size 8   # quick test

# Train from a custom dataset directory (e.g. without NWCSAF)
python train_models.py --mode mtg_lightning_no_nwcsaf --dataset_dir our_data/datasets/mtg_lightning_no_nwcsaf
```

The `--dataset_dir` flag overrides the default `{data_root}/datasets/{mode}` path, while `data_root` still provides `lightning_fraction.json` independently.

Output: `models/{mode}/` (saved model + `training_history.json`)

#### Step 7 — Evaluate

Loads the trained model, runs evaluation on the test set, and generates diagnostic plots.

```bash
python evaluate_coalition.py --mode msg_lightning --data_root ./our_data --model_dir ./models
python evaluate_coalition.py --mode msg_radar --data_root ./our_data --model_dir ./models
```

Output: `evaluation/eval_{mode}/` (plots + `evaluation_results.json`)

### Utility Scripts

```bash
# Compute lightning positive pixel fraction (needed for focal loss)
python lightning_fraction.py

# Generate dataset diagnostic plots (6 panels: diurnal cycle, spatial heatmap, etc.)
python data_statistics.py
python data_statistics.py --sequences our_data/train_data.csv

# Lightning activity statistics (per-day and per-timestep bar plots + CSV)
python our_data/lightning_data/visualize_lightning_stats.py --data_root our_data/lightning_data --output_dir our_data/data_statistics
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
4. **No more coordinate `.npy` files**: the old `coordinates/lat_{1,2}km.npy` and `lon_{1,2}km.npy` are gone. `regrid.py` rebuilds the source lat/lon arrays on demand from `mtg_constants.json` via `pyproj.Proj(proj='geos', ...)`.
5. **Old code preserved, inert**: the original `eumdac` + manual stitching path lives at the bottom of the file inside a `_DATASTORE_MANUAL_DISABLED` raw-string block; the previously-disabled MSG SEVIRI code is still in `_MSG_DISABLED`. Both parse but never execute — only the MTG-via-SFTP path is active.

> **Network**: you must be on a network with route to `192.168.11.223` (ANM internal/VPN) and have read access to the FCI storage path. The script fails fast if SFTP can't connect or the password file is missing.

#### MTG Helper Scripts

**`summarize_raw_chunks.py`** — scan `_raw_chunks/` and emit a CSV showing how many repeat cycles and chunk files are present per date. Useful to confirm a download is complete before processing.

```bash
python our_data/satellite_data/summarize_raw_chunks.py
python our_data/satellite_data/summarize_raw_chunks.py --raw_dir path/to/_raw_chunks --output summary.csv
```

**`inspect_mtg.py`** — reconstruct a CF-compliant NetCDF from a pipeline `.npy` (using `mtg_constants.json` for the geos grid) or from a regridded `.npy` (using `romania_grid_lats/lons.npy`), and optionally plot it with matplotlib. Use this to open the data in Panoply / QGIS or to sanity-check a frame.

```bash
# Plot pipeline output (geostationary grid)
python our_data/satellite_data/inspect_mtg.py --raw \
    --npy MTG/vis_06/nc4_2026-02-13-Romania_vis_06/nc4_2026-02-13-Romania_0930_vis_06.npy \
    --constants MTG/mtg_constants.json

# Plot regridded output (Romania EPSG:31700 grid)
python our_data/satellite_data/inspect_mtg.py --regridded \
    --npy regridded_data/satellite_data/MTG/vis_06/.../nc4_..._0930_vis_06.npy

# Save .nc without plotting
python our_data/satellite_data/inspect_mtg.py --raw --npy <path> --constants <path> --save_nc --no_plot
```

#### `regrid.py` — MTG branch updates

Two changes were needed to align `regrid.py` with the new pipeline output:

1. **Source grid reconstruction**: previously `regrid_satellite_mtg()` loaded the precomputed `coordinates/lat_{1,2}km.npy` / `lon_{1,2}km.npy` files written by the old pipeline. Those files no longer exist. The new code reads `our_data/satellite_data/MTG/mtg_constants.json`, builds a `pyproj.Proj(proj='geos', h=..., a=..., b=..., lon_0=..., sweep=...)` from the embedded projection parameters, and reconstructs 2-D source lat/lon arrays from `x_geos` / `y_geos` once per resolution. The KD-tree (`PrecomputedMapping`) caching strategy is unchanged.
2. **Input format**: MTG inputs are `.npy` arrays under `satellite_data/MTG/{channel}/nc4_{date}-Romania_{channel}/*.npy` instead of `.nc`. Radar, MSG (disabled), lightning, and NWCSAF paths are untouched.

The regrid output for MTG remains `.npy` on the Romania 1536×768 grid. Use `inspect_mtg.py --regridded` to get a CF NetCDF for any single regridded sample.

#### NWCSAF L2 Data Pipeline (SFTP + arrange)

`pipeline_nwcsaf.py` mirrors `pipeline_msg_mtg.py` but for SAFNWC L2 cloud products (CMIC + CTTH). It connects via SFTP, applies the standard filters (PLAX exclusion, date range, minute cadence), and arranges the result into per-date COALITION-4 directories in a single command.

```bash
# 1. Pick the training cadence first (writes our_data/timestep_config.json)
python validate_timestep.py --step_minutes 15

# 2. Download + arrange NWCSAF for a date range
python our_data/nwcsaf_data/pipeline_nwcsaf.py \
    --start 2026/03/13-0000 \
    --end   2026/05/11-2359 \
    --password_file password.txt
```

**Required arguments:**

| Flag | Description |
|------|-------------|
| `--start`, `--end` | Date/time range in `yyyy/mm/dd-hhmm` (UTC). End is *inclusive*. |
| `--password_file`, `-pw` | Text file containing the SSH password for `safnwc@192.168.11.212` on a single line. Keep out of git. **This is a different account from MTG (`anm@192.168.11.223`).** |

**Optional flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--cache_dir`, `-c` | Flat SFTP cache (kept around for resumability) | `our_data/nwcsaf_data/_raw_data/` |
| `--arranged_root`, `-a` | Root under which `{YYYY-MM-DD}-Romania/` dirs are created | `our_data/nwcsaf_data/` |
| `--products` | Products to ingest: `cmic`, `ctth`, or both | `cmic ctth` |
| `--timesteps` | Override the minute filter (e.g. `00 10 30 40`) or pass `all` for every native :00/.../:50 | read from `timestep_config.json` |
| `--skip_download` | Skip SFTP and only re-arrange what is already in the cache | off |
| `--no_arrange` | Skip the per-date arrange step (download only) | off |
| `--hardlinks` | Hard-link rather than copy when arranging — saves disk space on the same volume, falls back to copy when unsupported | off |

**Two-stage workflow inside the script:**

1. **SFTP download** — connects to `safnwc@192.168.11.212`, lists `/home/safnwc/prod_arch/CMIC/` and `/home/safnwc/prod_arch/CTTH/` via server-side `ls *YYYYMMDDT*.nc` globs, then drops anything that fails date / minute / product / PLAX filters before transferring. Already-present files in the cache are skipped. Lands matching files flat in `_raw_data/` with `[i/total] Downloading <filename>` / `[i/total] Already local: <filename>` progress per file (same format as the MTG pipeline).
2. **Arrange from cache** — non-destructively copies (or hard-links) each cached file into `{arranged_root}/{YYYY-MM-DD}-Romania/`. Re-applies the same filters defensively.

#### NWCSAF Helper Script

**`summarize_raw_data.py`** — scan `_raw_data/`, group files by date, and report:

- **Per-date CSV/table** with CMIC + CTTH file counts, `complete_pairs` (timesteps where both products are present), `incomplete_pairs`, `expected` (`len(minute_filter) × 24`), and `coverage_pct`.
- **Missing-timesteps JSON** with three categorisations per date — `missing_completely_times`, `partial_times` (one product missing), and `per_product_missing` — plus an overall summary.

The "expected" count is derived from `timestep_config.json` (or `--timesteps NN NN ...` override), so coverage stays meaningful no matter which cadence was chosen.

```bash
# Defaults: read cadence from timestep_config.json
python our_data/nwcsaf_data/summarize_raw_data.py

# Check completeness against every native :00 / .../ :50 timestep
python our_data/nwcsaf_data/summarize_raw_data.py --timesteps all

# Custom paths
python our_data/nwcsaf_data/summarize_raw_data.py \
    --raw_dir D:/backup/nwcsaf_raw --output backup_summary.csv \
    --missing backup_missing.json
```

#### OPERA Radar Pipeline (SFTP + reproject)

OPERA is an alternative radar source with two products:

| Product | Native resolution | Cadence | File format |
|---|---|---|---|
| Maximum reflectivity (dBZ) | 2 km | 15 min | HDF5 (`.h5`) |
| Instantaneous rainfall rate (mm/h) | 2 km | 15 min | HDF5 (`.h5`) |

Both products are listed in `product_cadences.json` as `opera_reflectivity: 15` and `opera_rainfall_rate: 15`. They raise the validator floor to 15 min when OPERA is in use — comment them out (rename with a leading underscore) in the cadences file if you're not using OPERA and want a finer training step.

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

Only `.h5` files are transferred; OPERA-internal metadata or index files in the same directory are skipped. Per-file `[i/total] Downloading <filename>` progress, identical to the MTG / NWCSAF pipelines.

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

##### Step 3 — Reproject to the Romania grid (via `regrid.py --opera`)

OPERA reprojection lives inside the unified `regrid.py` (the old standalone `regrid_opera.py` was removed). Reads each `.h5` file's `/where` metadata, builds `pyproj.Proj(projdef)` source projection, projects via `pyresample` KD-tree onto the EPSG:31700 Stereo70 grid (1536×768), and saves one **`.npy` per file** under `our_data/regridded_data/opera_data/{product}/nc4_{date}-Romania_{product}/`. The KD-tree mapping is built **once per product** from the first file and reused across the rest. Day folders run in parallel via the shared `ThreadPoolExecutor`, same as the other product families.

```bash
# Both OPERA products
python regrid.py --opera

# All products in a single run (radar + MTG + lightning + NWCSAF + OPERA)
python regrid.py --all

# Single date / custom worker count
python regrid.py --opera --date 2025-06-15 --workers 8
```

Output schema (one `.npy` per source `.h5`, plus shared sidecars):

| Path | Contents | Notes |
|---|---|---|
| `regridded_data/opera_data/{product}/nc4_{date}-Romania_{product}/nc4_{date}-Romania_{HHMM}_{product}.npy` | `float32` array on the 768×1536 Romania grid | `nodata` → NaN, `undetect` → 0 (no precipitation detected) |
| `regridded_data/opera_data/opera_constants.json` | Source projection per product: `projdef`, `xsize`, `ysize`, `xscale`, `yscale`, LL/UR corner coords | Written once from the first file of each product |
| `regridded_data/romania_grid_lats.npy`, `regridded_data/romania_grid_lons.npy` | Target lat/lon arrays on the Romania grid | Shared across every product (radar / MTG / lightning / NWCSAF / OPERA) |

To rebuild a CF-compliant NetCDF for inspection in Panoply / QGIS, point `inspect_mtg.py --regridded` at one of the `.npy` files — it auto-finds the shared `romania_grid_*.npy` via a walk-up.

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
├── nwcsaf/{cmic, ctth}/*.nc
└── lightning/*.kml
```

Run all arrange scripts from the project root:

```bash
# Radar: raw NetCDF → our_data/radar_data/{product}/nc4_{date}-Romania_{product}/
# Default: only keeps :00, :10, :30, :50 timesteps (nearest to 15-min grid)
python our_data/raw_data/radar_arrange.py --source_root our_data/raw_data/radar/netcdf --target_root our_data/radar_data
# Keep all timesteps (no filtering)
python our_data/raw_data/radar_arrange.py --source_root our_data/raw_data/radar/netcdf --target_root our_data/radar_data --timesteps all

# NWCSAF: see the dedicated section "NWCSAF L2 Data Pipeline (SFTP + arrange)"
# above — pipeline_nwcsaf.py handles download + per-date arrange in a single run.

# Lightning: date-based filenames (dd_mm_yyyy.kml)
python our_data/raw_data/lightning_arrange.py --source_root our_data/raw_data/lightning --target_root our_data/lightning_data

# Lightning: sequential filenames (lightning.kml, lightning (1).kml, ...)
python our_data/raw_data/lightning_arrange.py --source_root our_data/raw_data/lightning --target_root our_data/lightning_data --start-date 2026-03-01 --end-date 2026-03-31
python our_data/raw_data/lightning_arrange.py --source_root our_data/raw_data/lightning --target_root our_data/lightning_data --start-date 2026-03-01 --end-date 2026-03-31 --dry-run

# KML → 15-min NetCDF lightning maps (density, current, occurrence)
python our_data/lightning_data/read_kml_version2.py
python our_data/lightning_data/read_kml_version2.py --date 2025-05-15
```

The sequential lightning arrangement mode also generates a `lightning_filename_mapping.csv` mapping each original filename to its assigned date.

## Architecture Summary

The model uses a multi-branch encoder with resolution-specific input streams:

- **HR branch** (1 km): radar products + lightning + MTG vis_06 → ConvGRU over 3 input timesteps
- **LR branch** (2–3 km): satellite channels + NWCSAF → ConvGRU over 3 input timesteps

Branches merge at matching spatial scales during the encoder's downsampling stages. The decoder generates 3 future frames (T+15, T+30, T+45 min) autoregressively from the encoded state.

- **Lightning target**: weighted focal loss (γ=2) to handle severe class imbalance (~1% positive pixels)
- **Radar target**: categorical cross-entropy with 5 precipitation intensity classes

### Dynamic Model Construction

The model architecture is built dynamically from `metadata.json` saved by `create_datasets.py`. Each dataset records its input group names (e.g. `past_hr`, `past_mr`, `past_lr`), their shapes `[T, H, W, C]`, and the label type. `train_models.py` reads this metadata and:

1. Creates one input branch per group
2. Computes the spatial downsampling factor from the ratio of each input's resolution to the maximum resolution
3. Concatenates inputs that share the same resolution
4. Builds the encoder-decoder with the correct channel counts

This means a dataset created without NWCSAF (omitting `past_lr`) produces a model with no LR branch — no code changes needed. The same applies to any future input configuration changes.

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
| **Classical Shapley** | Source-group contribution by comparing models trained on different input subsets | 4+ trained models |

The script also produces a **prediction diagnostics** panel (MAE, RMSE, predicted-vs-target curves, and a per-sample MAE heatmap across lead times).

### Quick Start

```bash
conda activate tfenv

# Grad-CAM + Xi analysis only (fastest)
python feature_importance_analysis.py \
    --model models/coalition_mtg_lightning.keras \
    --data our_data/datasets/mtg_lightning/test \
    --methods gradcam_xi

# Grad-CAM + Xi + SHAP
python feature_importance_analysis.py \
    --model models/coalition_mtg_lightning.keras \
    --data our_data/datasets/mtg_lightning/test \
    --methods gradcam_xi shap

# All three methods (requires classical Shapley eval CSVs)
python feature_importance_analysis.py \
    --model models/coalition_mtg_lightning.keras \
    --data our_data/datasets/mtg_lightning/test \
    --methods gradcam_xi shap classical_shapley \
    --scores-dir results/eval_scores/
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Path to `.keras` model file | (required) |
| `--data` | Path to saved test dataset directory | (required) |
| `--output` | Output directory for results | `results/feature_importance` |
| `--methods` | Which analyses to run (`gradcam_xi`, `shap`, `classical_shapley`) | `gradcam_xi shap` |
| `--num-samples` | Number of test samples to average Grad-CAM over | `4` |
| `--scores-dir` | Directory with eval CSVs for classical Shapley | `None` |
| `--model-no-nwcsaf` | Path to model trained without NWCSAF (for impact comparison) | `None` |
| `--data-no-nwcsaf` | Test dataset for the no-NWCSAF model | `None` |

### Example: NWCSAF Impact Comparison

Train two models (with and without NWCSAF) and compare their Grad-CAM + Xi patterns:

```bash
# Train the full model (already done)
python train_models.py --mode mtg_lightning --epochs 10

# Train without NWCSAF (modify mode or remove past_lr input)
python train_models.py --mode mtg_lightning_no_nwcsaf --epochs 10

# Compare
python feature_importance_analysis.py \
    --model models/coalition_mtg_lightning.keras \
    --data our_data/datasets/mtg_lightning/test \
    --model-no-nwcsaf models/coalition_mtg_lightning_no_nwcsaf.keras \
    --data-no-nwcsaf our_data/datasets/mtg_lightning_no_nwcsaf/test \
    --methods gradcam_xi
```

### Example: Classical Shapley (4-model design)

With radar always present, permute the remaining 2 source groups (MR satellite, LR/NWCSAF):

| # | Sources | Model to train |
|---|---------|----------------|
| 1 | {HR} | Radar + lightning + MTG vis_06 only |
| 2 | {HR, MR} | Add MTG IR/WV satellite |
| 3 | {HR, LR} | Add NWCSAF |
| 4 | {HR, MR, LR} | Full model (already trained) |

After training all 4 and evaluating each on the test set, place the eval CSVs in a directory:

```
results/eval_scores/
├── eval-lightning-hr.csv
├── eval-lightning-hrmr.csv
├── eval-lightning-hrlr.csv
└── eval-lightning-hrmrlr.csv
```

Then run:

```bash
python feature_importance_analysis.py \
    --model models/coalition_mtg_lightning.keras \
    --data our_data/datasets/mtg_lightning/test \
    --methods gradcam_xi shap classical_shapley \
    --scores-dir results/eval_scores/
```

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
| `classical_shapley.csv` | Shapley value per source |
| `shapley_by_leadtime.png` | Normalised Shapley over lead time |
| `method_comparison.csv` | All methods side-by-side |
| `method_correlations.csv` | Spearman + Pearson between methods |
| `method_comparison.html` | Normalised bar chart comparing methods |
| `nwcsaf_impact.html` | With vs without NWCSAF Xi heatmaps |

## References

- Leinonen, J., et al. (2022). Seamless lightning nowcasting with recurrent-convolutional deep learning. *Monthly Weather Review*, 150(6).
- Czibula, G., et al. (2024). SepConv-based precipitation nowcasting using radar data. *Natural Hazards and Earth System Sciences*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks. *ICCV*.
- Chatterjee, S. (2021). A new coefficient of correlation. *Journal of the American Statistical Association*.

## License

This project is developed at ANM Romania under the EUMETSAT Training Placement Scheme.
