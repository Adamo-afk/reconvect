<div align="center">
  <img src="assets/reconvect-logo.gif" width="480" alt="ReConvect"/>
</div>

# COALITION-4 Nowcasting System — Romanian Adaptation

Recurrent-convolutional nowcasting of **rainfall intensity** and **lightning occurrence** over Romania, adapted from MeteoSwiss COALITION-4 (Leinonen et al.).

An encoder-forecaster (ResBlock + ConvGRU) ingests a multi-resolution stack of MTG FCI satellite channels, OPERA composite radar, and LINET lightning, and predicts three lead times: **T+15, T+30, T+45 min**.

- **Grid:** Romania, EPSG:31700 (Stereo70), **1536 × 768** @ ~1 km — a fixed 6 × 3 array of 18 patches of 256 × 256.
- **Area extent:** `(-177324, 77148, 1331353, 723370)` metres.
- **Cadence:** 15 min (configurable — Step 0).
- **Sample window:** 3 past steps (t−30, t−15, t0) → 3 future steps (t+15, t+30, t+45).

Precipitation is driven by the pan-European **OPERA** composite; lightning by **LINET**. Sample selection is DBSCAN over OPERA `rainfall_rate`, so every artefact is tagged `<mode>_dbscan`.

**Contents:** [Resolution tiers](#resolution-tiers) · [Modes](#modes) · [Setup](#environment-setup) · [Table 1 — Training](#table-1--training-a-model-from-scratch) · [Table 2 — Validation & inference](#table-2--validation-inference-visualisation--analysis) · [Table 3 — Architecture defaults](#table-3--architecture--training-defaults) · [Outputs](#outputs-reference) · [Thresholds](#thresholds-reference) · [Data products](#data-products)

---

## Resolution tiers

Two input tiers. **Always read the physical resolution, not the tier name.**

| Tier | Native | Patch | Pooling | Channels |
|---|---|---|---|---|
| `past_hr` — high resolution | 1 km | 256 × 256 | none | MTG `vis_06`; LINET `density`, `current`, `occurrence` |
| `past_mr` — medium resolution | 2 km | 128 × 128 | 2 × 2 avg | OPERA `reflectivity`, `rainfall_rate`; MTG `ir_38`, `ir_105`, `wv_63`, `wv_73` |

> **Why "medium" when MR is the coarsest tier?** There used to be a third tier (`past_lr`, 3 km, 64 × 64, 4 × 4 pooling) carrying MSG SEVIRI and NWCSAF. Both products were retired and the tier removed, so MR only reads as "middle" relative to that old MSG stack. The name is kept because `past_mr` is baked into the input-tensor names of every trained checkpoint and every dataset's `metadata.json`.

> **On-disk suffix caveat:** extracted patches are named `{variable}_{HHMM}_{HR|LR}.npy` — only **two** suffixes, where `_LR` means *"was pooled"*, not a tier name. Since the LR tier is gone, **every `_LR` patch on disk is MR: 128 × 128 at 2 km.**

### How multi-resolution inputs are reconciled

**1. Reproject everything to the same 1 km grid first.** `reproject.py` applies a precomputed pyresample KD-tree mapping from each product's native CRS onto the shared 1536 × 768 EPSG:31700 canvas. Afterwards every `.npy` is on the *same* grid and a given patch number maps to the same geographic tile across all products. Cross-scale alignment is paid once per cadence step, not at training time.

**2. Pool *down* to native resolution — never up.** `extract_patches.py` slices 256 × 256 1 km tiles, then average-pools 2 km products to 128 × 128. Upsampling would fabricate pixels: the model would see "1 km OPERA" carrying no genuine 1 km information and would overfit interpolation artefacts. Downsampling preserves the real resolution each instrument actually measured, forcing the encoder to cross scales honestly.

**3. Multi-branch encoder, merging at matching scales.**

```
INPUT  past_hr (256×256, 1 km) ─┐
                                ├─ ResBlock + ConvGRU ─stride 2─┐
                                │                                ▼
       past_mr (128×128, 2 km) ─┴────── concat ─────────────────[merge @128]
                                                                 │
                                                                 ▼
                                                            DECODER → T+15/30/45
```

**4. The model rebuilds itself from `metadata.json`.** Each dataset records its input group names and shapes `[T, H, W, C]`. The builder creates one branch per group, derives each branch's downsample factor as `max_res / branch_res` (1 for HR, 2 for MR), concatenates branches that share a resolution, and sizes the encoder-decoder accordingly. A dataset built from any subset of inputs produces a matching model with no code change; the Swin head from `--stage finetune` inherits the backbone's shape the same way.

---

## Modes

The mode name states its own track: `_rainfall` = OPERA rainfall 5-class, `_continuous` = OPERA rainfall regression, `_occurrence` = lightning binary. There are no aliases.

| Mode | HR inputs | MR inputs | Target |
|---|---|---|---|
| `mtg_opera_radar_only_rainfall` | `vis_06` | OPERA ×2 | rainfall 5-class |
| `mtg_opera_mtgmr_rainfall` | `vis_06` | OPERA + MTG IR/WV | rainfall 5-class |
| `mtg_lightning_opera_rainfall` | LINET ×3 + `vis_06` | OPERA + MTG IR/WV | rainfall 5-class |
| `mtg_opera_mtgmr_continuous` | `vis_06` | OPERA + MTG IR/WV | rainfall regression [0,1] |
| `mtg_lightning_opera_occurrence` | LINET ×3 + `vis_06` | OPERA + MTG IR/WV | lightning binary |
| `mtg_opera_occurrence` † | `vis_06` | OPERA + MTG IR/WV | lightning binary |

† **KD student only.** Not buildable via `create_datasets.py` — it trains on the teacher's dataset with `past_hr` sliced.

**Rainfall classes (mm/h):** `0: R<10 · 1: 10–20 · 2: 20–30 · 3: 30–40 · 4: R≥40`. The `_continuous` head regresses the same quantity normalised by 70 mm/h, so it bins back to these 5 classes for a like-for-like SepConv comparison.

`mtg_lightning_opera_rainfall` and `mtg_lightning_opera_occurrence` share an identical input stack and differ only in the label head — the natural dual-target pair to train side by side.

---

## Environment Setup

Requires **Conda**, an **NVIDIA GPU**, and **Windows**. Follow this order exactly; it sidesteps the two Windows traps.

1. **Install CUDA 11.2 + cuDNN 8.1 via the NVIDIA installers** — *not* `conda install cudatoolkit=11.2 cudnn=8.1.0`, which silently downgrades Python 3.10 → 3.9 (conda-forge has no CUDA 11.2 builds for 3.10). Then confirm `CUDA_PATH` is set — Python 3.8+ on Windows no longer honours `PATH` for DLL loading in C extensions, so TensorFlow finds CUDA through `CUDA_PATH` specifically:
   ```powershell
   setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"
   ```
   `setx` only affects newly-launched shells — open a fresh terminal afterwards.
2. **Create the env** (3.10 is the last Python with native Windows TF GPU support):
   ```powershell
   conda create -n tfenv python=3.10 -y; conda activate tfenv
   python --version    # MUST still print 3.10.x at every step below
   ```
3. **Geospatial stack via conda-forge first**, so `pyproj` / `proj` / `geopandas` / `cartopy` / `pyresample` share one PROJ ABI. Pip-installing these on Windows yields mismatched `proj.db` paths and every CRS lookup then fails with `Invalid projection: … no database context specified`:
   ```powershell
   conda install -c conda-forge -y pyproj proj geopandas pyogrio shapely cartopy pyresample pykdtree netCDF4 xarray hdf5plugin h5py
   ```
4. **Remaining deps, then the local package** (`c4dl.projection` is imported by several scripts):
   ```powershell
   pip install -r requirements.txt; pip install -e .
   ```
5. **Verify both PROJ and the GPU:**
   ```powershell
   python -c "import pyproj; print(pyproj.CRS('EPSG:4326'))"
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   An empty `[]` plus `Could not load dynamic library 'cudart64_110.dll'` means this shell can't see `CUDA_PATH` — reopen the terminal.

---

## Table 1 — Training a model from scratch

Run in step order. Steps 9a/9b are conditional on the track; 11–12 are optional.

> **Reading the tables.** Tokens in $\color{red}{\textbf{\textit{red bold italic}}}$ are **placeholders you must supply** — they are not in this repository. Everything else in the example commands is a literal path the pipeline produces or reads.
>
> **Date formats differ by script** — there are two conventions, and the end bound is not uniform:
>
> | Format | Used by | End bound |
> |---|---|---|
> | `yyyy/mm/dd-hhmm` | `pipeline_msg_mtg.py`, `pipeline_opera.py` | OPERA **inclusive**; MTG unspecified |
> | `YYYY-MM-DD` | `linet_export.py`, `read_kml_version2.py`, `reproject.py`, `identify_patches.py`, `extract_patches.py`, `predict_full_domain.py`, `validate_predictions.py`, `evaluate_*.py` | `linet_export` **EXCLUSIVE**; `identify_patches` **inclusive** |
> | `HH:MM` · integer | `--time`, `--start-time`, `--end-time` · `--hour` 0–23, `--year`, `--month` 1–12 | — |
>
> The exclusive bound on `linet_export.py` is the one that bites: `--end 2025-05-31` silently stops at 30 May.

| # | Script | Arguments | What the step does | Commands |
|---|---|---|---|---|
| **0** | `validate_timestep.py` | `--step_minutes N` desired master training cadence, in minutes · `--cadences_file PATH` per-product native cadence config file · `--output_path PATH` where the generated timestep config lands · `--print` show the existing config and exit | Picks the master cadence and derives each product's minute filter → `our_data/timestep_config.json`, read by every later step. | `python validate_timestep.py --step_minutes 15`<br>`python validate_timestep.py --print` |
| **1a** | `our_data/satellite_data/pipeline_msg_mtg.py` | `--start` `--end` `yyyy/mm/dd-hhmm` download window, both required · `--password_file PATH` text file holding the SSH password · `--products_file PATH` JSON listing which FCI channels to fetch · `--output_dir PATH` destination root for downloaded MTG data · `--timesteps` override the per-product minute filter · `--workers N` parallel download and processing worker count · `--full_disk` fetch full disk instead of Romania chunks · `--skip_download` reprocess local files without downloading again · `--fill-gaps` backfill shortfall from the EUMETSAT Data Store · `--eumdac_credentials PATH` two-line EUMDAC key and secret | Downloads + pre-processes MTG FCI L1C. | `python our_data/satellite_data/pipeline_msg_mtg.py --start 2025/05/01-0000 --end 2025/05/31-2350 --password_file `$\color{red}{\textbf{\textit{creds.txt}}}$ |
| **1b** | `our_data/opera_data/pipeline_opera.py` | `--start` `--end` `yyyy/mm/dd-hhmm` window, **end inclusive** · `--ssh_key PATH` SSH private key; excludes `--password_file` · `--password_file PATH` text file holding the SSH password · `--products` reflectivity, rainfall_rate, or both by default · `--remote_base PATH` remote EWC mount root directory · `--remote_host` `--remote_user` SSH endpoint and login account · `--cache_dir PATH` local OPERA destination root directory · `--timesteps` override the per-product minute filter | Fetches OPERA composite HDF5. If the default `--remote_base /eumetsatdata` errors with "No such file", pass `/home/eumetsatdata` — the mount root differs between EWC images. | `python our_data/opera_data/pipeline_opera.py --start 2025/05/01-0000 --end 2025/05/31-2359 --ssh_key `$\color{red}{\textbf{\textit{path/to/ssh-key}}}$<br>`… --remote_base /home/eumetsatdata` |
| **1c** | `our_data/lightning_data/linet_export.py` | `--start` `--end` `YYYY-MM-DD` period, **end EXCLUSIVE** · `--format` txt point list, kml, or asc · `--out PATH` destination root for the exported strokes · `--bbox` lon/lat rectangle limiting the export area · `--password_file PATH` text file holding the LINET password · `--lightning-type` 0 all, 1 cloud-to-ground, 2 intracloud · `--amp-threshold` minimum stroke amplitude to keep · `--daily-window` split the request into per-day windows · `--pause` seconds to wait between successive requests · `--force` re-download even when the output exists · `--dry-run` plan the fetch without downloading anything | Downloads LINET strokes. Use `--format kml`: it writes `{out}/kml_data/YYYY-MM-DD/…` which the rasteriser reads directly. | `python our_data/lightning_data/linet_export.py --start 2025-05-01 --end 2025-06-01 --format kml --out our_data/lightning_data --password_file `$\color{red}{\textbf{\textit{creds.txt}}}$<br>*(`--end 2025-06-01` to cover all of May — the bound is exclusive)* |
| **1d** | `our_data/lightning_data/read_kml_version2.py` | `--data_root PATH` root holding the downloaded KML files · `--output_root PATH` destination for the rasterised grid arrays · `--date YYYY-MM-DD` process one date instead of all · `--force` reprocess and overwrite the existing outputs | Rasterises strokes onto the 1 km Romania grid → `density`, `current`, `occurrence`. | `python our_data/lightning_data/read_kml_version2.py --data_root our_data` |
| **1e** | `our_data/satellite_data/summarize_mtg.py` | `--raw_dir PATH` directory of downloaded FCI chunk files · `--output PATH` per-date coverage summary CSV destination · `--missing PATH` missing-timestep JSON destination · `--timesteps` override the cadence minute filter · `--fill-from-datastore` backfill gaps from the EUMETSAT Data Store · `--eumdac_credentials PATH` two-line EUMDAC key and secret · `--fill-dry-run` list what would be fetched only · `--no-fill-incomplete` skip cycles missing just one chunk | Per-date MTG coverage against the cadence grid → `mtg_summary.csv` + `mtg_missing_timesteps.json`, both consumed by step 3. Optionally backfills the shortfall — see the section below. | `python our_data/satellite_data/summarize_mtg.py`<br>`python our_data/satellite_data/summarize_mtg.py --fill-from-datastore --fill-dry-run` |
| **1f** | `our_data/opera_data/summarize_opera_data.py` | `--data_dir PATH` local OPERA download root to scan · `--products` reflectivity, rainfall_rate, or both · `--timesteps` override the per-product minute filter · `--output PATH` per-date coverage summary CSV destination · `--missing PATH` missing-timestep JSON destination | Same for OPERA → `opera_summary.csv` + `opera_missing_timesteps.json`. Reports completeness per product against the 15-min grid. | `python our_data/opera_data/summarize_opera_data.py`<br>`python our_data/opera_data/summarize_opera_data.py --products opera_rainfall_rate` |
| **1g** | `our_data/lightning_data/summarize_lightning_data.py` | `--data_dir PATH` rasterised lightning `.npy` root to scan · `--output PATH` per-date coverage summary CSV destination · `--active PATH` per-timestep activity index CSV destination | Same for LINET → `lightning_summary.csv` + `lightning_active_steps.csv`. The activity index also feeds `lightning_fraction.py --scope_csv` and `visualize_lightning_stats.py`. | `python our_data/lightning_data/summarize_lightning_data.py` |
| **2** | `reproject.py` | `--satellite MTG` \| `--lightning` \| `--opera` \| `--all` product family; mutually exclusive, one required · `--data_root PATH` root containing the raw product folders · `--date YYYY-MM-DD` process a single date only · `--workers N` parallel day-folder worker processes | Regrids everything onto the 1536 × 768 EPSG:31700 canvas as `.npy`. Also writes the shared `romania_grid_{lats,lons}.npy` and per-source projection constants so the arrays stay self-recoverable. | `python reproject.py --all`<br>`python reproject.py --opera --workers 6`<br>`python reproject.py --satellite MTG --date 2025-05-14` |
| **3** | `intersect_product_coverage.py` | `--summary name=PATH` per-product coverage summary CSV · `--missing name=PATH` per-product missing-timestep JSON · `--active name=PATH` per-timestep activity index CSV · `--errors_log PATH` reprojection error log to subtract · `--timestep_config PATH` master cadence config to validate against · `--output_csv PATH` where the timestep manifest is written · `--output_plot PATH` destination for the coverage bar chart | Intersects per-product coverage into `timestep_manifest.csv` — the timesteps where *all* products exist. Gates step 5. | `python intersect_product_coverage.py --summary mtg=mtg_summary.csv --summary opera=opera_summary.csv` |
| **4** | `identify_patches.py` | `--threshold` mm/h rain-rate floor for DBSCAN clustering · `--eps` DBSCAN neighbourhood radius, in pixels · `--min_samples` minimum pixels needed to accept a cluster · `--data_root PATH` root holding the reprojected OPERA data · `--output_dir PATH` destination for the patch index CSV/JSON · `--date YYYY-MM-DD` single date; excludes `--start`/`--end` · `--start` `--end` `YYYY-MM-DD` range, **both bounds inclusive** · `--plot` save one PNG per active timestamp | DBSCAN over OPERA `rainfall_rate`; marks which of the 18 patches are convectively active per timestep → `patch_index.csv`. | `python identify_patches.py`<br>`python identify_patches.py --start 2025-05-01 --end 2025-05-31`<br>`python identify_patches.py --date 2025-05-14 --plot` |
| **5** | `extract_patch_seq_for_datasets.py` | `--past N` past steps required before the reference · `--future N` future steps required after the reference · `--test_frac` fraction of each block held for test · `--val_frac` fraction of each block held for validation · `--block_hours N` temporal block size; must divide 24 · `--manifest PATH` coverage gate; `none` disables the filter · `--data_root PATH` root holding the patch index CSV | Builds temporally-continuous sequences and the Czibula block-wise 80/10/10 split → `{train,validation,test}_data_dbscan.csv` + `sequence_meta_dbscan.json`. | `python extract_patch_seq_for_datasets.py`<br>`python extract_patch_seq_for_datasets.py --past 3 --future 3 --block_hours 12`<br>`python extract_patch_seq_for_datasets.py --manifest none` |
| **6** | `extract_patches.py` | `--products` satellite_MTG, lightning, opera; default all three · `--data_root PATH` root holding the reprojected full-domain canvases · `--output_dir PATH` destination for the sliced patch arrays · `--date YYYY-MM-DD` process a single date only | Slices 256 × 256 patches from the reprojected canvases, applying each variable's pooling factor → `patches/{date}/{var}_{HHMM}_{HR\|LR}.npy`. | `python extract_patches.py`<br>`python extract_patches.py --products satellite_MTG opera`<br>`python extract_patches.py --date 2025-05-14` |
| **7** | `compute_normalization_stats.py` | `--variables` restrict to a subset of variables · `--sample_fraction` fraction of pixels sampled per file · `--with_percentiles` also emit p01/p50/p99 and MAD · `--reservoir_size` reservoir sample size backing the percentiles · `--device auto\|cpu\|gpu` compute backend for the accumulation · `--no_split_filter` **DIAGNOSTIC ONLY — leaks val/test data** · `--train_csv PATH` training split scoping the statistics · `--sequence_meta PATH` sequence schema describing the sample window · `--timestep_config PATH` master cadence for per-product snapping · `--reproject_root PATH` root holding the reprojected product data · `--output PATH` destination for the statistics JSON · `--seed` RNG seed for reproducible pixel sampling | Per-variable mean/std over the **training split only** → `normalization_stats_dbscan.json`. Required by step 8; there is no fallback, and a missing variable fails loudly. | `python compute_normalization_stats.py`<br>`python compute_normalization_stats.py --variables ir_105 opera_rainfall_rate`<br>`python compute_normalization_stats.py --device gpu --with_percentiles` |
| **8** | `create_datasets.py` | `--mode` one of five buildable dataset modes · `--data_root PATH` root holding split CSVs and patches · `--output_root PATH` destination root for the TFRecord datasets | Applies transforms + label binning, writes TFRecord shards plus a per-split `metadata.json` (input shapes, label type, cadence) that drives model construction. | `python create_datasets.py --mode mtg_opera_mtgmr_rainfall`<br>`python create_datasets.py --mode mtg_lightning_opera_occurrence`<br>`python create_datasets.py --mode mtg_opera_mtgmr_continuous` |
| **9a** | `lightning_fraction.py` | `--scope_csv PATH` scope CSV; `none` scans every file · `--data_root PATH` root holding the lightning patch arrays · `--output PATH` destination for the focal-loss prior JSON | **`_occurrence` modes only** (`mtg_lightning_opera_occurrence`, `mtg_opera_occurrence`). Training-scope positive-pixel fraction → `lightning_fraction_dbscan.json`, the focal-loss prior. | `python lightning_fraction.py`<br>`python lightning_fraction.py --scope_csv none` |
| **9b** | `opera_rainfall_fraction.py` | `--scope_csv PATH` scope CSV; `none` scans every file · `--data_root PATH` root holding the OPERA patch arrays · `--output PATH` destination for the class-weight prior JSON | **`_rainfall` modes only, when `[radar_loss].weighting != none`.** Not needed by `_continuous`, which minimises weighted MSE. Per-class pixel fractions → `opera_rainfall_fraction_dbscan.json`, the class-weight prior. | `python opera_rainfall_fraction.py` |
| **10** | `train_models.py` | `--config PATH` training config carrying all hyperparameters · `--mode` train one mode instead of the list · `--stage base\|finetune\|both` base training, Swin head, or both · `--base_checkpoint PATH` frozen backbone for the finetune stage · `--dataset_dir PATH` override the derived dataset directory · `--output_dir PATH` destination for saved models and history · `--data_root PATH` root holding datasets and loss priors · `--fresh` ignore any saved per-epoch checkpoint · `--list-modes` print the mode registry and exit | Builds the encoder-forecaster from `metadata.json` and trains. `finetune` freezes the backbone and grafts a Swin head; `both` runs the two back-to-back in one process. Resumes from the per-epoch checkpoint unless `--fresh`. | `python train_models.py --list-modes`<br>`python train_models.py --mode mtg_opera_mtgmr_rainfall --stage base`<br>`python train_models.py --mode mtg_lightning_opera_occurrence --stage both`<br>`python train_models.py --config training.config`<br>`python train_models.py --mode mtg_opera_mtgmr_rainfall --stage base --fresh` |
| **11** | `train_lightning_kd.py` | `--kd_alpha` weight on the soft-teacher distillation loss · `--kd_temperature` softening temperature for both models' outputs · `--teacher_finetuned` distil from the Swin teacher instead · `--epochs` `--batch_size` training length and samples per step · `--learning_rate` Adam learning rate for the student · `--patience` early-stopping patience, counted in epochs · `--shuffle_buffer` samples held in RAM for shuffling · `--seed` RNG seed for reproducible student training · `--no_mixed_precision` disable fp16 compute on tensor cores · `--data_root` `--model_dir` dataset root and checkpoint destination | **Optional.** Distils the teacher into `mtg_opera_occurrence`, a student predicting lightning from satellite + OPERA with **no LINET at inference**. | `python train_lightning_kd.py`<br>`python train_lightning_kd.py --kd_alpha 0.5 --kd_temperature 6.0`<br>`python train_lightning_kd.py --teacher_finetuned` |
| **12** | `sepconv_ensemble_training.py` | `--mode` continuous-target mode supplying the training dataset · `--lead 1\|2\|3` train one lead only, not all · `--epochs` `--batch_size` training length and samples per step · `--data_root` `--model_dir` dataset root and checkpoint destination | **Optional baseline.** Three SepConv regression models (one per lead) on the same inputs as the continuous COALITION-4 mode. | `python sepconv_ensemble_training.py --mode mtg_opera_mtgmr_continuous`<br>`python sepconv_ensemble_training.py --mode mtg_opera_mtgmr_continuous --lead 1` |

### Ad-hoc helpers (no CLI)

Three scripts under `our_data/satellite_data/` are one-off checks with hardcoded
paths rather than pipeline steps, so they take no arguments and are not in the
tables above: `check_chunk_contents.py` (dump the variables inside a downloaded
FCI chunk), `check_chunk_names.py` (verify chunk numbering against the Romania
area), and `check_sign_convention_lat_lon.py` (confirm lat/lon ordering before
trusting a reprojection). Edit the path at the top and run directly.

`lightning_postproc.py`, `pipeline_config.py`, and
`our_data/satellite_data/datastore_fill.py` are imported libraries, not
entry points.

### MTG gap backfill from the EUMETSAT Data Store (step 1a)

The NMA internal server is the primary MTG source. Whatever never arrived there can be recovered from the EUMETSAT Data Store — opt-in, so a routine run never generates Data Store traffic on its own.

The cycle is **summarise → fill → re-summarise**, and it runs entirely inside `summarize_mtg.py`:

```bash
# after a download, in one step
python our_data/satellite_data/pipeline_msg_mtg.py --start … --end … \
    --password_file creds.txt --fill-gaps --eumdac_credentials eumdac.txt

# or standalone, to inspect the gaps before committing to a download
python our_data/satellite_data/summarize_mtg.py --fill-from-datastore --fill-dry-run
python our_data/satellite_data/summarize_mtg.py --fill-from-datastore \
    --eumdac_credentials eumdac.txt
```

| Detail | Behaviour |
|---|---|
| Collection | **FDHSI** (`EO:EUM:DAT:0662`) only — it carries all five channels at the resolutions used. HRFI would add `vis_06` at 500 m, which the pipeline pools away. |
| What gets filled | Both `missing_times` (no chunks) and `incomplete_times` (one of the two Romania chunks). `--no-fill-incomplete` restricts it to fully-absent cycles. |
| Chunks | 35 and 36, matching `ROMANIA_CHUNKS`. |
| Landing place | Straight into `_raw_chunks/` under native Data Store filenames — the same convention `parse_fci_filename` already reads, so no renaming step exists. |
| Credentials | `--eumdac_credentials PATH` (two lines: key, then secret), or `EUMDAC_KEY` / `EUMDAC_SECRET`. Get a key at <https://api.eumetsat.int/api-key/>. |
| Dependency | `eumdac`, imported lazily — the summary runs normally without it and only errors if the backfill is requested. |

**What gets recorded.** After the second summary, a `datastore_fill` block is written into `mtg_missing_timesteps.json` alongside the refreshed `mtg_summary.csv`:

```json
"datastore_fill": {
  "collection_id": "EO:EUM:DAT:0662",
  "files_downloaded": 42,
  "files": ["W_XX-EUMETSAT-Darmstadt,...,_0071_0035.nc", "…"],
  "timesteps_filled": { "2025-05-14": ["11:40", "11:50"] },
  "coverage_before_pct": 95.9, "coverage_after_pct": 99.1,
  "missing_before": 416, "missing_after": 38,
  "skipped_already_present": 0, "errors": []
}
```

The same count and file list print to the console. Partial downloads are deleted rather than left behind, so a re-run resumes cleanly and files already on disk are skipped.

### OPERA SFTP notes (step 1b)

Two distinct failures both surface as `cannot list …` — tell them apart by what follows.

| Symptom | Cause | What to do |
|---|---|---|
| `cannot list …: Permission denied` | The EWC VM rejects password authentication. | Switch to key auth: `--ssh_key ~/.ssh/id_ed25519`, or any other key registered under `claudiu@` on the server. |
| `cannot list …: No such file`, per date, at a mount root that already resolved | None — this is upstream. `--remote_base` auto-fallback already picked the correct EWC mount, so the remote directories genuinely do not exist for those dates. | Nothing to fix locally. Ask the NMA (National Meteorological Administration) data operators when the target range will land. |

```bash
python our_data/opera_data/pipeline_opera.py \
    --start 2025/01/01-0000 --end 2026/08/14-2359 \
    --ssh_key ~/.ssh/id_ed25519
```

---

## Table 2 — Validation, inference, visualisation & analysis

| # | Script | Arguments | What the step does | Commands |
|---|---|---|---|---|
| **1** | `evaluate_coalition.py` | `--mode` model variant to evaluate; required · `--split train\|validation\|test` which dataset split to score · `--finetuned` \| `--kd` evaluate Swin head or KD student · `--threshold` fixed decision threshold; else optimised on validation · `--plot_threshold` probability floor for the lightning visualisation · `--date YYYY-MM-DD` `--hour 0-23` reference for the sample figure · `--batch_size` samples per inference step · `--data_root` `--model_dir` `--output_dir` dataset, checkpoint and results locations | Full metric suite on a held-out split → `evaluation/eval_<run_tag>/evaluation_results.json` + plots. Per-lead metrics feed the coalition study. | `python evaluate_coalition.py --mode mtg_opera_mtgmr_rainfall`<br>`python evaluate_coalition.py --mode mtg_lightning_opera_occurrence --finetuned`<br>`python evaluate_coalition.py --mode mtg_opera_occurrence --kd`<br>`python evaluate_coalition.py --mode mtg_opera_mtgmr_rainfall --split validation` |
| **2** | `evaluate_sepconv_ensemble.py` | `--mode` continuous-target mode the baseline was trained on · `--split train\|validation\|test` which dataset split to score · `--date YYYY-MM-DD` `--hour 0-23` reference for the sample figure · `--batch_size` samples per inference step · `--data_root` `--model_dir` `--output_dir` dataset, checkpoint and results locations | Evaluates the SepConv baseline. Continuous predictions are also binned to the 5 rainfall classes so both comparison modes are reported. | `python evaluate_sepconv_ensemble.py --mode mtg_opera_mtgmr_continuous` |
| **3** | `validate_predictions.py` | `--track rainfall\|lightning\|kd` which validation pipeline runs; required · `--year` `--month 1-12` period to scan; both required · `--date YYYY-MM-DD` switches from extraction to visualisation mode · `--mode` model whose predictions are validated · `--finetuned` \| `--kd` validate Swin head or KD student · `--stride` Hann overlap stride for lightning inference · `--lightning_low_threshold` hysteresis LOW on the probability canvas · `--rainfall_threshold_mmh` rain-rate cut for sample selection · `--high_coverage_pct` coverage grade recorded in the summary · `--teacher_mode` `--student_mode` mode pair for the KD track · `--teacher_finetuned` `--no_student_kd` variant toggles for the KD pair · `--batch_size` samples per inference step · `--data_root` `--model_dir` `--output_dir` dataset, checkpoint and results locations | **Extraction (no `--date`):** scans the month for samples with ≥1 pixel ≥10 mm/h, runs inference, tunes the per-lead hysteresis HIGH by maximising aggregate CSI, writes CSV + summary JSON + metrics figure. **Visualisation (`--date`):** plots overlays for that day using the tuned thresholds. `kd` runs teacher and student on identical samples and tunes each independently. | `python validate_predictions.py --track rainfall --year 2025 --month 5`<br>`python validate_predictions.py --track rainfall --year 2025 --month 5 --finetuned`<br>`python validate_predictions.py --track lightning --year 2025 --month 5 --mode mtg_lightning_opera_occurrence`<br>`python validate_predictions.py --track lightning --year 2025 --month 5 --mode mtg_lightning_opera_occurrence --date 2025-05-14`<br>`python validate_predictions.py --track kd --year 2025 --month 5` |
| **4** | `predict_full_domain.py` | `--mode` model variant to run; required · `--date YYYY-MM-DD` reference date to predict; required · `--time HH:MM` a single reference time · `--hour 0-23` every step-aligned reference within one hour · `--start-time` `--end-time` `HH:MM` inclusive reference-time range · `--finetuned` \| `--kd` run Swin head or KD student · `--validation_summary PATH` load tuned per-lead hysteresis thresholds · `--lightning_low_threshold` `--lightning_high_threshold` hysteresis pair, lightning head · `--rainfall_low_threshold` `--rainfall_high_threshold` hysteresis pair, rainfall head · `--stride` Hann overlap stride for lightning inference · `--patches` restrict to a patch subset, rainfall only · `--threshold` fixed binarisation threshold overriding the default · `--batch_size` samples per inference step · `--no-plot` `--save-npy` skip PNGs; dump raw prediction canvases · `--data_root` `--model_dir` `--output_dir` data, checkpoint and output locations | **Operational inference on any date.** Reads the reprojected full-domain fields and slices all 18 patches on the fly — touches no `patch_index.csv`, split CSV, or pre-extracted patch tile, so it runs on dates the training pipeline has never seen. | `python predict_full_domain.py --mode mtg_opera_mtgmr_rainfall --date 2026-06-30`<br>`python predict_full_domain.py --mode mtg_opera_mtgmr_rainfall --date 2026-06-30 --hour 14`<br>`python predict_full_domain.py --mode mtg_lightning_opera_occurrence --date 2026-06-30 --time 14:30`<br>`python predict_full_domain.py --mode mtg_lightning_opera_occurrence --date 2026-06-30 --validation_summary validation/lightning_2025_05_summary.json`<br>`python predict_full_domain.py --mode mtg_opera_mtgmr_rainfall --date 2026-06-30 --start-time 12:00 --end-time 15:45 --save-npy` |
| **5** | `visualize_gt_vs_pred.py` | `--csv PATH` split CSV listing candidate references; required · `--mode` model variant to visualise; required · `--top_n N` how many highest-activity references to plot · `--finetuned` \| `--kd` visualise Swin head or KD student · `--eval_results PATH` evaluation JSON supplying the decision threshold · `--threshold` fixed threshold overriding the evaluation JSON · `--no_zoom` skip the per-patch zoom figure · `--no_aggregate_graphs` skip the rainfall aggregate plots · `--stride` Hann overlap stride for lightning inference · `--lightning_low_threshold` `--lightning_high_threshold` hysteresis pair, lightning head · `--rainfall_low_threshold` `--rainfall_high_threshold` hysteresis pair, rainfall head · `--validation_summary PATH` load tuned per-lead hysteresis thresholds · `--batch_size` samples per inference step · `--data_root` `--model_dir` `--output_dir` dataset, checkpoint and output locations | **Training-scope visualiser.** Ranks references in a split by qualifying-patch count and renders GT beside predictions, plus a zoom on the patch with the most GT activity. Hysteresis knobs mirror `predict_full_domain.py` so both render identical post-processing. | `python visualize_gt_vs_pred.py --csv our_data/test_data_dbscan.csv --mode mtg_opera_mtgmr_rainfall`<br>`python visualize_gt_vs_pred.py --csv our_data/validation_data_dbscan.csv --mode mtg_lightning_opera_occurrence --top_n 3`<br>`python visualize_gt_vs_pred.py --csv our_data/test_data_dbscan.csv --mode mtg_opera_occurrence --kd --no_zoom` |
| **6** | `generate_report.py` | `--year` `--month 1-12` reporting period; both required · `--track rainfall\|lightning\|both` which tracks the report covers · `--language ro\|en` output language; `en` skips translation · `--bilingual` render Romanian and English side by side · `--model TAG` Ollama model tag used for generation · `--temperature` sampling temperature; 0.0 collapses Gemma · `--seed` Ollama seed for reproducible generation · `--max_tokens` hard cap on tokens per call · `--refresh_cache` `--no_cache` rebuild or bypass the translation cache · `--skip_pdf` run generation without rendering the PDF · `--pred_coupling` couple cells from predictions, not ground truth · `--validation_dir` `--assets_dir` `--output` inputs, banner assets, PDF destination · `--data_root` `--model_dir` dataset and checkpoint locations | Builds a PDF from `validate_predictions.py` outputs with commentary from a local Ollama LLM. `--language en` skips the Romanian translation phase and halves the LLM calls. `--track both` requires both tracks' extraction outputs on disk. | `python generate_report.py --year 2025 --month 5`<br>`python generate_report.py --year 2025 --month 5 --language en`<br>`python generate_report.py --year 2025 --month 5 --track lightning --bilingual`<br>`python generate_report.py --year 2025 --month 5 --skip_pdf --no_cache` |
| **7** | `bundle_eval_scores.py` | `--mode MODE=LETTERS` repeatable mode-to-coalition-letter mapping · `--prefix` filename prefix for the emitted CSVs · `--metric` override the auto-detected scoring metric · `--eval_root` `--output_dir` evaluation source and CSV destination · `--finetuned` read the Swin head's evaluation results | Converts each mode's `evaluation_results.json` into the per-lead-time CSVs classical Shapley expects. Coalition letters encode which input groups a model saw (`o` = OPERA only, `om` = + MTG IR/WV). | `python bundle_eval_scores.py`<br>`python bundle_eval_scores.py --metric HSS`<br>`python bundle_eval_scores.py --mode "mtg_opera_radar_only_rainfall=o" --mode "mtg_opera_mtgmr_rainfall=om"` |
| **8** | `feature_importance_analysis.py` | `--model PATH` trained checkpoint to analyse · `--data PATH` test dataset directory feeding the analysis · `--output PATH` destination for the figures and CSVs · `--methods` gradcam_xi, shap, classical_shapley; repeatable · `--num-samples` how many samples to average over · `--scores-dir PATH` per-leadtime CSVs for classical Shapley · `--model-ablated` `--data-ablated` second model and dataset for ablation | Grad-CAM + Xi correlation (spatial attention), SHAP (pixel importance), and classical Shapley (source-level). The ablation pair diffs two Xi matrices to show how remaining inputs absorb a dropped group's role. | `python feature_importance_analysis.py --model models/coalition_mtg_opera_mtgmr_rainfall_dbscan.keras --data our_data/datasets/mtg_opera_mtgmr_rainfall_dbscan/test --output results/fi --methods gradcam_xi`<br>`… --methods gradcam_xi shap`<br>`… --model-ablated models/coalition_mtg_opera_radar_only_rainfall_dbscan.keras --data-ablated our_data/datasets/mtg_opera_radar_only_rainfall_dbscan/test` |
| **9** | `data_statistics.py` | `--split train\|validation\|test` which split to summarise · `--csv PATH` explicit CSV overriding the split default · `--data_root PATH` root holding the per-split CSVs | Six dataset diagnostic panels: diurnal cycle, spatial heatmap, daily timeline, simultaneously-active patches, samples per date, patch survival. | `python data_statistics.py`<br>`python data_statistics.py --split test` |
| **10** | `our_data/satellite_data/inspect_mtg.py` | `--raw` \| `--reprojected` which grid the input array is on · `--npy PATH` the MTG array to inspect · `--constants PATH` projection constants JSON for the raw grid · `--grid_dir PATH` directory holding the Romania grid coords · `--channel NAME` channel name, else inferred from filename · `--save_nc` also write a NetCDF beside the array · `--save_png PATH` save the figure instead of showing it · `--no_plot` skip plotting entirely | Renders a single MTG `.npy` on either the native geostationary grid or the reprojected Romania grid. Use it to confirm a channel landed correctly before trusting a whole reprojection run. | `python our_data/satellite_data/inspect_mtg.py --reprojected --npy path/to/vis_06.npy`<br>`python our_data/satellite_data/inspect_mtg.py --raw --npy path/to/chunk.npy --save_nc` |
| **11** | `our_data/lightning_data/inspect_lightning.py` | `--npy PATH` reprojected lightning array to inspect · `--output PATH` destination for the rendered figure · `--grid_dir PATH` directory holding the Romania grid coords | Same idea for a rasterised LINET field — a quick check that `read_kml_version2.py` put strokes where they belong on the 1 km grid. | `python our_data/lightning_data/inspect_lightning.py --npy path/to/occurrence.npy` |
| **12** | `our_data/lightning_data/visualize_lightning_stats.py` | `--csv PATH` lightning activity index CSV to plot · `--output_dir PATH` destination for the generated bar charts | Per-day and per-timestep lightning activity bar charts from `lightning_active_steps.csv` (step 1g). Plots only — reads no model. | `python our_data/lightning_data/visualize_lightning_stats.py` |

**Ablation pairs** for step 8 — the mode set is already an ablation ladder:

| Full model | Ablated model | Isolates |
|---|---|---|
| `mtg_opera_mtgmr_rainfall` | `mtg_opera_radar_only_rainfall` | MTG IR/WV |
| `mtg_lightning_opera_rainfall` | `mtg_opera_mtgmr_rainfall` | LINET lightning |
| `mtg_lightning_opera_occurrence` | `mtg_opera_occurrence` (KD student) | LINET lightning |

### End-to-end recipe for a new date

Steps 1–5 fetch and reproject; 6–7 run inference. No training artefact is touched.

```bash
# 1-3. Acquire MTG, OPERA, LINET for the date (see Table 1, steps 1a-1c)
# 4. Rasterise LINET onto the Romania grid
python our_data/lightning_data/read_kml_version2.py --data_root our_data --date 2026-06-30
# 5. Reproject MTG + OPERA
python reproject.py --satellite MTG --date 2026-06-30
python reproject.py --opera       --date 2026-06-30
# 6. Rainfall inference (5-class)
python predict_full_domain.py --mode mtg_opera_mtgmr_rainfall --date 2026-06-30
# 7. Lightning inference (binary occurrence)
python predict_full_domain.py --mode mtg_lightning_opera_occurrence --date 2026-06-30 \
    --validation_summary validation/lightning_2025_05_summary.json
```

---

## Table 3 — Architecture & training defaults

Everything under **COALITION-4** lives in [`training.config`](training.config); per-mode overrides go in `[mode.<name>]`.

| Parameter | Value | Scope | Source |
|---|---|---|---|
| **Architecture** ||||
| Encoder | ResBlock + ConvGRU (`ResGRU`), channels `[32, 64, 128]` | all | built from `metadata.json` |
| Decoder | reversed `[128, 64, 32]`, bilinear upsampling + skip connections | all | — |
| Input branches | one per tier, merged at matching scales | all | `INPUT_GROUP_KEYS` |
| Output head | `Conv2D(1, 1×1, sigmoid)` | lightning | — |
| Output head | `Conv2D(1, 1×1, sigmoid)` | rainfall continuous | — |
| Output head | `Conv2D(5, 1×1, softmax)` | rainfall 5-class | — |
| Past / future timesteps | 3 / 3 | all | `sequence_meta_dbscan.json` |
| **Base training** (`[defaults]`) ||||
| Optimizer | `Adam(lr=1e-3)` | base stage | — |
| Loss | `WeightedFocalLoss(gamma=2.0)`, prior from `lightning_fraction_dbscan.json` (~1 % positive pixels) | lightning | — |
| Loss | `WeightedFocalCategoricalCrossentropy` (see `[radar_loss]`) | rainfall 5-class | — |
| Loss | weighted MSE, bins `[15, 1, 2, 7, 15, 30, 1000]` — **identical to the SepConv baseline** | rainfall continuous | `RAINFALL_MSE_WEIGHTS` |
| Metrics | `iou_metric`, `true_pos`, `false_pos`, `false_neg` | lightning | — |
| Metrics | `accuracy` | rainfall 5-class | — |
| Metrics | `mae`, `mse` | rainfall continuous | — |
| Epochs / batch size | `20` / `32` | all | `[defaults]` |
| Dropout / normalisation | `0.1` / `layer` (`none` \| `batch` \| `layer`) | all | `[defaults]` |
| Shuffle buffer / seed | `256` samples / `0` | all | `[defaults]` |
| Mixed precision | `true` (fp16 on tensor cores) | all | `[defaults]` |
| **LR schedule** (`[lr_schedule]`) ||||
| Type | `cosine_warmup` — linear ramp `min_lr → initial_lr`, then cosine decay back | base stage | `[lr_schedule]` |
| Initial / min LR | `1e-3` / `1e-6` | base stage | `[lr_schedule]` |
| Warmup epochs | `3` | base stage | `[lr_schedule]` |
| **Early stopping** (`[early_stopping]`) ||||
| Monitor / mode | `val_loss` / `min` | all | `[early_stopping]` |
| Patience / min delta | `6` / `1e-5` | all | `[early_stopping]` |
| Restore best weights | `true` | all | `[early_stopping]` |
| **Radar loss** (`[radar_loss]`) ||||
| Weighting | `median` (`inverse` \| `median` \| `none`) | rainfall 5-class | `[radar_loss]` |
| Focal gamma / alpha max | `2.0` / `100.0` (class-weight cap) | rainfall 5-class | `[radar_loss]` |
| Label smoothing | `0.01` | rainfall 5-class | `[radar_loss]` |
| *Baseline equivalent* | `weighting = none` + `gamma = 0` → plain `CategoricalCrossentropy(label_smoothing=0.01)` | rainfall 5-class | — |
| **Swin fine-tune** (`[finetune]`, `--stage finetune`/`both`) ||||
| Optimizer | `AdamW`, `weight_decay = 0.01` (falls back to `Adam` where unavailable) | finetune | `[finetune]` |
| Initial / min LR, warmup | `3e-4` / `1e-6`, `3` epochs | finetune | `[finetune]` |
| Epochs | `20` | finetune | `[finetune]` |
| Swin blocks | `2` (block 0 = W-MSA, block 1 = SW-MSA) | finetune | `[finetune]` |
| Window size / heads | `8` / `4` | finetune | `[finetune]` |
| Head width / dropout | `c_shared = 64` / `0.1` | finetune | `[finetune]` |
| Backbone | frozen — only the head trains | finetune | — |
| **Knowledge distillation** (`train_lightning_kd.py`) ||||
| Loss | `α·L_soft·T² + (1−α)·WeightedFocalLoss` | KD student | — |
| Alpha / temperature | `0.7` / `4.0` (Hinton canonical) | KD student | `--kd_alpha` / `--kd_temperature` |
| Optimizer / LR | `Adam` / `1e-4` | KD student | `--learning_rate` |
| Epochs / batch / patience | `50` / `8` / `10` | KD student | CLI |
| **SepConv baseline** (`sepconv_ensemble_training.py`) ||||
| Architecture | 3 independent `SeparableConv2D` models, one per lead; kernel `5×5` | baseline | — |
| Optimizer | `Adam(lr=0.001, amsgrad=True)`, halve on plateau | baseline | — |
| Loss | weighted MSE — imported from `train_models` so it cannot drift | baseline | `RAINFALL_MSE_WEIGHTS` |
| Epochs / batch | `50` / `8` | baseline | CLI |
| Output | sigmoid `[0,1]` continuous; binned to 5 classes at evaluation | baseline | — |

---

## Outputs reference

### `predict_full_domain.py` → `inference/predict_<run_tag>[_finetuned|_kd]/`

**Lightning modes** — two PNGs per reference:
- `predict_<date>_<HHMM>.png` — 2 × 3. Row 1 GT occurrence; Row 2 hit / miss / false-alarm overlap of the Hann-blended + hysteresis prediction (orange = hit, blue = miss, red = false alarm).
- `predict_<date>_<HHMM>_hits.png` — 1 × 3, correctly-detected pixels only, subtitled `hits = N (X.X% of GT-active)`.

**Rainfall modes** — two PNGs per reference when OPERA GT is on disk (falls back to a pred-only 1 × 3 heatmap otherwise):
- `predict_<date>_<HHMM>.png` — 3 × 3. Row 1 GT class canvas (viridis-5); Row 2 zone overlap of the raw argmax; Row 3 same against the hysteresis-cleaned prediction.
- `predict_<date>_<HHMM>_perclass_hits.png` — 2 × 3, per-class hits raw vs hysteresis, subtitled with the per-class hit-rate breakdown.

**Optional:** `--save-npy` dumps raw `(3, 768, 1536)` canvases (plus `<stem>_hyst.npy` int32 class canvases for rainfall); `--no-plot` skips PNGs; `--patches "5,6,11,12"` restricts to a subset of the 18-patch grid (rainfall only — the lightning path always covers the full canvas via Hann overlap).

### `visualize_gt_vs_pred.py` → `full_domain_plots/full_domain_<run_tag>[…]/`

- **3-row full-domain figure** — Row 1 GT, Row 2 raw prediction, Row 3 post-processing overlap. Green/red patch numbering shows which patches DBSCAN selected vs padded.
- **3-row zoom figure** — same layout cropped to the qualifying patch with the most GT activity; Row 3 stats recomputed for that window only.
- **Rainfall-only aggregate graphs** (skip with `--no_aggregate_graphs`): `aggregate_pc_hist_{raw,hyst}.png` per-class p(c) histograms, `aggregate_class_count_distribution.png` per-class box plots, and `aggregate_pc_hist_3d/` — 10 interactive plotly HTMLs (5 classes × raw/hyst) where X = softmax bin, Y = patch #, Z = % of that patch's class-c pixels.

### `validate_predictions.py` → `validation/`

Two coverage metrics per (sample, lead): **`iou_mask`** (IoU of the binary ≥10 mm/h masks — structure only) and **`class_wt`** (per-class weighted overlap macro-averaged across the 5 classes — semantic-aware). Aggregate FAR / POD / CSI are computed per lead on the binary ≥10 mm/h event.

| File | Contents |
|---|---|
| `<track>_<year>_<month>_samples.csv` | One row per sample; both metrics × each lead time. |
| `<track>_<year>_<month>_summary.json` | Totals, counts above `--high_coverage_pct` per lead × metric, FAR/POD/CSI per lead, initial selection list, tuned `post_processing` thresholds. |
| `<track>_<year>_<month>_metrics.png` | Left: grouped FAR/POD/CSI bars per lead. Right: per-sample coverage scatter (IoU vs class-weighted, marker per lead). |
| `<track>_<year>_<month>_<date>_<HHMM>_<lead>.png` | Visualisation mode. Left: structure overlay (red = GT class == Pred class and both ≥10 mm/h). Right: 256 × 256 zoom into the most GT-active patch — red matched, blue misses, orange false alarms. |

Visualisation title colour: **green** if the date cleared the coverage threshold for that lead/metric, **orange** if selected but below it. A date absent from the initial selection raises `SystemExit`.

### `feature_importance_analysis.py` → `--output` (default `results/feature_importance/`)

| File | Content |
|---|---|
| `prediction_diagnostics.png` | 4-panel MAE / RMSE / value comparison / heatmap |
| `xi_matrix.csv`, `xi_heatmap.html`, `xi_bar_chart.html`, `xi_boxplots.html` | Xi correlations (inputs × timesteps) and its interactive views |
| `gradcam_rank*_*.png` | 5-panel Grad-CAM comparison plots |
| `shap_spatial_maps.png`, `shap_bar_chart.html`, `shap_importance.csv` | SHAP spatial + global importance |
| `method_comparison.{csv,html}`, `method_correlations.csv` | Grad-CAM + Xi vs SHAP, with Spearman + Pearson agreement |
| `xi_matrix_ablated.csv`, `ablation_impact.html` | Only with `--model-ablated` / `--data-ablated` |

---

## Knowledge distillation (lightning track)

Hinton-style teacher–student distillation producing a student that gives the same lightning prognosis **without LINET at inference** — useful whenever the LINET feed is late, missing, or itself being validated.

| | Teacher | Student |
|---|---|---|
| Mode | `mtg_lightning_opera_occurrence` | `mtg_opera_occurrence` |
| HR inputs | LINET (density + current + occurrence) + MTG `vis_06` | MTG `vis_06` only |
| MR inputs | OPERA reflectivity + rainfall_rate + MTG IR/WV | *(same)* |
| Label | Binary lightning occurrence at t+15/+30/+45 | *(same)* |
| Weights | `coalition_mtg_lightning_opera_occurrence_dbscan[_finetuned].keras` | `coalition_mtg_opera_occurrence_dbscan_kd.keras` |

**KD loss** (adapted for binary sigmoid outputs):

```
logit(p) = log(p / (1 - p))                        # inverse sigmoid
soft_x   = sigmoid(logit(p_x) / T)                 # temperature-softened
L_soft   = BCE(soft_teacher, soft_student) * T**2  # gradient-scale fix
L_hard   = WeightedFocalLoss(y_gt, student)        # supervised anchor
L_total  = alpha * L_soft + (1 - alpha) * L_hard
```

The student reuses the **teacher's dataset** directly, so both see identically-shuffled batches; LINET channels are sliced away on the fly (`teacher_hr[..., -1:]` = the trailing `vis_06` channel). No separate `mtg_opera_occurrence` dataset needs to exist.

---

## Automated report generation

Turns `validate_predictions.py` outputs into a standalone PDF for meteorologists who have never seen the model. Runs a local Ollama LLM for data-first English commentary, then (unless `--language en`) translates each paragraph to Romanian with a meteorological glossary in the system prompt. All prompts are **text-only** — figures are rendered for the human reader but never sent to the model, which reads pre-computed metadata instead. Output is deterministic given the same validation outputs, model tag, temperature, and seed.

**Sample-selection parity:** both tracks use the same OPERA-driven criterion (≥10 mm/h anywhere on the canvas at the reference timestep), so `initial_selection` holds the same references in each track's summary.

### Prerequisites

```powershell
& "C:\path\to\tfenv\python.exe" -m pip install fpdf2 ollama Pillow tzdata
ollama pull gemma3:27b-it-q4_K_M      # server on http://localhost:11434
```

`tzdata` is optional — it renders the cover timestamp in Europe/Bucharest (EET/EEST); without it a hand-rolled EU DST fallback applies.

### Contents

One PDF per (year, month): **cover** → **clickable table of contents** → **executive summary** → **per-lead metrics sections** (one per track × lead, with `metrics.png` embedded) → **per-reference event sections** (coupling-mask figure: blue = rainfall ≥10 mm/h only, orange = lightning only, red = coupled cells; caption built from per-cell bounding box, 8-way cardinal centroid, peak mm/h, and lightning-active count — cells under `MIN_CELL_SIZE_PIXELS` are dropped as noise) → **data appendix** with min/mean/max IoU per lead per track.

Every numeric claim comes from a pre-computed facts block; the LLM paraphrases, never invents.

---

## Thresholds reference

Tune these to change behaviour without touching the architecture.

| Constant | Default | CLI override | Purpose | Effect of changing |
|---|---|---|---|---|
| `DBSCAN_THRESHOLD` | `10` mm/h | `--threshold` | Rain-rate cut for training-patch selection. | Lower → weaker events enter training. Higher → smaller, more selective training set. |
| `DBSCAN_EPS` | `5` px | — | DBSCAN neighbourhood radius. | Larger → clusters merge. Smaller → cells fragment. |
| `DBSCAN_MIN_SAMPLES` | `20` px | — | Minimum cluster size; smaller regions become noise. | Lower → tiny cells qualify. Higher → only substantial storms. |
| `RAINFALL_THRESHOLD_MMH` | `10.0` mm/h | `--rainfall_threshold_mmh` | Validation **sample-selection** scope. The binary event for FAR/POD/CSI/IoU stays anchored to class ≥ 1 — that is the model's trained decision boundary. | Lower → more samples selected. Higher → only intense convection. |
| `HIGH_COVERAGE_PCT` | `90.0` % | `--high_coverage_pct` | Coverage above which a sample counts as "high coverage" per lead; drives the green/orange visualisation title. | Lower → more samples graded good. Higher → stricter grading. |
| `DEFAULT_STRIDE` | `128` px | `--stride` | Hann-blended inference stride — 50 % overlap → 55 patches per reference, which removes the 256-px tiling seams. | `64` → 75 % overlap, smoother but ~4× cost. `256` → no overlap, seams return. |
| `LIGHTNING_LOW_THRESHOLD` | `0.90` | `--lightning_low_threshold` | Hysteresis LOW: a pixel is positive iff `p ≥ low` **and** its 8-connected component contains a `p ≥ high` seed. | Lower → wider cells. Higher → tighter cells, fewer marginal detections. |
| `LIGHTNING_HIGH_GRID` | `0.91 … 0.99` step `0.01` | — | Sweep grid; the per-lead HIGH is tuned by maximising aggregate CSI and persisted in `summary.json → post_processing`. | Narrower → faster tuning, fewer operating points. |
| `DEFAULT_HIGH_THRESHOLD` | `0.95` | `--lightning_high_threshold` | Fallback HIGH when no `--validation_summary` is given. | Only affects inference without a tuned summary. |
| `DEFAULT_RAIN_LOW` / `DEFAULT_RAIN_HIGH` | `0.35` / `0.55` | `--rainfall_low_threshold` / `--rainfall_high_threshold` | Hysteresis on `p(argmax)` when the argmax is a rainy class. Lower than the lightning pair because probability is split across 5 classes. Rejected pixels drop to class 0. | Lower → more marginal rainy pixels survive. Higher → tighter blobs. |
| `RAINFALL_CLASS_EDGES` | `10, 20, 30, 40` mm/h | — | The 5-class boundaries, shared by the `_rainfall` and `_continuous` heads. | Changing requires retraining the head. |
| `RAINFALL_MAX_MMH` | `70` mm/h | — | Normalisation scale for the continuous head. | Changing requires retraining. |
| `DEFAULT_KD_ALPHA` | `0.7` | `--kd_alpha` | Weight on the soft-teacher loss. | Higher → student mimics teacher more, weaker GT anchoring. |
| `DEFAULT_KD_TEMPERATURE` | `4.0` | `--kd_temperature` | Softening temperature for both models' sigmoid outputs. | Higher → softer targets. `T=1` disables softening. |
| `STUDENT_HR_CHANNELS` | `1` | — | Trailing HR channels the student keeps (= `vis_06`). | Changing requires a matching HR layout change and retraining. |
| `MIN_CELL_SIZE_PIXELS` | `10` px | — | Minimum connected component reported as a coupled cell in the report. | Lower → more cells, longer captions. Higher → only large systems. |
| `DEFAULT_OLLAMA_SEED` / `_TEMPERATURE` / `_MAX_TOKENS` | `42` / `0.1` / `2000` | `--seed` / `--temperature` / `--max_tokens` | Report determinism and per-call length cap. | `temperature = 0.0` triggers empty-completion collapse in Gemma — keep it at `0.1`. |

---

## Data products

| Source | Products | Native cadence | Native resolution | Role | Entry point |
|---|---|---|---|---|---|
| **MTG FCI L1C** | `vis_06`, `ir_38`, `ir_105`, `wv_63`, `wv_73` | 10 min | 1 km (`vis_06`) / 2 km (IR/WV) | Satellite features (HR + MR) | `our_data/satellite_data/pipeline_msg_mtg.py` |
| **OPERA composite** | `reflectivity` (dBZ), `rainfall_rate` (mm/h) | 15 min | 2 km | Precipitation target + features | `our_data/opera_data/pipeline_opera.py` |
| **LINET** | `density`, `current`, `occurrence` | 10 min, filter-aligned | KML → 1 km grid | Lightning target + features | `our_data/lightning_data/linet_export.py`, `read_kml_version2.py` |

`opera_rainfall_rate_hr` is an HR-extracted alias of the same reprojected file, used as the 256 × 256 label so the output head matches the HR decoder.

**Satellite channel selection** — 5 channels chosen by physical property:

| Physical property | MTG FCI |
|---|---|
| Cloud optical thickness (VIS) | `vis_06` (0.6 µm) |
| Cloud phase discrimination (SWIR) | `ir_38` (3.8 µm) |
| Cloud top temperature (TIR) | `ir_105` (10.5 µm) |
| Upper-tropospheric moisture (WV) | `wv_63` (6.3 µm) |
| Mid-tropospheric moisture (WV) | `wv_73` (7.3 µm) |

---

## Directory structure

```
coalition4-rcnn/
├── c4dl/                     # shared library: projection + model layers
├── our_data/
│   ├── satellite_data/       # MTG acquisition
│   ├── opera_data/           # OPERA acquisition
│   ├── lightning_data/       # LINET export + rasterisation
│   ├── reprojected_data/     # Romania-grid canvases + grid coord sidecars
│   ├── patches/              # {var}_{HHMM}_{HR|LR}.npy per date
│   ├── datasets/             # <mode>_dbscan/{train,validation,test} TFRecords
│   ├── patch_index/          # patch_index.csv
│   ├── timestep_config.json  # master cadence + per-product minute filters
│   ├── sequence_meta_dbscan.json
│   └── normalization_stats_dbscan.json
├── models/                   # coalition_<run_tag>[_finetuned|_kd].keras + history
│   └── checkpoints/          # resumable per-epoch state
├── evaluation/               # eval_<run_tag>/evaluation_results.json + plots
├── inference/                # predict_full_domain output
├── validation/               # validate_predictions output + PDF reports
├── full_domain_plots/        # visualize_gt_vs_pred output
├── training.config           # all training hyperparameters
├── product_cadences.config   # per-product native cadences
├── run_opera.config          # end-to-end runbook (comments only)
└── pipeline_config.py        # SOURCE constant
```

---

## Key differences from MeteoSwiss COALITION-4

- **Domain:** Romania (EPSG:31700, 1536 × 768 @ 1 km) instead of Switzerland.
- **Radar:** OPERA European composite instead of MeteoSwiss RZC/CZC/LZC. NMA (National Meteorological Administration) radar was evaluated and dropped.
- **Satellite:** MTG FCI instead of MSG SEVIRI; the MSG branch and its 3 km LR tier were removed.
- **NWCSAF** cloud products were evaluated and dropped, which retired the LR tier.
- **Sample selection:** DBSCAN over OPERA `rainfall_rate` on an 18-patch grid, with a Czibula block-wise 80/10/10 split.
- **Additions:** optional Swin-transformer domain-adaptation head, knowledge distillation for a LINET-free lightning student, a SepConv regression baseline, and an automated LLM report pipeline.

---

## References

- Leinonen, J., Hamann, U., Germann, U. — *Seamless lightning nowcasting with recurrent-convolutional deep learning* (COALITION-4).
- Czibula, G. et al. — block-wise temporal splitting for meteorological nowcasting datasets.
- Hinton, G., Vinyals, O., Dean, J. — *Distilling the Knowledge in a Neural Network*.

## License

See [LICENSE](LICENSE).
