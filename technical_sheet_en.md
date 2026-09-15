# COALITION-4 RECONVECT — Technical Sheet

Convective nowcasting over Romania. Multi-resolution encoder-forecaster trained on
MTG FCI, OPERA radar and LINET lightning, predicting 5-class rainfall and binary
lightning occurrence on a shared 1 km grid.

```
repository   coalition4-rcnn   branch wp
grid         1536 x 768  EPSG:31700  (1 km)
cadence      15 min master step
archive      2025-01-01 .. 2026-08-13
```

> Artefacts marked **CRITICAL** are consumed downstream. Maintained alongside
> `README.md` and `technical_sheet_ro.md`; all three are updated as features land.

**Contents:** [1 Glance](#1-system-at-a-glance) · [2 Tiers](#2-resolution-tiers) ·
[3 Modes](#3-modes-inputs-and-labels) · [4 Data products](#4-data-products) ·
[5 Architecture](#5-architecture-and-training-defaults) · [6 Thresholds](#6-key-thresholds) ·
[7 Common steps](#7-running-plan--common-steps) · [8 Branch A](#8-branch-a-full-reconvect-training) ·
[9 Branch B](#9-branch-b-ablation-study-and-sepconv-ens-baseline) ·
[10 Branch C](#10-branch-c-seasonal-ensemble) · [11 Branch D](#11-branch-d-knowledge-distillation) ·
[12 Criticality](#12-artefact-criticality)

---

## 1. System at a glance

Three instruments are reprojected onto one 1 km canvas, sliced into 256 × 256 patches,
and fed to an encoder-forecaster whose architecture is **read from the dataset** rather
than hard-coded. Past and future step counts, input tiers and channel counts all come
from `metadata.json`, so changing the sequence window is a data decision, not a code change.

```
acquire -> reproject -> common coverage -> select patches -> build sequences -> slice patches
        -> statistics -> TFRecord datasets -> train -> infer -> validate -> report
```

| Task | Target |
|---|---|
| Rainfall, 5-class | OPERA `rainfall_rate` binned at 10/20/30/40 mm/h |
| Rainfall, `log_zscore` (SepConv baseline only) | OPERA `rainfall_rate`, denormalised and binned at the same edges for scoring |
| Lightning occurrence | LINET binary occurrence |

**Class imbalance.** The overwhelming majority of pixels carry no significant rainfall,
and the intense classes are rarer still by orders of magnitude. Every loss in the system is
weighted because of it: an unweighted objective on this distribution is minimised by
predicting the quiet value everywhere, which scores well and forecasts nothing. Patch
selection is convective — DBSCAN over the rain field, at a threshold that is configurable
(`--threshold`) — but *within* a selected patch every pixel contributes, low-intensity
ones included. There is no per-pixel cut-off.

| Storage item | Note |
|---|---|
| MTG `.npy` per repeat cycle | 5 channels |
| MTG raw `.nc` per chunk | 2 chunks per cycle |
| 7-Zip archive of a dataset | `-mx=5` reaches 4.8 % of source, `-mx=1` 11.5 % |
| zstd of the `.npy` stores | level 10, in place, ~8.5x measured across the archive |

---

## 2. Resolution tiers

Always read the physical resolution, not the tier name.

| Tier | Native | Patch | Pooling | Channels |
|---|---|---|---|---|
| `past_hr` (HR) | 1 km | 256 × 256 | none | MTG `vis_06`; LINET `density`, `current`, `occurrence` |
| `past_mr` (MR) | 2 km | 128 × 128 | 2 × 2 avg | OPERA `reflectivity`, `rainfall_rate`; MTG `ir_38`, `ir_105`, `wv_63`, `wv_73` |

**HR is the higher-resolution tier and MR the minimum-resolution one.** There are only
these two; extracted patches are named `{variable}_{HHMM}_{HR|MR}.npy` accordingly, and the
input tensors are `past_hr` and `past_mr`. One vocabulary throughout — code, filenames and
documentation.

**How the tiers are reconciled.** (1) Reproject everything onto the same 1 km grid first,
so a patch number maps to the same geographic tile across all products and cross-scale
alignment is paid once per cadence step. (2) Pool *down*, never up — upsampling would
fabricate pixels and the model would overfit interpolation artefacts.

---

## 3. Modes: inputs and labels

The mode name states its track: `_rainfall` = OPERA 5-class, `_logz` = OPERA rainfall in
`log_zscore` space (the SepConv baseline's own target), `_occurrence` = lightning binary.
The label is always HR (256 px) regardless of input tiers.

| Mode | HR inputs | MR inputs | Label |
|---|---|---|---|
| `mtg_opera_radar_only_rainfall` | `vis_06` | `opera_reflectivity` `opera_rainfall_rate` | rainfall 5-class |
| `mtg_opera_mtgmr_rainfall` | `vis_06` | + `ir_38` `ir_105` `wv_63` `wv_73` | rainfall 5-class |
| `mtg_lightning_opera_rainfall` | `density` `current` `occurrence` `vis_06` | + MTG IR/WV | rainfall 5-class |
| `mtg_lightning_opera_occurrence` | `density` `current` `occurrence` `vis_06` | + MTG IR/WV | lightning binary |
| `mtg_opera_occurrence` † | `vis_06` | + MTG IR/WV | lightning binary |
| `opera_radar_only_rainfall` ‡ | `opera_rainfall_rate_hr` | — | rainfall 5-class |
| `opera_sepconv_logz` ‡ | `opera_rainfall_rate_hr` | — | rainfall `log_zscore` |

† KD student only — not buildable by `create_datasets.py`; trains on the teacher's dataset.
‡ Baseline comparison pair, radar-only by design. Both carry the field in HR at 256 px so the input tensors are identical; the model's output resolution is its finest input, and this is the only mode with no other HR channel to hold it at 256.

Channel counts follow directly: an HR tensor is `(T, 256, 256, n_hr)`, MR is
`(T, 128, 128, n_mr)`. A mode with **no HR inputs has no `past_hr` tensor at all** — the
model is built from whatever groups the dataset provides.

`opera_rainfall_rate` and `opera_rainfall_rate_hr` are the same field at two tiers: the
2 km input pooled to 128 px, and the same reprojected field kept at 1 km / 256 px because
it is the label. `opera_sepconv_logz` is the one mode using the HR form as both.

**Rainfall classes (mm/h):** `0: R<10 · 1: 10–20 · 2: 20–30 · 3: 30–40 · 4: R≥40`

---

## 4. Data products

| Product | Channels | Cadence | Resolution | Source script |
|---|---|---|---|---|
| MTG FCI L1C | `vis_06` `ir_38` `ir_105` `wv_63` `wv_73` | 10 min | 1 km / 2 km | `pipeline_msg_mtg.py` |
| OPERA | `reflectivity` `rainfall_rate` | 15 min | 2 km | `pipeline_opera.py` |
| LINET | `density` `current` `occurrence` | 15 min | 1 km | `linet_export.py` |

MTG is delivered as 40 body chunks per repeat cycle; Romania falls in chunks 35 and 36
(`ROMANIA_CHUNKS`), applied identically by the NMA and Data Store paths. Lightning is
binned straight onto the Romania grid, so it needs no reprojection step.

| `--source` | Behaviour |
|---|---|
| `nma` | SFTP from the internal server over the requested window |
| `datastore` | Fetches only cycles listed in `mtg_missing_timesteps.json` |
| `both` | NMA first, then the Data Store for the remainder |
| `local` | No download — extracts raw already in `_raw_chunks/` |

Origin is recorded per cycle in `provenance.json` because the two sources deliver
identical native filenames; it is not recoverable from the data afterwards.

**Multi-disk storage.** The MTG store outgrows a single disk at ~47 MB per cycle.
`--spill_dir` rotates between stores whenever the active one drops below `--min_free_gb`,
re-evaluated before every window **and in both directions**, since `--delete_raw` returns
space to the disk it is reading from. `store_registry.py` records which date landed where;
`reproject.py` and `summarize_mtg.py` read every registered root.

---

## 5. Architecture and training defaults

| Parameter | Value | Scope |
|---|---|---|
| Encoder | ResBlock + ConvGRU (`ResGRU`), channels `[32, 64, 128]` | all |
| Decoder | reversed `[128, 64, 32]`, bilinear up + skips | all |
| Input branches | one per tier, merged at matching scales | all |
| Past / future steps | read from `sequence_meta_<source>[_<period>].json` — `w34`/`f34`: 3 / 4, `w44`: 4 / 4 | all |
| Optimizer | `Adam(lr=1e-3)` | base stage |
| Loss (lightning) | `WeightedFocalLoss(gamma=2.0)` | prior from `lightning_fraction` |
| Loss (rain classification) | `WeightedFocalCategoricalCrossentropy` | prior from `opera_rainfall_fraction` |
| Epochs / batch | `20` / `32` | all |
| Dropout / norm | `0.1` / `layer` | all |
| Mixed precision | `true` (fp16) | all |
| LR schedule | `cosine_warmup`, 1e-3 → 1e-6, 3 warmup epochs | base stage |
| Early stopping | `val_loss`, patience 6, restore best | all |
| Finetune head | Swin transformer | finetune stage |

The forecaster is **single-pass**: it starts from zeros repeated `future_timesteps` times
and runs the ConvGRU stack for the full horizon, emitting every frame at once. Nothing in
RECONVECT is autoregressive.

---

## 6. Key thresholds

The full table is in `README.md`; these are the ones that change results rather than presentation.

| Constant | Default | CLI | Purpose |
|---|---|---|---|
| `DBSCAN_THRESHOLD` | 10 mm/h module default; the master index records its own (8 mm/h today) and that is the default once it exists | `--threshold` | Rain-rate cut for training-patch selection |
| `selection_rule` | `pixel` | `--rule pixel\|box` | `pixel`: a tile holding ≥ 1 cluster pixel; `box`: a tile meeting the union of 256 × 256 boxes centred on the cluster centroids. Recorded in the index; the recorded rule is the default |
| `DBSCAN_EPS` / `MIN_SAMPLES` | 5 px / 20 px | — | Cluster radius and minimum size |
| `RAINFALL_CLASS_EDGES` | 10/20/30/40 | — | 5-class boundaries; changing requires retraining |
| `DEFAULT_RAIN_LOW` / `HIGH` | 0.35 / 0.55 | `--rainfall_*` | Hysteresis on `p(argmax)` for rainy classes |
| `LIGHTNING_LOW_THRESHOLD` | 0.90 | `--lightning_low_threshold` | Hysteresis LOW on the probability canvas |
| `DEFAULT_STRIDE` | 128 px | `--stride` | Hann-blended inference stride; 50 % overlap |
| `DEFAULT_KD_ALPHA` / `TEMPERATURE` | 0.7 / 4.0 | `--kd_alpha` | Soft-teacher weight and softening temperature |
| `alpha_max` | 100.0 | `[radar_loss]` | Clips per-class weights; unclipped destabilises fp16 |
| `label_smoothing` | 0.01 | `[radar_loss]` | Prevents `log(0)` when the Swin softmax saturates |

---

## 7. Running Plan — common steps

Every branch shares these. Each entry says what the script does **when run on its own**,
what it leaves behind, and which script picks that up. An output marked **CRITICAL** is
consumed downstream: **the pipeline cannot proceed without it.** A later stage will stop
outright when the file is absent, and — the case that costs more — will run to completion
on a file that no longer matches the data on disk, reporting results for an archive that
is not the one being used.

### 0. Master cadence
```bash
python validate_timestep.py --step_minutes 15
```
- **Does** — picks the master step and derives each product's minute filter from `product_cadences.config`.
- **Writes** — `our_data/timestep_config.json` **CRITICAL**
- **Read by** — every acquisition script, `reproject`, `identify_patches`, `extract_patch_seq`, `extract_patches`, the summarisers, the intersection. Changing the step here changes every stage below.

### 1a. MTG acquisition
```bash
python our_data/satellite_data/pipeline_msg_mtg.py --start ... --end ... --source nma|datastore|both|local
```
- **Does** — downloads FCI chunks 35/36 and extracts one `.npy` per channel per repeat cycle. Waves of `--workers` cycles; with `--delete_raw` the raw is reclaimed wave by wave, so peak raw on disk is one wave rather than the range.
- **Writes** — `MTG/<channel>/….npy`, `mtg_constants.json` **CRITICAL**
- **Writes** — `MTG/provenance.json` **CRITICAL** — which source each cycle came from. The two sources share native filenames, so origin is unrecoverable afterwards.
- **Alone** — `--source local` extracts raw already on disk with no download. `--delete_only` reclaims raw without re-extracting. `--record_existing` stamps provenance on data predating the ledger.
- **Read by** — `reproject.py`, `summarize_mtg.py`

### 1b. OPERA acquisition
```bash
python our_data/opera_data/pipeline_opera.py --start ... --end ... --ssh_key ...
```
- **Does** — fetches OPERA composite HDF5, mirroring the remote date hierarchy. Kept in native format so a projection fix never costs a re-download.
- **Writes** — `our_data/opera_data/{reflectivity,rainfall_rate}/YYYY/MM/DD/*.h5` **CRITICAL**
- **Read by** — `reproject.py`, `summarize_opera_data.py`

### 1c. LINET export
```bash
python our_data/lightning_data/linet_export.py --start ... --end ... --format kml
```
- **Does** — downloads raw stroke exports, one KML per day.
- **Note** — both bounds are inclusive, as everywhere else; `--start D --end D` exports one day.
- **Writes** — `<out>/kml_data/<date>/<date>.kml` **CRITICAL**
- **Read by** — `read_kml_version2.py`

### 2a. Rasterise lightning
```bash
python our_data/lightning_data/read_kml_version2.py
```
- **Does** — bins strokes straight onto the Romania grid at the label cadence; binning places them there, so no reprojection is needed.
- **Writes** — `lightning_data/{density,current,occurrence}/….npy` **CRITICAL**
- **Writes** — `filtered_out_reports/lightning_filtered_out_<date>.json` — strokes dropped for falling outside the grid. Audit only; nothing reads it.
- **Read by** — `extract_patches`, `compute_normalization_stats`, `summarize_lightning_data`

### 2b. Reproject onto the Romania grid
```bash
python reproject.py --all --workers 6
```
- **Does** — KD-tree resampling of MTG and OPERA onto the shared 1536 × 768 canvas. Afterwards a patch number maps to the same tile across all products.
- **Writes** — `our_data/reprojected_data/….npy` **CRITICAL**
- **Writes** — `romania_grid_lats.npy` / `_lons.npy` — the grid definition reused by plotting and NetCDF export.
- **Writes** — `reproject_<category>.log` — failures, subtracted from the manifest via `--errors_log` so a failed reprojection is not counted as present.
- **Alone** — `--mtg_dir` reads a store held on another drive; without it every root in the store index is walked in turn.
- **Read by** — `identify_patches`, `extract_patches`, `compute_normalization_stats`

### 3. Coverage summaries (one per product)
```bash
python our_data/<product>/summarize_<product>.py --start 2025-01-01 --end 2026-08-13 --chart
```
- **Does** — measures per-date coverage from the **extracted** output, not the raw downloads: a reprojection that failed after a successful download is invisible to a raw scan.
- **Writes** — `our_data/<product>_data/<product>_summary.csv` and `<product>_missing_timesteps.json` **CRITICAL** — beside the product they describe, not at the repository root, and anchored to the script rather than the working directory.
- **Graph** — `<product>_coverage.png` with `--chart`, in the same folder. Monthly bars with a line through the tops. Presentational; nothing reads it.
- **Note** — the MTG missing-timestep JSON is what the Data Store backfill requests: it fetches exactly the cycles named there. Pass `--start`/`--end`, or a date holding no files at all is never reported as missing and can never be requested.
- **Alone** — `summarize_mtg --npy_dir` accepts several roots and scans them as one archive, for a store split across drives. `--scan {npy,raw,reprojected}` picks which question is answered: `npy` (default) is the extracted store and drives the Data Store backfill, `raw` is what arrived before extraction, `reprojected` is what `extract_patches` will actually read. A `reprojected` scan's missing-list describes reprojection gaps and must **not** be fed to `--source datastore`.
- **Read by** — `intersect_product_coverage`; `pipeline_msg_mtg --source datastore`

### 4. Determining the common coverage period
```bash
python intersect_product_coverage.py --summary opera_rainfall_rate=our_data/opera_data/opera_summary.csv \
    [--summary opera_reflectivity=our_data/opera_data/opera_summary.csv] \
    [--summary mtg=our_data/satellite_data/mtg_summary.csv --summary lightning=our_data/lightning_data/lightning_summary.csv]
```
- **Does** — intersects the requested products into the timesteps where *all* of them are available. Which products are required is chosen per run, so a radar-only model is not held back by gaps in another instrument. Gate on every field the modes will read: `opera` alone is an alias for `opera_rainfall_rate`, and the full modes also read reflectivity.
- **Writes** — `our_data/timestep_manifest.csv` **CRITICAL** — the timesteps every later stage is allowed to draw on.
- **Graph** — `our_data/intersect_summary.png` — monthly lines, one per category, kept against each omission reason.
- **Note** — OPERA's two fields are separate keys, so a rainfall-only model keeps samples reflectivity happens to lack, and a model reading reflectivity is never handed a timestep without it.
- **Read by** — `extract_patch_seq_for_datasets`

### 5. Convective patch selection
```bash
python identify_patches.py --start 2025-01-01 --end 2026-08-13
```
- **Does** — DBSCAN over OPERA `rainfall_rate` (threshold, eps and min_samples default to the values recorded in the master index — 8 mm/h, 5, 20 today — else 10 / 5 / 20; `--threshold` overrides, with a warning that a rebuilt index invalidates the patch pool) marking which of the 18 patches are active per timestep. Selects **patches, not pixels**.
- **Rule** — `--rule pixel|box`, recorded in the index as `selection_rule` and the default once an index exists. `pixel` selects a tile holding ≥ 1 cluster pixel — every index so far. `box` centres a 256 × 256 box on each cluster centroid and selects a tile meeting the union of those boxes: every cell keeps 128 px of context on each side, and an edge cell brings its dry neighbour in with it. Changing the rule rebuilds a different index and invalidates the patch pool.
- **Writes** — `our_data/patch_index/patch_index.csv` and `.json` **CRITICAL**
- **Graph** — with `--date --plot`, one GIF per diagnostic per day, one frame per timestep: `plots/dbscan_<rule>_patch_selection/<date>.gif` (left: the field above the threshold, each cluster's centroid and a 256 × 256 box around it; right: the mask the rule in force tests — the cluster pixels under `pixel`, the union of centroid boxes with the cluster pixels drawn over it under `box` — the 6 × 3 grid dotted in red, tiles holding ≥ 1 mask pixel outlined in green) and `plots/patch_highlight/<date>.gif` (the field and the selection with borders). A NetCDF twin per active timestep, ~28 MB each, lands in `plots/nc/`. Diagnostics only; `--purge_plots` clears them.
- **Note** — one index serves every period, and its row order defines the patch axis of the saved arrays. A `--date` run never overwrites the master index.
- **Read by** — `extract_patch_seq_for_datasets`, `extract_patches`, `data_statistics`

### 6. Sequence windows and splits
```bash
python extract_patch_seq_for_datasets.py [--period LABEL --past N --future M --start ... --end ...]
```
- **Does** — builds temporally continuous sequences and the Czibula block split (6-hour blocks, 80/10/10).
- **Writes** — `{train,validation,test}_data_<source>[_<period>].csv` **CRITICAL**
- **Writes** — `sequence_meta_<source>[_<period>].json` **CRITICAL** — `past_steps`, `future_steps`, `step_minutes`. This is what makes the model horizon a property of the data.
- **Writes** — `extract_patch_seq_drops_….csv` — every candidate that did not survive, and why.
- **Read by** — `extract_patches`, `create_datasets`, `compute_normalization_stats`, the class priors, `verification_keys`, `data_statistics`

### 7. Slice patches
```bash
python extract_patches.py [--period LABEL] [--products opera ...]
```
- **Does** — slices 256 × 256 tiles from the full canvases; MR products are average-pooled to 128 px. Always down, never up.
- **Writes** — `our_data/patches/<date>/<var>_<HHMM>_{HR|MR}.npy` **CRITICAL**
- **Note** — output is **not** period-suffixed: every period writes into one shared tree, so a second period is largely a no-op over the overlap. The pool is invalidated by a rebuilt `patch_index.csv`, never by a new period.
- **Also writes** — `our_data/patches/<date>/_patch_index.json`, the active-patch lists each date was built from. Any timestep whose list has since moved is re-extracted instead of skipped, because a patch that becomes active inserts mid-list and shifts every later slot. **CRITICAL**
- **Alone** — `--audit_pool` reports which dates drifted and extracts nothing.
- **Read by** — `create_datasets`

### 8. Normalization statistics
```bash
python compute_normalization_stats.py [--period LABEL] [--variables ...]
```
- **Does** — per-variable mean/std over the **training keys only**.
- **Writes** — `normalization_stats_<source>[_<period>].json` **CRITICAL**
- **Note** — the invariant is that training and inversion use the same constants. Train under one set and invert with another and the recovered mm/h is wrong, biased with intensity, and nothing raises.
- **Read by** — `create_datasets`, `train_models`, `predict_full_domain`, `validate_predictions`, `sepconv_predict`, `evaluate_coalition`, `generate_report`

### 9. Class priors
```bash
python opera_rainfall_fraction.py [--period LABEL]
python lightning_fraction.py [--period LABEL]
```
- **Does** — measures the class balance of the model's **own** training split.
- **Writes** — `opera_rainfall_fraction_….json` / `lightning_fraction_….json` **CRITICAL**
- **Note** — scope and filename come from one tag. A prior computed over a different window describes a balance the model never sees, and the weighted loss then corrects for an imbalance that is not the one present.
- **Read by** — `train_models`, `sepconv_ensemble_training`

---

## 8. Branch A: full RECONVECT training

The headline model: all three modalities, 5-class rainfall. Runs straight on from step 9.

### A1. Build the dataset
```bash
python create_datasets.py --mode mtg_lightning_opera_rainfall [mtg_lightning_opera_occurrence ...] [--period LABEL] [--no-archive]
```
- **Writes** — `our_data/datasets/<run_tag>/{train,validation,test}/*.tfrecord` **CRITICAL**
- **Writes** — `metadata.json` per split **CRITICAL** — `input_shapes` and `label_shape`. Training reads its architecture from these, so a different window needs no code change.
- **Note** — without `--no-archive` the dataset is compressed and the shards deleted once verified, so training would need a restore first.

### A2. Train the base model
```bash
python train_models.py --config training.config --mode mtg_lightning_opera_rainfall --stage base
```
- **Does** — builds the encoder-forecaster from `metadata.json`. Restores an archived dataset automatically.
- **Writes** — `models/coalition_<run_tag>.keras` **CRITICAL**
- **Writes** — `models/history_<run_tag>.json` — mode, source, stage, label type, epochs, wall time.
- **Writes** — `models/coalition_<run_tag>.meta.json` **CRITICAL** — the period the model was trained on, checked before feature-importance analysis so a model is never explained with data it was trained on.

### A3. Finetune the Swin head
```bash
python train_models.py --config training.config --mode mtg_lightning_opera_rainfall --stage finetune
```
- **Writes** — `models/coalition_<run_tag>_finetuned.keras`

### A4. Full-domain inference
```bash
python predict_full_domain.py --mode ... --date YYYY-MM-DD
python predict_full_domain.py --mode ... --pick csi|active [--top_n N] --validation_summary validation/<summary>.json
```
- **Does** — stitches overlapping Hann-weighted patches into a full canvas at `--stride 128` (50 % overlap), removing the 256-px tiling seams.
- **Writes** — `inference/predict_<run_tag>/*.npy`, `*_hyst.npy` — saved as arrays so a threshold sweep never re-runs inference.
- **Graph** — `*_hits.png`, `*_perclass_hits.png`
- **Alone** — `--pick csi` runs the `--top_n` best samples (default 5) by mean CSI over leads of a validation run, `--pick active` the `--top_n` with the most ground-truth-active pixels over the leads; the outputs are prefixed `csi_top<NN>_` or `active_top<NN>_` so the two sets coexist. No ground truth is needed at run time. `--validation_summary` applies the tuned thresholds per lead for both tracks (LOW and HIGH for rainfall); without it the fallback is 0.20 / 0.25.

### A5. Validation and threshold tuning
```bash
python validate_predictions.py --track rainfall --split test --mode <mode> --period f34
python validate_predictions.py --track rainfall --year Y --month M --mode <mode> --period f34
python validate_predictions.py --track rainfall --split test --baseline --period w44
# every rainfall model in one run: two RECONVECT models and the baseline
python validate_predictions.py --track rainfall --split test --mode mtg_lightning_opera_rainfall opera_radar_only_rainfall --period f34 w34 --baseline_period w44
```
- **Does** — scans the month for samples with a pixel at or above the selected threshold (`--rainfall_threshold_mmh`, default 10 mm/h), runs inference, and tunes the hysteresis HIGH per lead by maximising aggregate CSI.
- **Scope** — `--split test` scores the reference timesteps of the test split (the held-out set); `--year --month` scores a calendar month; both together restrict the split to that month. The scope, including the selection threshold, is in every output name: `rainfall_test_thr8mmh_<tag>_*`, `rainfall_<Y>_<M>_thr8mmh_<tag>_*`, `rainfall_test_<Y>_<M>_thr8mmh_<tag>_*`; runs at different thresholds never overwrite each other.
- **Writes** — `validation/<stem>_summary.json` (`<stem>` = `rainfall_<scope>_<tag>`; the figures of the run go to `validation/<stem>/`) **CRITICAL** — tuned thresholds and the `per_patch` block. `<tag>` is the model's artifact tag, so several models validated on one scope coexist; `generate_report` takes `--rainfall_tag` when there is more than one.
- **Writes** — `…_samples.csv`, with `csi_t+<k>` and the hits / misses / false-alarm percentages per lead at the tuned HIGH.
- **Alone** — `--baseline` validates SepConv-ens post-processed into the same classes (no hysteresis); `--max_samples N` caps a trial run; `--weights latest` scores the per-epoch checkpoint instead of the final save (tag suffixed `_latest`). The same flag exists on A4 and A6.
- **Tuning** — on samples drawn from the validation split, never on the scored ones. Rainfall: phase 1 picks LOW per lead with a plain-threshold sweep on p(argmax) (`--rainfall_low_threshold` fixes it instead); phase 2 picks the (LOW, HIGH) pair per lead among windows of `--rainfall_window` tiled on both sides of that LOW out to `--rainfall_reach`; the scope is scored at the pair. `--tune_leads first` tunes on t+1 alone and reuses the result for every lead (both tracks). Lightning: LOW is the evaluation's tuned threshold (read from `evaluation/eval_<tag>/`, `--lightning_low_threshold` overrides), HIGH is swept above it per lead. `--max_samples` draws both sets reproducibly per month (`--month_batch`, `--seed`).
- **Population** — the extraction predicts the split's records from the TFRecords (`--datasets_root`), as the evaluation does, and pastes each reference's DBSCAN-selected patches back onto the canvas, where the hysteresis runs against the records' labels. Pixels outside those patches are not scored: the validation measures the model on the storm patches the dataset was built from, the same population as the evaluation, and not on the dry remainder of the domain. Summaries from before this change scored the whole canvas and are not comparable.
- **Cost** — one pass over the records per phase, about 5,300 patches per split for f34, no reprojected field read; the OPERA maxima the sample selection needs are cached in `our_data/data_statistics/opera_rainfall_max_cache.json`; the rainfall phase 2 replays the phase-1 predictions from memory (`--cache_gb`, default a third of the physical memory) instead of predicting the tuning samples again. The numbers are unchanged by any of this.
- **Population** — as on the rainfall track: the split's records, predicted in batches from the TFRecords (`--datasets_root`, `--batch_size`), the student from the teacher's records with the HR channels sliced, each reference's patches pasted back onto the canvas for the hysteresis against the records' labels. The probabilities are each patch's own output; the Hann blend across overlapping positions belongs to the inference and the per-date visualisation, not to the extraction.
- **Cost** — one component labelling per sample and lead serves every HIGH candidate (identical results to thresholding each one); one pass over the records per phase.
- **Graph** — `…_metrics.png` (FAR/POD/CSI bars), `…_coverage.png` (IoU against class-weighted overlap, rainfall), `…_hmf.png` (per-sample hits / misses / false alarms, marker per lead, red line at 50 %, first / median / last sample dated on the x axis), `…_hmf_percentiles.png` (per lead the p10–p90 / p25–p75 / median of those rates and the share of samples above and below 50 %), `…_hysteresis.png` (per lead, the gain or loss of hits / misses / false alarms from the raw decision, argmax for rainfall and p ≥ LOW for lightning, to the post-processed one, up always meaning better, and the rates of both), `…_tuning_low.png` and `…_tuning_high.png` (rainfall: the phase-1 sweep and the phase-2 windows per lead) or `…_tuning.png` (lightning: the HIGH sweep per lead). All drawn from the saved summary and CSV; `python validate_predictions.py --plots <summary.json>` redraws them for either track. Per-date overlays with `--date`
- **Read by** — `generate_report`, `build_patch_ensemble`, `bundle_eval_scores`

### A6. Figures and report
```bash
python visualize_gt_vs_pred.py --mode ... --csv our_data/test_data_<source>_<period>.csv
python visualize_gt_vs_pred.py --mode ... --csv ... --pick csi|active [--top_n N] --validation_summary validation/<summary>.json
python generate_report.py --year Y --month M
```
- **Note** — the visualiser builds its ground truth from the patch files of the `--csv` rows, so a picked timestep must be a row of that CSV (validate on `--split test` and pass the test CSV).
- **Writes** — `visualize_gt_vs_pred_plots/…`, `validation/report_<Y>_<M>.pdf`

---

## 9. Branch B: ablation study and SepConv-ens baseline

Two questions, deliberately separated. The ablation asks what the **architecture** is worth
at matched inputs; the modality ladder asks what MTG and lightning add. Both arms are
radar-only and **neither is autoregressive**: the comparison stops at t+4, which the
composition builds from observations alone.

| Tag | Window | Steps | For |
|---|---|---|---|
| `w44` | past=4 / future=4 | 9 | `opera_sepconv_logz` — t+2 and t+4 read `t−4` |
| `w34` | past=3 / future=4 | 8 | `opera_radar_only_rainfall` — 4 input frames |

### B1. Both windows (repeat of step 6)
```bash
python extract_patch_seq_for_datasets.py --period w44 --past 4 --future 4 --start ... --end ...
python extract_patch_seq_for_datasets.py --period w34 --past 3 --future 4 --start ... --end ...
```
Then repeat steps 7–9 for each period.

### B2. Dataset contamination check between the two runs
```bash
python verification_keys.py --write --reconvect_tag w34 --sepconv_tag w44
```
- **Does** — establishes the samples on which two independently split runs may honestly be compared. It intersects the two test splits, then removes every sample that appears in either model's training or validation data. Different sequence windows place the same sample on opposite sides of the split, so without this each model would be scored partly on data the other had learned from — and on a different population besides.
- **Writes** — `verification_keys_<source>_<a>_vs_<b>.json` **CRITICAL** — the name records which pair it describes.
- **Note** — run this **before** any model reads test data. `--sepconv_tag` is required: with several windows on disk there is no safe default.

### B3. Datasets
```bash
python create_datasets.py --mode opera_sepconv_logz        --period w44 --no-archive
python create_datasets.py --mode opera_radar_only_rainfall --period w34 --no-archive
```
- **Check** — both must print a period-suffixed statistics file with **no** `<- overridden` note. Each model is normalised by its own split, so `--global_stats` stays absent from both.

### B4. Train
```bash
python sepconv_ensemble_training.py --period w44
python train_models.py --config training.config --mode opera_radar_only_rainfall --period w34 --stage base
```
- **Does** — trains the baseline's three base models and the RECONVECT-architecture ablation. Each `Bm{k}` is named for its **own** lead — how far ahead it predicts from the end of its own input window — not for the horizon it serves: Bm1 supplies t+1, Bm3 supplies t+2 and t+3, and Bm5 supplies t+4 because its window is shifted back one step. The **forecast horizon is t+4 = 60 min**; the number in a base model's name is its lead in steps, never a horizon in minutes.
- **Writes** — `models/sepconv_<run_tag>_bm{1,3,5}.keras`, `history_sepconv_<run_tag>.json` **CRITICAL**
- **Also writes** — `models/checkpoints/<name>_latest.keras` after every epoch, with a `.json` sidecar carrying the next-epoch index. Both models resume from it; `--fresh` ignores it. So each run leaves **two** states: best weights in the final save, last epoch in the checkpoint.
- **Note** — both read the same `training.config`: `[defaults]` supplies `epochs` and `batch_size` to each, and the optional `[sepconv]` section overrides them, inheriting `learning_rate` from `[lr_schedule].initial_lr` and `es_patience` from `[early_stopping].patience`. A gap between the two models therefore cannot be a gap in training budget. The learning-rate *schedule* is deliberately not unified — RECONVECT uses cosine warmup, the baseline reproduces the paper's `ReduceLROnPlateau`.
- **Alone** — `--datasets_root` / `--output_dir` place the dataset and the checkpoints on another disk.

### B5. Evaluate
```bash
python evaluate_sepconv_ensemble.py --period w44
python evaluate_coalition.py --mode opera_radar_only_rainfall --period w34
```
- **Does** — composes t+1…t+4 through `sepconv_compose`, denormalises with the window's own statistics, and bins in mm/h at the same class edges RECONVECT uses — so the two cannot be told apart by their thresholds.
- **Writes** — `evaluation/eval_sepconv_<run_tag>/evaluation_results.json`
- **Graph** — `metrics_per_leadtime.png`; `--plot_samples N` renders observed vs predicted class and predicted mm/h. Lightning evaluation adds `threshold_sweep.png`, pooled CSI against the decision threshold on the validation split (where it is chosen) and on the test split, with the operating point and the test optimum marked; that threshold is the hysteresis LOW of the lightning validation.
- **Graph** — training curves, one figure per quantity (`training_loss.png`, `training_<metric>.png`; `training_loss_bm<k>.png` for the baseline) with the best-epoch and last-epoch cut-off lines taken from the history and the latest checkpoint sidecar. Leads follow the dataset's label depth.
- **Alone** — `--weights best|latest` chooses which of the two saved states to score. `best` is the final save; `latest` is the per-epoch checkpoint. Comparing them shows whether the epochs after the best one were overfitting, or whether early stopping cut a run that was still improving.


**Both evaluators must be given the same frozen key set.** **CRITICAL**
The two windows are split independently, so each model's own test split is a
different population from the other's *and* overlaps the other's training
data (w34 test 5420, w44 test 5125, shared 4745; 380 of the baseline's test
keys were seen by RECONVECT while fitting). Scoring restricted to the
intersection is what makes the numbers comparable:

```bash
python verification_keys.py --write --reconvect_tag f34 w34 --sepconv_tag w44
python evaluate_sepconv_ensemble.py --period w44 \
    --verification_keys our_data/verification_keys_dbscan_f34_w34_vs_w44.json
python evaluate_coalition.py --mode mtg_lightning_opera_rainfall --period f34 \
    --verification_keys our_data/verification_keys_dbscan_f34_w34_vs_w44.json
python evaluate_coalition.py --mode opera_radar_only_rainfall --period w34 \
    --verification_keys our_data/verification_keys_dbscan_f34_w34_vs_w44.json
```

`--reconvect_tag` takes several windows: the set is then the intersection of every named test split minus every key any train or validation split holds, so the full model, the ablation and the baseline are all scored on one population. With one window it is the pairwise set of before.

The filter matches on `(date, reference_utc, patch)` recorded in each shard,
so it is exact rather than positional. Datasets built before those fields
existed match nothing and are refused with a rebuild instruction.

### B6. Modality ladder (feature importance)
```bash
python bundle_eval_scores.py
python feature_importance_analysis.py --model ... --data ... --methods gradcam_xi shap
```
- **Does** — collapses several runs into per-lead CSVs, then runs Grad-CAM/Xi, SHAP and classical Shapley. The ablation pair diffs two Xi matrices to show how the remaining inputs absorb a dropped group's role.
- **Writes** — `eval_leadtime-<prefix>-<letters>.csv`, `results/feature_importance/…`
- **Note** — letters encode which inputs a run saw: `o` = OPERA only, `om` = OPERA + MTG IR/WV.

### B7. Compare every model of a track
```bash
python compare_models.py --track rainfall --split test --include_baseline
python compare_models.py --track rainfall --year Y --month M
python compare_models.py --track lightning --split test
```
- **Does** — reads every model's `evaluation_results.json` and tagged validation `_samples.csv` for one track and draws them together: per-lead metric curves (one figure per metric plus a grid), a paper-style table with the best value in bold, and the share of samples whose hit percentage (rainfall: GT-active pixels detected at the tuned HIGH; lightning: per-sample POD) clears 50…90 % per lead, per season and over the whole window.
- **Writes** — `comparison/<track>/metrics_per_leadtime_<metric>.png`, `comparison_table.{csv,md,tex}`, `hit_levels_<scope>/hit_levels.png`, `hit_levels_<scope>/hit_levels.csv`, `comparison_summary.json` (deliverables; nothing reads them). `<scope>` follows `--split` / `--year --month`: `test`, `<Y>_<M>`, `test_<Y>_<M>`, or `all_months` when every whole-month run is pooled.
- **Note** — `--include_baseline` has an effect on the rainfall track only: no lightning baseline exists yet. `--models`, `--label`, `--hit_levels` and `--seasons` narrow or rename what is drawn.

---

## 10. Branch C: seasonal ensemble

One member per season, selected per patch by measured skill. The member set is registered
once and every later stage checks against that registration rather than against whatever
happens to be on disk.

### C1. Register the plan
```bash
python create_datasets.py --mode <mode> --ensemble [--seasons_config ...]
```
- **Does** — enumerates members from the season definitions, reports coverage and overlaps against the available data, and appends the resulting plan.
- **Writes** — `our_data/ensemble_registry.json` **CRITICAL** — append-only; later stages read the last state.
- **Note** — a member below 90 % coverage is reported `PARTIAL` but is still buildable.

### C2. Build and train each member
```bash
python create_datasets.py --mode <mode> --period 2025warm
python train_models.py --config training.config --mode <mode> --period 2025warm --stage base
```
- **Note** — `--period` resolves from sequence metadata first, then the registry, so window tags such as `w44` work without being registered members.
- **Check** — `python train_models.py --check-ensemble --mode <mode>` reports which members have datasets built.

### C3. Score each member per patch
```bash
python validate_predictions.py --track rainfall --year Y --month M --mode <mode>
```
- **Does** — produces the `per_patch` block the selector reads. Run once per member.
- **Writes** — `validation/<stem>_summary.json` with `per_patch` **CRITICAL**

### C4. Select per patch
```bash
python build_patch_ensemble.py --mode <mode> --track rainfall --split test --min_samples N
```
- **Does** — chooses the best-scoring member for each of the 18 patches. Selection only — scoring happened in C3.
- **Scope** — `--split test`, or `--year --month`, plus `--rainfall_threshold_mmh` (default 10), name the members' validation runs the same way validate_predictions did.
- **Rule** — `--min_samples N` is required. A patch validated on at least N samples takes its own best member; below N it takes the overall best member (highest CSI pooled over every patch). If no patch reaches N, patches with samples keep their own best member and only patches with zero samples take the overall best. The manifest records the rule per patch.
- **Writes** — `our_data/ensemble_manifest_<mode>_<source>.json` **CRITICAL** — the routing table, with season and global fallbacks behind each assignment.
- **Read by** — `ensemble_inference`

### C5. Route at inference
`ensemble_inference.PatchEnsemble` resolves a patch to its assigned member, falling back to
the season member and then the global model.

---

## 11. Branch D: knowledge distillation

Trains a student that runs **without lightning input**, so the lightning track can be used
on dates where LINET is unavailable.

### D1. Train the teacher
```bash
python create_datasets.py --mode mtg_lightning_opera_occurrence --period f34
python train_models.py --config training.config --mode mtg_lightning_opera_occurrence --period f34 --stage base
```
- **Does** — the teacher receives the full input stack including LINET `density`, `current`, `occurrence`.
- **Writes** — `models/coalition_mtg_lightning_opera_occurrence_<source>.keras` **CRITICAL**

### D2. Distil the student
```bash
python train_lightning_kd.py --period f34 --epochs 30 --batch_size 32 --datasets_root E:/nowcasting/datasets --model_dir E:/nowcasting/models
```
- **Does** — trains on the **teacher's** dataset with `past_hr` sliced to the trailing `STUDENT_HR_CHANNELS` (= `vis_06`), so the student never sees lightning. Loss mixes soft teacher targets at `--kd_alpha 0.7`, temperature 4.0, with the ground truth.
- **Loss** — soft term: per-pixel KL between the temperature-softened teacher and student (T = 2 by default), weighted by the raw teacher probability plus a floor of 0.01 so the gradient sits where the teacher has an opinion, times T²; hard term: the teacher's focal loss against LINET; total = 0.7·soft + 0.3·hard. The logged soft value is the mismatch alone, not the teacher's entropy. `--kd_temperature`, `--kd_soft_weight`, `--kd_weight_floor`, `--kd_alpha` change these.
- **Learning rate** — the same cosine-with-warmup schedule as the base models, read from `[lr_schedule]` in `training.config` (`--config`; `--learning_rate` overrides its `initial_lr`); no plateau reduction on top.
- **Graph** — after the run, `evaluation/eval_<student_tag>_kd/training_{loss,l_soft,l_hard}.png`, with the exact values at the best and the last epoch in the legend.
- **Note** — `mtg_opera_occurrence` is **not** buildable by `create_datasets`: it exists only as a student over the teacher's data, so no second dataset is needed. `--period` names the teacher's period; the student is saved as `coalition_mtg_opera_occurrence_<source>_<period>_kd.keras`.
- **Writes** — `models/coalition_<student_run_tag>_kd.keras`, `history_…_kd.json` **CRITICAL**

### D3. Evaluate and validate the student like every other model
```bash
python evaluate_coalition.py --mode mtg_opera_occurrence --kd --period f34 --split test --datasets_root E:/nowcasting/datasets --model_dir E:/nowcasting/models
python validate_predictions.py --track lightning --split test --mode mtg_lightning_opera_occurrence mtg_opera_occurrence:kd --period f34 --model_dir E:/nowcasting/models
```
- **Does** — evaluation reads the teacher's dataset (named in the student's history) and slices `past_hr` to the student's channels, tunes the decision threshold on the validation split and scores the test split; the lightning validation track then takes that threshold as LOW, tunes HIGH per lead and scores the scope, for the teacher and the student in one run (`mode:kd` names the variant per entry). Both land in `compare_models --track lightning` as two models.
- **Writes** — `evaluation/eval_mtg_opera_occurrence_<source>_<period>_kd/`, `validation/lightning_<scope>_mtg_opera_occurrence_<source>_<period>_kd_*`
- **Alone** — `--track kd` is the older paired run: both models on identical samples with each one's HIGH tuned independently, written as `validation/kd_<scope>_*`.

---

## 12. Artefact criticality

**Shared by every period.** Five artefacts carry no source or period tag and are used by
every run, so rebuilding any of them affects all of them at once:

```
timestep_config.json      timestep_manifest.csv     patch_index.csv
our_data/patches/         reprojected_data/
```

**Tagged per run.** Everything else is named `<source>[_<period>]`, where `source` is
always `dbscan` (the DBSCAN sample-selection method — a constant, not a flag) and `period`
is the optional `--period` label. `run_tag` is `<mode>_<source>[_<period>]`, built by
`build_run_tag()` — the single source of truth for model, checkpoint and dataset names.

Note `--source` on `pipeline_msg_mtg.py` is unrelated: there it means download origin.

**Terminal artefacts — safe to delete.** Nothing downstream reads these; the script that
made them will make them again:

| Artefact | Producer |
|---|---|
| `<product>_coverage.png` | the three summarisers |
| `intersect_summary.png` | `intersect_product_coverage` |
| `patch_index/plots/{dbscan_<rule>_patch_selection,patch_highlight}/<date>.gif` and `plots/nc/` | `identify_patches --plot` |
| `mtg_store_distribution.png` | `store_registry --chart` |
| `inference/` figures, `visualize_gt_vs_pred_plots/` | `predict_full_domain`, `visualize_gt_vs_pred` |
| `evaluation/` figures | `evaluate_*` |
| `comparison/<track>/` | `compare_models` (deliverable) |
| `results/feature_importance/` | `feature_importance_analysis` |
| `validation/report_<Y>_<M>.pdf` | `generate_report` (deliverable) |
| `our_data/data_statistics/` | `data_statistics` |

### Patch pool staleness

`our_data/patches/` is shared by every period — a patch file depends only on
(date, time, variable, resolution) and `patch_index.csv`. A new period does not
invalidate it; a rebuilt `patch_index.csv` does. **CRITICAL**

A patch file is an array of tiles with no record of which tile is in which slot;
slot *k* means "the *k*-th active patch here". When a patch becomes active it
inserts mid-list and shifts every later slot:

```
file holds : [2, 3, 4,    7, 8, 9, 13, 14]
index says : [2, 3, 4, 5, 7, 8, 9, 13, 14]
             ok ok ok  ^-- shifted from here
```

Only the final slot goes out of range. The shifted ones read cleanly and return
the **wrong tile** — patches ~1,000 km apart on the domain — pairing one region's
input with another's label. Since `extract_patches` skips existing files, the
state is sticky.

| guard | effect |
|---|---|
| `our_data/patches/<date>/_patch_index.json` | Stamps the active list each file was built from; drifted timesteps are re-extracted, not skipped. |
| `create_datasets.StalePatchPool` | Out-of-range `idx_t*` raises instead of zero-filling. |

```bash
python extract_patches.py --audit_pool [--period TAG]
```

A *missing* variable still zero-fills — that is legitimate. An out-of-range slot
never is. Overwriting a stale file also removes its `.npy.zst` twin.

**Purge stale files before compressing the pool** — compression rewrites mtimes,
and for a pre-stamp pool mtime is the only staleness evidence.

### Where the data lives

All defaults resolve against the repository, not the working directory, so the
scripts run from anywhere without a `cd`.

| Root | Flag | Env var | Default |
|---|---|---|---|
| patches, CSVs, statistics | `--data_root` | `COALITION4_DATA_ROOT` | `<repo>/our_data` |
| TFRecord datasets | `--datasets_root` | `COALITION4_DATASETS_ROOT` | `<data_root>/datasets` |
| checkpoints | `--model_dir` | `COALITION4_MODEL_DIR` | `<repo>/models` |

Precedence: flag > env var > default. `datasets/` resolves separately from
`data_root` so the datasets can sit on a different disk from the terabyte-scale
patch pool — previously only possible with an NTFS junction. **CRITICAL**
Archive locks and in-use markers follow `--datasets_root`, keeping the
restore/reclaim lifecycle consistent with training.

### Reclaiming disk: compressing the `.npy` stores

The arrays are the bulk of the project — **5,292 GB across 1.10 M files**, against ~66 GB
of built datasets. They are compressed **in place** with zstd, not archived whole,
because the pipeline opens them one frame at a time by name. Every
reader resolves the logical `.npy` to whichever form is on disk, so **nothing needs
restoring before a run**.

```bash
python compress_datasets.py --npy-stats our_data/reprojected_data     # project first
python compress_datasets.py --compress-npy our_data/reprojected_data our_data/patches
python compress_datasets.py --restore-npy DIR                         # and back
python compress_datasets.py --move TAG --to ROOT      # carry a compressed dataset elsewhere and decompress it
```

| Target | On disk | Ratio | After |
|---|---|---|---|
| `our_data/patches/` | 66.8 GB | 11.3× | 5.9 GB |
| `reprojected_data/satellite_data/MTG/` | 1313.4 GB | 8.8× | 148.5 GB |
| `reprojected_data/opera_data/` | 519.3 GB | 37.6× | 13.8 GB |
| `our_data/lightning_data/` | 313.4 GB | 16180× | ~0 GB |
| MTG store (E: + G:) | 2765.2 GB | 6.1× | 454.0 GB |
| **Total** | **5292.3 GB** | **8.5×** | **622.3 GB** |

- **Level 10 on the `float32` exactly as stored** — no dtype change, no requantisation, no
  byte shuffle. The numerical base is untouched, so no calculation downstream can shift.
- **Deletion safety** — each file is written to `.tmp`, read back off disk and compared
  byte for byte against the original before anything is unlinked. The whole `.npy` is
  stored, header included, so a restore is byte-identical by construction. **CRITICAL**
- **Not solid archives** — measured over 24 consecutive frames, one solid stream buys 3 %
  (11.3× → 11.7×) and costs random access; a temporal delta is worse (10.8×).
- `romania_grid_lats.npy` / `_lons.npy` are never compressed — small, and read everywhere.

**Order that cannot be reversed.**

1. Statistics before the dataset that uses them.
2. The contamination check before any model reads test data.
3. Coverage summary before the Data Store backfill — the backfill requests exactly the cycles the missing-timestep JSON names.
4. Coverage summary before deleting raw chunks — once they are gone, `--scan raw` can no longer describe the archive, and only the `npy` and `reprojected` views remain valid.
5. Patch extraction before dataset creation, and re-extraction whenever `patch_index.csv` is rebuilt — a patch file records tiles by position, so a changed activity list silently shifts which tile each position holds.
