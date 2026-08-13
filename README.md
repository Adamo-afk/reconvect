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

| # | Script | Arguments | What the step does | Commands |
|---|---|---|---|---|
| **0** | `validate_timestep.py` | `--step_minutes` desired training step · `--cadences_file` product cadence config · `--output_path` · `--print` show existing config and exit | Picks the master cadence and derives each product's minute filter → `our_data/timestep_config.json`, read by every later step. | `python validate_timestep.py --step_minutes 15`<br>`python validate_timestep.py --print` |
| **1a** | `our_data/satellite_data/pipeline_msg_mtg.py` | `--start` `--end` range · `--password_file` 2-line credentials · `--products_file` channel list JSON · `--output_dir` · `--timesteps` · `--workers` · `--full_disk` · `--skip_download` | Downloads + pre-processes MTG FCI L1C. | `python our_data/satellite_data/pipeline_msg_mtg.py --start 2025/05/01-0000 --end 2025/05/31-2350 --password_file creds.txt` |
| **1b** | `our_data/opera_data/pipeline_opera.py` | `--start` `--end` · `--products` reflectivity/rainfall_rate · `--remote_base` EWC mount root · `--remote_host` `--remote_user` `--ssh_key` · `--password_file` · `--cache_dir` · `--timesteps` | Fetches OPERA composite HDF5. If the default `--remote_base /eumetsatdata` errors with "No such file", pass `/home/eumetsatdata` — the mount root differs between EWC images. | `python our_data/opera_data/pipeline_opera.py --start 2025-05-01 --end 2025-05-31`<br>`… --remote_base /home/eumetsatdata` |
| **1c** | `our_data/lightning_data/linet_export.py` | `--start` `--end` · `--format` txt/kml/asc · `--out` · `--bbox` · `--password_file` · `--lightning-type` · `--amp-threshold` · `--daily-window` · `--pause` · `--force` · `--dry-run` | Downloads LINET strokes. Use `--format kml`: it writes `{out}/kml_data/YYYY-MM-DD/…` which the rasteriser reads directly. | `python our_data/lightning_data/linet_export.py --start 2025-05-01 --end 2025-05-31 --format kml --out our_data/lightning_data --password_file creds.txt` |
| **1d** | `our_data/lightning_data/read_kml_version2.py` | `--data_root` · `--output_root` · `--date` single date · `--force` overwrite | Rasterises strokes onto the 1 km Romania grid → `density`, `current`, `occurrence`. | `python our_data/lightning_data/read_kml_version2.py --data_root our_data` |
| **2** | `reproject.py` | `--satellite MTG` \| `--lightning` \| `--opera` \| `--all` (mutually exclusive, required) · `--data_root` · `--date` · `--workers` | Regrids everything onto the 1536 × 768 EPSG:31700 canvas as `.npy`. Also writes the shared `romania_grid_{lats,lons}.npy` and per-source projection constants so the arrays stay self-recoverable. | `python reproject.py --all`<br>`python reproject.py --opera --workers 6`<br>`python reproject.py --satellite MTG --date 2025-05-14` |
| **3** | `intersect_product_coverage.py` | `--summary` per-product summary CSVs · `--missing` missing-timestep JSONs · `--active` activity index · `--errors_log` · `--timestep_config` · `--output_csv` · `--output_plot` | Intersects per-product coverage into `timestep_manifest.csv` — the timesteps where *all* products exist. Gates step 5. | `python intersect_product_coverage.py --summary mtg=mtg_summary.csv --summary opera=opera_summary.csv` |
| **4** | `identify_patches.py` | `--threshold` rain-rate cut mm/h (10) · `--eps` DBSCAN radius px (5) · `--min_samples` min cluster px (20) · `--data_root` `--output_dir` · `--date` \| `--start`/`--end` · `--plot` PNG per active step | DBSCAN over OPERA `rainfall_rate`; marks which of the 18 patches are convectively active per timestep → `patch_index.csv`. | `python identify_patches.py`<br>`python identify_patches.py --start 2025-05-01 --end 2025-05-31`<br>`python identify_patches.py --date 2025-05-14 --plot` |
| **5** | `extract_patch_seq_for_datasets.py` | `--past` past steps (2) · `--future` future steps (3) · `--test_frac` (0.1) · `--val_frac` (0.1) · `--block_hours` block size, must divide 24 (6) · `--manifest` path or `none` to disable the gate · `--data_root` | Builds temporally-continuous sequences and the Czibula block-wise 80/10/10 split → `{train,validation,test}_data_dbscan.csv` + `sequence_meta_dbscan.json`. | `python extract_patch_seq_for_datasets.py`<br>`python extract_patch_seq_for_datasets.py --past 3 --future 3 --block_hours 12`<br>`python extract_patch_seq_for_datasets.py --manifest none` |
| **6** | `extract_patches.py` | `--products` `satellite_MTG` / `lightning` / `opera` (default all) · `--data_root` `--output_dir` · `--date` | Slices 256 × 256 patches from the reprojected canvases, applying each variable's pooling factor → `patches/{date}/{var}_{HHMM}_{HR\|LR}.npy`. | `python extract_patches.py`<br>`python extract_patches.py --products satellite_MTG opera`<br>`python extract_patches.py --date 2025-05-14` |
| **7** | `compute_normalization_stats.py` | `--variables` subset · `--sample_fraction` · `--with_percentiles` p01/p50/p99 + MAD · `--reservoir_size` · `--device` auto/cpu/gpu · `--no_split_filter` **diagnostic only — leaks val/test** · `--train_csv` `--sequence_meta` `--timestep_config` `--reproject_root` `--output` `--seed` | Per-variable mean/std over the **training split only** → `normalization_stats_dbscan.json`. Required by step 8; there is no fallback, and a missing variable fails loudly. | `python compute_normalization_stats.py`<br>`python compute_normalization_stats.py --variables ir_105 opera_rainfall_rate`<br>`python compute_normalization_stats.py --device gpu --with_percentiles` |
| **8** | `create_datasets.py` | `--mode` one of the 5 buildable modes · `--data_root` · `--output_root` | Applies transforms + label binning, writes TFRecord shards plus a per-split `metadata.json` (input shapes, label type, cadence) that drives model construction. | `python create_datasets.py --mode mtg_opera_mtgmr_rainfall`<br>`python create_datasets.py --mode mtg_lightning_opera_occurrence`<br>`python create_datasets.py --mode mtg_opera_mtgmr_continuous` |
| **9a** | `lightning_fraction.py` | `--scope_csv` scope CSV or `none` for every file on disk · `--data_root` · `--output` | **Lightning modes only.** Training-scope positive-pixel fraction → `lightning_fraction_dbscan.json`, the focal-loss prior. | `python lightning_fraction.py`<br>`python lightning_fraction.py --scope_csv none` |
| **9b** | `opera_rainfall_fraction.py` | `--scope_csv` · `--data_root` · `--output` | **Rainfall modes only, when `[radar_loss].weighting != none`.** Per-class pixel fractions → `opera_rainfall_fraction_dbscan.json`, the class-weight prior. | `python opera_rainfall_fraction.py` |
| **10** | `train_models.py` | `--config` · `--mode` single mode (else `[modes].run`) · `--stage` `base`/`finetune`/`both` · `--base_checkpoint` frozen backbone · `--dataset_dir` override path · `--output_dir` `--data_root` · `--fresh` ignore checkpoint · `--list-modes` | Builds the encoder-forecaster from `metadata.json` and trains. `finetune` freezes the backbone and grafts a Swin head; `both` runs the two back-to-back in one process. Resumes from the per-epoch checkpoint unless `--fresh`. | `python train_models.py --list-modes`<br>`python train_models.py --mode mtg_opera_mtgmr_rainfall --stage base`<br>`python train_models.py --mode mtg_lightning_opera_occurrence --stage both`<br>`python train_models.py --config training.config`<br>`python train_models.py --mode mtg_opera_mtgmr_rainfall --stage base --fresh` |
| **11** | `train_lightning_kd.py` | `--kd_alpha` (0.7) · `--kd_temperature` (4.0) · `--teacher_finetuned` distil from the Swin teacher · `--epochs` (50) · `--batch_size` (8) · `--learning_rate` (1e-4) · `--patience` (10) · `--shuffle_buffer` · `--seed` · `--no_mixed_precision` · `--data_root` `--model_dir` | **Optional.** Distils the teacher into `mtg_opera_occurrence`, a student predicting lightning from satellite + OPERA with **no LINET at inference**. | `python train_lightning_kd.py`<br>`python train_lightning_kd.py --kd_alpha 0.5 --kd_temperature 6.0`<br>`python train_lightning_kd.py --teacher_finetuned` |
| **12** | `sepconv_ensemble_training.py` | `--mode` `mtg_opera_mtgmr_continuous` · `--lead` train only lead 1/2/3 · `--epochs` (50) · `--batch_size` (8) · `--data_root` `--model_dir` | **Optional baseline.** Three SepConv regression models (one per lead) on the same inputs as the continuous COALITION-4 mode. | `python sepconv_ensemble_training.py --mode mtg_opera_mtgmr_continuous`<br>`python sepconv_ensemble_training.py --mode mtg_opera_mtgmr_continuous --lead 1` |

### OPERA SFTP notes (step 1b)

Two distinct failures both surface as `cannot list …` — tell them apart by what follows.

| Symptom | Cause | What to do |
|---|---|---|
| `cannot list …: Permission denied` | The EWC VM rejects password authentication. | Switch to key auth: `--ssh_key ~/.ssh/id_ed25519`, or any other key registered under `claudiu@` on the server. |
| `cannot list …: No such file`, per date, at a mount root that already resolved | None — this is upstream. `--remote_base` auto-fallback already picked the correct EWC mount, so the remote directories genuinely do not exist for those dates. | Nothing to fix locally. Ask the NMA (National Meteorological Administration) data operators when the target range will land. |

```bash
python our_data/opera_data/pipeline_opera.py --start 2025-05-01 --end 2025-05-31 \
    --ssh_key ~/.ssh/id_ed25519
```

---

## Table 2 — Validation, inference, visualisation & analysis

| # | Script | Arguments | What the step does | Commands |
|---|---|---|---|---|
| **1** | `evaluate_coalition.py` | `--mode` (required) · `--split` train/validation/test · `--finetuned` \| `--kd` · `--threshold` fixed decision threshold (else optimised on validation) · `--plot_threshold` · `--date` `--hour` sample figure · `--batch_size` · `--data_root` `--model_dir` `--output_dir` | Full metric suite on a held-out split → `evaluation/eval_<run_tag>/evaluation_results.json` + plots. Per-lead metrics feed the coalition study. | `python evaluate_coalition.py --mode mtg_opera_mtgmr_rainfall`<br>`python evaluate_coalition.py --mode mtg_lightning_opera_occurrence --finetuned`<br>`python evaluate_coalition.py --mode mtg_opera_occurrence --kd`<br>`python evaluate_coalition.py --mode mtg_opera_mtgmr_rainfall --split validation` |
| **2** | `evaluate_sepconv_ensemble.py` | `--mode` `mtg_opera_mtgmr_continuous` · `--split` · `--date` `--hour` · `--batch_size` · `--data_root` `--model_dir` `--output_dir` | Evaluates the SepConv baseline. Continuous predictions are also binned to the 5 rainfall classes so both comparison modes are reported. | `python evaluate_sepconv_ensemble.py --mode mtg_opera_mtgmr_continuous` |
| **3** | `validate_predictions.py` | `--track` `rainfall`/`lightning`/`kd` (required) · `--year` `--month` (required) · `--date` switches to visualisation · `--mode` · `--finetuned` \| `--kd` · `--stride` Hann stride (128) · `--lightning_low_threshold` (0.90) · `--rainfall_threshold_mmh` selection cut (10) · `--high_coverage_pct` (90) · `--teacher_mode` `--student_mode` `--teacher_finetuned` `--no_student_kd` (kd track) · `--batch_size` · `--data_root` `--model_dir` `--output_dir` | **Extraction (no `--date`):** scans the month for samples with ≥1 pixel ≥10 mm/h, runs inference, tunes the per-lead hysteresis HIGH by maximising aggregate CSI, writes CSV + summary JSON + metrics figure. **Visualisation (`--date`):** plots overlays for that day using the tuned thresholds. `kd` runs teacher and student on identical samples and tunes each independently. | `python validate_predictions.py --track rainfall --year 2025 --month 5`<br>`python validate_predictions.py --track rainfall --year 2025 --month 5 --finetuned`<br>`python validate_predictions.py --track lightning --year 2025 --month 5 --mode mtg_lightning_opera_occurrence`<br>`python validate_predictions.py --track lightning --year 2025 --month 5 --mode mtg_lightning_opera_occurrence --date 2025-05-14`<br>`python validate_predictions.py --track kd --year 2025 --month 5` |
| **4** | `predict_full_domain.py` | `--mode` `--date` (required) · `--time` \| `--hour` \| `--start-time`/`--end-time` · `--finetuned` \| `--kd` · `--validation_summary` load tuned per-lead thresholds · `--lightning_low_threshold` `--lightning_high_threshold` · `--rainfall_low_threshold` `--rainfall_high_threshold` · `--stride` · `--patches` subset of the 18 (rainfall only) · `--threshold` · `--batch_size` · `--no-plot` `--save-npy` · `--data_root` `--model_dir` `--output_dir` | **Operational inference on any date.** Reads the reprojected full-domain fields and slices all 18 patches on the fly — touches no `patch_index.csv`, split CSV, or pre-extracted patch tile, so it runs on dates the training pipeline has never seen. | `python predict_full_domain.py --mode mtg_opera_mtgmr_rainfall --date 2026-06-30`<br>`python predict_full_domain.py --mode mtg_opera_mtgmr_rainfall --date 2026-06-30 --hour 14`<br>`python predict_full_domain.py --mode mtg_lightning_opera_occurrence --date 2026-06-30 --time 14:30`<br>`python predict_full_domain.py --mode mtg_lightning_opera_occurrence --date 2026-06-30 --validation_summary validation/lightning_2025_05_summary.json`<br>`python predict_full_domain.py --mode mtg_opera_mtgmr_rainfall --date 2026-06-30 --start-time 12:00 --end-time 15:45 --save-npy` |
| **5** | `visualize_gt_vs_pred.py` | `--csv` split CSV (required) · `--mode` (required) · `--top_n` highest-activity references (5) · `--finetuned` \| `--kd` · `--eval_results` threshold source · `--threshold` · `--no_zoom` · `--no_aggregate_graphs` · `--stride` · same 4 hysteresis flags as above · `--validation_summary` · `--batch_size` · `--data_root` `--model_dir` `--output_dir` | **Training-scope visualiser.** Ranks references in a split by qualifying-patch count and renders GT beside predictions, plus a zoom on the patch with the most GT activity. Hysteresis knobs mirror `predict_full_domain.py` so both render identical post-processing. | `python visualize_gt_vs_pred.py --csv our_data/test_data_dbscan.csv --mode mtg_opera_mtgmr_rainfall`<br>`python visualize_gt_vs_pred.py --csv our_data/validation_data_dbscan.csv --mode mtg_lightning_opera_occurrence --top_n 3`<br>`python visualize_gt_vs_pred.py --csv our_data/test_data_dbscan.csv --mode mtg_opera_occurrence --kd --no_zoom` |
| **6** | `generate_report.py` | `--year` `--month` (required) · `--track` `rainfall`/`lightning`/`both` · `--language` `ro`/`en` · `--bilingual` · `--model` Ollama tag · `--temperature` (0.1) · `--seed` (42) · `--max_tokens` (2000) · `--refresh_cache` `--no_cache` · `--skip_pdf` · `--pred_coupling` · `--validation_dir` `--assets_dir` `--output` `--data_root` `--model_dir` | Builds a PDF from `validate_predictions.py` outputs with commentary from a local Ollama LLM. `--language en` skips the Romanian translation phase and halves the LLM calls. `--track both` requires both tracks' extraction outputs on disk. | `python generate_report.py --year 2025 --month 5`<br>`python generate_report.py --year 2025 --month 5 --language en`<br>`python generate_report.py --year 2025 --month 5 --track lightning --bilingual`<br>`python generate_report.py --year 2025 --month 5 --skip_pdf --no_cache` |
| **7** | `bundle_eval_scores.py` | `--mode` repeatable `MODE=LETTERS` · `--prefix` · `--metric` override auto-detection · `--eval_root` `--output_dir` · `--finetuned` | Converts each mode's `evaluation_results.json` into the per-lead-time CSVs classical Shapley expects. Coalition letters encode which input groups a model saw (`o` = OPERA only, `om` = + MTG IR/WV). | `python bundle_eval_scores.py`<br>`python bundle_eval_scores.py --metric HSS`<br>`python bundle_eval_scores.py --mode "mtg_opera_radar_only_rainfall=o" --mode "mtg_opera_mtgmr_rainfall=om"` |
| **8** | `feature_importance_analysis.py` | `--model` checkpoint · `--data` test dataset dir · `--output` · `--methods` `gradcam_xi` / `shap` / `classical_shapley` · `--num-samples` · `--scores-dir` for classical Shapley · `--model-ablated` `--data-ablated` second model for the ablation diff | Grad-CAM + Xi correlation (spatial attention), SHAP (pixel importance), and classical Shapley (source-level). The ablation pair diffs two Xi matrices to show how remaining inputs absorb a dropped group's role. | `python feature_importance_analysis.py --model models/coalition_mtg_opera_mtgmr_rainfall_dbscan.keras --data our_data/datasets/mtg_opera_mtgmr_rainfall_dbscan/test --output results/fi --methods gradcam_xi`<br>`… --methods gradcam_xi shap`<br>`… --model-ablated models/coalition_mtg_opera_radar_only_rainfall_dbscan.keras --data-ablated our_data/datasets/mtg_opera_radar_only_rainfall_dbscan/test` |
| **9** | `data_statistics.py` | `--split` train/validation/test · `--csv` explicit override · `--data_root` | Six dataset diagnostic panels: diurnal cycle, spatial heatmap, daily timeline, simultaneously-active patches, samples per date, patch survival. | `python data_statistics.py`<br>`python data_statistics.py --split test` |

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
| Output head | `Conv2D(1, 1×1, sigmoid)` | lightning / continuous | — |
| Output head | `Conv2D(5, 1×1, softmax)` | rainfall 5-class | — |
| Past / future timesteps | 3 / 3 | all | `sequence_meta_dbscan.json` |
| **Base training** (`[defaults]`) ||||
| Optimizer | `Adam(lr=1e-3)` | base stage | — |
| Loss | `WeightedFocalLoss(gamma=2.0)`, prior from `lightning_fraction_dbscan.json` (~1 % positive pixels) | lightning | — |
| Loss | `WeightedFocalCategoricalCrossentropy` (see `[radar_loss]`) | rainfall 5-class | — |
| Metrics | `iou_metric`, `true_pos`, `false_pos`, `false_neg` | lightning | — |
| Metrics | `accuracy` | rainfall | — |
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
| Loss | weighted MSE, bin weights `[15, 1, 2, 7, 15, 30, 1000]` | baseline | — |
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
