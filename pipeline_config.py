"""
pipeline_config.py — COALITION-4 Romanian Adaptation
=====================================================
Project-wide constants that more than one pipeline stage needs to agree
on. Deliberately dependency-free (no TensorFlow, no numpy) so the light
CLIs can import it without paying for a TF import.

Sample-selection source
-----------------------
Every artefact this project writes is suffixed with the sample-selection
source: `datasets/<mode>_<source>/`, `coalition_<mode>_<source>.keras`,
`train_data_<source>.csv`, `normalization_stats_<source>.json`, and so
on. See train_models.build_run_tag.

Historically there were two sources — `dbscan` (DBSCAN clusters in the
OPERA rain field, via identify_patches.py) and `lightning`
(lightning-active windows). The lightning-driven track was retired:
patches are now extracted from OPERA only, so `dbscan` is the sole
source and `--source` is no longer a CLI flag anywhere.

The `source` *parameter* is still threaded through the functions that
build paths, defaulting to SOURCE. That keeps the on-disk naming stable
(nothing had to be renamed when the flag went away) and leaves a seam if
a second selection strategy is ever introduced.
"""

# The only sample-selection source. DBSCAN clustering over OPERA
# `rainfall_rate`, produced by identify_patches.py.
SOURCE = "dbscan"
