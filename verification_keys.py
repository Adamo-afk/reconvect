"""
verification_keys.py — the leakage-free key set for the baseline comparison
===========================================================================
RECONVECT and the SepConv-ens baseline need different sequence windows
(w34 = past=3/future=4 vs w44 = past=4/future=4), so they cannot share
one split. They are split independently, which creates a hazard: the
Czibula splitter assigns sequences to train/validation/test by position
inside each 6-hour block, and the same (date, reference_utc, patch)
lands in different splits under different window lengths.

Measured on w34 vs w44: the two test splits hold 5420 and 5125 keys but
share only 4745, so scoring each model on its own split compares them on
different populations. Worse, 380 of the baseline's test keys (7 train +
373 validation) were seen by RECONVECT during fitting.

The frozen set is not advisory. Pass it to the evaluator:

    python evaluate_sepconv_ensemble.py --period w44         --verification_keys our_data/verification_keys_dbscan_w34_vs_w44.json

Without it each model is scored on its full own split, and the
difference between them is partly a difference in test population.

The fix is not to trust the intersection but to prove it:

    verification set = (RECONVECT test) INTERSECT (SepConv test)
                       MINUS  every key in either model's train or
                              validation split

The subtraction is currently a no-op — the intersection happens to be
clean — but it is applied regardless, because "happens to be clean today"
is not a property that survives a re-split.

Usage
-----
    python verification_keys.py                    # report
    python verification_keys.py --write            # + freeze to JSON

Freeze the set before the test data is touched, and score from the
frozen file thereafter.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path

from pipeline_config import SOURCE, resolve_data_root
from periods import split_csv_name

# The two windows under comparison. `None` = RECONVECT's unsuffixed
# whole-archive split; "w48" = the past=4/future=8 set the baseline needs.
# Both windows are named on the command line. Hardcoding either one
# means the frozen key set silently describes a pair of splits that
# may not be the pair being compared.
RECONVECT_TAG = None   # None = the unsuffixed whole-archive split

# The frozen name carries BOTH window tags: a key set describes one
# specific pair of splits, and two pairs writing to one filename would
# leave a file that silently claims to cover a comparison it does not.
FROZEN_NAME = "verification_keys_{source}_{reconvect}_vs_{sepconv}.json"


def _as_tags(reconvect_tag) -> list:
    """One tag, several tags, or None (the unsuffixed split) as a list."""
    if reconvect_tag is None or isinstance(reconvect_tag, str):
        return [reconvect_tag]
    return list(reconvect_tag)


def frozen_name(source: str, reconvect_tag, sepconv_tag) -> str:
    """Filename for a frozen key set, naming every split it was built
    from: verification_keys_dbscan_f34_w34_vs_w44.json."""
    return FROZEN_NAME.format(
        source=source,
        reconvect="_".join(t or "base" for t in _as_tags(reconvect_tag)),
        sepconv=sepconv_tag)


def load_keys(csv_path: Path) -> set[tuple[str, str, int]]:
    """Every (date, reference_utc, patch) a split CSV contributes.

    One CSV row can carry several active patches, and each becomes its
    own sample downstream, so the patch number is part of the key.
    """
    if not csv_path.is_file():
        raise SystemExit(f"Split CSV not found: {csv_path}")
    keys: set[tuple[str, str, int]] = set()
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for patch in ast.literal_eval(row["patch_numbers"]):
                keys.add((row["date"], row["reference_utc"], int(patch)))
    return keys


def _splits(data_root: Path, tag) -> dict[str, set]:
    return {
        split: load_keys(data_root / split_csv_name(split, SOURCE, tag))
        for split in ("train", "validation", "test")
    }


def build(data_root: Path, reconvect_tag=RECONVECT_TAG,
          sepconv_tag=None) -> dict:
    """Compute the verification key set and everything worth reporting.

    `reconvect_tag` may name several RECONVECT windows (f34, w34, ...):
    the set is then the intersection of every test split, minus every
    key that any of the splits' train or validation sets holds, so all
    models are scored on one population.
    """
    tags = _as_tags(reconvect_tag)
    rs = {t: _splits(data_root, t) for t in tags}
    s = _splits(data_root, sepconv_tag)

    naive = set(s["test"])
    for r in rs.values():
        naive &= r["test"]
    seen = s["train"] | s["validation"]
    for r in rs.values():
        seen |= r["train"] | r["validation"]
    contaminated = naive & seen
    clean = naive - seen

    # Reported even though it does not affect the result: it is the
    # measure of how far the splits actually disagree, and it is the
    # number that would grow silently if any window changed.
    cross = {}
    for t, r in rs.items():
        name = t or "base"
        cross[f"sepconv_test_in_{name}_train"] = len(s["test"] & r["train"])
        cross[f"sepconv_test_in_{name}_val"] = len(s["test"] & r["validation"])
        cross[f"{name}_test_in_sepconv_train"] = len(r["test"] & s["train"])
        cross[f"{name}_test_in_sepconv_val"] = len(r["test"] & s["validation"])

    patches = Counter(k[2] for k in clean)
    return {
        "source": SOURCE,
        "reconvect_tag": tags[0] if len(tags) == 1 else None,
        "reconvect_tags": tags,
        "sepconv_tag": sepconv_tag,
        "counts": {
            "reconvect_test": {(t or "base"): len(r["test"]) for t, r in rs.items()},
            "sepconv_test": len(s["test"]),
            "naive_intersection": len(naive),
            "contaminated_dropped": len(contaminated),
            "clean": len(clean),
        },
        "cross_split_contamination": cross,
        "coverage": {
            "dates": len({k[0] for k in clean}),
            "reference_times": len({(k[0], k[1]) for k in clean}),
            "patches_present": sorted(patches),
            "patches_missing": [p for p in range(1, 19) if p not in patches],
            "per_patch": {str(p): n for p, n in sorted(patches.items())},
        },
        "keys": sorted([list(k) for k in clean]),
    }


def format_report(blob: dict) -> str:
    c = blob["counts"]
    cov = blob["coverage"]
    lines = [
        "=" * 70,
        "Verification key set — RECONVECT vs SepConv-ens",
        "=" * 70,
    ]
    for name, n in c["reconvect_test"].items():
        lines.append(f"  RECONVECT test ({name:<4}) : {n:>6}")
    lines += [
        f"  SepConv   test        : {c['sepconv_test']:>6}",
        f"  naive intersection    : {c['naive_intersection']:>6}",
        f"  dropped as contaminated: {c['contaminated_dropped']:>5}",
        f"  CLEAN verification set: {c['clean']:>6}",
        "",
        "  Cross-split contamination between the windows:",
    ]
    for k, v in blob["cross_split_contamination"].items():
        flag = "  <-- why the subtraction exists" if v else ""
        lines.append(f"    {k:<38} {v:>5}{flag}")
    lines += [
        "",
        f"  dates                 : {cov['dates']}",
        f"  reference times       : {cov['reference_times']}",
        f"  patches               : {len(cov['patches_present'])}/18 "
        f"present, missing {cov['patches_missing']}",
    ]
    if c["clean"] == 0:
        lines.append("\n  ERROR: nothing survives — the two splits share no "
                     "uncontaminated test key.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the leakage-free verification key set shared by "
                    "RECONVECT and the SepConv-ens baseline.")
    parser.add_argument("--data_root", default=str(resolve_data_root()))
    parser.add_argument("--reconvect_tag", nargs="*", default=None,
                        help="Window tag(s) of the RECONVECT split(s), e.g. "
                             "`f34 w34` to freeze one set every model is "
                             "scored on. Omit for the unsuffixed split.")
    parser.add_argument("--sepconv_tag", required=True,
                        help="Window tag of the baseline split, e.g. "
                             "w44. Required: with several windows on "
                             "disk there is no safe default.")
    parser.add_argument("--write", action="store_true",
                        help="Freeze the key set to JSON. Do this BEFORE "
                             "the test data is scored, and read from the "
                             "frozen file afterwards.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    reconvect = args.reconvect_tag if args.reconvect_tag else None
    blob = build(data_root, reconvect_tag=reconvect,
                 sepconv_tag=args.sepconv_tag)
    print(format_report(blob))

    if args.write:
        out = Path(args.output) if args.output else (
            data_root / frozen_name(SOURCE, reconvect, args.sepconv_tag))
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2)
        print(f"\n  Frozen -> {out}  ({blob['counts']['clean']} keys)")
    else:
        print("\n  (report only — pass --write to freeze)")
    return 0 if blob["counts"]["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
