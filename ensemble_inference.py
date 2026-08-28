"""
ensemble_inference.py — route each patch to its assigned ensemble member
========================================================================
`build_patch_ensemble.py` decides, per patch, which seasonal member scores
best and writes that decision to a manifest. This module reads the
manifest back and answers the one question inference needs:

    for this target date, which model predicts patch N?

Resolution order
----------------
1. The manifest's per-patch assignment. This is the verified answer -
   the member that won patch N on the validation split.
2. For a patch no member scored (nothing convective ever happened there
   in the scoring split), the member whose season contains the target
   date. A seasonal default beats an arbitrary one.
3. Failing that - the target month belongs to no season, or that member
   is absent - the member that won the most patches overall.

Steps 2 and 3 are reported, not silent: a patch served by a fallback is
not backed by evidence and the caller should be able to say so.

Determinism
-----------
Nothing here re-scores anything. The manifest is the ensemble, so two
validation runs against the same manifest resolve identically. The
assignment changes only when `build_patch_ensemble.py` is re-run, which
is what `knowledge_cutoff` in the manifest marks.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from periods import DEFAULT_SEASONS, parse_date, season_for_date


class PatchEnsemble:
    """A manifest plus lazily-loaded member models."""

    def __init__(self, manifest: dict, model_loader=None):
        self.manifest = manifest
        self.assignment = {
            int(k): v for k, v in manifest.get("patch_assignment", {}).items()
        }
        self.members = manifest.get("members", {})
        self.seasons = manifest.get("seasons") or DEFAULT_SEASONS
        self._loader = model_loader
        self._cache: dict[str, object] = {}

    # -- construction ------------------------------------------------------

    @classmethod
    def from_path(cls, path: str | Path, model_loader=None) -> "PatchEnsemble":
        p = Path(path)
        if not p.is_file():
            raise SystemExit(
                f"Ensemble manifest not found: {p}\n"
                f"Build one with:\n"
                f"    python build_patch_ensemble.py --mode <mode>"
            )
        with open(p, encoding="utf-8") as fh:
            return cls(json.load(fh), model_loader=model_loader)

    # -- properties --------------------------------------------------------

    @property
    def knowledge_cutoff(self) -> str | None:
        return self.manifest.get("knowledge_cutoff")

    @property
    def member_labels(self) -> list[str]:
        return sorted(self.members)

    def _most_patches_won(self) -> str | None:
        if not self.assignment:
            return sorted(self.members)[0] if self.members else None
        tally: dict[str, int] = {}
        for a in self.assignment.values():
            tally[a["member"]] = tally.get(a["member"], 0) + 1
        # Sort by count then label so ties resolve reproducibly.
        return sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    # -- routing -----------------------------------------------------------

    def resolve(self, patch: int, target_date: date | str) -> tuple[str, str]:
        """Return (member_label, why) for one patch on one date."""
        assigned = self.assignment.get(int(patch))
        if assigned and assigned["member"] in self.members:
            return assigned["member"], "assigned"

        day = parse_date(target_date)
        season = season_for_date(day, self.seasons)
        if season:
            for label in self.members:
                blob = self.members[label]
                # A member label is `<year><season>`; match on the season
                # suffix so any year's warm member can serve a warm date.
                if label.endswith(season):
                    return label, f"season fallback ({season})"

        fallback = self._most_patches_won()
        if fallback is None:
            raise SystemExit("Manifest lists no members — nothing to route to.")
        return fallback, "global fallback"

    def routing_table(self, target_date: date | str,
                      patches: list[int]) -> dict[int, dict]:
        """Resolve a whole set of patches at once."""
        table: dict[int, dict] = {}
        for patch in patches:
            label, why = self.resolve(patch, target_date)
            table[patch] = {"member": label, "reason": why}
        return table

    def describe_routing(self, table: dict[int, dict]) -> str:
        by_member: dict[str, list[int]] = {}
        fallbacks: list[int] = []
        for patch, info in sorted(table.items()):
            by_member.setdefault(info["member"], []).append(patch)
            if info["reason"] != "assigned":
                fallbacks.append(patch)

        lines = [f"Ensemble routing (cutoff {self.knowledge_cutoff}):"]
        for label, patches in sorted(by_member.items()):
            lines.append(f"  {label:<14} patches {patches}")
        if fallbacks:
            lines.append(
                f"  NOTE: patches {fallbacks} had no verified assignment and "
                f"fell back — their member is a default, not a measured "
                f"winner."
            )
        return "\n".join(lines)

    # -- models ------------------------------------------------------------

    def model_path(self, label: str) -> str:
        blob = self.members.get(label)
        if blob is None:
            raise KeyError(f"Member {label!r} is not in the manifest.")
        return blob["model"]

    def get_model(self, label: str):
        """Load (and cache) a member's model via the injected loader.

        Models are cached because a single inference run touches at most
        one model per member but many patches per model.
        """
        if self._loader is None:
            raise RuntimeError(
                "PatchEnsemble was constructed without a model_loader; "
                "pass one to load member weights."
            )
        if label not in self._cache:
            self._cache[label] = self._loader(self.model_path(label))
        return self._cache[label]


def default_manifest_path(model_dir: str | Path, mode: str,
                          source: str) -> Path:
    from build_patch_ensemble import MANIFEST_NAME_FMT
    return Path(model_dir) / MANIFEST_NAME_FMT.format(mode=mode, source=source)
