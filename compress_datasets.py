"""
compress_datasets.py — archive, restore and reclaim built TFRecord datasets
===========================================================================
TFRecord shards compress extremely well: the label tensors are one-hot and
mostly zeros, and the normalised inputs repeat heavily. Measured on a real
shard from this project:

    -mx=1   11.5% of original    (~8.7x)   ~1s per 286 MB
    -mx=5    4.8% of original   (~20.8x)  ~19s per 286 MB

An ensemble is built one member at a time, each in its own directory, so
members are archived independently as they finish - a completed member
need not occupy disk while the next one trains.

Lifecycle
---------
    create_datasets --period 2025warm
      -> writes shards, spawns a detached ARCHIVE job, returns immediately

    train_models --period 2025warm
      -> restores the dataset if it is archive-only (blocking; training
         cannot start without the bytes)
      -> takes an in-use marker for the duration
      -> on completion spawns a detached RECLAIM job

Compression happens ONCE, right after creation. Training does not modify a
dataset, so the original archive stays valid across a restore: "reclaim"
therefore means *verify the archive, then delete the directory* - seconds,
not another compression pass. It only compresses when no archive exists.

Concurrency
-----------
Each dataset gets a lock in `<datasets_root>/_archive_jobs/`, so two
7-Zip processes can never write the same archive. Training takes an
`.inuse` marker on the same dataset; an archive job that finds one keeps
the archive and skips its delete step, leaving `pending-delete` for a
later reclaim to finish. That is what makes "build A, immediately train A"
safe.

Deletion safety
---------------
The source is removed only after ALL of:

  1. 7-Zip exits 0 on the `a` (add) command;
  2. `7z t` passes on the resulting archive;
  3. the archive's file listing matches the file count found on disk.

Any failure leaves both copies. A restore never deletes the archive. A
reclaim that finds a corrupt archive refuses to delete and says so.

Usage
-----
    python compress_datasets.py                          # list everything
    python compress_datasets.py --compress TAG           # archive + delete
    python compress_datasets.py --npy-stats DIR          # project the saving
    python compress_datasets.py --compress-npy DIR       # zstd the .npy tree
    python compress_datasets.py --restore-npy DIR        # and back again
    python compress_datasets.py --compress TAG --background
    python compress_datasets.py --restore  TAG           # extract back
    python compress_datasets.py --reclaim  TAG           # drop the copy on disk
    python compress_datasets.py --reclaim-all            # sweep leftovers
    python compress_datasets.py --jobs                   # background job state

Two different jobs
------------------
Datasets are archived WHOLE, with 7-Zip, because training streams them
start to finish and a restore is cheap relative to a training run.

The .npy stores cannot work that way: the pipeline opens them one frame at
a time, by name, from a dozen scripts. They are compressed IN PLACE
instead - `foo.npy` becomes `foo.npy.zst` - and every reader goes through
`load_array()`, so nothing needs restoring before a run. Solid bundling
was measured and rejected: over a full day of frames it buys 3% (11.3x ->
11.7x) and costs random access, while a frame-to-frame temporal delta is
actually worse (10.8x) because the inter-scan sensor noise compresses
less well than the frames do.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pipeline_config import resolve_data_root, resolve_datasets_root

# Usual install locations; `--sevenzip` overrides. 7-Zip is not on PATH in a
# default Windows install, so looking only at PATH would fail for most users.
SEVENZIP_CANDIDATES = (
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
)

ARCHIVE_SUFFIX = ".7z"
DEFAULT_LEVEL = 5

# Background bookkeeping lives here: one log, one lock and one status file
# per dataset.
JOBS_DIRNAME = "_archive_jobs"

# Simultaneous background jobs. With half the cores each, two already use
# the whole machine.
DEFAULT_MAX_CONCURRENT = 2


def default_workers() -> int:
    """Threads to hand 7-Zip.

    Half the logical cores: archiving runs alongside training, and LZMA2
    will otherwise take the whole machine and starve the input pipeline
    feeding the GPU.
    """
    return max(1, (os.cpu_count() or 4) // 2)


# =============================================================================
# 7-Zip
# =============================================================================

def find_sevenzip(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(f"7-Zip not found at {p}")
        return p
    for candidate in SEVENZIP_CANDIDATES:
        if Path(candidate).is_file():
            return Path(candidate)
    found = shutil.which("7z") or shutil.which("7za")
    if found:
        return Path(found)
    raise SystemExit(
        "7-Zip not found. Install it from https://www.7-zip.org/ or pass "
        "--sevenzip <path to 7z.exe>.\n"
        f"Looked in: {', '.join(SEVENZIP_CANDIDATES)} and PATH."
    )


def _run_7z(exe: Path, args: list[str],
            cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke 7-Zip without a shell.

    An argument list bypasses shell parsing entirely, which matters here:
    routing 7-Zip's switches through a shell is how arguments like
    `-mx=5` get mangled.

    `cwd` matters for `a`: 7-Zip stores each entry's path exactly as it was
    given on the command line, so archiving `our_data/datasets/<tag>` buries
    the whole tree under that prefix inside the archive. Running from the
    datasets root and naming just the tag keeps the archive rooted at the
    dataset itself.
    """
    return subprocess.run([str(exe), *args], capture_output=True, text=True,
                          cwd=None if cwd is None else str(cwd))


def archive_prefix(exe: Path, archive: Path) -> str:
    """The path every entry in `archive` sits under, as stored.

    Archives written before the working-directory fix carry the full
    `our_data/datasets/<tag>`; newer ones carry just `<tag>`. Restore has to
    place the tree so it lands at the same target either way, and the only
    authority on which layout an archive uses is the archive itself.

    Returns '' when 7-Zip cannot list it, which makes restore fall back to
    the modern layout.
    """
    result = _run_7z(exe, ["l", "-slt", "-ba", str(archive)])
    if result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        if line.startswith("Path = "):
            return line[len("Path = "):].strip().replace("\\", "/")
    return ""


# =============================================================================
# Sizes
# =============================================================================

def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def count_files(path: Path) -> int:
    return sum(len(files) for _root, _dirs, files in os.walk(path))


# =============================================================================
# Locks, markers, jobs
# =============================================================================

def jobs_dir(datasets_root: Path) -> Path:
    d = Path(datasets_root) / JOBS_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pid_alive(pid: int) -> bool:
    """Whether a PID is still running, so stale locks can be reclaimed."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        SYNCHRONIZE = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


class _Marker:
    """A PID-stamped file used as a lock or an in-use flag.

    Created atomically with O_EXCL so two processes cannot both believe
    they hold it. A marker whose PID is gone is treated as stale and
    reclaimed, so a crashed run does not block the dataset forever.
    """

    def __init__(self, path: Path, purpose: str):
        self.path = path
        self.purpose = purpose

    def held_by(self) -> int | None:
        """PID currently holding this marker, or None (clearing if stale)."""
        if not self.path.is_file():
            return None
        try:
            blob = json.loads(self.path.read_text())
            pid = int(blob.get("pid", -1))
        except (json.JSONDecodeError, ValueError, OSError):
            self.path.unlink(missing_ok=True)
            return None
        if _pid_alive(pid):
            return pid
        self.path.unlink(missing_ok=True)   # stale: owner is gone
        return None

    def acquire(self) -> bool:
        if self.held_by() is not None:
            return False
        try:
            fd = os.open(str(self.path),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({
                "pid": os.getpid(),
                "purpose": self.purpose,
                "started": datetime.now(timezone.utc)
                                   .strftime("%Y-%m-%dT%H:%M:%SZ"),
            }, fh)
        return True

    def release(self) -> None:
        self.path.unlink(missing_ok=True)

    def __enter__(self):
        self.acquired = self.acquire()
        return self

    def __exit__(self, *exc):
        if getattr(self, "acquired", False):
            self.release()
        return False


def lock_for(datasets_root: Path, run_tag: str) -> _Marker:
    return _Marker(jobs_dir(datasets_root) / f"{run_tag}.lock", "archive")


def inuse_for(datasets_root: Path, run_tag: str) -> _Marker:
    return _Marker(jobs_dir(datasets_root) / f"{run_tag}.inuse", "training")


def write_status(datasets_root: Path, run_tag: str, action: str,
                 outcome: str, detail: str = "") -> None:
    path = jobs_dir(datasets_root) / f"{run_tag}.status"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "action": action,
            "outcome": outcome,
            "detail": detail,
            "finished": datetime.now(timezone.utc)
                                .strftime("%Y-%m-%dT%H:%M:%SZ"),
        }, fh, indent=2)


def running_jobs(datasets_root: Path) -> list[str]:
    """Dataset tags with a live archive lock."""
    out = []
    for lock in jobs_dir(datasets_root).glob("*.lock"):
        tag = lock.stem
        if lock_for(datasets_root, tag).held_by() is not None:
            out.append(tag)
    return sorted(out)


def spawn_job(action: str, run_tag: str, datasets_root: Path,
              level: int, workers: int, sevenzip: str | None = None,
              max_concurrent: int = DEFAULT_MAX_CONCURRENT) -> bool:
    """Launch a detached compress/reclaim job for one dataset.

    Detached on purpose: the caller (dataset creation, or training) must
    return immediately so the next member can start. The child outlives
    the parent and writes to its own log.
    """
    datasets_root = Path(datasets_root)
    if lock_for(datasets_root, run_tag).held_by() is not None:
        print(f"  [job] {run_tag} already has a job running — not starting "
              f"a second one.")
        return False

    live = running_jobs(datasets_root)
    if len(live) >= max_concurrent:
        print(f"  [job] {len(live)} job(s) already running "
              f"({', '.join(live)}); skipping background {action} for "
              f"{run_tag}.")
        print(f"        Run it later with: python compress_datasets.py "
              f"--{action} {run_tag}")
        return False

    log_path = jobs_dir(datasets_root) / f"{run_tag}.log"
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        f"--{action}", run_tag,
        "--datasets_root", str(datasets_root),
        "--level", str(level),
        "--workers", str(workers),
    ]
    if sevenzip:
        cmd += ["--sevenzip", sevenzip]

    flags = 0
    kwargs: dict = {}
    if os.name == "nt":
        flags = (subprocess.DETACHED_PROCESS
                 | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        kwargs["start_new_session"] = True

    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, creationflags=flags,
                            close_fds=True, **kwargs)
    print(f"  [job] {action} {run_tag} started in the background "
          f"(pid {proc.pid}, {workers} threads)")
    print(f"        log: {log_path}")
    return True


# =============================================================================
# Discovery
# =============================================================================

class DatasetInfo:
    """One dataset directory (or its archive) plus what is known about it."""

    def __init__(self, run_tag: str, datasets_root: Path):
        self.run_tag = run_tag
        self.datasets_root = Path(datasets_root)
        self.path = self.datasets_root / run_tag
        self.archive = self.datasets_root / f"{run_tag}{ARCHIVE_SUFFIX}"
        self.meta: dict = {}
        self.splits: list[str] = []
        self.registered: bool | None = None
        self._size: int | None = None

        if self.path.is_dir():
            for split in ("train", "validation", "test"):
                split_dir = self.path / split
                if split_dir.is_dir():
                    self.splits.append(split)
                    meta_path = split_dir / "metadata.json"
                    if not self.meta and meta_path.is_file():
                        try:
                            self.meta = json.loads(meta_path.read_text())
                        except json.JSONDecodeError:
                            pass

    @property
    def on_disk(self) -> bool:
        return self.path.is_dir()

    @property
    def archived(self) -> bool:
        return self.archive.is_file()

    @property
    def in_use_by(self) -> int | None:
        return inuse_for(self.datasets_root, self.run_tag).held_by()

    @property
    def job_pid(self) -> int | None:
        return lock_for(self.datasets_root, self.run_tag).held_by()

    @property
    def state(self) -> str:
        if self.job_pid:
            return "job running"
        if self.on_disk and self.archived:
            return "both"
        if self.archived:
            return "archived"
        if self.on_disk:
            return "on disk"
        return "missing"

    @property
    def reclaimable(self) -> bool:
        """Archived, still on disk, and nobody is using it."""
        return (self.on_disk and self.archived
                and self.in_use_by is None and self.job_pid is None)

    @property
    def size(self) -> int:
        if self._size is None:
            self._size = dir_size(self.path) if self.on_disk else 0
        return self._size

    @property
    def archive_size(self) -> int:
        return self.archive.stat().st_size if self.archived else 0

    @property
    def period(self) -> str | None:
        blob = self.meta.get("period")
        return blob.get("label") if isinstance(blob, dict) else None


def discover(datasets_root: Path, data_root: Path) -> list[DatasetInfo]:
    """Every dataset under datasets_root, whether on disk or archived."""
    datasets_root = Path(datasets_root)
    if not datasets_root.is_dir():
        return []

    tags: set[str] = set()
    for child in datasets_root.iterdir():
        if child.is_dir() and child.name != JOBS_DIRNAME:
            tags.add(child.name)
        elif child.suffix == ARCHIVE_SUFFIX:
            tags.add(child.stem)

    infos = [DatasetInfo(tag, datasets_root) for tag in sorted(tags)]

    # Cross-reference the registered ensemble plan: a dataset whose period
    # is a registered member belongs to the ensemble; one that is not is
    # either the unscoped whole-archive dataset or a leftover.
    try:
        from ensemble_plan import load_last_state
        state = load_last_state(data_root)
    except Exception:                                    # noqa: BLE001
        state = None
    if state:
        labels = {m["label"] for m in state.get("members", [])}
        for info in infos:
            period = info.period or _period_from_tag(info.run_tag, labels)
            info.registered = (period in labels) if period else False
    return infos


def _period_from_tag(run_tag: str, labels: set[str]) -> str | None:
    """Recover a period label from a run tag when metadata is unreadable.

    An archived dataset has no readable metadata.json, so the tag suffix
    is the only clue left. Matched against known labels rather than
    parsed, since mode names contain underscores too.
    """
    for label in labels:
        if run_tag.endswith(f"_{label}"):
            return label
    return None


# =============================================================================
# Listing
# =============================================================================

def print_listing(infos: list[DatasetInfo],
                  datasets_root: Path | None = None) -> None:
    if not infos:
        # Name the root actually searched: with --datasets_root or
        # $COALITION4_DATASETS_ROOT it is often not our_data/datasets.
        where = datasets_root if datasets_root else "our_data/datasets/"
        print(f"No datasets found under {where}.")
        return

    width = max(len(i.run_tag) for i in infos)
    print("=" * 78)
    print("Available datasets")
    print("=" * 78)
    print(f"  {'dataset':<{width}}  {'state':<12} {'on disk':>10} "
          f"{'archive':>10}  {'period':<10} splits")
    print("  " + "-" * (width + 55))

    total_disk = total_archive = 0
    for info in infos:
        total_disk += info.size
        total_archive += info.archive_size
        flags = ""
        if info.registered:
            flags += " *"
        if info.in_use_by:
            flags += f" [in use pid {info.in_use_by}]"
        print(
            f"  {info.run_tag:<{width}}  {info.state:<12} "
            f"{human(info.size) if info.on_disk else '-':>10} "
            f"{human(info.archive_size) if info.archived else '-':>10}  "
            f"{(info.period or '-'):<10} "
            f"{','.join(info.splits) if info.splits else '-'}{flags}"
        )

    print("  " + "-" * (width + 55))
    print(f"  {'TOTAL':<{width}}  {'':<12} {human(total_disk):>10} "
          f"{human(total_archive):>10}")
    if any(i.registered for i in infos):
        print("\n  * registered ensemble member")

    reclaimable = [i for i in infos if i.reclaimable]
    if reclaimable:
        freeable = sum(i.size for i in reclaimable)
        print(f"\n  {len(reclaimable)} dataset(s) are archived AND still on "
              f"disk — {human(freeable)} reclaimable:")
        print(f"      python compress_datasets.py --reclaim-all")

    unarchived = [i for i in infos if i.on_disk and not i.archived]
    if unarchived:
        raw = sum(i.size for i in unarchived)
        print(f"\n  {len(unarchived)} dataset(s) not yet archived "
              f"({human(raw)}); compressing would free about "
              f"{human(raw * 0.95)} (measured ~4.8% of original at -mx=5):")
        for i in unarchived:
            print(f"      python compress_datasets.py --compress {i.run_tag}")


def print_jobs(datasets_root: Path) -> None:
    d = jobs_dir(datasets_root)
    rows: list[tuple[str, str, str]] = []
    for lock in sorted(d.glob("*.lock")):
        tag = lock.stem
        pid = lock_for(datasets_root, tag).held_by()
        if pid:
            rows.append((tag, f"RUNNING (pid {pid})", ""))
    for status in sorted(d.glob("*.status")):
        tag = status.stem
        if any(r[0] == tag for r in rows):
            continue
        try:
            blob = json.loads(status.read_text())
        except json.JSONDecodeError:
            continue
        rows.append((tag, f"{blob['outcome']} ({blob['action']})",
                     blob.get("detail", "")))

    if not rows:
        print("No archive jobs recorded.")
        return
    width = max(len(r[0]) for r in rows)
    print("Archive jobs:")
    for tag, state, detail in rows:
        print(f"  {tag:<{width}}  {state}" + (f"  — {detail}" if detail else ""))
    print(f"\nLogs: {d}")


# =============================================================================
# Actions
# =============================================================================

def archive_is_valid(info: DatasetInfo, exe: Path) -> bool:
    if not info.archived:
        return False
    return _run_7z(exe, ["t", str(info.archive)]).returncode == 0


def compress(info: DatasetInfo, exe: Path, level: int, workers: int,
             keep: bool, dry_run: bool) -> bool:
    """Archive one dataset, verify it, then optionally delete the source."""
    print("=" * 78)
    print(f"Compressing {info.run_tag}")
    print("=" * 78)

    if not info.on_disk:
        print(f"  SKIP: nothing on disk at {info.path}")
        return False
    if info.archived:
        print(f"  Archive already exists at {info.archive} — "
              f"switching to reclaim.")
        return reclaim(info, exe, level, workers, dry_run)

    n_files = count_files(info.path)
    print(f"  Source   : {info.path}")
    print(f"  Size     : {human(info.size)} across {n_files} file(s)")
    print(f"  Archive  : {info.archive}")
    print(f"  Level    : -mx={level}   Threads: {workers}")
    print(f"  Delete   : {'no (--keep)' if keep else 'yes, after verification'}")

    if dry_run:
        print("  --dry-run: nothing written.")
        return False

    lock = lock_for(info.datasets_root, info.run_tag)
    if not lock.acquire():
        print(f"  SKIP: another job holds the lock for {info.run_tag}.")
        return False

    try:
        started = time.time()
        print("\n  [1/3] compressing ...", flush=True)
        # Run from the datasets root and name just the tag, so entries
        # are stored as <tag>/... rather than our_data/datasets/<tag>/...
        result = _run_7z(exe, [
            "a", "-t7z", f"-mx={level}", f"-mmt={workers}",
            "-bso0", "-bsp2", str(info.archive.resolve()), info.run_tag,
        ], cwd=info.datasets_root)
        if result.returncode != 0:
            print(f"  ERROR: 7-Zip exited {result.returncode}")
            print(result.stderr.strip()[:2000])
            info.archive.unlink(missing_ok=True)   # no half-written archive
            write_status(info.datasets_root, info.run_tag, "compress",
                         "failed", f"7z exit {result.returncode}")
            return False

        elapsed = time.time() - started
        archive_size = info.archive.stat().st_size
        ratio = archive_size / info.size * 100 if info.size else 0
        print(f"        {human(info.size)} -> {human(archive_size)} "
              f"({ratio:.1f}%) in {elapsed / 60:.1f} min")

        # The source is about to be deleted, so the archive is tested
        # before anything is removed, not trusted on exit code alone.
        print("  [2/3] verifying archive ...", flush=True)
        if not archive_is_valid(info, exe):
            print("  ERROR: archive failed verification — source intact.")
            write_status(info.datasets_root, info.run_tag, "compress",
                         "failed", "verification failed")
            return False

        listing = _run_7z(exe, ["l", "-slt", str(info.archive)])
        archived_files = listing.stdout.count("\nPath = ") - 1
        if archived_files < n_files:
            print(f"  ERROR: archive holds {archived_files} file(s) but "
                  f"{n_files} were on disk — source intact.")
            write_status(info.datasets_root, info.run_tag, "compress",
                         "failed", "file count mismatch")
            return False
        print(f"        OK — {archived_files} file(s) verified")

        if keep:
            print("  [3/3] keeping source (--keep).")
            write_status(info.datasets_root, info.run_tag, "compress",
                         "ok", "source kept")
            return True

        return _delete_source(info, step="[3/3]", action="compress")
    finally:
        lock.release()


def _delete_source(info: DatasetInfo, step: str, action: str) -> bool:
    """Remove the uncompressed copy, unless something is using it."""
    holder = info.in_use_by
    if holder is not None:
        print(f"  {step} dataset is IN USE by pid {holder} — keeping the "
              f"directory.")
        print(f"        The archive is valid; reclaim it later with:")
        print(f"        python compress_datasets.py --reclaim {info.run_tag}")
        write_status(info.datasets_root, info.run_tag, action,
                     "pending-delete", f"in use by pid {holder}")
        return True

    print(f"  {step} removing uncompressed source ...", flush=True)
    size = info.size
    try:
        shutil.rmtree(info.path)
    except OSError as exc:
        print(f"  ERROR: could not remove {info.path}: {exc}\n"
              f"         The archive is valid; delete the directory by hand.")
        write_status(info.datasets_root, info.run_tag, action,
                     "failed", f"rmtree: {exc}")
        return False

    print(f"        freed {human(size)}")
    write_status(info.datasets_root, info.run_tag, action, "ok",
                 f"freed {human(size)}")
    print(f"\n  Restore with: python compress_datasets.py "
          f"--restore {info.run_tag}")
    return True


def reclaim(info: DatasetInfo, exe: Path, level: int, workers: int,
            dry_run: bool) -> bool:
    """Make a dataset archive-only.

    Reuses a valid existing archive and simply deletes the directory -
    the case after training, which does not modify the dataset. Falls
    back to a full compression when no archive exists.
    """
    print("=" * 78)
    print(f"Reclaiming {info.run_tag}")
    print("=" * 78)

    if not info.on_disk:
        print("  Nothing on disk — already reclaimed.")
        return True
    if not info.archived:
        print("  No archive yet — compressing first.")
        return compress(info, exe, level, workers, keep=False,
                        dry_run=dry_run)

    holder = info.in_use_by
    if holder is not None:
        print(f"  SKIP: in use by pid {holder}.")
        return False

    print(f"  Archive  : {info.archive} ({human(info.archive_size)})")
    print(f"  On disk  : {human(info.size)}")
    if dry_run:
        print("  --dry-run: nothing removed.")
        return False

    lock = lock_for(info.datasets_root, info.run_tag)
    if not lock.acquire():
        print(f"  SKIP: another job holds the lock for {info.run_tag}.")
        return False
    try:
        print("  [1/2] verifying existing archive ...", flush=True)
        if not archive_is_valid(info, exe):
            print("  ERROR: existing archive FAILED verification. Refusing "
                  "to delete the only good copy.")
            print(f"         Delete {info.archive} and re-run --compress "
                  f"to rebuild it.")
            write_status(info.datasets_root, info.run_tag, "reclaim",
                         "failed", "archive verification failed")
            return False
        print("        OK")
        return _delete_source(info, step="[2/2]", action="reclaim")
    finally:
        lock.release()


def restore(info: DatasetInfo, exe: Path, dry_run: bool) -> bool:
    """Extract an archived dataset back onto disk."""
    print("=" * 78)
    print(f"Restoring {info.run_tag}")
    print("=" * 78)

    if not info.archived:
        print(f"  ERROR: no archive at {info.archive}")
        return False
    if info.on_disk:
        print(f"  Already on disk at {info.path} — nothing to do.")
        return True

    print(f"  Archive : {info.archive} ({human(info.archive_size)})")
    print(f"  Target  : {info.path}")
    if dry_run:
        print("  --dry-run: nothing extracted.")
        return False

    # Extract so the stored prefix lands exactly on info.path: step back
    # one directory per prefix component. A modern archive stores '<tag>'
    # (one component -> datasets_root); a legacy one stores
    # 'our_data/datasets/<tag>' (three -> the project root).
    prefix = archive_prefix(exe, info.archive) or info.run_tag
    out_dir = info.path.resolve()
    for _ in Path(prefix).parts:
        out_dir = out_dir.parent
    if prefix != info.run_tag:
        print(f"  Layout  : archive stores '{prefix}' -> extracting to "
              f"{out_dir}")

    started = time.time()
    result = _run_7z(exe, [
        "x", str(info.archive.resolve()), f"-o{out_dir}", "-y",
        "-bso0", "-bsp2",
    ])
    if result.returncode != 0:
        print(f"  ERROR: 7-Zip exited {result.returncode}")
        print(result.stderr.strip()[:2000])
        write_status(info.datasets_root, info.run_tag, "restore",
                     "failed", f"7z exit {result.returncode}")
        return False

    print(f"  Restored -> {info.path} in {(time.time() - started) / 60:.1f} min")
    print("  The archive was kept, so reclaiming later is just a delete.")
    write_status(info.datasets_root, info.run_tag, "restore", "ok", "")
    return True


def ensure_available(run_tag: str, datasets_root, exe: Path | None = None,
                     auto_restore: bool = True) -> Path | None:
    """Make a dataset usable, extracting it first if it is archive-only.

    The entry point training uses. Blocking on purpose: training cannot
    proceed without the bytes, so there is nothing useful to do in
    parallel. Returns the dataset directory, or None when the dataset is
    neither on disk nor archived.
    """
    datasets_root = Path(datasets_root)
    info = DatasetInfo(run_tag, datasets_root)

    if info.on_disk:
        return info.path
    if not info.archived:
        return None
    if not auto_restore:
        return None

    exe = exe or find_sevenzip(None)
    print(f"\nDataset {run_tag} is archived — extracting "
          f"{human(info.archive_size)} before training ...")
    if not restore(info, exe, dry_run=False):
        raise SystemExit(f"Failed to restore {run_tag} from {info.archive}")
    return info.path


# =============================================================================
# CLI
# =============================================================================

# =============================================================================
# .npy compression (zstd)
# =============================================================================
# TFRecord datasets are archived whole with 7-Zip above. The .npy that feed
# them cannot be: they are read one file at a time, by name, from a dozen
# scripts, so they have to stay individually addressable. They are instead
# compressed in place - one frame per file - and read back through
# `load_array` below, which resolves a logical `foo.npy` to whichever of
# `foo.npy` / `foo.npy.zst` is actually on disk.
#
# What gets stored is the ENTIRE .npy file, header included, not the array
# buffer. A restore is then byte-identical by construction, and the verify
# step before any delete is a plain byte comparison - dtype, shape, byte
# order and fill values cannot drift.
#
# Level 10 on the float32 exactly as stored: no dtype change, no requant,
# no byte shuffle, so the numerical base is untouched. Measured across
# every folder this tool targets (1.10 M files, 5292 GB) it reaches 8.5x
# overall: 5.7x on the raw MTG store, 8.8x on reprojected MTG, 37.6x on
# OPERA, >7000x on lightning, whose grids are nearly empty. Higher levels
# cost far more than they return - 19 reaches 10.7x at 3 MB/s against 10's
# ~35 MB/s, which is days of extra CPU over a store this size.

NPY_EXT = ".npy"
ZST_EXT = ".zst"
NPY_ZST_EXT = NPY_EXT + ZST_EXT
DEFAULT_ZSTD_LEVEL = 10

# Below this gain the plain file is kept: the read-side indirection is not
# worth it, and a file that will not compress is usually already dense.
MIN_ZSTD_GAIN = 1.05

# The two grid files are read by nearly every script in the repo, including
# ad-hoc notebooks that will never go through `load_array`. They are ~9 MB
# in total, so leaving them plain costs nothing and removes a whole class
# of "why is this one path broken" failure.
NPY_NEVER_COMPRESS = ("romania_grid_lats.npy", "romania_grid_lons.npy")

_DATE_IN_NAME = __import__("re").compile(r"(\d{4}-\d{2}-\d{2}|\d{8})")


def _zstd():
    """Import zstandard with an actionable message if it is missing."""
    try:
        import zstandard
    except ImportError:
        raise SystemExit(
            "The `zstandard` package is required for .npy compression.\n"
            "    pip install zstandard"
        )
    return zstandard


# ---------------------------------------------------------------- read side

def array_path(path) -> Path:
    """Resolve a logical array path to the file that actually exists.

    Accepts either `foo.npy` or `foo.npy.zst` and returns whichever is on
    disk, preferring the uncompressed copy. Raises FileNotFoundError if
    neither is, so a genuinely missing frame still fails loudly.
    """
    p = Path(path)
    plain = Path(str(p)[:-len(ZST_EXT)]) if p.name.endswith(NPY_ZST_EXT) else p
    packed = plain.with_name(plain.name + ZST_EXT)
    if plain.is_file():
        return plain
    if packed.is_file():
        return packed
    raise FileNotFoundError(f"neither {plain} nor {packed} exists")


def array_exists(path) -> bool:
    """True if the frame is on disk in either form."""
    try:
        array_path(path)
        return True
    except FileNotFoundError:
        return False


def load_array(path, mmap_mode=None):
    """np.load that transparently reads `.npy.zst`.

    Drop-in for `np.load(path)` at every call site in the pipeline: pass
    the same logical `.npy` path whether or not the frame has been
    compressed.
    """
    import numpy as np

    resolved = array_path(path)
    if resolved.suffix != ZST_EXT:
        return np.load(resolved, mmap_mode=mmap_mode, allow_pickle=False)
    if mmap_mode is not None:
        raise ValueError(
            f"{resolved} is zstd-compressed and cannot be memory-mapped. "
            f"Restore it first: python compress_datasets.py --restore-npy "
            f"{resolved.parent}"
        )
    import io
    with open(resolved, "rb") as fh:
        raw = _zstd().ZstdDecompressor().decompress(fh.read())
    return np.load(io.BytesIO(raw), allow_pickle=False)


def save_array(path, arr, compress: bool = False, level: int = DEFAULT_ZSTD_LEVEL):
    """np.save that can write the compressed form directly.

    Writers stay on the plain form by default; compression is a separate,
    verifiable pass rather than something that happens silently mid-run.
    """
    import io

    import numpy as np

    p = Path(path)
    if p.name.endswith(NPY_ZST_EXT):
        p = Path(str(p)[:-len(ZST_EXT)])
    if not compress:
        np.save(p, arr, allow_pickle=False)
        return p
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    out = p.with_name(p.name + ZST_EXT)
    with open(out, "wb") as fh:
        fh.write(_zstd().ZstdCompressor(level=level).compress(buf.getvalue()))
    return out


def list_arrays(directory) -> list[str]:
    """Sorted LOGICAL array names in one directory.

    Always returns `*.npy` names with any `.zst` stripped, so the callers
    that parse the filename (`base = name[:-len('.npy')]`, timestamp
    slicing, channel suffix matching) keep working untouched whether the
    directory is compressed or not. Deduplicates when both forms exist.
    """
    d = Path(directory)
    if not d.is_dir():
        return []
    names = set()
    for entry in os.scandir(d):
        if not entry.is_file():
            continue
        n = entry.name
        if n.endswith(NPY_ZST_EXT):
            names.add(n[:-len(ZST_EXT)])
        elif n.endswith(NPY_EXT):
            names.add(n)
    return sorted(names)


def find_arrays(root, compressed=None):
    """Walk `root` yielding real array paths.

    compressed=None  -> both forms
    compressed=False -> only plain .npy   (what --compress-npy consumes)
    compressed=True  -> only .npy.zst     (what --restore-npy consumes)
    """
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.endswith(NPY_ZST_EXT):
                if compressed is not False:
                    yield Path(dirpath) / name
            elif name.endswith(NPY_EXT):
                if compressed is not True:
                    yield Path(dirpath) / name


# ------------------------------------------------------------- write side

def _date_key(path: Path, root: Path) -> str:
    """Group files by the dated folder they sit under.

    Both layouts in this project carry the date in a directory name -
    `nc4_2025-01-01-Romania_ir_105/` for the product and reprojected
    stores, `2025-01-03/` for the patches - so the same walk batches
    either one without being told which it is looking at.
    """
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    for part in reversed(parts[:-1]):
        m = _DATE_IN_NAME.search(part)
        if m:
            d = m.group(1)
            return d if "-" in d else f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return "(undated)"


def _compress_one(task):
    """Compress one .npy, verify it round-trips, then drop the original.

    Runs in a worker process. The original is unlinked only after the
    freshly written file has been read back and compared byte for byte,
    so an interrupted or corrupt write can never cost data.
    """
    src_s, level, keep = task
    src = Path(src_s)
    dst = src.with_name(src.name + ZST_EXT)
    try:
        if dst.is_file():
            return ("exists", src_s, 0, 0, "")
        original = src.read_bytes()
        packed = _zstd().ZstdCompressor(level=level).compress(original)
        if len(original) / max(len(packed), 1) < MIN_ZSTD_GAIN:
            return ("nogain", src_s, len(original), len(original), "")

        tmp = dst.with_name(dst.name + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(packed)
        # Read back from disk, not from the buffer in memory: this is the
        # step that catches a short write or a bad sector.
        with open(tmp, "rb") as fh:
            if _zstd().ZstdDecompressor().decompress(fh.read()) != original:
                tmp.unlink(missing_ok=True)
                return ("error", src_s, 0, 0, "round-trip mismatch")
        os.replace(tmp, dst)
        if not keep:
            src.unlink()
        return ("done", src_s, len(original), len(packed), "")
    except Exception as exc:
        return ("error", src_s, 0, 0, f"{type(exc).__name__}: {exc}")


def _restore_one(task):
    """Decompress one .npy.zst back to a plain .npy and drop the archive."""
    src_s, keep = task
    src = Path(src_s)
    dst = Path(str(src)[:-len(ZST_EXT)])
    try:
        if dst.is_file():
            return ("exists", src_s, 0, 0, "")
        packed_size = src.stat().st_size          # before it is unlinked
        with open(src, "rb") as fh:
            raw = _zstd().ZstdDecompressor().decompress(fh.read())
        tmp = dst.with_name(dst.name + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(raw)
        if tmp.stat().st_size != len(raw):
            tmp.unlink(missing_ok=True)
            return ("error", src_s, 0, 0, "short write")
        os.replace(tmp, dst)
        if not keep:
            src.unlink()
        return ("done", src_s, packed_size, len(raw), "")
    except Exception as exc:
        return ("error", src_s, 0, 0, f"{type(exc).__name__}: {exc}")


def _verify_one(src_s):
    """Decompress and discard: proves the archive is readable."""
    try:
        with open(src_s, "rb") as fh:
            raw = _zstd().ZstdDecompressor().decompress(fh.read())
        import io

        import numpy as np
        np.load(io.BytesIO(raw), allow_pickle=False)
        return ("done", src_s, os.path.getsize(src_s), len(raw), "")
    except Exception as exc:
        return ("error", src_s, 0, 0, f"{type(exc).__name__}: {exc}")


# Windows caps a ProcessPoolExecutor at 61 workers, and each worker holds a
# whole frame in memory anyway, so there is nothing to gain past that.
MAX_NPY_WORKERS = 61


def npy_default_workers() -> int:
    """Half the cores: this runs for hours, often alongside a training job."""
    return max(1, min((os.cpu_count() or 4) // 2, MAX_NPY_WORKERS))


def run_npy_pass(roots: list[Path], action: str, level: int, workers: int,
                 keep: bool, dry_run: bool, limit: int | None = None) -> int:
    """Compress / restore / verify every array under `roots`, by date.

    Returns the number of failures. Progress is reported one line per
    dated folder, because that is the unit the pipeline itself is
    organised in and the unit a resumed run picks up from.
    """
    from concurrent.futures import ProcessPoolExecutor

    workers = max(1, min(workers, MAX_NPY_WORKERS))
    want_compressed = {"compress": False, "restore": True,
                       "verify": True}[action]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"ERROR: no such path: {root}")
            return 1
        files.extend(find_arrays(root, compressed=want_compressed))
    files = [f for f in files if f.name not in NPY_NEVER_COMPRESS
             and not f.name.endswith(".tmp")]
    if limit:
        files = files[:limit]

    if not files:
        print(f"Nothing to {action}: no matching arrays under "
              f"{', '.join(str(r) for r in roots)}")
        return 0

    root0 = roots[0]
    groups: dict[str, list[Path]] = {}
    for f in files:
        groups.setdefault(_date_key(f, root0), []).append(f)

    total_bytes = sum(f.stat().st_size for f in files)
    print(f"Action     : {action}")
    print(f"Roots      : {', '.join(str(r) for r in roots)}")
    print(f"Arrays     : {len(files):,} across {len(groups):,} dated folder(s)")
    print(f"On disk    : {human(total_bytes)}")
    if action == "compress":
        print(f"Level      : zstd-{level}   (float32 kept exactly as stored)")
    print(f"Workers    : {workers}")
    if action == "verify":
        print("Source     : untouched (read-only pass)")
    elif keep:
        print("Source     : KEPT alongside the output")
    else:
        print(f"Source     : {'.npy' if action == 'compress' else '.npy.zst'}"
              f" deleted, but only once the round-trip is verified")
    if dry_run:
        print("\n[dry run] nothing will be written.")
        for date in sorted(groups)[:10]:
            g = groups[date]
            print(f"  {date}  {len(g):5,} file(s)  "
                  f"{human(sum(p.stat().st_size for p in g))}")
        if len(groups) > 10:
            print(f"  ... and {len(groups)-10:,} more dated folder(s)")
        return 0
    print()

    src_total = dst_total = 0
    counts = {"done": 0, "exists": 0, "nogain": 0, "error": 0}
    errors: list[str] = []
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, date in enumerate(sorted(groups), 1):
            batch = groups[date]
            if action == "compress":
                tasks = [(str(p), level, keep) for p in batch]
                results = pool.map(_compress_one, tasks, chunksize=8)
            elif action == "restore":
                tasks = [(str(p), keep) for p in batch]
                results = pool.map(_restore_one, tasks, chunksize=8)
            else:
                results = pool.map(_verify_one, [str(p) for p in batch],
                                   chunksize=8)

            g_src = g_dst = 0
            for status, path_s, s_bytes, d_bytes, msg in results:
                counts[status] = counts.get(status, 0) + 1
                g_src += s_bytes
                g_dst += d_bytes
                if status == "error":
                    errors.append(f"{path_s}: {msg}")
            src_total += g_src
            dst_total += g_dst

            # Always quote the ratio the same way round - packed against
            # plain - so compress and restore lines read alike instead of
            # one of them showing a confusing "0.1x".
            plain, packed = ((g_src, g_dst) if action == "compress"
                             else (g_dst, g_src))
            ratio = plain / packed if packed else 0.0
            elapsed = time.perf_counter() - t0
            rate = max(g_src, g_dst) and (
                (src_total if action == "compress" else dst_total)
                / 1e6 / elapsed if elapsed else 0)
            print(f"  [{i:>5}/{len(groups)}] {date}  {len(batch):5,} file(s)  "
                  f"{human(g_src):>9} -> {human(g_dst):>9}  "
                  f"{ratio:5.1f}x   {rate:5.0f} MB/s", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n{'-'*66}")
    print(f"  processed : {counts['done']:,}")
    if counts["exists"]:
        print(f"  skipped   : {counts['exists']:,} (target already present)")
    if counts["nogain"]:
        print(f"  left plain: {counts['nogain']:,} "
              f"(gain below {MIN_ZSTD_GAIN:.2f}x)")
    if src_total and dst_total:
        plain, packed = ((src_total, dst_total) if action == "compress"
                         else (dst_total, src_total))
        verb = {"compress": "reclaimed", "restore": "given back to disk",
                "verify": "would be given back"}[action]
        print(f"  {human(src_total)} -> {human(dst_total)}  "
              f"({plain/packed:.1f}x, {human(plain - packed)} {verb})")
    print(f"  elapsed   : {elapsed/60:.1f} min")
    if errors:
        print(f"\n  FAILURES  : {len(errors)}")
        for e in errors[:20]:
            print(f"    {e}")
        if len(errors) > 20:
            print(f"    ... and {len(errors)-20} more")
        print("  Sources for failed files were NOT deleted.")
    return len(errors)


def npy_stats(roots: list[Path], level: int, sample: int = 40) -> int:
    """Project the saving without writing anything.

    Samples `sample` arrays per root and scales the measured ratio by the
    real byte count on disk, which is what makes the estimate worth
    reading before committing hours of CPU.
    """
    import random

    import numpy as np

    cctx = _zstd().ZstdCompressor(level=level)
    dctx = _zstd().ZstdDecompressor()
    random.seed(7)

    print(f"{'root':44} {'files':>9} {'on disk':>10} {'ratio':>7} "
          f"{'after':>10}")
    grand_raw = grand_after = 0
    for root in roots:
        files = [f for f in find_arrays(root, compressed=False)
                 if f.name not in NPY_NEVER_COMPRESS]
        if not files:
            print(f"{str(root)[-44:]:44} {'-- no plain .npy':>9}")
            continue
        on_disk = sum(f.stat().st_size for f in files)
        picks = random.sample(files, min(sample, len(files)))
        raw = comp = 0
        for f in picks:
            b = f.read_bytes()
            c = cctx.compress(b)
            if dctx.decompress(c) != b:
                print(f"  ROUND-TRIP FAILED on {f}")
                return 1
            raw += len(b)
            comp += len(c)
        ratio = raw / comp
        after = on_disk / ratio
        grand_raw += on_disk
        grand_after += after
        print(f"{str(root)[-44:]:44} {len(files):9,} {human(on_disk):>10} "
              f"{ratio:6.1f}x {human(after):>10}")
    if grand_after:
        print(f"\n{'TOTAL':44} {'':9} {human(grand_raw):>10} "
              f"{grand_raw/grand_after:6.1f}x {human(grand_after):>10}")
        print(f"reclaimed: {human(grand_raw - grand_after)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List, compress, restore and reclaim TFRecord datasets, "
                    "and compress the .npy stores that feed them."
    )
    parser.add_argument("--data_root", default=None, metavar="PATH",
                        help="Root holding patches/, split CSVs and "
                             "statistics (default: the our_data/ beside "
                             "this script, or $COALITION4_DATA_ROOT).")
    parser.add_argument("--datasets_root", default=None, metavar="PATH",
                        help="Root holding the built TFRecord datasets "
                             "(default: <data_root>/datasets, or "
                             "$COALITION4_DATASETS_ROOT).")
    parser.add_argument("--compress", nargs="+", metavar="DATASET",
                        help="Archive, verify, then delete the uncompressed "
                             "shards.")
    parser.add_argument("--restore", nargs="+", metavar="DATASET",
                        help="Extract dataset(s) back onto disk. The archive "
                             "is kept.")
    parser.add_argument("--reclaim", nargs="+", metavar="DATASET",
                        help="Drop the on-disk copy of a dataset that is "
                             "already archived (verifying the archive "
                             "first). Compresses if no archive exists.")
    parser.add_argument("--reclaim-all", action="store_true",
                        help="Sweep every dataset that is archived, still on "
                             "disk and not in use. The cleanup for leftovers "
                             "from an interrupted run.")
    parser.add_argument("--jobs", action="store_true",
                        help="Show background archive job state and exit.")
    parser.add_argument("--background", action="store_true",
                        help="Run the requested --compress / --reclaim as a "
                             "detached job and return immediately.")
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL,
                        choices=[0, 1, 3, 5, 7, 9],
                        help=f"7-Zip -mx level (default: {DEFAULT_LEVEL}). "
                             f"Measured on this project's shards: -mx=1 "
                             f"reaches ~11.5%% of original and is ~19x "
                             f"faster; -mx=5 reaches ~4.8%%.")
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help=f"7-Zip threads, -mmt (default: "
                             f"{default_workers()} = half the logical "
                             f"cores, leaving headroom for training).")
    parser.add_argument("--max-concurrent", type=int,
                        default=DEFAULT_MAX_CONCURRENT,
                        help=f"Maximum simultaneous background jobs "
                             f"(default: {DEFAULT_MAX_CONCURRENT}).")
    parser.add_argument("--keep", action="store_true",
                        help="Compress but do not delete the source.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen; change nothing.")
    parser.add_argument("--sevenzip", default=None,
                        help="Path to 7z.exe (default: autodetect).")

    npy = parser.add_argument_group(
        ".npy stores",
        "Compress the raw arrays in place with zstd, one file per frame, "
        "so they stay individually addressable. Works on any folder tree "
        "containing .npy - patches, reprojected_data, and the satellite "
        "and lightning product folders - and batches by dated subfolder. "
        "The float32 is stored exactly as it is: no dtype change, no "
        "requantisation. Readers go through load_array() and need no "
        "restore.")
    npy.add_argument("--compress-npy", nargs="+", metavar="PATH",
                     help="Compress every .npy under PATH to .npy.zst, "
                          "verifying each round-trip before deleting the "
                          "original.")
    npy.add_argument("--restore-npy", nargs="+", metavar="PATH",
                     help="Decompress every .npy.zst under PATH back to "
                          ".npy.")
    npy.add_argument("--verify-npy", nargs="+", metavar="PATH",
                     help="Decompress every .npy.zst under PATH and parse "
                          "the array, without writing anything.")
    npy.add_argument("--npy-stats", nargs="+", metavar="PATH",
                     help="Sample PATH and project the saving. Changes "
                          "nothing.")
    npy.add_argument("--zstd-level", type=int, default=DEFAULT_ZSTD_LEVEL,
                     help=f"zstd level (default: {DEFAULT_ZSTD_LEVEL}). "
                          f"Measured on this project: 10 gives 8.5x at "
                          f"~35 MB/s per worker; 19 gives 10.7x at 3 MB/s.")
    npy.add_argument("--npy-workers", type=int, default=npy_default_workers(),
                     help=f"Worker processes (default: "
                          f"{npy_default_workers()} = cores minus one).")
    npy.add_argument("--npy-limit", type=int, default=None,
                     help="Stop after N files. For trying it on a small "
                          "sample first.")
    args = parser.parse_args()

    npy_roots = (args.compress_npy or args.restore_npy or args.verify_npy
                 or args.npy_stats)
    if npy_roots:
        roots = [Path(p) for p in npy_roots]
        if args.npy_stats:
            return npy_stats(roots, args.zstd_level)
        action = ("compress" if args.compress_npy else
                  "restore" if args.restore_npy else "verify")
        return 1 if run_npy_pass(
            roots, action, args.zstd_level, args.npy_workers,
            keep=args.keep or action == "verify",
            dry_run=args.dry_run, limit=args.npy_limit) else 0

    data_root = resolve_data_root(args.data_root)
    datasets_root = resolve_datasets_root(args.data_root,
                                          args.datasets_root)

    if args.jobs:
        print_jobs(datasets_root)
        return 0

    infos = discover(datasets_root, data_root)
    by_tag = {i.run_tag: i for i in infos}

    reclaim_tags = list(args.reclaim or [])
    if args.reclaim_all:
        sweep = [i.run_tag for i in infos if i.reclaimable]
        if not sweep:
            print("Nothing to reclaim: no dataset is both archived and still "
                  "on disk.")
            if not (args.compress or args.restore or reclaim_tags):
                return 0
        reclaim_tags.extend(t for t in sweep if t not in reclaim_tags)

    if not args.compress and not args.restore and not reclaim_tags:
        print_listing(infos, datasets_root)
        return 0

    requested = (args.compress or []) + (args.restore or []) + reclaim_tags
    unknown = [t for t in requested if t not in by_tag]
    if unknown:
        print(f"ERROR: unknown dataset(s): {unknown}")
        print(f"Known: {sorted(by_tag) or '(none)'}")
        return 2

    # Background dispatch: hand each tag to a detached child and return.
    if args.background:
        for tag in (args.compress or []):
            spawn_job("compress", tag, datasets_root, args.level,
                      args.workers, args.sevenzip, args.max_concurrent)
        for tag in reclaim_tags:
            spawn_job("reclaim", tag, datasets_root, args.level,
                      args.workers, args.sevenzip, args.max_concurrent)
        if args.restore:
            print("NOTE: --restore is not backgrounded; training needs the "
                  "bytes before it can start.")
            exe = find_sevenzip(args.sevenzip)
            for tag in args.restore:
                restore(by_tag[tag], exe, args.dry_run)
        return 0

    exe = find_sevenzip(args.sevenzip)
    print(f"Using 7-Zip: {exe}   threads: {args.workers}\n")

    failures = 0
    for tag in (args.compress or []):
        if not compress(by_tag[tag], exe, args.level, args.workers,
                        args.keep, args.dry_run):
            failures += 1
        print()
    for tag in reclaim_tags:
        if not reclaim(by_tag[tag], exe, args.level, args.workers,
                       args.dry_run):
            failures += 1
        print()
    for tag in (args.restore or []):
        if not restore(by_tag[tag], exe, args.dry_run):
            failures += 1
        print()

    if not args.dry_run:
        print_listing(discover(datasets_root, data_root), datasets_root)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
