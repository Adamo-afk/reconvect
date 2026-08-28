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
    python compress_datasets.py --compress TAG --background
    python compress_datasets.py --restore  TAG           # extract back
    python compress_datasets.py --reclaim  TAG           # drop the copy on disk
    python compress_datasets.py --reclaim-all            # sweep leftovers
    python compress_datasets.py --jobs                   # background job state
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

def print_listing(infos: list[DatasetInfo]) -> None:
    if not infos:
        print("No datasets found under our_data/datasets/.")
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

def main() -> int:
    parser = argparse.ArgumentParser(
        description="List, compress, restore and reclaim TFRecord datasets."
    )
    parser.add_argument("--data_root", default="./our_data")
    parser.add_argument("--datasets_root", default=None,
                        help="Default: <data_root>/datasets")
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
    args = parser.parse_args()

    data_root = Path(args.data_root)
    datasets_root = (Path(args.datasets_root) if args.datasets_root
                     else data_root / "datasets")

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
        print_listing(infos)
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
        print_listing(discover(datasets_root, data_root))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
