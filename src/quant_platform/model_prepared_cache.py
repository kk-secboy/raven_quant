"""Bounded trusted prepared-data cache with cross-process reader leases.

Lock files are permanent inode identities; never unlink them. Linux consumers
may hold a shared flock on ``lease_path`` through its read-only sandbox mount.
Windows uses conservative exclusive byte locks instead of shared reader locks.
"""

from __future__ import annotations

import errno
import math
import os
import re
import shutil
import stat
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, BinaryIO

from .model_prepared_data import (
    canonical_key,
    load_prepared_data,
    manifest_sha256,
    publish_prepared_data,
)

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_KEY = re.compile(r"[0-9a-f]{64}")
_STAGING = re.compile(r"([0-9a-f]{64})\.[0-9a-f]{32}")


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("prepared data cache exceeded its execution deadline")


def _safe(path: Path) -> Path:
    """Reject link/reparse traversal, including in ancestors, before file I/O."""
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("prepared cache path contains a link or reparse point")
        if part != path and not stat.S_ISDIR(info.st_mode):
            raise ValueError("prepared cache ancestor is not a directory")
    return path


def _mkdir(path: Path) -> Path:
    path = _safe(path)
    path.mkdir(parents=True, exist_ok=True)
    if not stat.S_ISDIR(_safe(path).lstat().st_mode):
        raise ValueError("prepared cache directory has an invalid type")
    return path


def _open_lock(path: Path, *, create: bool = True) -> BinaryIO:
    path = _safe(path)
    flags = os.O_RDWR | (os.O_CREAT if create else 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("prepared cache lease is not a regular file")
        if os.name != "nt":
            os.fchmod(descriptor, 0o644)  # Unprivileged read-only sandbox must acquire LOCK_SH.
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        return os.fdopen(descriptor, "r+b", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


def _lock(
    stream: BinaryIO, *, exclusive: bool, blocking: bool = True, deadline: float | None = None,
) -> bool:
    while True:
        _check_deadline(deadline)
        try:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                fcntl.flock(stream.fileno(), mode | fcntl.LOCK_NB)
            stream._prepared_locked = True
            return True
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            if not blocking:
                return False
            remaining = 0.05 if deadline is None else max(0, deadline - time.monotonic())
            time.sleep(min(0.05, remaining))


def _unlock(stream: BinaryIO) -> None:
    if not getattr(stream, "_prepared_locked", False):
        return  # A timed-out shared-to-exclusive upgrade already released its old lock.
    if os.name == "nt":
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    stream._prepared_locked = False


@contextmanager
def _locked(
    path: Path, *, exclusive: bool, blocking: bool = True, deadline: float | None = None,
    create: bool = True,
):
    _check_deadline(deadline)
    with _open_lock(path, create=create) as stream:
        acquired = _lock(stream, exclusive=exclusive, blocking=blocking, deadline=deadline)
        try:
            yield stream if acquired else None
        finally:
            if acquired:
                _unlock(stream)


def _size(directory: Path) -> int:
    if not stat.S_ISDIR(_safe(directory).lstat().st_mode):
        raise ValueError("prepared cache entry is not a directory")
    total = 0
    for current, directories, files in os.walk(directory, followlinks=False):
        for name in directories:
            if not stat.S_ISDIR(_safe(Path(current) / name).lstat().st_mode):
                raise ValueError("prepared cache contains an invalid directory")
        for name in files:
            info = _safe(Path(current) / name).lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("prepared cache contains a nonregular or hardlinked file")
            total += info.st_size
    return total


def _remove(directory: Path, parent: Path) -> None:
    directory = _safe(directory)
    if directory.parent != _safe(parent) or directory == parent:
        raise ValueError("prepared cache deletion escaped its exact parent")
    _size(directory)  # Inspect the complete tree before deleting any of it.
    shutil.rmtree(directory)


def _remove_idle_staging(root: Path, staging: Path, deadline: float | None = None) -> bool:
    """Caller owns this key exclusively; never manufacture a missing producer lease."""
    try:
        with _locked(staging / "producer.lease", exclusive=True, blocking=False,
                     deadline=deadline, create=False) as producer:
            idle = producer is not None
    except FileNotFoundError:
        return False  # Unknown legacy staging remains protected and charged to the budget.
    if idle:
        # Close before deletion for Windows; the key lock and unique UUID prevent ABA.
        _remove(staging, root / "building")
    return idle


@contextmanager
def _staging(root: Path, key: str, deadline: float | None):
    staging = _mkdir(root / "building" / f"{key}.{uuid.uuid4().hex}")
    try:
        with _locked(staging / "producer.lease", exclusive=False, deadline=deadline):
            yield staging
    finally:
        # A surviving preparer keeps its own read-only shared lease after parent failure.
        _remove_idle_staging(root, staging)


def _collect_staging(
    root: Path, *, protected: str, exclusive_key: str | None,
    ignore: Path | None, deadline: float | None,
) -> int:
    total = 0
    for staging in (root / "building").iterdir():
        _check_deadline(deadline)
        if staging == ignore:
            continue
        match = _STAGING.fullmatch(staging.name)
        if match is not None:
            key = match[1]
            if key != protected or key == exclusive_key:
                owner = nullcontext(True) if key == exclusive_key else _locked(
                    root / "locks" / f"{key}.lock", exclusive=True,
                    blocking=False, deadline=deadline,
                )
                with owner as key_lease:
                    if key_lease is not None and _remove_idle_staging(root, staging, deadline):
                        continue
        total += _size(staging)
    return total


def _touch(root: Path, key: str) -> None:
    target = _safe(root / "access" / key)
    temporary = target.with_name(f".{key}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(str(time.time_ns()), encoding="ascii")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _last_used(root: Path, entry: Path) -> int:
    access = _safe(root / "access" / entry.name)
    if not access.exists():
        return entry.stat().st_mtime_ns
    if not stat.S_ISREG(access.stat().st_mode) or access.stat().st_size > 32:
        raise ValueError("prepared cache access metadata is invalid")
    try:
        return int(access.read_text(encoding="ascii"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("prepared cache access metadata is invalid") from exc


def _make_room(
    root: Path, *, max_bytes: int, min_free_bytes: int, protected: str,
    ignore_building: Path | None = None, deadline: float | None = None,
    exclusive_key: str | None = None,
) -> bool:
    """Caller owns the budget lock; entry leases make GC safe against consumers."""
    entries = []
    for entry in (root / "entries").iterdir():
        _check_deadline(deadline)
        if _KEY.fullmatch(entry.name) is None:
            raise ValueError("prepared cache contains an unexpected entry name")
        entries.append((entry, _size(entry), _last_used(root, entry)))
    total = sum(size for _, size, _ in entries) + _collect_staging(
        root, protected=protected, exclusive_key=exclusive_key,
        ignore=ignore_building, deadline=deadline,
    )
    for entry, size, _last in sorted(entries, key=lambda item: (item[2], item[0].name)):
        _check_deadline(deadline)
        if total <= max_bytes and shutil.disk_usage(root).free >= min_free_bytes:
            return True
        if entry.name == protected:
            continue
        with _locked(root / "locks" / f"{entry.name}.lock", exclusive=True,
                     blocking=False, deadline=deadline) as lease:
            if lease is None:
                continue
            _remove(entry, root / "entries")
            _safe(root / "access" / entry.name).unlink(missing_ok=True)
            total -= size
    _check_deadline(deadline)
    return total <= max_bytes and shutil.disk_usage(root).free >= min_free_bytes


def _verify(entry: Path, contract: Mapping, deadline: float | None) -> None:
    _check_deadline(deadline)
    load_prepared_data(entry, expected_contract=contract)
    _check_deadline(deadline)


@contextmanager
def prepared_data_cache(
    cache_root: Path,
    *,
    contract: Mapping,
    build: Callable[[Path], None],
    max_bytes: int = 128 * 1024**3,
    min_free_bytes: int = 32 * 1024**3,
    deadline: float | None = None,
) -> Iterator[dict[str, Any] | None]:
    """Lease a verified entry or yield None when shared-cache capacity is unavailable.

    ``build`` is trusted and writes a *new* directory at the supplied path.
    It never receives the shared entries directory. Corruption is an error,
    not a cache miss. The context covers the complete model subprocess lifetime.
    """
    for name, value in (("max_bytes", max_bytes), ("min_free_bytes", min_free_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"prepared cache {name} must be a nonnegative integer")
    if deadline is not None and (
        isinstance(deadline, bool) or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
    ):
        raise ValueError("prepared cache deadline must be a finite monotonic timestamp")
    _check_deadline(deadline)
    started = time.monotonic()
    key = canonical_key(contract)
    root = _mkdir(cache_root)
    for name in ("entries", "locks", "access", "building"):
        _mkdir(root / name)
    entry = root / "entries" / key
    lease_path = root / "locks" / f"{key}.lock"
    hit, ready = False, False
    with _locked(lease_path, exclusive=False, deadline=deadline) as lease:
        if entry.exists() or entry.is_symlink():
            _verify(entry, contract, deadline)
            hit = ready = True
        else:
            # Recheck after taking exclusive ownership: another builder may win.
            _unlock(lease)
            _lock(lease, exclusive=True, deadline=deadline)
            if entry.exists() or entry.is_symlink():
                _verify(entry, contract, deadline)
                hit = ready = True
            else:
                with _locked(root / "locks" / "budget.lock", exclusive=True, deadline=deadline):
                    room = max_bytes > 0 and _make_room(
                        root, max_bytes=max_bytes, min_free_bytes=min_free_bytes, protected=key,
                        deadline=deadline, exclusive_key=key,
                    )
                    if room:
                        try:
                            with _staging(root, key, deadline) as staging:
                                fresh = staging / "entry"
                                _check_deadline(deadline)
                                build(fresh)
                                _check_deadline(deadline)
                                _verify(fresh, contract, deadline)
                                size = _size(fresh)
                                if size <= max_bytes and _make_room(
                                    root, max_bytes=max_bytes - size,
                                    min_free_bytes=min_free_bytes, protected=key,
                                    ignore_building=staging, deadline=deadline, exclusive_key=key,
                                ):
                                    _check_deadline(deadline)
                                    published = publish_prepared_data(
                                        fresh, root / "entries", contract=contract,
                                    )
                                    _check_deadline(deadline)
                                    if published != entry:
                                        raise ValueError(
                                            "prepared cache publisher returned another entry"
                                        )
                                    ready = True
                        except OSError as exc:
                            if exc.errno not in (errno.ENOSPC, errno.EDQUOT):
                                raise
            if os.name != "nt":
                _lock(lease, exclusive=False, deadline=deadline)
        if not ready:
            _check_deadline(deadline)
            yield None
            return
        if hit:
            room = max_bytes > 0 and _size(entry) <= max_bytes
            # A hit adds no data bytes: do not queue it behind another long build.
            with _locked(root / "locks" / "budget.lock", exclusive=True,
                         blocking=False, deadline=deadline) as budget:
                if room and budget is not None:
                    room = _make_room(
                        root, max_bytes=max_bytes, min_free_bytes=min_free_bytes, protected=key,
                        deadline=deadline,
                    )
            if not room:
                _check_deadline(deadline)
                yield None
                return
        _touch(root, key)
        _check_deadline(deadline)
        seal = manifest_sha256(entry)
        _check_deadline(deadline)
        yield {"entry": entry, "manifest_sha256": seal, "cache_hit": hit,
               "lease_path": lease_path, "elapsed_seconds": time.monotonic() - started}
