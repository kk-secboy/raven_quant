"""One non-blocking host lock for destructive QuantLab control-plane work."""

from __future__ import annotations

import functools
import os
import stat
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import ParamSpec, TypeVar

_LOCK_NAME = "quantlab-control-plane.lock"
_STATE_GUARD = threading.Lock()
_HELD: dict[Path, tuple[int, int, int]] = {}

P = ParamSpec("P")
R = TypeVar("R")


class ControlPlaneBusyError(RuntimeError):
    """Raised when another control-plane writer already owns the host lock."""


def default_control_plane_lock_path() -> Path:
    if os.name == "posix":
        return Path("/run/lock") / _LOCK_NAME
    return Path(tempfile.gettempdir()) / _LOCK_NAME


def _lock_file_descriptor(file_descriptor: int) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return

    import msvcrt

    if os.fstat(file_descriptor).st_size == 0:
        os.write(file_descriptor, b"\0")
        os.fsync(file_descriptor)
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    msvcrt.locking(file_descriptor, msvcrt.LK_NBLCK, 1)


def _unlock_file_descriptor(file_descriptor: int) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.flock(file_descriptor, fcntl.LOCK_UN)
        return

    import msvcrt

    os.lseek(file_descriptor, 0, os.SEEK_SET)
    msvcrt.locking(file_descriptor, msvcrt.LK_UNLCK, 1)


@contextmanager
def control_plane_lock(path: Path | None = None) -> Iterator[Path]:
    """Acquire the process-reentrant, cross-process control-plane lock."""

    target = (path or default_control_plane_lock_path()).expanduser().resolve()
    owner = threading.get_ident()
    with _STATE_GUARD:
        held = _HELD.get(target)
        if held is not None:
            file_descriptor, held_owner, depth = held
            if held_owner != owner:
                raise ControlPlaneBusyError(
                    f"another control-plane operation owns {target}"
                )
            _HELD[target] = (file_descriptor, owner, depth + 1)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_CREAT | os.O_RDWR
            flags |= getattr(os, "O_NOFOLLOW", 0)
            file_descriptor = os.open(target, flags, 0o600)
            try:
                metadata = os.fstat(file_descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise RuntimeError("control-plane lock path is not a regular file")
                if os.name == "posix" and metadata.st_uid != os.geteuid():
                    raise RuntimeError("control-plane lock is owned by another user")
                if os.name == "posix" and stat.S_IMODE(metadata.st_mode) & 0o022:
                    raise RuntimeError("control-plane lock is group/world writable")
                try:
                    _lock_file_descriptor(file_descriptor)
                except OSError as exc:
                    raise ControlPlaneBusyError(
                        f"another control-plane operation owns {target}"
                    ) from exc
            except Exception:
                os.close(file_descriptor)
                raise
            _HELD[target] = (file_descriptor, owner, 1)

    try:
        yield target
    finally:
        with _STATE_GUARD:
            file_descriptor, held_owner, depth = _HELD[target]
            if held_owner != owner:
                raise RuntimeError("control-plane lock ownership changed unexpectedly")
            if depth > 1:
                _HELD[target] = (file_descriptor, owner, depth - 1)
            else:
                try:
                    _unlock_file_descriptor(file_descriptor)
                finally:
                    os.close(file_descriptor)
                    del _HELD[target]


def control_plane_locked(function: Callable[P, R]) -> Callable[P, R]:
    """Decorate a control-plane mutation with the shared host lock."""

    @functools.wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with control_plane_lock():
            return function(*args, **kwargs)

    return wrapped
