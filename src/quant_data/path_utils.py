from __future__ import annotations

import re
from pathlib import Path

_WINDOWS_ABSOLUTE_PATH = re.compile(r"^(?P<drive>[A-Za-z]):(?P<tail>/.*)$")


def to_wsl_path(path: Path) -> str:
    """Map an absolute Windows path to WSL, regardless of the current host OS."""
    raw = str(path).replace("\\", "/")
    match = _WINDOWS_ABSOLUTE_PATH.fullmatch(raw)
    if match:
        return f"/mnt/{match.group('drive').lower()}{match.group('tail')}"

    normalized = path.resolve().as_posix()
    match = _WINDOWS_ABSOLUTE_PATH.fullmatch(normalized)
    if match:
        return f"/mnt/{match.group('drive').lower()}{match.group('tail')}"
    return normalized
