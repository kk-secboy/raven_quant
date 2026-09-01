from __future__ import annotations

import hashlib
from pathlib import Path

BASELINE_MODEL_STUB_SHA256 = (
    "7eb9694655c24a8ebd8f61d792c66a95d68a56c7cf8eb00b2d60fbb11e3ca9d3"
)
_SOURCE_RELATIVE_PATH = Path("scripts") / "baseline_model_stub.py"
_PACKAGED_RELATIVE_PATH = Path("_artifacts") / "baseline_model_stub.py"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_governed_baseline_model_stub(
    project_root: Path | None = None,
    *,
    package_root: Path | None = None,
) -> Path:
    """Resolve and verify the immutable model stub in either runtime layout.

    Editable/source checkouts retain the canonical script under ``scripts``.
    A built wheel carries the same bytes inside ``quant_platform/_artifacts``.
    The integrity check fails closed before either path can become a governed
    research artifact.
    """

    installed_package_root = package_root or Path(__file__).resolve().parent
    candidates: list[Path] = []
    if project_root is not None:
        candidates.append(project_root / _SOURCE_RELATIVE_PATH)
    candidates.append(installed_package_root / _PACKAGED_RELATIVE_PATH)

    for candidate in candidates:
        if not candidate.is_file():
            continue
        actual_sha256 = _sha256_file(candidate)
        if actual_sha256 != BASELINE_MODEL_STUB_SHA256:
            raise ValueError(
                "governed platform model stub failed its SHA-256 integrity check"
            )
        return candidate.resolve()
    raise ValueError("governed platform model stub is unavailable")
