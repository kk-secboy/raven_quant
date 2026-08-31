from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

from quant_platform.runtime_source_closure import (
    position_risk_source_closure_inventory,
    position_risk_source_closure_sha256,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_overlay(project_root: Path, expected_closure_sha256: str) -> dict[str, object]:
    expected = str(expected_closure_sha256 or "").strip().lower()
    if not _SHA256.fullmatch(expected):
        raise ValueError("expected closure must be a lowercase SHA-256 digest")
    root = project_root.resolve()
    actual = position_risk_source_closure_sha256(root)
    if actual != expected:
        raise ValueError(
            f"runtime source closure mismatch: expected {expected}, observed {actual}"
        )

    inventory = position_risk_source_closure_inventory(root)["inventory"]
    checked = 0
    for item in inventory:
        relative = str(item["path"])
        if not relative.startswith("src/") or "#" in relative:
            continue
        parts = Path(relative).parts
        module = ".".join((*parts[1:-1], Path(parts[-1]).stem))
        if parts[-1] == "__init__.py":
            module = ".".join(parts[1:-1])
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise ValueError(f"installed runtime module is missing: {module}")
        installed = Path(spec.origin).resolve()
        source = (root / relative).resolve()
        if installed == source or "site-packages" not in installed.parts:
            raise ValueError(f"runtime module is not imported from site-packages: {module}")
        if _sha256(installed) != _sha256(source):
            raise ValueError(f"installed runtime module differs from /app/src: {module}")
        checked += 1
    if checked < 1:
        raise ValueError("runtime closure did not contain installed Python modules")
    return {
        "status": "ok",
        "project_root": str(root),
        "closure_sha256": actual,
        "installed_modules_verified": checked,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/app"))
    parser.add_argument("--expected-closure-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_overlay(args.project_root, args.expected_closure_sha256),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
