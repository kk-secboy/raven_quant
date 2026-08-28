#!/usr/bin/env python3
"""Materialize one frozen factor-library view from a sealed Qlib provider.

This command is intentionally resumable and is designed for the server's
low-priority Qlib queue. It never mutates the source provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_data.qlib_builder import verify_qlib_output_manifest
from quant_platform.feature_set_registry import get_feature_set


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-set-id", default="unified-research-v1")
    parser.add_argument("--universe", default="cn_all")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64:
        raise ValueError("factor library batch size must be in [1, 64]")

    import qlib
    from qlib.data import D

    provider = Path(args.provider_uri).resolve(strict=True)
    provenance = json.loads(
        (provider / "metadata" / "provenance.json").read_text(encoding="utf-8")
    )
    verify_qlib_output_manifest(provider, provenance)
    feature_set = get_feature_set(args.feature_set_id)
    output = Path(args.output).resolve()
    values_root = output / "values"
    values_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.is_file()
        else {
            "contract_version": "factor-library-materialization-v1",
            "feature_set_id": feature_set["id"],
            "feature_set_definition_sha256": feature_set["definition_sha256"],
            "dataset_identity_sha256": provenance["dataset_identity_sha256"],
            "universe": args.universe,
            "start": args.start,
            "end": args.end,
            "completed": {},
            "blocked": {},
        }
    )
    identity = {
        key: checkpoint.get(key)
        for key in (
            "feature_set_id",
            "feature_set_definition_sha256",
            "dataset_identity_sha256",
            "universe",
            "start",
            "end",
        )
    }
    expected_identity = {
        "feature_set_id": feature_set["id"],
        "feature_set_definition_sha256": feature_set["definition_sha256"],
        "dataset_identity_sha256": provenance["dataset_identity_sha256"],
        "universe": args.universe,
        "start": args.start,
        "end": args.end,
    }
    if identity != expected_identity:
        raise ValueError("factor library checkpoint belongs to another frozen input")

    qlib.init(provider_uri=str(provider), region="cn")
    instruments = D.instruments(args.universe)
    pending = [
        (name, expression)
        for name, expression in feature_set["features"].items()
        if name not in checkpoint["completed"] and name not in checkpoint["blocked"]
    ]
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        expressions = [expression for _, expression in batch]
        try:
            frame = D.features(
                instruments,
                expressions,
                start_time=args.start,
                end_time=args.end,
                freq="day",
            )
        except Exception as exc:
            # Resolve failures individually so one optional missing field does
            # not hide calculable definitions in the same batch.
            for name, expression in batch:
                try:
                    single = D.features(
                        instruments,
                        [expression],
                        start_time=args.start,
                        end_time=args.end,
                        freq="day",
                    )
                    _persist_one(values_root, name, single.iloc[:, 0], checkpoint)
                except Exception as single_exc:
                    checkpoint["blocked"][name] = {
                        "reason": str(single_exc)[:1000],
                        "batch_error": str(exc)[:1000],
                    }
                _write_json_atomic(checkpoint_path, checkpoint)
            continue
        for index, (name, _expression) in enumerate(batch):
            _persist_one(values_root, name, frame.iloc[:, index], checkpoint)
        _write_json_atomic(checkpoint_path, checkpoint)

    result = {
        **expected_identity,
        "contract_version": "factor-library-materialization-v1",
        "completed_count": len(checkpoint["completed"]),
        "blocked_count": len(checkpoint["blocked"]),
        "completed": checkpoint["completed"],
        "blocked": checkpoint["blocked"],
        "status": "complete" if not checkpoint["blocked"] else "complete_with_blockers",
    }
    _write_json_atomic(output / "manifest.json", result)


def _persist_one(
    values_root: Path,
    name: str,
    values: pd.Series,
    checkpoint: dict[str, Any],
) -> None:
    safe_name = hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]
    target = values_root / f"{safe_name}.h5"
    frame = values.to_frame("factor").swaplevel().sort_index()
    frame.index.names = ["datetime", "instrument"]
    temporary = target.with_suffix(".tmp.h5")
    frame.to_hdf(temporary, key="data", mode="w")
    os.replace(temporary, target)
    checkpoint["completed"][name] = {
        "relative_path": target.relative_to(values_root.parent).as_posix(),
        "sha256": _sha256(target),
        "rows": len(frame),
        "finite": int(pd.to_numeric(frame["factor"], errors="coerce").notna().sum()),
    }


if __name__ == "__main__":
    main()
