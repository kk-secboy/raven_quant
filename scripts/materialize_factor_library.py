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
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_data.qlib_builder import verify_qlib_output_manifest
from quant_platform.feature_set_registry import get_feature_set, resolve_feature_set

RECENT_SESSION_LIMIT = 512
STRATEGY_HEALTH_SESSION_LIMIT = 64


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


def _bounded_window_from_calendar(
    provider: Path,
    *,
    requested_start: str,
    requested_end: str,
    session_limit: int,
) -> tuple[str, str]:
    """Resolve the last N requested sessions without opening factor data."""

    if session_limit < 1:
        raise ValueError("factor materialization session limit is invalid")
    start = date.fromisoformat(requested_start)
    end = date.fromisoformat(requested_end)
    if start > end:
        raise ValueError("factor materialization requested window is invalid")
    calendar_path = provider / "calendars" / "day.txt"
    try:
        calendar = sorted(
            {
                date.fromisoformat(line.strip()[:10])
                for line in calendar_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        )
    except (OSError, ValueError) as exc:
        raise ValueError("Qlib daily calendar is missing or invalid") from exc
    selected = [session for session in calendar if start <= session <= end]
    if not selected or selected[-1] != end:
        raise ValueError("Qlib daily calendar does not reach the requested end")
    bounded = selected[-session_limit:]
    return bounded[0].isoformat(), bounded[-1].isoformat()


def _query_features(
    data_api: Any,
    instruments: Any,
    expressions: list[str],
    *,
    materialized_start: str,
    materialized_end: str,
) -> pd.DataFrame:
    return data_api.features(
        instruments,
        expressions,
        start_time=materialized_start,
        end_time=materialized_end,
        freq="day",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-set-id", default="unified-research-v1")
    parser.add_argument("--feature-set-definition")
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
    embedded_feature_set = None
    if args.feature_set_definition:
        embedded_feature_set = json.loads(
            Path(args.feature_set_definition).read_text(encoding="utf-8")
        )
    feature_set = (
        resolve_feature_set(args.feature_set_id, embedded_feature_set)
        if embedded_feature_set is not None
        else get_feature_set(args.feature_set_id)
    )
    storage_mode = (
        "recent_only"
        if str(feature_set["id"]).startswith("strategy-health:")
        else "full_and_recent"
    )
    session_limit = (
        STRATEGY_HEALTH_SESSION_LIMIT
        if storage_mode == "recent_only"
        else RECENT_SESSION_LIMIT
    )
    materialized_start, materialized_end = (
        _bounded_window_from_calendar(
            provider,
            requested_start=args.start,
            requested_end=args.end,
            session_limit=session_limit,
        )
        if storage_mode == "recent_only"
        else (args.start, args.end)
    )
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
            "requested_start": args.start,
            "requested_end": args.end,
            "materialized_start": materialized_start,
            "materialized_end": materialized_end,
            "session_limit": session_limit,
            "storage_mode": storage_mode,
            "completed": {},
            "blocked": {},
        }
    )
    expected_identity = {
        "feature_set_id": feature_set["id"],
        "feature_set_definition_sha256": feature_set["definition_sha256"],
        "dataset_identity_sha256": provenance["dataset_identity_sha256"],
        "universe": args.universe,
        "start": args.start,
        "end": args.end,
    }
    if storage_mode == "recent_only":
        expected_identity.update(
            {
                "storage_mode": storage_mode,
                "requested_start": args.start,
                "requested_end": args.end,
                "materialized_start": materialized_start,
                "materialized_end": materialized_end,
                "session_limit": session_limit,
            }
        )
    identity = {key: checkpoint.get(key) for key in expected_identity}
    if identity != expected_identity:
        raise ValueError("factor library checkpoint belongs to another frozen input")

    qlib.init(provider_uri=str(provider), region="cn")
    instruments = D.instruments(args.universe)
    pending = [
        (name, expression)
        for name, expression in feature_set["features"].items()
        if not _completed_with_recent(
            checkpoint["completed"].get(name), storage_mode=storage_mode
        )
        and name not in checkpoint["blocked"]
    ]
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        expressions = [expression for _, expression in batch]
        try:
            frame = _query_features(
                D,
                instruments,
                expressions,
                materialized_start=materialized_start,
                materialized_end=materialized_end,
            )
        except Exception as exc:
            # Resolve failures individually so one optional missing field does
            # not hide calculable definitions in the same batch.
            for name, expression in batch:
                try:
                    single = _query_features(
                        D,
                        instruments,
                        [expression],
                        materialized_start=materialized_start,
                        materialized_end=materialized_end,
                    )
                    _persist_one(
                        values_root,
                        name,
                        single.iloc[:, 0],
                        checkpoint,
                        storage_mode=storage_mode,
                        recent_session_limit=session_limit,
                    )
                except Exception as single_exc:
                    checkpoint["blocked"][name] = {
                        "reason": str(single_exc)[:1000],
                        "batch_error": str(exc)[:1000],
                    }
                _write_json_atomic(checkpoint_path, checkpoint)
            continue
        for index, (name, _expression) in enumerate(batch):
            _persist_one(
                values_root,
                name,
                frame.iloc[:, index],
                checkpoint,
                storage_mode=storage_mode,
                recent_session_limit=session_limit,
            )
        _write_json_atomic(checkpoint_path, checkpoint)

    result = {
        **expected_identity,
        "contract_version": "factor-library-materialization-v1",
        "feature_set": feature_set,
        "storage_mode": storage_mode,
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
    *,
    storage_mode: str,
    recent_session_limit: int,
) -> None:
    safe_name = hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]
    frame = values.to_frame("factor").swaplevel().sort_index()
    frame.index.names = ["datetime", "instrument"]
    dates = pd.DatetimeIndex(
        pd.to_datetime(frame.index.get_level_values("datetime"), errors="raise")
    ).tz_localize(None).normalize()
    recent_sessions = dates.unique().sort_values()[-recent_session_limit:]
    recent = frame.loc[dates.isin(recent_sessions)]
    recent_root = values_root.parent / "recent"
    recent_root.mkdir(parents=True, exist_ok=True)
    recent_target = recent_root / f"{safe_name}.parquet"
    recent_temporary = recent_target.with_suffix(".tmp.parquet")
    recent.to_parquet(
        recent_temporary,
        compression="zstd",
        engine="pyarrow",
    )
    os.replace(recent_temporary, recent_target)
    completed = {
        "finite": int(pd.to_numeric(frame["factor"], errors="coerce").notna().sum()),
        "recent_relative_path": recent_target.relative_to(
            values_root.parent
        ).as_posix(),
        "recent_sha256": _sha256(recent_target),
        "recent_rows": len(recent),
        "recent_start": (
            recent_sessions[0].date().isoformat() if len(recent_sessions) else None
        ),
        "recent_end": (
            recent_sessions[-1].date().isoformat() if len(recent_sessions) else None
        ),
        "recent_session_limit": recent_session_limit,
    }
    if storage_mode == "full_and_recent":
        target = values_root / f"{safe_name}.h5"
        temporary = target.with_suffix(".tmp.h5")
        frame.to_hdf(temporary, key="data", mode="w")
        os.replace(temporary, target)
        completed.update(
            {
                "relative_path": target.relative_to(values_root.parent).as_posix(),
                "sha256": _sha256(target),
                "rows": len(frame),
            }
        )
    checkpoint["completed"][name] = completed


def _completed_with_recent(value: Any, *, storage_mode: str) -> bool:
    if not isinstance(value, dict):
        return False
    required = {
        "recent_relative_path",
        "recent_sha256",
        "recent_start",
        "recent_end",
    }
    if storage_mode == "full_and_recent":
        required.update({"relative_path", "sha256", "rows"})
    return required.issubset(value)


if __name__ == "__main__":
    main()
