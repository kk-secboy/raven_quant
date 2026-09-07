"""Compare full prepared data with independently reopened numeric artifacts.

Run prepare, verify and warm in separate, equivalently limited containers. This
does not drop the host page cache, fit models, or open a new formal OOS window.
The existing manifest controls the complete market, features and data periods.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_sandbox_runner import (  # noqa: E402
    initialize_model_qlib,
    model_memory_snapshot,
    resolve_manifest_label_contract,
)

from quant_platform import model_data_handler, model_data_request, model_prepared_data  # noqa: E402
from quant_platform.model_data_handler import (  # noqa: E402
    build_model_handler_from_prepared_data,
    prepare_memory_bounded_model_data,
)
from quant_platform.model_data_request import prepared_data_request  # noqa: E402
from quant_platform.model_prepared_data import (  # noqa: E402
    PreparedModelData,
    canonical_key,
    load_prepared_data,
    manifest_sha256,
    sha256file,
    write_prepared_data,
)

BENCHMARK_VERSION = "model-prepared-data-benchmark-v1"
_FETCH_SESSIONS = 16


def _encoded(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(_encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def _update_array(digest: Any, values: np.ndarray) -> None:
    """Hash the exact logical C-order bytes with at most one MiB of copying."""
    if values.ndim != 1 or values.dtype.hasobject:
        raise ValueError("benchmark oracle requires one-dimensional numeric arrays")
    step = max(1, (1024 * 1024) // values.dtype.itemsize)
    for start in range(0, len(values), step):
        block = np.ascontiguousarray(values[start:start + step])
        digest.update(memoryview(block).cast("B"))


def _array_oracle(values: np.ndarray) -> dict[str, Any]:
    digest = hashlib.sha256()
    _update_array(digest, values)
    return {"dtype": values.dtype.str, "shape": list(values.shape), "sha256": digest.hexdigest()}


def _index_levels(index: pd.MultiIndex) -> dict[str, Any]:
    dates = index.levels[0]
    instruments = index.levels[1].tolist()
    return {
        "names": list(index.names),
        "datetime": _array_oracle(dates.as_unit("ns").asi8),
        "datetime_timezone": str(dates.tz) if dates.tz is not None else None,
        "datetime_frequency": dates.freqstr,
        "instruments": {
            "count": len(instruments), "sha256": hashlib.sha256(_encoded(instruments)).hexdigest(),
        },
    }


def _index_oracle(index: pd.MultiIndex) -> dict[str, Any]:
    return {
        "rows": len(index), "levels": _index_levels(index),
        "codes": [_array_oracle(np.asarray(codes)) for codes in index.codes],
    }


def _frame_oracle(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "index": _index_oracle(frame.index),
        "column_names": list(frame.columns.names),
        "columns": [
            {"column": list(column), **_array_oracle(frame.iloc[:, position].to_numpy(copy=False))}
            for position, column in enumerate(frame.columns)
        ],
    }


def _data_oracle(data: PreparedModelData) -> dict[str, Any]:
    return {
        "index": _index_oracle(data.index),
        "features": [
            {"column": list(column), **_array_oracle(values)}
            for column, values in data.features.items()
        ],
        "infer_labels": _frame_oracle(data.infer_labels),
        "learn_labels": _frame_oracle(data.learn_labels),
    }


def _segments(manifest: dict, provider: Path, label: dict) -> dict[str, list[str]]:
    """Use the runner's purged validation and prediction-as-test DatasetH segments."""
    periods = manifest["periods"]
    calendar = [
        line.strip() for line in (provider / "calendars" / "day.txt").read_text().splitlines()
        if line.strip()
    ]
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("benchmark provider calendar must contain ordered unique sessions")
    positions = {date: position for position, date in enumerate(calendar)}
    if periods["valid_end"] >= periods["test_start"]:
        raise ValueError("benchmark validation reaches the sealed formal OOS")
    purge = int(label["purge_sessions"])
    try:
        gap = positions[periods["valid_start"]] - positions[periods["train_end"]] - 1
        fit_end = positions[periods["valid_end"]] - purge
        if gap < purge or fit_end < positions[periods["valid_start"]]:
            raise ValueError("benchmark periods do not preserve governed label purge")
        if periods["test_start"] in positions:
            embargo = positions[periods["test_start"]] - positions[periods["valid_end"]] - 1
            if embargo < int(label["embargo_sessions"]):
                raise ValueError("benchmark periods do not preserve governed OOS embargo")
        prediction = manifest.get("prediction_segment") or "valid"
        if prediction not in {"valid", "test"}:
            raise ValueError("benchmark prediction segment is invalid")
        if prediction == "test" and manifest.get("final_oos_opened") is not True:
            raise ValueError("benchmark cannot open a sealed formal OOS segment")
        return {
            "train": [periods["train_start"], periods["train_end"]],
            "valid": [periods["valid_start"], calendar[fit_end]],
            "test": [periods[f"{prediction}_start"], periods[f"{prediction}_end"]],
        }
    except KeyError as exc:
        raise ValueError("benchmark periods are not represented by the provider calendar") from exc


def _fetch_oracle(
    data: PreparedModelData, segments: Mapping[str, list[str]],
    stage: Callable[[str], None],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Exercise actual Qlib fetch in bounded date chunks, hashing each column in order.

    A full 158-column multi-year matrix is never retained. Six segment/data-key
    hashes cover train/valid/test for both inference and learning semantics.
    """
    from qlib.data.dataset.handler import DataHandlerLP

    adapter_start = time.perf_counter()
    handler = build_model_handler_from_prepared_data(data)
    timings = {"handler_adapter_seconds": time.perf_counter() - adapter_start}
    output = {}
    first_fetch = True
    for data_key in (DataHandlerLP.DK_I, DataHandlerLP.DK_L):
        for name, (start, end) in segments.items():
            key = f"{data_key}:{name}"
            dates = data.index.levels[0]
            dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
            columns = None
            index_levels = None
            code_hashes = [hashlib.sha256(), hashlib.sha256()]
            code_dtypes = None
            hashes: list[Any] = []
            rows = 0
            fetch_seconds = 0.0
            hash_seconds = 0.0
            chunks = 0
            for offset in range(0, max(len(dates), 1), _FETCH_SESSIONS):
                group = dates[offset:offset + _FETCH_SESSIONS]
                selector = (
                    slice(str(group[0].date()), str(group[-1].date()))
                    if len(group) else slice(start, end)
                )
                began = time.perf_counter()
                frame = handler.fetch(
                    selector, level="datetime", col_set=["feature", "label"], data_key=data_key,
                )
                elapsed = time.perf_counter() - began
                fetch_seconds += elapsed
                if first_fetch:
                    timings["first_fetch_seconds"] = elapsed
                    timings["first_fetch_rows"] = len(frame)
                    first_fetch = False
                began = time.perf_counter()
                current_columns = [
                    {"column": list(column), "dtype": frame.iloc[:, position].dtype.str}
                    for position, column in enumerate(frame.columns)
                ]
                current_levels = _index_levels(frame.index)
                current_code_dtypes = [np.asarray(code).dtype.str for code in frame.index.codes]
                if columns is None:
                    columns, index_levels = current_columns, current_levels
                    code_dtypes = current_code_dtypes
                    hashes = [hashlib.sha256() for _ in columns]
                elif (
                    columns != current_columns or index_levels != current_levels
                    or code_dtypes != current_code_dtypes
                ):
                    raise ValueError("chunked Qlib fetch changed its column or index-level binding")
                for position, digest in enumerate(hashes):
                    _update_array(digest, frame.iloc[:, position].to_numpy(copy=False))
                for position, digest in enumerate(code_hashes):
                    _update_array(digest, np.asarray(frame.index.codes[position]))
                rows += len(frame)
                chunks += 1
                del frame
                hash_seconds += time.perf_counter() - began
            output[key] = {
                "period": [start, end], "rows": rows, "chunks": chunks,
                "index_levels": index_levels,
                "index_codes": [
                    {"dtype": dtype, "shape": [rows], "sha256": digest.hexdigest()}
                    for dtype, digest in zip(code_dtypes, code_hashes, strict=True)
                ],
                "columns": [
                    {**column, "shape": [rows], "sha256": digest.hexdigest()}
                    for column, digest in zip(columns, hashes, strict=True)
                ],
            }
            timings[f"fetch_seconds:{key}"] = fetch_seconds
            timings[f"fetch_oracle_hash_seconds:{key}"] = hash_seconds
            stage(f"segment_verified:{key}")
    return output, timings


def _producer_identity(path: Path | None) -> dict[str, str]:
    actual = {
        "model_data_handler_sha256": sha256file(Path(model_data_handler.__file__)),
        "model_data_request_sha256": sha256file(Path(model_data_request.__file__)),
        "model_prepared_data_sha256": sha256file(Path(model_prepared_data.__file__)),
        "qlib_version": importlib.metadata.version("pyqlib"),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
    }
    if path is not None:
        expected = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(expected, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) or not value
            for key, value in expected.items()
        ):
            raise ValueError("benchmark producer identity must map strings to nonempty strings")
        for key, value in actual.items():
            if key in expected and expected[key] != value:
                raise ValueError(f"benchmark producer identity mismatch: {key}")
        actual.update(expected)
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "verify", "warm"), required=True)
    parser.add_argument("--additional-factors", type=Path)
    parser.add_argument("--producer-identity", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    manifest_path = args.manifest.resolve(strict=True)
    provider = args.provider.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("contract_version") != "model-sandbox-input-v1"
    ):
        raise ValueError("benchmark requires an existing governed model sandbox manifest")
    if manifest.get("inference_only") is True or manifest.get("live_retrain") is True:
        raise ValueError("benchmark is limited to the existing model research preparation path")
    label = resolve_manifest_label_contract(manifest)
    segments = _segments(manifest, provider, label)
    factors = args.additional_factors
    if factors is None and manifest.get("additional_factors_path"):
        factors = Path(manifest["additional_factors_path"])
    if factors is not None:
        factors = factors.resolve(strict=True)
    identity = _producer_identity(args.producer_identity)
    contract = prepared_data_request(
        manifest, provider=provider, label_contract=label, producer_identity=identity,
        additional_factors=factors,
    )
    output = args.output.absolute()
    if args.mode == "prepare":
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError("benchmark verification requires an existing preparation output")
    run_id = f"{args.mode}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    events_path = output / f"{run_id}-memory.jsonl"
    memory_stages = []

    def stage(name: str) -> None:
        value = model_memory_snapshot(name)
        value["elapsed_seconds"] = time.perf_counter() - started
        memory_stages.append(value)
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(_encoded(value).decode("utf-8") + "\n")
        print(json.dumps({"stage": name, "elapsed_seconds": value["elapsed_seconds"]}), flush=True)

    report: dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION, "run_id": run_id, "mode": args.mode,
        "started_at": datetime.now(UTC).isoformat(), "pid": os.getpid(),
        "manifest_sha256": sha256file(manifest_path), "contract_key": canonical_key(contract),
        "producer_identity": identity, "segments": segments,
        "test_segment_meaning": "existing runner prediction alias; no new formal OOS access",
        "scope": "full governed data and bounded segment fetch; no model fit or prediction proof",
        "cache_conditions": {
            "fresh_process_required": True, "qlib_kernels": 1,
            "os_page_cache": "uncontrolled; not dropped or claimed cold",
            "application_prepared_artifact": "absent" if args.mode == "prepare" else "reused",
            "cold_write_meaning": "first artifact creation, not a cold filesystem guarantee",
            "warm_open_meaning": "new process, full SHA verification and mmap of existing artifact",
            "first_fetch_meaning": f"first at-most-{_FETCH_SESSIONS}-session matrix, all columns",
        },
        "timings": {}, "memory_stages": memory_stages,
    }
    stage("started")
    try:
        import qlib

        began = time.perf_counter()
        initialize_model_qlib(
            qlib, output=output / f"{run_id}-runtime", provider_uri=str(provider), kernels=1,
        )
        report["timings"]["qlib_init_seconds"] = time.perf_counter() - began
        stage("qlib_initialized")
        prepared = output / "prepared"
        if args.mode == "prepare":
            began = time.perf_counter()
            data = prepare_memory_bounded_model_data(
                features=manifest["feature_set"]["features"],
                label_expression=label["label_expression"],
                instruments=manifest.get("universe", "cn_all"), start_time=contract["train_start"],
                end_time=contract["load_end"], fit_end_time=contract["train_end"],
                additional_factors_path=factors, on_stage=stage,
            )
            report["timings"]["prepare_seconds"] = time.perf_counter() - began
            stage("uncached_preparation_complete")
        else:
            baseline = json.loads((output / "oracle.json").read_text(encoding="utf-8"))
            if (
                baseline["manifest_sha256"] != report["manifest_sha256"]
                or baseline["contract_key"] != report["contract_key"]
            ):
                raise ValueError("benchmark source manifest or producer changed between processes")
            began = time.perf_counter()
            data = load_prepared_data(
                prepared, expected_contract=contract,
                expected_manifest_sha256=baseline["prepared_manifest_sha256"],
            )
            timing_name = "verify_seconds" if args.mode == "verify" else "warm_open_seconds"
            report["timings"][timing_name] = time.perf_counter() - began
            stage("prepared_artifact_verified_and_mapped")
        began = time.perf_counter()
        oracle = _data_oracle(data)
        report["timings"]["data_oracle_hash_seconds"] = time.perf_counter() - began
        stage("data_oracle_complete")
        fetch_oracle, fetch_timings = _fetch_oracle(data, segments, stage)
        report["timings"].update(fetch_timings)
        if args.mode == "prepare":
            began = time.perf_counter()
            write_prepared_data(prepared, contract=contract, data=data)
            report["timings"]["cold_write_seconds"] = time.perf_counter() - began
            pinned = manifest_sha256(prepared)
            _write_json(output / "oracle.json", {
                "benchmark_version": BENCHMARK_VERSION,
                "manifest_sha256": report["manifest_sha256"],
                "contract_key": report["contract_key"],
                "prepared_manifest_sha256": pinned, "data": oracle, "fetch": fetch_oracle,
            })
            report["prepared_manifest_sha256"] = pinned
            report["equivalence"] = "baseline recorded; independent verification pending"
            stage("prepared_artifact_written")
        else:
            if oracle != baseline["data"]:
                raise ValueError(
                    "prepared artifact differs from full uncached numeric/index oracle"
                )
            if fetch_oracle != baseline["fetch"]:
                raise ValueError("prepared handler differs from uncached segment fetch oracle")
            report["prepared_manifest_sha256"] = baseline["prepared_manifest_sha256"]
            report["equivalence"] = "exact raw-byte, dtype, shape, index and segment-fetch match"
        report["status"] = "succeeded"
        stage("completed")
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        stage("failed")
        raise
    finally:
        report["timings"]["total_process_work_seconds"] = time.perf_counter() - started
        report["finished_at"] = datetime.now(UTC).isoformat()
        _write_json(output / f"{run_id}-report.json", report)


if __name__ == "__main__":
    main()
