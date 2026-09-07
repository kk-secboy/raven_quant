"""Immutable, numeric-only prepared model data shared across isolated experiments.

Only the trusted controller may write the cache. Model sandboxes receive read-only
mounts; this format deliberately contains no pickle or executable Python objects.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_VERSION = "model-prepared-data-v1"
_JSON_LIMIT = 16 * 1024 * 1024
_FLOAT_DTYPES = frozenset({"<f4", "<f8", ">f4", ">f8"})
_INDEX_FIELDS = frozenset(
    {
        "names", "datetime_level", "instrument_levels", "datetime_codes",
        "instrument_codes", "timezone", "freq",
    }
)


@dataclass
class PreparedModelData:
    index: pd.MultiIndex
    features: dict[tuple[str, str], np.ndarray]
    infer_labels: pd.DataFrame
    learn_labels: pd.DataFrame


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("prepared data JSON keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("prepared data contract must contain strict JSON values")


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _json_value(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, UnicodeError) as exc:
        raise ValueError("prepared data contract must contain finite strict JSON") from exc


def canonical_key(contract: Mapping) -> str:
    if not isinstance(contract, Mapping):
        raise ValueError("prepared data contract must be a mapping")
    return hashlib.sha256(_json_bytes(contract)).hexdigest()


def _safe_path(path: Path, *, directory: bool) -> Path:
    path = Path(os.path.abspath(path))
    for component in (path, *path.parents):
        try:
            info = component.lstat()
        except OSError as exc:
            raise ValueError(f"prepared data path is unavailable: {component}") from exc
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("prepared data paths must not contain symbolic links or reparses")
        wanted = stat.S_ISDIR if component != path or directory else stat.S_ISREG
        if not wanted(info.st_mode):
            raise ValueError("prepared data path has an invalid file type")
    return path


def sha256file(path: Path) -> str:
    path = _safe_path(path, directory=False)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError("prepared data file cannot be read") from exc
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    path = _safe_path(path, directory=False)
    if path.stat().st_size > _JSON_LIMIT:
        raise ValueError("prepared data JSON exceeds its size limit")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("prepared data JSON contains a duplicate key")
            result[key] = value
        return result

    def bad_constant(value: str) -> None:
        raise ValueError(f"prepared data JSON contains non-finite value: {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
            parse_constant=bad_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("prepared data JSON is invalid") from exc


def prepared_manifest(directory: Path) -> dict:
    """Read metadata only; callers must use load_prepared_data to trust its contents."""
    directory = _safe_path(directory, directory=True)
    value = _read_json(directory / "manifest.json")
    if not isinstance(value, dict):
        raise ValueError("prepared data manifest must be an object")
    return value


def manifest_sha256(directory: Path) -> str:
    return sha256file(_safe_path(directory, directory=True) / "manifest.json")


def _names(names: Any, length: int) -> list[str | None]:
    if not isinstance(names, (list, tuple)) or len(names) != length:
        raise ValueError("prepared data index names are invalid")
    if any(name is not None and not isinstance(name, str) for name in names):
        raise ValueError("prepared data index names must be strings or null")
    return list(names)


def _column(column: Any) -> tuple[str, str]:
    if not isinstance(column, (list, tuple)) or len(column) != 2:
        raise ValueError("prepared data column must be a pair of strings")
    if any(not isinstance(part, str) for part in column):
        raise ValueError("prepared data column must be a pair of strings")
    return tuple(column)


def _numeric(array: Any, count: int) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise ValueError("prepared data columns must be NumPy arrays")
    if array.ndim != 1 or len(array) != count or array.dtype.str not in _FLOAT_DTYPES:
        raise ValueError("prepared data columns must be one-dimensional float32 or float64")
    return array


def _check_index(index: Any) -> None:
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise ValueError("prepared data index must have datetime and instrument levels")
    if not isinstance(index.levels[0], pd.DatetimeIndex):
        raise ValueError("prepared data first index level must contain datetimes")
    if any(not isinstance(value, str) for value in index.levels[1]):
        raise ValueError("prepared data instrument levels must contain strings")
    _names(index.names, 2)
    for codes, level in zip(index.codes, index.levels, strict=True):
        if len(codes) != len(index) or np.any(codes < -1) or np.any(codes >= len(level)):
            raise ValueError("prepared data index codes are invalid")


def _same_index(left: pd.MultiIndex, right: pd.MultiIndex) -> bool:
    return (
        isinstance(right, pd.MultiIndex)
        and left.nlevels == right.nlevels
        and list(left.names) == list(right.names)
        and all(a.equals(b) for a, b in zip(left.levels, right.levels, strict=True))
        and all(np.array_equal(a, b) for a, b in zip(left.codes, right.codes, strict=True))
    )


def _check_label_index(index: pd.MultiIndex, learn: pd.MultiIndex) -> None:
    # A prepared label frame can drop rows but cannot introduce or duplicate rows.
    # Check by index membership/counts, independent of its retained level encoding.
    if not learn.isin(index).all():
        raise ValueError("prepared learning labels are not a subset of the feature index")
    if learn.has_duplicates:
        counts = index.value_counts(dropna=False)
        learn_counts = learn.value_counts(dropna=False)
        if (learn_counts > counts.reindex(learn_counts.index, fill_value=0)).any():
            raise ValueError("prepared learning labels duplicate feature rows")


def _write_file(directory: Path, name: str, content: Any, files: dict, *, array: bool) -> None:
    path = directory / name
    try:
        with path.open("xb") as stream:
            if array:
                np.save(stream, content, allow_pickle=False)
            else:
                stream.write(_json_bytes(content))
            stream.flush()
            os.fsync(stream.fileno())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot write prepared data file: {name}") from exc
    files[name] = {
        "sha256": sha256file(path), "size_bytes": path.stat().st_size,
        "shape": list(content.shape) if array else None,
        "dtype": content.dtype.str if array else None,
    }


def _sync_directory(directory: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_index(directory: Path, index: pd.MultiIndex, prefix: str, files: dict) -> dict:
    _check_index(index)
    dates = index.levels[0]
    spec = {
        "names": list(index.names), "datetime_level": f"{prefix}datetime-level.npy",
        "instrument_levels": f"{prefix}instrument-levels.json",
        "datetime_codes": f"{prefix}datetime-codes.npy",
        "instrument_codes": f"{prefix}instrument-codes.npy",
        "timezone": str(dates.tz) if dates.tz is not None else None,
        "freq": dates.freqstr,
    }
    _write_file(directory, spec["datetime_level"], dates.as_unit("ns").asi8, files, array=True)
    _write_file(directory, spec["instrument_levels"], index.levels[1].tolist(), files, array=False)
    for position, name in enumerate(("datetime_codes", "instrument_codes")):
        _write_file(
            directory, spec[name], np.asarray(index.codes[position], dtype=np.int64),
            files, array=True,
        )
    return spec


def _write_labels(directory: Path, frame: pd.DataFrame, prefix: str, files: dict) -> dict:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("prepared labels must be a DataFrame")
    multi = isinstance(frame.columns, pd.MultiIndex)
    columns = []
    for position, column in enumerate(frame.columns):
        name = f"{prefix}-label-{position:04d}.npy"
        values = _numeric(frame.iloc[:, position].to_numpy(copy=False), len(frame))
        _write_file(directory, name, values, files, array=True)
        columns.append({"column": list(_column(column)), "file": name})
    return {
        "columns": columns, "column_index_kind": "multiindex" if multi else "index",
        "column_names": _names(frame.columns.names, 2 if multi else 1),
    }


def write_prepared_data(directory: Path, *, contract: Mapping, data: PreparedModelData) -> dict:
    """Write a fresh staging directory; never overwrite or publish an existing entry."""
    key = canonical_key(contract)
    _check_index(data.index)
    if not isinstance(data.features, dict) or not data.features:
        raise ValueError("prepared data must contain feature columns")
    for column, array in data.features.items():
        _column(column)
        _numeric(array, len(data.index))
    for frame in (data.infer_labels, data.learn_labels):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("prepared labels must be a DataFrame")
        _check_index(frame.index)
    if not _same_index(data.index, data.infer_labels.index):
        raise ValueError("prepared inference labels must have the exact feature index")
    _check_label_index(data.index, data.learn_labels.index)
    directory = Path(directory).absolute()
    _safe_path(directory.parent, directory=True)
    try:
        directory.mkdir()
    except OSError as exc:
        raise ValueError("prepared staging directory must not already exist") from exc
    files: dict = {}
    features = []
    index = _write_index(directory, data.index, "", files)
    learn_index = _write_index(directory, data.learn_labels.index, "learn-", files)
    for position, (column, array) in enumerate(data.features.items()):
        name = f"feature-{position:04d}.npy"
        _write_file(directory, name, array, files, array=True)
        features.append({"column": list(column), "file": name})
    infer_labels = _write_labels(directory, data.infer_labels, "infer", files)
    learn_labels = _write_labels(directory, data.learn_labels, "learn", files)
    manifest = {
        "version": _VERSION, "contract": _json_value(contract), "key": key,
        "data_shape": {
            "rows": len(data.index), "features": len(features),
            "infer_labels": len(data.infer_labels.columns),
            "learn_rows": len(data.learn_labels), "learn_labels": len(data.learn_labels.columns),
        },
        "index": index, "learn_index": learn_index, "features": features,
        "infer_labels": infer_labels, "learn_labels": learn_labels, "files": files,
    }
    # The manifest is the completion marker and is intentionally written last.
    _write_file(directory, "manifest.json", manifest, {}, array=False)
    _sync_directory(directory)
    return manifest


def _fields(value: Any, expected: set | frozenset, description: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"prepared data {description} has invalid fields")
    return value


def _read_file(directory: Path, name: str, files: dict, used: set, *, array: bool) -> Any:
    # Names are supplied by this module, never interpreted from untrusted paths.
    if name not in files or name in used:
        raise ValueError("prepared data file binding is missing or duplicated")
    used.add(name)
    metadata = _fields(files[name], {"sha256", "size_bytes", "shape", "dtype"}, "file metadata")
    path = _safe_path(directory / name, directory=False)
    if (
        type(metadata["size_bytes"]) is not int
        or metadata["size_bytes"] < 0
        or path.stat().st_size != metadata["size_bytes"]
        or sha256file(path) != metadata["sha256"]
    ):
        raise ValueError("prepared data file checksum or size mismatch")
    if not array:
        if metadata["shape"] is not None or metadata["dtype"] is not None:
            raise ValueError("prepared data JSON file metadata is invalid")
        return _read_json(path)
    if (
        not isinstance(metadata["shape"], list) or len(metadata["shape"]) != 1
        or type(metadata["shape"][0]) is not int or metadata["shape"][0] < 0
        or not isinstance(metadata["dtype"], str)
        or metadata["dtype"] not in (_FLOAT_DTYPES | {"<i8", ">i8"})
    ):
        raise ValueError("prepared data array shape or dtype is invalid")
    try:
        result = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, TypeError, ValueError, EOFError) as exc:
        raise ValueError("prepared data array cannot be loaded safely") from exc
    if (
        not isinstance(result, np.ndarray) or list(result.shape) != metadata["shape"]
        or result.dtype.str != metadata["dtype"] or result.flags.writeable
    ):
        raise ValueError("prepared data array does not match its metadata")
    return result


def _read_index(directory: Path, spec: Any, prefix: str, files: dict, used: set, rows: int):
    spec = _fields(spec, _INDEX_FIELDS, "index")
    _names(spec["names"], 2)
    expected = {
        "datetime_level": f"{prefix}datetime-level.npy",
        "instrument_levels": f"{prefix}instrument-levels.json",
        "datetime_codes": f"{prefix}datetime-codes.npy",
        "instrument_codes": f"{prefix}instrument-codes.npy",
    }
    if any(spec[field] != name for field, name in expected.items()):
        raise ValueError("prepared data index path binding is invalid")
    timezone, frequency = spec["timezone"], spec["freq"]
    if any(value is not None and not isinstance(value, str) for value in (timezone, frequency)):
        raise ValueError("prepared data datetime metadata is invalid")
    dates = _read_file(directory, expected["datetime_level"], files, used, array=True)
    instruments = _read_file(directory, expected["instrument_levels"], files, used, array=False)
    codes = [
        _read_file(directory, expected[field], files, used, array=True)
        for field in ("datetime_codes", "instrument_codes")
    ]
    if dates.dtype.kind != "i" or any(array.dtype.kind != "i" for array in codes):
        raise ValueError("prepared data index arrays must use int64")
    if not isinstance(instruments, list) or any(not isinstance(item, str) for item in instruments):
        raise ValueError("prepared data instrument levels must contain strings")
    if len(set(instruments)) != len(instruments) or len(np.unique(dates)) != len(dates):
        raise ValueError("prepared data index levels must be unique")
    for array, size in zip(codes, (len(dates), len(instruments)), strict=True):
        if len(array) != rows or np.any(array < -1) or np.any(array >= size):
            raise ValueError("prepared data index codes are outside their bound levels")
    try:
        date_level = pd.to_datetime(dates, unit="ns", utc=timezone is not None)
        if timezone is not None:
            date_level = date_level.tz_convert(timezone)
        date_level = pd.DatetimeIndex(date_level, freq=frequency)
        return pd.MultiIndex(
            levels=[date_level, pd.Index(instruments, dtype=object)],
            codes=codes, names=spec["names"], verify_integrity=True,
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("prepared data index cannot be reconstructed") from exc


def _read_labels(directory, spec, prefix, files, used, index, count):
    spec = _fields(spec, {"columns", "column_index_kind", "column_names"}, "labels")
    if not isinstance(spec["columns"], list) or len(spec["columns"]) != count:
        raise ValueError("prepared data label column count is invalid")
    kind = spec["column_index_kind"]
    if not isinstance(kind, str) or kind not in {"multiindex", "index"}:
        raise ValueError("prepared data label column index kind is invalid")
    names = _names(spec["column_names"], 2 if kind == "multiindex" else 1)
    arrays, columns = {}, []
    for position, entry in enumerate(spec["columns"]):
        entry = _fields(entry, {"column", "file"}, "label column")
        name = f"{prefix}-label-{position:04d}.npy"
        if entry["file"] != name:
            raise ValueError("prepared data label path binding is invalid")
        columns.append(_column(entry["column"]))
        arrays[position] = _numeric(
            _read_file(directory, name, files, used, array=True), len(index),
        )
    result = pd.DataFrame(arrays, index=index, copy=False)
    if kind == "multiindex":
        result.columns = pd.MultiIndex.from_tuples(columns, names=names)
    else:
        result.columns = pd.Index(columns, tupleize_cols=False, name=names[0])
    return result


def load_prepared_data(
    directory: Path, *, expected_contract: Mapping, expected_manifest_sha256: str | None = None,
) -> PreparedModelData:
    """Verify every bound file before returning read-only memory-mapped columns."""
    directory = _safe_path(directory, directory=True)
    key = canonical_key(expected_contract)
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256(directory) != expected_manifest_sha256
    ):
        raise ValueError("prepared data manifest checksum mismatch")
    manifest = _fields(
        prepared_manifest(directory),
        {
            "version", "contract", "key", "data_shape", "index", "learn_index",
            "features", "infer_labels", "learn_labels", "files",
        }, "manifest",
    )
    if (
        manifest["version"] != _VERSION or manifest["key"] != key
        or _json_bytes(manifest["contract"]) != _json_bytes(expected_contract)
        or canonical_key(manifest["contract"]) != key
    ):
        raise ValueError("prepared data contract identity mismatch")
    shape = _fields(
        manifest["data_shape"], {"rows", "features", "infer_labels", "learn_rows", "learn_labels"},
        "data shape",
    )
    if (
        any(type(value) is not int or value < 0 for value in shape.values())
        or shape["features"] < 1
    ):
        raise ValueError("prepared data dimensions must be nonnegative integers")
    files = manifest["files"]
    if not isinstance(files, dict):
        raise ValueError("prepared data file inventory must be an object")
    used: set[str] = set()
    index = _read_index(directory, manifest["index"], "", files, used, shape["rows"])
    learn_index = _read_index(
        directory, manifest["learn_index"], "learn-", files, used, shape["learn_rows"],
    )
    _check_label_index(index, learn_index)
    entries = manifest["features"]
    if not isinstance(entries, list) or len(entries) != shape["features"]:
        raise ValueError("prepared data feature count is invalid")
    features = {}
    for position, entry in enumerate(entries):
        entry = _fields(entry, {"column", "file"}, "feature column")
        column, name = _column(entry["column"]), f"feature-{position:04d}.npy"
        if column in features or entry["file"] != name:
            raise ValueError("prepared data feature identity or path binding is invalid")
        features[column] = _numeric(
            _read_file(directory, name, files, used, array=True), len(index),
        )
    infer = _read_labels(
        directory, manifest["infer_labels"], "infer", files, used, index, shape["infer_labels"],
    )
    learn = _read_labels(
        directory, manifest["learn_labels"], "learn", files, used, learn_index,
        shape["learn_labels"],
    )
    if (
        set(files) != used
        or {entry.name for entry in directory.iterdir()} != used | {"manifest.json"}
    ):
        raise ValueError("prepared data directory has unbound or unexpected files")
    return PreparedModelData(index, features, infer, learn)


def _rename_no_replace(source: Path, target: Path) -> None:
    if os.name == "nt":
        os.rename(source, target)
        return
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise ValueError("atomic prepared data publication requires renameat2 on this platform")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error, os.strerror(error), str(target))
        raise OSError(error, os.strerror(error), str(target))


def publish_prepared_data(staging: Path, cache_root: Path, *, contract: Mapping) -> Path:
    """Atomically publish a verified staging tree; a valid concurrent winner is retained."""
    staging = _safe_path(staging, directory=True)
    load_prepared_data(staging, expected_contract=contract)
    cache_root = Path(cache_root).absolute()
    # Validate every existing ancestor before creating controller-owned directories.
    existing = cache_root
    while not existing.exists() and not existing.is_symlink():
        existing = existing.parent
    _safe_path(existing, directory=True)
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError("prepared data cache root cannot be created") from exc
    cache_root = _safe_path(cache_root, directory=True)
    target = cache_root / canonical_key(contract)
    if target == staging:
        return target
    if target.exists() or target.is_symlink():
        load_prepared_data(target, expected_contract=contract)
        return target
    try:
        _rename_no_replace(staging, target)
        _sync_directory(cache_root)
        if staging.parent != cache_root:
            _sync_directory(staging.parent)
    except FileExistsError:
        load_prepared_data(target, expected_contract=contract)
    except OSError as exc:
        raise ValueError(
            "prepared data publication requires an atomic same-filesystem rename"
        ) from exc
    return target
