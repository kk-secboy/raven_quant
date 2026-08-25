from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import uuid
from datetime import date
from pathlib import Path

from .parameter_experiments import split_research_period

_MANIFEST = "quantlab-rdagent-dataset-view.json"


def isolate_rdagent_periods(periods: dict[str, str]) -> dict[str, str]:
    """Keep RD-Agent's own train/valid/test loop strictly before final OOS."""

    required = (
        "train_start",
        "train_end",
        "valid_start",
        "valid_end",
        "test_start",
        "test_end",
    )
    try:
        parsed = {key: date.fromisoformat(str(periods[key])) for key in required}
    except (KeyError, ValueError) as exc:
        raise ValueError("governed research periods must contain valid ISO dates") from exc
    if not (
        parsed["train_start"]
        <= parsed["train_end"]
        < parsed["valid_start"]
        <= parsed["valid_end"]
        < parsed["test_start"]
        <= parsed["test_end"]
    ):
        raise ValueError("governed research periods must be ordered and non-overlapping")

    internal = split_research_period(parsed["valid_start"], parsed["valid_end"])
    isolated = {
        "train_start": parsed["train_start"].isoformat(),
        "train_end": parsed["train_end"].isoformat(),
        "valid_start": internal["in_sample"]["start"],
        "valid_end": internal["in_sample"]["end"],
        "test_start": internal["out_of_sample"]["start"],
        "test_end": internal["out_of_sample"]["end"],
    }
    if date.fromisoformat(isolated["test_end"]) >= parsed["test_start"]:
        raise ValueError("RD-Agent research periods overlap the reserved final test")
    return isolated


def prepare_rdagent_dataset_view(
    source: str | Path,
    destination: str | Path,
    *,
    cutoff: str,
) -> Path:
    """Create a day-frequency Qlib provider with no bytes after ``cutoff``."""

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if source_path == destination_path or source_path in destination_path.parents:
        raise ValueError("RD-Agent dataset view must be outside the source Qlib dataset")
    calendar_path = source_path / "calendars" / "day.txt"
    instruments_path = source_path / "instruments"
    features_path = source_path / "features"
    if not calendar_path.is_file() or not instruments_path.is_dir() or not features_path.is_dir():
        raise ValueError("RD-Agent requires a complete day-frequency Qlib dataset")
    governed_instruments_path = instruments_path / "cn_all.txt"
    if not governed_instruments_path.is_file():
        raise ValueError("RD-Agent requires the governed cn_all instrument universe")

    cutoff_date = date.fromisoformat(cutoff)
    full_calendar = [
        line.strip()
        for line in calendar_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected_calendar = [
        value for value in full_calendar if date.fromisoformat(value) <= cutoff_date
    ]
    if not selected_calendar:
        raise ValueError("RD-Agent dataset cutoff precedes the Qlib trading calendar")
    cutoff_index = len(selected_calendar) - 1
    effective_cutoff = selected_calendar[-1]
    governed_lines = _truncate_instruments(
        governed_instruments_path,
        effective_cutoff,
    )
    if not governed_lines:
        raise ValueError("RD-Agent cn_all universe is empty at the governed cutoff")
    governed_output = ("\n".join(governed_lines) + "\n").encode("utf-8")
    expected_manifest = {
        "schema_version": 2,
        "source": str(source_path),
        "market": "cn_all",
        "default_market_alias": "cn_all",
        "source_instruments_sha256": hashlib.sha256(
            governed_instruments_path.read_bytes()
        ).hexdigest(),
        "instruments_sha256": hashlib.sha256(governed_output).hexdigest(),
        "requested_cutoff": cutoff_date.isoformat(),
        "effective_cutoff": effective_cutoff,
        "calendar_rows": len(selected_calendar),
    }

    manifest_path = destination_path / _MANIFEST
    if destination_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("existing RD-Agent dataset view is incomplete") from exc
        if existing != expected_manifest:
            raise ValueError("existing RD-Agent dataset view has a different cutoff or source")
        for name in ("cn_all.txt", "all.txt"):
            cached_instruments = destination_path / "instruments" / name
            if (
                not cached_instruments.is_file()
                or hashlib.sha256(cached_instruments.read_bytes()).hexdigest()
                != expected_manifest["instruments_sha256"]
            ):
                raise ValueError(
                    "existing RD-Agent dataset view has an invalid cn_all universe"
                )
        return destination_path

    temporary = destination_path.with_name(
        f"{destination_path.name}.building-{uuid.uuid4().hex}"
    )
    try:
        (temporary / "calendars").mkdir(parents=True)
        (temporary / "instruments").mkdir()
        (temporary / "features").mkdir()
        (temporary / "calendars" / "day.txt").write_text(
            "\n".join(selected_calendar) + "\n", encoding="utf-8"
        )
        for path in instruments_path.glob("*.txt"):
            lines = (
                governed_lines
                if path == governed_instruments_path
                else _truncate_instruments(path, effective_cutoff)
            )
            target = temporary / "instruments" / path.name
            if path == governed_instruments_path or path.name == "all.txt":
                target.write_bytes(governed_output)
            else:
                target.write_text(
                    "\n".join(lines) + ("\n" if lines else ""),
                    encoding="utf-8",
                )
        # Upstream factor source generation calls D.instruments() without a
        # market argument. In this sealed RD-Agent-only view, make that default
        # alias identical to the governed cn_all universe used for independent
        # recomputation. The full platform Qlib provider remains unchanged.
        (temporary / "instruments" / "all.txt").write_bytes(governed_output)

        copied = 0
        for path in features_path.rglob("*.day.bin"):
            target = temporary / "features" / path.relative_to(features_path)
            if _copy_truncated_day_feature(path, target, cutoff_index):
                copied += 1
        if copied == 0:
            raise ValueError("RD-Agent Qlib dataset has no usable day feature binaries")
        (temporary / _MANIFEST).write_text(
            json.dumps(expected_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination_path


def _truncate_instruments(path: Path, cutoff: str) -> list[str]:
    cutoff_date = date.fromisoformat(cutoff)
    result: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if len(fields) < 3:
            continue
        start = date.fromisoformat(fields[1])
        end = date.fromisoformat(fields[2])
        if start > cutoff_date:
            continue
        fields[2] = min(end, cutoff_date).isoformat()
        result.append("\t".join(fields))
    return result


def _copy_truncated_day_feature(source: Path, destination: Path, cutoff_index: int) -> bool:
    size = source.stat().st_size
    if size < 8 or size % 4:
        raise ValueError(f"invalid Qlib day feature binary: {source}")
    with source.open("rb") as handle:
        header = handle.read(4)
        start_value = struct.unpack("<f", header)[0]
        if not math.isfinite(start_value) or not start_value.is_integer():
            raise ValueError(f"invalid Qlib day feature start index: {source}")
        start_index = int(start_value)
        available_values = size // 4 - 1
        retained_values = min(available_values, cutoff_index - start_index + 1)
        if retained_values <= 0:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = handle.read(retained_values * 4)
        if len(payload) != retained_values * 4:
            raise ValueError(f"truncated Qlib day feature binary: {source}")
        with destination.open("wb") as output:
            output.write(header)
            output.write(payload)
    return True
