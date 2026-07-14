from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "run_pair_paper_step.py"
    spec = importlib.util.spec_from_file_location("run_pair_paper_step_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pair_paper_script_loads_execution_evidence_and_writes_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    script = _script_module()
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "output"
    manifest = {
        "pair": {"leg_y": "SH510300", "leg_x": "SZ159919"},
        "as_of_date": "2026-07-08",
        "dataset_start": "2024-01-01",
        "state": {
            "status": "active",
            "cash": 5_000_000,
            "nav": 5_000_000,
            "high_water_mark": 5_000_000,
            "position_direction": 0,
            "quantity_y": 0,
            "quantity_x": 0,
            "entry_nav": None,
            "holding_days": 0,
        },
        "config": {},
        "daily_provenance": {
            "dataset_identity_sha256": "a" * 64,
            "snapshot_manifest_sha256": "b" * 64,
        },
        "minute_dataset": {"manifest_sha256": "c" * 64},
        "shortability_dataset": {"source_sha256": "d" * 64},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    reads: list[tuple[str, str, str]] = []

    def read(path: Path, *, legs: set[str], start: str, end: str):
        reads.append((path.name, start, end))
        assert legs == {"SH510300", "SZ159919"}
        return {"path": path.name}

    monkeypatch.setattr(script, "_read_parquet_dataset", read)
    monkeypatch.setattr(script, "_normalize_minute", lambda value, **_kwargs: ("minute", value))
    monkeypatch.setattr(
        script, "_normalize_shortability", lambda value, **_kwargs: ("shortability", value)
    )
    monkeypatch.setattr(script, "_daily_market", lambda *_args, **_kwargs: "daily")
    monkeypatch.setattr(
        script,
        "run_pair_paper_step",
        lambda daily, minute, **kwargs: {
            "status": "ok",
            "daily": daily,
            "minute_loaded": minute[0] == "minute",
            "as_of_date": kwargs["as_of_date"],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pair_paper_step.py",
            "--provider-uri",
            str(tmp_path / "qlib"),
            "--minute-path",
            str(tmp_path / "minute"),
            "--shortability-path",
            str(tmp_path / "shortability"),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
        ],
    )

    script.main()

    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["daily"] == "daily"
    assert result["minute_loaded"] is True
    assert len(reads) == 2
    assert reads[0][1] == "2026-07-08"
    assert result["provenance"]["daily_dataset_identity_sha256"] == "a" * 64
    assert result["provenance"]["daily_snapshot_manifest_sha256"] == "b" * 64
    assert result["provenance"]["minute_snapshot_manifest_sha256"] == "c" * 64
    assert result["provenance"]["shortability_evidence_sha256"] == "d" * 64
    assert len(result["provenance"]["strategy_config_sha256"]) == 64
    assert len(result["provenance"]["execution_manifest_sha256"]) == 64
    assert len(result["provenance"]["pair_engine_sha256"]) == 64
