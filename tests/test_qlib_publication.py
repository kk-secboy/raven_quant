from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from quant_data import cli
from quant_data.execution_contract import (
    DAILY_QLIB_FIELD_CONTRACT_VERSION,
    INDEX_VOLUME_POLICY,
    QLIB_DAILY_AMOUNT_UNIT,
    QLIB_DAILY_VOLUME_UNIT,
    TUSHARE_DAILY_AMOUNT_UNIT,
    TUSHARE_DAILY_VOLUME_UNIT,
    TUSHARE_HAND_SIZE,
)
from quant_data.history_bounds import GOVERNED_DAILY_STOCK_SCOPE_VERSION
from quant_data.qlib_builder import build_qlib_output_manifest
from quant_data.qlib_publication import (
    build_qlib_publication_receipt,
    validate_qlib_publication_receipt,
)
from quant_data.snapshot_lineage import canonical_sha256
from quant_data.universe import (
    GOVERNED_DAILY_ETF_WHITELIST,
    governed_daily_etf_whitelist_contract,
)
from quant_platform.job_commands.data import data_qlib_command
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database


def _write_object(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value).encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _publication(data_root: Path, *, recovered: bool) -> tuple[str, Path]:
    requested = "cn-20080101-20260904"
    source_sha = _write_object(
        data_root / "snapshots" / requested / "manifest.json",
        {"name": requested, "lineage_id": "a" * 64},
    )
    published = f"{requested}-ingested-fix" if recovered else requested
    snapshot_sha = source_sha
    if recovered:
        recovery = {
            "source_snapshot": requested,
            "source_snapshot_manifest_sha256": source_sha,
        }
        recovery["receipt_sha256"] = canonical_sha256(recovery)
        snapshot_sha = _write_object(
            data_root / "snapshots" / published / "manifest.json",
            {
                "name": published,
                "lineage_id": "a" * 64,
                "ingested_at_recovery": recovery,
                "lineage_contract": {"kind": "qlib_daily_source_ingested_at_successor"},
            },
        )
    output = data_root / "qlib" / published
    for relative, content in (
        ("calendars/day.txt", b"2026-09-04\n"),
        ("instruments/cn_all.txt", b"SH600000\t2026-09-04\t2026-09-04\n"),
        ("features/sh600000/close.day.bin", b"sealed-feature"),
    ):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    _write_object(
        output / "metadata" / "provenance.json",
        {
            "frequency": "day",
            "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
            "source_volume_unit": TUSHARE_DAILY_VOLUME_UNIT,
            "qlib_volume_unit": QLIB_DAILY_VOLUME_UNIT,
            "source_amount_unit": TUSHARE_DAILY_AMOUNT_UNIT,
            "qlib_amount_unit": QLIB_DAILY_AMOUNT_UNIT,
            "source_hand_size": TUSHARE_HAND_SIZE,
            "index_volume_policy": INDEX_VOLUME_POLICY,
            "lineage_verified": True,
            "governed_etf_whitelist": {
                **governed_daily_etf_whitelist_contract(),
                "status": "ready",
                "included_symbols": list(GOVERNED_DAILY_ETF_WHITELIST),
                "missing_symbols": [],
            },
            "execution_controls": {"scope_version": GOVERNED_DAILY_STOCK_SCOPE_VERSION},
            "snapshot_name": published,
            "snapshot_manifest_sha256": snapshot_sha,
            "source_lineage_id": "a" * 64,
            "dataset_identity_sha256": "b" * 64,
            "dataset_lineage_id": "c" * 64,
            "output_manifest": build_qlib_output_manifest(output),
        },
    )
    return requested, output


def _worker(data_root: Path):
    worker = object.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace(data_root=data_root)
    worker.notify = lambda: None
    created = []

    def create(kind, payload, log_path, **kwargs):
        job = {"kind": kind, "payload": payload, "log_path": str(log_path), **kwargs}
        created.append(job)
        return job

    worker.store = SimpleNamespace(create=create)
    return worker, created


@pytest.mark.parametrize("recovered", [False, True])
@pytest.mark.parametrize("explicit_steps", [False, True])
def test_published_qlib_receipt_binds_pipeline_baseline_to_actual_sealed_output(
    tmp_path: Path, monkeypatch, recovered: bool, explicit_steps: bool
) -> None:
    requested, output = _publication(tmp_path, recovered=recovered)
    worker, created = _worker(tmp_path)
    payload = {
        "pipeline_id": "daily-pipeline",
        "profile": "full",
        "snapshot_name": requested,
        "start": "2008-01-01",
        "end": "2026-09-04",
    }
    if explicit_steps:
        payload.update({"pipeline_steps": [{"kind": "qlib_baseline"}], "pipeline_next_index": 0})
    job = {"id": "qlib-job", "kind": "data_qlib", "payload": payload}
    command, result_path, _ = data_qlib_command(worker, job)
    assert command[command.index("--result") + 1] == str(result_path)
    monkeypatch.setattr(cli, "load_context", lambda **_: SimpleNamespace(
        settings=worker.settings,
        storage=SimpleNamespace(snapshots_root=tmp_path / "snapshots"),
    ))
    monkeypatch.setattr(cli, "_build_qlib", lambda *args, **kwargs: output)
    cli.build_qlib_command(snapshot_name=requested, result_path=result_path)
    receipt = json.loads(result_path.read_text(encoding="utf-8"))
    successor = worker._queue_data_pipeline_successor(job, result=receipt)

    assert len(created) == 1
    assert successor["payload"]["dataset"] == output.name
    assert successor["payload"]["dataset_path"] == str(output.resolve())
    assert successor["payload"]["dataset_identity_sha256"] == "b" * 64
    assert successor["payload"]["qlib_publication"] == receipt
    assert successor["payload"]["snapshot_name"] == requested
    assert successor["payload"]["pipeline_snapshot_name"] == requested
    assert "qlib_publication" not in job["payload"]


@pytest.mark.parametrize("damage", ["feature", "identity", "source", "receipt"])
def test_changed_publication_cannot_enqueue_a_baseline(tmp_path: Path, damage: str) -> None:
    requested, output = _publication(tmp_path, recovered=True)
    receipt = build_qlib_publication_receipt(tmp_path, requested, output)
    if damage == "feature":
        (output / "features/sh600000/close.day.bin").write_bytes(b"changed-feature")
    elif damage == "identity":
        path = output / "metadata/provenance.json"
        provenance = json.loads(path.read_text())
        provenance["dataset_identity_sha256"] = "d" * 64
        _write_object(path, provenance)
    elif damage == "source":
        _write_object(tmp_path / "snapshots" / requested / "manifest.json", {"changed": True})
    else:
        receipt["requested_snapshot_manifest_sha256"] = "d" * 64
    worker, created = _worker(tmp_path)
    with pytest.raises(ValueError):
        worker._queue_data_pipeline_successor(
            {"kind": "data_qlib", "payload": {"snapshot_name": requested}}, result=receipt
        )
    assert created == []


def test_missing_receipt_never_guesses_a_recovery_directory(tmp_path: Path) -> None:
    requested, _ = _publication(tmp_path, recovered=True)
    worker, created = _worker(tmp_path)
    with pytest.raises(ValueError, match="requires a matching publication receipt"):
        worker._queue_data_pipeline_successor(
            {"kind": "data_qlib", "payload": {"snapshot_name": requested}}
        )
    assert created == []


def test_unrelated_or_external_publication_is_rejected(tmp_path: Path) -> None:
    requested, output = _publication(tmp_path, recovered=True)
    receipt = build_qlib_publication_receipt(tmp_path, requested, output)
    other_requested, other_output = _publication(tmp_path / "other-root", recovered=False)
    with pytest.raises(ValueError, match="outside its governed root"):
        validate_qlib_publication_receipt(
            tmp_path, requested, {**receipt, "dataset_path": str(other_output)}
        )
    _write_object(tmp_path / "snapshots" / "unrelated" / "manifest.json", {"different": True})
    with pytest.raises(ValueError, match="not a proven recovery"):
        build_qlib_publication_receipt(tmp_path, "unrelated", output)
    assert other_requested == requested


def test_staging_build_cannot_issue_a_publication_receipt(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter, match="requires a complete Qlib build"):
        cli.build_qlib_command(staging_only=True, result_path=tmp_path / "result.json")
