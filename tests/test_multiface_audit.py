from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from quant_platform import multiface_audit as audit

pytestmark = pytest.mark.no_database


class FakeLedger:
    def __init__(self) -> None:
        self.candidates: list[dict[str, Any]] = []

    def find_candidate(self, *, name: str, values_sha256: str) -> dict[str, Any] | None:
        return next(
            (
                item
                for item in self.candidates
                if item["name"] == name and item["values_sha256"] == values_sha256
            ),
            None,
        )

    def list_candidates(
        self, *, run_id: str | None = None, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        rows = self.candidates
        if status:
            rows = [item for item in rows if item.get("status") == status]
        return rows[:limit]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _seed_qlib(data_root: Path, *, version: int = 5) -> tuple[str, str]:
    dataset = "cn-audit"
    identity = "a" * 64
    contract = {
        "version": version,
        "daily_fields": ["turnover_rate", "pe_ttm"],
        "fundamental_fields": {"fina_indicator": {"roe": "fund_roe"}},
        "capital_flow_fields": list(audit.CAPITAL_FLOW_FIELDS) if version >= 5 else [],
        "missing_fundamental_fields": {},
        "all_null_fundamental_fields": {},
        "missing_capital_flow_fields": {},
        "all_null_capital_flow_fields": {},
    }
    _write_json(
        data_root / "qlib" / dataset / "metadata" / "provenance.json",
        {
            "dataset_identity_sha256": identity,
            "dataset_lineage_id": "b" * 64,
            "lineage_verified": True,
            "snapshot_name": "snapshot-audit",
            "source_start_date": "2008-01-01",
            "source_end_date": "2026-08-03",
            "fields": list(audit.TECHNICAL_FIELDS),
            "research_features": contract,
        },
    )
    return dataset, identity


def _seed_factors(data_root: Path, ledger: FakeLedger, identity: str) -> None:
    for index, (name, directory) in enumerate(audit._factor_directories(data_root).items()):
        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / f"{name}.parquet"
        artifact.write_bytes(f"factor:{name}".encode())
        values_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        _write_json(
            directory / f"{name}.json",
            {
                "factor": name,
                "artifact": artifact.name,
                "sha256": values_sha256,
                "rows": 25,
                "availability_policy": {name: "available only after source publication"},
                "source": {"dataset": "governed-test-source"},
            },
        )
        ledger.candidates.append(
            {
                "id": f"candidate-{index}",
                "name": name,
                "status": "rejected",
                "values_path": str(artifact),
                "values_sha256": values_sha256,
                "variables": {"source": {"dataset": "governed-test-source"}},
                # Economic rejection is still a valid, honest evaluation.
                "latest_evaluation": {
                    "id": f"evaluation-{index}",
                    "dataset_identity_sha256": identity,
                    "gate_status": "failed",
                    "gate_reasons": ["no stable alpha"],
                    "metrics": {
                        "rolling_walk_forward": {
                            "status": "completed",
                            "passed": False,
                            "fold_count": 3,
                            "purge_days": 5,
                            "embargo_days": 5,
                            "uses_final_test_data": False,
                        }
                    },
                    "recompute_evidence": {
                        "config": {"require_rolling_walk_forward": True}
                    },
                },
            }
        )


def _seed_labels(data_root: Path) -> Path:
    labels_dir = data_root / "announcements" / "nlp" / "labels" / "snapshot-audit"
    artifact = labels_dir / "event_market_response_labels.parquet"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"training-labels")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    _write_json(
        labels_dir / "event_market_response_labels.json",
        {
            "schema_version": "event-market-response.v1",
            "role": "training_label_only",
            "artifact": artifact.name,
            "sha256": digest,
            "rows": 10,
            "horizons": [1, 3, 5, 20],
            "forbidden_consumers": list(audit.FORBIDDEN_LABEL_CONSUMERS),
            "source": {"snapshot_name": "snapshot-audit"},
        },
    )
    return artifact


def _ready_fixture(tmp_path: Path) -> tuple[str, FakeLedger, Path]:
    dataset, identity = _seed_qlib(tmp_path)
    ledger = FakeLedger()
    _seed_factors(tmp_path, ledger, identity)
    labels = _seed_labels(tmp_path)
    return dataset, ledger, labels


def test_multiface_audit_accepts_evaluated_but_unprofitable_factors(tmp_path: Path) -> None:
    dataset, ledger, _ = _ready_fixture(tmp_path)

    report = audit.audit_multiface_readiness(
        tmp_path,
        dataset=dataset,
        ledger=ledger,
        now=datetime(2026, 8, 8, tzinfo=UTC),
    )

    assert report["ok"] is True
    assert all(item["ready"] for item in report["faces"].values())
    assert all(
        item["evaluation"]["gate_status"] == "failed"
        for item in report["information_factors"]
    )
    assert report["semantics"]["ready_does_not_mean_profitable"] is True
    assert len(report["report_sha256"]) == 64

    path = audit.write_multiface_report(tmp_path, report)
    assert path.is_file()
    latest = json.loads(
        (tmp_path / "verification" / "multiface-latest.json").read_text(encoding="utf-8")
    )
    assert latest["report_sha256"] == report["report_sha256"]


def test_obsolete_qlib_contract_keeps_technical_ready_but_blocks_capital(
    tmp_path: Path,
) -> None:
    dataset, identity = _seed_qlib(tmp_path, version=4)
    ledger = FakeLedger()
    _seed_factors(tmp_path, ledger, identity)
    _seed_labels(tmp_path)

    report = audit.audit_multiface_readiness(tmp_path, dataset=dataset, ledger=ledger)

    assert report["ok"] is False
    assert report["faces"]["technical"]["ready"] is True
    assert report["faces"]["capital_flow"]["ready"] is False
    assert "obsolete" in " ".join(report["qlib"]["errors"])
    assert set(report["qlib"]["capital_flow"]["missing"]) == set(
        audit.CAPITAL_FLOW_FIELDS
    )


def test_market_response_label_leak_fails_closed(tmp_path: Path) -> None:
    dataset, ledger, labels = _ready_fixture(tmp_path)
    ledger.candidates.append(
        {
            "id": "leaked-label",
            "name": "market_recognition_20d",
            "status": "awaiting_evaluation",
            "values_path": str(labels),
            "values_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
            "variables": {"role": "training_label_only"},
            "latest_evaluation": None,
        }
    )

    report = audit.audit_multiface_readiness(tmp_path, dataset=dataset, ledger=ledger)

    assert report["ok"] is False
    assert report["faces"]["market_recognition"]["ready"] is False
    assert report["market_recognition"]["candidate_leaks"] == [
        {"id": "leaked-label", "name": "market_recognition_20d"}
    ]


def test_pretest_insufficient_evidence_is_audited_without_relaxing_thresholds(
    tmp_path: Path,
) -> None:
    dataset, ledger, _ = _ready_fixture(tmp_path)
    evaluation = ledger.candidates[0]["latest_evaluation"]
    evaluation["gate_status"] = "insufficient_evidence"
    evaluation["gate_reasons"] = ["independent event days below minimum"]
    evaluation["metrics"] = None

    report = audit.audit_multiface_readiness(tmp_path, dataset=dataset, ledger=ledger)

    assert report["ok"] is True
    first = report["information_factors"][0]
    assert first["evaluation"]["rolling"]["mode"] == "pretest_insufficient_evidence"
    assert first["evaluation"]["rolling"]["strict_config"] is True
