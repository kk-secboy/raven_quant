from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_platform import information_schedule as schedule_module
from quant_platform.announcement_nlp import FACTOR_NAME
from quant_platform.api import ScheduleCreateRequest
from quant_platform.information_schedule import (
    latest_verified_snapshot,
    normalize_information_factor_refresh_payload,
    normalize_information_schedule_payload,
)
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database


def _snapshot(
    data_root: Path,
    name: str,
    *,
    end: str,
    ok: bool,
    errors: list[str] | None = None,
) -> None:
    root = data_root / "snapshots" / name
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"name": name, "start_date": "2008-01-01", "end_date": end}),
        encoding="utf-8",
    )
    (root / "verification.json").write_text(
        json.dumps({"ok": ok, "errors": errors or []}), encoding="utf-8"
    )


def test_information_schedule_is_raw_only_and_bounded_by_default() -> None:
    payload = normalize_information_schedule_payload({})

    assert payload["regulatory_only"] is True
    assert payload["enable_nlp"] is False
    assert payload["include_corpus_nlp"] is False
    assert payload["include_event_labels"] is False
    assert payload["include_factor_evaluation"] is False
    assert payload["factor_evaluation"] is None
    assert payload["announcement_nlp_limit"] == 500
    assert payload["corpus_nlp_limit"] == 500


def test_information_factor_refresh_requires_sources_weekday_and_frozen_evaluation() -> None:
    evaluation = {
        "dataset": "qlib-frozen",
        "periods": {
            "train_start": "2010-01-01",
            "train_end": "2019-12-31",
            "valid_start": "2020-01-01",
            "valid_end": "2022-12-31",
            "test_start": "2023-01-09",
            "test_end": "2026-08-03",
        },
    }
    payload = normalize_information_factor_refresh_payload(
        {
            "sources": ["news_flash", "report_rc", "news_flash"],
            "weekday": 4,
            "factor_evaluation": evaluation,
        }
    )
    assert payload["sources"] == ["news_flash", "report_rc"]
    assert payload["weekday"] == 4
    assert payload["factor_evaluation"]["dataset"] == "qlib-frozen"

    with pytest.raises(ValueError, match="non-empty"):
        normalize_information_factor_refresh_payload(
            {"sources": [], "factor_evaluation": evaluation}
        )
    with pytest.raises(ValueError, match="between 0 and 4"):
        normalize_information_factor_refresh_payload(
            {"weekday": 6, "factor_evaluation": evaluation}
        )
    with pytest.raises(ValueError, match="unsupported values"):
        normalize_information_factor_refresh_payload(
            {"sources": ["social_media"], "factor_evaluation": evaluation}
        )

    request = ScheduleCreateRequest(
        name="weekly structured information factors",
        kind="information_factor_refresh",
        payload={
            "sources": ["report_rc"],
            "weekday": 4,
            "factor_evaluation": evaluation,
        },
    )
    assert request.kind == "information_factor_refresh"


def test_structured_information_steps_are_allowed_in_durable_chain(
    tmp_path: Path,
) -> None:
    class FakeStore:
        def create(self, kind, payload, log_path, *, idempotency_key):
            return {
                "id": idempotency_key,
                "kind": kind,
                "payload": payload,
                "log_path": str(log_path),
            }

    worker = object.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace(data_root=tmp_path / "data")
    worker.store = FakeStore()
    worker.notify = lambda: None
    kinds = [
        "report_rc_factor_register",
        "major_news_mentions",
        "major_news_mentions_factor_register",
        "news_flash_factors",
        "news_flash_factor_register",
        "information_factor_evaluate",
    ]
    current = {
        "kind": "report_rc_factors",
        "payload": {
            "pipeline_id": "structured-information-fixture",
            "profile": "information_factor_refresh",
            "start": "2010-01-01",
            "end": "2026-08-03",
            "snapshot_name": "information-factors-20260803",
            "pipeline_steps": [
                {"kind": kind, "payload": {}} for kind in kinds
            ],
            "pipeline_next_index": 0,
        },
    }
    created = []
    for kind in kinds:
        current = worker._queue_data_pipeline_successor(current)
        assert current["kind"] == kind
        created.append(current)
    assert worker._has_data_pipeline_successor(created[-1]) is False


def test_information_schedule_requires_explicit_nlp_and_finite_limits() -> None:
    with pytest.raises(ValueError, match="enable_nlp=true"):
        normalize_information_schedule_payload({"include_corpus_nlp": True})
    with pytest.raises(ValueError, match="between 1 and 10000"):
        normalize_information_schedule_payload({"enable_nlp": True, "announcement_nlp_limit": 0})

    payload = normalize_information_schedule_payload(
        {
            "enable_nlp": True,
            "announcement_nlp_limit": 250,
            "corpus_nlp_limit": 300,
            "corpus_datasets": ["major_news", "irm_qa_sz", "major_news"],
        }
    )
    assert payload["include_corpus_nlp"] is True
    assert payload["include_event_labels"] is True
    assert payload["corpus_datasets"] == ["irm_qa_sz", "major_news"]


def test_information_schedule_requires_explicit_frozen_factor_evaluation() -> None:
    with pytest.raises(ValueError, match="include_factor_evaluation=true"):
        normalize_information_schedule_payload(
            {"enable_nlp": True, "factor_evaluation": {"dataset": "qlib"}}
        )
    with pytest.raises(ValueError, match="periods must contain exactly"):
        normalize_information_schedule_payload(
            {
                "enable_nlp": True,
                "include_factor_evaluation": True,
                "factor_evaluation": {"dataset": "qlib-fixture", "periods": {}},
            }
        )
    with pytest.raises(ValueError, match="purge/embargo"):
        normalize_information_schedule_payload(
            {
                "enable_nlp": True,
                "include_factor_evaluation": True,
                "factor_evaluation": {
                    "dataset": "qlib-fixture",
                    "periods": {
                        "train_start": "2020-01-01",
                        "train_end": "2021-12-31",
                        "valid_start": "2022-01-01",
                        "valid_end": "2023-12-31",
                        "test_start": "2024-01-05",
                        "test_end": "2025-12-31",
                    },
                },
            }
        )

    payload = normalize_information_schedule_payload(
        {
            "enable_nlp": True,
            "include_factor_evaluation": True,
            "factor_evaluation": {
                "dataset": "qlib-fixture",
                "periods": {
                    "train_start": "2020-01-01",
                    "train_end": "2021-12-31",
                    "valid_start": "2022-01-01",
                    "valid_end": "2023-12-31",
                    "test_start": "2024-01-08",
                    "test_end": "2025-12-31",
                },
                "universe": "cn_all",
                "benchmark": "sh000300",
            },
        }
    )

    assert payload["include_factor_evaluation"] is True
    assert payload["factor_evaluation"]["dataset"] == "qlib-fixture"
    assert payload["factor_evaluation"]["benchmark"] == "SH000300"


def test_latest_verified_snapshot_ignores_failed_invalid_and_future_candidates(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _snapshot(data_root, "verified-old", end="2025-01-02", ok=True)
    _snapshot(data_root, "verified-current", end="2025-01-03", ok=True)
    _snapshot(data_root, "failed-newer", end="2025-01-04", ok=False)
    _snapshot(data_root, "errors-newer", end="2025-01-04", ok=True, errors=["bad"])
    _snapshot(data_root, "future", end="2025-01-06", ok=True)
    invalid = data_root / "snapshots" / "invalid"
    invalid.mkdir(parents=True)
    (invalid / "manifest.json").write_text("not json", encoding="utf-8")

    assert latest_verified_snapshot(data_root, as_of=date(2025, 1, 3)) == "verified-current"


def test_information_evaluation_resolves_only_pinned_reproducible_daily_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "data" / "qlib" / "qlib-frozen"
    calendar_path = dataset_path / "calendars" / "day.txt"
    calendar_path.parent.mkdir(parents=True)
    start = date(2020, 1, 1)
    calendar_path.write_text(
        "\n".join((start + timedelta(days=offset)).isoformat() for offset in range(2200)),
        encoding="utf-8",
    )
    dataset = {
        "name": "qlib-frozen",
        "path": str(dataset_path),
        "ready": True,
        "reproducible": True,
        "frequency": "day",
        "start_date": "2020-01-01",
        "end_date": "2026-01-01",
        "provenance": {"dataset_identity_sha256": "a" * 64},
    }
    monkeypatch.setattr(schedule_module, "list_qlib_datasets", lambda _root: [dataset])
    monkeypatch.setattr(
        schedule_module, "require_daily_qlib_contract", lambda _provenance: None
    )
    evaluation = normalize_information_schedule_payload(
        {
            "enable_nlp": True,
            "include_factor_evaluation": True,
            "factor_evaluation": {
                "dataset": "qlib-frozen",
                "periods": {
                    "train_start": "2020-01-01",
                    "train_end": "2021-12-31",
                    "valid_start": "2022-01-01",
                    "valid_end": "2023-12-31",
                    "test_start": "2024-01-08",
                    "test_end": "2025-12-31",
                },
            },
        }
    )["factor_evaluation"]

    assert (
        schedule_module.resolve_information_evaluation_dataset(
            tmp_path / "data", evaluation
        )["name"]
        == "qlib-frozen"
    )


def test_information_steps_form_a_durable_idempotent_successor_chain(
    tmp_path: Path,
) -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.by_key: dict[str, dict] = {}

        def create(
            self,
            kind: str,
            payload: dict,
            log_path: Path,
            *,
            idempotency_key: str,
        ) -> dict:
            if idempotency_key not in self.by_key:
                self.by_key[idempotency_key] = {
                    "id": f"job-{len(self.by_key) + 1}",
                    "kind": kind,
                    "payload": payload,
                    "log_path": str(log_path),
                }
            return self.by_key[idempotency_key]

    worker = object.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace(data_root=tmp_path / "data")
    worker.store = FakeStore()
    worker.notify = lambda: None
    steps = [
        {
            "kind": "announcement_nlp",
            "payload": {
                "start": "2025-01-01",
                "end": "2025-01-02",
                "categories": ["regulatory_letter"],
                "limit": 100,
            },
        },
        {
            "kind": "corpus_nlp",
            "payload": {
                "start": "2025-01-01",
                "end": "2025-01-02",
                "datasets": ["major_news"],
                "limit": 100,
            },
        },
        {
            "kind": "event_market_response",
            "payload": {
                "snapshot_name": "cn-verified",
                "horizons": [1, 3, 5, 20],
                "benchmark_code": "000300.SH",
            },
        },
        {
            "kind": "information_factor_evaluate",
            "payload": {
                "dataset": "qlib-fixture",
                "dataset_path": str(tmp_path / "data" / "qlib" / "qlib-fixture"),
                "dataset_identity_sha256": "a" * 64,
                "periods": {
                    "train_start": "2020-01-01",
                    "train_end": "2021-12-31",
                    "valid_start": "2022-01-01",
                    "valid_end": "2023-12-31",
                    "test_start": "2024-01-08",
                    "test_end": "2025-01-02",
                },
                "universe": "cn_all",
                "benchmark": "SH000300",
                "factor_names": [FACTOR_NAME],
            },
        },
    ]
    current = {
        "kind": "cninfo_announcements_download",
        "payload": {
            "pipeline_id": "information-fixture",
            "profile": "information",
            "start": "2024-12-30",
            "end": "2025-01-02",
            "snapshot_name": "information-20250102",
            "pipeline_steps": steps,
            "pipeline_next_index": 0,
        },
    }

    created = []
    expected_kinds = (
        "announcement_nlp",
        "announcement_factor_register",
        "corpus_nlp",
        "corpus_factor_register",
        "event_market_response",
        "information_factor_evaluate",
    )
    steps.insert(
        1,
        {
            "kind": "announcement_factor_register",
            "payload": {"factor_name": "all", "actor": "information-scheduler"},
        },
    )
    steps.insert(
        3,
        {
            "kind": "corpus_factor_register",
            "payload": {"factor_name": "all", "actor": "information-scheduler"},
        },
    )
    for expected in expected_kinds:
        successor = worker._queue_data_pipeline_successor(current)
        assert successor["kind"] == expected
        assert worker._queue_data_pipeline_successor(current)["id"] == successor["id"]
        created.append(successor)
        current = successor

    assert created[-2]["payload"]["snapshot_name"] == "cn-verified"
    assert created[-1]["payload"]["dataset_identity_sha256"] == "a" * 64
    assert worker._has_data_pipeline_successor(created[-1]) is False

    announcement_command, result_path, environment = worker._command(created[1])
    assert announcement_command[-5:] == [
        "register-announcement-factor",
        "--factor-name",
        "all",
        "--actor",
        "information-scheduler",
    ]
    assert result_path is None and environment == {}
