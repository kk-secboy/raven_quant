from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quant_platform import forward_only_rehabilitation as rehabilitation
from quant_platform.forward_only_rehabilitation import (
    EVIDENCE_MODE_REPLAY,
    REPLAY_AUTHORITY,
    SOURCE_BACKTEST_ID,
    SOURCE_DATASET,
    SOURCE_EXECUTION_CONTRACT_HASH,
    SOURCE_PERIODS,
    SOURCE_RULES_SHA256,
    SOURCE_VERSION_ID,
    audit_incomplete_family_artifacts,
    rehabilitation_forward_thresholds,
    require_replay_config,
)
from quant_platform.strategy_recipes import get_strategy_recipe
from quant_platform.strategy_store import StrategyStore
from quant_platform.transparent_baseline_bootstrap import (
    TransparentBaselineBootstrapService,
    _build_forward_only_rehabilitation_plan,
)
from quant_platform.transparent_baseline_lockbox import (
    BOOTSTRAP_CONFIG_KEY,
    LOCKBOX_CONFIG_KEY,
)
from quant_platform.transparent_baseline_runner import (
    FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION,
    FORWARD_ONLY_REHABILITATION_TARGET_RUNNER_SHA256,
    FORWARD_ONLY_REHABILITATION_TARGET_RUNTIME_BUNDLE_SHA256,
    TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION,
    TRANSPARENT_BASELINE_JOB_RUNNER_FIELD,
    TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
    WORKER_RUNTIME_IMAGE_DIGEST_ENV,
)

pytestmark = pytest.mark.no_database

_WORKER_IMAGE = "sha256:" + "d" * 64


def _source_version() -> dict:
    return {
        "id": SOURCE_VERSION_ID,
        "strategy_rules_sha256": SOURCE_RULES_SHA256,
        "execution_contract_hash": SOURCE_EXECUTION_CONTRACT_HASH,
        "horizon_profile": "short_1_5d",
        "config": {
            "recipe_id": "short_relative_strength",
            "recipe_version": TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION,
        },
    }


def _build_plan(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setenv(WORKER_RUNTIME_IMAGE_DIGEST_ENV, _WORKER_IMAGE)
    monkeypatch.setattr(
        "quant_platform.transparent_baseline_bootstrap._sha256_file",
        lambda _path: FORWARD_ONLY_REHABILITATION_TARGET_RUNNER_SHA256,
    )
    monkeypatch.setattr(
        "quant_platform.transparent_baseline_bootstrap.position_risk_bundle_sha256",
        lambda _root: FORWARD_ONLY_REHABILITATION_TARGET_RUNTIME_BUNDLE_SHA256,
    )
    return _build_forward_only_rehabilitation_plan(
        source_version=_source_version(),
        dataset={"name": SOURCE_DATASET},
        consumed_oos_vintage_id="vintage-consumed-once",
    )


def test_v18_plan_is_an_opened_descriptive_replay_without_a_new_lockbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _build_plan(monkeypatch)
    config = plan["config"]

    assert config["recipe_version"] == FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
    assert config["evidence_mode"] == EVIDENCE_MODE_REPLAY
    assert LOCKBOX_CONFIG_KEY not in config
    assert plan["formal_periods"] == SOURCE_PERIODS
    assert config["forward_only_rehabilitation"]["consumed_oos_vintage_id"] == (
        "vintage-consumed-once"
    )
    window = config[BOOTSTRAP_CONFIG_KEY]["research_window_contract"]
    assert window["historical_replay_opened"] is True
    assert window["consumed_oos_replayed"] is True
    assert window["sealed_final_oos"] is False
    assert window["unseen_oos"] is False
    assert window["authority"] == REPLAY_AUTHORITY
    assert require_replay_config(config)["replay_periods"] == SOURCE_PERIODS

    # The ordinary scheduler remains on v17.  Only the dedicated one-shot
    # emits v18, so swing/long cannot accidentally acquire replay authority.
    assert (
        get_strategy_recipe("short_relative_strength")["version"]
        == TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION
    )
    assert (
        get_strategy_recipe("swing_trend")["version"]
        == TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION
    )
    assert (
        get_strategy_recipe("long_quality_value")["version"]
        == TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION
    )

    tampered = deepcopy(config)
    tampered[LOCKBOX_CONFIG_KEY] = {"forbidden": True}
    with pytest.raises(ValueError, match="exact forward-only rehabilitation"):
        require_replay_config(tampered)


def test_replay_config_requires_the_sealed_runner_bundle_and_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(_build_plan(monkeypatch)["config"])
    config[BOOTSTRAP_CONFIG_KEY][TRANSPARENT_BASELINE_RUNNER_FIELD] = "f" * 64
    with pytest.raises(ValueError, match="exact forward-only rehabilitation"):
        require_replay_config(config)


def _source_attempt_root(data_root: Path) -> Path:
    return (
        data_root
        / "artifacts"
        / "formal-backtest-recoveries"
        / SOURCE_BACKTEST_ID
        / "attempt-2"
    )


def test_incomplete_family_receipt_hashes_only_partial_baseline_artifacts(
    tmp_path: Path,
) -> None:
    root = _source_attempt_root(tmp_path)
    (root / "baseline" / "raw").mkdir(parents=True)
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    (root / "baseline" / "raw" / "relative_strength.parquet").write_bytes(b"partial")

    evidence = audit_incomplete_family_artifacts(
        data_root=tmp_path,
        observed_at=datetime(2026, 8, 31, tzinfo=UTC),
    )

    missing = {item["artifact_kind"] for item in evidence if item["status"] == "missing"}
    assert missing == {
        "trial_daily_returns_matrix",
        "trial_score_grid_matrix",
        "trial_candidate_manifest_matrix",
    }
    partial = [item for item in evidence if item["status"] == "partial"]
    assert len(partial) == 2
    assert all(len(str(item["sha256"])) == 64 for item in partial)

    (root / "result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="conservative fallback is forbidden"):
        audit_incomplete_family_artifacts(
            data_root=tmp_path,
            observed_at=datetime(2026, 8, 31, tzinfo=UTC),
        )


def test_source_artifact_audit_rejects_escape_and_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rehabilitation,
        "SOURCE_ATTEMPT_ARTIFACT_RELATIVE_PATH",
        "../outside-attempt",
    )
    with pytest.raises(ValueError, match="escapes the governed data root"):
        audit_incomplete_family_artifacts(
            data_root=tmp_path,
            observed_at=datetime(2026, 8, 31, tzinfo=UTC),
        )

    monkeypatch.setattr(
        rehabilitation,
        "SOURCE_ATTEMPT_ARTIFACT_RELATIVE_PATH",
        (
            "artifacts/formal-backtest-recoveries/"
            f"{SOURCE_BACKTEST_ID}/attempt-2"
        ),
    )
    root = _source_attempt_root(tmp_path)
    root.mkdir(parents=True)
    original = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path.name == "attempt-2" or original(path),
    )
    with pytest.raises(ValueError, match="contains a symlink"):
        audit_incomplete_family_artifacts(
            data_root=tmp_path,
            observed_at=datetime(2026, 8, 31, tzinfo=UTC),
        )


def _job_payload(version: dict, backtest: dict, dataset: dict) -> dict:
    bootstrap = version["config"][BOOTSTRAP_CONFIG_KEY]
    return {
        "backtest_id": backtest["id"],
        "strategy_version_id": version["id"],
        "dataset": dataset["name"],
        "dataset_path": dataset["path"],
        "execution_dataset": None,
        "periods": backtest["periods"],
        TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: bootstrap[
            TRANSPARENT_BASELINE_RUNNER_FIELD
        ],
        TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD: bootstrap[
            TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD
        ],
        TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: bootstrap[
            TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD
        ],
    }


def test_existing_replay_must_have_a_complete_exact_backtest_and_job(
    tmp_path: Path,
) -> None:
    version = {
        "id": "new-v18-version",
        "evidence_mode": EVIDENCE_MODE_REPLAY,
        "config": {
            BOOTSTRAP_CONFIG_KEY: {
                TRANSPARENT_BASELINE_RUNNER_FIELD: "a" * 64,
                TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: "b" * 64,
                TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: _WORKER_IMAGE,
            }
        },
    }
    dataset = {"name": SOURCE_DATASET, "path": "/data/qlib/source"}
    plan = {"formal_periods": SOURCE_PERIODS}

    class MissingBacktestStrategies:
        @staticmethod
        def list_backtests(**_kwargs) -> list[dict]:
            return []

    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        strategies=MissingBacktestStrategies(),
        jobs=object(),
        promotions=object(),
        lockboxes=object(),
    )
    with pytest.raises(ValueError, match="formal backtest is missing"):
        service._ensure_backtest_job(
            plan=plan,
            version=version,
            dataset=dataset,
            allow_create=False,
        )

    incomplete = {
        "id": "new-v18-backtest",
        "dataset": SOURCE_DATASET,
        "execution_dataset": None,
        "periods": SOURCE_PERIODS,
        "evidence_mode": EVIDENCE_MODE_REPLAY,
        "status": "queued",
        "job_id": None,
    }

    class MissingJobStrategies:
        @staticmethod
        def list_backtests(**_kwargs) -> list[dict]:
            return [incomplete]

    service.strategies = MissingJobStrategies()
    with pytest.raises(ValueError, match="attached job is missing"):
        service._ensure_backtest_job(
            plan=plan,
            version=version,
            dataset=dataset,
            allow_create=False,
        )

    complete = {**incomplete, "status": "succeeded", "job_id": "new-v18-job"}
    payload = _job_payload(version, complete, dataset)

    class CompleteStrategies:
        @staticmethod
        def list_backtests(**_kwargs) -> list[dict]:
            return [complete]

    class CompleteJobs:
        create_calls = 0

        @classmethod
        def create(cls, *_args, **_kwargs):
            cls.create_calls += 1
            raise AssertionError("complete replay reuse must not create another job")

        @staticmethod
        def get(job_id: str) -> dict:
            assert job_id == "new-v18-job"
            return {
                "id": job_id,
                "kind": "strategy_backtest",
                "status": "succeeded",
                "payload": payload,
                "idempotency_key": (
                    "transparent-baseline:new-v18-version:new-v18-backtest"
                ),
                "max_attempts": 1,
            }

    service.strategies = CompleteStrategies()
    service.jobs = CompleteJobs()
    result = service._ensure_backtest_job(
        plan=plan,
        version=version,
        dataset=dataset,
        allow_create=False,
    )
    assert result["backtest_action"] == "reused"
    assert result["job_action"] == "reused"
    assert result["job"]["id"] == "new-v18-job"
    assert CompleteJobs.create_calls == 0


def test_replay_uses_dedicated_admission_and_one_stricter_forward_gate(
    tmp_path: Path,
) -> None:
    class Strategies:
        admitted = 0
        approved = 0

        def __init__(self) -> None:
            self.state = {
                "id": "new-v18-version",
                "status": "draft",
                "promotion_stage": None,
                "evidence_mode": EVIDENCE_MODE_REPLAY,
            }

        def get_version(self, _version_id: str) -> dict:
            return dict(self.state)

        def admit_forward_only_rehabilitation(self, *_args, **_kwargs) -> None:
            self.admitted += 1
            self.state.update(status="approved", promotion_stage="paper")

        def approve(self, *_args, **_kwargs) -> None:
            self.approved += 1
            raise AssertionError("ordinary approve must never admit a replay")

    class Promotions:
        @staticmethod
        def prepare_paper_stage(version_id: str, *, actor: str) -> dict:
            return {"strategy_version_id": version_id, "actor": actor}

    strategies = Strategies()
    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        strategies=strategies,
        jobs=object(),
        promotions=Promotions(),
        lockboxes=object(),
    )
    result = service._advance_paper(
        version_id="new-v18-version",
        backtest={"status": "succeeded"},
        actor="system:test",
    )
    assert result["state"] == "paper_validating"
    assert strategies.admitted == 1
    assert strategies.approved == 0

    thresholds = rehabilitation_forward_thresholds(
        {
            "min_forward_calendar_days": 90,
            "min_forward_trading_days": 60,
            "min_decision_batches": 60,
            "min_closed_round_trips": 30,
        }
    )
    assert thresholds["min_forward_calendar_days"] == 365
    assert thresholds["min_forward_trading_days"] == 252


def test_ordinary_strategy_approve_rejects_replay_before_any_write() -> None:
    store = object.__new__(StrategyStore)
    store.get_version = lambda _version_id: {"evidence_mode": EVIDENCE_MODE_REPLAY}

    with pytest.raises(ValueError, match="dedicated forward-only admission"):
        store.approve(
            "new-v18-version",
            actor="operator",
            reason="This replay cannot use ordinary approval.",
        )
