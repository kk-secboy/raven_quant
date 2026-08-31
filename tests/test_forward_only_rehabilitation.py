from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quant_platform import forward_only_rehabilitation as rehabilitation
from quant_platform import transparent_baseline_bootstrap as bootstrap
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
from quant_platform.strategy_store import (
    StrategyStore,
    _bind_current_transparent_runtime_identity,
)
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


@pytest.mark.parametrize("recipe_id", ["swing_trend", "long_quality_value"])
def test_v18_runtime_is_rejected_for_every_non_short_recipe(recipe_id: str) -> None:
    with pytest.raises(ValueError, match="restricted to the exact short"):
        _bind_current_transparent_runtime_identity(
            {
                "recipe_id": recipe_id,
                "recipe_version": (
                    FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
                ),
                "evidence_mode": "sealed_final_oos",
            }
        )


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
        "artifact_path": str(
            tmp_path / "artifacts" / "backtests" / "new-v18-backtest"
        ),
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


def test_exact_v18_version_resumes_one_frozen_backtest_then_reuses_it(
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
    dataset = {
        "name": SOURCE_DATASET,
        "path": "/data/qlib/source",
        "calendar": ["2018-11-08", "2019-11-20"],
        "dataset_lineage_id": "c" * 64,
        "dataset_identity_sha256": "d" * 64,
    }
    plan = {"formal_periods": SOURCE_PERIODS}

    class Strategies:
        create_calls = 0
        attach_calls = 0
        backtest: dict | None = None

        @classmethod
        def list_backtests(cls, **_kwargs) -> list[dict]:
            return [dict(cls.backtest)] if cls.backtest is not None else []

        @classmethod
        def create_backtest(cls, **kwargs) -> dict:
            cls.create_calls += 1
            assert kwargs["version_id"] == version["id"]
            assert kwargs["periods"] == SOURCE_PERIODS
            assert kwargs["dataset_identity_sha256"] == "d" * 64
            assert not any(key.startswith("capital_oos") for key in kwargs)
            cls.backtest = {
                "id": "new-v18-backtest",
                "dataset": SOURCE_DATASET,
                "execution_dataset": None,
                "periods": SOURCE_PERIODS,
                "artifact_path": str(
                    tmp_path / "artifacts" / "backtests" / "new-v18-backtest"
                ),
                "evidence_mode": EVIDENCE_MODE_REPLAY,
                "status": "queued",
                "job_id": None,
            }
            return dict(cls.backtest)

        @classmethod
        def attach_job_once(cls, backtest_id: str, job_id: str) -> None:
            cls.attach_calls += 1
            assert backtest_id == "new-v18-backtest"
            assert job_id == "new-v18-job"
            assert cls.backtest is not None
            cls.backtest["job_id"] = job_id

        @classmethod
        def get_backtest(cls, _backtest_id: str) -> dict:
            assert cls.backtest is not None
            return dict(cls.backtest)

    class Jobs:
        create_calls = 0
        job: dict | None = None

        @classmethod
        def create(cls, kind: str, payload: dict, _log_path: Path, **kwargs) -> dict:
            cls.create_calls += 1
            assert kind == "strategy_backtest"
            assert kwargs["max_attempts"] == 1
            assert kwargs["idempotency_key"] == (
                "transparent-baseline:new-v18-version:new-v18-backtest"
            )
            cls.job = {
                "id": "new-v18-job",
                "kind": kind,
                "status": "queued",
                "payload": payload,
                "idempotency_key": kwargs["idempotency_key"],
                "max_attempts": kwargs["max_attempts"],
            }
            return dict(cls.job)

        @classmethod
        def get(cls, job_id: str) -> dict:
            assert cls.job is not None and job_id == cls.job["id"]
            return dict(cls.job)

    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        strategies=Strategies(),
        jobs=Jobs(),
        promotions=object(),
        lockboxes=object(),
    )
    created = service._ensure_backtest_job(
        plan=plan,
        version=version,
        dataset=dataset,
        allow_create=True,
    )
    reused = service._ensure_backtest_job(
        plan=plan,
        version=version,
        dataset=dataset,
        allow_create=True,
    )

    assert created["backtest_action"] == "created"
    assert created["job_action"] == "created"
    assert reused["backtest_action"] == "reused"
    assert reused["job_action"] == "reused"
    assert Strategies.create_calls == 1
    assert Jobs.create_calls == 1
    assert Strategies.attach_calls == 1


def test_v18_job_creation_crash_recovers_only_the_same_one_attempt_job(
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
    backtest = {
        "id": "new-v18-backtest",
        "dataset": SOURCE_DATASET,
        "execution_dataset": None,
        "periods": SOURCE_PERIODS,
        "artifact_path": str(
            tmp_path / "artifacts" / "backtests" / "new-v18-backtest"
        ),
        "evidence_mode": EVIDENCE_MODE_REPLAY,
        "status": "queued",
        "job_id": None,
    }

    class Strategies:
        attach_calls = 0

        @staticmethod
        def list_backtests(**_kwargs) -> list[dict]:
            return [dict(backtest)]

        @classmethod
        def attach_job_once(cls, _backtest_id: str, job_id: str) -> None:
            cls.attach_calls += 1
            if cls.attach_calls == 1:
                raise RuntimeError("simulated process crash before attachment")
            backtest["job_id"] = job_id

        @staticmethod
        def get_backtest(_backtest_id: str) -> dict:
            return dict(backtest)

    class Jobs:
        create_calls = 0
        unique_jobs: dict[str, dict] = {}

        @classmethod
        def create(cls, kind: str, payload: dict, _log_path: Path, **kwargs) -> dict:
            cls.create_calls += 1
            key = kwargs["idempotency_key"]
            candidate = {
                "id": "new-v18-job",
                "kind": kind,
                "status": "queued",
                "payload": payload,
                "idempotency_key": key,
                "max_attempts": kwargs["max_attempts"],
            }
            existing = cls.unique_jobs.setdefault(key, candidate)
            assert existing == candidate
            return dict(existing)

        @classmethod
        def get(cls, job_id: str) -> dict:
            [job] = cls.unique_jobs.values()
            assert job_id == job["id"]
            return dict(job)

    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        strategies=Strategies(),
        jobs=Jobs(),
        promotions=object(),
        lockboxes=object(),
    )
    with pytest.raises(RuntimeError, match="simulated process crash"):
        service._ensure_backtest_job(
            plan=plan,
            version=version,
            dataset=dataset,
            allow_create=True,
        )
    recovered = service._ensure_backtest_job(
        plan=plan,
        version=version,
        dataset=dataset,
        allow_create=True,
    )
    reused = service._ensure_backtest_job(
        plan=plan,
        version=version,
        dataset=dataset,
        allow_create=True,
    )

    assert recovered["job"]["id"] == "new-v18-job"
    assert recovered["job_action"] == "created"
    assert reused["job_action"] == "reused"
    assert Jobs.create_calls == 2
    assert len(Jobs.unique_jobs) == 1
    assert next(iter(Jobs.unique_jobs.values()))["max_attempts"] == 1


def test_v18_backtest_creation_race_reuses_one_exact_row(
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
    dataset = {
        "name": SOURCE_DATASET,
        "path": "/data/qlib/source",
        "calendar": ["2018-11-08", "2019-11-20"],
        "dataset_lineage_id": "c" * 64,
        "dataset_identity_sha256": "d" * 64,
    }
    plan = {"formal_periods": SOURCE_PERIODS}
    raced_backtest = {
        "id": "new-v18-backtest",
        "dataset": SOURCE_DATASET,
        "execution_dataset": None,
        "periods": SOURCE_PERIODS,
        "artifact_path": str(
            tmp_path / "artifacts" / "backtests" / "new-v18-backtest"
        ),
        "evidence_mode": EVIDENCE_MODE_REPLAY,
        "status": "queued",
        "job_id": None,
    }

    class Strategies:
        list_calls = 0
        attach_calls = 0

        @classmethod
        def list_backtests(cls, **_kwargs) -> list[dict]:
            cls.list_calls += 1
            return [] if cls.list_calls == 1 else [dict(raced_backtest)]

        @staticmethod
        def create_backtest(**_kwargs) -> dict:
            raise ValueError("concurrent caller already created the formal row")

        @classmethod
        def attach_job_once(cls, _backtest_id: str, job_id: str) -> None:
            cls.attach_calls += 1
            raced_backtest["job_id"] = job_id

        @staticmethod
        def get_backtest(_backtest_id: str) -> dict:
            return dict(raced_backtest)

    class Jobs:
        create_calls = 0

        @classmethod
        def create(cls, kind: str, payload: dict, _log_path: Path, **kwargs) -> dict:
            cls.create_calls += 1
            return {
                "id": "new-v18-job",
                "kind": kind,
                "status": "queued",
                "payload": payload,
                "idempotency_key": kwargs["idempotency_key"],
                "max_attempts": kwargs["max_attempts"],
            }

    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        strategies=Strategies(),
        jobs=Jobs(),
        promotions=object(),
        lockboxes=object(),
    )
    result = service._ensure_backtest_job(
        plan=plan,
        version=version,
        dataset=dataset,
        allow_create=True,
    )

    assert result["backtest_action"] == "reused"
    assert result["backtest"]["id"] == "new-v18-backtest"
    assert Strategies.list_calls == 2
    assert Strategies.attach_calls == 1
    assert Jobs.create_calls == 1


def test_v18_resume_rejects_duplicates_and_tampered_retry_budget(
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
    backtest = {
        "id": "new-v18-backtest",
        "dataset": SOURCE_DATASET,
        "execution_dataset": None,
        "periods": SOURCE_PERIODS,
        "artifact_path": str(
            tmp_path / "artifacts" / "backtests" / "new-v18-backtest"
        ),
        "evidence_mode": EVIDENCE_MODE_REPLAY,
        "status": "queued",
        "job_id": "new-v18-job",
    }
    payload = _job_payload(version, backtest, dataset)

    class DuplicatedStrategies:
        @staticmethod
        def list_backtests(**_kwargs) -> list[dict]:
            return [dict(backtest), {**backtest, "id": "duplicate"}]

    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        strategies=DuplicatedStrategies(),
        jobs=object(),
        promotions=object(),
        lockboxes=object(),
    )
    with pytest.raises(ValueError, match="more than one formal backtest"):
        service._ensure_backtest_job(
            plan=plan,
            version=version,
            dataset=dataset,
            allow_create=True,
        )

    class TamperedArtifactStrategies:
        attach_calls = 0

        @staticmethod
        def list_backtests(**_kwargs) -> list[dict]:
            return [
                {
                    **backtest,
                    "job_id": None,
                    "artifact_path": str(
                        tmp_path / "artifacts" / "formal-backtest-recoveries" / SOURCE_BACKTEST_ID
                    ),
                }
            ]

        @classmethod
        def attach_job_once(cls, *_args) -> None:
            cls.attach_calls += 1

    class NoJobCreation:
        create_calls = 0

        @classmethod
        def create(cls, *_args, **_kwargs):
            cls.create_calls += 1
            raise AssertionError("tampered artifact paths must fail before job creation")

    service.strategies = TamperedArtifactStrategies()
    service.jobs = NoJobCreation()
    with pytest.raises(ValueError, match="backtest differs from the frozen plan"):
        service._ensure_backtest_job(
            plan=plan,
            version=version,
            dataset=dataset,
            allow_create=True,
        )
    assert NoJobCreation.create_calls == 0
    assert TamperedArtifactStrategies.attach_calls == 0

    class ExactStrategies:
        attach_calls = 0

        @staticmethod
        def list_backtests(**_kwargs) -> list[dict]:
            return [dict(backtest)]

        @classmethod
        def attach_job_once(cls, *_args) -> None:
            cls.attach_calls += 1

    class TamperedJobs:
        @staticmethod
        def get(_job_id: str) -> dict:
            return {
                "id": "new-v18-job",
                "kind": "strategy_backtest",
                "status": "queued",
                "payload": payload,
                "idempotency_key": (
                    "transparent-baseline:new-v18-version:new-v18-backtest"
                ),
                "max_attempts": 2,
            }

    service.strategies = ExactStrategies()
    service.jobs = TamperedJobs()
    with pytest.raises(ValueError, match="differs from the frozen plan"):
        service._ensure_backtest_job(
            plan=plan,
            version=version,
            dataset=dataset,
            allow_create=True,
        )
    assert ExactStrategies.attach_calls == 0


def test_strategy_backtest_job_attachment_is_compare_and_set() -> None:
    statements: list = []

    class Result:
        def __init__(self, *, rowcount: int = 0, job_id: str | None = None) -> None:
            self.rowcount = rowcount
            self.job_id = job_id

        def first(self):
            if self.job_id is None:
                return None
            return type("AttachedRow", (), {"job_id": self.job_id})()

    class Connection:
        @staticmethod
        def execute(statement):
            statements.append(statement)
            if statement.is_update:
                return Result(rowcount=0)
            return Result(job_id="concurrent-different-job")

    class Transaction:
        def __enter__(self) -> Connection:
            return Connection()

        def __exit__(self, *_args) -> None:
            return None

    class Engine:
        @staticmethod
        def begin() -> Transaction:
            return Transaction()

    store = object.__new__(StrategyStore)
    store.engine = Engine()
    with pytest.raises(ValueError, match="already attached to a different job"):
        store.attach_job_once("new-v18-backtest", "new-v18-job")

    assert len(statements) == 2
    update_sql = str(statements[0])
    assert "job_id IS NULL OR" in update_sql
    assert "job_id =" in update_sql
    assert statements[0].is_update is True
    assert statements[1].is_select is True


def test_ordinary_baseline_still_revalidates_an_attached_job_via_idempotent_create(
    tmp_path: Path,
) -> None:
    version = {
        "id": "ordinary-v17-version",
        "config": {BOOTSTRAP_CONFIG_KEY: {}},
    }
    periods = {
        "historical_start": "2008-01-02",
        "historical_end": "2024-12-31",
        "start": "2025-01-02",
        "end": "2025-12-31",
    }
    backtest = {
        "id": "ordinary-v17-backtest",
        "dataset": "daily-ready",
        "execution_dataset": None,
        "periods": periods,
        "evidence_mode": "sealed_final_oos",
        "status": "queued",
        "job_id": "unrelated-attached-job",
    }

    class Strategies:
        attach_calls = 0

        @staticmethod
        def list_backtests(**_kwargs) -> list[dict]:
            return [dict(backtest)]

        @classmethod
        def attach_job(cls, *_args) -> None:
            cls.attach_calls += 1

    class Jobs:
        create_calls = 0

        @classmethod
        def create(cls, kind: str, payload: dict, _log_path: Path, **kwargs) -> dict:
            cls.create_calls += 1
            assert kind == "strategy_backtest"
            assert kwargs["idempotency_key"] == (
                "transparent-baseline:ordinary-v17-version:ordinary-v17-backtest"
            )
            return {
                "id": "exact-idempotent-job",
                "status": "queued",
                "payload": payload,
            }

        @staticmethod
        def get(_job_id: str) -> dict:
            raise AssertionError("ordinary active rows must re-enter idempotent create")

    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        strategies=Strategies(),
        jobs=Jobs(),
        promotions=object(),
        lockboxes=object(),
    )
    with pytest.raises(ValueError, match="attached to a different job"):
        service._ensure_backtest_job(
            plan={"formal_periods": periods},
            version=version,
            dataset={"name": "daily-ready", "path": "/data/qlib/daily-ready"},
            allow_create=True,
        )
    assert Jobs.create_calls == 1
    assert Strategies.attach_calls == 0


@pytest.mark.parametrize(
    ("status", "promotion_stage", "expected_resume"),
    [
        ("draft", None, True),
        ("approved", "paper", False),
        ("suspended", "suspended", False),
    ],
)
def test_one_shot_reuses_exact_version_and_resumes_only_an_untouched_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    promotion_stage: str | None,
    expected_resume: bool,
) -> None:
    plan = {
        "config": {
            "recipe_version": FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION,
            "evidence_mode": EVIDENCE_MODE_REPLAY,
        },
        "formal_periods": SOURCE_PERIODS,
    }
    source = {
        **_source_version(),
        "strategy_id": "source-family",
        "benchmark": "SH000300",
        "universe": "csi300",
    }
    version = {
        "id": "new-v18-version",
        "benchmark": source["benchmark"],
        "universe": source["universe"],
        "factors": [],
        "config": plan["config"],
        "evidence_mode": EVIDENCE_MODE_REPLAY,
        "status": status,
        "promotion_stage": promotion_stage,
    }
    dataset = {
        "name": SOURCE_DATASET,
        "path": "/data/qlib/source",
    }

    class Transaction:
        def __enter__(self) -> object:
            return object()

        def __exit__(self, *_args) -> None:
            return None

    class Strategies:
        class Engine:
            @staticmethod
            def begin() -> Transaction:
                return Transaction()

        engine = Engine()

        @staticmethod
        def get_version(version_id: str) -> dict:
            assert version_id == SOURCE_VERSION_ID
            return dict(source)

        @staticmethod
        def get(strategy_id: str) -> dict:
            assert strategy_id == "source-family"
            return {"versions": [dict(version)]}

        @staticmethod
        def create_version_if_absent(*_args, **_kwargs) -> dict:
            raise AssertionError("the exact existing v18 version must be reused")

    monkeypatch.setattr(bootstrap, "_select_forward_only_dataset", lambda _rows: dataset)
    monkeypatch.setattr(
        bootstrap,
        "_build_forward_only_rehabilitation_plan",
        lambda **_kwargs: plan,
    )
    monkeypatch.setattr(bootstrap, "require_source_cancellation", lambda _connection: None)
    monkeypatch.setattr(
        bootstrap,
        "require_source_cash_only_lockbox",
        lambda _connection: {"scope": "cash_only"},
    )
    monkeypatch.setattr(
        bootstrap,
        "require_consumed_vintage",
        lambda *_args, **_kwargs: type("Vintage", (), {"id": "consumed-vintage"})(),
    )
    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        dataset_loader=lambda _root: [dataset],
        strategies=Strategies(),
        jobs=object(),
        promotions=object(),
        lockboxes=object(),
    )
    observed: dict = {}

    def ensure(**kwargs) -> dict:
        observed.update(kwargs)
        if not kwargs["allow_create"]:
            raise ValueError("existing replay chain is incomplete")
        backtest_id = "new-v18-backtest"
        return {
            "backtest": {
                "id": backtest_id,
                "status": "queued",
                "artifact_path": str(service.artifact_root / backtest_id),
                "evidence_mode": EVIDENCE_MODE_REPLAY,
            },
            "backtest_action": "created",
            "job": {"id": "new-v18-job", "status": "queued"},
            "job_action": "created",
        }

    service._ensure_backtest_job = ensure
    service._advance_paper = lambda **_kwargs: {
        "state": "formal_backtest_pending",
        "paper_stage": None,
    }

    result = service.reconcile_forward_only_rehabilitation(actor="system:test")

    assert result["status"] == ("pending" if expected_resume else "failed")
    assert result["strategy_version_action"] == "reused"
    assert observed["allow_create"] is expected_resume
    assert observed["version"] == version
    assert observed["plan"] == plan
    if not expected_resume:
        assert result["errors"] == ["existing replay chain is incomplete"]


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
