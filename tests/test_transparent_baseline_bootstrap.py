from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from governance_fixtures import governed_etf_ready_evidence
from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import DBAPIError

from quant_data.database import (
    audit_events,
    backtest_runs,
    jobs,
    oos_vintages,
    open_database,
    strategy_versions,
    transparent_baseline_pre_result_repairs,
)
from quant_data.execution_contract import DAILY_QLIB_FIELD_CONTRACT_VERSION
from quant_data.history_bounds import GOVERNED_DAILY_STOCK_SCOPE_VERSION
from quant_platform.api import StrategyConfigRequest
from quant_platform.promotion import PromotionStore
from quant_platform.research_horizon import research_horizon_contract
from quant_platform.strategy_recipes import (
    TRANSPARENT_RESEARCH_BASELINE_IDS,
    get_strategy_recipe,
)
from quant_platform.strategy_store import StrategyStore, _normalize_multifactor_contract
from quant_platform.transparent_baseline_bootstrap import (
    TransparentBaselineBootstrapService,
    _feature_set,
    _select_dataset,
)
from quant_platform.transparent_baseline_lockbox import (
    BOOTSTRAP_CONFIG_KEY,
    LOCKBOX_CONFIG_KEY,
    TransparentBaselineLockboxStore,
    build_joint_lockbox,
    build_lockbox_member,
    canonical_sha256,
    lockbox_member_link,
    validate_joint_lockbox,
    validate_pre_result_repair_receipt,
)
from scripts.run_multifactor_backtest import _promotion_dataset_descriptors


def _calendar(count: int = 4300) -> list[str]:
    result: list[str] = []
    current = date(2008, 1, 2)
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current += timedelta(days=1)
    return result


def _formal_periods(calendar: list[str], recipe_id: str) -> dict[str, str]:
    starts = {
        "short_relative_strength": 2721,
        "swing_trend": 2828,
        "long_quality_value": 2954,
    }
    oos = {
        "short_relative_strength": 252,
        "swing_trend": 504,
        "long_quality_value": 756,
    }
    start = starts[recipe_id]
    return {
        "historical_start": calendar[0],
        "historical_end": calendar[2700],
        "start": calendar[start],
        "end": calendar[start + oos[recipe_id] - 1],
    }


def _base_plan(calendar: list[str], recipe_id: str) -> dict:
    recipe = get_strategy_recipe(recipe_id)
    horizon = research_horizon_contract(str(recipe["horizon"]))
    formal = _formal_periods(calendar, recipe_id)
    raw = StrategyConfigRequest.model_validate(
        {
            **recipe["config_overrides"],
            "recipe_id": recipe["id"],
            "recipe_version": recipe["version"],
            "outer_purge_days": horizon.purge_sessions,
            "outer_embargo_days": max(int(horizon.embargo_sessions or 0), 20),
            "min_backtest_days": horizon.sealed_oos_sessions,
        }
    ).model_dump()
    window = {
        "recipe_id": recipe_id,
        "formal_periods": formal,
        "calendar_end": calendar[-1],
    }
    raw[BOOTSTRAP_CONFIG_KEY] = {
        "contract_version": "transparent-baseline-bootstrap-v1",
        "recipe_id": recipe_id,
        "recipe_version": recipe["version"],
        "recipe_sha256": canonical_sha256(recipe),
        "dataset": "daily-ready",
        "dataset_identity_sha256": "a" * 64,
        "dataset_lineage_id": "b" * 64,
        "feature_set": {
            "id": f"transparent-baseline:{recipe_id}",
            "definition_sha256": canonical_sha256(recipe_id),
        },
        "research_periods": {
            "train_start": formal["historical_start"],
            "train_end": calendar[2500],
            "valid_start": calendar[2520],
            "valid_end": formal["historical_end"],
            "test_start": formal["start"],
            "test_end": formal["end"],
        },
        "formal_periods": formal,
        "research_window_contract": window,
        "research_window_contract_sha256": canonical_sha256(window),
    }
    base = _normalize_multifactor_contract(
        raw,
        factor_count=0,
        creating_family=True,
    )
    return {
        "recipe": recipe,
        "recipe_sha256": raw[BOOTSTRAP_CONFIG_KEY]["recipe_sha256"],
        "feature_set": raw[BOOTSTRAP_CONFIG_KEY]["feature_set"],
        "periods": raw[BOOTSTRAP_CONFIG_KEY]["research_periods"],
        "formal_periods": formal,
        "research_window_contract_sha256": raw[BOOTSTRAP_CONFIG_KEY][
            "research_window_contract_sha256"
        ],
        "base_config": base,
        "lockbox_member": build_lockbox_member(
            config=base,
            formal_periods=formal,
        ),
    }


def _plans(calendar: list[str]) -> tuple[list[dict], dict]:
    plans = [
        _base_plan(calendar, recipe_id)
        for recipe_id in TRANSPARENT_RESEARCH_BASELINE_IDS
    ]
    lockbox = build_joint_lockbox(
        dataset="daily-ready",
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
        members=[plan["lockbox_member"] for plan in plans],
    )
    for plan in plans:
        plan["config"] = _normalize_multifactor_contract(
            {
                **plan["base_config"],
                LOCKBOX_CONFIG_KEY: lockbox,
            },
            factor_count=0,
            creating_family=True,
        )
        assert lockbox_member_link(plan["config"])["batch_sha256"] == (
            lockbox["batch_sha256"]
        )
    return plans, lockbox


def _retarget_plans(
    plans: list[dict],
    *,
    dataset: str,
    identity: str,
    lineage: str,
    recipe_version: str,
    change_economic_rule: bool = False,
) -> tuple[list[dict], dict]:
    result = deepcopy(plans)
    for plan in result:
        base = deepcopy(plan["base_config"])
        base["recipe_version"] = recipe_version
        if change_economic_rule:
            # ``topk`` is compiled from the public rule IR and would be
            # rejected before the repair guard is reached.  ``n_drop`` is a
            # valid persisted trading-policy field, so changing it proves the
            # repair guard itself rejects an economic change.
            base["n_drop"] = int(base["n_drop"]) + 1
        bootstrap = base[BOOTSTRAP_CONFIG_KEY]
        bootstrap.update(
            {
                "recipe_version": recipe_version,
                "recipe_sha256": canonical_sha256(
                    {
                        "recipe_id": plan["recipe"]["id"],
                        "recipe_version": recipe_version,
                    }
                ),
                "dataset": dataset,
                "dataset_identity_sha256": identity,
                "dataset_lineage_id": lineage,
            }
        )
        plan["base_config"] = _normalize_multifactor_contract(
            base,
            factor_count=0,
            creating_family=True,
        )
        plan["lockbox_member"] = build_lockbox_member(
            config=plan["base_config"],
            formal_periods=plan["formal_periods"],
        )
    lockbox = build_joint_lockbox(
        dataset=dataset,
        dataset_identity_sha256=identity,
        dataset_lineage_id=lineage,
        members=[plan["lockbox_member"] for plan in result],
    )
    for plan in result:
        plan["config"] = _normalize_multifactor_contract(
            {**plan["base_config"], LOCKBOX_CONFIG_KEY: lockbox},
            factor_count=0,
            creating_family=True,
        )
    return result, lockbox


def _repair_receipt_payload(members: list[dict], *, target_recipe_version: str) -> dict:
    payload = {
        "contract_version": "transparent-baseline-pre-result-repair-v1",
        "source_release_commit": "e" * 40,
        "target_recipe_version": target_recipe_version,
        "target_eligibility_contract": "cn-stock-etf-point-in-time-eligibility-v3",
        "target_stock_scope_contract": "cn-mainland-a-share-daily-scope-v1",
        "reason_codes": [
            "empty_eligible_session_pandas_concat_failure",
            "pre_2023_bse_history_outside_governed_scope",
        ],
        "performance_information_used": False,
        "members": members,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def _prepare_repair_store_case(
    database_url: str,
    tmp_path: Path,
    *,
    receipt_after_result: bool = False,
    change_economic_rule: bool = False,
) -> tuple[
    StrategyStore,
    list[dict],
    list[dict],
    list[dict],
    str,
]:
    calendar = _calendar()
    current_plans, _ = _plans(calendar)
    source_plans, source_lockbox = _retarget_plans(
        current_plans,
        dataset="source-daily",
        identity="a" * 64,
        lineage="b" * 64,
        recipe_version="source-recipe-v6",
    )
    target_recipe_version = get_strategy_recipe("short_relative_strength")["version"]
    target_plans, _ = _retarget_plans(
        current_plans,
        dataset="target-daily",
        identity="d" * 64,
        lineage="e" * 64,
        recipe_version=target_recipe_version,
        change_economic_rule=change_economic_rule,
    )
    store = StrategyStore(database_url)
    families: dict[str, dict] = {}
    source_versions: list[dict] = []
    for plan in source_plans:
        recipe = plan["recipe"]
        family = store.create(
            name=f"repair-store:{recipe['id']}",
            description="Transparent baseline pre-result repair fixture.",
            benchmark=recipe["benchmark"],
            universe=recipe["universe"],
            factors=[],
            config=plan["config"],
            actor="test",
            economic_hypothesis_group=f"transparent-public-control:{recipe['id']}",
        )
        families[str(recipe["id"])] = family
        source_versions.append(family["versions"][0])
    TransparentBaselineLockboxStore(database_url).reserve(
        versions=source_versions,
        dataset="source-daily",
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
    )

    engine = open_database(database_url)
    errors = {
        "swing_trend": (
            "ValueError: cannot concatenate unaligned mixed dimensional NDFrame objects"
        ),
        "long_quality_value": (
            "ValueError: formal execution starts before native price-limit controls "
            "are complete (2022-12-31)"
        ),
    }
    source_members: list[dict] = []
    source_backtests: dict[str, dict] = {}
    for index, (plan, version) in enumerate(
        zip(source_plans, source_versions, strict=True),
        start=1,
    ):
        recipe_id = str(plan["recipe"]["id"])
        backtest = store.create_backtest(
            version_id=str(version["id"]),
            dataset="source-daily",
            periods=plan["formal_periods"],
            artifact_path=tmp_path / "backtests",
            trading_dates=calendar,
            dataset_lineage_id="b" * 64,
            dataset_identity_sha256="a" * 64,
        )
        source_backtests[recipe_id] = backtest
        artifact = Path(str(backtest["artifact_path"]))
        artifact.mkdir(parents=True, exist_ok=True)
        manifest = json.dumps(
            {
                "backtest_id": backtest["id"],
                "strategy_version_id": version["id"],
                "dataset": "source-daily",
                "periods": plan["formal_periods"],
                "config": version["config"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        (artifact / "manifest.json").write_bytes(manifest)
        job_id = f"{index + 6:x}" * 32
        error = errors.get(recipe_id)
        status = "running" if error is None else "failed"
        now = datetime.now(UTC)
        with engine.begin() as connection:
            connection.execute(
                insert(jobs).values(
                    id=job_id,
                    kind="strategy_backtest",
                    idempotency_key=f"repair-source:{job_id}",
                    status=status,
                    payload_json={"backtest_id": backtest["id"]},
                    progress_json=None,
                    log_path=str(tmp_path / f"{job_id}.log"),
                    exit_code=(1 if error else None),
                    error=error,
                    attempts=1,
                    max_attempts=1,
                    next_attempt_at=None,
                    cancel_requested_at=None,
                    created_at=now,
                    started_at=now,
                    finished_at=(now if error else None),
                )
            )
        store.attach_job(str(backtest["id"]), job_id)
        store.mark_backtest(str(backtest["id"]), status, error=error)
        source_members.append(
            {
                "backtest_id": str(backtest["id"]),
                "strategy_version_id": str(version["id"]),
                "job_id": job_id,
                "dataset": "source-daily",
                "periods": dict(plan["formal_periods"]),
                "status": status,
                "job_status": status,
                "error": error,
                "metrics_absent": True,
                "result_absent": True,
                "files": [
                    {
                        "path": "manifest.json",
                        "bytes": len(manifest),
                        "sha256": hashlib.sha256(manifest).hexdigest(),
                    }
                ],
            }
        )

    short = source_backtests["short_relative_strength"]
    short_artifact = Path(str(short["artifact_path"]))
    short_result = short_artifact / "reports" / "result.json"
    short_job_id = next(
        item["job_id"]
        for item in source_members
        if item["backtest_id"] == str(short["id"])
    )

    def finish_short() -> None:
        short_result.parent.mkdir(parents=True, exist_ok=True)
        short_result.write_text('{"metrics":{"return":1.0}}', encoding="utf-8")
        store.mark_backtest(str(short["id"]), "succeeded", metrics={"return": 1.0})
        with engine.begin() as connection:
            connection.execute(
                update(jobs)
                .where(jobs.c.id == short_job_id)
                .values(
                    status="succeeded",
                    exit_code=0,
                    error=None,
                    finished_at=datetime.now(UTC),
                )
            )

    if receipt_after_result:
        finish_short()
    receipt = _repair_receipt_payload(
        source_members,
        target_recipe_version=target_recipe_version,
    )
    with engine.begin() as connection:
        event_id = connection.execute(
            insert(audit_events)
            .values(
                user_id=None,
                username="system:test",
                action="transparent_baseline_pre_result_repair_registered",
                method="INTERNAL",
                path="transparent-baseline/pre-result-repair",
                status_code=201,
                ip_hash=None,
                user_agent="pytest",
                details_json=receipt,
                created_at=datetime.now(UTC),
            )
            .returning(audit_events.c.id)
        ).scalar_one()
    if not receipt_after_result:
        finish_short()

    target_versions: list[dict] = []
    for plan in target_plans:
        recipe_id = str(plan["recipe"]["id"])
        target_versions.append(
            store.create_version_if_absent(
                str(families[recipe_id]["id"]),
                benchmark=str(plan["recipe"]["benchmark"]),
                universe=str(plan["recipe"]["universe"]),
                factors=[],
                config=plan["config"],
                actor="test",
            )
        )
    assert source_lockbox["batch_sha256"]
    return store, target_plans, target_versions, source_members, str(event_id)


@pytest.mark.no_database
def test_joint_lockbox_requires_exact_three_members_and_detects_tampering() -> None:
    plans, lockbox = _plans(_calendar())

    assert validate_joint_lockbox(lockbox) == lockbox
    for plan in plans:
        assert plan["recipe_sha256"] == canonical_sha256(plan["recipe"])
        assert len(_feature_set(plan["recipe"])["features"]) == len(
            plan["recipe"].get("factor_baseline") or []
        )
    with pytest.raises(ValueError, match="exactly the three"):
        build_joint_lockbox(
            dataset="daily-ready",
            dataset_identity_sha256="a" * 64,
            dataset_lineage_id="b" * 64,
            members=[plan["lockbox_member"] for plan in plans[:2]],
        )

    tampered = {**lockbox, "dataset": "different"}
    with pytest.raises(ValueError, match="digest or members changed"):
        validate_joint_lockbox(tampered)

    changed_recipe = deepcopy(plans[0]["config"])
    changed_recipe[BOOTSTRAP_CONFIG_KEY]["recipe_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs from its joint-lockbox member"):
        lockbox_member_link(changed_recipe)


@pytest.mark.no_database
def test_pre_result_repair_receipt_is_hashed_and_contains_no_performance_artifacts() -> None:
    periods = {
        "historical_start": "2008-01-02",
        "historical_end": "2021-12-31",
        "start": "2022-01-04",
        "end": "2025-01-03",
    }
    members = [
        {
            "backtest_id": "1" * 32,
            "strategy_version_id": "4" * 32,
            "job_id": "7" * 32,
            "dataset": "source-daily",
            "periods": periods,
            "status": "running",
            "job_status": "running",
            "error": None,
            "metrics_absent": True,
            "result_absent": True,
            "files": [
                {"path": "manifest.json", "bytes": 10, "sha256": "a" * 64},
                {
                    "path": "baseline/composite.parquet",
                    "bytes": 20,
                    "sha256": "b" * 64,
                },
            ],
        },
        {
            "backtest_id": "2" * 32,
            "strategy_version_id": "5" * 32,
            "job_id": "8" * 32,
            "dataset": "source-daily",
            "periods": periods,
            "status": "failed",
            "job_status": "failed",
            "error": (
                "ValueError: cannot concatenate unaligned mixed dimensional "
                "NDFrame objects"
            ),
            "metrics_absent": True,
            "result_absent": True,
            "files": [{"path": "manifest.json", "bytes": 11, "sha256": "c" * 64}],
        },
        {
            "backtest_id": "3" * 32,
            "strategy_version_id": "6" * 32,
            "job_id": "9" * 32,
            "dataset": "source-daily",
            "periods": periods,
            "status": "failed",
            "job_status": "failed",
            "error": (
                "ValueError: formal execution starts before native price-limit "
                "controls are complete (2022-12-31)"
            ),
            "metrics_absent": True,
            "result_absent": True,
            "files": [{"path": "manifest.json", "bytes": 12, "sha256": "d" * 64}],
        },
    ]
    payload = {
        "contract_version": "transparent-baseline-pre-result-repair-v1",
        "source_release_commit": "e" * 40,
        "target_recipe_version": get_strategy_recipe(
            "short_relative_strength"
        )["version"],
        "target_eligibility_contract": "cn-stock-etf-point-in-time-eligibility-v3",
        "target_stock_scope_contract": "cn-mainland-a-share-daily-scope-v1",
        "reason_codes": [
            "empty_eligible_session_pandas_concat_failure",
            "pre_2023_bse_history_outside_governed_scope",
        ],
        "performance_information_used": False,
        "members": members,
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}

    assert validate_pre_result_repair_receipt(receipt)["members"] == members

    result_tamper = deepcopy(receipt)
    result_tamper["members"][0]["files"].append(
        {"path": "daily_returns.parquet", "bytes": 1, "sha256": "f" * 64}
    )
    result_payload = dict(result_tamper)
    result_payload.pop("receipt_sha256")
    result_tamper["receipt_sha256"] = canonical_sha256(result_payload)
    with pytest.raises(ValueError, match="result artifact"):
        validate_pre_result_repair_receipt(result_tamper)


@pytest.mark.no_database
def test_latest_ready_dataset_does_not_fall_back_when_its_seal_is_broken(
    monkeypatch,
) -> None:
    from quant_platform import transparent_baseline_bootstrap as module

    def validate(dataset: dict) -> dict:
        if dataset["name"] == "latest-broken":
            raise ValueError("latest ready daily Qlib dataset is not reproducibly sealed")
        return dataset

    monkeypatch.setattr(module, "_validate_dataset", validate)
    with pytest.raises(ValueError, match="not reproducibly sealed"):
        _select_dataset(
            [
                {
                    "name": "older-good",
                    "ready": True,
                    "frequency": "day",
                    "end_date": "2026-08-27",
                },
                {
                    "name": "latest-broken",
                    "ready": True,
                    "frequency": "day",
                    "end_date": "2026-08-28",
                },
            ],
            anchored_name=None,
        )


@pytest.mark.no_database
def test_recipe_vnext_ignores_old_family_anchor_but_recovers_partial_current_batch() -> None:
    recipe_id = "short_relative_strength"
    recipe = get_strategy_recipe(recipe_id)
    old = {
        "config": {
            "recipe_id": recipe_id,
            "recipe_version": "retained-old-recipe",
            BOOTSTRAP_CONFIG_KEY: {
                "recipe_sha256": "0" * 64,
                "dataset": "old-dataset",
            },
        }
    }
    families = {
        recipe_id: {"versions": [old]},
        "swing_trend": None,
        "long_quality_value": None,
    }

    assert TransparentBaselineBootstrapService._anchored_dataset(families) is None

    current = deepcopy(old)
    current["config"].update(
        {
            "recipe_version": recipe["version"],
            BOOTSTRAP_CONFIG_KEY: {
                "recipe_sha256": canonical_sha256(recipe),
                "dataset": "current-batch-dataset",
            },
        }
    )
    families[recipe_id] = {"versions": [old, current]}
    assert (
        TransparentBaselineBootstrapService._anchored_dataset(families)
        == "current-batch-dataset"
    )

    families[recipe_id] = {"versions": [old, current, deepcopy(current)]}
    with pytest.raises(ValueError, match="duplicate current-recipe"):
        TransparentBaselineBootstrapService._anchored_dataset(families)


@pytest.mark.no_database
def test_recipe_vnext_stays_in_family_and_uses_atomic_exact_create(tmp_path: Path) -> None:
    recipe = get_strategy_recipe("short_relative_strength")
    old = {
        "id": "old-version",
        "benchmark": recipe["benchmark"],
        "universe": recipe["universe"],
        "factors": [],
        "config": {"recipe_version": "old-release"},
        "horizon_profile": recipe["horizon"],
    }
    expected_config = {
        "recipe_id": recipe["id"],
        "recipe_version": recipe["version"],
        "material_contract": "current-release",
    }
    family = {"id": "same-family", "versions": [old]}

    class AtomicStrategies:
        calls = 0

        def create_version_if_absent(self, strategy_id: str, **values) -> dict:
            assert strategy_id == "same-family"
            assert values["config"] == expected_config
            self.calls += 1
            exact = [
                item for item in family["versions"] if item["config"] == expected_config
            ]
            if exact:
                return exact[0]
            created = {
                "id": "current-version",
                "benchmark": values["benchmark"],
                "universe": values["universe"],
                "factors": [],
                "config": dict(values["config"]),
                "horizon_profile": recipe["horizon"],
            }
            family["versions"].append(created)
            return created

        @staticmethod
        def create_version(*_args, **_kwargs) -> dict:
            raise AssertionError("managed reconciliation must use atomic exact-create")

        @staticmethod
        def get_by_name(name: str) -> dict:
            assert name == "QuantLab透明基线：1至5日短线相对强弱"
            return family

    strategies = AtomicStrategies()
    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        strategies=strategies,
        jobs=object(),
        promotions=object(),
        lockboxes=object(),
    )
    plan = {"recipe": recipe, "config": expected_config}
    first = service._ensure_versions(
        plans=[plan],
        families={recipe["id"]: family},
        actor="test-bootstrap",
    )
    second_plan = {"recipe": recipe, "config": expected_config}
    second = service._ensure_versions(
        plans=[second_plan],
        families={recipe["id"]: family},
        actor="test-bootstrap",
    )

    assert [item["id"] for item in family["versions"]] == [
        "old-version",
        "current-version",
    ]
    assert first[0]["id"] == second[0]["id"] == "current-version"
    assert plan["family_action"] == "version_created"
    assert second_plan["family_action"] == "reused"
    assert strategies.calls == 1


@pytest.mark.no_database
def test_backtest_creation_race_recovers_only_the_exact_frozen_run(
    tmp_path: Path,
) -> None:
    formal_periods = {
        "historical_start": "2008-01-02",
        "historical_end": "2024-12-31",
        "start": "2025-01-02",
        "end": "2025-12-31",
    }
    backtest = {
        "id": "formal-backtest",
        "strategy_version_id": "baseline-version",
        "dataset": "daily-ready",
        "execution_dataset": None,
        "periods": formal_periods,
        "status": "queued",
        "job_id": None,
    }

    class RacingStrategies:
        reads = 0

        def list_backtests(self, *, version_id: str, limit: int) -> list[dict]:
            assert version_id == "baseline-version"
            assert limit == 10
            self.reads += 1
            return [] if self.reads == 1 else [dict(backtest)]

        @staticmethod
        def create_backtest(**_values) -> dict:
            raise ValueError("reserved final test has already been consumed")

        @staticmethod
        def attach_job(backtest_id: str, job_id: str) -> None:
            assert backtest_id == "formal-backtest"
            assert job_id == "formal-job"
            backtest["job_id"] = job_id

        @staticmethod
        def get_backtest(backtest_id: str) -> dict:
            assert backtest_id == "formal-backtest"
            return dict(backtest)

    class IdempotentJobs:
        @staticmethod
        def create(kind: str, payload: dict, _log_path: Path, **options) -> dict:
            assert kind == "strategy_backtest"
            assert payload["backtest_id"] == "formal-backtest"
            assert options["idempotency_key"] == (
                "transparent-baseline:baseline-version:formal-backtest"
            )
            return {"id": "formal-job", "status": "queued"}

    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        strategies=RacingStrategies(),
        jobs=IdempotentJobs(),
        promotions=object(),
        lockboxes=object(),
    )
    result = service._ensure_backtest_job(
        plan={"formal_periods": formal_periods},
        version={"id": "baseline-version"},
        dataset={
            "name": "daily-ready",
            "path": str(tmp_path / "daily-ready"),
            "calendar": ["2025-01-02", "2025-12-31"],
            "dataset_identity_sha256": "a" * 64,
            "dataset_lineage_id": "b" * 64,
        },
    )

    assert result["backtest_action"] == "reused"
    assert result["job_action"] == "created"
    assert result["backtest"]["job_id"] == "formal-job"


@pytest.mark.no_database
def test_approval_race_recovers_only_an_approved_paper_transition(
    tmp_path: Path,
) -> None:
    class RacingStrategies:
        reads = 0

        def get_version(self, version_id: str) -> dict:
            assert version_id == "baseline-version"
            self.reads += 1
            if self.reads == 1:
                return {"status": "draft", "promotion_stage": None}
            return {"status": "approved", "promotion_stage": "paper"}

        @staticmethod
        def approve(*_args, **_kwargs) -> None:
            raise ValueError("another reconcile approved this version")

    class IdempotentPromotions:
        @staticmethod
        def prepare_paper_stage(version_id: str, *, actor: str) -> dict:
            assert version_id == "baseline-version"
            assert actor == "test-bootstrap"
            return {"stage": "paper", "status": "awaiting_simulation"}

    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        strategies=RacingStrategies(),
        jobs=object(),
        promotions=IdempotentPromotions(),
        lockboxes=object(),
    )
    result = service._advance_paper(
        version_id="baseline-version",
        backtest={"status": "succeeded"},
        actor="test-bootstrap",
    )

    assert result["state"] == "paper_validating"
    assert result["paper_stage"]["status"] == "awaiting_simulation"


def test_joint_lockbox_allows_only_its_three_overlapping_one_shot_vintages(
    database_url: str,
    tmp_path: Path,
) -> None:
    calendar = _calendar()
    plans, lockbox = _plans(calendar)
    strategies = StrategyStore(database_url)
    versions = []
    for plan in plans:
        recipe = plan["recipe"]
        family = strategies.create(
            name=f"lockbox-test:{recipe['id']}",
            description="Predeclared transparent baseline joint OOS member.",
            benchmark=recipe["benchmark"],
            universe=recipe["universe"],
            factors=[],
            config=plan["config"],
            actor="test",
        )
        versions.append(family["versions"][0])

    reserved = TransparentBaselineLockboxStore(database_url).reserve(
        versions=versions,
        dataset="daily-ready",
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
    )
    assert reserved["batch_sha256"] == lockbox["batch_sha256"]
    assert {item["status"] for item in reserved["members"]} == {"reserved"}

    for plan, version in zip(plans, versions, strict=True):
        backtest = strategies.create_backtest(
            version_id=version["id"],
            dataset="daily-ready",
            periods=plan["formal_periods"],
            artifact_path=tmp_path / "backtests",
            trading_dates=calendar,
            dataset_lineage_id="b" * 64,
            dataset_identity_sha256="a" * 64,
        )
        assert backtest["status"] == "queued"

    recovered = TransparentBaselineLockboxStore(database_url).reserve(
        versions=versions,
        dataset="daily-ready",
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
    )
    assert {item["status"] for item in recovered["members"]} == {"consumed"}

    duplicate = strategies.create(
        name="lockbox-test:temporary-fourth",
        description="A duplicate version must not steal a preregistered member.",
        benchmark=plans[0]["recipe"]["benchmark"],
        universe=plans[0]["recipe"]["universe"],
        factors=[],
        config=plans[0]["config"],
        actor="test",
    )["versions"][0]
    with pytest.raises(ValueError, match="different baseline strategy"):
        strategies.create_backtest(
            version_id=duplicate["id"],
            dataset="daily-ready",
            periods=plans[0]["formal_periods"],
            artifact_path=tmp_path / "fourth",
            trading_dates=calendar,
            dataset_lineage_id="b" * 64,
            dataset_identity_sha256="a" * 64,
        )


def test_new_lineage_cannot_reopen_overlapping_baseline_oos_without_repair_receipt(
    database_url: str,
) -> None:
    calendar = _calendar()
    source_plans, _ = _plans(calendar)
    store = StrategyStore(database_url)
    families: dict[str, dict] = {}
    source_versions = []
    for plan in source_plans:
        recipe_id = str(plan["recipe"]["id"])
        family = store.create(
            name=f"lockbox-lineage-guard:{recipe_id}",
            description="Source transparent baseline lockbox.",
            benchmark=str(plan["recipe"]["benchmark"]),
            universe=str(plan["recipe"]["universe"]),
            factors=[],
            config=plan["config"],
            actor="test",
        )
        families[recipe_id] = family
        source_versions.append(family["versions"][0])
    TransparentBaselineLockboxStore(database_url).reserve(
        versions=source_versions,
        dataset="daily-ready",
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
    )

    target_plans = deepcopy(source_plans)
    for plan in target_plans:
        base = deepcopy(plan["base_config"])
        base["recipe_version"] = "test-target-recipe"
        bootstrap = base[BOOTSTRAP_CONFIG_KEY]
        bootstrap.update(
            {
                "recipe_version": "test-target-recipe",
                "recipe_sha256": "c" * 64,
                "dataset": "daily-repaired",
                "dataset_identity_sha256": "d" * 64,
                "dataset_lineage_id": "e" * 64,
            }
        )
        plan["base_config"] = _normalize_multifactor_contract(
            base,
            factor_count=0,
            creating_family=True,
        )
        plan["lockbox_member"] = build_lockbox_member(
            config=plan["base_config"],
            formal_periods=plan["formal_periods"],
        )
    target_lockbox = build_joint_lockbox(
        dataset="daily-repaired",
        dataset_identity_sha256="d" * 64,
        dataset_lineage_id="e" * 64,
        members=[plan["lockbox_member"] for plan in target_plans],
    )
    target_versions = []
    for plan in target_plans:
        recipe_id = str(plan["recipe"]["id"])
        config = _normalize_multifactor_contract(
            {**plan["base_config"], LOCKBOX_CONFIG_KEY: target_lockbox},
            factor_count=0,
            creating_family=True,
        )
        created = store.create_version_if_absent(
            str(families[recipe_id]["id"]),
            benchmark=str(plan["recipe"]["benchmark"]),
            universe=str(plan["recipe"]["universe"]),
            factors=[],
            config=config,
            actor="test",
        )
        target_versions.append(created)

    with pytest.raises(ValueError, match="exactly one valid pre-result repair receipt"):
        TransparentBaselineLockboxStore(database_url).reserve(
            versions=target_versions,
            dataset="daily-repaired",
            dataset_identity_sha256="d" * 64,
            dataset_lineage_id="e" * 64,
        )


def test_preregistered_pre_result_repair_opens_one_append_only_target_batch(
    database_url: str,
    tmp_path: Path,
) -> None:
    store, target_plans, target_versions, source_members, event_id = (
        _prepare_repair_store_case(database_url, tmp_path)
    )
    target_lockbox = validate_joint_lockbox(
        target_versions[0]["config"][LOCKBOX_CONFIG_KEY]
    )
    lockboxes = TransparentBaselineLockboxStore(database_url)

    reserved = lockboxes.reserve(
        versions=target_versions,
        dataset="target-daily",
        dataset_identity_sha256="d" * 64,
        dataset_lineage_id="e" * 64,
    )
    repeated = lockboxes.reserve(
        versions=target_versions,
        dataset="target-daily",
        dataset_identity_sha256="d" * 64,
        dataset_lineage_id="e" * 64,
    )

    assert reserved["batch_sha256"] == target_lockbox["batch_sha256"]
    assert {item["status"] for item in reserved["members"]} == {"reserved"}
    assert repeated["members"] == reserved["members"]
    assert repeated["pre_result_repair"] == reserved["pre_result_repair"]
    repair = reserved["pre_result_repair"]
    assert repair["source_audit_event_id"] == int(event_id)
    assert repair["source_backtest_ids"] == sorted(
        item["backtest_id"] for item in source_members
    )
    assert len(repair["results_created_after_preregistration"]) == 1
    assert repair["performance_information_used"] is False

    engine = open_database(database_url)
    with engine.connect() as connection:
        registry_rows = connection.execute(
            select(transparent_baseline_pre_result_repairs)
        ).all()
        vintages = connection.execute(select(oos_vintages)).all()
    assert len(registry_rows) == 1
    assert str(registry_rows[0].target_batch_sha256) == target_lockbox["batch_sha256"]
    assert len(vintages) == 6
    assert all(
        row.consumed_at is not None
        for row in vintages
        if str(row.dataset_lineage_id) == "b" * 64
    )
    assert all(
        row.consumed_at is None
        for row in vintages
        if str(row.dataset_lineage_id) == "e" * 64
    )

    with pytest.raises(DBAPIError, match="append-only"):
        with engine.begin() as connection:
            connection.execute(
                update(transparent_baseline_pre_result_repairs).values(
                    target_recipe_version="tampered"
                )
            )

    # A third lineage would be a second look at the same OOS windows. Even a
    # new immutable strategy version cannot turn the one repair into a retry
    # loop, and no source/target vintage is rewritten to make room for it.
    third_plans, _ = _retarget_plans(
        target_plans,
        dataset="third-daily",
        identity="1" * 64,
        lineage="2" * 64,
        recipe_version=get_strategy_recipe("short_relative_strength")["version"],
    )
    third_versions = []
    target_by_recipe = {
        str(item["config"]["recipe_id"]): item for item in target_versions
    }
    for plan in third_plans:
        recipe_id = str(plan["recipe"]["id"])
        third_versions.append(
            store.create_version_if_absent(
                str(target_by_recipe[recipe_id]["strategy_id"]),
                benchmark=str(plan["recipe"]["benchmark"]),
                universe=str(plan["recipe"]["universe"]),
                factors=[],
                config=plan["config"],
                actor="test",
            )
        )
    with pytest.raises(ValueError, match="more than one prior batch"):
        lockboxes.reserve(
            versions=third_versions,
            dataset="third-daily",
            dataset_identity_sha256="1" * 64,
            dataset_lineage_id="2" * 64,
        )
    with engine.connect() as connection:
        assert connection.scalar(
            select(func.count()).select_from(
                transparent_baseline_pre_result_repairs
            )
        ) == 1
        assert connection.scalar(select(func.count()).select_from(oos_vintages)) == 6


def test_pre_result_repair_receipt_registered_after_result_is_rejected(
    database_url: str,
    tmp_path: Path,
) -> None:
    _, _, target_versions, _, _ = _prepare_repair_store_case(
        database_url,
        tmp_path,
        receipt_after_result=True,
    )

    with pytest.raises(ValueError, match="exactly one valid pre-result repair receipt"):
        TransparentBaselineLockboxStore(database_url).reserve(
            versions=target_versions,
            dataset="target-daily",
            dataset_identity_sha256="d" * 64,
            dataset_lineage_id="e" * 64,
        )
    engine = open_database(database_url)
    with engine.connect() as connection:
        assert connection.scalar(
            select(func.count()).select_from(
                transparent_baseline_pre_result_repairs
            )
        ) == 0
        assert connection.scalar(select(func.count()).select_from(oos_vintages)) == 3


def test_pre_result_repair_cannot_change_an_economic_rule(
    database_url: str,
    tmp_path: Path,
) -> None:
    _, _, target_versions, _, _ = _prepare_repair_store_case(
        database_url,
        tmp_path,
        change_economic_rule=True,
    )

    with pytest.raises(ValueError, match="exactly one valid pre-result repair receipt"):
        TransparentBaselineLockboxStore(database_url).reserve(
            versions=target_versions,
            dataset="target-daily",
            dataset_identity_sha256="d" * 64,
            dataset_lineage_id="e" * 64,
        )
    engine = open_database(database_url)
    with engine.connect() as connection:
        assert connection.scalar(
            select(func.count()).select_from(
                transparent_baseline_pre_result_repairs
            )
        ) == 0
        assert connection.scalar(select(func.count()).select_from(oos_vintages)) == 3


def test_three_daily_baseline_artifacts_load_and_attach_paper_simulations(
    database_url: str,
    tmp_path: Path,
) -> None:
    """The exact three zero-factor controls can leave awaiting_simulation."""

    calendar = _calendar()
    plans, _ = _plans(calendar)
    strategies = StrategyStore(database_url)
    versions = []
    for plan in plans:
        recipe = plan["recipe"]
        family = strategies.create(
            name=f"daily-descriptor-test:{recipe['id']}",
            description="Transparent daily baseline promotion descriptor fixture.",
            benchmark=recipe["benchmark"],
            universe=recipe["universe"],
            factors=[],
            config=plan["config"],
            actor="test",
        )
        versions.append(family["versions"][0])

    TransparentBaselineLockboxStore(database_url).reserve(
        versions=versions,
        dataset="daily-ready",
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
    )
    provenance = {
        "frequency": "day",
        "dataset_identity_sha256": "a" * 64,
        "dataset_lineage_id": "b" * 64,
        "source_lineage_id": "c" * 64,
        "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
        "source_volume_unit": "hand",
        "qlib_volume_unit": "share",
        "source_amount_unit": "thousand_cny",
        "qlib_amount_unit": "cny",
        "source_hand_size": 100,
        "index_volume_policy": "excluded_non_tradable_benchmark",
        "governed_etf_whitelist": governed_etf_ready_evidence(),
        "lineage_verified": True,
        "execution_controls": {
            "formal_execution_requires_native_controls": True,
            "scope_version": GOVERNED_DAILY_STOCK_SCOPE_VERSION,
            "native_complete_from": "2008-01-01",
        },
    }
    promotion = PromotionStore(database_url)
    engine = open_database(database_url)

    for plan, version in zip(plans, versions, strict=True):
        recipe = plan["recipe"]
        artifact = tmp_path / f"formal-{recipe['id']}"
        artifact.mkdir()
        descriptors = _promotion_dataset_descriptors(
            daily_dataset_name="daily-ready",
            daily_provenance=provenance,
            execution_method=str(version["config"]["execution_method"]),
            execution_frequency=str(version["config"]["execution_frequency"]),
            formal_execution_start=str(plan["formal_periods"]["start"]),
        )
        (artifact / "datasets.json").write_text(
            json.dumps(descriptors, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        backtest = strategies.create_backtest(
            version_id=str(version["id"]),
            dataset="daily-ready",
            periods=plan["formal_periods"],
            artifact_path=artifact,
            trading_dates=calendar,
            dataset_lineage_id="b" * 64,
            dataset_identity_sha256="a" * 64,
        )
        strategies.mark_backtest(
            str(backtest["id"]),
            "succeeded",
            metrics={"fixture": "daily execution descriptor contract"},
        )
        with engine.begin() as connection:
            connection.execute(
                strategy_versions.update()
                .where(strategy_versions.c.id == str(version["id"]))
                .values(status="approved", promotion_stage="paper")
            )

        promotion.prepare_paper_stage(str(version["id"]), actor="test")
        loaded = promotion._load_backtest_datasets(str(version["id"]))
        assert loaded == descriptors
        assert loaded["execution"] == loaded["daily"]
        stage = promotion.attach_paper_simulation(
            str(version["id"]),
            actor="test",
            daily_dataset=loaded["daily"],
            execution_dataset=loaded["execution"],
            initial_cash=100_000,
        )
        assert stage["status"] == "active"
        assert stage["simulation_portfolio_id"]


def test_reconcile_is_idempotent_and_invalid_success_cannot_enter_paper(
    database_url: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    from quant_platform import transparent_baseline_bootstrap as module

    calendar = _calendar()
    plans, _ = _plans(calendar)
    by_recipe = {str(plan["recipe"]["id"]): plan for plan in plans}
    dataset = {
        "name": "daily-ready",
        "path": str(tmp_path / "qlib" / "daily-ready"),
        "ready": True,
        "reproducible": True,
        "output_files_verified": True,
        "frequency": "day",
        "start_date": calendar[0],
        "end_date": calendar[-1],
        "trading_days": len(calendar),
        "dataset_identity_sha256": "a" * 64,
        "dataset_lineage_id": "b" * 64,
        "calendar": calendar,
    }
    monkeypatch.setattr(module, "_validate_dataset", lambda value: dict(value))
    monkeypatch.setattr(
        module,
        "_plan_member",
        lambda *, recipe_id, dataset: {
            **by_recipe[recipe_id],
            "base_config": dict(by_recipe[recipe_id]["base_config"]),
            "lockbox_member": dict(by_recipe[recipe_id]["lockbox_member"]),
        },
    )
    service = TransparentBaselineBootstrapService(
        database_url=database_url,
        data_root=tmp_path,
        dataset_loader=lambda _root: [dataset],
    )

    first = service.reconcile(actor="test-bootstrap")
    second = service.reconcile(actor="test-bootstrap")
    assert first["status"] == "pending"
    assert second["status"] == "pending"
    assert all(item["backtest"]["action"] == "reused" for item in second["members"])
    assert all(item["job"]["action"] == "reused" for item in second["members"])

    engine = open_database(database_url)
    with engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(backtest_runs)) == 3
        assert connection.scalar(select(func.count()).select_from(jobs)) == 3
        assert connection.scalar(select(func.count()).select_from(oos_vintages)) == 3

    first_backtest_id = str(first["members"][0]["backtest"]["id"])
    StrategyStore(database_url).mark_backtest(
        first_backtest_id,
        "succeeded",
        metrics={"fabricated": True},
    )
    blocked = service.reconcile(actor="test-bootstrap")
    blocked_member = blocked["members"][0]
    version = StrategyStore(database_url).get_version(
        str(blocked_member["strategy_version_id"])
    )
    assert blocked["status"] == "failed"
    assert blocked_member["errors"]
    assert version["status"] == "draft"
    assert version["promotion_stage"] is None
    assert blocked["recommendation_enabled_created"] is False
    assert blocked["investor_profile_bypassed"] is False
    assert blocked["safety_mode_changed"] is False
