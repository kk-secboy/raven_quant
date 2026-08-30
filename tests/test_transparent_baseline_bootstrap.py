from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
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
from quant_platform.research_automation import ResearchWindowUnavailableError
from quant_platform.research_horizon import research_horizon_contract
from quant_platform.strategy_recipes import (
    TRANSPARENT_RESEARCH_BASELINE_IDS,
    get_strategy_recipe,
)
from quant_platform.strategy_store import StrategyStore, _normalize_multifactor_contract
from quant_platform.transparent_baseline_bootstrap import (
    FAMILY_NAMES,
    TransparentBaselineBootstrapService,
    _feature_set,
    _plan_member,
    _require_native_formal_oos,
    _select_dataset,
)
from quant_platform.transparent_baseline_lockbox import (
    BOOTSTRAP_CONFIG_KEY,
    CANONICAL_LF_PACKAGING_ERROR,
    CANONICAL_LF_PACKAGING_SOURCE_BINDINGS,
    CANONICAL_LF_PACKAGING_SOURCE_OBSERVED_RUNNER_SHA256,
    CANONICAL_LF_PACKAGING_TARGET_RECIPE_VERSION,
    LOCKBOX_CONFIG_KEY,
    LOCKBOX_CONTRACT_VERSION_V2,
    LOCKBOX_CONTRACT_VERSION_V3,
    LOCKBOX_LINK_VERSION_V2,
    OPTIMIZER_APPLICABILITY_REASON,
    OPTIMIZER_APPLICABILITY_REPAIR_GENERATION,
    OPTIMIZER_APPLICABILITY_SOURCE_BACKTEST_IDS,
    OPTIMIZER_APPLICABILITY_SOURCE_COMMIT,
    OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION,
    OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256,
    PRE_RESULT_REPAIR_CONTRACT_VERSION_V2,
    TransparentBaselineLockboxStore,
    build_joint_lockbox,
    build_lockbox_member,
    build_unopened_history_selection,
    canonical_sha256,
    lockbox_member_link,
    validate_joint_lockbox,
    validate_lockbox_link,
    validate_pre_result_repair_receipt,
)
from quant_platform.transparent_baseline_repair import (
    build_optimizer_applicability_receipt,
    register_canonical_lf_packaging_repair,
    register_optimizer_applicability_repair,
)
from quant_platform.transparent_baseline_runner import (
    TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
    WORKER_RUNTIME_IMAGE_DIGEST_ENV,
    target_runner_for_recipe,
    target_runtime_bundle_for_recipe,
    target_worker_runtime_image_for_recipe,
)
from scripts.run_multifactor_backtest import _promotion_dataset_descriptors

_WORKER_IMAGE_DIGEST = "sha256:" + "d" * 64


@pytest.fixture(autouse=True)
def _sealed_worker_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WORKER_RUNTIME_IMAGE_DIGEST_ENV, _WORKER_IMAGE_DIGEST)


def _calendar(count: int = 4300) -> list[str]:
    result: list[str] = []
    current = date(2008, 1, 2)
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current += timedelta(days=1)
    return result


def _pre_2022_server_calendar() -> list[str]:
    pre_cost = [
        item.date().isoformat()
        for item in pd.bdate_range(end="2015-07-31", periods=1844)
    ]
    raw_cost = [
        item.date().isoformat()
        for item in pd.bdate_range("2015-08-03", "2022-07-05")
    ]
    indices = sorted(
        {
            round(index * (len(raw_cost) - 1) / (1683 - 1))
            for index in range(1683)
        }
    )
    return pre_cost + [raw_cost[index] for index in indices]


def _research_dataset(
    calendar: list[str],
    *,
    path: Path,
    name: str = "daily-ready",
    identity: str = "a" * 64,
    lineage: str = "b" * 64,
) -> dict:
    required_fields = {
        "amount",
        "close",
        "fund_debt_to_assets",
        "fund_op_profit_yoy",
        "fund_quarter_revenue_yoy",
        "fund_roa",
        "fund_roe",
        "fund_roic",
        "fund_sales_cash_to_revenue",
        "high",
        "low",
        "pb",
        "pe_ttm",
    }
    coverage = {
        field: {
            "research_available_from": calendar[0],
            "available_to": calendar[-1],
            "years": [],
        }
        for field in sorted(required_fields)
    }
    provenance = {
        "dataset_identity_sha256": identity,
        "dataset_lineage_id": lineage,
        "dataset_contract_sha256": "c" * 64,
        "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
        "fields": sorted(required_fields),
        "field_units": {},
        "research_features": {},
        "source_start_date": calendar[0],
        "source_end_date": calendar[-1],
        "field_year_coverage": {
            "version": "qlib-field-year-source-coverage-v1",
            "evidence_status": "complete",
            "fields": coverage,
        },
        "execution_controls": {
            "formal_execution_requires_native_controls": True,
            "scope_version": GOVERNED_DAILY_STOCK_SCOPE_VERSION,
            "native_complete_from": calendar[0],
        },
    }
    return {
        "name": name,
        "path": str(path),
        "ready": True,
        "reproducible": True,
        "output_files_verified": True,
        "frequency": "day",
        "start_date": calendar[0],
        "end_date": calendar[-1],
        "trading_days": len(calendar),
        "dataset_identity_sha256": identity,
        "dataset_lineage_id": lineage,
        "lineage_id": lineage,
        "provenance": provenance,
        "calendar": calendar,
    }


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
    target_runner_sha256 = target_runner_for_recipe(recipe["id"], recipe["version"])
    if target_runner_sha256 is not None:
        raw[BOOTSTRAP_CONFIG_KEY][TRANSPARENT_BASELINE_RUNNER_FIELD] = (
            target_runner_sha256
        )
    target_runtime_bundle_sha256 = target_runtime_bundle_for_recipe(
        recipe["id"], recipe["version"]
    )
    if target_runtime_bundle_sha256 is not None:
        raw[BOOTSTRAP_CONFIG_KEY][TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD] = (
            target_runtime_bundle_sha256
        )
    target_worker_runtime_image_digest = target_worker_runtime_image_for_recipe(
        recipe["id"], recipe["version"]
    )
    if target_worker_runtime_image_digest is not None:
        raw[BOOTSTRAP_CONFIG_KEY][TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD] = (
            target_worker_runtime_image_digest
        )
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
        target_runner_sha256 = target_runner_for_recipe(
            plan["recipe"]["id"], recipe_version
        )
        if target_runner_sha256 is None:
            bootstrap.pop(TRANSPARENT_BASELINE_RUNNER_FIELD, None)
        else:
            bootstrap[TRANSPARENT_BASELINE_RUNNER_FIELD] = target_runner_sha256
        target_runtime_bundle_sha256 = target_runtime_bundle_for_recipe(
            plan["recipe"]["id"], recipe_version
        )
        if target_runtime_bundle_sha256 is None:
            bootstrap.pop(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD, None)
        else:
            bootstrap[TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD] = (
                target_runtime_bundle_sha256
            )
        target_worker_runtime_image_digest = target_worker_runtime_image_for_recipe(
            plan["recipe"]["id"], recipe_version
        )
        if target_worker_runtime_image_digest is None:
            bootstrap.pop(TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD, None)
        else:
            bootstrap[TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD] = (
                target_worker_runtime_image_digest
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


def _prepare_optimizer_repair_store_case(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutation: str | None = None,
    insert_receipt: bool = True,
    target_version_ids_by_recipe: dict[str, str] | None = None,
) -> tuple[StrategyStore, list[dict], dict[str, str]]:
    """Create the exact failed v7 family and a governed v8 replacement."""

    from quant_platform import strategy_store as strategy_store_module

    calendar = _calendar()
    current_plans, _ = _plans(calendar)
    source_plans, _ = _retarget_plans(
        current_plans,
        dataset="same-daily",
        identity="a" * 64,
        lineage="b" * 64,
        recipe_version="qlib-rdagent-single-mainline-2026-08-30-v7",
    )
    store = StrategyStore(database_url)
    families: dict[str, dict] = {}
    source_versions: list[dict] = []
    for plan in source_plans:
        recipe_id = str(plan["recipe"]["id"])
        family = store.create(
            name=f"optimizer-repair:{recipe_id}",
            description="Exact v7 optimizer-applicability failure fixture.",
            benchmark=str(plan["recipe"]["benchmark"]),
            universe=str(plan["recipe"]["universe"]),
            factors=[],
            config=plan["config"],
            actor="test",
            economic_hypothesis_group=f"transparent-public-control:{recipe_id}",
        )
        families[recipe_id] = family
        source_versions.append(family["versions"][0])
    TransparentBaselineLockboxStore(database_url).reserve(
        versions=source_versions,
        dataset="same-daily",
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
    )

    backtest_ids = {
        "short_relative_strength": "f51d7fa2f4fd463e97fd5f6990b3721c",
        "swing_trend": "8090c21aa11546bd9d59f732975afc25",
        "long_quality_value": "9c8a75ac646f452e8a5666bacd708936",
    }
    job_ids = {
        "short_relative_strength": "3ce200c5bcec4129a12312ac04efe269",
        "swing_trend": "1416e51e689945b9abb482967c5a7956",
        "long_quality_value": "0e01042cbeca49b192e351438ce4a969",
    }
    ordered_backtest_ids = iter(
        [backtest_ids[str(plan["recipe"]["id"])] for plan in source_plans]
    )

    class _FixedUuid:
        def __init__(self, value: str) -> None:
            self.hex = value

    source_members: list[dict] = []
    engine = open_database(database_url)
    with monkeypatch.context() as local_patch:
        local_patch.setattr(
            strategy_store_module.uuid,
            "uuid4",
            lambda: _FixedUuid(next(ordered_backtest_ids)),
        )
        for plan, version in zip(source_plans, source_versions, strict=True):
            recipe_id = str(plan["recipe"]["id"])
            backtest_id = backtest_ids[recipe_id]
            artifact = tmp_path / "optimizer-repair" / backtest_id
            backtest = store.create_backtest(
                version_id=str(version["id"]),
                dataset="same-daily",
                periods=plan["formal_periods"],
                artifact_path=artifact,
                trading_dates=calendar,
                dataset_lineage_id="b" * 64,
                dataset_identity_sha256="a" * 64,
            )
            assert backtest["id"] == backtest_id
            artifact.mkdir(parents=True, exist_ok=True)
            manifest = json.dumps(
                {"backtest_id": backtest_id, "recipe_id": recipe_id},
                sort_keys=True,
            ).encode()
            (artifact / "manifest.json").write_bytes(manifest)
            error = (
                "ValueError: optimizer requires 60 complete point-in-time "
                "return observations"
            )
            now = datetime.now(UTC)
            with engine.begin() as connection:
                connection.execute(
                    insert(jobs).values(
                        id=job_ids[recipe_id],
                        kind="strategy_backtest",
                        idempotency_key=f"optimizer-repair:{backtest_id}",
                        status="failed",
                        payload_json={"backtest_id": backtest_id},
                        progress_json=None,
                        log_path=str(tmp_path / f"{backtest_id}.log"),
                        exit_code=1,
                        error=error,
                        attempts=1,
                        max_attempts=1,
                        next_attempt_at=None,
                        cancel_requested_at=None,
                        created_at=now,
                        started_at=now,
                        finished_at=now,
                    )
                )
            store.attach_job(backtest_id, job_ids[recipe_id])
            store.mark_backtest(backtest_id, "failed", error=error)
            source_members.append(
                {
                    "backtest_id": backtest_id,
                    "strategy_version_id": str(version["id"]),
                    "job_id": job_ids[recipe_id],
                    "dataset": "same-daily",
                    "periods": dict(plan["formal_periods"]),
                    "status": "failed",
                    "job_status": "failed",
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

    receipt = build_optimizer_applicability_receipt(source_members)
    if insert_receipt:
        with engine.begin() as connection:
            connection.execute(
                insert(audit_events).values(
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
            )
    if mutation == "source_metrics":
        first_artifact = (
            tmp_path
            / "optimizer-repair"
            / backtest_ids["short_relative_strength"]
            / "metrics.json"
        )
        first_artifact.write_text('{"return": 0.1}', encoding="utf-8")

    target_dataset = "same-daily" if mutation != "data" else "changed-daily"
    target_identity = "a" * 64 if mutation != "data" else "c" * 64
    target_lineage = "b" * 64 if mutation != "data" else "d" * 64
    target_plans, _ = _retarget_plans(
        current_plans,
        dataset=target_dataset,
        identity=target_identity,
        lineage=target_lineage,
        recipe_version=OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION,
        change_economic_rule=mutation == "economic",
    )
    if mutation == "oos":
        changed = target_plans[0]
        changed_end = (
            date.fromisoformat(changed["formal_periods"]["end"]) + timedelta(days=1)
        ).isoformat()
        changed["formal_periods"]["end"] = changed_end
        changed["base_config"][BOOTSTRAP_CONFIG_KEY]["formal_periods"][
            "end"
        ] = changed_end
        changed["lockbox_member"] = build_lockbox_member(
            config=changed["base_config"],
            formal_periods=changed["formal_periods"],
        )
        target_lockbox = build_joint_lockbox(
            dataset=target_dataset,
            dataset_identity_sha256=target_identity,
            dataset_lineage_id=target_lineage,
            members=[plan["lockbox_member"] for plan in target_plans],
        )
        for plan in target_plans:
            plan["config"] = _normalize_multifactor_contract(
                {**plan["base_config"], LOCKBOX_CONFIG_KEY: target_lockbox},
                factor_count=0,
                creating_family=True,
            )

    def create_target_versions() -> list[dict]:
        result: list[dict] = []
        for plan in target_plans:
            recipe_id = str(plan["recipe"]["id"])
            result.append(
                store.create_version_if_absent(
                    str(families[recipe_id]["id"]),
                    benchmark=str(plan["recipe"]["benchmark"]),
                    universe=str(plan["recipe"]["universe"]),
                    factors=[],
                    config=plan["config"],
                    actor="test",
                )
            )
        return result

    if target_version_ids_by_recipe is None:
        target_versions = create_target_versions()
    else:
        ordered_target_ids = iter(
            [
                target_version_ids_by_recipe[str(plan["recipe"]["id"])]
                for plan in target_plans
            ]
        )
        with monkeypatch.context() as local_patch:
            local_patch.setattr(
                strategy_store_module.uuid,
                "uuid4",
                lambda: _FixedUuid(next(ordered_target_ids)),
            )
            target_versions = create_target_versions()
    return store, target_versions, {
        "dataset": target_dataset,
        "identity": target_identity,
        "lineage": target_lineage,
    }


@pytest.mark.no_database
def test_joint_lockbox_requires_exact_three_members_and_detects_tampering() -> None:
    plans, lockbox = _plans(_calendar())

    assert validate_joint_lockbox(lockbox) == lockbox
    for plan in plans:
        assert plan["recipe_sha256"] == canonical_sha256(plan["recipe"])
        assert len(_feature_set(plan["recipe"])["features"]) == len(
            plan["recipe"].get("factor_baseline") or []
        )
    with pytest.raises(ValueError, match="all three"):
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
def test_v2_optimizer_repair_accepts_only_exact_failed_production_attempts() -> None:
    periods = {
        "historical_start": "2008-01-02",
        "historical_end": "2024-01-19",
        "start": "2024-01-22",
        "end": "2026-02-26",
    }
    members = []
    for index, backtest_id in enumerate(
        sorted(OPTIMIZER_APPLICABILITY_SOURCE_BACKTEST_IDS), start=1
    ):
        members.append(
            {
                "backtest_id": backtest_id,
                "strategy_version_id": f"{index:x}" * 32,
                "job_id": f"{index + 3:x}" * 32,
                "dataset": "cn-20080101-20260828-v6-79a88b3",
                "periods": periods,
                "status": "failed",
                "job_status": "failed",
                "error": (
                    "ValueError: optimizer requires 60 complete point-in-time "
                    "return observations"
                ),
                "metrics_absent": True,
                "result_absent": True,
                "files": [
                    {
                        "path": "manifest.json",
                        "bytes": 10 + index,
                        "sha256": f"{index + 6:x}" * 64,
                    },
                    {
                        "path": "baseline/composite.parquet",
                        "bytes": 20 + index,
                        "sha256": f"{index + 9:x}" * 64,
                    },
                ],
            }
        )

    receipt = build_optimizer_applicability_receipt(members)
    assert receipt["contract_version"] == PRE_RESULT_REPAIR_CONTRACT_VERSION_V2
    assert receipt["repair_generation"] == OPTIMIZER_APPLICABILITY_REPAIR_GENERATION
    assert receipt["source_release_commit"] == OPTIMIZER_APPLICABILITY_SOURCE_COMMIT
    assert receipt["target_recipe_version"] == (
        OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION
    )
    assert receipt[TRANSPARENT_BASELINE_RUNNER_FIELD] == (
        OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
    )
    assert receipt["reason_codes"] == [OPTIMIZER_APPLICABILITY_REASON]

    result = deepcopy(receipt)
    result["members"][0]["files"].append(
        {"path": "metrics.json", "bytes": 1, "sha256": "f" * 64}
    )
    result_payload = dict(result)
    result_payload.pop("receipt_sha256")
    result["receipt_sha256"] = canonical_sha256(result_payload)
    with pytest.raises(ValueError, match="result artifact"):
        validate_pre_result_repair_receipt(result)

    metrics = deepcopy(receipt)
    metrics["members"][0]["metrics_absent"] = False
    metrics_payload = dict(metrics)
    metrics_payload.pop("receipt_sha256")
    metrics["receipt_sha256"] = canonical_sha256(metrics_payload)
    with pytest.raises(ValueError, match="had a result"):
        validate_pre_result_repair_receipt(metrics)

    wrong_marker = deepcopy(receipt)
    wrong_marker["members"][0]["error"] = "ValueError: arbitrary runner failure"
    marker_payload = dict(wrong_marker)
    marker_payload.pop("receipt_sha256")
    wrong_marker["receipt_sha256"] = canonical_sha256(marker_payload)
    with pytest.raises(ValueError, match="error is not exact"):
        validate_pre_result_repair_receipt(wrong_marker)

    generic_retry = deepcopy(receipt)
    generic_retry["members"][0]["backtest_id"] = "a" * 32
    generic_payload = dict(generic_retry)
    generic_payload.pop("receipt_sha256")
    generic_retry["receipt_sha256"] = canonical_sha256(generic_payload)
    with pytest.raises(ValueError, match="backtests are not allowlisted"):
        validate_pre_result_repair_receipt(generic_retry)

    wrong_runner = deepcopy(receipt)
    wrong_runner[TRANSPARENT_BASELINE_RUNNER_FIELD] = "f" * 64
    runner_payload = dict(wrong_runner)
    runner_payload.pop("receipt_sha256")
    wrong_runner["receipt_sha256"] = canonical_sha256(runner_payload)
    with pytest.raises(ValueError, match="source or target is not allowlisted"):
        validate_pre_result_repair_receipt(wrong_runner)


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
                    "provenance": {
                        "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
                    },
                },
                {
                    "name": "latest-broken",
                    "ready": True,
                    "frequency": "day",
                    "end_date": "2026-08-28",
                    "provenance": {
                        "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
                    },
                },
            ],
            anchored_name=None,
        )


@pytest.mark.no_database
def test_current_recipe_does_not_fall_back_from_newer_legacy_daily_contract(
    monkeypatch,
) -> None:
    from quant_platform import transparent_baseline_bootstrap as module

    selected: list[str] = []

    def validate(dataset: dict) -> dict:
        selected.append(str(dataset["name"]))
        if (
            dataset["provenance"]["field_contract_version"]
            != DAILY_QLIB_FIELD_CONTRACT_VERSION
        ):
            raise ValueError(
                "current transparent baselines require the fail-closed daily Qlib "
                "field contract; rebuild the dataset"
            )
        return dataset

    monkeypatch.setattr(module, "_validate_dataset", validate)
    current = {
        "name": "cn-current-v6",
        "ready": True,
        "frequency": "day",
        "end_date": "2026-08-28",
        "provenance": {
            "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
        },
    }
    legacy = {
        "name": "zz-newer-name-but-v5",
        "ready": True,
        "frequency": "day",
        "end_date": "2026-08-28",
        "provenance": {
            "field_contract_version": "daily-qlib-field-v5-governed-domestic-etf",
        },
    }

    with pytest.raises(ValueError, match="current transparent baselines require"):
        _select_dataset([current, legacy], anchored_name=None)
    assert selected == [legacy["name"]]


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
        "status": "draft",
        "promotion_stage": None,
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
    family = {"id": "same-family", "status": "draft", "versions": [old]}

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
                "status": "draft",
                "promotion_stage": None,
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
            assert payload[TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD] == (
                _WORKER_IMAGE_DIGEST
            )
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
        version={
            "id": "baseline-version",
            "config": {
                BOOTSTRAP_CONFIG_KEY: {
                    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: (
                        _WORKER_IMAGE_DIGEST
                    )
                }
            },
        },
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
        bootstrap.pop(TRANSPARENT_BASELINE_RUNNER_FIELD, None)
        bootstrap.pop(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD, None)
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


def test_v2_optimizer_repair_opens_new_same_lineage_oos_rows(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, target_versions, target = _prepare_optimizer_repair_store_case(
        database_url,
        tmp_path,
        monkeypatch,
    )
    result = TransparentBaselineLockboxStore(database_url).reserve(
        versions=target_versions,
        dataset=target["dataset"],
        dataset_identity_sha256=target["identity"],
        dataset_lineage_id=target["lineage"],
    )

    assert result["pre_result_repair"]["repair_generation"] == (
        OPTIMIZER_APPLICABILITY_REPAIR_GENERATION
    )
    assert {item["status"] for item in result["members"]} == {"reserved"}
    assert result["scope"].startswith("lineage:" + "b" * 64 + ":repair:")
    engine = open_database(database_url)
    with engine.connect() as connection:
        vintages = connection.execute(select(oos_vintages)).all()
        repair = connection.execute(
            select(transparent_baseline_pre_result_repairs)
        ).one()
    assert len(vintages) == 6
    assert len({str(row.scope) for row in vintages}) == 2
    assert str(repair.source_dataset_lineage_id) == "b" * 64
    assert str(repair.target_dataset_lineage_id) == "b" * 64


def test_v2_optimizer_repair_consumes_only_its_preregistered_scope(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, target_versions, target = _prepare_optimizer_repair_store_case(
        database_url,
        tmp_path,
        monkeypatch,
    )
    reservation = TransparentBaselineLockboxStore(database_url).reserve(
        versions=target_versions,
        dataset=target["dataset"],
        dataset_identity_sha256=target["identity"],
        dataset_lineage_id=target["lineage"],
    )
    repair_scope = str(reservation["scope"])

    for version in target_versions:
        bootstrap = dict(version["config"][BOOTSTRAP_CONFIG_KEY])
        store.create_backtest(
            version_id=str(version["id"]),
            dataset=target["dataset"],
            periods=dict(bootstrap["formal_periods"]),
            artifact_path=tmp_path / "optimizer-repair-v8" / str(version["id"]),
            trading_dates=_calendar(),
            dataset_lineage_id=target["lineage"],
            dataset_identity_sha256=target["identity"],
        )

    engine = open_database(database_url)
    with engine.connect() as connection:
        vintages = connection.execute(select(oos_vintages)).all()
    source_rows = [row for row in vintages if str(row.scope) != repair_scope]
    repair_rows = [row for row in vintages if str(row.scope) == repair_scope]
    assert len(source_rows) == 3
    assert len(repair_rows) == 3
    assert all(row.consumed_at is not None for row in source_rows)
    assert all(row.consumed_at is not None for row in repair_rows)


def test_v3_packaging_repair_opens_and_consumes_a_new_exact_chain_scope(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_platform import strategy_store as strategy_store_module
    from quant_platform import transparent_baseline_repair as repair_module

    source_by_recipe = {
        "short_relative_strength": "1ca979f22a0d4e2981e6e5c5e478f583",
        "swing_trend": "7b4386b52cf24d6d913dae3b232fbe7f",
        "long_quality_value": "0c5f5a0b66284a66ba68225afd22b623",
    }
    source_version_ids = {
        recipe_id: CANONICAL_LF_PACKAGING_SOURCE_BINDINGS[backtest_id][
            "strategy_version_id"
        ]
        for recipe_id, backtest_id in source_by_recipe.items()
    }
    store, v8_versions, target = _prepare_optimizer_repair_store_case(
        database_url,
        tmp_path,
        monkeypatch,
        target_version_ids_by_recipe=source_version_ids,
    )
    lockboxes = TransparentBaselineLockboxStore(database_url)
    v8_reservation = lockboxes.reserve(
        versions=v8_versions,
        dataset=target["dataset"],
        dataset_identity_sha256=target["identity"],
        dataset_lineage_id=target["lineage"],
    )

    class _FixedUuid:
        def __init__(self, value: str) -> None:
            self.hex = value

    calendar = _calendar()
    ordered_backtests = iter(
        [source_by_recipe[str(version["config"]["recipe_id"])] for version in v8_versions]
    )
    engine = open_database(database_url)
    with monkeypatch.context() as local_patch:
        local_patch.setattr(
            strategy_store_module.uuid,
            "uuid4",
            lambda: _FixedUuid(next(ordered_backtests)),
        )
        for version in v8_versions:
            recipe_id = str(version["config"]["recipe_id"])
            backtest_id = source_by_recipe[recipe_id]
            binding = CANONICAL_LF_PACKAGING_SOURCE_BINDINGS[backtest_id]
            bootstrap = dict(version["config"][BOOTSTRAP_CONFIG_KEY])
            artifact = tmp_path / "packaging-repair-v8" / backtest_id
            artifact.mkdir(parents=True)
            created = store.create_backtest(
                version_id=str(version["id"]),
                dataset=target["dataset"],
                periods=dict(bootstrap["formal_periods"]),
                artifact_path=artifact,
                trading_dates=calendar,
                dataset_lineage_id=target["lineage"],
                dataset_identity_sha256=target["identity"],
            )
            assert created["id"] == backtest_id
            now = datetime.now(UTC)
            with engine.begin() as connection:
                connection.execute(
                    insert(jobs).values(
                        id=binding["job_id"],
                        kind="strategy_backtest",
                        idempotency_key=f"packaging-repair:{backtest_id}",
                        status="failed",
                        payload_json={
                            "backtest_id": backtest_id,
                            "strategy_version_id": str(version["id"]),
                            "dataset": target["dataset"],
                            "periods": dict(bootstrap["formal_periods"]),
                            "transparent_baseline_runner_sha256": (
                                OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
                            ),
                        },
                        progress_json=None,
                        log_path=str(tmp_path / f"{backtest_id}.log"),
                        exit_code=1,
                        error=CANONICAL_LF_PACKAGING_ERROR,
                        attempts=1,
                        max_attempts=1,
                        next_attempt_at=None,
                        cancel_requested_at=None,
                        created_at=now,
                        started_at=now,
                        finished_at=now,
                    )
                )
            store.attach_job(backtest_id, binding["job_id"])
            store.mark_backtest(
                backtest_id,
                "failed",
                error=CANONICAL_LF_PACKAGING_ERROR,
            )

    # This exercises the historical v8-to-v9 receipt after the repository has
    # advanced to the v11 runner.  The current project runner must not be
    # rebound to the sealed v9 identity (and is explicitly rejected by the
    # packaging-repair unit tests), so inject only the archived v9 digest at an
    # explicit historical path for this database lifecycle fixture.
    historical_v9_runner = tmp_path / "historical-v9-run_multifactor_backtest.py"
    historical_v9_runner.write_text("archived v9 runner fixture\n", encoding="utf-8")
    expected_v9_runner_sha256 = target_runner_for_recipe(
        "short_relative_strength", CANONICAL_LF_PACKAGING_TARGET_RECIPE_VERSION
    )
    real_file_sha256 = repair_module._file_sha256
    with monkeypatch.context() as repair_patch:
        repair_patch.setattr(
            repair_module,
            "_file_sha256",
            lambda path: (
                expected_v9_runner_sha256
                if Path(path).resolve() == historical_v9_runner.resolve()
                else real_file_sha256(Path(path))
            ),
        )
        registered = register_canonical_lf_packaging_repair(
            database_url,
            backtest_ids=list(source_by_recipe.values()),
            actor="system:test-v3",
            source_runner_observed_sha256=(
                CANONICAL_LF_PACKAGING_SOURCE_OBSERVED_RUNNER_SHA256
            ),
            target_runner_path=historical_v9_runner,
        )
    assert registered["status"] == "registered"
    assert all(member["files"] == [] for member in registered["receipt"]["members"])

    current_plans, _ = _plans(calendar)
    v9_plans, _ = _retarget_plans(
        current_plans,
        dataset=target["dataset"],
        identity=target["identity"],
        lineage=target["lineage"],
        recipe_version=CANONICAL_LF_PACKAGING_TARGET_RECIPE_VERSION,
    )
    family_by_recipe = {
        str(version["config"]["recipe_id"]): str(version["strategy_id"])
        for version in v8_versions
    }
    v9_versions = [
        store.create_version_if_absent(
            family_by_recipe[str(plan["recipe"]["id"])],
            benchmark=str(plan["recipe"]["benchmark"]),
            universe=str(plan["recipe"]["universe"]),
            factors=[],
            config=plan["config"],
            actor="test",
        )
        for plan in v9_plans
    ]
    v9_reservation = lockboxes.reserve(
        versions=v9_versions,
        dataset=target["dataset"],
        dataset_identity_sha256=target["identity"],
        dataset_lineage_id=target["lineage"],
    )
    assert v9_reservation["scope"] != v8_reservation["scope"]
    assert v9_reservation["pre_result_repair"]["repair_generation"] == (
        "v8-to-v9-canonical-lf-packaging"
    )

    for version in v9_versions:
        bootstrap = dict(version["config"][BOOTSTRAP_CONFIG_KEY])
        store.create_backtest(
            version_id=str(version["id"]),
            dataset=target["dataset"],
            periods=dict(bootstrap["formal_periods"]),
            artifact_path=tmp_path / "packaging-repair-v9" / str(version["id"]),
            trading_dates=calendar,
            dataset_lineage_id=target["lineage"],
            dataset_identity_sha256=target["identity"],
        )

    with engine.connect() as connection:
        vintages = connection.execute(select(oos_vintages)).all()
        repairs = connection.execute(
            select(transparent_baseline_pre_result_repairs)
        ).all()
    assert len(vintages) == 9
    assert len({str(row.scope) for row in vintages}) == 3
    assert len(repairs) == 2
    v9_rows = [row for row in vintages if str(row.scope) == v9_reservation["scope"]]
    assert len(v9_rows) == 3
    assert all(row.consumed_at is not None for row in v9_rows)


def test_optimizer_repair_registration_helper_seals_exact_v7_ids(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, target_versions, target = _prepare_optimizer_repair_store_case(
        database_url,
        tmp_path,
        monkeypatch,
        insert_receipt=False,
    )
    backtest_ids = sorted(OPTIMIZER_APPLICABILITY_SOURCE_BACKTEST_IDS)
    registered = register_optimizer_applicability_repair(
        database_url,
        backtest_ids=backtest_ids,
        actor="system:test-helper",
    )
    repeated = register_optimizer_applicability_repair(
        database_url,
        backtest_ids=backtest_ids,
        actor="system:test-helper",
    )

    assert registered["status"] == "registered"
    assert repeated["status"] == "already_registered"
    assert repeated["audit_event_id"] == registered["audit_event_id"]
    result = TransparentBaselineLockboxStore(database_url).reserve(
        versions=target_versions,
        dataset=target["dataset"],
        dataset_identity_sha256=target["identity"],
        dataset_lineage_id=target["lineage"],
    )
    assert result["pre_result_repair"]["source_backtest_ids"] == backtest_ids


@pytest.mark.parametrize(
    "mutation", ["economic", "data", "oos", "source_metrics"]
)
def test_v2_optimizer_repair_rejects_target_contract_changes(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _, target_versions, target = _prepare_optimizer_repair_store_case(
        database_url,
        tmp_path,
        monkeypatch,
        mutation=mutation,
    )

    with pytest.raises(ValueError, match="exactly one valid pre-result repair receipt"):
        TransparentBaselineLockboxStore(database_url).reserve(
            versions=target_versions,
            dataset=target["dataset"],
            dataset_identity_sha256=target["identity"],
            dataset_lineage_id=target["lineage"],
        )
    engine = open_database(database_url)
    with engine.connect() as connection:
        assert connection.scalar(
            select(func.count()).select_from(
                transparent_baseline_pre_result_repairs
            )
        ) == 0
        assert connection.scalar(select(func.count()).select_from(oos_vintages)) == 3


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


def test_v12_reconcile_uses_only_history_before_prior_transparent_oos(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_platform import transparent_baseline_bootstrap as module

    calendar = _calendar(6000)
    dataset = _research_dataset(calendar, path=tmp_path / "daily-ready")
    current_recipe_version = get_strategy_recipe("short_relative_strength")[
        "version"
    ]
    latest_selection = build_unopened_history_selection(
        calendar_days=calendar,
        current_recipe_version=current_recipe_version,
        prior_batches=[],
    )
    latest_dataset = {
        **dataset,
        "source_calendar": calendar,
        "unopened_history_selection": latest_selection,
    }
    latest_plans = [
        _plan_member(recipe_id=recipe_id, dataset=latest_dataset)
        for recipe_id in TRANSPARENT_RESEARCH_BASELINE_IDS
    ]

    source_recipe_version = "qlib-rdagent-single-mainline-2026-08-30-v11"
    source_plans = deepcopy(latest_plans)
    for plan in source_plans:
        base = deepcopy(plan["base_config"])
        base["recipe_version"] = source_recipe_version
        bootstrap = base[BOOTSTRAP_CONFIG_KEY]
        bootstrap["recipe_version"] = source_recipe_version
        bootstrap["recipe_sha256"] = canonical_sha256(
            {
                "recipe_id": plan["recipe"]["id"],
                "recipe_version": source_recipe_version,
            }
        )
        bootstrap.pop("unopened_history_selection")
        bootstrap[TRANSPARENT_BASELINE_RUNNER_FIELD] = target_runner_for_recipe(
            str(plan["recipe"]["id"]), source_recipe_version
        )
        bootstrap[TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD] = (
            target_runtime_bundle_for_recipe(
                str(plan["recipe"]["id"]), source_recipe_version
            )
        )
        bootstrap.pop(TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD, None)
        plan["base_config"] = _normalize_multifactor_contract(
            base,
            factor_count=0,
            creating_family=True,
        )
        plan["lockbox_member"] = build_lockbox_member(
            config=plan["base_config"],
            formal_periods=plan["formal_periods"],
        )
    source_lockbox = build_joint_lockbox(
        dataset=str(dataset["name"]),
        dataset_identity_sha256=str(dataset["dataset_identity_sha256"]),
        dataset_lineage_id=str(dataset["dataset_lineage_id"]),
        members=[plan["lockbox_member"] for plan in source_plans],
    )
    strategies = StrategyStore(database_url)
    source_versions: list[dict] = []
    for plan in source_plans:
        recipe_id = str(plan["recipe"]["id"])
        config = _normalize_multifactor_contract(
            {**plan["base_config"], LOCKBOX_CONFIG_KEY: source_lockbox},
            factor_count=0,
            creating_family=True,
        )
        family = strategies.create(
            name=FAMILY_NAMES[recipe_id],
            description="Prior transparent v11 final-OOS fixture.",
            benchmark=str(plan["recipe"]["benchmark"]),
            universe=str(plan["recipe"]["universe"]),
            factors=[],
            config=config,
            actor="test",
            economic_hypothesis_group=f"transparent-public-control:{recipe_id}",
        )
        source_versions.append(family["versions"][0])
    lockboxes = TransparentBaselineLockboxStore(database_url)
    source_reservation = lockboxes.reserve(
        versions=source_versions,
        dataset=str(dataset["name"]),
        dataset_identity_sha256=str(dataset["dataset_identity_sha256"]),
        dataset_lineage_id=str(dataset["dataset_lineage_id"]),
    )
    source_vintage_ids = {
        str(item["oos_vintage_id"]) for item in source_reservation["members"]
    }
    earliest_source_start = min(
        str(item["test_start"]) for item in source_reservation["members"]
    )

    monkeypatch.setattr(module, "_validate_dataset", lambda value: dict(value))
    service = TransparentBaselineBootstrapService(
        database_url=database_url,
        data_root=tmp_path,
        dataset_loader=lambda _root: [dataset],
    )
    first = service.reconcile(actor="test-v12-unopened-history")
    repeated = service.reconcile(actor="test-v12-unopened-history")

    assert first["status"] == repeated["status"] == "pending"
    selection = first["unopened_history_selection"]
    assert selection == repeated["unopened_history_selection"]
    assert selection["selection_mode"] == (
        "unopened_history_before_prior_transparent_oos"
    )
    assert selection["earliest_prior_final_oos_start"] == earliest_source_start
    assert selection["selected_calendar_end"] < earliest_source_start
    assert selection["prior_results_or_metrics_read"] is False
    assert selection["prior_windows_treatment"] == (
        "ordinary_historical_validation_only"
    )
    assert len(selection["prior_batches"]) == 1
    assert selection["prior_batches"][0]["batch_sha256"] == source_lockbox[
        "batch_sha256"
    ]
    assert all(
        str(member["formal_periods"]["end"]) < earliest_source_start
        for member in first["members"]
    )
    assert first["joint_lockbox"]["contract_version"] == (
        LOCKBOX_CONTRACT_VERSION_V2
    )
    assert len(first["joint_lockbox"]["members"]) == 3
    assert first["joint_lockbox"]["pre_result_repair"] is None
    assert first["joint_lockbox"]["batch_sha256"] == repeated[
        "joint_lockbox"
    ]["batch_sha256"]
    assert all(
        item["backtest"]["action"] == "reused" for item in repeated["members"]
    )

    recomputed = lockboxes.resolve_unopened_history_selection(
        calendar_days=calendar,
        current_recipe_version=current_recipe_version,
        anchored_selection=None,
    )
    assert recomputed["evidence"] == selection

    engine = open_database(database_url)
    with engine.connect() as connection:
        recorded_vintages = connection.execute(select(oos_vintages)).all()
        assert len(recorded_vintages) == 6
        assert source_vintage_ids <= {str(row.id) for row in recorded_vintages}
        assert connection.scalar(
            select(func.count()).select_from(
                transparent_baseline_pre_result_repairs
            )
        ) == 0


@pytest.mark.no_database
def test_unopened_history_cutoff_fails_when_no_horizon_window_can_mature(
    tmp_path: Path,
) -> None:
    calendar = _calendar(1200)
    prior_members = []
    for index, recipe_id in enumerate(TRANSPARENT_RESEARCH_BASELINE_IDS, start=1):
        recipe = get_strategy_recipe(recipe_id)
        prior_members.append(
            {
                "oos_vintage_id": f"{index:x}" * 32,
                "strategy_version_id": f"{index + 3:x}" * 32,
                "recipe_id": recipe_id,
                "horizon_profile": recipe["horizon"],
                "recipe_version": "prior-v11",
                "test_start": calendar[900],
                "test_end": calendar[1000 + index],
                "first_opened_at": "2026-08-30T00:00:00+00:00",
                "sealed_member_set_sha256": f"{index + 6:x}" * 64,
            }
        )
    prior_batch = {
        "batch_sha256": "f" * 64,
        "recipe_version": "prior-v11",
        "earliest_final_oos_start": calendar[900],
        "latest_final_oos_end": max(item["test_end"] for item in prior_members),
        "members": sorted(prior_members, key=lambda item: item["recipe_id"]),
    }
    prior_batch["members_sha256"] = canonical_sha256(prior_batch["members"])
    selection = build_unopened_history_selection(
        calendar_days=calendar,
        current_recipe_version=get_strategy_recipe("long_quality_value")["version"],
        prior_batches=[prior_batch],
    )
    dataset = _research_dataset(calendar, path=tmp_path / "insufficient")
    planning_dataset = {
        **dataset,
        "source_calendar": calendar,
        "calendar": calendar[: selection["selected_calendar_trading_days"]],
        "unopened_history_selection": selection,
    }

    with pytest.raises(ValueError, match="frozen final OOS"):
        _plan_member(recipe_id="long_quality_value", dataset=planning_dataset)


@pytest.mark.no_database
def test_reconcile_preserves_every_unavailable_horizon_when_no_plan_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_platform import transparent_baseline_bootstrap as module

    calendar = _calendar(1200)
    dataset = _research_dataset(calendar, path=tmp_path / "all-unavailable")
    current_recipe_version = get_strategy_recipe("short_relative_strength")[
        "version"
    ]
    selection = build_unopened_history_selection(
        calendar_days=calendar,
        current_recipe_version=current_recipe_version,
        prior_batches=[],
    )

    class Lockboxes:
        @staticmethod
        def resolve_preregistered_single_member_repair(**_kwargs: object) -> None:
            return None

        @staticmethod
        def resolve_unopened_history_selection(**_kwargs: object) -> dict:
            return {"calendar": list(calendar), "evidence": dict(selection)}

    def unavailable(*, recipe_id: str, dataset: dict) -> dict:
        assert dataset["unopened_history_selection"] == selection
        evidence = {
            "contract_version": "test-unavailable-horizon-v1",
            "recipe_id": recipe_id,
            "capital_evaluation_eligible": False,
        }
        raise ResearchWindowUnavailableError(
            f"{recipe_id} has no honest window",
            evidence=evidence,
        )

    monkeypatch.setattr(module, "_validate_dataset", lambda value: dict(value))
    monkeypatch.setattr(module, "_plan_member", unavailable)
    service = TransparentBaselineBootstrapService.__new__(
        TransparentBaselineBootstrapService
    )
    service.data_root = tmp_path
    service.dataset_loader = lambda _root: [dataset]
    service.lockboxes = Lockboxes()
    service._existing_families = lambda: {
        recipe_id: None for recipe_id in TRANSPARENT_RESEARCH_BASELINE_IDS
    }

    result = service.reconcile(actor="test-all-unavailable")

    assert result["status"] == "failed"
    assert result["errors"] == [
        "no transparent baseline horizon has enough unopened evidence"
    ]
    assert result["joint_lockbox"]["status"] == "unavailable"
    unavailable_horizons = result["joint_lockbox"]["unavailable_horizons"]
    assert [item["recipe_id"] for item in unavailable_horizons] == list(
        TRANSPARENT_RESEARCH_BASELINE_IDS
    )
    assert all(item["status"] == "unavailable" for item in unavailable_horizons)
    assert [item["recipe_id"] for item in result["members"]] == list(
        TRANSPARENT_RESEARCH_BASELINE_IDS
    )
    assert all(item["state"] == "unavailable" for item in result["members"])
    assert all(item["sleeve_action"] == "remain_in_cash" for item in result["members"])


@pytest.mark.no_database
def test_reconcile_prefers_exact_single_member_repair_history_and_same_oos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_platform import transparent_baseline_bootstrap as module

    calendar = _calendar(4300)
    repaired_calendar = calendar[:2897]
    dataset = _research_dataset(calendar, path=tmp_path / "single-member-repair")
    source_start = repaired_calendar[-257]
    source_end = repaired_calendar[-6]
    repair_selection = {
        "calendar": repaired_calendar,
        "evidence": {
            "contract_version": "transparent-baseline-unopened-history-selection-v2",
            "current_recipe_version": get_strategy_recipe(
                "short_relative_strength"
            )["version"],
            "selected_calendar_end": repaired_calendar[-1],
            "selected_calendar_trading_days": len(repaired_calendar),
            "receipt_sha256": "a" * 64,
            "performance_information_used": False,
        },
        "repair_receipt": {"receipt_sha256": "a" * 64},
        "source_batch_sha256": "b" * 64,
        "source_lockbox": {
            "members": [
                {
                    "recipe_id": "short_relative_strength",
                    "horizon_profile": "short_1_5d",
                    "test_start": source_start,
                    "test_end": source_end,
                }
            ],
            "unavailable_horizons": [
                {"recipe_id": "swing_trend"},
                {"recipe_id": "long_quality_value"},
            ],
        },
    }
    calls = {"repair": 0, "ordinary": 0}
    seen_calendars: list[list[str]] = []

    class Lockboxes:
        @staticmethod
        def resolve_preregistered_single_member_repair(**values: object) -> dict:
            calls["repair"] += 1
            assert values["calendar_days"] == calendar
            return deepcopy(repair_selection)

        @staticmethod
        def resolve_unopened_history_selection(**_values: object) -> dict:
            calls["ordinary"] += 1
            raise AssertionError("exact preregistered repair must bypass ordinary left shift")

    def plan_member(*, recipe_id: str, dataset: dict) -> dict:
        seen_calendars.append(list(dataset["calendar"]))
        if recipe_id == "short_relative_strength":
            return {
                "lockbox_member": {
                    "recipe_id": recipe_id,
                    "horizon_profile": "short_1_5d",
                    "test_start": source_start,
                    "test_end": source_end,
                }
            }
        evidence = {
            "contract_version": "test-source-unavailable-v1",
            "recipe_id": recipe_id,
            "capital_evaluation_eligible": False,
        }
        raise ResearchWindowUnavailableError(
            f"{recipe_id} remains unavailable",
            evidence=evidence,
        )

    monkeypatch.setattr(module, "_select_dataset", lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr(module, "_plan_member", plan_member)

    def stop_after_planning(**_kwargs: object) -> dict:
        raise ValueError("stop after exact repair planning")

    monkeypatch.setattr(module, "build_joint_lockbox", stop_after_planning)
    service = TransparentBaselineBootstrapService.__new__(
        TransparentBaselineBootstrapService
    )
    service.data_root = tmp_path
    service.dataset_loader = lambda _root: [dataset]
    service.lockboxes = Lockboxes()
    service._existing_families = lambda: {
        recipe_id: None for recipe_id in TRANSPARENT_RESEARCH_BASELINE_IDS
    }

    result = service.reconcile(actor="test-exact-single-member-repair")

    assert calls == {"repair": 1, "ordinary": 0}
    assert seen_calendars == [repaired_calendar] * 3
    assert result["unopened_history_selection"] == repair_selection["evidence"]
    assert result["errors"] == ["stop after exact repair planning"]

    changed = deepcopy(repair_selection)
    changed_plan = {
        "lockbox_member": {
            "recipe_id": "short_relative_strength",
            "horizon_profile": "short_1_5d",
            "test_start": calendar[-300],
            "test_end": source_end,
        }
    }
    with pytest.raises(ValueError, match="changed the frozen short final OOS"):
        module._require_preregistered_single_member_repair_oos(
            repair_selection=changed,
            plans=[changed_plan],
            unavailable_horizons=[
                {"recipe_id": "swing_trend"},
                {"recipe_id": "long_quality_value"},
            ],
        )


@pytest.mark.no_database
def test_partial_v12_lockbox_preregisters_short_swing_and_seals_long_cash(
    tmp_path: Path,
) -> None:
    calendar = _pre_2022_server_calendar()
    dataset = _research_dataset(calendar, path=tmp_path / "server-pre-2022")
    selection = build_unopened_history_selection(
        calendar_days=calendar,
        current_recipe_version=get_strategy_recipe("short_relative_strength")[
            "version"
        ],
        prior_batches=[],
    )
    planning_dataset = {
        **dataset,
        "source_calendar": calendar,
        "calendar": calendar,
        "unopened_history_selection": selection,
    }
    plans = [
        _plan_member(recipe_id=recipe_id, dataset=planning_dataset)
        for recipe_id in ("short_relative_strength", "swing_trend")
    ]
    with pytest.raises(ResearchWindowUnavailableError) as captured:
        _plan_member(recipe_id="long_quality_value", dataset=planning_dataset)
    evidence = dict(captured.value.evidence)
    unavailable = {
        "recipe_id": "long_quality_value",
        "horizon_profile": "long_1_3y",
        "status": "unavailable",
        "reason": str(captured.value),
        "evidence": evidence,
        "evidence_sha256": canonical_sha256(evidence),
    }

    lockbox = build_joint_lockbox(
        dataset=str(dataset["name"]),
        dataset_identity_sha256=str(dataset["dataset_identity_sha256"]),
        dataset_lineage_id=str(dataset["dataset_lineage_id"]),
        members=[plan["lockbox_member"] for plan in plans],
        unopened_history_selection=selection,
        unavailable_horizons=[unavailable],
    )

    assert validate_joint_lockbox(lockbox) == lockbox
    assert lockbox["contract_version"] == LOCKBOX_CONTRACT_VERSION_V3
    assert [item["recipe_id"] for item in lockbox["members"]] == [
        "short_relative_strength",
        "swing_trend",
    ]
    assert lockbox["unavailable_horizons"] == [unavailable]
    for plan in plans:
        config = _normalize_multifactor_contract(
            {**dict(plan["base_config"]), LOCKBOX_CONFIG_KEY: lockbox},
            factor_count=0,
            creating_family=True,
        )
        link = lockbox_member_link(config)
        assert link is not None
        assert validate_lockbox_link(link) == link
        assert link["contract_version"] == LOCKBOX_LINK_VERSION_V2
        assert len(link["member_sha256s"]) == 2

    falsely_unavailable = deepcopy(unavailable)
    falsely_unavailable["evidence"]["capital_evaluation_eligible"] = True
    falsely_unavailable["evidence_sha256"] = canonical_sha256(
        falsely_unavailable["evidence"]
    )
    with pytest.raises(
        ValueError,
        match="unavailable baseline horizon evidence is invalid",
    ):
        build_joint_lockbox(
            dataset=str(dataset["name"]),
            dataset_identity_sha256=str(dataset["dataset_identity_sha256"]),
            dataset_lineage_id=str(dataset["dataset_lineage_id"]),
            members=[plan["lockbox_member"] for plan in plans],
            unopened_history_selection=selection,
            unavailable_horizons=[falsely_unavailable],
        )


@pytest.mark.no_database
def test_swing_oos_is_unavailable_when_native_price_limits_leave_fewer_than_504_sessions(
    tmp_path: Path,
) -> None:
    calendar = _calendar()
    dataset = _research_dataset(calendar, path=tmp_path / "native-limit-boundary")
    selection = build_unopened_history_selection(
        calendar_days=calendar,
        current_recipe_version=get_strategy_recipe("swing_trend")["version"],
        prior_batches=[],
    )
    planning_dataset = {
        **dataset,
        "source_calendar": calendar,
        "calendar": calendar,
        "unopened_history_selection": selection,
    }
    unrestricted = _plan_member(
        recipe_id="swing_trend",
        dataset=planning_dataset,
    )
    original_start = str(unrestricted["formal_periods"]["start"])
    original_start_index = calendar.index(original_start)
    first_native_session = calendar[original_start_index + 10]
    planning_dataset["provenance"]["execution_controls"][
        "native_complete_from"
    ] = first_native_session

    with pytest.raises(ResearchWindowUnavailableError) as captured:
        _plan_member(recipe_id="swing_trend", dataset=planning_dataset)

    evidence = captured.value.evidence
    assert evidence["capital_evaluation_eligible"] is False
    assert evidence["capital_evaluation_unavailable_reason"] == (
        "insufficient_native_execution_controlled_sessions_before_immutable_cutoff"
    )
    assert evidence["rejected_test_start"] == original_start
    assert evidence["proposed_test_start"] == first_native_session
    assert evidence["required_sealed_oos_sessions"] == 504
    assert evidence["available_native_controlled_oos_sessions"] == 494


@pytest.mark.no_database
def test_native_price_limit_boundary_uses_complete_remaining_oos_when_available(
    tmp_path: Path,
) -> None:
    calendar = _calendar()
    dataset = _research_dataset(calendar, path=tmp_path / "native-limit-shift")
    dataset["provenance"]["field_contract_version"] = (
        "daily-qlib-field-v5-governed-domestic-etf"
    )
    boundary = calendar[1010]
    maturity_cutoff = calendar[1600]
    dataset["provenance"]["execution_controls"]["native_complete_from"] = boundary
    periods = {
        "train_start": calendar[0],
        "train_end": calendar[700],
        "valid_start": calendar[800],
        "valid_end": calendar[999],
        "test_start": calendar[1000],
        "test_end": maturity_cutoff,
    }
    evidence = {
        "latest_mature_label_sessions": {
            "21": calendar[1705],
            "63": calendar[1663],
            "126": maturity_cutoff,
        }
    }

    adjusted = _require_native_formal_oos(
        dataset=dataset,
        periods=periods,
        evidence=evidence,
        horizon_profile="swing_1_6m",
    )

    assert adjusted["test_start"] == boundary
    assert adjusted["test_end"] == maturity_cutoff
    assert sum(
        adjusted["test_start"] <= day <= adjusted["test_end"]
        for day in calendar
    ) >= 504
    assert adjusted == _require_native_formal_oos(
        dataset=dataset,
        periods=periods,
        evidence=evidence,
        horizon_profile="swing_1_6m",
    )


@pytest.mark.no_database
def test_plan_member_rebinds_shifted_native_control_oos_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_platform import transparent_baseline_bootstrap as module

    calendar = _calendar()
    dataset = _research_dataset(calendar, path=tmp_path / "native-limit-plan-shift")
    selection = build_unopened_history_selection(
        calendar_days=calendar,
        current_recipe_version=get_strategy_recipe("swing_trend")["version"],
        prior_batches=[],
    )
    planning_dataset = {
        **dataset,
        "source_calendar": calendar,
        "calendar": calendar,
        "unopened_history_selection": selection,
    }
    recipe = get_strategy_recipe("swing_trend")
    feature_set = _feature_set(recipe)
    resolved, evidence = module.resolve_research_window_contract(
        dict(planning_dataset),
        calendar,
        horizon_profile="swing_1_6m",
        feature_set=feature_set,
        universe=str(recipe["universe"]),
    )
    native_start = str(resolved["test_start"])
    fake_periods = {
        **resolved,
        "test_start": calendar[calendar.index(native_start) - 10],
    }
    original_contract = evidence["research_window_contract"]
    effective_calendar = [
        day
        for day in calendar
        if original_contract["calendar_start"]
        <= day
        <= original_contract["calendar_end"]
    ]
    fake_contract = module.build_research_window_contract(
        dataset=planning_dataset,
        calendar_days=effective_calendar,
        periods=fake_periods,
        period_resolution=evidence,
        horizon_profile="swing_1_6m",
        feature_set=feature_set,
        universe=str(recipe["universe"]),
    )
    fake_evidence = {
        **evidence,
        "final_test_trading_days": fake_contract.sealed_oos_sessions,
        "research_window_contract": fake_contract.to_dict(),
        "research_window_contract_sha256": fake_contract.sha256,
    }
    planning_dataset["provenance"]["field_contract_version"] = (
        "daily-qlib-field-v5-governed-domestic-etf"
    )
    planning_dataset["provenance"]["execution_controls"][
        "native_complete_from"
    ] = native_start
    monkeypatch.setattr(
        module,
        "resolve_research_window_contract",
        lambda *_args, **_kwargs: (fake_periods, fake_evidence),
    )

    plan = _plan_member(recipe_id="swing_trend", dataset=planning_dataset)

    assert plan["periods"]["test_start"] == native_start
    assert plan["periods"]["test_end"] == resolved["test_end"]
    rebound = plan["base_config"][BOOTSTRAP_CONFIG_KEY]["research_window_contract"]
    assert rebound["periods"] == plan["periods"]
    assert rebound["sealed_oos_sessions"] == 504
    assert plan["research_window_contract_sha256"] == canonical_sha256(rebound)


def test_reconcile_starts_short_and_marks_native_control_ineligible_horizons_unavailable(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_platform import transparent_baseline_bootstrap as module

    calendar = _calendar()
    dataset = _research_dataset(calendar, path=tmp_path / "native-limit-reconcile")
    selection = build_unopened_history_selection(
        calendar_days=calendar,
        current_recipe_version=get_strategy_recipe("swing_trend")["version"],
        prior_batches=[],
    )
    planning_dataset = {
        **dataset,
        "source_calendar": calendar,
        "calendar": calendar,
        "unopened_history_selection": selection,
    }
    short = _plan_member(
        recipe_id="short_relative_strength",
        dataset=planning_dataset,
    )
    swing = _plan_member(recipe_id="swing_trend", dataset=planning_dataset)
    boundary_index = calendar.index(str(swing["formal_periods"]["start"])) + 10
    boundary = calendar[boundary_index]
    assert boundary < str(short["formal_periods"]["start"])
    dataset["provenance"]["execution_controls"]["native_complete_from"] = boundary

    monkeypatch.setattr(module, "_validate_dataset", lambda value: dict(value))
    service = TransparentBaselineBootstrapService(
        database_url=database_url,
        data_root=tmp_path,
        dataset_loader=lambda _root: [dataset],
    )
    result = service.reconcile(actor="test-native-limit-boundary")

    assert result["status"] == "pending"
    members = {str(item["horizon_profile"]): item for item in result["members"]}
    assert members["short_1_5d"]["state"] == "formal_backtest_pending"
    for horizon in ("swing_1_6m", "long_1_3y"):
        assert members[horizon]["state"] == "unavailable"
        assert members[horizon]["sleeve_action"] == "remain_in_cash"
        assert members[horizon]["unavailable_evidence"][
            "capital_evaluation_unavailable_reason"
        ] == (
            "insufficient_native_execution_controlled_sessions_before_immutable_cutoff"
        )
    assert len(result["joint_lockbox"]["members"]) == 1
    assert len(result["joint_lockbox"]["unavailable_horizons"]) == 2
    with open_database(database_url).connect() as connection:
        assert connection.scalar(select(func.count()).select_from(oos_vintages)) == 1
        assert connection.scalar(select(func.count()).select_from(strategy_versions)) == 1


def test_v12_reconcile_starts_available_horizons_and_keeps_long_sleeve_cash(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_platform import transparent_baseline_bootstrap as module

    calendar = _pre_2022_server_calendar()
    dataset = _research_dataset(calendar, path=tmp_path / "server-pre-2022-db")
    monkeypatch.setattr(module, "_validate_dataset", lambda value: dict(value))
    service = TransparentBaselineBootstrapService(
        database_url=database_url,
        data_root=tmp_path,
        dataset_loader=lambda _root: [dataset],
    )

    first = service.reconcile(actor="test-partial-v12")
    repeated = service.reconcile(actor="test-partial-v12")

    assert first["status"] == repeated["status"] == "pending"
    assert first["joint_lockbox"]["contract_version"] == (
        LOCKBOX_CONTRACT_VERSION_V3
    )
    assert len(first["joint_lockbox"]["members"]) == 2
    members = {str(item["horizon_profile"]): item for item in first["members"]}
    assert members["short_1_5d"]["state"] == "formal_backtest_pending"
    assert members["swing_1_6m"]["state"] == "formal_backtest_pending"
    assert members["long_1_3y"]["state"] == "unavailable"
    assert members["long_1_3y"]["sleeve_action"] == "remain_in_cash"
    assert members["long_1_3y"]["strategy_version_id"] is None
    assert members["long_1_3y"]["unavailable_evidence"][
        "capital_evaluation_eligible"
    ] is False
    assert all(
        repeated_member["backtest"]["action"] == "reused"
        for repeated_member in repeated["members"]
        if repeated_member["state"] != "unavailable"
    )
    with open_database(database_url).connect() as connection:
        assert connection.scalar(select(func.count()).select_from(oos_vintages)) == 2
        assert connection.scalar(select(func.count()).select_from(strategy_versions)) == 2

    next_recipe = service.lockboxes.resolve_unopened_history_selection(
        calendar_days=calendar,
        current_recipe_version="qlib-rdagent-single-mainline-future-test-v13",
    )
    next_evidence = next_recipe["evidence"]
    available_starts = [
        str(item["formal_periods"]["start"])
        for item in first["members"]
        if item["state"] != "unavailable"
    ]
    assert next_evidence["earliest_prior_final_oos_start"] == min(
        available_starts
    )
    assert next_evidence["selected_calendar_end"] < min(available_starts)
    assert len(next_evidence["prior_batches"]) == 1
    assert len(next_evidence["prior_batches"][0]["members"]) == 2


@pytest.mark.no_database
def test_v12_unopened_history_selection_ignores_its_own_rows_and_jointly_locks(
    tmp_path: Path,
) -> None:
    calendar = _calendar(6000)
    current_version = get_strategy_recipe("short_relative_strength")["version"]
    prior_version = "qlib-rdagent-single-mainline-2026-08-30-v11"
    rows: list[SimpleNamespace] = []
    versions: list[SimpleNamespace] = []

    def add_batch(recipe_version: str, *, start_index: int, ordinal: int) -> None:
        batch_sha256 = canonical_sha256(
            {"recipe_version": recipe_version, "ordinal": ordinal}
        )
        member_hashes = [
            canonical_sha256({"batch": batch_sha256, "member": index})
            for index in range(3)
        ]
        for index, recipe_id in enumerate(TRANSPARENT_RESEARCH_BASELINE_IDS):
            recipe = get_strategy_recipe(recipe_id)
            strategy_version_id = f"{ordinal * 10 + index + 1:032x}"
            oos_vintage_id = f"{ordinal * 10 + index + 4:032x}"
            link = {
                "contract_version": "transparent-baseline-joint-lockbox-link-v1",
                "batch_sha256": batch_sha256,
                "member_sha256": member_hashes[index],
                "member_sha256s": sorted(member_hashes),
                "recipe_id": recipe_id,
                "horizon_profile": recipe["horizon"],
            }
            sealed = {
                "strategy_version_id": strategy_version_id,
                "transparent_baseline_lockbox": link,
            }
            rows.append(
                SimpleNamespace(
                    id=oos_vintage_id,
                    test_start=date.fromisoformat(calendar[start_index + index]),
                    test_end=date.fromisoformat(calendar[start_index + 300 + index]),
                    first_opened_at=datetime(2026, 8, 30, ordinal, tzinfo=UTC),
                    sealed_candidate_set_json=sealed,
                    sealed_candidate_set_sha256=canonical_sha256(sealed),
                )
            )
            versions.append(
                SimpleNamespace(
                    id=strategy_version_id,
                    config_json={
                        "recipe_id": recipe_id,
                        "recipe_version": recipe_version,
                        "horizon_profile": recipe["horizon"],
                    },
                )
            )

    # Current v12 rows sit earlier than v11 by construction. If retries read
    # their own rows, the cutoff would incorrectly move from 4999 to 4499.
    add_batch(prior_version, start_index=5000, ordinal=1)
    add_batch(current_version, start_index=4500, ordinal=2)
    original_row_ids = [row.id for row in rows]

    class Result:
        def __init__(self, values: list[SimpleNamespace]) -> None:
            self.values = values

        def all(self) -> list[SimpleNamespace]:
            return list(self.values)

    class Connection:
        def __enter__(self) -> Connection:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def execute(statement: object) -> Result:
            selected = tuple(statement.selected_columns)  # type: ignore[attr-defined]
            return Result(versions if len(selected) == 2 else rows)

    class Engine:
        @staticmethod
        def connect() -> Connection:
            return Connection()

    store = TransparentBaselineLockboxStore.__new__(
        TransparentBaselineLockboxStore
    )
    store.engine = Engine()
    first = store.resolve_unopened_history_selection(
        calendar_days=calendar,
        current_recipe_version=current_version,
        anchored_selection=None,
    )
    repeated = store.resolve_unopened_history_selection(
        calendar_days=calendar,
        current_recipe_version=current_version,
        anchored_selection=None,
    )

    assert first == repeated
    assert [row.id for row in rows] == original_row_ids
    evidence = first["evidence"]
    assert evidence["selected_calendar_end"] == calendar[4999]
    assert evidence["earliest_prior_final_oos_start"] == calendar[5000]
    assert evidence["prior_results_or_metrics_read"] is False
    assert evidence["prior_windows_treatment"] == (
        "ordinary_historical_validation_only"
    )
    assert [item["recipe_version"] for item in evidence["prior_batches"]] == [
        prior_version
    ]
    assert "pre_result_repair" not in evidence

    dataset = _research_dataset(calendar, path=tmp_path / "unopened-v12")
    planning_dataset = {
        **dataset,
        "source_calendar": calendar,
        "calendar": first["calendar"],
        "unopened_history_selection": evidence,
    }
    plans = [
        _plan_member(recipe_id=recipe_id, dataset=planning_dataset)
        for recipe_id in TRANSPARENT_RESEARCH_BASELINE_IDS
    ]
    assert all(
        plan["base_config"][BOOTSTRAP_CONFIG_KEY][
            TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD
        ]
        == _WORKER_IMAGE_DIGEST
        for plan in plans
    )
    assert all(
        plan["formal_periods"]["end"] < calendar[5000] for plan in plans
    )
    lockbox = build_joint_lockbox(
        dataset=str(dataset["name"]),
        dataset_identity_sha256=str(dataset["dataset_identity_sha256"]),
        dataset_lineage_id=str(dataset["dataset_lineage_id"]),
        members=[plan["lockbox_member"] for plan in plans],
        unopened_history_selection=evidence,
    )
    assert validate_joint_lockbox(lockbox) == lockbox
    assert lockbox["contract_version"] == LOCKBOX_CONTRACT_VERSION_V2
    assert len(lockbox["members"]) == 3
    assert lockbox["unopened_history_selection"] == evidence
    assert "pre_result_repair" not in lockbox


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

    def _selected_history_plan(*, recipe_id: str, dataset: dict) -> dict:
        plan = deepcopy(by_recipe[recipe_id])
        base_config = deepcopy(plan["base_config"])
        base_config[BOOTSTRAP_CONFIG_KEY]["unopened_history_selection"] = deepcopy(
            dataset["unopened_history_selection"]
        )
        base_config = _normalize_multifactor_contract(
            base_config,
            factor_count=0,
            creating_family=True,
        )
        return {
            **plan,
            "base_config": base_config,
            "lockbox_member": build_lockbox_member(
                config=base_config,
                formal_periods=plan["formal_periods"],
            ),
        }

    monkeypatch.setattr(
        module,
        "_plan_member",
        _selected_history_plan,
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
