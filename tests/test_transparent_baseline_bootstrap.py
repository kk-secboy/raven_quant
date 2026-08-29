from __future__ import annotations

import json
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import pytest
from governance_fixtures import governed_etf_ready_evidence
from sqlalchemy import func, select

from quant_data.database import (
    backtest_runs,
    jobs,
    oos_vintages,
    open_database,
    strategy_versions,
)
from quant_data.execution_contract import DAILY_QLIB_FIELD_CONTRACT_VERSION
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
