from __future__ import annotations

from pathlib import Path

import pytest

from quant_platform.strategy_recipes import get_strategy_recipe
from quant_platform.transparent_baseline_bootstrap import (
    TransparentBaselineBootstrapService,
)
from quant_platform.transparent_baseline_lockbox import LOCKBOX_CONFIG_KEY


class _NoopPromotions:
    calls = 0

    def prepare_paper_stage(self, version_id: str, *, actor: str) -> dict:
        self.calls += 1
        return {
            "strategy_version_id": version_id,
            "created_by": actor,
            "status": "awaiting_simulation",
        }


@pytest.mark.no_database
@pytest.mark.parametrize(
    "status", ["watch", "paused", "restricted", "suspended", "rejected", "retired"]
)
def test_terminal_version_is_never_reapproved_by_bootstrap(
    tmp_path: Path,
    status: str,
) -> None:
    class TerminalStrategies:
        approve_calls = 0

        @staticmethod
        def get_version(version_id: str) -> dict:
            assert version_id == "baseline-version"
            return {"status": status, "promotion_stage": None}

        def approve(self, *_args, **_kwargs) -> None:
            self.approve_calls += 1

    strategies = TerminalStrategies()
    promotions = _NoopPromotions()
    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        strategies=strategies,
        jobs=object(),
        promotions=promotions,
        lockboxes=object(),
    )

    result = service._advance_paper(
        version_id="baseline-version",
        backtest={"status": "succeeded"},
        actor="system:reconcile-test",
    )

    assert result == {
        "state": "governed_no_op",
        "paper_stage": None,
        "lifecycle": {"status": status, "promotion_stage": None},
    }
    assert strategies.approve_calls == 0
    assert promotions.calls == 0


@pytest.mark.no_database
@pytest.mark.parametrize(
    "status", ["watch", "paused", "restricted", "suspended", "rejected", "retired"]
)
def test_terminal_exact_version_is_not_reused(
    tmp_path: Path,
    status: str,
) -> None:
    recipe = get_strategy_recipe("short_relative_strength")
    expected_config = {"immutable": "current-v8"}
    version = {
        "id": "terminal-version",
        "status": status,
        "promotion_stage": None,
        "benchmark": recipe["benchmark"],
        "universe": recipe["universe"],
        "factors": [],
        "config": expected_config,
        "horizon_profile": recipe["horizon"],
    }
    family = {
        "id": "baseline-family",
        "status": "approved",
        "versions": [version],
    }
    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        strategies=object(),
        jobs=object(),
        promotions=object(),
        lockboxes=object(),
    )

    plan = {"recipe": recipe, "config": expected_config}
    versions = service._ensure_versions(
        plans=[plan],
        families={recipe["id"]: family},
        actor="system:reconcile-test",
    )

    assert versions == [version]
    assert plan["family_action"] == "reused"
    assert plan["lifecycle_action"] == "governed_no_op"
    assert plan["lifecycle"] == {"status": status, "promotion_stage": None}


@pytest.mark.no_database
@pytest.mark.parametrize(
    "promotion_stage",
    ["recommendation_enabled", "watch", "restricted", "suspended", "retired"],
)
def test_governed_promotion_stage_is_a_noop(
    tmp_path: Path,
    promotion_stage: str,
) -> None:
    class GovernedStrategies:
        approve_calls = 0

        @staticmethod
        def get_version(version_id: str) -> dict:
            assert version_id == "baseline-version"
            return {"status": "approved", "promotion_stage": promotion_stage}

        def approve(self, *_args, **_kwargs) -> None:
            self.approve_calls += 1

    strategies = GovernedStrategies()
    promotions = _NoopPromotions()
    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        strategies=strategies,
        jobs=object(),
        promotions=promotions,
        lockboxes=object(),
    )

    result = service._advance_paper(
        version_id="baseline-version",
        backtest={"status": "succeeded"},
        actor="system:reconcile-test",
    )

    assert result["state"] == "governed_no_op"
    assert result["lifecycle"] == {
        "status": "approved",
        "promotion_stage": promotion_stage,
    }
    assert strategies.approve_calls == 0
    assert promotions.calls == 0


@pytest.mark.no_database
def test_unknown_lifecycle_still_fails_closed(tmp_path: Path) -> None:
    class UnknownStrategies:
        @staticmethod
        def get_version(version_id: str) -> dict:
            assert version_id == "baseline-version"
            return {"status": "approved", "promotion_stage": None}

    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        strategies=UnknownStrategies(),
        jobs=object(),
        promotions=object(),
        lockboxes=object(),
    )

    with pytest.raises(ValueError, match="unknown or inconsistent"):
        service._advance_paper(
            version_id="baseline-version",
            backtest={"status": "succeeded"},
            actor="system:reconcile-test",
        )


@pytest.mark.no_database
def test_all_governed_members_reconcile_as_noop_without_backtest_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_platform import transparent_baseline_bootstrap as module
    from quant_platform.strategy_recipes import TRANSPARENT_RESEARCH_BASELINE_IDS

    lockbox = {"contract_version": "test-lockbox"}
    families: dict[str, dict] = {}
    for recipe_id in TRANSPARENT_RESEARCH_BASELINE_IDS:
        recipe = get_strategy_recipe(recipe_id)
        config = {
            "recipe_id": recipe_id,
            "recipe_version": recipe["version"],
            LOCKBOX_CONFIG_KEY: lockbox,
        }
        version = {
            "id": f"version-{recipe_id}",
            "status": "approved",
            "promotion_stage": "recommendation_enabled",
            "benchmark": recipe["benchmark"],
            "universe": recipe["universe"],
            "factors": [],
            "config": config,
            "horizon_profile": recipe["horizon"],
        }
        families[recipe_id] = {
            "id": f"family-{recipe_id}",
            "status": "approved",
            "versions": [version],
        }

    class GovernedStrategies:
        @staticmethod
        def get_by_name(name: str) -> dict:
            recipe_id = next(
                key for key, family_name in module.FAMILY_NAMES.items() if family_name == name
            )
            return families[recipe_id]

    class Lockboxes:
        @staticmethod
        def reserve(**_values) -> dict:
            raise AssertionError("governed no-op must not reopen lockbox reservation")

    monkeypatch.setattr(
        module,
        "_select_dataset",
        lambda _datasets, anchored_name: {
            "name": anchored_name or "daily-ready",
            "start_date": "2008-01-02",
            "end_date": "2026-08-28",
            "trading_days": 4500,
            "dataset_identity_sha256": "a" * 64,
            "dataset_lineage_id": "b" * 64,
        },
    )

    def plan_member(*, recipe_id: str, dataset: dict) -> dict:
        del dataset
        recipe = get_strategy_recipe(recipe_id)
        return {
            "recipe": recipe,
            "base_config": {
                "recipe_id": recipe_id,
                "recipe_version": recipe["version"],
            },
            "lockbox_member": {"recipe_id": recipe_id},
            "research_window_contract_sha256": "c" * 64,
            "formal_periods": {
                "historical_start": "2008-01-02",
                "historical_end": "2023-12-29",
                "start": "2024-01-02",
                "end": "2026-08-28",
            },
        }

    monkeypatch.setattr(module, "_plan_member", plan_member)
    monkeypatch.setattr(module, "build_joint_lockbox", lambda **_values: lockbox)
    monkeypatch.setattr(
        module,
        "_normalize_multifactor_contract",
        lambda values, **_options: dict(values),
    )
    monkeypatch.setattr(module, "lockbox_member_link", lambda _config: None)

    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        dataset_loader=lambda _root: [],
        strategies=GovernedStrategies(),
        jobs=object(),
        promotions=object(),
        lockboxes=Lockboxes(),
    )
    result = service.reconcile(actor="system:reconcile-test")

    assert result["status"] == "no_op"
    assert result["errors"] == []
    assert result["joint_lockbox"] == {
        "status": "governed_no_op",
        "reason": "all transparent baselines are governance controlled",
    }
    assert [member["state"] for member in result["members"]] == [
        "governed_no_op",
        "governed_no_op",
        "governed_no_op",
    ]


@pytest.mark.no_database
def test_fresh_draft_v8_remains_reconcilable(tmp_path: Path) -> None:
    recipe = get_strategy_recipe("short_relative_strength")
    expected_config = {"immutable": "current-v8"}
    version = {
        "id": "fresh-v8-version",
        "status": "draft",
        "promotion_stage": None,
        "benchmark": recipe["benchmark"],
        "universe": recipe["universe"],
        "factors": [],
        "config": expected_config,
        "horizon_profile": recipe["horizon"],
    }
    family = {
        "id": "baseline-family",
        "status": "draft",
        "versions": [version],
    }
    service = TransparentBaselineBootstrapService(
        database_url="unused",
        data_root=tmp_path,
        strategies=object(),
        jobs=object(),
        promotions=object(),
        lockboxes=object(),
    )

    reconciled = service._ensure_versions(
        plans=[{"recipe": recipe, "config": expected_config}],
        families={recipe["id"]: family},
        actor="system:reconcile-test",
    )

    assert reconciled == [version]
