from __future__ import annotations

from pathlib import Path

import pytest

from quant_platform.strategy_recipes import get_strategy_recipe
from quant_platform.transparent_baseline_bootstrap import (
    TransparentBaselineBootstrapService,
)


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
@pytest.mark.parametrize("status", ["paused", "restricted", "suspended", "retired"])
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

    with pytest.raises(ValueError, match="operator/risk controlled"):
        service._advance_paper(
            version_id="baseline-version",
            backtest={"status": "succeeded"},
            actor="system:reconcile-test",
        )

    assert strategies.approve_calls == 0
    assert promotions.calls == 0


@pytest.mark.no_database
@pytest.mark.parametrize("status", ["paused", "restricted", "suspended", "retired"])
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

    with pytest.raises(ValueError, match="operator/risk controlled"):
        service._ensure_versions(
            plans=[{"recipe": recipe, "config": expected_config}],
            families={recipe["id"]: family},
            actor="system:reconcile-test",
        )


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
