from __future__ import annotations

from copy import deepcopy

import pytest

from quant_platform.feature_set_registry import get_feature_set
from quant_platform.horizon_factor_bundle import (
    build_horizon_factor_bundle,
    validate_horizon_factor_bundle,
)
from quant_platform.promotion import (
    MAX_FORWARD_CHALLENGERS_PER_HORIZON,
    _autopilot_paper_accounts_compete,
    require_horizon_challenger_capacity,
)

pytestmark = pytest.mark.no_database


def _binding(*, horizon: str, label: int, feature_set: dict) -> dict:
    return {
        "horizon_profile": horizon,
        "label_horizon_sessions": label,
        "feature_set_id": feature_set["id"],
        "feature_set_sha256": feature_set["definition_sha256"],
        "binding_sha256": "a" * 64,
        "dataset_identity_sha256": "b" * 64,
    }


def _incumbent() -> dict:
    return {
        "kind": "model",
        "candidate_id": "model-incumbent",
        "evidence_sha256": "c" * 64,
    }


def test_alpha_base_and_rdagent_additions_form_one_horizon_identity() -> None:
    alpha360 = get_feature_set("qlib-alpha360")
    factors = [
        {"candidate_id": "rd-factor-1", "code_sha256": "1" * 64},
        {"candidate_id": "rd-factor-2", "code_sha256": "2" * 64},
    ]

    bundle = build_horizon_factor_bundle(
        feature_set=alpha360,
        incremental_factors=factors,
        research_label_binding=_binding(
            horizon="short_1_5d", label=5, feature_set=alpha360
        ),
        incumbent_prediction=_incumbent(),
    )

    assert validate_horizon_factor_bundle(bundle) == bundle
    assert bundle["horizon_profile"] == "short_1_5d"
    assert bundle["primary_label_policy"]["policy_sha256"] == (
        "f90f34e67b4721c0e7b82181007872cc099093e80ea92f8ed3d2d88e3e7adfdc"
    )
    assert bundle["base_feature_set"] == {
        "id": "qlib-alpha360",
        "definition_sha256": alpha360["definition_sha256"],
        "contract_version": alpha360["contract_version"],
        "source": alpha360["source"],
        "feature_count": len(alpha360["features"]),
        "feature_names_sha256": bundle["base_feature_set"][
            "feature_names_sha256"
        ],
    }
    assert bundle["incremental_factors"] == factors
    assert bundle["incremental_challenge"]["comparison"] == (
        "joint_vs_frozen_incumbent"
    )
    assert bundle["authority"] == "research_only"


def test_alpha_base_alone_has_an_initial_horizon_champion_identity() -> None:
    alpha158 = get_feature_set("qlib-alpha158")

    bundle = build_horizon_factor_bundle(
        feature_set=alpha158,
        incremental_factors=[],
        research_label_binding=_binding(
            horizon="long_1_3y", label=252, feature_set=alpha158
        ),
    )

    assert validate_horizon_factor_bundle(bundle) == bundle
    assert bundle["incremental_factors"] == []
    assert bundle["incremental_challenge"] == {
        "mode": "baseline_seed",
        "incumbent_kind": None,
        "incumbent_candidate_id": None,
        "incumbent_evidence_sha256": None,
        "comparison": None,
        "required_ablations": [],
    }
    assert bundle["id"].startswith("horizon-factor-bundle:long_1_3y:")


def test_factor_bundle_identity_is_horizon_specific_and_immutable() -> None:
    alpha158 = get_feature_set("qlib-alpha158")
    factor = [{"candidate_id": "rd-factor", "code_sha256": "3" * 64}]
    first = build_horizon_factor_bundle(
        feature_set=alpha158,
        incremental_factors=factor,
        research_label_binding=_binding(
            horizon="swing_1_6m", label=63, feature_set=alpha158
        ),
        incumbent_prediction=_incumbent(),
    )
    second = build_horizon_factor_bundle(
        feature_set=alpha158,
        incremental_factors=[
            {"candidate_id": "rd-factor-2", "code_sha256": "4" * 64}
        ],
        research_label_binding=_binding(
            horizon="swing_1_6m", label=63, feature_set=alpha158
        ),
        incumbent_prediction=_incumbent(),
    )
    assert first["id"] != second["id"]
    assert first["bundle_sha256"] != second["bundle_sha256"]

    tampered = deepcopy(first)
    tampered["incremental_factors"][0]["code_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="digest"):
        validate_horizon_factor_bundle(tampered)

    with pytest.raises(ValueError, match="primary horizon label"):
        build_horizon_factor_bundle(
            feature_set=alpha158,
            incremental_factors=factor,
            research_label_binding=_binding(
                horizon="swing_1_6m", label=126, feature_set=alpha158
            ),
            incumbent_prediction=_incumbent(),
        )


class _CapacityConnection:
    def __init__(self, challenger_ids: list[str]) -> None:
        self.challenger_ids = challenger_ids
        self.locked = False
        self.scalar_sql = ""

    def execute(self, _statement, parameters=None):
        if parameters is not None:
            self.locked = True
        return None

    def scalars(self, statement):
        self.scalar_sql = str(
            statement.compile(compile_kwargs={"literal_binds": True})
        )
        return list(self.challenger_ids)


def test_forward_challenger_capacity_is_hard_capped_per_horizon() -> None:
    allowed = _CapacityConnection(["first"])
    evidence = require_horizon_challenger_capacity(
        allowed,
        horizon_profile="short_1_5d",
        version_id="second",
    )
    assert allowed.locked is True
    assert evidence["challenger_count"] == MAX_FORWARD_CHALLENGERS_PER_HORIZON
    assert "strategy_promotion_stages" in allowed.scalar_sql
    assert "status IN ('active', 'awaiting_simulation')" in allowed.scalar_sql
    assert "frozen" not in allowed.scalar_sql
    assert "strategy_versions.status = 'approved'" in allowed.scalar_sql
    assert "retired" not in allowed.scalar_sql

    full = _CapacityConnection(["first", "second"])
    with pytest.raises(ValueError, match="governed limit"):
        require_horizon_challenger_capacity(
            full,
            horizon_profile="short_1_5d",
            version_id="third",
        )

    retry = _CapacityConnection(["first", "second"])
    retried = require_horizon_challenger_capacity(
        retry,
        horizon_profile="short_1_5d",
        version_id="second",
    )
    assert retried["challenger_count"] == MAX_FORWARD_CHALLENGERS_PER_HORIZON


def test_explicit_horizon_shadow_accounts_do_not_supersede_each_other() -> None:
    assert not _autopilot_paper_accounts_compete("short_1_5d", "short_1_5d")
    assert not _autopilot_paper_accounts_compete("swing_1_6m", "swing_1_6m")
    assert not _autopilot_paper_accounts_compete("long_1_3y", "long_1_3y")
    assert _autopilot_paper_accounts_compete(
        "legacy_ambiguous", "short_1_5d"
    )
