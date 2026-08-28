from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import insert, select, text, update
from sqlalchemy.engine import Engine

from quant_data.database import (
    factor_candidates,
    factor_definition_similarity_edges,
    factor_definitions,
    factor_evaluations,
    factor_library_members,
    factor_library_versions,
    factor_similarity_edges,
    research_sota_members,
    research_sota_versions,
    row_dict,
)

from .factor_library import (
    ECONOMIC_FAMILIES,
    FACTOR_DEFINITIONS,
    FACTOR_LIBRARY_CONTRACT_VERSION,
    canonical_sha256,
    compile_qlib_expression,
    library_release_definition,
)
from .upstream_versions import QLIB_COMMIT

SOTA_POLICY_VERSION = "research-sota-policy-v1"
INCREMENTAL_EVIDENCE_VERSION = "factor-sota-increment-v2-frozen-model-paired"
_EXPECTED_PREDECESSOR_UNSET = object()


def _now() -> datetime:
    return datetime.now(UTC)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ResearchSotaPolicy:
    cluster_threshold: float = 0.75
    near_duplicate_threshold: float = 0.95
    max_factors_per_family: int = 3
    max_factor_weight: float = 0.25
    max_family_weight: float = 0.35

    def __post_init__(self) -> None:
        if not 0 < self.cluster_threshold < self.near_duplicate_threshold <= 1:
            raise ValueError("SOTA correlation thresholds are invalid")
        if self.max_factors_per_family < 1:
            raise ValueError("SOTA family count limit must be positive")
        if not 0 < self.max_factor_weight <= self.max_family_weight <= 1:
            raise ValueError("SOTA factor/family weight limits are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {"version": SOTA_POLICY_VERSION, **asdict(self)}


def relationship_for_correlation(
    value: float, policy: ResearchSotaPolicy | None = None
) -> str:
    policy = policy or ResearchSotaPolicy()
    correlation = abs(float(value))
    if not 0 <= correlation <= 1:
        raise ValueError("factor correlation must be in [0, 1]")
    if correlation >= policy.near_duplicate_threshold:
        return "near_duplicate"
    if correlation > policy.cluster_threshold:
        return "clustered"
    return "independent"


def governed_economic_family(
    proposed: str | None,
    required_fields: Iterable[str],
    expression: str = "",
) -> tuple[str, tuple[str, ...]]:
    family = str(proposed or "mixed").strip().lower()
    if family not in ECONOMIC_FAMILIES:
        family = "mixed"
    fields = set(required_fields)
    inferred: set[str] = set()
    if any(field.startswith("fund_") for field in fields):
        inferred.add(
            "growth"
            if any("yoy" in field or "growth" in field for field in fields)
            else "quality"
        )
    if fields & {"pe_ttm", "pb", "ps_ttm", "dv_ttm"}:
        inferred.add("value")
    if any(field.startswith("mf_") for field in fields):
        inferred.add("capital_flow")
    if fields & {"volume", "amount", "turnover_rate", "turnover_rate_f", "volume_ratio"}:
        inferred.add("liquidity")
    price_fields = fields & {"open", "close", "high", "low", "vwap"}
    if price_fields and re.search(r"\bStd\s*\(", expression):
        inferred.add("volatility_risk")
    if price_fields and any(
        re.search(rf"\b{function}\s*\(", expression)
        for function in ("Ref", "Mean", "EMA", "Slope", "Rsquare")
    ):
        inferred.add("trend")
    if price_fields and any(
        re.search(rf"\b{function}\s*\(", expression)
        for function in ("Min", "Max", "Rank")
    ):
        inferred.add("mean_reversion")
    if price_fields and not inferred:
        inferred.add("price_action")
    if not inferred:
        inferred.add(family)
    primary = next(iter(inferred)) if len(inferred) == 1 else "mixed"
    return primary, tuple(sorted(inferred))


def validate_incremental_evidence(value: dict[str, Any]) -> None:
    if value.get("version") != INCREMENTAL_EVIDENCE_VERSION:
        raise ValueError("SOTA incremental evidence version is invalid")
    model = value.get("frozen_model")
    if not isinstance(model, dict):
        raise ValueError("SOTA incremental evidence requires a frozen model")
    if str(model.get("kind") or "") in {
        "",
        "fixed_cross_sectional_zscore_linear",
        "point_estimate_linear_composite",
    }:
        raise ValueError("SOTA incremental evidence cannot use a fixed linear score proxy")
    for key in (
        "model_artifact_id",
        "model_artifact_sha256",
        "training_recipe_sha256",
        "feature_contract_sha256",
    ):
        text = str(model.get(key) or "")
        if not text or (key.endswith("sha256") and len(text) != 64):
            raise ValueError(f"SOTA incremental frozen model {key} is invalid")
    window_policy = value.get("window_role_policy")
    if not (
        isinstance(window_policy, dict)
        and window_policy.get("version") == "nested-profile-roles-v1"
        and window_policy.get("recent_role") == "ranking_and_significance"
        and window_policy.get("balanced_role") == "non_degradation"
        and window_policy.get("robust_role") == "direction_and_crash_stress"
        and window_policy.get("nested_windows_count_as_independent") is False
        and window_policy.get("combined_profile_p_value") is None
    ):
        raise ValueError("SOTA nested profiles must not be counted as independent votes")
    evaluation_ids = value.get("evaluation_ids")
    if not isinstance(evaluation_ids, dict) or set(evaluation_ids) != {
        "recent_3y",
        "balanced_5y",
        "robust_10y",
    }:
        raise ValueError("SOTA incremental evidence requires governed evaluation ids")
    if any(len(str(item or "")) != 32 for item in evaluation_ids.values()):
        raise ValueError("SOTA incremental evaluation identity is invalid")
    multiplicity = value.get("multiplicity")
    if not (
        isinstance(multiplicity, dict)
        and multiplicity.get("method") in {"benjamini_hochberg", "holm_bonferroni"}
        and int(multiplicity.get("hypothesis_count") or 0) >= 1
        and str(multiplicity.get("experiment_family_id") or "")
    ):
        raise ValueError("SOTA incremental evidence lacks multiplicity governance")
    for name in ("rank_ic_q_value", "cost_return_q_value"):
        q_value = multiplicity.get(name)
        if not isinstance(q_value, (int, float)) or not 0 <= float(q_value) <= 0.10:
            raise ValueError(f"SOTA incremental {name} must pass the adjusted significance gate")
    profiles = value.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != {
        "recent_3y",
        "balanced_5y",
        "robust_10y",
    }:
        raise ValueError("SOTA incremental evidence requires all three profiles")
    for profile_id, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"SOTA incremental profile {profile_id} is invalid")
        for name in (
            "delta_rank_ic",
            "delta_cost_adjusted_return",
            "baseline_rank_ic",
            "proposed_rank_ic",
            "baseline_cost_adjusted_return",
            "proposed_cost_adjusted_return",
        ):
            metric = profile.get(name)
            if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
                raise ValueError(f"SOTA incremental profile {profile_id} has invalid {name}")
        if len(str(profile.get("evaluation_evidence_sha256") or "")) != 64:
            raise ValueError(
                f"SOTA incremental profile {profile_id} evaluation evidence is unsealed"
            )
        prediction = profile.get("prediction_evidence")
        if not isinstance(prediction, dict) or any(
            len(str(prediction.get(name) or "")) != 64
            for name in (
                "baseline_prediction_sha256",
                "proposed_prediction_sha256",
                "paired_index_sha256",
            )
        ):
            raise ValueError(f"SOTA incremental profile {profile_id} predictions are unsealed")
        if prediction.get("final_oos_observations_exposed") is not False:
            raise ValueError("SOTA incremental predictions exposed the final OOS window")
        if profile.get("hard_gate_status") != "passed":
            raise ValueError(f"SOTA incremental profile {profile_id} failed a hard gate")
    recent = profiles["recent_3y"]
    balanced = profiles["balanced_5y"]
    robust = profiles["robust_10y"]
    if float(recent["delta_rank_ic"]) <= 0 or float(
        recent["delta_cost_adjusted_return"]
    ) <= 0:
        raise ValueError("recent SOTA RankIC and cost-adjusted return must both improve")
    rank_test = recent.get("paired_rank_ic_hac")
    return_test = recent.get("paired_cost_return_bootstrap")
    if not (
        isinstance(rank_test, dict)
        and rank_test.get("status") == "ok"
        and float(rank_test.get("mean") or 0.0) > 0
        and isinstance(rank_test.get("p_value"), (int, float))
        and float(rank_test["p_value"]) <= 0.05
    ):
        raise ValueError("recent RankIC increment is not significant under the paired HAC test")
    interval = return_test.get("confidence_interval_95") if isinstance(return_test, dict) else None
    if not (
        isinstance(return_test, dict)
        and return_test.get("status") == "ok"
        and isinstance(interval, list)
        and len(interval) == 2
        and float(interval[0]) > 0
        and isinstance(return_test.get("one_sided_p_value"), (int, float))
        and float(return_test["one_sided_p_value"]) <= 0.05
    ):
        raise ValueError(
            "recent cost-return increment is not significant under the paired block bootstrap"
        )
    if float(balanced["delta_rank_ic"]) < 0 or float(
        balanced["delta_cost_adjusted_return"]
    ) < 0:
        raise ValueError("balanced SOTA evidence must not deteriorate")
    if (
        robust.get("stability_gate_status") != "passed"
        or float(robust["proposed_rank_ic"]) <= 0
        or float(robust["proposed_cost_adjusted_return"]) < 0
    ):
        raise ValueError("robust SOTA profile must retain direction and avoid a cost crash")


def validate_sota_roll_forward(
    value: dict[str, Any] | None,
    *,
    predecessor_id: str | None,
    target_dataset: str,
    target_dataset_identity_sha256: str,
    target_lineage_id: str,
    target_end_date: str,
) -> None:
    """Validate the immutable exact/roll-forward predecessor contract."""

    if (
        not target_dataset
        or len(target_dataset_identity_sha256) != 64
        or not target_lineage_id
    ):
        raise ValueError("SOTA target dataset lineage binding is invalid")
    try:
        target_end = date.fromisoformat(target_end_date)
    except ValueError as exc:
        raise ValueError("SOTA target dataset end date is invalid") from exc
    if predecessor_id is None:
        if value is not None:
            raise ValueError("bootstrap SOTA cannot declare a predecessor roll-forward")
        return
    if not isinstance(value, dict):
        raise ValueError("SOTA predecessor is missing its roll-forward contract")
    if (
        value.get("contract_version") != "sota-roll-forward-v1"
        or value.get("mode") not in {"exact", "roll_forward"}
        or str(value.get("predecessor_id") or "") != predecessor_id
        or str(value.get("target_dataset") or "") != target_dataset
        or str(value.get("target_dataset_identity_sha256") or "")
        != target_dataset_identity_sha256
        or str(value.get("dataset_lineage_id") or "") != target_lineage_id
        or value.get("lineage_verified") is not True
        or str(value.get("target_end_date") or "") != target_end_date
    ):
        raise ValueError("SOTA predecessor roll-forward binding is invalid")
    for key in (
        "source_dataset_identity_sha256",
        "source_sota_evidence_sha256",
        "source_feature_set_definition_sha256",
        "source_member_set_sha256",
    ):
        if len(str(value.get(key) or "")) != 64:
            raise ValueError(f"SOTA predecessor {key} is invalid")
    try:
        source_end = date.fromisoformat(str(value.get("source_end_date") or ""))
    except ValueError as exc:
        raise ValueError("SOTA predecessor dates are invalid") from exc
    if value["mode"] == "exact":
        if (
            value.get("source_dataset_identity_sha256")
            != target_dataset_identity_sha256
            or source_end != target_end
        ):
            raise ValueError("exact SOTA predecessor changed dataset identity or end date")
    elif (
        value.get("source_dataset_identity_sha256")
        == target_dataset_identity_sha256
        or source_end >= target_end
    ):
        raise ValueError("SOTA roll-forward dataset is not a strict monotonic successor")


def validate_factor_definition_immutability(value: dict[str, Any]) -> str:
    """Rebuild one governed definition identity from its current DB fields."""

    compiled = compile_qlib_expression(str(value.get("expression") or ""))
    required_fields = list(value.get("required_fields") or [])
    identity = {
        "contract_version": FACTOR_LIBRARY_CONTRACT_VERSION,
        "expression_sha256": compiled.expression_sha256,
        "economic_family": str(value.get("economic_family") or ""),
        "required_fields": required_fields,
        "max_lookback_days": int(value.get("max_lookback_days") or 0),
        "qlib_commit": str(value.get("qlib_commit") or ""),
    }
    definition_sha256 = canonical_sha256(identity)
    if (
        compiled.expression_sha256 != value.get("expression_sha256")
        or list(compiled.required_fields) != required_fields
        or int(compiled.max_lookback_days) != int(value.get("max_lookback_days") or 0)
        or value.get("qlib_commit") != QLIB_COMMIT
        or value.get("status") != "registered"
        or definition_sha256 != value.get("definition_sha256")
    ):
        raise ValueError("SOTA factor definition changed in place")
    return definition_sha256


def _sota_member_set_sha256(connection: Any, version_id: str) -> str:
    rows = connection.execute(
        select(
            research_sota_members,
            factor_candidates.c.code_sha256,
            factor_candidates.c.values_sha256,
            factor_definitions.c.expression,
            factor_definitions.c.expression_sha256,
            factor_definitions.c.definition_sha256,
            factor_definitions.c.required_fields_json,
            factor_definitions.c.max_lookback_days,
            factor_definitions.c.economic_family.label("definition_economic_family"),
            factor_definitions.c.qlib_commit,
            factor_definitions.c.status,
        )
        .join(
            factor_candidates,
            factor_candidates.c.id == research_sota_members.c.factor_candidate_id,
        )
        .join(
            factor_definitions,
            factor_definitions.c.id == research_sota_members.c.factor_definition_id,
        )
        .where(research_sota_members.c.sota_version_id == version_id)
        .order_by(research_sota_members.c.member_rank)
    ).all()
    if not rows:
        raise ValueError("SOTA predecessor has no immutable members")
    manifest: list[dict[str, Any]] = []
    for row in rows:
        incremental = dict(row.incremental_evidence_json or {})
        definition_sha256 = validate_factor_definition_immutability(
            {
                "expression": row.expression,
                "expression_sha256": row.expression_sha256,
                "definition_sha256": row.definition_sha256,
                "required_fields": list(row.required_fields_json or []),
                "max_lookback_days": row.max_lookback_days,
                "economic_family": row.definition_economic_family,
                "qlib_commit": row.qlib_commit,
                "status": row.status,
            }
        )
        if canonical_sha256(incremental) != row.incremental_evidence_sha256:
            raise ValueError("SOTA predecessor member evidence changed in place")
        manifest.append(
            {
                "member_rank": int(row.member_rank),
                "factor_candidate_id": str(row.factor_candidate_id),
                "factor_definition_id": str(row.factor_definition_id),
                "factor_definition_sha256": definition_sha256,
                "candidate_code_sha256": str(row.code_sha256),
                "candidate_values_sha256": str(row.values_sha256),
                "incremental_evidence_sha256": str(row.incremental_evidence_sha256),
            }
        )
    return canonical_sha256(manifest)


def _sota_feature_set_sha256(connection: Any, version: Any) -> str:
    rows = connection.execute(
        select(
            factor_definitions.c.id,
            factor_definitions.c.expression,
            research_sota_members.c.member_rank,
        )
        .join(
            research_sota_members,
            research_sota_members.c.factor_definition_id == factor_definitions.c.id,
        )
        .where(research_sota_members.c.sota_version_id == version.id)
        .order_by(research_sota_members.c.member_rank)
    ).all()
    features = {
        f"SOTA_{int(row.member_rank):03d}_{str(row.id)[-8:]}": str(row.expression)
        for row in rows
    }
    definition = {
        "contract_version": "governed-feature-set-v2-research-sota",
        "id": f"sota:{version.id}",
        "name": f"Research SOTA {version.id}",
        "source": str(version.id),
        "features": features,
        "sota_evidence_sha256": str(version.evidence_sha256),
    }
    return canonical_sha256(definition)


def validate_sota_members(
    members: list[dict[str, Any]],
    policy: ResearchSotaPolicy,
) -> None:
    if not members:
        raise ValueError("research SOTA must contain at least one factor")
    ids = [str(item.get("factor_candidate_id") or "") for item in members]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("research SOTA factor candidate ids must be non-empty and unique")
    clusters = [str(item.get("similarity_cluster_id") or "") for item in members]
    if any(not item for item in clusters) or len(clusters) != len(set(clusters)):
        raise ValueError("research SOTA may retain only one factor per similarity cluster")
    member_families = [
        {
            str(value)
            for value in [
                item.get("economic_family"),
                *(item.get("family_tags") or []),
            ]
            if str(value) in ECONOMIC_FAMILIES and str(value) != "mixed"
        }
        or {"mixed"}
        for item in members
    ]
    family_counts = Counter(
        family for families in member_families for family in families
    )
    if "" in family_counts or any(
        count > policy.max_factors_per_family for count in family_counts.values()
    ):
        raise ValueError("research SOTA exceeds the economic-family factor limit")
    for item in members:
        validate_incremental_evidence(dict(item.get("incremental_evidence") or {}))

    weights = [item.get("weight") for item in members]
    if all(value is None for value in weights):
        return
    if any(not isinstance(value, (int, float)) for value in weights):
        raise ValueError("research SOTA weights must be either all present or all absent")
    numeric = [float(value) for value in weights]
    if abs(sum(numeric) - 1.0) > 1e-8:
        raise ValueError("research SOTA explicit weights must sum to one")
    if any(value < 0 or value > policy.max_factor_weight for value in numeric):
        raise ValueError("research SOTA exceeds the single-factor weight limit")
    family_weights: dict[str, float] = defaultdict(float)
    for families, weight in zip(member_families, numeric, strict=True):
        for family in families:
            family_weights[family] += weight
    if any(value > policy.max_family_weight + 1e-8 for value in family_weights.values()):
        raise ValueError("research SOTA exceeds the family weight limit")


class FactorLibraryStore:
    def __init__(
        self,
        engine: Engine,
        *,
        policy: ResearchSotaPolicy | None = None,
    ) -> None:
        self.engine = engine
        self.policy = policy or ResearchSotaPolicy()

    def sync_builtin_library(self) -> dict[str, Any]:
        release = library_release_definition()
        now = _now()
        with self.engine.begin() as connection:
            for definition in FACTOR_DEFINITIONS:
                existing = connection.execute(
                    select(factor_definitions).where(factor_definitions.c.id == definition.id)
                ).first()
                if existing is not None:
                    if row_dict(existing)["definition_sha256"] != definition.definition_sha256:
                        raise ValueError(
                            f"factor definition {definition.id} changed without a new identity"
                        )
                    continue
                connection.execute(
                    insert(factor_definitions).values(
                        id=definition.id,
                        name=definition.name,
                        expression=definition.expression,
                        expression_sha256=definition.expression_sha256,
                        definition_sha256=definition.definition_sha256,
                        required_fields_json=list(definition.required_fields),
                        max_lookback_days=definition.max_lookback_days,
                        economic_family=definition.economic_family,
                        family_tags_json=list(definition.family_tags),
                        aliases_json=list(definition.aliases),
                        source_refs_json=list(definition.source_refs),
                        availability_policy=definition.availability_policy,
                        qlib_commit=QLIB_COMMIT,
                        status="registered",
                        created_at=now,
                    )
                )
            existing_release = connection.execute(
                select(factor_library_versions).where(
                    factor_library_versions.c.id == release["id"]
                )
            ).first()
            if existing_release is None:
                connection.execute(
                    update(factor_library_versions)
                    .where(factor_library_versions.c.status == "active")
                    .values(status="retired", retired_at=now)
                )
                connection.execute(
                    insert(factor_library_versions).values(
                        id=release["id"],
                        contract_version=release["contract_version"],
                        definition_sha256=release["definition_sha256"],
                        member_count=release["member_count"],
                        source_alias_counts_json=release["source_alias_counts"],
                        qlib_commit=release["qlib_commit"],
                        status="active",
                        created_at=now,
                    )
                )
                connection.execute(
                    insert(factor_library_members),
                    [
                        {
                            "library_version_id": release["id"],
                            "factor_definition_id": definition.id,
                            "ordinal": ordinal,
                        }
                        for ordinal, definition in enumerate(
                            sorted(FACTOR_DEFINITIONS, key=lambda item: item.id)
                        )
                    ],
                )
            elif row_dict(existing_release)["definition_sha256"] != release[
                "definition_sha256"
            ]:
                raise ValueError("factor library release identity changed in place")
        return self.get_library_version(release["id"])

    def list_definitions(
        self,
        *,
        family: str | None = None,
        source: str | None = None,
        status: str | None = None,
        available_fields: Iterable[str] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        if limit < 1 or limit > 2000:
            raise ValueError("factor definition limit must be in [1, 2000]")
        statement = select(factor_definitions)
        if family:
            statement = statement.where(factor_definitions.c.economic_family == family)
        if status:
            statement = statement.where(factor_definitions.c.status == status)
        statement = statement.order_by(
            factor_definitions.c.economic_family, factor_definitions.c.name
        ).limit(limit)
        available = set(available_fields) if available_fields is not None else None
        with self.engine.connect() as connection:
            rows = [row_dict(row) for row in connection.execute(statement)]
        result: list[dict[str, Any]] = []
        for row in rows:
            aliases = list(row.pop("aliases_json") or [])
            refs = list(row.pop("source_refs_json") or [])
            if source and not any(item.startswith(f"{source}:") for item in aliases):
                continue
            required = list(row.pop("required_fields_json") or [])
            missing = sorted(set(required) - available) if available is not None else []
            calculability = (
                "not_assessed"
                if available is None
                else ("blocked_missing_fields" if missing else "available")
            )
            result.append(
                {
                    **row,
                    "required_fields": required,
                    "family_tags": list(row.pop("family_tags_json") or []),
                    "aliases": aliases,
                    "source_refs": refs,
                    "calculability": calculability,
                    "missing_fields": missing,
                }
            )
        return result

    def list_library_versions(self) -> list[dict[str, Any]]:
        statement = select(factor_library_versions).order_by(
            factor_library_versions.c.created_at.desc()
        )
        with self.engine.connect() as connection:
            rows = [row_dict(row) for row in connection.execute(statement)]
        for row in rows:
            row["source_alias_counts"] = row.pop("source_alias_counts_json")
        return rows

    def get_library_version(self, version_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(factor_library_versions).where(
                    factor_library_versions.c.id == version_id
                )
            ).first()
        if row is None:
            raise KeyError(version_id)
        result = row_dict(row)
        result["source_alias_counts"] = result.pop("source_alias_counts_json")
        return result

    def register_candidate_expression(
        self,
        candidate_id: str,
        *,
        name: str,
        expression: str,
        proposed_family: str | None,
        source_ref: str,
    ) -> dict[str, Any]:
        compiled = compile_qlib_expression(expression)
        family, tags = governed_economic_family(
            proposed_family, compiled.required_fields, compiled.expression
        )
        definition_id = f"factor-{compiled.expression_sha256[:24]}"
        identity = {
            "contract_version": FACTOR_LIBRARY_CONTRACT_VERSION,
            "expression_sha256": compiled.expression_sha256,
            "economic_family": family,
            "required_fields": compiled.required_fields,
            "max_lookback_days": compiled.max_lookback_days,
            "qlib_commit": QLIB_COMMIT,
        }
        now = _now()
        with self.engine.begin() as connection:
            candidate = connection.execute(
                select(factor_candidates).where(factor_candidates.c.id == candidate_id)
            ).first()
            if candidate is None:
                raise KeyError(candidate_id)
            existing = connection.execute(
                select(factor_definitions).where(
                    factor_definitions.c.expression_sha256
                    == compiled.expression_sha256
                )
            ).first()
            if existing is None:
                connection.execute(
                    insert(factor_definitions).values(
                        id=definition_id,
                        name=name,
                        expression=compiled.expression,
                        expression_sha256=compiled.expression_sha256,
                        definition_sha256=canonical_sha256(identity),
                        required_fields_json=list(compiled.required_fields),
                        max_lookback_days=compiled.max_lookback_days,
                        economic_family=family,
                        family_tags_json=list(tags),
                        aliases_json=[f"rdagent:{candidate_id}"],
                        source_refs_json=[source_ref],
                        availability_policy=(
                            "next_session_after_announcement"
                            if any(field.startswith("fund_") for field in compiled.required_fields)
                            else "after_same_session_close"
                        ),
                        qlib_commit=QLIB_COMMIT,
                        status="registered",
                        created_at=now,
                    )
                )
            else:
                definition_id = str(existing.id)
                family = str(existing.economic_family)
                tags = tuple(existing.family_tags_json or ())
            connection.execute(
                update(factor_candidates)
                .where(factor_candidates.c.id == candidate_id)
                .values(
                    factor_definition_id=definition_id,
                    economic_family=family,
                    family_tags_json=list(tags),
                    formulation=compiled.expression,
                    updated_at=now,
                )
            )
        return self.get_definition(definition_id)

    def register_expression_definition(
        self,
        *,
        name: str,
        expression: str,
        proposed_family: str | None,
        alias: str,
        source_ref: str,
    ) -> dict[str, Any]:
        compiled = compile_qlib_expression(expression)
        family, tags = governed_economic_family(
            proposed_family, compiled.required_fields, compiled.expression
        )
        definition_id = f"factor-{compiled.expression_sha256[:24]}"
        identity = {
            "contract_version": FACTOR_LIBRARY_CONTRACT_VERSION,
            "expression_sha256": compiled.expression_sha256,
            "economic_family": family,
            "required_fields": compiled.required_fields,
            "max_lookback_days": compiled.max_lookback_days,
            "qlib_commit": QLIB_COMMIT,
        }
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(factor_definitions).where(
                    factor_definitions.c.expression_sha256
                    == compiled.expression_sha256
                )
            ).first()
            if existing is None:
                connection.execute(
                    insert(factor_definitions).values(
                        id=definition_id,
                        name=name,
                        expression=compiled.expression,
                        expression_sha256=compiled.expression_sha256,
                        definition_sha256=canonical_sha256(identity),
                        required_fields_json=list(compiled.required_fields),
                        max_lookback_days=compiled.max_lookback_days,
                        economic_family=family,
                        family_tags_json=list(tags),
                        aliases_json=[alias],
                        source_refs_json=[source_ref],
                        availability_policy=(
                            "next_session_after_announcement"
                            if any(
                                field.startswith("fund_")
                                for field in compiled.required_fields
                            )
                            else "after_same_session_close"
                        ),
                        qlib_commit=QLIB_COMMIT,
                        status="registered",
                        created_at=_now(),
                    )
                )
            else:
                definition_id = str(existing.id)
        return self.get_definition(definition_id)

    def get_definition(self, definition_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(factor_definitions).where(factor_definitions.c.id == definition_id)
            ).first()
        if row is None:
            raise KeyError(definition_id)
        result = row_dict(row)
        result["required_fields"] = result.pop("required_fields_json")
        result["family_tags"] = result.pop("family_tags_json")
        result["aliases"] = result.pop("aliases_json")
        result["source_refs"] = result.pop("source_refs_json")
        return result

    def record_similarity(
        self,
        *,
        left_candidate_id: str,
        right_candidate_id: str,
        dataset_identity_sha256: str,
        profile_id: str,
        mean_abs_spearman: float,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        left, right = sorted((left_candidate_id, right_candidate_id))
        if left == right:
            raise ValueError("factor similarity requires two distinct candidates")
        relationship = relationship_for_correlation(mean_abs_spearman, self.policy)
        cluster_id = (
            f"cluster-{_sha256_text(f'{dataset_identity_sha256}:{profile_id}:{left}:{right}')[:20]}"
            if relationship != "independent"
            else None
        )
        identity = {
            "dataset_identity_sha256": dataset_identity_sha256,
            "profile_id": profile_id,
            "left_factor_candidate_id": left,
            "right_factor_candidate_id": right,
            "mean_abs_spearman": float(mean_abs_spearman),
            "relationship": relationship,
            "cluster_id": cluster_id,
            "evidence": evidence,
        }
        edge_id = uuid.uuid5(uuid.NAMESPACE_URL, canonical_sha256(identity)).hex
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(factor_similarity_edges).where(
                    factor_similarity_edges.c.id == edge_id
                )
            ).first()
            if existing is None:
                connection.execute(
                    insert(factor_similarity_edges).values(
                        id=edge_id,
                        evidence_json=evidence,
                        evidence_sha256=canonical_sha256(evidence),
                        created_at=_now(),
                        **{key: value for key, value in identity.items() if key != "evidence"},
                    )
                )
            row = existing or connection.execute(
                select(factor_similarity_edges).where(
                    factor_similarity_edges.c.id == edge_id
                )
            ).first()
        return row_dict(row)

    def assign_similarity_clusters(
        self,
        *,
        candidate_ids: Iterable[str],
        correlations: Iterable[dict[str, Any]],
        dataset_identity_sha256: str,
        profile_id: str,
    ) -> dict[str, str]:
        ids = sorted({str(value) for value in candidate_ids if str(value)})
        correlation_items = [dict(item) for item in correlations]
        parents = {value: value for value in ids}

        def find(value: str) -> str:
            while parents[value] != value:
                parents[value] = parents[parents[value]]
                value = parents[value]
            return value

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[max(left_root, right_root)] = min(left_root, right_root)

        for item in correlation_items:
            left = str(item["left_candidate_id"])
            right = str(item["right_candidate_id"])
            if left not in parents or right not in parents or left == right:
                raise ValueError("similarity cluster input contains an unknown candidate")
            value = abs(float(item["mean_abs_spearman"]))
            self.record_similarity(
                left_candidate_id=left,
                right_candidate_id=right,
                dataset_identity_sha256=dataset_identity_sha256,
                profile_id=profile_id,
                mean_abs_spearman=value,
                evidence=dict(item.get("evidence") or {}),
            )
            if relationship_for_correlation(value, self.policy) != "independent":
                union(left, right)
        components: dict[str, list[str]] = defaultdict(list)
        for candidate_id in ids:
            components[find(candidate_id)].append(candidate_id)
        assignments: dict[str, str] = {}
        for members in components.values():
            identity = f"{dataset_identity_sha256}:{profile_id}:{':'.join(sorted(members))}"
            cluster_id = f"cluster-{_sha256_text(identity)[:20]}"
            assignments.update({candidate_id: cluster_id for candidate_id in members})
        with self.engine.begin() as connection:
            for candidate_id, cluster_id in assignments.items():
                connection.execute(
                    update(factor_candidates)
                    .where(factor_candidates.c.id == candidate_id)
                    .values(similarity_cluster_id=cluster_id, updated_at=_now())
                )
            for item in correlation_items:
                left, right = sorted(
                    (
                        str(item["left_candidate_id"]),
                        str(item["right_candidate_id"]),
                    )
                )
                relationship = relationship_for_correlation(
                    abs(float(item["mean_abs_spearman"])), self.policy
                )
                connection.execute(
                    update(factor_similarity_edges)
                    .where(
                        factor_similarity_edges.c.dataset_identity_sha256
                        == dataset_identity_sha256,
                        factor_similarity_edges.c.profile_id == profile_id,
                        factor_similarity_edges.c.left_factor_candidate_id == left,
                        factor_similarity_edges.c.right_factor_candidate_id == right,
                    )
                    .values(
                        cluster_id=(
                            assignments[left]
                            if relationship != "independent"
                            else None
                        )
                    )
                )
        return assignments

    def import_definition_similarity_clusters(
        self,
        *,
        library_version_id: str,
        dataset_identity_sha256: str,
        edges: Iterable[dict[str, Any]],
    ) -> int:
        imported = 0
        with self.engine.begin() as connection:
            member_ids = {
                str(value)
                for value in connection.scalars(
                    select(factor_library_members.c.factor_definition_id).where(
                        factor_library_members.c.library_version_id
                        == library_version_id
                    )
                )
            }
            if not member_ids:
                raise ValueError("factor similarity library version is unavailable")
            for item in edges:
                left, right = sorted(
                    (
                        str(item["left_factor_definition_id"]),
                        str(item["right_factor_definition_id"]),
                    )
                )
                if left not in member_ids or right not in member_ids or left == right:
                    raise ValueError("definition similarity edge is outside the frozen library")
                correlation = abs(float(item["mean_abs_spearman"]))
                relationship = relationship_for_correlation(correlation, self.policy)
                if relationship == "independent":
                    raise ValueError("only governed correlated definition edges are persisted")
                evidence = dict(item.get("evidence") or {})
                identity = {
                    "library_version_id": library_version_id,
                    "dataset_identity_sha256": dataset_identity_sha256,
                    "left_factor_definition_id": left,
                    "right_factor_definition_id": right,
                    "mean_abs_spearman": correlation,
                    "relationship": relationship,
                    "cluster_id": str(item["cluster_id"]),
                }
                edge_id = uuid.uuid5(
                    uuid.NAMESPACE_URL, canonical_sha256(identity)
                ).hex
                existing = connection.execute(
                    select(factor_definition_similarity_edges.c.id).where(
                        factor_definition_similarity_edges.c.id == edge_id
                    )
                ).first()
                if existing is not None:
                    continue
                connection.execute(
                    insert(factor_definition_similarity_edges).values(
                        id=edge_id,
                        evidence_json=evidence,
                        evidence_sha256=canonical_sha256(evidence),
                        created_at=_now(),
                        **identity,
                    )
                )
                imported += 1
        return imported

    def activate_sota(
        self,
        *,
        dataset: str,
        dataset_identity_sha256: str,
        universe: str,
        label_horizon_days: int,
        periods: dict[str, Any],
        members: list[dict[str, Any]],
        evidence: dict[str, Any],
        actor: str,
        expected_predecessor_id: str | None | object = _EXPECTED_PREDECESSOR_UNSET,
    ) -> dict[str, Any]:
        if not actor.strip():
            raise ValueError("research SOTA actor is required")
        explicit_predecessor = expected_predecessor_id is not _EXPECTED_PREDECESSOR_UNSET
        expected_id = (
            str(expected_predecessor_id)
            if explicit_predecessor and expected_predecessor_id is not None
            else None
        )
        roll_forward = evidence.get("predecessor_roll_forward")
        if explicit_predecessor:
            validate_sota_roll_forward(
                roll_forward,
                predecessor_id=expected_id,
                target_dataset=dataset,
                target_dataset_identity_sha256=dataset_identity_sha256,
                target_lineage_id=str(evidence.get("dataset_lineage_id") or ""),
                target_end_date=str(evidence.get("dataset_end_date") or ""),
            )
        validate_sota_members(members, self.policy)
        policy_json = self.policy.to_dict()
        now = _now()
        with self.engine.begin() as connection:
            # Serialise one atomic SOTA increment per dataset/universe/horizon.
            # The explicit predecessor check prevents two completed jobs from
            # both consuming the same frozen baseline during concurrent import.
            connection.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:scope))"),
                {
                    "scope": (
                        f"research-sota:{dataset}:{universe}:"
                        f"{label_horizon_days}"
                    )
                },
            )
            if expected_id is not None:
                connection.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext(:scope))"),
                    {"scope": f"research-sota-predecessor:{expected_id}"},
                )
            base_library = connection.execute(
                select(factor_library_versions).where(
                    factor_library_versions.c.status == "active"
                )
            ).first()
            if base_library is None:
                raise ValueError("active factor library is unavailable")
            predecessor = connection.execute(
                select(research_sota_versions).where(
                    research_sota_versions.c.dataset == dataset,
                    research_sota_versions.c.universe == universe,
                    research_sota_versions.c.label_horizon_days == label_horizon_days,
                    research_sota_versions.c.status == "active",
                )
            ).first()
            actual_predecessor_id = str(predecessor.id) if predecessor else None
            if explicit_predecessor and actual_predecessor_id != expected_id:
                if actual_predecessor_id is None and expected_id is not None:
                    predecessor = connection.execute(
                        select(research_sota_versions)
                        .where(research_sota_versions.c.id == expected_id)
                        .with_for_update()
                    ).first()
                    if (
                        predecessor is None
                        or predecessor.status != "active"
                        or predecessor.universe != universe
                        or int(predecessor.label_horizon_days) != label_horizon_days
                    ):
                        raise ValueError(
                            "research SOTA frozen predecessor was already consumed or changed"
                        )
                else:
                    raise ValueError(
                        "research SOTA frozen predecessor was already consumed or changed"
                    )
            if predecessor is not None and explicit_predecessor:
                source_is_target = (
                    str(predecessor.dataset_identity_sha256)
                    == dataset_identity_sha256
                )
                if (
                    not isinstance(roll_forward, dict)
                    or str(roll_forward.get("source_dataset") or "")
                    != str(predecessor.dataset)
                    or str(roll_forward.get("source_dataset_identity_sha256") or "")
                    != str(predecessor.dataset_identity_sha256)
                    or roll_forward.get("mode")
                    != ("exact" if source_is_target else "roll_forward")
                    or canonical_sha256(predecessor.evidence_json or {})
                    != predecessor.evidence_sha256
                    or canonical_sha256(predecessor.policy_json or {})
                    != predecessor.policy_sha256
                    or str(roll_forward.get("source_sota_evidence_sha256") or "")
                    != str(predecessor.evidence_sha256)
                    or str(
                        roll_forward.get("source_feature_set_definition_sha256") or ""
                    )
                    != _sota_feature_set_sha256(connection, predecessor)
                    or str(roll_forward.get("source_member_set_sha256") or "")
                    != _sota_member_set_sha256(connection, str(predecessor.id))
                ):
                    raise ValueError(
                        "research SOTA predecessor definition or hash changed in place"
                    )
            for member in members:
                candidate = connection.execute(
                    select(factor_candidates).where(
                        factor_candidates.c.id == member["factor_candidate_id"]
                    )
                ).first()
                if candidate is None or candidate.status != "promoted":
                    raise ValueError("research SOTA members must be promoted factors")
                if not candidate.factor_definition_id:
                    raise ValueError("research SOTA factor has no immutable definition")
                if int(candidate.label_horizon_days or 0) != label_horizon_days:
                    raise ValueError("research SOTA label horizon does not match candidate")
                retained_source = None
                if (
                    predecessor is not None
                    and member.get("action") == "retained"
                ):
                    retained_source = connection.execute(
                        select(research_sota_members).where(
                            research_sota_members.c.sota_version_id == predecessor.id,
                            research_sota_members.c.factor_candidate_id == candidate.id,
                        )
                    ).first()
                    retained_incremental = dict(
                        retained_source.incremental_evidence_json or {}
                    ) if retained_source is not None else {}
                    if (
                        retained_source is None
                        or str(retained_source.factor_definition_id)
                        != str(member.get("factor_definition_id") or "")
                        or str(retained_source.factor_evaluation_id)
                        != str(member.get("factor_evaluation_id") or "")
                        or str(retained_source.economic_family)
                        != str(member.get("economic_family") or "")
                        or set(retained_source.family_tags_json or [])
                        != set(member.get("family_tags") or [])
                        or str(retained_source.similarity_cluster_id)
                        != str(member.get("similarity_cluster_id") or "")
                        or retained_source.weight != member.get("weight")
                        or retained_incremental
                        != dict(member.get("incremental_evidence") or {})
                        or canonical_sha256(retained_incremental)
                        != retained_source.incremental_evidence_sha256
                    ):
                        raise ValueError(
                            "roll-forward retained SOTA member changed after freezing"
                        )
                expected_member_identity = (
                    str(predecessor.dataset_identity_sha256)
                    if retained_source is not None
                    else dataset_identity_sha256
                )
                expected_member_periods = (
                    dict(predecessor.periods_json or {})
                    if retained_source is not None
                    else periods
                )
                consensus = candidate.profile_consensus_json
                admission_path = str(candidate.admission_path or "standalone")
                if admission_path == "incremental":
                    incremental = dict(candidate.incremental_evidence_json or {})
                    validate_incremental_evidence(incremental)
                    if (
                        canonical_sha256(incremental)
                        != candidate.incremental_evidence_sha256
                        or incremental != dict(member.get("incremental_evidence") or {})
                        or incremental.get("factor_candidate_id") != candidate.id
                        or incremental.get("dataset_identity_sha256")
                        != expected_member_identity
                        or incremental.get("profile_periods") != expected_member_periods
                        or incremental.get("evaluation_ids", {}).get("recent_3y")
                        != member["factor_evaluation_id"]
                    ):
                        raise ValueError(
                            "research SOTA incremental admission evidence changed"
                        )
                elif (
                    not isinstance(consensus, dict)
                    or consensus.get("status") != "passed"
                    or consensus.get("dataset_identity_sha256")
                    != expected_member_identity
                    or set(consensus.get("evaluation_ids") or {})
                    != {"recent_3y", "balanced_5y", "robust_10y"}
                    or consensus.get("profile_periods") != expected_member_periods
                ):
                    raise ValueError(
                        "research SOTA factor lacks matching three-window governance"
                    )
                predicates = [
                    factor_evaluations.c.id == member["factor_evaluation_id"],
                    factor_evaluations.c.factor_candidate_id == candidate.id,
                ]
                if admission_path != "incremental":
                    predicates.append(factor_evaluations.c.gate_status == "passed")
                evaluation = connection.execute(
                    select(factor_evaluations).where(*predicates)
                ).first()
                if evaluation is None:
                    raise ValueError("research SOTA factor evaluation is not governed")
                if admission_path == "incremental":
                    from quant_platform.research_store import FactorGatePolicy

                    layers = FactorGatePolicy().evaluate_layers(evaluation.metrics_json or {})
                    if layers["hard_status"] != "passed":
                        raise ValueError("hard-gate failure cannot enter research SOTA")
                if str(candidate.economic_family) != str(member["economic_family"]):
                    raise ValueError("research SOTA economic family does not match candidate")
                if set(candidate.family_tags_json or []) != set(
                    member.get("family_tags") or []
                ):
                    raise ValueError("research SOTA family tags do not match candidate")
                if str(candidate.similarity_cluster_id or "") != str(
                    member["similarity_cluster_id"]
                ):
                    raise ValueError("research SOTA similarity cluster changed")
            library_id = self._ensure_library_for_members(
                connection,
                base_library_id=str(base_library.id),
                definition_ids={
                    str(
                        connection.execute(
                            select(factor_candidates.c.factor_definition_id).where(
                                factor_candidates.c.id == item["factor_candidate_id"]
                            )
                        ).scalar_one()
                    )
                    for item in members
                },
                now=now,
            )
            if predecessor is not None:
                connection.execute(
                    update(research_sota_versions)
                    .where(research_sota_versions.c.id == predecessor.id)
                    .values(status="superseded", superseded_at=now)
                )
            version_identity = {
                "predecessor_id": str(predecessor.id) if predecessor else None,
                "library_version_id": library_id,
                "dataset": dataset,
                "dataset_identity_sha256": dataset_identity_sha256,
                "universe": universe,
                "label_horizon_days": label_horizon_days,
                "periods": periods,
                "policy_sha256": canonical_sha256(policy_json),
                "members": [
                    {
                        key: member.get(key)
                        for key in (
                            "factor_candidate_id",
                            "factor_evaluation_id",
                            "economic_family",
                            "similarity_cluster_id",
                            "weight",
                            "action",
                            "replaced_factor_candidate_id",
                        )
                    }
                    for member in members
                ],
                "evidence_sha256": canonical_sha256(evidence),
            }
            version_id = f"sota-{canonical_sha256(version_identity)[:24]}"
            connection.execute(
                insert(research_sota_versions).values(
                    id=version_id,
                    library_version_id=library_id,
                    predecessor_id=predecessor.id if predecessor else None,
                    dataset=dataset,
                    dataset_identity_sha256=dataset_identity_sha256,
                    universe=universe,
                    label_horizon_days=label_horizon_days,
                    periods_json=periods,
                    policy_json=policy_json,
                    policy_sha256=canonical_sha256(policy_json),
                    evidence_json=evidence,
                    evidence_sha256=canonical_sha256(evidence),
                    status="active",
                    created_by=actor,
                    created_at=now,
                    activated_at=now,
                )
            )
            for rank, member in enumerate(members):
                incremental = dict(member["incremental_evidence"])
                candidate = connection.execute(
                    select(factor_candidates).where(
                        factor_candidates.c.id == member["factor_candidate_id"]
                    )
                ).one()
                connection.execute(
                    insert(research_sota_members).values(
                        sota_version_id=version_id,
                        factor_candidate_id=candidate.id,
                        factor_definition_id=candidate.factor_definition_id,
                        factor_evaluation_id=member["factor_evaluation_id"],
                        economic_family=member["economic_family"],
                        family_tags_json=list(member.get("family_tags") or []),
                        similarity_cluster_id=member["similarity_cluster_id"],
                        member_rank=rank,
                        weight=member.get("weight"),
                        action=member.get("action") or "added",
                        replaced_factor_candidate_id=member.get(
                            "replaced_factor_candidate_id"
                        ),
                        incremental_evidence_json=incremental,
                        incremental_evidence_sha256=canonical_sha256(incremental),
                    )
                )
        return self.get_sota(version_id)

    def _ensure_library_for_members(
        self,
        connection: Any,
        *,
        base_library_id: str,
        definition_ids: set[str],
        now: datetime,
    ) -> str:
        current_ids = {
            str(value)
            for value in connection.scalars(
                select(factor_library_members.c.factor_definition_id).where(
                    factor_library_members.c.library_version_id == base_library_id
                )
            )
        }
        combined = sorted(current_ids | definition_ids)
        if current_ids == set(combined):
            return base_library_id
        definitions = {
            str(row.id): str(row.definition_sha256)
            for row in connection.execute(
                select(
                    factor_definitions.c.id,
                    factor_definitions.c.definition_sha256,
                ).where(factor_definitions.c.id.in_(combined))
            )
        }
        if set(definitions) != set(combined):
            raise ValueError("SOTA library contains an unknown factor definition")
        identity = {
            "contract_version": FACTOR_LIBRARY_CONTRACT_VERSION,
            "qlib_commit": QLIB_COMMIT,
            "member_definition_sha256": sorted(definitions.values()),
            "member_count": len(combined),
        }
        digest = canonical_sha256(identity)
        existing = connection.execute(
            select(factor_library_versions).where(
                factor_library_versions.c.definition_sha256 == digest
            )
        ).first()
        if existing is not None:
            return str(existing.id)
        version_id = f"factor-library-{digest[:24]}"
        connection.execute(
            update(factor_library_versions)
            .where(factor_library_versions.c.status == "active")
            .values(status="retired", retired_at=now)
        )
        connection.execute(
            insert(factor_library_versions).values(
                id=version_id,
                contract_version=FACTOR_LIBRARY_CONTRACT_VERSION,
                definition_sha256=digest,
                member_count=len(combined),
                source_alias_counts_json={
                    "inherited": len(current_ids),
                    "research_sota": len(set(combined) - current_ids),
                },
                qlib_commit=QLIB_COMMIT,
                status="active",
                created_at=now,
            )
        )
        connection.execute(
            insert(factor_library_members),
            [
                {
                    "library_version_id": version_id,
                    "factor_definition_id": definition_id,
                    "ordinal": ordinal,
                }
                for ordinal, definition_id in enumerate(combined)
            ],
        )
        return version_id

    def list_sota(self, *, limit: int = 50) -> list[dict[str, Any]]:
        statement = (
            select(research_sota_versions)
            .order_by(research_sota_versions.c.created_at.desc())
            .limit(limit)
        )
        with self.engine.connect() as connection:
            rows = [row_dict(row) for row in connection.execute(statement)]
        for row in rows:
            row["periods"] = row.pop("periods_json")
            row["policy"] = row.pop("policy_json")
            row["evidence"] = row.pop("evidence_json")
        return rows

    def get_sota(self, version_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(research_sota_versions).where(
                    research_sota_versions.c.id == version_id
                )
            ).first()
            if row is None:
                raise KeyError(version_id)
            members = connection.execute(
                select(research_sota_members)
                .where(research_sota_members.c.sota_version_id == version_id)
                .order_by(research_sota_members.c.member_rank)
            ).all()
        result = row_dict(row)
        result["periods"] = result.pop("periods_json")
        result["policy"] = result.pop("policy_json")
        result["evidence"] = result.pop("evidence_json")
        decoded_members: list[dict[str, Any]] = []
        for item in members:
            decoded = row_dict(item)
            decoded["family_tags"] = decoded.pop("family_tags_json")
            decoded["incremental_evidence"] = decoded.pop(
                "incremental_evidence_json"
            )
            decoded_members.append(decoded)
        return {**result, "members": decoded_members}

    def sota_feature_set(self, version_id: str) -> dict[str, Any]:
        sota = self.get_sota(version_id)
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    factor_definitions.c.id,
                    factor_definitions.c.name,
                    factor_definitions.c.expression,
                    research_sota_members.c.member_rank,
                )
                .join(
                    research_sota_members,
                    research_sota_members.c.factor_definition_id
                    == factor_definitions.c.id,
                )
                .where(research_sota_members.c.sota_version_id == version_id)
                .order_by(research_sota_members.c.member_rank)
            ).all()
        features = {
            f"SOTA_{int(row.member_rank):03d}_{str(row.id)[-8:]}": str(row.expression)
            for row in rows
        }
        definition = {
            "contract_version": "governed-feature-set-v2-research-sota",
            "id": f"sota:{version_id}",
            "name": f"Research SOTA {version_id}",
            "source": version_id,
            "features": features,
            "sota_evidence_sha256": sota["evidence_sha256"],
        }
        return {**definition, "definition_sha256": canonical_sha256(definition)}
