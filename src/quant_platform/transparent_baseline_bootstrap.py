"""Idempotent one-step bootstrap for the three transparent public baselines."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

from quant_data.config import Settings
from quant_data.execution_contract import (
    DAILY_QLIB_FIELD_CONTRACT_VERSION,
    require_daily_qlib_contract,
    require_native_daily_execution_controls,
)
from quant_platform.forward_only_rehabilitation import (
    EVIDENCE_MODE_REPLAY,
    REPLAY_MARKERS,
    SOURCE_BACKTEST_ID,
    SOURCE_DATASET,
    SOURCE_DATASET_IDENTITY_SHA256,
    SOURCE_DATASET_LINEAGE_ID,
    SOURCE_EXECUTION_CONTRACT_HASH,
    SOURCE_INTERRUPTION_RECEIPT_AUTHORITY,
    SOURCE_INTERRUPTION_RECOVERY_RECEIPT_SHA256,
    SOURCE_JOB_ID,
    SOURCE_PERIODS,
    SOURCE_RULES_SHA256,
    SOURCE_VERSION_ID,
    require_consumed_vintage,
    require_source_cancellation,
    require_source_cash_only_lockbox,
)
from quant_platform.job_store import JobStore
from quant_platform.promotion import PromotionStore
from quant_platform.research_automation import (
    ResearchWindowUnavailableError,
    resolve_research_window_contract,
)
from quant_platform.research_horizon import research_horizon_contract
from quant_platform.research_window import build_research_window_contract
from quant_platform.services import list_qlib_datasets
from quant_platform.strategy_recipes import (
    TRANSPARENT_RESEARCH_BASELINE_IDS,
    get_strategy_recipe,
)
from quant_platform.strategy_store import StrategyStore, _normalize_multifactor_contract
from quant_platform.transparent_baseline_lockbox import (
    BOOTSTRAP_CONFIG_KEY,
    LOCKBOX_CONFIG_KEY,
    TransparentBaselineLockboxStore,
    build_joint_lockbox,
    build_lockbox_member,
    canonical_sha256,
    lockbox_member_link,
    validate_unopened_history_selection,
)
from quant_platform.transparent_baseline_runner import (
    FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION,
    TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION,
    TRANSPARENT_BASELINE_JOB_RUNNER_FIELD,
    TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
    position_risk_bundle_sha256,
    target_runner_for_recipe,
    target_runtime_bundle_for_recipe,
    target_worker_runtime_image_for_recipe,
)

BOOTSTRAP_CONTRACT_VERSION = "transparent-baseline-bootstrap-v2"
RECONCILE_RESULT_VERSION = "transparent-baseline-reconcile-v2"
FORWARD_ONLY_RECONCILE_RESULT_VERSION = (
    "transparent-baseline-forward-only-rehabilitation-reconcile-v1"
)
FORWARD_ONLY_BOOTSTRAP_CONTRACT_VERSION = (
    "transparent-baseline-forward-only-replay-bootstrap-v1"
)
FORWARD_ONLY_WINDOW_CONTRACT_VERSION = "consumed-historical-replay-window-v1"
DEFAULT_ACTOR = "system:transparent-baseline-bootstrap"
_RECONCILABLE_FAMILY_STATUSES = frozenset({"draft", "approved"})
_RECONCILABLE_VERSION_LIFECYCLES = frozenset(
    {
        ("draft", None),
        ("approved", "paper"),
    }
)
_GOVERNED_NOOP_STATUSES = frozenset(
    {"watch", "paused", "restricted", "suspended", "rejected", "retired"}
)
_GOVERNED_NOOP_PROMOTION_STAGES = frozenset(
    {"recommendation_enabled", "watch", "paused", "restricted", "suspended", "retired"}
)
FAMILY_NAMES = {
    "short_relative_strength": "QuantLab透明基线：1至5日短线相对强弱",
    "swing_trend": "QuantLab透明基线：1至6个月波段趋势",
    "long_quality_value": "QuantLab透明基线：1至3年质量价值",
}


def _unavailable_member_results(
    unavailable_horizons: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project unavailable research lanes without inventing a lockbox member."""

    return [
        {
            "recipe_id": item["recipe_id"],
            "horizon_profile": item["horizon_profile"],
            "family_action": None,
            "strategy_version_id": None,
            "research_window_contract_sha256": None,
            "formal_periods": None,
            "state": "unavailable",
            "backtest": None,
            "job": None,
            "paper_stage": None,
            "lifecycle": None,
            "errors": [],
            "unavailable_reason": item["reason"],
            "unavailable_evidence": item["evidence"],
            "unavailable_evidence_sha256": item["evidence_sha256"],
            "sleeve_action": "remain_in_cash",
        }
        for item in unavailable_horizons
    ]


def _require_preregistered_single_member_repair_oos(
    *,
    repair_selection: Mapping[str, Any],
    plans: Sequence[Mapping[str, Any]],
    unavailable_horizons: Sequence[Mapping[str, Any]],
) -> None:
    """Keep the exact preregistered short repair on its source final OOS.

    Receipt eligibility, source-result absence and the frozen history-selection
    v2 contract belong to ``TransparentBaselineLockboxStore``.  Bootstrap only
    enforces the final planning invariant it owns: the one repaired short lane
    must use the same OOS dates, while the same two source lanes remain cash.
    """

    source_lockbox = repair_selection.get("source_lockbox")
    if not isinstance(source_lockbox, Mapping):
        raise ValueError("single-member repair has no frozen source lockbox")
    source_members = source_lockbox.get("members")
    if (
        not isinstance(source_members, Sequence)
        or isinstance(source_members, (str, bytes))
        or len(source_members) != 1
        or not isinstance(source_members[0], Mapping)
    ):
        raise ValueError("single-member repair source must contain exactly one member")
    source_member = source_members[0]
    if (
        str(source_member.get("recipe_id") or "") != "short_relative_strength"
        or str(source_member.get("horizon_profile") or "") != "short_1_5d"
    ):
        raise ValueError("single-member repair source is not the short baseline")
    if len(plans) != 1:
        raise ValueError("single-member repair must plan exactly one target member")
    target_member = plans[0].get("lockbox_member")
    if not isinstance(target_member, Mapping):
        raise ValueError("single-member repair target has no lockbox member")
    if (
        str(target_member.get("recipe_id") or "") != "short_relative_strength"
        or str(target_member.get("horizon_profile") or "") != "short_1_5d"
        or str(target_member.get("test_start") or "")
        != str(source_member.get("test_start") or "")
        or str(target_member.get("test_end") or "")
        != str(source_member.get("test_end") or "")
    ):
        raise ValueError("single-member repair changed the frozen short final OOS")
    if {str(item.get("recipe_id") or "") for item in unavailable_horizons} != {
        "swing_trend",
        "long_quality_value",
    }:
        raise ValueError("single-member repair changed the unavailable source lanes")


def _sha256(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def _calendar(dataset: Mapping[str, Any]) -> list[str]:
    path = Path(str(dataset.get("path") or "")) / "calendars" / "day.txt"
    try:
        raw = [
            date.fromisoformat(line.strip()).isoformat()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError) as exc:
        raise ValueError("latest ready daily Qlib calendar is missing or invalid") from exc
    if not raw or raw != sorted(set(raw)):
        raise ValueError("latest ready daily Qlib calendar must be ordered and unique")
    if (
        str(dataset.get("start_date") or "") != raw[0]
        or str(dataset.get("end_date") or "") != raw[-1]
        or int(dataset.get("trading_days") or 0) != len(raw)
    ):
        raise ValueError("Qlib catalog dates differ from the real daily calendar")
    return raw


def _validate_dataset(dataset: Mapping[str, Any]) -> dict[str, Any]:
    if not dataset.get("ready") or str(dataset.get("frequency") or "") != "day":
        raise ValueError("selected Qlib dataset is not a ready daily publication")
    if not dataset.get("reproducible") or not dataset.get("output_files_verified"):
        raise ValueError("latest ready daily Qlib dataset is not reproducibly sealed")
    provenance = dataset.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("latest ready daily Qlib dataset has no provenance")
    require_daily_qlib_contract(dict(provenance))
    if provenance.get("field_contract_version") != DAILY_QLIB_FIELD_CONTRACT_VERSION:
        raise ValueError(
            "current transparent baselines require the fail-closed daily Qlib "
            "field contract; rebuild the dataset"
        )
    identity = _sha256(
        provenance.get("dataset_identity_sha256"),
        field="dataset_identity_sha256",
    )
    lineage = _sha256(
        dataset.get("lineage_id") or provenance.get("dataset_lineage_id"),
        field="dataset_lineage_id",
    )
    if provenance.get("dataset_lineage_id") != lineage:
        raise ValueError("Qlib catalog and provenance lineage identities differ")
    result = dict(dataset)
    result["provenance"] = dict(provenance)
    result["dataset_identity_sha256"] = identity
    result["dataset_lineage_id"] = lineage
    result["calendar"] = _calendar(result)
    return result


def _select_dataset(
    datasets: Sequence[Mapping[str, Any]], *, anchored_name: str | None
) -> dict[str, Any]:
    daily_ready = [
        dict(item)
        for item in datasets
        if item.get("ready") and str(item.get("frequency") or "") == "day"
    ]
    if not daily_ready:
        raise ValueError("no ready daily Qlib dataset is available")
    if anchored_name is not None:
        matches = [item for item in daily_ready if item.get("name") == anchored_name]
        if len(matches) != 1:
            raise ValueError("the partially frozen baseline dataset is no longer ready")
        return _validate_dataset(matches[0])
    newest = max(
        daily_ready,
        key=lambda item: (str(item.get("end_date") or ""), str(item.get("name") or "")),
    )
    # Do not silently fall back to an older reproducible snapshot when the
    # latest ready publication itself has broken provenance.
    return _validate_dataset(newest)


def _feature_set(recipe: Mapping[str, Any]) -> dict[str, Any]:
    definition = {
        "contract_version": "transparent-baseline-feature-set-v1",
        "recipe_id": recipe["id"],
        "recipe_version": recipe["version"],
        "features": {
            str(item["id"]): str(item["qlib_expression"])
            for item in recipe.get("factor_baseline") or []
        },
    }
    if not definition["features"]:
        raise ValueError("transparent baseline recipe has no Qlib factors")
    return {
        **definition,
        "id": f"transparent-baseline:{recipe['id']}",
        "definition_sha256": canonical_sha256(definition),
    }


def _validated_recipe_config(values: Mapping[str, Any]) -> dict[str, Any]:
    # StrategyConfigRequest currently owns the complete governed defaults.  It
    # is imported lazily so SchedulerEngine can import this service without a
    # module-initialization cycle through quant_platform.api.
    from quant_platform.api import StrategyConfigRequest

    return StrategyConfigRequest.model_validate(dict(values)).model_dump()


def _require_native_formal_oos(
    *,
    dataset: Mapping[str, Any],
    periods: Mapping[str, str],
    evidence: Mapping[str, Any],
    horizon_profile: str,
) -> dict[str, str]:
    """Reject a frozen OOS that predates native A-share execution controls.

    The daily runner already fails closed on this boundary.  Transparent
    baselines must apply the same check *before* preregistration, otherwise a
    one-shot OOS can be consumed by a job that was impossible to execute.

    A rolling horizon already ends at the latest session that leaves the full
    label-maturity tail.  Therefore moving its start forward also moves its
    fixed-length end forward.  If the immutable unopened-history prefix cannot
    hold that complete shifted window, the honest result is ``unavailable``;
    shortening the OOS or reading into a prior opened batch is forbidden.
    """

    provenance = dataset.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("transparent baseline dataset has no provenance")
    try:
        require_native_daily_execution_controls(
            dict(provenance),
            start=str(periods["test_start"]),
        )
        return dict(periods)
    except ValueError as exc:
        if "formal execution starts before native price-limit controls" not in str(exc):
            raise
        execution_boundary_error = exc
        controls = provenance.get("execution_controls")
        if not isinstance(controls, Mapping):
            raise
        raw_boundary = str(controls.get("native_complete_from") or "")[:10]
        try:
            boundary = date.fromisoformat(raw_boundary)
        except ValueError:
            raise

    calendar = [str(day) for day in dataset.get("calendar") or []]
    first_native_session = next(
        (day for day in calendar if date.fromisoformat(day) >= boundary),
        None,
    )
    horizon = research_horizon_contract(horizon_profile)
    required_sessions = int(horizon.sealed_oos_sessions or 0)
    maturity = evidence.get("latest_mature_label_sessions")
    if not isinstance(maturity, Mapping):
        raise ValueError("research window has no label-maturity evidence")
    maturity_cutoff = str(maturity.get(str(max(horizon.label_horizons_sessions))) or "")
    available_sessions = (
        sum(
            first_native_session <= day <= maturity_cutoff
            for day in calendar
        )
        if first_native_session is not None and maturity_cutoff
        else 0
    )
    if (
        first_native_session is not None
        and maturity_cutoff
        and available_sessions >= required_sessions
    ):
        # ``test_end`` is already the latest label-mature session allowed by
        # the immutable research-window contract.  Keep that sealed cutoff and
        # move the start to the first natively controlled session.  The result
        # therefore remains a complete (possibly conservative, longer) OOS,
        # never consumes the label-maturity tail, and is deterministic for the
        # same calendar and provenance.
        return {
            **dict(periods),
            "test_start": first_native_session,
            "test_end": maturity_cutoff,
        }
    unavailable_evidence = {
        "contract_version": "native-execution-oos-resolution-v1",
        "horizon_profile": horizon_profile,
        "native_complete_from": boundary.isoformat(),
        "first_native_controlled_trading_session": first_native_session,
        "rejected_test_start": str(periods["test_start"]),
        "proposed_test_start": first_native_session,
        "label_maturity_cutoff_session": maturity_cutoff or None,
        "required_sealed_oos_sessions": required_sessions,
        "available_native_controlled_oos_sessions": available_sessions,
        "capital_evaluation_eligible": False,
        "capital_evaluation_unavailable_reason": (
            "insufficient_native_execution_controlled_sessions_before_immutable_cutoff"
        ),
    }
    raise ResearchWindowUnavailableError(
        "formal OOS would start before native price-limit controls; moving it to "
        f"{first_native_session or boundary.isoformat()} leaves {available_sessions} "
        f"sessions before the immutable label-maturity cutoff, but {required_sessions} "
        "are required",
        evidence=unavailable_evidence,
    ) from execution_boundary_error


def _plan_member(
    *, recipe_id: str, dataset: Mapping[str, Any]
) -> dict[str, Any]:
    recipe = get_strategy_recipe(recipe_id)
    recipe_sha256 = canonical_sha256(recipe)
    feature_set = _feature_set(recipe)
    raw_selection = dataset.get("unopened_history_selection")
    selection = validate_unopened_history_selection(
        raw_selection,
        calendar_days=dataset.get("source_calendar") or dataset["calendar"],
    )
    if selection["current_recipe_version"] != recipe["version"]:
        raise ValueError(
            "transparent baseline history selection belongs to another recipe version"
        )
    periods, evidence = resolve_research_window_contract(
        dict(dataset),
        list(dataset["calendar"]),
        horizon_profile=str(recipe["horizon"]),
        feature_set=feature_set,
        universe=str(recipe["universe"]),
    )
    original_research_window = evidence.get("research_window_contract")
    if not isinstance(original_research_window, Mapping):
        raise ValueError("research window resolver returned no immutable contract")
    adjusted_periods = _require_native_formal_oos(
        dataset=dataset,
        periods=periods,
        evidence=evidence,
        horizon_profile=str(recipe["horizon"]),
    )
    if adjusted_periods != periods:
        calendar_start = str(original_research_window.get("calendar_start") or "")
        calendar_end = str(original_research_window.get("calendar_end") or "")
        effective_calendar = [
            str(day)
            for day in dataset.get("calendar") or []
            if calendar_start <= str(day) <= calendar_end
        ]
        rebound = build_research_window_contract(
            dataset=dataset,
            calendar_days=effective_calendar,
            periods=adjusted_periods,
            period_resolution=evidence,
            horizon_profile=str(recipe["horizon"]),
            feature_set=feature_set,
            universe=str(recipe["universe"]),
        )
        periods = adjusted_periods
        evidence = {
            **evidence,
            "final_test_trading_days": int(rebound.sealed_oos_sessions),
            "research_window_contract": rebound.to_dict(),
            "research_window_contract_sha256": rebound.sha256,
        }
    research_window = evidence.get("research_window_contract")
    research_window_sha256 = _sha256(
        evidence.get("research_window_contract_sha256"),
        field="research_window_contract_sha256",
    )
    if not isinstance(research_window, Mapping):
        raise ValueError("research window resolver returned no immutable contract")
    formal_periods = {
        "historical_start": periods["train_start"],
        "historical_end": periods["valid_end"],
        "start": periods["test_start"],
        "end": periods["test_end"],
    }
    horizon = research_horizon_contract(str(recipe["horizon"]))
    raw_config = {
        **deepcopy(dict(recipe["config_overrides"])),
        "recipe_id": recipe["id"],
        "recipe_version": recipe["version"],
        "outer_purge_days": int(evidence["purge_trading_days"]),
        "outer_embargo_days": int(evidence["embargo_trading_days"]),
        "min_backtest_days": int(horizon.sealed_oos_sessions or 0),
    }
    config = _validated_recipe_config(raw_config)
    bootstrap = {
        "contract_version": BOOTSTRAP_CONTRACT_VERSION,
        "recipe_id": recipe["id"],
        "recipe_version": recipe["version"],
        "recipe_sha256": recipe_sha256,
        "dataset": dataset["name"],
        "dataset_identity_sha256": dataset["dataset_identity_sha256"],
        "dataset_lineage_id": dataset["dataset_lineage_id"],
        "feature_set": feature_set,
        "research_periods": dict(periods),
        "formal_periods": formal_periods,
        "research_window_contract": dict(research_window),
        "research_window_contract_sha256": research_window_sha256,
        "unopened_history_selection": selection,
    }
    target_runner_sha256 = target_runner_for_recipe(recipe["id"], recipe["version"])
    if target_runner_sha256 is not None:
        bootstrap[TRANSPARENT_BASELINE_RUNNER_FIELD] = target_runner_sha256
    target_runtime_bundle_sha256 = target_runtime_bundle_for_recipe(
        recipe["id"], recipe["version"]
    )
    if target_runtime_bundle_sha256 is not None:
        bootstrap[TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD] = (
            target_runtime_bundle_sha256
        )
    target_worker_runtime_image_digest = target_worker_runtime_image_for_recipe(
        recipe["id"], recipe["version"]
    )
    if target_worker_runtime_image_digest is not None:
        bootstrap[TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD] = (
            target_worker_runtime_image_digest
        )
    config[BOOTSTRAP_CONFIG_KEY] = bootstrap
    normalized = _normalize_multifactor_contract(
        config,
        factor_count=0,
        creating_family=True,
    )
    return {
        "recipe": recipe,
        "recipe_sha256": recipe_sha256,
        "feature_set": feature_set,
        "periods": periods,
        "formal_periods": formal_periods,
        "research_window_contract_sha256": research_window_sha256,
        "base_config": normalized,
        "lockbox_member": build_lockbox_member(
            config=normalized,
            formal_periods=formal_periods,
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_forward_only_dataset(datasets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matches = [dict(item) for item in datasets if item.get("name") == SOURCE_DATASET]
    if len(matches) != 1:
        raise ValueError("forward-only rehabilitation source dataset is missing or duplicated")
    dataset = matches[0]
    if (
        str(dataset.get("dataset_identity_sha256") or "")
        != SOURCE_DATASET_IDENTITY_SHA256
        or str(dataset.get("dataset_lineage_id") or "")
        != SOURCE_DATASET_LINEAGE_ID
        or not isinstance(dataset.get("calendar"), Sequence)
        or isinstance(dataset.get("calendar"), (str, bytes))
        or not dataset.get("calendar")
        or not str(dataset.get("path") or "").strip()
    ):
        raise ValueError("forward-only rehabilitation dataset identity or calendar changed")
    return dataset


def _build_forward_only_rehabilitation_plan(
    *,
    source_version: Mapping[str, Any],
    dataset: Mapping[str, Any],
    consumed_oos_vintage_id: str | None,
) -> dict[str, Any]:
    """Build v18 descriptive replay config without creating a new lockbox."""

    recipe = deepcopy(get_strategy_recipe("short_relative_strength"))
    if (
        str(recipe.get("version") or "")
        != TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION
    ):
        raise ValueError("forward-only rehabilitation source recipe generation changed")
    # v18 changes evidence authority and runtime identity, not economics.  It
    # must never become the ordinary three-horizon recipe generation because
    # only this exact short-horizon one-shot is eligible for rehabilitation.
    recipe["version"] = FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
    recipe_version = FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
    source_config = source_version.get("config")
    if not isinstance(source_config, Mapping):
        raise ValueError("source v17 StrategyVersion config is missing")
    if (
        str(source_version.get("id") or "") != SOURCE_VERSION_ID
        or str(source_version.get("strategy_rules_sha256") or "")
        != SOURCE_RULES_SHA256
        or str(source_version.get("execution_contract_hash") or "")
        != SOURCE_EXECUTION_CONTRACT_HASH
        or str(source_version.get("horizon_profile") or "") != "short_1_5d"
        or source_config.get("recipe_id") != "short_relative_strength"
        or source_config.get("recipe_version")
        != TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION
    ):
        raise ValueError("source v17 economic strategy changed")
    consumed_vintage_id = str(consumed_oos_vintage_id or "").strip()
    feature_set = _feature_set(recipe)
    raw_config = {
        **deepcopy(dict(recipe["config_overrides"])),
        "recipe_id": recipe["id"],
        # Validate the unchanged economic recipe against its ordinary v17
        # release contract first.  The dedicated builder then changes only the
        # evidence/runtime generation to v18 before the stricter replay
        # normalizer verifies every frozen binding below.
        "recipe_version": TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION,
    }
    config = _validated_recipe_config(raw_config)
    config["recipe_version"] = recipe_version
    binding = {
        "contract_version": "forward-only-rehabilitation-v1",
        "source_strategy_version_id": SOURCE_VERSION_ID,
        "source_backtest_id": SOURCE_BACKTEST_ID,
        "source_job_id": SOURCE_JOB_ID,
        "source_interruption_recovery_receipt_sha256": (
            SOURCE_INTERRUPTION_RECOVERY_RECEIPT_SHA256
        ),
        "source_interruption_receipt_authority": (
            SOURCE_INTERRUPTION_RECEIPT_AUTHORITY
        ),
        "dataset": SOURCE_DATASET,
        "dataset_identity_sha256": SOURCE_DATASET_IDENTITY_SHA256,
        "dataset_lineage_id": SOURCE_DATASET_LINEAGE_ID,
        "strategy_rules_sha256": SOURCE_RULES_SHA256,
        "execution_contract_hash": SOURCE_EXECUTION_CONTRACT_HASH,
        "replay_periods": dict(SOURCE_PERIODS),
        "source_attempt_artifact_relative_path": (
            "artifacts/formal-backtest-recoveries/"
            f"{SOURCE_BACKTEST_ID}/attempt-2"
        ),
        **(
            {"consumed_oos_vintage_id": consumed_vintage_id}
            if consumed_vintage_id
            else {}
        ),
    }
    window_core = {
        "contract_version": FORWARD_ONLY_WINDOW_CONTRACT_VERSION,
        **REPLAY_MARKERS,
        "periods": dict(SOURCE_PERIODS),
        **(
            {"consumed_oos_vintage_id": consumed_vintage_id}
            if consumed_vintage_id
            else {}
        ),
    }
    window = {**window_core, "contract_sha256": canonical_sha256(window_core)}
    target_runner = target_runner_for_recipe(recipe["id"], recipe_version)
    target_bundle = target_runtime_bundle_for_recipe(recipe["id"], recipe_version)
    target_image = target_worker_runtime_image_for_recipe(recipe["id"], recipe_version)
    project_root = Path(__file__).resolve().parents[2]
    observed_runner = _sha256_file(project_root / "scripts" / "run_multifactor_backtest.py")
    observed_bundle = position_risk_bundle_sha256(project_root)
    if (
        target_runner != observed_runner
        or target_bundle != observed_bundle
        or target_image is None
    ):
        raise ValueError(
            "v18 runner, runtime source closure, or worker image is not the sealed release"
        )
    bootstrap = {
        "contract_version": FORWARD_ONLY_BOOTSTRAP_CONTRACT_VERSION,
        "recipe_id": recipe["id"],
        "recipe_version": recipe_version,
        "recipe_sha256": canonical_sha256(recipe),
        "dataset": SOURCE_DATASET,
        "dataset_identity_sha256": SOURCE_DATASET_IDENTITY_SHA256,
        "dataset_lineage_id": SOURCE_DATASET_LINEAGE_ID,
        "feature_set": feature_set,
        "formal_periods": dict(SOURCE_PERIODS),
        "research_window_contract": window,
        "research_window_contract_sha256": window["contract_sha256"],
        TRANSPARENT_BASELINE_RUNNER_FIELD: target_runner,
        TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: target_bundle,
        TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: target_image,
    }
    final_config = _normalize_multifactor_contract(
        {
            **config,
            "evidence_mode": EVIDENCE_MODE_REPLAY,
            "forward_only_rehabilitation": binding,
            BOOTSTRAP_CONFIG_KEY: bootstrap,
        },
        factor_count=0,
        creating_family=False,
    )
    if LOCKBOX_CONFIG_KEY in final_config:
        raise ValueError("forward-only replay must not create a new OOS lockbox")
    if (
        final_config.get("strategy_rules_sha256") != SOURCE_RULES_SHA256
        or final_config.get("execution_contract_hash")
        != SOURCE_EXECUTION_CONTRACT_HASH
    ):
        raise ValueError("v18 replay changed the v17 economic or execution strategy")
    return {
        "recipe": recipe,
        "config": final_config,
        "formal_periods": dict(SOURCE_PERIODS),
        "consumed_oos_vintage_id": consumed_vintage_id or None,
    }


class TransparentBaselineBootstrapService:
    """Reconcile the public controls without granting recommendation authority."""

    def __init__(
        self,
        *,
        database_url: str,
        data_root: Path,
        dataset_loader: Callable[[Path], list[dict[str, Any]]] = list_qlib_datasets,
        strategies: StrategyStore | None = None,
        jobs: JobStore | None = None,
        promotions: PromotionStore | None = None,
        lockboxes: TransparentBaselineLockboxStore | None = None,
    ) -> None:
        self.database_url = database_url
        self.data_root = data_root.resolve()
        self.dataset_loader = dataset_loader
        self.strategies = strategies or StrategyStore(database_url)
        self.jobs = jobs or JobStore(database_url)
        self.promotions = promotions or PromotionStore(database_url)
        self.lockboxes = lockboxes or TransparentBaselineLockboxStore(database_url)
        self.artifact_root = self.data_root / "artifacts" / "backtests"
        self.log_root = self.data_root / "platform" / "logs"

    def _existing_families(self) -> dict[str, dict[str, Any] | None]:
        return {
            recipe_id: self.strategies.get_by_name(FAMILY_NAMES[recipe_id])
            for recipe_id in TRANSPARENT_RESEARCH_BASELINE_IDS
        }

    @staticmethod
    def _lifecycle_action(
        value: Mapping[str, Any],
        *,
        entity: str,
    ) -> str:
        """Classify bootstrap-owned work without reviving a governed lifecycle.

        Managed reconciliation owns only the initial ``draft`` research path
        and the idempotent ``approved``/``paper`` continuation it created.
        Explicit recommendation, operator and risk states are safe terminal
        no-ops for this bootstrap. Unknown or inconsistent states still fail
        closed instead of being guessed into either path.
        """

        status = str(value.get("status") or "").strip()
        if entity == "family":
            if status in _RECONCILABLE_FAMILY_STATUSES:
                return "reconcile"
            raise ValueError(
                "transparent baseline family lifecycle is unknown or inconsistent: "
                f"{status or 'missing'}"
            )
        if entity != "version":
            raise ValueError(f"unsupported transparent baseline lifecycle entity: {entity}")
        stage = value.get("promotion_stage")
        lifecycle = (status, str(stage).strip() if stage is not None else None)
        if lifecycle in _RECONCILABLE_VERSION_LIFECYCLES:
            return "reconcile"
        if (
            status in _GOVERNED_NOOP_STATUSES
            or lifecycle[1] in _GOVERNED_NOOP_PROMOTION_STAGES
        ):
            return "governed_no_op"
        rendered_stage = lifecycle[1] or "none"
        raise ValueError(
            "transparent baseline version lifecycle is unknown or inconsistent: "
            f"status={status or 'missing'}, promotion_stage={rendered_stage}"
        )

    @classmethod
    def _governed_noop_result(
        cls,
        version: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if cls._lifecycle_action(version, entity="version") != "governed_no_op":
            return None
        return {
            "state": "governed_no_op",
            "paper_stage": None,
            "lifecycle": {
                "status": str(version.get("status") or ""),
                "promotion_stage": version.get("promotion_stage"),
            },
        }

    @staticmethod
    def _anchored_dataset(
        families: Mapping[str, Mapping[str, Any] | None]
    ) -> str | None:
        """Anchor only a partially-created *current* recipe batch.

        Older recipe/dataset versions remain immutable history.  They must not
        pin a material vNext recipe to the original bootstrap dataset forever.
        """

        anchors: set[str] = set()
        for recipe_id, family in families.items():
            if family is None:
                continue
            versions = list(family.get("versions") or [])
            recipe = get_strategy_recipe(recipe_id)
            recipe_sha256 = canonical_sha256(recipe)
            current = []
            for version in versions:
                config = version.get("config") or {}
                bootstrap = config.get(BOOTSTRAP_CONFIG_KEY)
                if (
                    config.get("recipe_id") == recipe_id
                    and config.get("recipe_version") == recipe["version"]
                    and isinstance(bootstrap, Mapping)
                    and bootstrap.get("recipe_sha256") == recipe_sha256
                ):
                    current.append(version)
            if len(current) > 1:
                raise ValueError(
                    f"{recipe_id} transparent baseline has duplicate current-recipe versions"
                )
            if not current:
                continue
            bootstrap = current[0]["config"][BOOTSTRAP_CONFIG_KEY]
            anchors.add(str(bootstrap.get("dataset") or ""))
        if not anchors:
            return None
        if len(anchors) != 1 or "" in anchors:
            raise ValueError("partial transparent baselines do not share one frozen dataset")
        return next(iter(anchors))

    @staticmethod
    def _anchored_unopened_history_selection(
        families: Mapping[str, Mapping[str, Any] | None]
    ) -> dict[str, Any] | None:
        """Recover the immutable cutoff from a partially created current batch."""

        selections: list[dict[str, Any]] = []
        for recipe_id, family in families.items():
            if family is None:
                continue
            recipe = get_strategy_recipe(recipe_id)
            recipe_sha256 = canonical_sha256(recipe)
            current = []
            for version in family.get("versions") or []:
                config = version.get("config") or {}
                bootstrap = config.get(BOOTSTRAP_CONFIG_KEY)
                if (
                    config.get("recipe_id") == recipe_id
                    and config.get("recipe_version") == recipe["version"]
                    and isinstance(bootstrap, Mapping)
                    and bootstrap.get("recipe_sha256") == recipe_sha256
                ):
                    current.append(bootstrap)
            if len(current) > 1:
                raise ValueError(
                    f"{recipe_id} transparent baseline has duplicate current-recipe versions"
                )
            if not current:
                continue
            raw_selection = current[0].get("unopened_history_selection")
            if not isinstance(raw_selection, Mapping):
                raise ValueError(
                    "current transparent baseline has no frozen history selection"
                )
            selections.append(validate_unopened_history_selection(raw_selection))
        if not selections:
            return None
        selection_hashes = {
            str(item["selection_sha256"]) for item in selections
        }
        if len(selection_hashes) != 1:
            raise ValueError(
                "partial transparent baselines do not share one history cutoff"
            )
        return selections[0]

    def _ensure_versions(
        self,
        *,
        plans: list[dict[str, Any]],
        families: Mapping[str, Mapping[str, Any] | None],
        actor: str,
    ) -> list[dict[str, Any]]:
        versions: list[dict[str, Any]] = []
        for plan in plans:
            recipe = plan["recipe"]
            recipe_id = str(recipe["id"])
            expected_config = plan["config"]
            family = families[recipe_id]
            if family is None:
                try:
                    family = self.strategies.create(
                        name=FAMILY_NAMES[recipe_id],
                        description=(
                            f"{recipe['description']} 由系统冻结公开配方、真实交易日历和"
                            "一次性正式样本外窗口；仅通过严格审批后进入隔离模拟。"
                        ),
                        benchmark=str(recipe["benchmark"]),
                        universe=str(recipe["universe"]),
                        factors=[],
                        config=expected_config,
                        actor=actor,
                        economic_hypothesis_group=(
                            f"transparent-public-control:{recipe_id}"
                        ),
                    )
                except ValueError:
                    # A concurrent reconcile may have won the unique family
                    # name. Re-read and validate the exact immutable version.
                    family = self.strategies.get_by_name(FAMILY_NAMES[recipe_id])
                    if family is None:
                        raise
                plan["family_action"] = "created"
            else:
                self._lifecycle_action(family, entity="family")
                exact = [
                    version
                    for version in family.get("versions") or []
                    if version.get("benchmark") == recipe["benchmark"]
                    and version.get("universe") == recipe["universe"]
                    and version.get("factors") == []
                    and version.get("config") == expected_config
                    and version.get("horizon_profile") == recipe["horizon"]
                ]
                if len(exact) > 1:
                    raise ValueError(
                        f"{recipe_id} transparent baseline current version is duplicated"
                    )
                if not exact:
                    try:
                        self.strategies.create_version_if_absent(
                            str(family["id"]),
                            benchmark=str(recipe["benchmark"]),
                            universe=str(recipe["universe"]),
                            factors=[],
                            config=expected_config,
                            actor=actor,
                        )
                    except ValueError:
                        # Recover only if a concurrent reconcile materialized
                        # the exact immutable vNext version.
                        refreshed = self.strategies.get_by_name(
                            FAMILY_NAMES[recipe_id]
                        )
                        if refreshed is None:
                            raise
                        family = refreshed
                    else:
                        family = self.strategies.get_by_name(
                            FAMILY_NAMES[recipe_id]
                        )
                        if family is None:
                            raise ValueError(
                                "transparent baseline family disappeared after version creation"
                            )
                    plan["family_action"] = "version_created"
                else:
                    plan["family_action"] = "reused"
            exact = [
                version
                for version in family.get("versions") or []
                if version.get("benchmark") == recipe["benchmark"]
                and version.get("universe") == recipe["universe"]
                and version.get("factors") == []
                and version.get("config") == expected_config
                and version.get("horizon_profile") == recipe["horizon"]
            ]
            if len(exact) != 1:
                raise ValueError(
                    f"{recipe_id} current frozen StrategyVersion is missing or duplicated"
                )
            version = exact[0]
            lifecycle_action = self._lifecycle_action(version, entity="version")
            if lifecycle_action == "governed_no_op":
                plan["lifecycle_action"] = lifecycle_action
                plan["lifecycle"] = {
                    "status": str(version.get("status") or ""),
                    "promotion_stage": version.get("promotion_stage"),
                }
            plan["version_id"] = str(version["id"])
            versions.append(version)
        return versions

    def _ensure_backtest_job(
        self,
        *,
        plan: dict[str, Any],
        version: Mapping[str, Any],
        dataset: Mapping[str, Any],
        allow_create: bool = True,
    ) -> dict[str, Any]:
        backtests = self.strategies.list_backtests(
            version_id=str(version["id"]),
            limit=10,
        )
        if len(backtests) > 1:
            raise ValueError("transparent baseline has more than one formal backtest")
        if backtests:
            backtest = backtests[0]
            if (
                backtest.get("dataset") != dataset["name"]
                or backtest.get("execution_dataset") is not None
                or backtest.get("periods") != plan["formal_periods"]
                or backtest.get("evidence_mode")
                != version.get("evidence_mode", "sealed_final_oos")
            ):
                raise ValueError("existing formal backtest differs from the frozen lockbox")
            backtest_action = "reused"
        else:
            if not allow_create:
                raise ValueError(
                    "existing forward-only v18 replay is partial: formal backtest is missing"
                )
            try:
                backtest = self.strategies.create_backtest(
                    version_id=str(version["id"]),
                    dataset=str(dataset["name"]),
                    execution_dataset=None,
                    periods=dict(plan["formal_periods"]),
                    artifact_path=self.artifact_root,
                    trading_dates=[
                        date.fromisoformat(day) for day in dataset["calendar"]
                    ],
                    dataset_lineage_id=str(dataset["dataset_lineage_id"]),
                    dataset_identity_sha256=str(dataset["dataset_identity_sha256"]),
                )
                backtest_action = "created"
            except ValueError:
                # Another scheduler/process may have consumed the exact
                # preregistered one-shot vintage after our initial read.  Only
                # recover when that race produced the one immutable backtest
                # this member is allowed to own; otherwise preserve the
                # original fail-closed error.
                raced = self.strategies.list_backtests(
                    version_id=str(version["id"]),
                    limit=10,
                )
                if len(raced) != 1:
                    raise
                backtest = raced[0]
                if (
                    backtest.get("dataset") != dataset["name"]
                    or backtest.get("execution_dataset") is not None
                    or backtest.get("periods") != plan["formal_periods"]
                ):
                    raise
                backtest_action = "reused"
        status = str(backtest.get("status") or "")
        job: dict[str, Any] | None = None
        job_action = "not_required"
        payload = {
            "backtest_id": str(backtest["id"]),
            "strategy_version_id": str(version["id"]),
            "dataset": str(dataset["name"]),
            "dataset_path": str(dataset["path"]),
            "execution_dataset": None,
            "periods": dict(backtest["periods"]),
        }
        bootstrap = dict(
            dict(version.get("config") or {}).get(BOOTSTRAP_CONFIG_KEY) or {}
        )
        target_runner_sha256 = bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
        if target_runner_sha256 is not None:
            payload[TRANSPARENT_BASELINE_JOB_RUNNER_FIELD] = target_runner_sha256
        target_runtime_bundle_sha256 = bootstrap.get(
            TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD
        )
        if target_runtime_bundle_sha256 is not None:
            payload[TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD] = (
                target_runtime_bundle_sha256
            )
        target_worker_runtime_image_digest = bootstrap.get(
            TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD
        )
        if target_worker_runtime_image_digest is not None:
            payload[TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD] = (
                target_worker_runtime_image_digest
            )
        idempotency_key = f"transparent-baseline:{version['id']}:{backtest['id']}"
        if not allow_create:
            attached_job_id = str(backtest.get("job_id") or "")
            if not attached_job_id:
                raise ValueError(
                    "existing forward-only v18 replay is partial: attached job is missing"
                )
            job = self.jobs.get(attached_job_id)
            if (
                attached_job_id == SOURCE_JOB_ID
                or job.get("kind") != "strategy_backtest"
                or job.get("payload") != payload
                or job.get("idempotency_key") != idempotency_key
                or int(job.get("max_attempts") or 0) != 1
            ):
                raise ValueError(
                    "existing forward-only v18 replay job differs from the frozen plan"
                )
            allowed_job_statuses = {
                "queued": {"queued"},
                "running": {"queued", "running"},
                "succeeded": {"succeeded"},
                "failed": {"failed", "cancelled"},
                "cancelled": {"failed", "cancelled"},
            }
            if str(job.get("status") or "") not in allowed_job_statuses.get(status, set()):
                raise ValueError(
                    "existing forward-only v18 replay job/backtest lifecycle is partial"
                )
            job_action = "reused"
        if status in {"queued", "running"}:
            if allow_create:
                job = self.jobs.create(
                    "strategy_backtest",
                    payload,
                    self.log_root / f"strategy-backtest-{backtest['id']}.log",
                    dedupe_active_kind=False,
                    idempotency_key=idempotency_key,
                    max_attempts=1,
                )
                job_action = "reused" if backtest.get("job_id") else "created"
            assert job is not None
            if backtest.get("job_id") not in {None, str(job["id"])}:
                raise ValueError("formal backtest is attached to a different job")
            if str(job.get("status") or "") in {"failed", "cancelled"}:
                raise ValueError(
                    "formal backtest job is terminal and cannot reopen its one-shot OOS"
                )
            self.strategies.attach_job(str(backtest["id"]), str(job["id"]))
            backtest = self.strategies.get_backtest(str(backtest["id"]))
        return {
            "backtest": backtest,
            "backtest_action": backtest_action,
            "job": job,
            "job_action": job_action,
        }

    def _advance_paper(
        self,
        *,
        version_id: str,
        backtest: Mapping[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        status = str(backtest.get("status") or "")
        if status != "succeeded":
            return {
                "state": "formal_backtest_failed"
                if status in {"failed", "cancelled"}
                else "formal_backtest_pending",
                "paper_stage": None,
            }
        version = self.strategies.get_version(version_id)
        governed_noop = self._governed_noop_result(version)
        if governed_noop is not None:
            return governed_noop
        if version.get("status") != "approved":
            try:
                if version.get("evidence_mode") == "consumed_historical_replay":
                    self.strategies.admit_forward_only_rehabilitation(
                        version_id,
                        actor=actor,
                        reason=(
                            "Exact consumed-history public baseline replay passed every "
                            "descriptive hard gate and enters strict forward-only paper "
                            "observation without sealed or unseen OOS authority."
                        ),
                    )
                else:
                    self.strategies.approve(
                        version_id,
                        actor=actor,
                        reason=(
                            "Transparent public baseline passed its immutable Qlib formal "
                            "OOS, historical, cost, PIT, statistical, risk and execution "
                            "gates."
                        ),
                    )
            except ValueError:
                # Concurrent scheduler ticks may both observe the successful
                # immutable backtest.  Recover only if the competing caller
                # completed the exact safe approval transition.
                raced = self.strategies.get_version(version_id)
                governed_noop = self._governed_noop_result(raced)
                if governed_noop is not None:
                    return governed_noop
                if raced.get("status") != "approved":
                    raise
        version = self.strategies.get_version(version_id)
        governed_noop = self._governed_noop_result(version)
        if governed_noop is not None:
            return governed_noop
        if version.get("status") != "approved":
            raise ValueError("strict approval did not produce an approved paper version")
        # StrategyStore.approve opens this automatically. A second idempotent
        # call recovers the safe state where approval committed but stage setup
        # failed. Investor capital/permissions are still required by PromotionStore.
        stage = self.promotions.prepare_paper_stage(version_id, actor=actor)
        version = self.strategies.get_version(version_id)
        governed_noop = self._governed_noop_result(version)
        if governed_noop is not None:
            return governed_noop
        if version.get("status") != "approved":
            raise ValueError("strict approval did not end in paper validation")
        return {"state": "paper_validating", "paper_stage": stage}

    def reconcile_forward_only_rehabilitation(
        self,
        *,
        actor: str = DEFAULT_ACTOR,
    ) -> dict[str, Any]:
        """Create/reuse the one exact v18 replay and its ordinary backtest job.

        This entry point never reserves a new OOS vintage and never reopens the
        v17 job.  It replays the already-consumed window under descriptive-only
        authority, then reuses the existing paper/promotion lifecycle.
        """

        result: dict[str, Any] = {
            "contract_version": FORWARD_ONLY_RECONCILE_RESULT_VERSION,
            "status": "failed",
            "source_strategy_version_id": SOURCE_VERSION_ID,
            "source_backtest_id": SOURCE_BACKTEST_ID,
            "source_job_id": SOURCE_JOB_ID,
            "strategy_version_id": None,
            "backtest": None,
            "job": None,
            "state": "failed",
            "source_cash_only_lockbox": None,
            "errors": [],
        }
        try:
            source = self.strategies.get_version(SOURCE_VERSION_ID)
            dataset = _select_forward_only_dataset(self.dataset_loader(self.data_root))
            recipe_version = FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
            preliminary_plan = _build_forward_only_rehabilitation_plan(
                source_version=source,
                dataset=dataset,
                consumed_oos_vintage_id=None,
            )
            with self.strategies.engine.begin() as connection:
                require_source_cancellation(connection)
                source_cash_only = require_source_cash_only_lockbox(connection)
                vintage = require_consumed_vintage(
                    connection,
                    version_config=preliminary_plan["config"],
                    dataset_identity_sha256=SOURCE_DATASET_IDENTITY_SHA256,
                    dataset_lineage_id=SOURCE_DATASET_LINEAGE_ID,
                )
            plan = _build_forward_only_rehabilitation_plan(
                source_version=source,
                dataset=dataset,
                consumed_oos_vintage_id=str(vintage.id),
            )
            result["source_cash_only_lockbox"] = source_cash_only
            family = self.strategies.get(str(source["strategy_id"]))
            current_candidates = [
                version
                for version in family.get("versions") or []
                if (version.get("config") or {}).get("recipe_version") == recipe_version
            ]
            exact_candidates = [
                version
                for version in current_candidates
                if version.get("benchmark") == source["benchmark"]
                and version.get("universe") == source["universe"]
                and version.get("factors") == []
                and version.get("config") == plan["config"]
            ]
            if len(current_candidates) != len(exact_candidates) or len(exact_candidates) > 1:
                raise ValueError(
                    "strategy family contains a conflicting or duplicate v18 replay version"
                )
            version_action = "reused" if exact_candidates else "created"
            if exact_candidates:
                version = exact_candidates[0]
            else:
                version = self.strategies.create_version_if_absent(
                    str(source["strategy_id"]),
                    benchmark=str(source["benchmark"]),
                    universe=str(source["universe"]),
                    factors=[],
                    config=plan["config"],
                    actor=actor,
                )
            if (
                str(version.get("id") or "") == SOURCE_VERSION_ID
                or version.get("config") != plan["config"]
                or version.get("evidence_mode") != EVIDENCE_MODE_REPLAY
            ):
                raise ValueError("v18 StrategyVersion differs from the frozen replay plan")
            result["strategy_version_id"] = str(version["id"])
            result["strategy_version_action"] = version_action
            queued = self._ensure_backtest_job(
                plan=plan,
                version=version,
                dataset=dataset,
                allow_create=version_action == "created",
            )
            backtest = queued["backtest"]
            job = queued["job"]
            expected_artifact_path = (
                self.artifact_root / str(backtest["id"])
            ).resolve()
            if (
                str(backtest["id"]) == SOURCE_BACKTEST_ID
                or Path(str(backtest.get("artifact_path") or "")).resolve()
                != expected_artifact_path
                or (job is not None and str(job["id"]) == SOURCE_JOB_ID)
            ):
                raise ValueError(
                    "v18 rehabilitation must use a new exact backtest/job artifact identity"
                )
            result["backtest"] = {
                "id": str(backtest["id"]),
                "status": str(backtest["status"]),
                "action": queued["backtest_action"],
                "evidence_mode": backtest.get("evidence_mode"),
            }
            result["job"] = (
                {
                    "id": str(job["id"]),
                    "status": str(job["status"]),
                    "action": queued["job_action"],
                }
                if job is not None
                else None
            )
            advanced = self._advance_paper(
                version_id=str(version["id"]),
                backtest=backtest,
                actor=actor,
            )
            result.update(advanced)
            result["status"] = (
                "paper_validating"
                if advanced["state"] == "paper_validating"
                else "no_op"
                if advanced["state"] == "governed_no_op"
                else "failed"
                if advanced["state"] == "formal_backtest_failed"
                else "pending"
            )
        except Exception as exc:  # noqa: BLE001 - one-shot emits a durable report
            result["errors"].append(str(exc))
        return result

    def reconcile(self, *, actor: str = DEFAULT_ACTOR) -> dict[str, Any]:
        result: dict[str, Any] = {
            "contract_version": RECONCILE_RESULT_VERSION,
            "status": "failed",
            "dataset": None,
            "unopened_history_selection": None,
            "joint_lockbox": None,
            "members": [],
            "recommendation_enabled_created": False,
            "investor_profile_bypassed": False,
            "safety_mode_changed": False,
            "errors": [],
        }
        try:
            families = self._existing_families()
            anchored = self._anchored_dataset(families)
            dataset = _select_dataset(
                self.dataset_loader(self.data_root),
                anchored_name=anchored,
            )
            result["dataset"] = {
                key: dataset[key]
                for key in (
                    "name",
                    "start_date",
                    "end_date",
                    "trading_days",
                    "dataset_identity_sha256",
                    "dataset_lineage_id",
                )
            }
            current_recipe_versions = {
                str(get_strategy_recipe(recipe_id)["version"])
                for recipe_id in TRANSPARENT_RESEARCH_BASELINE_IDS
            }
            if len(current_recipe_versions) != 1:
                raise ValueError(
                    "transparent baseline batch mixes current recipe versions"
                )
            current_recipe_version = next(iter(current_recipe_versions))
            repair_selection = (
                self.lockboxes.resolve_preregistered_single_member_repair(
                    calendar_days=dataset["calendar"],
                    current_recipe_version=current_recipe_version,
                )
            )
            history_selection = repair_selection
            if history_selection is None:
                history_selection = self.lockboxes.resolve_unopened_history_selection(
                    calendar_days=dataset["calendar"],
                    current_recipe_version=current_recipe_version,
                    anchored_selection=self._anchored_unopened_history_selection(
                        families
                    ),
                )
            selection_evidence = dict(history_selection["evidence"])
            research_dataset = {
                **dataset,
                "source_calendar": list(dataset["calendar"]),
                "calendar": list(history_selection["calendar"]),
                "unopened_history_selection": selection_evidence,
            }
            result["unopened_history_selection"] = selection_evidence
            plans: list[dict[str, Any]] = []
            unavailable_horizons: list[dict[str, Any]] = []
            for recipe_id in TRANSPARENT_RESEARCH_BASELINE_IDS:
                try:
                    plans.append(
                        _plan_member(recipe_id=recipe_id, dataset=research_dataset)
                    )
                except ResearchWindowUnavailableError as exc:
                    recipe = get_strategy_recipe(recipe_id)
                    evidence = dict(exc.evidence)
                    unavailable_horizons.append(
                        {
                            "recipe_id": recipe_id,
                            "horizon_profile": str(recipe["horizon"]),
                            "status": "unavailable",
                            "reason": str(exc),
                            "evidence": evidence,
                            "evidence_sha256": canonical_sha256(evidence),
                        }
                    )
            if repair_selection is not None:
                _require_preregistered_single_member_repair_oos(
                    repair_selection=repair_selection,
                    plans=plans,
                    unavailable_horizons=unavailable_horizons,
                )
            if not plans:
                reason = "no transparent baseline horizon has enough unopened evidence"
                # There is no statistical member to reserve, so building a
                # joint lockbox would be dishonest.  Still project every
                # deterministic exclusion into the reconcile report instead
                # of losing it behind the generic terminal error.
                result["joint_lockbox"] = {
                    "status": "unavailable",
                    "reason": reason,
                    "unavailable_horizons": deepcopy(unavailable_horizons),
                }
                result["members"] = _unavailable_member_results(
                    unavailable_horizons
                )
                result["errors"].append(reason)
                return result
            lockbox = build_joint_lockbox(
                dataset=str(dataset["name"]),
                dataset_identity_sha256=str(dataset["dataset_identity_sha256"]),
                dataset_lineage_id=str(dataset["dataset_lineage_id"]),
                members=[plan["lockbox_member"] for plan in plans],
                unopened_history_selection=selection_evidence,
                unavailable_horizons=unavailable_horizons,
            )
            for plan in plans:
                plan["config"] = _normalize_multifactor_contract(
                    {
                        **dict(plan["base_config"]),
                        LOCKBOX_CONFIG_KEY: deepcopy(lockbox),
                    },
                    factor_count=0,
                    creating_family=True,
                )
                # Validate the full config before it can reach StrategyStore.
                lockbox_member_link(plan["config"])
            versions = self._ensure_versions(
                plans=plans,
                families=families,
                actor=actor,
            )
            if all(
                plan.get("lifecycle_action") == "governed_no_op" for plan in plans
            ):
                # The bootstrap no longer owns any member in this batch. Do not
                # even reopen the lockbox reservation transaction; the existing
                # strategy, paper and recommendation evidence stays untouched.
                reservation = {
                    "status": "governed_no_op",
                    "reason": "all transparent baselines are governance controlled",
                }
            else:
                reservation = self.lockboxes.reserve(
                    versions=versions,
                    dataset=str(dataset["name"]),
                    dataset_identity_sha256=str(dataset["dataset_identity_sha256"]),
                    dataset_lineage_id=str(dataset["dataset_lineage_id"]),
                )
            # Reservation rows exist only for horizons that consume an OOS
            # vintage.  Preserve the validated v3 cash-only declarations in
            # the reconcile projection as well, so operators and the novice
            # UI can distinguish an intentionally unavailable sleeve from a
            # missing strategy lane without opening StrategyVersion internals.
            result["joint_lockbox"] = {
                **reservation,
                **(
                    {
                        "unavailable_horizons": deepcopy(
                            lockbox["unavailable_horizons"]
                        )
                    }
                    if lockbox.get("unavailable_horizons")
                    else {}
                ),
            }
        except Exception as exc:  # noqa: BLE001 - reconcile must return a failed result
            result["errors"].append(str(exc))
            return result

        member_results = _unavailable_member_results(unavailable_horizons)
        for plan, version in zip(plans, versions, strict=True):
            member = {
                "recipe_id": plan["recipe"]["id"],
                "horizon_profile": plan["recipe"]["horizon"],
                "family_action": plan.get("family_action"),
                "strategy_version_id": str(version["id"]),
                "research_window_contract_sha256": plan[
                    "research_window_contract_sha256"
                ],
                "formal_periods": dict(plan["formal_periods"]),
                "state": "failed",
                "backtest": None,
                "job": None,
                "paper_stage": None,
                "lifecycle": plan.get("lifecycle"),
                "errors": [],
            }
            if plan.get("lifecycle_action") == "governed_no_op":
                member["state"] = "governed_no_op"
                member_results.append(member)
                continue
            try:
                queued = self._ensure_backtest_job(
                    plan=plan,
                    version=version,
                    dataset=research_dataset,
                )
                backtest = queued["backtest"]
                member.update(
                    {
                        "backtest": {
                            "id": backtest["id"],
                            "status": backtest["status"],
                            "action": queued["backtest_action"],
                        },
                        "job": (
                            {
                                "id": queued["job"]["id"],
                                "status": queued["job"]["status"],
                                "action": queued["job_action"],
                            }
                            if queued["job"] is not None
                            else None
                        ),
                    }
                )
                advanced = self._advance_paper(
                    version_id=str(version["id"]),
                    backtest=backtest,
                    actor=actor,
                )
                member.update(advanced)
            except Exception as exc:  # noqa: BLE001 - isolate one failed horizon
                member["errors"].append(str(exc))
            member_results.append(member)
        recipe_order = {
            recipe_id: index
            for index, recipe_id in enumerate(TRANSPARENT_RESEARCH_BASELINE_IDS)
        }
        member_results.sort(key=lambda item: recipe_order[str(item["recipe_id"])])
        result["members"] = member_results
        if any(
            item["errors"] or item["state"] == "formal_backtest_failed"
            for item in member_results
        ):
            result["status"] = "failed"
        elif all(
            item["state"] in {"governed_no_op", "unavailable"}
            for item in member_results
        ):
            result["status"] = "no_op"
        elif all(
            item["state"]
            in {"paper_validating", "governed_no_op", "unavailable"}
            for item in member_results
        ):
            result["status"] = "paper_validating"
        else:
            result["status"] = "pending"
        return result


def reconcile(settings: Settings, *, actor: str = DEFAULT_ACTOR) -> dict[str, Any]:
    """SchedulerEngine-compatible entry point; no scheduler state is mutated."""

    return TransparentBaselineBootstrapService(
        database_url=settings.database_url,
        data_root=settings.data_root,
    ).reconcile(actor=actor)


def reconcile_forward_only_rehabilitation(
    settings: Settings,
    *,
    actor: str = DEFAULT_ACTOR,
) -> dict[str, Any]:
    """One-shot command entry point for the exact consumed-history v18 replay."""

    return TransparentBaselineBootstrapService(
        database_url=settings.database_url,
        data_root=settings.data_root,
    ).reconcile_forward_only_rehabilitation(actor=actor)
