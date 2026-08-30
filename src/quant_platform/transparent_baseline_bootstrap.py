"""Idempotent one-step bootstrap for the three transparent public baselines."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

from quant_data.config import Settings
from quant_data.execution_contract import require_daily_qlib_contract
from quant_platform.job_store import JobStore
from quant_platform.promotion import PromotionStore
from quant_platform.research_automation import resolve_research_window_contract
from quant_platform.research_horizon import research_horizon_contract
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
)
from quant_platform.transparent_baseline_runner import (
    TRANSPARENT_BASELINE_JOB_RUNNER_FIELD,
    TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD,
    target_runner_for_recipe,
    target_runtime_bundle_for_recipe,
)

BOOTSTRAP_CONTRACT_VERSION = "transparent-baseline-bootstrap-v1"
RECONCILE_RESULT_VERSION = "transparent-baseline-reconcile-v1"
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


def _plan_member(
    *, recipe_id: str, dataset: Mapping[str, Any]
) -> dict[str, Any]:
    recipe = get_strategy_recipe(recipe_id)
    recipe_sha256 = canonical_sha256(recipe)
    feature_set = _feature_set(recipe)
    periods, evidence = resolve_research_window_contract(
        dict(dataset),
        list(dataset["calendar"]),
        horizon_profile=str(recipe["horizon"]),
        feature_set=feature_set,
        universe=str(recipe["universe"]),
    )
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
            ):
                raise ValueError("existing formal backtest differs from the frozen lockbox")
            backtest_action = "reused"
        else:
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
        if status in {"queued", "running"}:
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
            job = self.jobs.create(
                "strategy_backtest",
                payload,
                self.log_root / f"strategy-backtest-{backtest['id']}.log",
                dedupe_active_kind=False,
                idempotency_key=(
                    f"transparent-baseline:{version['id']}:{backtest['id']}"
                ),
                max_attempts=1,
            )
            job_action = "reused" if backtest.get("job_id") else "created"
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
                self.strategies.approve(
                    version_id,
                    actor=actor,
                    reason=(
                        "Transparent public baseline passed its immutable Qlib formal OOS, "
                        "historical, cost, PIT, statistical, risk and execution gates."
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

    def reconcile(self, *, actor: str = DEFAULT_ACTOR) -> dict[str, Any]:
        result: dict[str, Any] = {
            "contract_version": RECONCILE_RESULT_VERSION,
            "status": "failed",
            "dataset": None,
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
            plans = [
                _plan_member(recipe_id=recipe_id, dataset=dataset)
                for recipe_id in TRANSPARENT_RESEARCH_BASELINE_IDS
            ]
            lockbox = build_joint_lockbox(
                dataset=str(dataset["name"]),
                dataset_identity_sha256=str(dataset["dataset_identity_sha256"]),
                dataset_lineage_id=str(dataset["dataset_lineage_id"]),
                members=[plan["lockbox_member"] for plan in plans],
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
            result["joint_lockbox"] = reservation
        except Exception as exc:  # noqa: BLE001 - reconcile must return a failed result
            result["errors"].append(str(exc))
            return result

        member_results: list[dict[str, Any]] = []
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
                    dataset=dataset,
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
        result["members"] = member_results
        if any(
            item["errors"] or item["state"] == "formal_backtest_failed"
            for item in member_results
        ):
            result["status"] = "failed"
        elif all(item["state"] == "governed_no_op" for item in member_results):
            result["status"] = "no_op"
        elif all(
            item["state"] in {"paper_validating", "governed_no_op"}
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
