from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict
from datetime import UTC, date, datetime
from math import isfinite
from pathlib import Path
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    backtest_runs,
    factor_candidates,
    factor_evaluations,
    oos_vintages,
    open_database,
    research_campaigns,
    row_dict,
    strategies,
    strategy_events,
    strategy_factors,
    strategy_pairs,
    strategy_versions,
)
from quant_data.execution_contract import (
    require_daily_qlib_contract,
    require_minute_execution_contract,
    require_strategy_execution_contract,
    strategy_execution_contract_hash,
)
from quant_platform.cost_model import KNOWN_COST_SCHEDULE_VERSIONS, CostModelConfig
from quant_platform.eligibility import ELIGIBILITY_CONTRACT_VERSION
from quant_platform.formal_validation import (
    FORMAL_VALIDATION_CONTRACT_VERSION,
    SIGNAL_DECAY_FRONTIER_VERSION,
)
from quant_platform.pair_trading import PairTradingConfig
from quant_platform.qlib_backtest import (
    COMPONENT_COST_STRESS_MULTIPLIERS,
    QLIB_ENGINE_VERSION,
)
from quant_platform.qlib_factor_baseline import (
    FACTOR_SOURCE_QLIB_BASELINE,
    baseline_manifest_failures,
    bind_factor_source_config,
)
from quant_platform.qlib_workflow import require_qlib_workflow_identity
from quant_platform.statistical_validation import DEFLATED_SHARPE_METHOD_VERSION
from quant_platform.strategy_catalog import require_capital_eligible_strategy_type
from quant_platform.upstream_versions import QLIB_COMMIT, RDAGENT_COMMIT


def _now() -> datetime:
    return datetime.now(UTC)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _version_contract_columns(
    config: dict[str, Any], *, strategy_type: str
) -> dict[str, Any]:
    if strategy_type == "pair":
        signal_frequency = "day"
        signal_horizon = "1d"
        execution_frequency = "1min"
        contract_hash = _canonical_sha256(
            {
                "strategy_type": "pair",
                "signal_frequency": signal_frequency,
                "signal_horizon": signal_horizon,
                "execution_frequency": execution_frequency,
                "config": config,
            }
        )
    else:
        signal_frequency = str(config.get("signal_frequency") or "day")
        signal_horizon = f"{int(config.get('signal_period') or 1)}bar"
        execution_frequency = str(config.get("execution_frequency") or "day")
        contract_hash = str(config.get("execution_contract_hash") or "")
        if not _is_sha256(contract_hash):
            raise ValueError("strategy execution contract hash is required")
    return {
        "signal_frequency": signal_frequency,
        "signal_horizon": signal_horizon,
        "execution_frequency": execution_frequency,
        "execution_contract_hash": contract_hash,
        "qlib_version": f"0.0.dev0+g{QLIB_COMMIT}",
        "qlib_commit": QLIB_COMMIT,
        "rdagent_version": f"0.0.dev0+g{RDAGENT_COMMIT}",
        "rdagent_commit": RDAGENT_COMMIT,
    }


def _normalize_multifactor_contract(
    config: dict[str, Any], *, factor_count: int, creating_family: bool
) -> dict[str, Any]:
    normalized = bind_factor_source_config(
        config,
        factor_count=factor_count,
        creating_family=creating_family,
    )
    normalized.setdefault("signal_frequency", "day")
    normalized.setdefault("signal_period", 1)
    normalized.setdefault("execution_frequency", "day")
    normalized.setdefault("execution_lag_bars", 1)
    normalized.setdefault("execution_method", "open")
    normalized.setdefault("execution_days", 1)
    normalized.setdefault("execution_slice_minutes", 20)
    normalized.setdefault("max_execution_slices", 24)
    normalized["execution_contract_hash"] = strategy_execution_contract_hash(normalized)
    require_strategy_execution_contract(normalized)
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenario_artifact_failures(
    scenarios: dict[str, Any], artifact_root: Path
) -> list[str]:
    """Validate each scenario's immutable artifacts against the artifact root."""

    failures: list[str] = []
    for scenario_name, scenario in scenarios.items():
        artifacts = scenario.get("artifacts") if isinstance(scenario, dict) else None
        valid_artifacts = isinstance(artifacts, dict) and set(artifacts) == {
            "daily_report",
            "fills",
            "metrics",
        }
        if valid_artifacts:
            for entry in artifacts.values():
                if not isinstance(entry, dict) or not _is_sha256(entry.get("sha256")):
                    valid_artifacts = False
                    break
                try:
                    artifact_path = (artifact_root / str(entry["path"])).resolve()
                    artifact_path.relative_to(artifact_root)
                except (KeyError, ValueError):
                    valid_artifacts = False
                    break
                if not artifact_path.is_file() or _sha256_file(artifact_path) != entry["sha256"]:
                    valid_artifacts = False
                    break
        if not valid_artifacts:
            failures.append(
                f"robustness scenario {scenario_name} has no complete immutable artifacts"
            )
    return failures


def _formal_validation_failures(
    version: dict[str, Any], metrics: dict[str, Any]
) -> list[str]:
    evidence = metrics.get("formal_validation")
    if not isinstance(evidence, dict):
        return ["formal validation evidence is required"]
    failures: list[str] = []
    if evidence.get("contract_version") != FORMAL_VALIDATION_CONTRACT_VERSION:
        failures.append("formal validation contract version is missing or obsolete")
    if evidence.get("status") != "passed" or metrics.get(
        "formal_validation_passed"
    ) is not True:
        failures.append("formal validation suite did not pass")

    outer = evidence.get("outer_walk_forward")
    coverage = outer.get("candidate_coverage") if isinstance(outer, dict) else {}
    trials = int((metrics.get("deflated_sharpe") or {}).get("trials") or 1)
    config = version.get("config", {})
    minimum_outer_test_metric = float(
        config.get("minimum_outer_test_excess_return", 0.0)
    )
    minimum_outer_test_pass_rate = float(
        config.get("minimum_outer_test_pass_rate", 0.60)
    )
    outer_folds = outer.get("folds") if isinstance(outer, dict) else None

    def valid_outer_fold(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        try:
            test_metric = float(item.get("test_metric"))
        except (TypeError, ValueError):
            return False
        recorded_passed = item.get("test_passed")
        return isinstance(recorded_passed, bool) and isfinite(test_metric) and recorded_passed == (
            test_metric > minimum_outer_test_metric
        )

    valid_outer_folds = (
        isinstance(outer_folds, list)
        and bool(outer_folds)
        and all(valid_outer_fold(item) for item in outer_folds)
    )
    if valid_outer_folds:
        outer_test_metrics = [float(item["test_metric"]) for item in outer_folds]
        calculated_test_pass_rate = sum(
            bool(item["test_passed"]) for item in outer_folds
        ) / len(outer_folds)
        calculated_mean_test_metric = sum(outer_test_metrics) / len(outer_test_metrics)
    else:
        calculated_test_pass_rate = float("-inf")
        calculated_mean_test_metric = float("-inf")
    try:
        recorded_test_pass_rate = float(outer.get("test_pass_rate"))
        recorded_mean_test_metric = float(outer.get("mean_test_metric"))
    except (AttributeError, TypeError, ValueError):
        recorded_test_pass_rate = float("-inf")
        recorded_mean_test_metric = float("-inf")

    if (
        not isinstance(outer, dict)
        or outer.get("status") != "completed"
        or outer.get("passed") is not True
        or int(outer.get("fold_count") or 0) < 3
        or int((coverage or {}).get("required_group_trials") or 0) != trials
        or int((coverage or {}).get("provided_candidates") or 0) != trials
        or recorded_test_pass_rate < minimum_outer_test_pass_rate
        or recorded_mean_test_metric <= minimum_outer_test_metric
        or not isinstance(outer_folds, list)
        or len(outer_folds) != int(outer.get("fold_count") or 0)
        or not valid_outer_folds
        or abs(recorded_test_pass_rate - calculated_test_pass_rate) > 1e-12
        or abs(recorded_mean_test_metric - calculated_mean_test_metric) > 1e-12
    ):
        failures.append(
            "outer walk-forward must cover the complete candidate set and pass OOS gates"
        )

    baseline = version.get("config", {}).get("baseline_definition")
    expected_components = len(version.get("factors") or []) + len(
        (baseline or {}).get("factors") or []
    )
    ablation = evidence.get("ablation")
    if (
        not isinstance(ablation, dict)
        or ablation.get("status") != "passed"
        or len(ablation.get("runs") or []) != expected_components
        or any(
            not isinstance(item, dict)
            or item.get("passed") is not True
            or not isinstance(item.get("metrics"), dict)
            for item in ablation.get("runs") or []
        )
    ):
        failures.append("complete passing component ablation evidence is required")

    decay = evidence.get("signal_decay")
    if (
        not isinstance(decay, dict)
        or decay.get("status") != "completed"
        or decay.get("frontier_version") != SIGNAL_DECAY_FRONTIER_VERSION
        or decay.get("maximum_supported_delay_bars") is None
        or not decay.get("runs")
    ):
        failures.append("signal-decay evidence did not establish a supported response delay")

    bootstrap = evidence.get("paired_block_bootstrap")
    interval = (
        bootstrap.get("confidence_interval_95")
        if isinstance(bootstrap, dict)
        else None
    )
    if (
        not isinstance(bootstrap, dict)
        or bootstrap.get("status") != "ok"
        or not isinstance(interval, list)
        or len(interval) != 2
        or float(interval[0]) <= 0
    ):
        failures.append(
            "paired moving-block bootstrap did not show positive baseline increment"
        )

    multiple = evidence.get("multiple_testing")
    if trials == 1:
        valid_multiple = (
            isinstance(multiple, dict)
            and multiple.get("status") == "not_applicable_single_trial"
            and len(multiple.get("holm_adjusted_p_values") or []) == 1
        )
    else:
        pbo = multiple.get("pbo") if isinstance(multiple, dict) else None
        valid_multiple = (
            isinstance(multiple, dict)
            and multiple.get("status") == "ok"
            and len(multiple.get("holm_adjusted_p_values") or []) == trials
            and isinstance(pbo, dict)
            and pbo.get("status") == "ok"
            and pbo.get("pbo") is not None
        )
    if not valid_multiple:
        failures.append(
            "Holm/PBO evidence must cover the shared hypothesis-group trial count"
        )
    return failures


def _multifactor_manifest_failures(
    version: dict[str, Any], backtest: dict[str, Any], metrics: dict[str, Any]
) -> list[str]:
    provenance = metrics.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    artifact_root = Path(str(backtest["artifact_path"]))
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.is_file():
        return ["strategy backtest manifest artifact is missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ["strategy backtest manifest artifact is unreadable"]
    if not isinstance(manifest, dict):
        return ["strategy backtest manifest artifact must be a JSON object"]

    failures: list[str] = []
    try:
        require_qlib_workflow_identity(provenance.get("qlib_workflow"))
    except ValueError as exc:
        failures.append(str(exc))
    manifest_sha256 = provenance.get("execution_manifest_sha256")
    if not _is_sha256(manifest_sha256) or _sha256_file(manifest_path) != manifest_sha256:
        failures.append("strategy backtest manifest does not match its SHA-256 provenance")
    config_sha256 = provenance.get("strategy_config_sha256")
    expected_config_sha256 = _canonical_sha256(version.get("config"))
    if config_sha256 != expected_config_sha256:
        failures.append("strategy config does not match its SHA-256 provenance")
    if _canonical_sha256(manifest.get("config")) != expected_config_sha256:
        failures.append("strategy backtest manifest config does not match the immutable version")
    for field, expected in (
        ("strategy_version_id", version.get("id")),
        ("dataset", backtest.get("dataset")),
        ("execution_dataset", backtest.get("execution_dataset")),
        ("benchmark", version.get("benchmark")),
        ("universe", version.get("universe")),
    ):
        if manifest.get(field) != expected:
            failures.append(f"strategy backtest manifest {field} does not match the run")
    if manifest.get("periods") != backtest.get("periods"):
        failures.append("strategy backtest manifest periods do not match the run")

    expected_factors = {
        str(item["factor_candidate_id"]): {
            "weight": float(item["weight"]),
            "direction": int(item["direction"]),
            "code_path": item.get("code_path"),
            "code_sha256": item.get("code_sha256"),
            "values_path": item.get("values_path"),
        }
        for item in version.get("factors", [])
    }
    manifest_items = manifest.get("factors")
    if not isinstance(manifest_items, list):
        failures.append("strategy backtest manifest factors are missing")
        return failures
    manifest_factors: dict[str, dict[str, Any]] = {}
    for item in manifest_items:
        if not isinstance(item, dict) or not str(item.get("candidate_id") or ""):
            failures.append("strategy backtest manifest contains an invalid factor")
            continue
        candidate_id = str(item["candidate_id"])
        if candidate_id in manifest_factors:
            failures.append("strategy backtest manifest contains duplicate factors")
        manifest_factors[candidate_id] = item
    if set(manifest_factors) != set(expected_factors):
        failures.append("strategy backtest manifest factors do not match the immutable version")
    for candidate_id, expected in expected_factors.items():
        item = manifest_factors.get(candidate_id)
        if item is None:
            continue
        try:
            numeric_matches = (
                abs(float(item.get("weight")) - expected["weight"]) <= 1e-12
                and int(item.get("direction")) == expected["direction"]
            )
        except (TypeError, ValueError):
            numeric_matches = False
        if not numeric_matches or item.get("code_sha256") != expected["code_sha256"]:
            failures.append(
                f"strategy backtest manifest factor {candidate_id} does not match the version"
            )
    code_hashes = provenance.get("factor_code_sha256")
    if not isinstance(code_hashes, dict) or code_hashes != {
        candidate_id: item["code_sha256"] for candidate_id, item in expected_factors.items()
    }:
        failures.append("factor code provenance does not match the immutable version")
    value_hashes = provenance.get("factor_values_sha256")
    for candidate_id, item in expected_factors.items():
        for artifact_kind, path_value, hashes in (
            ("code", item["code_path"], code_hashes),
            ("values", item["values_path"], value_hashes),
        ):
            artifact = Path(str(path_value)) if path_value else None
            recorded = hashes.get(candidate_id) if isinstance(hashes, dict) else None
            if artifact is None or not artifact.is_file():
                failures.append(f"factor {candidate_id} {artifact_kind} artifact is missing")
            elif not _is_sha256(recorded) or _sha256_file(artifact) != recorded:
                failures.append(
                    f"factor {candidate_id} {artifact_kind} artifact does not match provenance"
                )
    failures.extend(
        baseline_manifest_failures(
            config=version.get("config") or {},
            factor_count=len(version.get("factors") or []),
            artifact_root=artifact_root,
            manifest=manifest,
            provenance=provenance,
        )
    )
    return failures


def _pair_artifact_failures(
    version: dict[str, Any], backtest: dict[str, Any], metrics: dict[str, Any]
) -> list[str]:
    artifact_root = Path(str(backtest["artifact_path"]))
    manifest_path = artifact_root / "manifest.json"
    pair_manifest_path = artifact_root / "pair_artifact_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ["pair backtest artifact manifests are missing or unreadable"]
    if not isinstance(manifest, dict) or not isinstance(pair_manifest, dict):
        return ["pair backtest artifact manifests must be JSON objects"]
    provenance = metrics.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    failures: list[str] = []
    try:
        require_qlib_workflow_identity(provenance.get("qlib_workflow"))
    except ValueError as exc:
        failures.append(str(exc))
    if provenance.get("execution_manifest_sha256") != _sha256_file(manifest_path):
        failures.append("pair execution manifest does not match its SHA-256 provenance")
    if provenance.get("pair_artifact_manifest_sha256") != _sha256_file(
        pair_manifest_path
    ):
        failures.append("pair artifact manifest does not match its SHA-256 provenance")
    expected_config_sha256 = _canonical_sha256(version.get("config") or {})
    expected_pair = {
        key: (version.get("pair") or {}).get(key)
        for key in ("leg_y", "leg_x", "asset_class", "shorting_mode")
    }
    for candidate in (manifest, pair_manifest):
        observed_pair = {
            key: dict(candidate.get("pair") or {}).get(key) for key in expected_pair
        }
        if (
            candidate.get("backtest_id") != backtest.get("id")
            or candidate.get("strategy_version_id") != version.get("id")
            or candidate.get("dataset") != backtest.get("dataset")
            or candidate.get("periods") != backtest.get("periods")
            or candidate.get("execution_contract_hash")
            != version.get("execution_contract_hash")
            or observed_pair != expected_pair
        ):
            failures.append(
                "pair artifact manifest does not match the immutable strategy/backtest"
            )
            break
    if pair_manifest.get("format_version") != "pair-replay-artifact-v1":
        failures.append("pair artifact manifest format is unsupported")
    if pair_manifest.get("strategy_config_sha256") != expected_config_sha256:
        failures.append("pair artifact strategy config identity does not match the version")
    if _canonical_sha256(manifest.get("config") or {}) != expected_config_sha256:
        failures.append("pair execution manifest config does not match the version")
    files = pair_manifest.get("files")
    if not isinstance(files, dict):
        failures.append("pair artifact file manifest is missing")
        return failures
    for name in (
        "daily_returns.parquet",
        "daily_ledger.parquet",
        "kalman_spread.parquet",
        "trades.json",
        "rejections.json",
    ):
        evidence = files.get(name)
        path = artifact_root / name
        if (
            not isinstance(evidence, dict)
            or not path.is_file()
            or path.stat().st_size != int(evidence.get("bytes") or -1)
            or _sha256_file(path) != str(evidence.get("sha256") or "")
        ):
            failures.append(f"pair artifact {name} failed immutable verification")
    return failures


class StrategyStore:
    """Immutable strategy versions backed by promoted factors and audited approvals."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.engine = open_database(database_url)

    @staticmethod
    def _factor_evidence(
        connection: Any, factors: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        candidate_ids = [item["candidate_id"] for item in factors]
        candidate_rows = connection.execute(
            select(factor_candidates).where(factor_candidates.c.id.in_(candidate_ids))
        ).all()
        candidates = {str(row.id): row for row in candidate_rows}
        missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in candidates]
        if missing:
            raise ValueError(f"factor candidates not found: {', '.join(missing)}")
        not_promoted = [row.name for row in candidate_rows if row.status != "promoted"]
        if not_promoted:
            raise ValueError(
                "strategy versions may only use promoted factors: " + ", ".join(not_promoted)
            )
        evidence: dict[str, dict[str, Any]] = {}
        for candidate_id, candidate in candidates.items():
            if not candidate.promoted_evaluation_id or not _is_sha256(
                candidate.promotion_evidence_sha256
            ):
                raise ValueError(
                    f"promoted factor {candidate_id} has no immutable promotion evidence"
                )
            evaluation = connection.execute(
                select(factor_evaluations).where(
                    factor_evaluations.c.id == candidate.promoted_evaluation_id,
                    factor_evaluations.c.factor_candidate_id == candidate_id,
                )
            ).first()
            if not evaluation:
                raise ValueError(f"promoted factor {candidate_id} is not bound to its evaluation")
            if (
                evaluation.gate_status != "passed"
                or evaluation.is_legacy
                or str(evaluation.evaluator_version) != "factor-gate-v3-hac-bh"
                or evaluation.evidence_sha256 != candidate.promotion_evidence_sha256
                or not _is_sha256(evaluation.evidence_sha256)
            ):
                raise ValueError(f"promoted factor {candidate_id} has invalid promotion evidence")
            if (
                _canonical_sha256(evaluation.metrics_json) != evaluation.metrics_sha256
                or _canonical_sha256(evaluation.policy_json) != evaluation.policy_sha256
            ):
                raise ValueError(f"promoted factor {candidate_id} evaluation provenance is invalid")
            for artifact_kind, path_value, expected in (
                ("code", candidate.code_path, candidate.code_sha256),
                ("values", candidate.values_path, candidate.values_sha256),
                ("evaluation", evaluation.artifact_path, evaluation.artifact_sha256),
            ):
                artifact = Path(str(path_value)) if path_value else None
                if (
                    artifact is None
                    or not artifact.is_file()
                    or not _is_sha256(expected)
                    or _sha256_file(artifact) != expected
                ):
                    raise ValueError(
                        f"promoted factor {candidate_id} {artifact_kind} "
                        "evidence is missing or changed"
                    )
            if (
                evaluation.candidate_code_sha256 != candidate.code_sha256
                or evaluation.candidate_values_sha256 != candidate.values_sha256
            ):
                raise ValueError(
                    f"promoted factor {candidate_id} artifacts do not match its evaluation"
                )
            evidence[candidate_id] = {
                "id": str(evaluation.id),
                "direction": -1 if evaluation.metrics_json.get("direction") == "inverted" else 1,
            }
        return evidence

    def create(
        self,
        *,
        name: str,
        description: str,
        benchmark: str,
        universe: str,
        factors: list[dict[str, Any]],
        config: dict[str, Any],
        actor: str,
        economic_hypothesis_group: str | None = None,
        hypothesis_group_cap: float = 0.70,
    ) -> dict[str, Any]:
        config = _normalize_multifactor_contract(
            config, factor_count=len(factors), creating_family=True
        )
        if len({item["candidate_id"] for item in factors}) != len(factors):
            raise ValueError("factor candidates must be unique within a strategy version")
        total_weight = sum(abs(float(item["weight"])) for item in factors)
        if factors and total_weight <= 0:
            raise ValueError("factor weights must not all be zero")
        strategy_id = uuid.uuid4().hex
        group = str(economic_hypothesis_group or strategy_id).strip()
        if not group or len(group) > 200:
            raise ValueError("economic hypothesis group must contain 1 to 200 characters")
        if not 0 < float(hypothesis_group_cap) <= 0.70:
            raise ValueError("hypothesis group capital cap must be in (0, 0.70]")
        version_id = uuid.uuid4().hex
        now = _now()
        try:
            with self.engine.begin() as connection:
                evaluation_evidence = self._factor_evidence(connection, factors)
                connection.execute(
                    insert(strategies).values(
                        id=strategy_id,
                        name=name,
                        description=description,
                        status="draft",
                        economic_hypothesis_group=group,
                        hypothesis_group_cap=float(hypothesis_group_cap),
                        created_by=actor,
                        created_at=now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    insert(strategy_versions).values(
                        id=version_id,
                        strategy_id=strategy_id,
                        version=1,
                        status="draft",
                        strategy_type="multifactor",
                        **_version_contract_columns(config, strategy_type="multifactor"),
                        benchmark=benchmark,
                        universe=universe,
                        config_json=config,
                        created_by=actor,
                        created_at=now,
                    )
                )
                factor_rows = [
                        {
                            "strategy_version_id": version_id,
                            "factor_candidate_id": item["candidate_id"],
                            "factor_evaluation_id": evaluation_evidence[item["candidate_id"]]["id"],
                            "weight": float(item["weight"]) / total_weight,
                            "direction": evaluation_evidence[item["candidate_id"]]["direction"],
                            "created_at": now,
                        }
                        for item in factors
                    ]
                if factor_rows:
                    connection.execute(insert(strategy_factors), factor_rows)
                self._event(
                    connection,
                    strategy_id=strategy_id,
                    version_id=version_id,
                    event_type="strategy.created",
                    actor=actor,
                    payload={"benchmark": benchmark, "universe": universe},
                )
        except IntegrityError as exc:
            raise ValueError(f"strategy name {name!r} already exists") from exc
        return self.get(strategy_id)

    def create_version(
        self,
        strategy_id: str,
        *,
        benchmark: str,
        universe: str,
        factors: list[dict[str, Any]],
        config: dict[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        config = _normalize_multifactor_contract(
            config, factor_count=len(factors), creating_family=False
        )
        if len({item["candidate_id"] for item in factors}) != len(factors):
            raise ValueError("factor candidates must be unique within a strategy version")
        total_weight = sum(abs(float(item["weight"])) for item in factors)
        if factors and total_weight <= 0:
            raise ValueError("factor weights must not all be zero")
        version_id = uuid.uuid4().hex
        now = _now()
        try:
            with self.engine.begin() as connection:
                strategy = connection.execute(
                    select(strategies).where(strategies.c.id == strategy_id).with_for_update()
                ).first()
                if strategy is None:
                    raise KeyError(strategy_id)
                family_type = connection.scalar(
                    select(strategy_versions.c.strategy_type)
                    .where(strategy_versions.c.strategy_id == strategy_id)
                    .limit(1)
                )
                if family_type != "multifactor":
                    raise ValueError("pair strategy families require a pair strategy version")
                evaluation_evidence = self._factor_evidence(connection, factors)
                latest = connection.scalar(
                    select(func.max(strategy_versions.c.version)).where(
                        strategy_versions.c.strategy_id == strategy_id
                    )
                )
                version_number = int(latest or 0) + 1
                connection.execute(
                    insert(strategy_versions).values(
                        id=version_id,
                        strategy_id=strategy_id,
                        version=version_number,
                        status="draft",
                        strategy_type="multifactor",
                        **_version_contract_columns(config, strategy_type="multifactor"),
                        benchmark=benchmark,
                        universe=universe,
                        config_json=config,
                        created_by=actor,
                        created_at=now,
                    )
                )
                factor_rows = [
                        {
                            "strategy_version_id": version_id,
                            "factor_candidate_id": item["candidate_id"],
                            "factor_evaluation_id": evaluation_evidence[item["candidate_id"]]["id"],
                            "weight": float(item["weight"]) / total_weight,
                            "direction": evaluation_evidence[item["candidate_id"]]["direction"],
                            "created_at": now,
                        }
                        for item in factors
                    ]
                if factor_rows:
                    connection.execute(insert(strategy_factors), factor_rows)
                connection.execute(
                    update(strategies).where(strategies.c.id == strategy_id).values(updated_at=now)
                )
                self._event(
                    connection,
                    strategy_id=strategy_id,
                    version_id=version_id,
                    event_type="strategy.version_created",
                    actor=actor,
                    payload={
                        "version": version_number,
                        "benchmark": benchmark,
                        "universe": universe,
                    },
                )
        except IntegrityError as exc:
            raise ValueError("strategy version creation conflicted with another request") from exc
        return self.get_version(version_id)

    @staticmethod
    def _validate_pair_definition(
        *,
        leg_y: str,
        leg_x: str,
        asset_class: str,
        shorting_mode: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        first = leg_y.strip().upper()
        second = leg_x.strip().upper()
        if not first or not second or first == second:
            raise ValueError("pair strategy requires two distinct instruments")
        if asset_class not in {"etf", "stock", "mixed"}:
            raise ValueError("pair asset_class must be etf, stock, or mixed")
        if shorting_mode != "margin_borrow":
            raise ValueError("only governed margin_borrow pair strategies are supported")
        validated = PairTradingConfig(**config)
        return {
            "leg_y": first,
            "leg_x": second,
            "asset_class": asset_class,
            "shorting_mode": shorting_mode,
            "config": asdict(validated),
        }

    def create_pair(
        self,
        *,
        name: str,
        description: str,
        leg_y: str,
        leg_x: str,
        asset_class: str,
        shorting_mode: str,
        config: dict[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        if not name.strip() or not description.strip() or not actor.strip():
            raise ValueError("pair strategy name, description, and actor are required")
        definition = self._validate_pair_definition(
            leg_y=leg_y,
            leg_x=leg_x,
            asset_class=asset_class,
            shorting_mode=shorting_mode,
            config=config,
        )
        strategy_id = uuid.uuid4().hex
        version_id = uuid.uuid4().hex
        now = _now()
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(strategies).values(
                        id=strategy_id,
                        name=name.strip(),
                        description=description.strip(),
                        status="draft",
                        created_by=actor.strip(),
                        created_at=now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    insert(strategy_versions).values(
                        id=version_id,
                        strategy_id=strategy_id,
                        version=1,
                        status="draft",
                        strategy_type="pair",
                        **_version_contract_columns(
                            definition["config"], strategy_type="pair"
                        ),
                        benchmark="CASH",
                        universe=f"pair:{definition['leg_y']}:{definition['leg_x']}",
                        config_json=definition["config"],
                        created_by=actor.strip(),
                        created_at=now,
                    )
                )
                connection.execute(
                    insert(strategy_pairs).values(
                        strategy_version_id=version_id,
                        leg_y=definition["leg_y"],
                        leg_x=definition["leg_x"],
                        asset_class=definition["asset_class"],
                        shorting_mode=definition["shorting_mode"],
                        created_at=now,
                    )
                )
                self._event(
                    connection,
                    strategy_id=strategy_id,
                    version_id=version_id,
                    event_type="strategy.pair_created",
                    actor=actor.strip(),
                    payload={key: definition[key] for key in ("leg_y", "leg_x", "asset_class")},
                )
        except IntegrityError as exc:
            raise ValueError(f"strategy name {name!r} already exists") from exc
        return self.get(strategy_id)

    def create_pair_version(
        self,
        strategy_id: str,
        *,
        leg_y: str,
        leg_x: str,
        asset_class: str,
        shorting_mode: str,
        config: dict[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        if not actor.strip():
            raise ValueError("pair strategy version actor is required")
        definition = self._validate_pair_definition(
            leg_y=leg_y,
            leg_x=leg_x,
            asset_class=asset_class,
            shorting_mode=shorting_mode,
            config=config,
        )
        version_id = uuid.uuid4().hex
        now = _now()
        try:
            with self.engine.begin() as connection:
                strategy = connection.execute(
                    select(strategies).where(strategies.c.id == strategy_id).with_for_update()
                ).first()
                if strategy is None:
                    raise KeyError(strategy_id)
                family_type = connection.scalar(
                    select(strategy_versions.c.strategy_type)
                    .where(strategy_versions.c.strategy_id == strategy_id)
                    .limit(1)
                )
                if family_type != "pair":
                    raise ValueError("multifactor strategy families require promoted factors")
                latest = connection.scalar(
                    select(func.max(strategy_versions.c.version)).where(
                        strategy_versions.c.strategy_id == strategy_id
                    )
                )
                version_number = int(latest or 0) + 1
                connection.execute(
                    insert(strategy_versions).values(
                        id=version_id,
                        strategy_id=strategy_id,
                        version=version_number,
                        status="draft",
                        strategy_type="pair",
                        **_version_contract_columns(
                            definition["config"], strategy_type="pair"
                        ),
                        benchmark="CASH",
                        universe=f"pair:{definition['leg_y']}:{definition['leg_x']}",
                        config_json=definition["config"],
                        created_by=actor.strip(),
                        created_at=now,
                    )
                )
                connection.execute(
                    insert(strategy_pairs).values(
                        strategy_version_id=version_id,
                        leg_y=definition["leg_y"],
                        leg_x=definition["leg_x"],
                        asset_class=definition["asset_class"],
                        shorting_mode=definition["shorting_mode"],
                        created_at=now,
                    )
                )
                connection.execute(
                    update(strategies).where(strategies.c.id == strategy_id).values(updated_at=now)
                )
                self._event(
                    connection,
                    strategy_id=strategy_id,
                    version_id=version_id,
                    event_type="strategy.pair_version_created",
                    actor=actor.strip(),
                    payload={
                        "version": version_number,
                        **{key: definition[key] for key in ("leg_y", "leg_x", "asset_class")},
                    },
                )
        except IntegrityError as exc:
            raise ValueError(
                "pair strategy version creation conflicted with another request"
            ) from exc
        return self.get_version(version_id)

    def get(self, strategy_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(strategies).where(strategies.c.id == strategy_id)
            ).first()
            if row is None:
                raise KeyError(strategy_id)
            result = row_dict(row)
            versions = connection.execute(
                select(strategy_versions)
                .where(strategy_versions.c.strategy_id == strategy_id)
                .order_by(strategy_versions.c.version.desc())
            ).all()
        result["versions"] = [self.get_version(str(item.id)) for item in versions]
        return result

    def get_by_name(self, name: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            strategy_id = connection.scalar(
                select(strategies.c.id).where(strategies.c.name == name)
            )
        return self.get(str(strategy_id)) if strategy_id else None

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        statement = select(strategies).order_by(strategies.c.updated_at.desc()).limit(limit)
        with self.engine.connect() as connection:
            ids = [str(row.id) for row in connection.execute(statement)]
        return [self.get(strategy_id) for strategy_id in ids]

    def list_pairs(self, limit: int = 100) -> list[dict[str, Any]]:
        statement = (
            select(strategies.c.id)
            .join(
                strategy_versions,
                strategy_versions.c.strategy_id == strategies.c.id,
            )
            .join(
                strategy_pairs,
                strategy_pairs.c.strategy_version_id == strategy_versions.c.id,
            )
            .group_by(strategies.c.id)
            .order_by(strategies.c.updated_at.desc())
            .limit(limit)
        )
        with self.engine.connect() as connection:
            ids = [str(row.id) for row in connection.execute(statement)]
        return [self.get(strategy_id) for strategy_id in ids]

    def get_version(self, version_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    strategy_versions,
                    strategies.c.economic_hypothesis_group,
                    strategies.c.hypothesis_group_cap,
                )
                .join(strategies, strategies.c.id == strategy_versions.c.strategy_id)
                .where(strategy_versions.c.id == version_id)
            ).first()
            if row is None:
                raise KeyError(version_id)
            factor_rows = connection.execute(
                select(
                    strategy_factors,
                    factor_candidates.c.name,
                    factor_candidates.c.code_path,
                    factor_candidates.c.values_path,
                    factor_candidates.c.code_sha256,
                    factor_candidates.c.experiment_family_id,
                    factor_candidates.c.label_horizon_days,
                    factor_candidates.c.experiment_count,
                )
                .join(
                    factor_candidates,
                    factor_candidates.c.id == strategy_factors.c.factor_candidate_id,
                )
                .where(strategy_factors.c.strategy_version_id == version_id)
            ).all()
            pair_row = connection.execute(
                select(strategy_pairs).where(strategy_pairs.c.strategy_version_id == version_id)
            ).first()
        result = row_dict(row)
        result["config"] = result.pop("config_json")
        result["factors"] = [row_dict(item) for item in factor_rows]
        result["pair"] = row_dict(pair_row) if pair_row else None
        result["factor_source_mode"] = result["config"].get(
            "factor_source_mode", "promoted_only"
        )
        result["baseline_definition_sha256"] = result["config"].get(
            "baseline_definition_sha256"
        )
        return result

    def hypothesis_group_evidence(self, version_id: str) -> dict[str, Any]:
        """Return the immutable family-wide trial count used by formal gates.

        Factor experiment families carry their declared count (including
        non-winning variants); multiple versions/model wrappers also count as
        trials.  Renaming or versioning therefore cannot reset DSR/PBO inputs.
        """

        version = self.get_version(version_id)
        group = str(version["economic_hypothesis_group"])
        with self.engine.connect() as connection:
            version_rows = connection.execute(
                select(strategy_versions.c.id)
                .join(strategies, strategies.c.id == strategy_versions.c.strategy_id)
                .where(
                    strategies.c.economic_hypothesis_group == group,
                    strategy_versions.c.is_legacy.is_(False),
                )
            ).all()
            factor_rows = connection.execute(
                select(
                    factor_candidates.c.experiment_family_id,
                    factor_candidates.c.id,
                    factor_candidates.c.experiment_count,
                )
                .join(
                    strategy_factors,
                    strategy_factors.c.factor_candidate_id == factor_candidates.c.id,
                )
                .join(
                    strategy_versions,
                    strategy_versions.c.id == strategy_factors.c.strategy_version_id,
                )
                .join(strategies, strategies.c.id == strategy_versions.c.strategy_id)
                .where(
                    strategies.c.economic_hypothesis_group == group,
                    strategy_versions.c.is_legacy.is_(False),
                )
            ).all()
        family_counts: dict[str, int] = {}
        for row in factor_rows:
            family = str(row.experiment_family_id or row.id)
            family_counts[family] = max(
                family_counts.get(family, 0),
                int(row.experiment_count or 1),
            )
        version_ids = sorted(str(row.id) for row in version_rows)
        shared_count = max(1, len(version_ids), sum(family_counts.values()))
        return {
            "economic_hypothesis_group": group,
            "hypothesis_group_cap": float(version["hypothesis_group_cap"]),
            "shared_experiment_count": shared_count,
            "strategy_version_ids": version_ids,
            "experiment_family_counts": dict(sorted(family_counts.items())),
        }

    def create_backtest(
        self,
        *,
        version_id: str,
        dataset: str,
        periods: dict[str, str],
        artifact_path: Path,
        execution_dataset: str | None = None,
    ) -> dict[str, Any]:
        version = self.get_version(version_id)
        backtest_id = uuid.uuid4().hex
        artifact_directory = (
            artifact_path / backtest_id
            if artifact_path.name == "backtests"
            else artifact_path
        )
        with self.engine.begin() as connection:
            if version.get("strategy_type") == "multifactor":
                prior = connection.execute(
                    select(backtest_runs.c.id).where(
                        backtest_runs.c.strategy_version_id == version_id
                    )
                ).first()
                if prior is not None:
                    raise ValueError("a frozen strategy version may run the final test only once")
                baseline_only = (
                    version["config"].get("factor_source_mode")
                    == FACTOR_SOURCE_QLIB_BASELINE
                    and not version["factors"]
                )
                factor_windows = connection.execute(
                    select(
                        factor_evaluations.c.id,
                        factor_evaluations.c.factor_candidate_id,
                        factor_evaluations.c.dataset,
                        factor_evaluations.c.dataset_identity_sha256,
                        factor_evaluations.c.test_start,
                        factor_evaluations.c.test_end,
                        factor_evaluations.c.evaluator_version,
                        factor_evaluations.c.final_test_consumed_at,
                    )
                    .join(
                        strategy_factors,
                        strategy_factors.c.factor_evaluation_id == factor_evaluations.c.id,
                    )
                    .where(strategy_factors.c.strategy_version_id == version_id)
                ).all()
                requested_start = date.fromisoformat(periods["start"])
                requested_end = date.fromisoformat(periods["end"])
                if not baseline_only and (not factor_windows or any(
                    item.dataset != dataset
                    or str(item.evaluator_version) != "factor-gate-v3-hac-bh"
                    or requested_start != item.test_start
                    or requested_end != item.test_end
                    for item in factor_windows
                )):
                    raise ValueError(
                        "formal backtest must exactly match the reserved final-test window"
                    )
                if any(item.final_test_consumed_at is not None for item in factor_windows):
                    raise ValueError("reserved final test has already been consumed")
                consumed_at = _now()
                if factor_windows:
                    # Cross-campaign seal (design draft 4.1/12.1): the reserved
                    # final OOS window is a one-time vintage keyed by research
                    # scope + dataset identity + calendar window. New evaluation
                    # rows from renamed or new campaigns cannot re-open it.
                    self._seal_and_consume_oos_vintage(
                        connection,
                        candidate_ids=sorted(
                            {str(item.factor_candidate_id) for item in factor_windows}
                        ),
                        dataset_identities={
                            str(item.dataset_identity_sha256 or "") for item in factor_windows
                        },
                        dataset=dataset,
                        test_start=requested_start,
                        test_end=requested_end,
                        consumed_at=consumed_at,
                    )
                for item in factor_windows:
                    key = hashlib.sha256(
                        f"{version_id}:{item.id}:{dataset}:{periods['start']}:{periods['end']}".encode()
                    ).hexdigest()
                    connection.execute(
                        update(factor_evaluations)
                        .where(
                            factor_evaluations.c.id == item.id,
                            factor_evaluations.c.final_test_consumed_at.is_(None),
                        )
                        .values(final_test_key=key, final_test_consumed_at=consumed_at)
                    )
            connection.execute(
                insert(backtest_runs).values(
                    id=backtest_id,
                    strategy_version_id=version_id,
                    dataset=dataset,
                    execution_dataset=execution_dataset,
                    signal_frequency=version["signal_frequency"],
                    execution_frequency=version["execution_frequency"],
                    execution_contract_hash=version["execution_contract_hash"],
                    qlib_version=version["qlib_version"],
                    qlib_commit=version["qlib_commit"],
                    rdagent_version=version["rdagent_version"],
                    rdagent_commit=version["rdagent_commit"],
                    status="queued",
                    periods_json=periods,
                    artifact_path=str(artifact_directory),
                    created_at=_now(),
                )
            )
        return self.get_backtest(backtest_id)

    @staticmethod
    def _seal_and_consume_oos_vintage(
        connection: Any,
        *,
        candidate_ids: list[str],
        dataset_identities: set[str],
        dataset: str,
        test_start: date,
        test_end: date,
        consumed_at: datetime,
    ) -> str:
        """Seal and consume the OOS vintage for a reserved final-test window.

        The vintage key is (scope, dataset identity, calendar window). Scope is
        the immutable research program id when the candidate lineage belongs to
        exactly one program, otherwise the dataset identity itself; it never
        contains campaign/hypothesis-family/strategy names, so renaming or
        recreating those cannot mint a fresh vintage for the same window.
        """

        dataset_identity = (
            next(iter(dataset_identities))
            if len(dataset_identities) == 1 and dataset_identities != {""}
            else f"name:{dataset}"
        )
        program_ids = {
            str(row.research_program_id)
            for row in connection.execute(
                select(research_campaigns.c.research_program_id).where(
                    research_campaigns.c.research_run_id.in_(
                        select(factor_candidates.c.research_run_id).where(
                            factor_candidates.c.id.in_(candidate_ids)
                        )
                    ),
                    research_campaigns.c.research_program_id.is_not(None),
                )
            )
        }
        scope = (
            f"program:{next(iter(program_ids))}"
            if len(program_ids) == 1
            else f"dataset:{dataset_identity}"
        )
        row = connection.execute(
            select(oos_vintages)
            .where(
                oos_vintages.c.scope == scope,
                oos_vintages.c.dataset_identity == dataset_identity,
                oos_vintages.c.test_start == test_start,
                oos_vintages.c.test_end == test_end,
            )
            .with_for_update()
        ).first()
        if row is not None:
            sealed = set((row.sealed_candidate_set_json or {}).get("candidate_ids") or [])
            if not set(candidate_ids) <= sealed:
                raise ValueError(
                    "final test window is sealed and this candidate is not in the "
                    "sealed candidate set"
                )
            if row.consumed_at is not None:
                raise ValueError("reserved final test has already been consumed")
            connection.execute(
                update(oos_vintages)
                .where(oos_vintages.c.id == row.id)
                .values(consumed_at=consumed_at)
            )
            return str(row.id)
        sealed_set = {"candidate_ids": candidate_ids}
        vintage_id = uuid.uuid4().hex
        connection.execute(
            insert(oos_vintages).values(
                id=vintage_id,
                scope=scope,
                dataset_identity=dataset_identity,
                test_start=test_start,
                test_end=test_end,
                sealed_at=consumed_at,
                first_opened_at=consumed_at,
                consumed_at=consumed_at,
                sealed_candidate_set_json=sealed_set,
                sealed_candidate_set_sha256=_canonical_sha256(sealed_set),
                created_at=consumed_at,
            )
        )
        return vintage_id

    def attach_job(self, backtest_id: str, job_id: str) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(backtest_runs).where(backtest_runs.c.id == backtest_id).values(job_id=job_id)
            )
            if not result.rowcount:
                raise KeyError(backtest_id)

    def mark_backtest(
        self,
        backtest_id: str,
        status: str,
        *,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        now = _now()
        values: dict[str, Any] = {"status": status, "error": error}
        if metrics is not None:
            values["metrics_json"] = metrics
        if status == "running":
            values["started_at"] = now
        if status in {"succeeded", "failed", "cancelled"}:
            values["finished_at"] = now
        with self.engine.begin() as connection:
            result = connection.execute(
                update(backtest_runs).where(backtest_runs.c.id == backtest_id).values(**values)
            )
            if not result.rowcount:
                raise KeyError(backtest_id)

    def requeue_backtest(self, backtest_id: str) -> None:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(backtest_runs.c.status, strategy_versions.c.strategy_type)
                .join(
                    strategy_versions,
                    strategy_versions.c.id == backtest_runs.c.strategy_version_id,
                )
                .where(backtest_runs.c.id == backtest_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(backtest_id)
            if row.status not in {"failed", "cancelled"}:
                raise ValueError("only failed or cancelled backtests may be requeued")
            if row.strategy_type == "multifactor":
                raise ValueError("a formal final test cannot be rerun")
            connection.execute(
                update(backtest_runs)
                .where(backtest_runs.c.id == backtest_id)
                .values(
                    status="queued",
                    metrics_json=None,
                    error=None,
                    started_at=None,
                    finished_at=None,
                )
            )

    def validate_backtest_artifacts(self, backtest_id: str, metrics: dict[str, Any]) -> None:
        """Validate immutable strategy artifacts before a worker reports success."""

        backtest = self.get_backtest(backtest_id)
        version = self.get_version(backtest["strategy_version_id"])
        failures = (
            _pair_artifact_failures(version, backtest, metrics)
            if version.get("strategy_type") == "pair"
            else _multifactor_manifest_failures(version, backtest, metrics)
        )
        if failures:
            raise ValueError("strategy backtest artifact validation failed: " + "; ".join(failures))

    def get_backtest(self, backtest_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(backtest_runs).where(backtest_runs.c.id == backtest_id)
            ).first()
        if row is None:
            raise KeyError(backtest_id)
        result = row_dict(row)
        result["periods"] = result.pop("periods_json")
        result["metrics"] = result.pop("metrics_json")
        return result

    def list_backtests(
        self, version_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        statement = select(backtest_runs)
        if version_id:
            statement = statement.where(backtest_runs.c.strategy_version_id == version_id)
        statement = statement.order_by(backtest_runs.c.created_at.desc()).limit(limit)
        with self.engine.connect() as connection:
            rows = [row_dict(row) for row in connection.execute(statement)]
        for row in rows:
            row["periods"] = row.pop("periods_json")
            row["metrics"] = row.pop("metrics_json")
        return rows

    def _approve_pair(
        self,
        version: dict[str, Any],
        backtest: dict[str, Any],
        *,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        # Research-only gate (design 6.4.3/13): pair strategies are offline
        # statistical research and can never be approved for capital use.
        # The verdict comes from the strategy catalog, the single source of truth.
        require_capital_eligible_strategy_type("pair", action="批准")
        config = PairTradingConfig(**version["config"])
        metrics = dict(backtest["metrics"] or {})
        failures: list[str] = []
        if actor == version["created_by"]:
            failures.append("pair strategy approval requires a second operator")
        if not backtest.get("execution_dataset"):
            failures.append("pair backtest requires an immutable minute execution dataset")
        if (
            metrics.get("backtest_engine") != "quantlab_pair"
            or metrics.get("pair_native_backtest") is not True
        ):
            failures.append("a native QuantLab pair backtest is required")
        pair = version.get("pair") or {}
        if metrics.get("leg_y") != pair.get("leg_y") or metrics.get("leg_x") != pair.get("leg_x"):
            failures.append("pair backtest instruments do not match the strategy version")
        evidence = metrics.get("initial_pair_evidence")
        if not isinstance(evidence, dict):
            failures.append("initial pair correlation and cointegration evidence is required")
            evidence = {}
        checks: dict[str, tuple[Any, Any, str]] = {
            "correlation": (evidence.get("correlation"), config.min_correlation, "min"),
            "cointegration_pvalue": (
                evidence.get("cointegration_pvalue"),
                config.max_cointegration_pvalue,
                "max",
            ),
            "hedge_ratio_min": (
                evidence.get("hedge_ratio"),
                config.min_hedge_ratio,
                "min",
            ),
            "hedge_ratio_max": (
                evidence.get("hedge_ratio"),
                config.max_hedge_ratio,
                "max",
            ),
            "max_drawdown": (
                abs(float(metrics["max_drawdown"]))
                if metrics.get("max_drawdown") is not None
                else None,
                config.max_drawdown,
                "max",
            ),
            "sharpe_ratio": (metrics.get("sharpe_ratio"), config.min_sharpe_ratio, "min"),
            "closed_trade_count": (
                metrics.get("closed_trade_count"),
                config.min_closed_trades,
                "min",
            ),
            "trading_days": (metrics.get("trading_days"), config.min_backtest_days, "min"),
            "rolling_cointegration_pass_rate": (
                metrics.get("rolling_cointegration_pass_rate"),
                config.min_rolling_cointegration_pass_rate,
                "min",
            ),
            "pair_robustness_pass_rate": (
                metrics.get("pair_robustness_pass_rate"),
                config.min_robustness_pass_rate,
                "min",
            ),
            "capacity_fill_ratio": (
                metrics.get("capacity_fill_ratio"),
                config.min_capacity_fill_ratio,
                "min",
            ),
        }
        for name, (value, threshold, mode) in checks.items():
            if (
                value is None
                or (mode == "max" and value > threshold)
                or (mode == "min" and value < threshold)
            ):
                failures.append(f"{name}={value} violates {mode} {threshold}")
        for name in (
            "minute_execution_enforced",
            "shortability_enforced",
            "market_controls_enforced",
            "atomic_pair_execution_enforced",
            "transaction_costs_enforced",
            "borrow_cost_enforced",
        ):
            if metrics.get(name) is not True:
                failures.append(f"{name} is required for pair strategy approval")
        if metrics.get("open_position_at_end") is not False:
            failures.append("pair backtest must finish without an open spread position")
        if metrics.get("cost_schedule_version") not in KNOWN_COST_SCHEDULE_VERSIONS:
            failures.append("pair backtest cost schedule is missing or obsolete")
        provenance = metrics.get("provenance")
        if not isinstance(provenance, dict):
            failures.append("reproducible pair backtest provenance is required")
        else:
            try:
                require_qlib_workflow_identity(provenance.get("qlib_workflow"))
            except ValueError as exc:
                failures.append(str(exc))
            for field in (
                "daily_dataset_identity_sha256",
                "daily_snapshot_manifest_sha256",
                "minute_snapshot_manifest_sha256",
                "strategy_config_sha256",
                "execution_manifest_sha256",
                "pair_engine_sha256",
                "shortability_evidence_sha256",
            ):
                if not _is_sha256(provenance.get(field)):
                    failures.append(f"provenance {field} must be a SHA-256 digest")
        if failures:
            raise ValueError("pair strategy risk gate failed: " + "; ".join(failures))
        now = _now()
        with self.engine.begin() as connection:
            connection.execute(
                update(strategy_versions)
                .where(
                    strategy_versions.c.strategy_id == version["strategy_id"],
                    strategy_versions.c.status == "approved",
                )
                .values(status="retired")
            )
            connection.execute(
                update(strategy_versions)
                .where(strategy_versions.c.id == version["id"])
                .values(
                    status="approved",
                    approved_by=actor,
                    approval_reason=reason,
                    approved_at=now,
                )
            )
            connection.execute(
                update(strategies)
                .where(strategies.c.id == version["strategy_id"])
                .values(status="approved", updated_at=now)
            )
            self._event(
                connection,
                strategy_id=version["strategy_id"],
                version_id=version["id"],
                event_type="strategy.pair_approved",
                actor=actor,
                payload={
                    "reason": reason,
                    "backtest_id": backtest["id"],
                    "gate_evidence": {name: value[0] for name, value in checks.items()},
                },
            )
        return self.get_version(version["id"])

    def approve(self, version_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        if not actor.strip() or len(reason.strip()) < 10:
            raise ValueError("actor and a meaningful approval reason are required")
        version = self.get_version(version_id)
        if version.get("is_legacy"):
            raise ValueError("legacy strategy versions must be recreated through evaluation v2")
        backtests = self.list_backtests(version_id=version_id, limit=1)
        if not backtests or backtests[0]["status"] != "succeeded" or not backtests[0]["metrics"]:
            raise ValueError("strategy version requires a successful backtest before approval")
        metrics = backtests[0]["metrics"]
        if backtests[0].get("is_legacy"):
            raise ValueError("legacy backtests cannot approve a new strategy")
        config = version["config"]
        if version.get("strategy_type") == "pair":
            return self._approve_pair(
                version,
                backtests[0],
                actor=actor,
                reason=reason,
            )
        drawdown = metrics.get("max_drawdown")
        checks = {
            "tracking_error": (
                metrics.get("tracking_error"),
                config["max_tracking_error"],
                "max",
            ),
            "max_drawdown": (
                abs(float(drawdown)) if drawdown is not None else None,
                config["max_drawdown"],
                "max",
            ),
            "average_turnover": (
                metrics.get("average_turnover"),
                config["max_turnover"],
                "max",
            ),
            "information_ratio": (
                metrics.get("information_ratio"),
                config["min_information_ratio"],
                "min",
            ),
            "sharpe_ratio": (
                metrics.get("sharpe_ratio"),
                config.get("min_sharpe_ratio", 0.0),
                "min",
            ),
            "sortino_ratio": (
                metrics.get("sortino_ratio"),
                config.get("min_sortino_ratio", 0.0),
                "min",
            ),
            "deflated_sharpe_probability": (
                metrics.get("deflated_sharpe_probability"),
                0.95,
                "min",
            ),
            "robustness_pass_rate": (metrics.get("robustness_pass_rate"), 1.0, "min"),
            "rolling_pass_rate": (
                metrics.get("rolling_pass_rate"),
                config.get("min_rolling_pass_rate", 0.60),
                "min",
            ),
            "rolling_window_count": (
                metrics.get("rolling_window_count"),
                config.get("min_rolling_windows", 3),
                "min",
            ),
            "event_stress_count": (
                metrics.get("event_stress_count"),
                config.get("event_count", 5),
                "min",
            ),
            "event_stress_pass_rate": (
                metrics.get("event_stress_pass_rate"),
                config.get("min_event_stress_pass_rate", 0.60),
                "min",
            ),
            "trading_days": (
                metrics.get("trading_days"),
                config.get("min_backtest_days", 504),
                "min",
            ),
            "closed_trade_count": (
                metrics.get("closed_trade_count"),
                config.get("min_closed_trades", 20),
                "min",
            ),
            "win_rate": (
                metrics.get("win_rate"),
                config.get("min_win_rate", 0.0),
                "min",
            ),
            "profit_loss_ratio": (
                metrics.get("profit_loss_ratio"),
                config.get("min_profit_loss_ratio", 0.0),
                "min",
            ),
            "capacity_curve_points": (
                metrics.get("capacity_curve_points"),
                3,
                "min",
            ),
        }
        execution_method = str(config.get("execution_method", "open"))
        if execution_method in {"twap", "vwap", "next_bar"}:
            checks["capacity_fill_ratio"] = (
                metrics.get("capacity_fill_ratio"),
                config.get("min_capacity_fill_ratio", 0.95),
                "min",
            )
        failures = []
        try:
            require_strategy_execution_contract(config)
        except ValueError as exc:
            failures.append(str(exc))
        if (
            metrics.get("backtest_engine") != "qlib"
            or metrics.get("qlib_native_backtest") is not True
        ):
            failures.append("a Qlib-native backtest is required for approval")
        provenance = metrics.get("provenance")
        expected_factors = {str(item["factor_candidate_id"]) for item in version["factors"]}
        if not isinstance(provenance, dict):
            failures.append("reproducible backtest provenance is required for approval")
        else:
            try:
                require_qlib_workflow_identity(provenance.get("qlib_workflow"))
            except ValueError as exc:
                failures.append(str(exc))
            for field in (
                "dataset_identity_sha256",
                "snapshot_manifest_sha256",
                "qlib_builder_sha256",
                "strategy_config_sha256",
                "execution_manifest_sha256",
            ):
                if not _is_sha256(provenance.get(field)):
                    failures.append(f"provenance {field} must be a SHA-256 digest")
            for field in ("factor_values_sha256", "factor_code_sha256"):
                hashes = provenance.get(field)
                if not isinstance(hashes, dict) or set(hashes) != expected_factors:
                    failures.append(f"provenance {field} does not match strategy factors")
                elif not all(_is_sha256(value) for value in hashes.values()):
                    failures.append(f"provenance {field} contains an invalid SHA-256 digest")
            if execution_method in {"twap", "vwap", "next_bar"}:
                for field in (
                    "execution_dataset_identity_sha256",
                    "execution_snapshot_manifest_sha256",
                    "execution_qlib_builder_sha256",
                ):
                    if not _is_sha256(provenance.get(field)):
                        failures.append(f"provenance {field} must be a SHA-256 digest")
            if (
                not str(provenance.get("qlib_version") or "").strip()
                or provenance.get("qlib_version") == "unknown"
            ):
                failures.append("provenance qlib_version is required")
            qlib_commit = str(provenance.get("qlib_commit") or "")
            if len(qlib_commit) != 40 or any(
                character not in "0123456789abcdef" for character in qlib_commit.lower()
            ):
                failures.append("provenance qlib_commit must identify the pinned upstream")
            try:
                require_daily_qlib_contract(provenance)
            except ValueError as exc:
                failures.append(str(exc))
            if provenance.get("backtest_engine_version") != QLIB_ENGINE_VERSION:
                failures.append("backtest engine version is obsolete or inconsistent")
            if not str(provenance.get("policy_version") or "").strip():
                failures.append("PortfolioPolicy provenance is required")
        if not isinstance(provenance, dict) or metrics.get("policy_version") != provenance.get(
            "policy_version"
        ):
            failures.append("PortfolioPolicy version is missing or inconsistent")
        execution_model_evidence = metrics.get("execution_model")
        if (
            not isinstance(execution_model_evidence, dict)
            or execution_model_evidence.get("strategy_contract_hash")
            != config.get("execution_contract_hash")
        ):
            failures.append("strategy execution contract evidence is missing or inconsistent")
        if metrics.get("event_stress_passed") is not True:
            failures.append("event stress scenarios did not satisfy the configured result gate")
        if (metrics.get("event_stress") or {}).get("state_source") != (
            "full_backtest_carried_positions"
        ):
            failures.append("event stress did not inherit the formal backtest state")
        event_stress = metrics.get("event_stress") or {}
        event_items = event_stress.get("events")
        if (
            event_stress.get("position_state_method") != "formal_fill_ledger_v1"
            or not isinstance(event_items, list)
            or len(event_items) < int(config.get("event_count", 5))
            or any(
                not isinstance(item, dict)
                or item.get("state_source") != "full_backtest_carried_positions"
                or item.get("return_state_source") != "full_backtest_report_slice"
                or not isinstance(item.get("start_holdings"), dict)
                or not isinstance(item.get("state_fill_count"), int)
                for item in event_items
            )
        ):
            failures.append("event stress carried-position evidence is incomplete")
        robustness = metrics.get("robustness")
        artifact_root = Path(backtests[0]["artifact_path"]).resolve()
        if (
            not isinstance(robustness, dict)
            or robustness.get("passed") is not True
            or robustness.get("pass_rate") != 1.0
            or set(robustness.get("scenarios") or {})
            != {"double_cost", "turnover_75pct", "topk_80pct", "zero_retention_buffer"}
        ):
            failures.append("all four independent robustness scenarios are required")
        else:
            failures.extend(
                _scenario_artifact_failures(robustness["scenarios"], artifact_root)
            )
        component_stress = metrics.get("component_cost_stress")
        if (
            not isinstance(component_stress, dict)
            or component_stress.get("passed") is not True
            or component_stress.get("pass_rate") != 1.0
            or set(component_stress.get("scenarios") or {})
            != set(COMPONENT_COST_STRESS_MULTIPLIERS)
        ):
            failures.append(
                "all component cost stress scenarios are required "
                "(commission/slippage/impact/fill-rate)"
            )
        else:
            failures.extend(
                _scenario_artifact_failures(component_stress["scenarios"], artifact_root)
            )
        if metrics.get("sortino_status") != "ok":
            failures.append("Sortino is undefined or non-finite")
        deflated = metrics.get("deflated_sharpe")
        if (
            not isinstance(deflated, dict)
            or deflated.get("status") != "ok"
            or deflated.get("method_version") != DEFLATED_SHARPE_METHOD_VERSION
        ):
            failures.append("Deflated Sharpe evidence is missing or invalid")
        failures.extend(_formal_validation_failures(version, metrics))
        if metrics.get("capacity_curve_passed") is not True:
            failures.append("capacity curve did not satisfy the configured result gate")
        eligibility = metrics.get("eligibility")
        if (
            not isinstance(eligibility, dict)
            or eligibility.get("contract_version") != ELIGIBILITY_CONTRACT_VERSION
            or int(eligibility.get("rows") or 0) <= 0
            or int(eligibility.get("eligible_rows") or 0) <= 0
        ):
            failures.append("point-in-time eligibility evidence is missing or empty")
        elif config.get("require_regulatory_events") and not eligibility.get(
            "regulatory_data_available"
        ):
            failures.append("required regulatory violation data is unavailable")
        if execution_method in {"twap", "vwap", "next_bar"}:
            execution_model = metrics.get("execution_model")
            if not backtests[0].get("execution_dataset"):
                failures.append("minute execution dataset is required for approval")
            if (
                not isinstance(execution_model, dict)
                or execution_model.get("method") != execution_method
                or execution_model.get("frequency") in {None, "day"}
                or execution_model.get("price_assumption")
                not in {"minute bar vwap fills", "next eligible minute bar vwap"}
                or execution_model.get("strategy_contract_hash")
                != config.get("execution_contract_hash")
                or metrics.get("minute_execution_enforced") is not True
            ):
                failures.append("minute-native execution evidence is required")
            try:
                require_minute_execution_contract(
                    {
                        "frequency": (execution_model or {}).get("frequency"),
                        "execution_contract_version": provenance.get(
                            "execution_contract_version"
                        )
                        if isinstance(provenance, dict)
                        else None,
                        "lineage_verified": provenance.get("execution_lineage_verified")
                        if isinstance(provenance, dict)
                        else None,
                        "fields": provenance.get("execution_fields")
                        if isinstance(provenance, dict)
                        else None,
                        "source_datasets": provenance.get("execution_source_datasets")
                        if isinstance(provenance, dict)
                        else None,
                        "source_unit_contracts": provenance.get(
                            "execution_source_unit_contracts"
                        )
                        if isinstance(provenance, dict)
                        else None,
                    },
                    frequency=(execution_model or {}).get("frequency"),
                )
            except ValueError as exc:
                failures.append(str(exc))
            if isinstance(provenance, dict) and provenance.get(
                "source_lineage_id"
            ) != provenance.get("execution_source_lineage_id"):
                failures.append("daily and minute backtest datasets do not share source lineage")
        cost_model = metrics.get("cost_model")
        if not isinstance(cost_model, dict):
            failures.append("the unified cost model is required for approval")
        else:
            try:
                effective_costs = CostModelConfig.from_mapping(cost_model)
                backtest_start_date = date.fromisoformat(backtests[0]["periods"]["start"])
                backtest_end_date = date.fromisoformat(backtests[0]["periods"]["end"])
                if date.fromisoformat(effective_costs.effective_from) > backtest_start_date or (
                    effective_costs.effective_to is not None
                    and date.fromisoformat(effective_costs.effective_to) < backtest_end_date
                ):
                    failures.append("cost schedule does not cover the full backtest period")
            except (TypeError, ValueError) as exc:
                failures.append(f"cost schedule is invalid: {exc}")
        failures.extend(_multifactor_manifest_failures(version, backtests[0], metrics))
        if isinstance(cost_model, dict) and float(cost_model.get("min_commission", -1.0)) < float(
            config.get("min_commission", 5.0)
        ):
            failures.append("minimum commission evidence is below the configured value")
        for name, (value, threshold, mode) in checks.items():
            if (
                value is None
                or (mode == "max" and value > threshold)
                or (mode == "min" and value < threshold)
            ):
                failures.append(f"{name}={value} violates {mode} {threshold}")
        backtest_start = date.fromisoformat(backtests[0]["periods"]["start"])
        backtest_end = date.fromisoformat(backtests[0]["periods"]["end"])
        with self.engine.connect() as connection:
            for factor in version["factors"]:
                evaluation = connection.execute(
                    select(
                        factor_evaluations.c.test_start,
                        factor_evaluations.c.test_end,
                        factor_evaluations.c.evaluator_version,
                        factor_evaluations.c.is_legacy,
                        factor_evaluations.c.dataset_identity_sha256,
                    ).where(
                        factor_evaluations.c.id == factor["factor_evaluation_id"],
                        factor_evaluations.c.factor_candidate_id == factor["factor_candidate_id"],
                    )
                ).first()
                if evaluation is None:
                    failures.append(
                        f"factor {factor['factor_candidate_id']} has no out-of-sample evidence"
                    )
                elif evaluation.is_legacy or str(evaluation.evaluator_version) != (
                    "factor-gate-v3-hac-bh"
                ):
                    failures.append(
                        f"factor {factor['factor_candidate_id']} uses a legacy evaluation"
                    )
                elif evaluation.dataset_identity_sha256 != provenance.get(
                    "dataset_identity_sha256"
                ):
                    failures.append(
                        f"factor {factor['factor_candidate_id']} dataset identity does not match"
                    )
                elif backtest_start < evaluation.test_start or backtest_end > evaluation.test_end:
                    failures.append(
                        f"backtest {backtest_start}..{backtest_end} falls outside factor "
                        f"test window {evaluation.test_start}..{evaluation.test_end}"
                    )
        if failures:
            raise ValueError("strategy risk gate failed: " + "; ".join(failures))
        now = _now()
        with self.engine.begin() as connection:
            connection.execute(
                update(strategy_versions)
                .where(
                    strategy_versions.c.strategy_id == version["strategy_id"],
                    strategy_versions.c.status == "approved",
                )
                .values(status="retired")
            )
            connection.execute(
                update(strategy_versions)
                .where(strategy_versions.c.id == version_id)
                .values(
                    status="approved",
                    # Design 6.11: passing the formal hard gate automatically
                    # moves the version to the isolated paper stage; forward
                    # evidence starts accumulating from zero.
                    promotion_stage="paper",
                    approved_by=actor,
                    approval_reason=reason,
                    approved_at=now,
                )
            )
            connection.execute(
                update(strategies)
                .where(strategies.c.id == version["strategy_id"])
                .values(status="approved", updated_at=now)
            )
            self._event(
                connection,
                strategy_id=version["strategy_id"],
                version_id=version_id,
                event_type="strategy.approved",
                actor=actor,
                payload={
                    "reason": reason,
                    "backtest_id": backtests[0]["id"],
                    "gate_evidence": {name: value[0] for name, value in checks.items()},
                },
            )
        # Design 6.11/7.4: candidate -> paper is automatic once the formal
        # hard gate passes. The isolated paper stage opens after the approval
        # commit; a stage-opening failure never rolls back a passed gate and
        # is traceable through strategy events (retry via open_paper_stage).
        from .promotion import PromotionStore

        promotion = PromotionStore(self.database_url)
        try:
            promotion.open_paper_stage(version_id, actor=actor)
        except Exception as exc:  # noqa: BLE001 - approval is already committed
            promotion.record_paper_stage_failure(version_id, actor=actor, error=str(exc))
        return self.get_version(version_id)

    @staticmethod
    def _event(
        connection: Any,
        *,
        strategy_id: str,
        version_id: str | None,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
    ) -> None:
        connection.execute(
            insert(strategy_events).values(
                strategy_id=strategy_id,
                strategy_version_id=version_id,
                event_type=event_type,
                actor=actor,
                payload_json=payload,
                created_at=_now(),
            )
        )
