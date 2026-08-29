"""Resolve activity-health drift from sealed formal and materialized factors."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from quant_data.database import (
    backtest_runs,
    factor_candidates,
    factor_definitions,
    jobs,
    model_artifacts,
    model_candidates,
    open_database,
    strategy_factors,
    strategy_versions,
)

from .feature_drift import (
    FEATURE_DRIFT_FORMULA_VERSION,
    FEATURE_DRIFT_OBSERVATION_VERSION,
    build_strategy_health_feature_set,
    sha256_file,
    validate_factor_psi_observation,
)
from .feature_set_registry import get_feature_set
from .model_calibration_drift import (
    MODEL_CALIBRATION_FORMULA_VERSION,
    MODEL_CALIBRATION_OBSERVATION_VERSION,
    validate_model_calibration_observation,
)
from .research_horizon import canonical_sha256
from .strategy_artifact_manifest import validate_backtest_artifact_manifest
from .strategy_health_reference import (
    STRATEGY_HEALTH_REFERENCE_NAME,
    factor_drift_from_reference,
    factor_values_sha256,
    live_model_calibration_from_reference,
    validate_strategy_health_reference,
)


class StrategyFeatureDriftSource:
    """Build one deterministic observation for an active StrategyVersion."""

    def __init__(self, database_url: str, data_root: Path) -> None:
        self.engine = open_database(database_url)
        self.data_root = Path(data_root).resolve()

    def feature_set(self, version_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            version_row = connection.execute(
                select(strategy_versions).where(strategy_versions.c.id == version_id)
            ).mappings().first()
            if version_row is None:
                raise KeyError(version_id)
            candidate_rows = connection.execute(
                select(
                    strategy_factors.c.factor_candidate_id,
                    factor_candidates.c.factor_definition_id,
                    factor_definitions.c.expression,
                    factor_definitions.c.expression_sha256,
                    factor_definitions.c.status,
                )
                .join(
                    factor_candidates,
                    factor_candidates.c.id == strategy_factors.c.factor_candidate_id,
                )
                .outerjoin(
                    factor_definitions,
                    factor_definitions.c.id == factor_candidates.c.factor_definition_id,
                )
                .where(strategy_factors.c.strategy_version_id == version_id)
            ).mappings().all()
        version = dict(version_row)
        version["config"] = dict(version.pop("config_json") or {})
        if str(version["config"].get("signal_source") or "factor_score") == (
            "model_prediction"
        ):
            return self._model_feature_set(version)
        version["factors"] = [dict(item) for item in candidate_rows]
        expressions: dict[str, str] = {}
        for row in candidate_rows:
            candidate_id = str(row["factor_candidate_id"])
            if (
                not row.get("factor_definition_id")
                or str(row.get("status") or "") != "active"
                or not row.get("expression")
            ):
                raise ValueError(
                    f"strategy challenger {candidate_id} has no active governed Qlib definition"
                )
            expressions[candidate_id] = str(row["expression"])
        return build_strategy_health_feature_set(
            version,
            candidate_expressions=expressions,
        )

    def _model_feature_set(self, version: dict[str, Any]) -> dict[str, Any]:
        config = dict(version["config"])
        candidate_ids = [
            str(item)
            for item in (
                config.get("model_component_candidate_ids")
                or [config.get("model_candidate_id")]
            )
            if str(item or "")
        ]
        if not candidate_ids:
            raise ValueError("model strategy has no frozen model candidate")
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(model_candidates).where(model_candidates.c.id.in_(candidate_ids))
            ).mappings().all()
        by_id = {str(row["id"]): dict(row) for row in rows}
        if set(by_id) != set(candidate_ids):
            raise ValueError("model strategy candidate evidence is incomplete")
        features: dict[str, str] = {}
        feature_sources: list[dict[str, Any]] = []
        label_contracts: dict[str, dict[str, Any]] = {}
        for candidate_id in candidate_ids:
            row = by_id[candidate_id]
            base = dict(row.get("base_features_manifest_json") or {})
            if (
                canonical_sha256(base)
                != str(row.get("base_features_manifest_sha256") or "")
                or base.get("definition_sha256")
                != row.get("feature_set_definition_sha256")
                or not isinstance(base.get("feature_expressions"), dict)
                or sorted(base.get("feature_names") or [])
                != sorted(base["feature_expressions"])
            ):
                raise ValueError("model strategy frozen feature manifest is invalid")
            registered = get_feature_set(str(base.get("feature_set_id") or ""))
            if (
                registered["definition_sha256"] != base["definition_sha256"]
                or registered["features"] != base["feature_expressions"]
            ):
                raise ValueError("model strategy feature registry changed")
            for factor_id, expression in registered["features"].items():
                if factor_id in features and features[factor_id] != expression:
                    raise ValueError("ensemble model feature expressions conflict")
                features[factor_id] = expression
            feature_sources.append(
                {
                    "model_candidate_id": candidate_id,
                    "base_features_manifest_sha256": str(
                        row["base_features_manifest_sha256"]
                    ),
                    "feature_set_definition_sha256": str(
                        row["feature_set_definition_sha256"]
                    ),
                }
            )
            admission = dict(row.get("admission_evidence_json") or {})
            for profile in (admission.get("profiles") or {}).values():
                for cell in ((profile or {}).get("seeds") or {}).values():
                    contract = (cell or {}).get("model_label_contract")
                    digest = str(
                        (cell or {}).get("model_label_contract_sha256") or ""
                    )
                    if not isinstance(contract, dict) or canonical_sha256(contract) != digest:
                        raise ValueError("model strategy label contract is invalid")
                    label_contracts[digest] = dict(contract)
        if len(label_contracts) != 1:
            raise ValueError("model strategy candidates do not share one label contract")
        label_sha256, label_contract = next(iter(label_contracts.items()))
        if label_contract.get("horizon_profile") != version.get("horizon_profile"):
            raise ValueError("model strategy label belongs to another horizon")
        drift_factor_ids = sorted(features)
        factor_contract = {
            "contract_version": "strategy-health-model-factor-contract-v1",
            "strategy_version_id": str(version["id"]),
            "strategy_rules_sha256": _sha256(
                version.get("strategy_rules_sha256"), field="strategy rules"
            ),
            "feature_sources": feature_sources,
            "label_contract_sha256": label_sha256,
            "label_contract": label_contract,
            "features": features,
        }
        factor_set_sha256 = canonical_sha256(factor_contract)
        definition = {
            "contract_version": "strategy-health-model-feature-set-v1",
            "id": (
                f"strategy-health:{version['id']}:{factor_set_sha256[:16]}"
            ),
            "name": f"Activity-health inputs for model strategy {version['id']}",
            "features": features,
            "source": f"strategy-model-signal:{version['id']}:{factor_set_sha256}",
            "materialization_contract": {
                "contract_version": "strategy-health-recent-materialization-v1",
                "session_limit": 64,
                "storage_mode": "recent_only",
            },
        }
        return {
            **definition,
            "definition_sha256": canonical_sha256(definition),
            "factor_set_sha256": factor_set_sha256,
            "factor_contract": factor_contract,
            "signal_source": "model_prediction",
            "drift_factor_ids": drift_factor_ids,
            "model_label_contract": label_contract,
            "model_label_contract_sha256": label_sha256,
        }

    def observe(
        self,
        *,
        version_id: str,
        formal_backtest_id: str,
        current_dataset_identity_sha256: str,
        current_dataset_lineage_id: str,
    ) -> dict[str, Any]:
        feature_set = self.feature_set(version_id)
        with self.engine.connect() as connection:
            version = connection.execute(
                select(strategy_versions).where(strategy_versions.c.id == version_id)
            ).mappings().one()
            backtest = connection.execute(
                select(backtest_runs)
                .where(
                    backtest_runs.c.id == formal_backtest_id,
                    backtest_runs.c.strategy_version_id == version_id,
                    backtest_runs.c.status == "succeeded",
                    backtest_runs.c.is_legacy.is_(False),
                )
            ).mappings().first()
        if backtest is None:
            raise ValueError("strategy feature drift formal backtest identity is invalid")
        backtest = dict(backtest)
        version = dict(version)
        metrics = dict(backtest.get("metrics_json") or {})
        provenance = metrics.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("formal backtest provenance is missing")
        reference_dataset_identity = _sha256(
            provenance.get("dataset_identity_sha256"),
            field="formal dataset identity",
        )
        formal_lineage = _sha256(
            ((metrics.get("capital_oos_receipt") or {}).get("dataset_lineage_id"))
            or provenance.get("dataset_lineage_id")
            or "",
            field="formal dataset lineage",
        )
        if formal_lineage != current_dataset_lineage_id:
            raise ValueError("current factor dataset does not extend the formal lineage")

        artifact_root = Path(str(backtest.get("artifact_path") or "")).resolve()
        formal_manifest_sha256 = _sha256(
            provenance.get("artifact_manifest_sha256"),
            field="formal artifact manifest",
        )
        validate_backtest_artifact_manifest(
            artifact_root,
            expected_sha256=formal_manifest_sha256,
        )
        execution_manifest_path = artifact_root / "manifest.json"
        if (
            not execution_manifest_path.is_file()
            or sha256_file(execution_manifest_path)
            != _sha256(
                provenance.get("execution_manifest_sha256"),
                field="formal execution manifest",
            )
        ):
            raise ValueError("formal execution manifest changed")
        execution_manifest = _json_object(execution_manifest_path)
        if (
            str(execution_manifest.get("backtest_id") or "") != formal_backtest_id
            or str(execution_manifest.get("strategy_version_id") or "") != version_id
            or execution_manifest.get("dataset") != backtest.get("dataset")
        ):
            raise ValueError("formal execution manifest identity changed")

        drift_factor_ids = set(feature_set.get("drift_factor_ids") or feature_set["features"])
        reference_path = artifact_root / STRATEGY_HEALTH_REFERENCE_NAME
        reference_receipt = provenance.get("strategy_health_reference")
        if (
            not isinstance(reference_receipt, dict)
            or reference_receipt.get("path") != STRATEGY_HEALTH_REFERENCE_NAME
            or not reference_path.is_file()
            or reference_path.is_symlink()
            or sha256_file(reference_path) != reference_receipt.get("sha256")
        ):
            raise ValueError("formal strategy-health reference is missing or changed")
        reference = validate_strategy_health_reference(
            _json_object(reference_path),
            strategy_version_id=version_id,
            formal_backtest_id=formal_backtest_id,
            formal_dataset_identity_sha256=reference_dataset_identity,
            formal_dataset_lineage_id=formal_lineage,
            strategy_rules_sha256=_sha256(
                version.get("strategy_rules_sha256"), field="strategy rules"
            ),
            expected_factor_ids=drift_factor_ids,
            signal_source=str(feature_set.get("signal_source") or "factor_score"),
        )
        if reference_receipt.get("reference_sha256") != reference["reference_sha256"]:
            raise ValueError("formal strategy-health reference receipt changed")
        materialization = self._current_materialization(
            current_dataset_identity_sha256=current_dataset_identity_sha256,
            feature_set=feature_set,
        )
        factors: dict[str, dict[str, Any]] = {}
        current_hashes: dict[str, str] = {}
        for factor_id in sorted(drift_factor_ids):
            values, digest = self._current_factor(
                materialization=materialization,
                factor_id=factor_id,
            )
            factors[factor_id] = factor_drift_from_reference(
                values,
                reference=reference["factors"][factor_id],
                as_of=date.fromisoformat(str(materialization["manifest"]["end"])),
                current_window_sessions=20,
                current_file_sha256=digest,
            )
            current_hashes[factor_id] = digest
            del values
        periods = dict(backtest.get("periods_json") or {})
        reference_start = str(periods.get("start") or "")
        reference_end = str(periods.get("end") or "")
        as_of = pd.Timestamp(str(materialization["manifest"]["end"])).date()
        contract = {
            "contract_version": "strategy-feature-drift-reference-v1",
            "strategy_version_id": version_id,
            "strategy_rules_sha256": _sha256(
                version.get("strategy_rules_sha256"), field="strategy rules"
            ),
            "factor_set_sha256": str(feature_set["factor_set_sha256"]),
            "feature_set_definition_sha256": str(feature_set["definition_sha256"]),
            "factor_ids": sorted(factors),
            "formal_backtest_id": str(backtest["id"]),
            "formal_artifact_manifest_sha256": formal_manifest_sha256,
            "reference_dataset_identity_sha256": reference_dataset_identity,
            "reference_start": reference_start,
            "reference_end": reference_end,
            "current_window_sessions": 20,
            "bins": 10,
            "minimum_reference_sessions": 20,
            "minimum_reference_observations": 500,
            "minimum_current_observations": 100,
        }
        contract = {**contract, "contract_sha256": canonical_sha256(contract)}
        drifts = np.asarray(
            [float(item["normalized_psi"]) for item in factors.values()], dtype=float
        )
        max_factor_id = max(
            factors, key=lambda item: float(factors[item]["normalized_psi"])
        )
        observation = {
            "contract_version": FEATURE_DRIFT_OBSERVATION_VERSION,
            "formula_version": FEATURE_DRIFT_FORMULA_VERSION,
            "metric": "feature_drift",
            "metric_semantics": "max_per_factor_reference_decile_psi_transformed_1_minus_exp",
            "strategy_version_id": version_id,
            "strategy_rules_sha256": contract["strategy_rules_sha256"],
            "factor_set_sha256": contract["factor_set_sha256"],
            "feature_set_definition_sha256": contract[
                "feature_set_definition_sha256"
            ],
            "factor_ids": sorted(factors),
            "formal_backtest_id": formal_backtest_id,
            "formal_artifact_manifest_sha256": formal_manifest_sha256,
            "reference_contract_sha256": reference["reference_sha256"],
            "reference_dataset_identity_sha256": reference_dataset_identity,
            "current_dataset_identity_sha256": current_dataset_identity_sha256,
            "current_dataset_lineage_id": current_dataset_lineage_id,
            "materialization_manifest_sha256": materialization["manifest_sha256"],
            "reference_start": reference_start,
            "reference_end": reference_end,
            "current_start": min(item["current_start"] for item in factors.values()),
            "current_end": as_of.isoformat(),
            "current_window_sessions_requested": 20,
            "factor_count": len(factors),
            "factors": factors,
            "aggregation": {
                "method": "maximum",
                "max_factor_id": max_factor_id,
                "maximum": float(drifts.max()),
                "median": float(np.median(drifts)),
                "p75": float(np.quantile(drifts, 0.75)),
            },
            "feature_drift": float(drifts.max()),
            "observed_at": f"{as_of.isoformat()}T15:00:00+08:00",
        }
        observation = {
            **observation,
            "observation_sha256": canonical_sha256(observation),
        }
        return validate_factor_psi_observation(
            observation,
            strategy_version_id=version_id,
            current_dataset_identity_sha256=current_dataset_identity_sha256,
            expected_as_of=as_of,
        )

    def observe_model_calibration(
        self,
        *,
        version_id: str,
        formal_backtest_id: str,
        current_dataset_identity_sha256: str,
        current_dataset_lineage_id: str,
        signal_date: date,
    ) -> dict[str, Any]:
        feature_set = self.feature_set(version_id)
        if feature_set.get("signal_source") != "model_prediction":
            raise ValueError("factor strategy has no model calibration contract")
        with self.engine.connect() as connection:
            version = connection.execute(
                select(strategy_versions).where(strategy_versions.c.id == version_id)
            ).mappings().one()
            backtest = connection.execute(
                select(backtest_runs).where(
                    backtest_runs.c.id == formal_backtest_id,
                    backtest_runs.c.strategy_version_id == version_id,
                    backtest_runs.c.status == "succeeded",
                    backtest_runs.c.is_legacy.is_(False),
                )
            ).mappings().first()
        if backtest is None:
            raise ValueError("model calibration formal backtest identity is invalid")
        backtest = dict(backtest)
        provenance = dict((backtest.get("metrics_json") or {}).get("provenance") or {})
        formal_identity = _sha256(
            provenance.get("dataset_identity_sha256"), field="formal dataset identity"
        )
        formal_lineage = _sha256(
            ((backtest.get("metrics_json") or {}).get("capital_oos_receipt") or {}).get(
                "dataset_lineage_id"
            )
            or provenance.get("dataset_lineage_id"),
            field="formal dataset lineage",
        )
        if formal_lineage != current_dataset_lineage_id:
            raise ValueError("current model dataset does not extend the formal lineage")
        artifact_root = Path(str(backtest.get("artifact_path") or "")).resolve()
        validate_backtest_artifact_manifest(
            artifact_root,
            expected_sha256=provenance.get("artifact_manifest_sha256"),
        )
        execution_manifest_path = artifact_root / "manifest.json"
        if (
            not execution_manifest_path.is_file()
            or sha256_file(execution_manifest_path)
            != _sha256(
                provenance.get("execution_manifest_sha256"),
                field="formal execution manifest",
            )
        ):
            raise ValueError("formal model execution manifest changed")
        execution_manifest = _json_object(execution_manifest_path)
        if (
            execution_manifest.get("backtest_id") != formal_backtest_id
            or execution_manifest.get("strategy_version_id") != version_id
            or execution_manifest.get("dataset") != backtest.get("dataset")
        ):
            raise ValueError("formal model execution manifest identity changed")
        reference_path = artifact_root / STRATEGY_HEALTH_REFERENCE_NAME
        reference_receipt = provenance.get("strategy_health_reference")
        if (
            not isinstance(reference_receipt, dict)
            or reference_receipt.get("path") != STRATEGY_HEALTH_REFERENCE_NAME
            or not reference_path.is_file()
            or reference_path.is_symlink()
            or sha256_file(reference_path) != reference_receipt.get("sha256")
        ):
            raise ValueError("formal model calibration reference is missing or changed")
        reference = validate_strategy_health_reference(
            _json_object(reference_path),
            strategy_version_id=version_id,
            formal_backtest_id=formal_backtest_id,
            formal_dataset_identity_sha256=formal_identity,
            formal_dataset_lineage_id=formal_lineage,
            strategy_rules_sha256=_sha256(
                version.get("strategy_rules_sha256"), field="strategy rules"
            ),
            expected_factor_ids=set(feature_set["drift_factor_ids"]),
            signal_source="model_prediction",
        )
        if reference_receipt.get("reference_sha256") != reference["reference_sha256"]:
            raise ValueError("formal model calibration reference receipt changed")
        materialization = self._current_materialization(
            current_dataset_identity_sha256=current_dataset_identity_sha256,
            feature_set=feature_set,
        )
        label_horizon = int(
            feature_set["model_label_contract"]["label_horizon_sessions"]
        )
        artifact_limit = min(512, max(64, label_horizon + 64))
        live_label, label_hash, label_provenance_sha256 = self._current_label(
            materialization=materialization,
            current_dataset_identity_sha256=current_dataset_identity_sha256,
            current_dataset_lineage_id=current_dataset_lineage_id,
            label_contract=feature_set["model_label_contract"],
            signal_date=signal_date,
            session_limit=artifact_limit,
        )
        with self.engine.connect() as connection:
            formal_rows = connection.execute(
                select(model_artifacts)
                .where(
                    model_artifacts.c.strategy_version_id == version_id,
                    model_artifacts.c.training_kind == "formal_oos",
                )
            ).mappings().all()
            rows = connection.execute(
                select(model_artifacts)
                .where(
                    model_artifacts.c.strategy_version_id == version_id,
                    model_artifacts.c.artifact_key.like("live-refit-%"),
                )
                .order_by(model_artifacts.c.created_at.desc())
                .limit(artifact_limit)
            ).mappings().all()
        if len(formal_rows) != 1:
            raise ValueError("model calibration requires one formal OOS prediction artifact")
        formal_row = dict(formal_rows[0])
        formal_prediction_path = Path(str(formal_row.get("artifact_path") or "")).resolve()
        if (
            str(formal_row.get("predictions_sha256") or "")
            != reference["model_calibration"]["formal_predictions_sha256"]
            or not formal_prediction_path.is_file()
            or formal_prediction_path.is_symlink()
            or sha256_file(formal_prediction_path)
            != reference["model_calibration"]["formal_predictions_sha256"]
        ):
            raise ValueError("formal model prediction reference changed")
        live_frames: list[pd.DataFrame] = []
        live_hashes: dict[str, str] = {}
        current_signal_bound = False
        for raw in reversed(rows):
            row = dict(raw)
            key = str(row.get("artifact_key") or "")
            if not key.startswith("live-refit-"):
                continue
            try:
                prediction_date = date.fromisoformat(
                    f"{key[11:15]}-{key[15:17]}-{key[17:19]}"
                )
            except ValueError as exc:
                raise ValueError("live model artifact key has an invalid date") from exc
            if prediction_date > signal_date:
                continue
            predictions = _read_model_predictions(row)
            observed_dates = {
                item.date()
                for item in pd.DatetimeIndex(
                    predictions.index.get_level_values("datetime")
                ).normalize()
            }
            if observed_dates != {prediction_date}:
                raise ValueError("live model predictions do not match their artifact date")
            if prediction_date == signal_date:
                if row.get("dataset_identity_sha256") != current_dataset_identity_sha256:
                    raise ValueError("paper signal used a model artifact from another dataset")
                current_signal_bound = True
            live_frames.append(predictions)
            live_hashes[str(row["id"])] = str(row["predictions_sha256"])
        if not current_signal_bound:
            raise ValueError("latest paper signal has no exact model prediction artifact")
        live_predictions = pd.concat(live_frames).sort_index() if live_frames else pd.DataFrame()
        live = live_model_calibration_from_reference(
            live_predictions,
            live_label,
            reference=reference["model_calibration"],
        )
        formal = dict(reference["model_calibration"])
        observation = {
            "contract_version": MODEL_CALIBRATION_OBSERVATION_VERSION,
            "formula_version": MODEL_CALIBRATION_FORMULA_VERSION,
            "metric": "model_calibration_drift",
            "strategy_version_id": version_id,
            "label_horizon_sessions": label_horizon,
            "label_contract_sha256": formal["label_contract_sha256"],
            "formal_predictions_sha256": formal["formal_predictions_sha256"],
            "live_prediction_hashes": dict(sorted(live_hashes.items())),
            "label_materialization_manifest_sha256": label_provenance_sha256,
            "label_materialized_file_sha256": label_hash,
            "label_source": "exact_current_qlib_bounded_query",
            "label_query_session_limit": artifact_limit,
            "current_dataset_identity_sha256": current_dataset_identity_sha256,
            "current_dataset_lineage_id": current_dataset_lineage_id,
            "formal_start": formal["formal_start"],
            "formal_end": formal["formal_end"],
            "formal_sessions": formal["formal_sessions"],
            "formal_observations": formal["formal_observations"],
            "live_start": live["live_start"],
            "live_end": live["live_end"],
            "live_sessions": live["live_sessions"],
            "live_observations": live["live_observations"],
            "as_of": signal_date.isoformat(),
            "formal_deciles": formal["formal_deciles"],
            "live_deciles": live["live_deciles"],
            "formal_realized_return_scale": formal[
                "formal_realized_return_scale"
            ],
            "standardized_decile_rmse": live["standardized_decile_rmse"],
            "model_calibration_drift": live["model_calibration_drift"],
            "formal_values_sha256": formal["formal_values_sha256"],
            "live_values_sha256": live["live_values_sha256"],
            "formal_reference_sha256": formal["summary_sha256"],
            "observed_at": f"{signal_date.isoformat()}T15:00:00+08:00",
        }
        observation = {
            **observation,
            "observation_sha256": canonical_sha256(observation),
        }
        return validate_model_calibration_observation(
            observation,
            strategy_version_id=version_id,
            current_dataset_identity_sha256=current_dataset_identity_sha256,
            expected_as_of=signal_date,
        )

    def _current_label(
        self,
        *,
        materialization: dict[str, Any],
        current_dataset_identity_sha256: str,
        current_dataset_lineage_id: str,
        label_contract: dict[str, Any],
        signal_date: date,
        session_limit: int,
    ) -> tuple[pd.DataFrame, str, str]:
        """Load one bounded live label expression directly from the sealed provider."""

        dataset_path = Path(str(materialization.get("dataset_path") or "")).resolve(
            strict=True
        )
        qlib_root = (self.data_root / "qlib").resolve(strict=True)
        if not dataset_path.is_relative_to(qlib_root):
            raise ValueError("strategy health dataset path escapes the Qlib root")
        provenance_path = dataset_path / "metadata" / "provenance.json"
        provenance = _json_object(provenance_path)
        if (
            provenance.get("dataset_identity_sha256")
            != current_dataset_identity_sha256
            or provenance.get("dataset_lineage_id") != current_dataset_lineage_id
        ):
            raise ValueError("strategy health live label dataset identity changed")
        calendar_path = dataset_path / "calendars" / "day.txt"
        try:
            calendar = sorted(
                {
                    date.fromisoformat(line.strip()[:10])
                    for line in calendar_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                    and date.fromisoformat(line.strip()[:10]) <= signal_date
                }
            )
        except (OSError, ValueError) as exc:
            raise ValueError("strategy health Qlib calendar is invalid") from exc
        if not calendar or calendar[-1] != signal_date:
            raise ValueError("strategy health Qlib calendar does not reach the signal date")
        selected = calendar[-session_limit:]
        expression = str(label_contract.get("label_expression") or "")
        if not expression:
            raise ValueError("model calibration label expression is missing")
        import qlib
        from qlib.data import D

        qlib.init(provider_uri=str(dataset_path), region="cn")
        values = D.features(
            D.instruments("cn_all"),
            [expression],
            start_time=selected[0].isoformat(),
            end_time=selected[-1].isoformat(),
            freq="day",
        )
        if not isinstance(values, pd.DataFrame) or values.shape[1] != 1:
            raise ValueError("model calibration live label query is invalid")
        return values, factor_values_sha256(values), sha256_file(provenance_path)

    def _reference_values(
        self,
        *,
        artifact_root: Path,
        execution_manifest: dict[str, Any],
        provenance: dict[str, Any],
        feature_set: dict[str, Any],
    ) -> tuple[dict[str, pd.Series | pd.DataFrame], dict[str, str]]:
        expected = set(feature_set["features"])
        values: dict[str, pd.Series | pd.DataFrame] = {}
        hashes: dict[str, str] = {}
        baseline = execution_manifest.get("baseline")
        baseline_raw_hashes = provenance.get("baseline_raw_values_sha256")
        if isinstance(baseline, dict):
            if (
                baseline.get("definition_sha256")
                != (execution_manifest.get("config") or {}).get(
                    "baseline_definition_sha256"
                )
                or not isinstance(baseline_raw_hashes, dict)
            ):
                raise ValueError("formal baseline feature identity changed")
            raw = ((baseline.get("artifacts") or {}).get("raw")) or {}
            for factor_id, entry in raw.items():
                if factor_id not in expected:
                    continue
                path = _artifact_path(artifact_root, entry.get("path"))
                digest = _sha256(entry.get("sha256"), field="formal baseline factor")
                if baseline_raw_hashes.get(factor_id) != digest or sha256_file(path) != digest:
                    raise ValueError(f"formal baseline factor changed: {factor_id}")
                values[factor_id] = pd.read_parquet(path)
                hashes[factor_id] = digest

        formal_hashes = provenance.get("formal_factor_values_sha256")
        for item in execution_manifest.get("factors") or []:
            candidate_id = str(item.get("candidate_id") or "")
            factor_id = f"candidate__{candidate_id}"
            if factor_id not in expected:
                continue
            artifact = item.get("formal_factor_artifact")
            if not isinstance(artifact, dict) or not isinstance(formal_hashes, dict):
                raise ValueError(f"formal challenger factor is missing: {candidate_id}")
            path = _artifact_path(artifact_root, artifact.get("path"))
            digest = _sha256(artifact.get("sha256"), field="formal challenger factor")
            if formal_hashes.get(candidate_id) != digest or sha256_file(path) != digest:
                raise ValueError(f"formal challenger factor changed: {candidate_id}")
            values[factor_id] = pd.read_hdf(path)
            hashes[factor_id] = digest
        if set(values) != expected:
            raise ValueError("formal factor reference is incomplete")
        return values, hashes

    def _current_materialization(
        self,
        *,
        current_dataset_identity_sha256: str,
        feature_set: dict[str, Any],
    ) -> dict[str, Any]:
        root = (
            self.data_root
            / "artifacts"
            / "factor-library-materializations"
            / current_dataset_identity_sha256
            / str(feature_set["definition_sha256"])[:16]
        ).resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("strategy factor materialization is not ready")
        manifest_sha256 = sha256_file(manifest_path)
        manifest = _json_object(manifest_path)
        if (
            manifest.get("contract_version") != "factor-library-materialization-v1"
            or manifest.get("dataset_identity_sha256")
            != current_dataset_identity_sha256
            or manifest.get("feature_set_id") != feature_set["id"]
            or manifest.get("feature_set_definition_sha256")
            != feature_set["definition_sha256"]
            or manifest.get("universe") != "cn_all"
            or manifest.get("status") not in {"complete", "complete_with_blockers"}
            or manifest.get("feature_set")
            != _materialized_feature_definition(feature_set)
            or manifest.get("storage_mode") != "recent_only"
            or int(manifest.get("session_limit") or 0) != 64
            or manifest.get("materialized_end") != manifest.get("requested_end")
            or not isinstance(manifest.get("completed"), dict)
            or not set(feature_set["features"]).issubset(manifest["completed"])
            or any(
                not _valid_recent_materialization_entry(
                    item,
                    expected_end=str(manifest.get("materialized_end") or ""),
                )
                for item in manifest["completed"].values()
            )
        ):
            raise ValueError("strategy factor materialization identity is invalid")
        with self.engine.connect() as connection:
            receipts = connection.execute(
                select(jobs.c.payload_json, jobs.c.progress_json)
                .where(
                    jobs.c.kind == "factor_library_materialize",
                    jobs.c.status == "succeeded",
                    jobs.c.payload_json["dataset_identity_sha256"].as_string()
                    == current_dataset_identity_sha256,
                    jobs.c.payload_json[
                        "feature_set_definition_sha256"
                    ].as_string()
                    == feature_set["definition_sha256"],
                )
                .order_by(jobs.c.finished_at.desc())
                .limit(2)
            ).all()
        matched = next(
            (
                (dict(row.payload_json or {}), dict(row.progress_json or {}))
                for row in receipts
                if (row.progress_json or {}).get("materialization_manifest_sha256")
                == manifest_sha256
            ),
            None,
        )
        if matched is None:
            raise ValueError("strategy factor materialization has no sealed job receipt")
        receipt_payload, _receipt_progress = matched
        dataset_path = Path(str(receipt_payload.get("dataset_path") or "")).resolve(
            strict=True
        )
        if (
            receipt_payload.get("dataset_identity_sha256")
            != current_dataset_identity_sha256
            or receipt_payload.get("feature_set_definition_sha256")
            != feature_set["definition_sha256"]
        ):
            raise ValueError("strategy factor materialization receipt identity changed")
        return {
            "root": root,
            "manifest": manifest,
            "manifest_sha256": manifest_sha256,
            "dataset_path": dataset_path,
        }

    @staticmethod
    def _current_factor(
        *,
        materialization: dict[str, Any],
        factor_id: str,
    ) -> tuple[pd.DataFrame, str]:
        manifest = materialization["manifest"]
        entry = (manifest.get("completed") or {}).get(factor_id)
        if not isinstance(entry, dict):
            raise ValueError(f"strategy factor materialization is missing {factor_id}")
        path = _artifact_path(
            Path(materialization["root"]), entry.get("recent_relative_path")
        )
        digest = _sha256(entry.get("recent_sha256"), field="materialized factor")
        if sha256_file(path) != digest:
            raise ValueError(f"materialized factor changed: {factor_id}")
        return pd.read_parquet(path), digest

    def _current_values(
        self,
        *,
        current_dataset_identity_sha256: str,
        feature_set: dict[str, Any],
        require_job_receipt: bool = True,
        factor_ids: set[str] | None = None,
        recent: bool = True,
    ) -> tuple[
        dict[str, pd.Series | pd.DataFrame],
        dict[str, str],
        dict[str, Any],
    ]:
        root = (
            self.data_root
            / "artifacts"
            / "factor-library-materializations"
            / current_dataset_identity_sha256
            / str(feature_set["definition_sha256"])[:16]
        ).resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("strategy factor materialization is not ready")
        manifest_sha256 = sha256_file(manifest_path)
        manifest = _json_object(manifest_path)
        if (
            manifest.get("contract_version") != "factor-library-materialization-v1"
            or manifest.get("dataset_identity_sha256")
            != current_dataset_identity_sha256
            or manifest.get("feature_set_id") != feature_set["id"]
            or manifest.get("feature_set_definition_sha256")
            != feature_set["definition_sha256"]
            or manifest.get("universe") != "cn_all"
            or manifest.get("status") not in {"complete", "complete_with_blockers"}
            or manifest.get("feature_set")
            != _materialized_feature_definition(feature_set)
        ):
            raise ValueError("strategy factor materialization identity is invalid")
        if require_job_receipt:
            with self.engine.connect() as connection:
                receipts = connection.execute(
                    select(jobs.c.payload_json, jobs.c.progress_json)
                    .where(
                        jobs.c.kind == "factor_library_materialize",
                        jobs.c.status == "succeeded",
                    )
                    .order_by(jobs.c.finished_at.desc())
                    .limit(200)
                ).all()
            matched_receipt = next(
                (
                    dict(row.progress_json or {})
                    for row in receipts
                    if (row.payload_json or {}).get("dataset_identity_sha256")
                    == current_dataset_identity_sha256
                    and (row.payload_json or {}).get(
                        "feature_set_definition_sha256"
                    )
                    == feature_set["definition_sha256"]
                ),
                None,
            )
            if (
                matched_receipt is None
                or matched_receipt.get("materialization_manifest_sha256")
                != manifest_sha256
            ):
                raise ValueError("strategy factor materialization has no sealed job receipt")
        completed = manifest.get("completed")
        selected_ids = set(factor_ids or feature_set["features"])
        if not selected_ids.issubset(feature_set["features"]):
            raise ValueError("strategy factor materialization selection is invalid")
        if not isinstance(completed, dict) or not selected_ids.issubset(completed):
            raise ValueError("strategy factor materialization is incomplete")
        values: dict[str, pd.Series | pd.DataFrame] = {}
        hashes: dict[str, str] = {}
        for factor_id in sorted(selected_ids):
            entry = completed.get(factor_id)
            if not isinstance(entry, dict):
                raise ValueError(f"strategy factor materialization is missing {factor_id}")
            path_field = "recent_relative_path" if recent else "relative_path"
            hash_field = "recent_sha256" if recent else "sha256"
            path = _artifact_path(root, entry.get(path_field))
            digest = _sha256(entry.get(hash_field), field="materialized factor")
            if sha256_file(path) != digest:
                raise ValueError(f"materialized factor changed: {factor_id}")
            values[factor_id] = (
                pd.read_parquet(path) if recent else pd.read_hdf(path)
            )
            hashes[factor_id] = digest
        return values, hashes, {
            "manifest": manifest,
            "manifest_sha256": manifest_sha256,
        }


def _read_model_predictions(row: dict[str, Any]) -> pd.DataFrame:
    path = Path(str(row.get("artifact_path") or "")).resolve()
    digest = _sha256(row.get("predictions_sha256"), field="model predictions")
    evidence = dict(row.get("training_evidence_json") or {})
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != digest
        or str(row.get("artifact_sha256") or "") != digest
        or canonical_sha256(evidence)
        != str(row.get("training_evidence_sha256") or "")
    ):
        raise ValueError("model prediction artifact changed")
    if path.suffix.lower() == ".parquet":
        value = pd.read_parquet(path)
    elif path.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        value = pd.read_hdf(path)
    else:
        raise ValueError("model prediction artifact format is unsupported")
    if isinstance(value, pd.Series):
        value = value.to_frame("score")
    if not isinstance(value, pd.DataFrame) or value.shape[1] != 1:
        raise ValueError("model prediction artifact must contain one score")
    return value


def _materialized_feature_definition(feature_set: dict[str, Any]) -> dict[str, Any]:
    value = {
        key: feature_set[key]
        for key in (
            "contract_version",
            "id",
            "name",
            "features",
            "source",
            "definition_sha256",
        )
    }
    if "materialization_contract" in feature_set:
        value["materialization_contract"] = feature_set["materialization_contract"]
    return value


def _valid_recent_materialization_entry(
    value: Any, *, expected_end: str
) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        recent_start = date.fromisoformat(str(value.get("recent_start") or ""))
        recent_end = date.fromisoformat(str(value.get("recent_end") or ""))
    except ValueError:
        return False
    return (
        int(value.get("recent_session_limit") or 0) == 64
        and recent_start <= recent_end
        and recent_end.isoformat() == expected_end
        and value.get("relative_path") is None
        and value.get("sha256") is None
    )


def _artifact_path(root: Path, value: Any) -> Path:
    relative = PurePosixPath(str(value or ""))
    if (
        not relative.parts
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("factor artifact path is unsafe")
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError("factor artifact is missing or redirected")
    return path


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"artifact JSON is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"artifact JSON must be an object: {path.name}")
    return value


def _sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest
