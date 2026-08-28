from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    model_artifacts,
    open_database,
    row_dict,
    strategy_versions,
)

from .model_recompute import (
    GOVERNED_MODEL_ENGINES,
    governed_checkpoint_format,
    verify_governed_checkpoint,
)
from .model_research_governance import (
    MODEL_REFIT_POLICY,
    MODEL_REFIT_POLICY_SHA256,
    verify_model_prediction_artifact,
)
from .strategy_store import StrategyStore

MODEL_ARTIFACT_CONTRACT_VERSION = "model-artifact-lifecycle-v2-checkpointed"
INITIAL_FORMAL_ARTIFACT_VALID_DAYS = 7


def _now() -> datetime:
    return datetime.now(UTC)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _structural_model_data_contract(value: dict[str, Any]) -> dict[str, Any]:
    """Remove only rolling date values; every semantic field stays frozen."""

    result = json.loads(json.dumps(value, ensure_ascii=False))
    normalization = result.get("feature_normalization")
    if isinstance(normalization, dict):
        normalization.pop("fit_start_time", None)
        normalization.pop("fit_end_time", None)
    return result


def _frozen_model_engine(model_signal: dict[str, Any]) -> str:
    recipe = dict(model_signal.get("recipe") or {})
    recipe_hyperparameters = dict(recipe.get("model_hyperparameters") or {})
    signal_hyperparameters = dict(model_signal.get("model_hyperparameters") or {})
    engine = str(
        recipe.get("model_engine")
        or recipe_hyperparameters.get("model_engine")
        or signal_hyperparameters.get("model_engine")
        or "rdagent_pytorch"
    )
    if engine not in GOVERNED_MODEL_ENGINES:
        raise ValueError("frozen strategy requests an ungoverned model engine")
    return engine


class ModelArtifactStore:
    """Immutable fitted-model artifacts under one frozen StrategySpec."""

    def __init__(self, database_url: str) -> None:
        self.strategies = StrategyStore(database_url)
        self.engine = open_database(database_url)

    @staticmethod
    def _verify_fitted_checkpoint(row: Any) -> None:
        evidence = row.training_evidence_json
        evidence_sha256 = str(row.training_evidence_sha256 or "").lower()
        model_data_contract = (
            evidence.get("model_data_contract") if isinstance(evidence, dict) else None
        )
        if (
            not isinstance(evidence, dict)
            or not _is_sha256(evidence_sha256)
            or _canonical_sha256(evidence) != evidence_sha256
            or not _is_sha256(row.model_data_contract_sha256)
            or not isinstance(model_data_contract, dict)
            or _canonical_sha256(model_data_contract)
            != str(row.model_data_contract_sha256)
        ):
            raise ValueError("ModelArtifact fitted-model provenance is incomplete")
        verify_governed_checkpoint(
            Path(str(row.checkpoint_path or "")).resolve(),
            model_engine=str(evidence.get("model_engine") or ""),
            checkpoint_format=str(row.checkpoint_format or ""),
            expected_sha256=str(row.checkpoint_sha256 or ""),
        )

    def strategy_spec_sha256(self, strategy_version_id: str) -> str:
        version = self.strategies.get_version(strategy_version_id)
        factors = [
            {
                "candidate_id": item["factor_candidate_id"],
                "factor_evaluation_id": item["factor_evaluation_id"],
                "weight": item["weight"],
                "direction": item["direction"],
            }
            for item in version.get("factors") or []
        ]
        return _canonical_sha256(
            {
                "strategy_version_id": str(version["id"]),
                "strategy_type": version["strategy_type"],
                "signal_frequency": version["signal_frequency"],
                "signal_horizon": version["signal_horizon"],
                "execution_frequency": version["execution_frequency"],
                "execution_contract_hash": version["execution_contract_hash"],
                "benchmark": version["benchmark"],
                "universe": version["universe"],
                "config": version["config"],
                "factors": factors,
                "pair": version.get("pair"),
            }
        )

    def _governed_execution_environment_sha256(
        self, strategy_version_id: str
    ) -> str:
        """Rebuild the one admitted runtime identity for a model StrategySpec."""

        version = self.strategies.get_version(strategy_version_id)
        with self.engine.connect() as connection:
            evidence = self.strategies._model_signal_evidence(  # noqa: SLF001
                connection,
                version["config"],
            )
        if evidence is None:
            raise ValueError("ModelArtifact StrategySpec has no governed model evidence")
        admission = dict(evidence["candidate"].admission_evidence_json or {})
        binding = evidence.get("formal_admission_binding")
        model_grid = binding.get("model_grid") if isinstance(binding, dict) else None
        admitted_hash = str(admission.get("execution_environment_sha256") or "").lower()
        bound_hash = (
            str(model_grid.get("execution_environment_sha256") or "").lower()
            if isinstance(model_grid, dict)
            else ""
        )
        if (
            not _is_sha256(admitted_hash)
            or not _is_sha256(bound_hash)
            or admitted_hash != bound_hash
        ):
            raise ValueError(
                "model admission has no single immutable execution environment"
            )
        return bound_hash

    def create(
        self,
        *,
        strategy_version_id: str,
        artifact_key: str,
        model_recipe: dict[str, Any],
        dataset: str,
        dataset_identity_sha256: str,
        execution_environment_sha256: str,
        training_start: date,
        training_end: date,
        data_cutoff_at: datetime,
        valid_until: datetime,
        artifact_path: str | Path,
        predictions_sha256: str,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        checkpoint_format: str,
        model_data_contract_sha256: str,
        training_kind: str,
        training_evidence: dict[str, Any],
        actor: str,
        scheduled_refit_at: datetime | None = None,
        allow_dataset_rollover: bool = False,
    ) -> dict[str, Any]:
        version = self.strategies.get_version(strategy_version_id)
        if version.get("is_legacy"):
            raise ValueError("legacy StrategySpec versions cannot own ModelArtifacts")
        model_signal = version.get("model_signal")
        if str(version.get("config", {}).get("signal_source") or "factor_score") != (
            "model_prediction"
        ) or not isinstance(model_signal, dict):
            raise ValueError(
                "ModelArtifacts may only be registered for a governed model-prediction "
                "StrategySpec"
            )
        if training_end < training_start:
            raise ValueError("model training window is invalid")
        if data_cutoff_at.tzinfo is None or data_cutoff_at.utcoffset() is None:
            raise ValueError("model data cutoff must include a timezone")
        if valid_until.tzinfo is None or valid_until.utcoffset() is None:
            raise ValueError("model validity deadline must include a timezone")
        if valid_until <= data_cutoff_at:
            raise ValueError("model validity must extend beyond its data cutoff")
        if scheduled_refit_at is not None and (
            scheduled_refit_at.tzinfo is None
            or scheduled_refit_at.utcoffset() is None
        ):
            raise ValueError("scheduled refit timestamp must include a timezone")
        if (
            not _is_sha256(dataset_identity_sha256)
            or not _is_sha256(predictions_sha256)
            or not _is_sha256(checkpoint_sha256)
            or not _is_sha256(model_data_contract_sha256)
        ):
            raise ValueError(
                "ModelArtifact requires immutable dataset, prediction, checkpoint and "
                "data-contract hashes"
            )
        governed_environment_sha256 = self._governed_execution_environment_sha256(
            strategy_version_id
        )
        if (
            not _is_sha256(execution_environment_sha256)
            or execution_environment_sha256.lower() != governed_environment_sha256
        ):
            raise ValueError(
                "ModelArtifact execution environment does not match independent admission"
            )
        creator = actor.strip()
        key = artifact_key.strip()
        if len(creator) < 2 or not key or not dataset.strip() or not model_recipe:
            raise ValueError("ModelArtifact identity, recipe, dataset and actor are required")
        path = Path(artifact_path).resolve()
        if not path.is_file():
            raise ValueError("model artifact file does not exist")
        artifact_sha256 = _file_sha256(path)
        fitted_checkpoint_path = Path(checkpoint_path).resolve()
        if not isinstance(training_evidence, dict):
            raise ValueError("ModelArtifact training evidence is required")
        model_engine = str(training_evidence.get("model_engine") or "")
        model_data_contract = training_evidence.get("model_data_contract")
        if training_kind not in {
            "formal_oos",
            "monthly_retrain",
            "early_retrain",
            "daily_inference",
        }:
            raise ValueError("ModelArtifact training kind is invalid")
        if (
            not model_engine
            or model_engine != _frozen_model_engine(model_signal)
            or not isinstance(model_data_contract, dict)
            or _canonical_sha256(model_data_contract)
            != model_data_contract_sha256.lower()
        ):
            raise ValueError("ModelArtifact training evidence is required")
        verify_governed_checkpoint(
            fitted_checkpoint_path,
            model_engine=model_engine,
            checkpoint_format=checkpoint_format,
            expected_sha256=checkpoint_sha256,
        )
        training_evidence_sha256 = _canonical_sha256(training_evidence)
        spec_sha256 = self.strategy_spec_sha256(strategy_version_id)
        recipe_sha256 = _canonical_sha256(model_recipe)
        if recipe_sha256 != str(model_signal.get("model_recipe_sha256") or ""):
            raise ValueError(
                "ModelArtifact recipe does not match the model candidate frozen in StrategySpec"
            )
        if (
            not allow_dataset_rollover
            and dataset.strip() != str(model_signal.get("dataset") or "")
        ):
            raise ValueError("ModelArtifact dataset does not match the frozen model candidate")
        if artifact_sha256 != predictions_sha256.lower() or path.suffix.lower() not in {
            ".parquet",
            ".h5",
            ".hdf",
            ".hdf5",
        }:
            raise ValueError(
                "model-signal ModelArtifact must be the immutable prediction table "
                "whose file hash equals predictions_sha256"
            )
        now = _now()

        try:
            with self.engine.begin() as connection:
                active = connection.execute(
                    select(model_artifacts).where(
                        model_artifacts.c.strategy_version_id == strategy_version_id,
                        model_artifacts.c.status == "active",
                    )
                ).first()
                if active is not None and str(active.model_recipe_sha256) != recipe_sha256:
                    raise ValueError(
                        "routine refit changed the frozen model recipe; create a new "
                        "StrategySpec version"
                    )
                connection.execute(
                    insert(model_artifacts).values(
                        id=uuid.uuid4().hex,
                        strategy_version_id=strategy_version_id,
                        artifact_key=key,
                        status="candidate",
                        strategy_spec_sha256=spec_sha256,
                        model_recipe_sha256=recipe_sha256,
                        model_recipe_json=model_recipe,
                        dataset=dataset.strip(),
                        dataset_identity_sha256=dataset_identity_sha256.lower(),
                        execution_environment_sha256=governed_environment_sha256,
                        training_start=training_start,
                        training_end=training_end,
                        data_cutoff_at=data_cutoff_at.astimezone(UTC),
                        scheduled_refit_at=(
                            scheduled_refit_at.astimezone(UTC)
                            if scheduled_refit_at is not None
                            else None
                        ),
                        valid_until=valid_until.astimezone(UTC),
                        artifact_path=str(path),
                        artifact_sha256=artifact_sha256,
                        predictions_sha256=predictions_sha256.lower(),
                        checkpoint_path=str(fitted_checkpoint_path),
                        checkpoint_sha256=checkpoint_sha256.lower(),
                        checkpoint_format=checkpoint_format,
                        model_data_contract_sha256=model_data_contract_sha256.lower(),
                        training_kind=training_kind,
                        training_evidence_json=training_evidence,
                        training_evidence_sha256=training_evidence_sha256,
                        created_by=creator,
                        created_at=now,
                    )
                )
        except IntegrityError as exc:
            raise ValueError(f"ModelArtifact key {key!r} already exists") from exc
        return self.get_by_key(strategy_version_id, key)

    def create_from_formal_backtest(
        self,
        *,
        strategy_version_id: str,
        source_backtest_id: str,
        valid_until: datetime | None,
        actor: str,
        backtests_root: str | Path,
    ) -> dict[str, Any]:
        """Register only a server-produced, revalidated formal model artifact.

        The public API supplies an opaque backtest id, never host paths, recipes,
        datasets, or digests.  Every such value is rebuilt from the frozen
        StrategySpec and the succeeded formal-backtest evidence.
        """

        backtest = self.strategies.get_backtest(source_backtest_id)
        if str(backtest["strategy_version_id"]) != str(strategy_version_id):
            raise ValueError("source backtest belongs to another StrategySpec version")
        if str(backtest.get("status") or "") != "succeeded":
            raise ValueError("source backtest has not succeeded")
        metrics = backtest.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("source backtest has no validated metrics")

        governed_root = Path(backtests_root).resolve()
        artifact_root = Path(str(backtest.get("artifact_path") or "")).resolve()
        expected_root = (governed_root / source_backtest_id).resolve()
        if artifact_root != expected_root or not artifact_root.is_dir():
            raise ValueError("source backtest artifact root is outside governed storage")
        self.strategies.validate_backtest_artifacts(source_backtest_id, metrics)

        manifest_path = artifact_root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("source backtest manifest is unreadable") from exc
        if not isinstance(manifest, dict):
            raise ValueError("source backtest manifest must be an object")

        version = self.strategies.get_version(strategy_version_id)
        model_signal = version.get("model_signal")
        frozen_model = manifest.get("model_candidate")
        formal = manifest.get("formal_model_artifact")
        if not isinstance(model_signal, dict) or not isinstance(frozen_model, dict):
            raise ValueError("StrategySpec is not a governed model-prediction version")
        if not isinstance(formal, dict):
            raise ValueError("source backtest has no formal model artifact")
        candidate_manifest = frozen_model.get("candidate_manifest")
        training = frozen_model.get("training_periods")
        evidence = formal.get("evidence")
        if not all(isinstance(item, dict) for item in (candidate_manifest, training, evidence)):
            raise ValueError("formal model identity or training evidence is incomplete")
        candidate_manifest = dict(candidate_manifest)
        training = dict(training)
        evidence = dict(evidence)
        formal_environment_sha256 = str(
            evidence.get("execution_environment_sha256") or ""
        ).lower()
        formal_validation = metrics.get("formal_validation")
        formal_admission = (
            formal_validation.get("model_admission")
            if isinstance(formal_validation, dict)
            else None
        )
        model_grid = (
            formal_admission.get("model_grid")
            if isinstance(formal_admission, dict)
            else None
        )
        admitted_environment_sha256 = (
            str(model_grid.get("execution_environment_sha256") or "").lower()
            if isinstance(model_grid, dict)
            else ""
        )
        if (
            not _is_sha256(formal_environment_sha256)
            or not _is_sha256(admitted_environment_sha256)
            or formal_environment_sha256 != admitted_environment_sha256
        ):
            raise ValueError(
                "formal model artifact used another or unbound execution environment"
            )
        recipe = candidate_manifest.get("recipe")
        if (
            not isinstance(recipe, dict)
            or _canonical_sha256(recipe) != str(model_signal.get("model_recipe_sha256") or "")
            or candidate_manifest.get("id") != model_signal.get("model_candidate_id")
            or candidate_manifest.get("dataset") != model_signal.get("dataset")
            or candidate_manifest.get("dataset_identity_sha256")
            != model_signal.get("dataset_identity_sha256")
            or evidence.get("dataset_identity_sha256")
            != model_signal.get("dataset_identity_sha256")
        ):
            raise ValueError("formal model artifact disagrees with its frozen StrategySpec")

        relative_predictions = Path(str(formal.get("predictions_path") or ""))
        predictions_path = (artifact_root / relative_predictions).resolve()
        try:
            predictions_path.relative_to(artifact_root)
        except ValueError as exc:
            raise ValueError("formal model prediction path escapes the backtest") from exc
        predictions_sha256 = str(formal.get("predictions_sha256") or "").lower()
        if (
            relative_predictions.is_absolute()
            or not predictions_path.is_file()
            or not _is_sha256(predictions_sha256)
            or _file_sha256(predictions_path) != predictions_sha256
        ):
            raise ValueError("formal model predictions failed immutable verification")

        relative_checkpoint = Path(str(formal.get("checkpoint_path") or ""))
        checkpoint_path = (artifact_root / relative_checkpoint).resolve()
        try:
            checkpoint_path.relative_to(artifact_root)
        except ValueError as exc:
            raise ValueError("formal model checkpoint path escapes the backtest") from exc
        checkpoint_sha256 = str(formal.get("checkpoint_sha256") or "").lower()
        resource_policy = evidence.get("resource_policy")
        model_engine = (
            str(resource_policy.get("model_engine") or "")
            if isinstance(resource_policy, dict)
            else ""
        )
        checkpoint_format = governed_checkpoint_format(model_engine)
        verify_governed_checkpoint(
            checkpoint_path,
            model_engine=model_engine,
            checkpoint_format=checkpoint_format,
            expected_sha256=checkpoint_sha256,
        )
        model_data_contract = formal.get("model_data_contract")
        model_data_contract_sha256 = str(
            formal.get("model_data_contract_sha256") or ""
        ).lower()
        if (
            not isinstance(model_data_contract, dict)
            or not _is_sha256(model_data_contract_sha256)
            or _canonical_sha256(model_data_contract) != model_data_contract_sha256
            or evidence.get("model_data_contract_sha256")
            != model_data_contract_sha256
        ):
            raise ValueError("formal model artifact has no immutable data contract")

        try:
            training_start = date.fromisoformat(str(training["train_start"]))
            training_end = date.fromisoformat(str(training["valid_end"]))
            cutoff_date = date.fromisoformat(str(candidate_manifest["pre_final_end"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("formal model training window is invalid") from exc
        data_cutoff_at = datetime(
            cutoff_date.year,
            cutoff_date.month,
            cutoff_date.day,
            23,
            59,
            59,
            999999,
            tzinfo=UTC,
        )
        if valid_until is not None and (
            valid_until.tzinfo is None or valid_until.utcoffset() is None
        ):
            raise ValueError("model validity deadline must include a timezone")
        current = _now()
        effective_valid_until = valid_until or (
            max(current, data_cutoff_at)
            + timedelta(days=INITIAL_FORMAL_ARTIFACT_VALID_DAYS)
        )
        artifact_key = f"formal-backtest-{source_backtest_id}"
        try:
            existing = self.get_by_key(strategy_version_id, artifact_key)
        except KeyError:
            existing = None
        if existing is not None:
            expected_spec_sha256 = self.strategy_spec_sha256(strategy_version_id)
            expected_recipe_sha256 = _canonical_sha256(recipe)
            if (
                existing.get("status") not in {"candidate", "active"}
                or existing.get("strategy_spec_sha256") != expected_spec_sha256
                or existing.get("model_recipe_sha256") != expected_recipe_sha256
                or existing.get("model_recipe") != recipe
                or existing.get("dataset") != str(model_signal["dataset"])
                or existing.get("dataset_identity_sha256")
                != str(model_signal["dataset_identity_sha256"])
                or existing.get("execution_environment_sha256")
                != admitted_environment_sha256
                or existing.get("training_start") != training_start
                or existing.get("training_end") != training_end
                or Path(str(existing.get("artifact_path") or "")).resolve()
                != predictions_path
                or existing.get("artifact_sha256") != predictions_sha256
                or existing.get("predictions_sha256") != predictions_sha256
                or Path(str(existing.get("checkpoint_path") or "")).resolve()
                != checkpoint_path
                or existing.get("checkpoint_sha256") != checkpoint_sha256
                or existing.get("checkpoint_format") != checkpoint_format
                or existing.get("model_data_contract_sha256")
                != model_data_contract_sha256
                or existing.get("training_kind") != "formal_oos"
                or existing.get("valid_until") <= current
                or existing.get("scheduled_refit_at") is None
                or existing.get("scheduled_refit_at") > current
                or (
                    valid_until is None
                    and existing.get("valid_until") > effective_valid_until
                )
                or (
                    valid_until is not None
                    and existing.get("valid_until") != valid_until.astimezone(UTC)
                )
            ):
                raise ValueError(
                    "formal backtest ModelArtifact key exists with different or "
                    "expired evidence"
                )
            return existing
        training_evidence = {
            "model_engine": model_engine,
            "operation": "formal_oos",
            "periods": training,
            "source_backtest_id": source_backtest_id,
            "execution_evidence_sha256": str(
                evidence.get("evidence_sha256") or ""
            ),
            "model_data_contract": model_data_contract,
        }
        return self.create(
            strategy_version_id=strategy_version_id,
            artifact_key=artifact_key,
            model_recipe=recipe,
            dataset=str(model_signal["dataset"]),
            dataset_identity_sha256=str(model_signal["dataset_identity_sha256"]),
            execution_environment_sha256=admitted_environment_sha256,
            training_start=training_start,
            training_end=training_end,
            data_cutoff_at=data_cutoff_at,
            valid_until=effective_valid_until,
            artifact_path=predictions_path,
            predictions_sha256=predictions_sha256,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_format=checkpoint_format,
            model_data_contract_sha256=model_data_contract_sha256,
            training_kind="formal_oos",
            training_evidence=training_evidence,
            actor=actor,
            scheduled_refit_at=_now(),
        )

    def create_from_live_refit(
        self,
        *,
        strategy_version_id: str,
        source_model_artifact_id: str,
        result_path: str | Path,
        actor: str,
        valid_for_days: int = 4,
    ) -> dict[str, Any]:
        """Register one server-produced daily prediction and fitted checkpoint.

        Daily inference must reuse the active checkpoint byte-for-byte.  A
        monthly/evidenced retrain must create a new governed-format checkpoint.
        Model identity, recipe, dataset and all digests are re-bound to the
        approved StrategySpec and active source artifact before rotation.
        """

        source = self.get(source_model_artifact_id)
        if str(source["strategy_version_id"]) != str(strategy_version_id):
            raise ValueError("model refit source belongs to another StrategySpec")
        source_status = str(source.get("status") or "")
        if source_status not in {"active", "retired"}:
            raise ValueError("model refit source is no longer the active ModelArtifact")
        path = Path(result_path).resolve()
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("model refit result is unreadable") from exc
        if (
            not isinstance(result, dict)
            or result.get("contract_version") != "model-live-refresh-v2"
        ):
            raise ValueError("model refit result contract is invalid")
        recorded_evidence_sha256 = str(result.get("evidence_sha256") or "")
        unsigned = {key: value for key, value in result.items() if key != "evidence_sha256"}
        if not _is_sha256(recorded_evidence_sha256) or _canonical_sha256(unsigned) != (
            recorded_evidence_sha256
        ):
            raise ValueError("model refit result evidence hash is invalid")
        operation = str(result.get("operation") or "")
        inference_only = operation == "inference"
        if operation not in {"inference", "retrain"}:
            raise ValueError("model live refresh operation is invalid")
        if (
            result.get("status") != "passed"
            or result.get("final_oos_opened") is not False
            or (result.get("inference_only") is True) is not inference_only
            or result.get("strategy_version_id") != strategy_version_id
            or result.get("source_model_artifact_id") != source_model_artifact_id
        ):
            raise ValueError("model refit execution state is invalid")
        if (
            result.get("refit_policy") != MODEL_REFIT_POLICY
            or result.get("refit_policy_sha256") != MODEL_REFIT_POLICY_SHA256
            or _canonical_sha256(result["refit_policy"])
            != result["refit_policy_sha256"]
        ):
            raise ValueError("model refit changed the frozen training policy")

        version = self.strategies.get_version(strategy_version_id)
        model_signal = version.get("model_signal")
        if not isinstance(model_signal, dict):
            raise ValueError("model refit StrategySpec has no governed model signal")
        if (
            result.get("model_candidate_id") != model_signal.get("model_candidate_id")
            or result.get("model_code_sha256") != model_signal.get("model_code_sha256")
            or result.get("model_recipe_sha256") != model_signal.get("model_recipe_sha256")
            or result.get("feature_set_definition_sha256")
            != model_signal.get("feature_set_definition_sha256")
            or not str(result.get("dataset") or "").strip()
            or not _is_sha256(result.get("dataset_lineage_id"))
            or result.get("dataset_lineage_id") != model_signal.get("dataset_lineage_id")
            or not _is_sha256(result.get("dataset_identity_sha256"))
            or model_signal.get("refit_policy") != MODEL_REFIT_POLICY
            or model_signal.get("refit_policy_sha256") != MODEL_REFIT_POLICY_SHA256
        ):
            raise ValueError("model refit changed the frozen StrategySpec identity")

        periods = result.get("periods")
        coverage = result.get("coverage")
        if not isinstance(periods, dict) or not isinstance(coverage, dict):
            raise ValueError("model refit periods or coverage evidence are missing")
        try:
            signal_date = date.fromisoformat(str(result["signal_date"]))
            observed_training_start = date.fromisoformat(str(periods["train_start"]))
            observed_training_end = date.fromisoformat(str(periods["valid_end"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("model refit periods are invalid") from exc
        if (
            periods.get("test_start") != signal_date.isoformat()
            or periods.get("test_end") != signal_date.isoformat()
            or not (observed_training_start <= observed_training_end < signal_date)
            or coverage.get("coverage_gate_passed") is not True
            or coverage.get("test_start") != signal_date.isoformat()
            or coverage.get("test_end") != signal_date.isoformat()
        ):
            raise ValueError("model refit did not produce exact one-day live coverage")
        execution = result.get("execution_evidence")
        if not isinstance(execution, dict):
            raise ValueError("model refit execution evidence is missing")
        execution_hash = str(execution.get("evidence_sha256") or "")
        unsigned_execution = {
            key: value for key, value in execution.items() if key != "evidence_sha256"
        }
        if (
            execution.get("sandbox_mode") != "docker-isolated"
            or execution.get("network_mode") != "none"
            or execution.get("root_filesystem_read_only") is not True
            or execution.get("final_oos_opened") is not False
            or (execution.get("inference_only") is True) is not inference_only
            or (execution.get("live_retrain") is True) is not (not inference_only)
            or not _is_sha256(execution_hash)
            or _canonical_sha256(unsigned_execution) != execution_hash
            or execution.get("execution_environment_sha256")
            != source.get("execution_environment_sha256")
            or result.get("execution_environment_sha256")
            != source.get("execution_environment_sha256")
        ):
            raise ValueError("model refit execution evidence is invalid")

        expected_bundle = model_signal.get("bundle_factors") or []
        observed_bundle = result.get("bundle_factor_evidence") or []
        if not isinstance(expected_bundle, list) or not isinstance(observed_bundle, list):
            raise ValueError("model refit bundle-factor evidence is invalid")
        if [
            (str(item.get("candidate_id") or ""), str(item.get("code_sha256") or ""))
            for item in observed_bundle
        ] != [
            (str(item.get("candidate_id") or ""), str(item.get("code_sha256") or ""))
            for item in expected_bundle
        ] or any(
            not isinstance(item.get("pit_invariance"), dict)
            or item["pit_invariance"].get("status") != "passed"
            or not isinstance(item.get("coverage"), dict)
            or item["coverage"].get("coverage_gate_passed") is not True
            or not isinstance(item.get("execution"), dict)
            or item["execution"].get("sandbox_mode") != "docker-isolated"
            for item in observed_bundle
        ):
            raise ValueError("model refit bundle factors do not match the frozen bundle")

        result_root = path.parent.resolve()
        relative_predictions = Path(str(result.get("predictions_path") or ""))
        if relative_predictions.is_absolute():
            raise ValueError("model refit prediction path must be relative")
        predictions_path = (result_root / relative_predictions).resolve()
        try:
            predictions_path.relative_to(result_root)
        except ValueError as exc:
            raise ValueError(
                "model refit prediction path escapes its governed artifact root"
            ) from exc
        predictions_sha256 = str(result.get("predictions_sha256") or "").lower()
        if (
            not predictions_path.is_file()
            or not _is_sha256(predictions_sha256)
            or _file_sha256(predictions_path) != predictions_sha256
        ):
            raise ValueError("model refit predictions failed immutable verification")
        verify_model_prediction_artifact(
            predictions_path,
            expected_sha256=predictions_sha256,
            test_start=signal_date.isoformat(),
            test_end=signal_date.isoformat(),
            trading_days=[signal_date.isoformat()],
        )
        checkpoint_sha256 = str(result.get("checkpoint_sha256") or "").lower()
        checkpoint_format = str(result.get("checkpoint_format") or "")
        model_engine = str(
            (source.get("training_evidence") or {}).get("model_engine") or ""
        )
        if not model_engine:
            raise ValueError("source ModelArtifact has no governed model engine")
        if inference_only:
            if (
                result.get("checkpoint_path") is not None
                or result.get("checkpoint_reused") is not True
                or checkpoint_sha256 != source.get("checkpoint_sha256")
                or checkpoint_format != source.get("checkpoint_format")
            ):
                raise ValueError("daily inference did not reuse the active checkpoint")
            checkpoint_path = Path(str(source.get("checkpoint_path") or "")).resolve()
        else:
            relative_checkpoint = Path(str(result.get("checkpoint_path") or ""))
            checkpoint_path = (result_root / relative_checkpoint).resolve()
            try:
                checkpoint_path.relative_to(result_root)
            except ValueError as exc:
                raise ValueError(
                    "model refit checkpoint escapes its governed artifact root"
                ) from exc
            if relative_checkpoint.is_absolute() or result.get("checkpoint_reused") is True:
                raise ValueError("model retraining did not create a new checkpoint")
        verify_governed_checkpoint(
            checkpoint_path,
            model_engine=model_engine,
            checkpoint_format=checkpoint_format,
            expected_sha256=checkpoint_sha256,
        )
        model_data_contract = result.get("model_data_contract")
        model_data_contract_sha256 = str(
            result.get("model_data_contract_sha256") or ""
        ).lower()
        source_training_evidence = source.get("training_evidence") or {}
        source_model_data_contract = source_training_evidence.get(
            "model_data_contract"
        )
        if (
            not isinstance(model_data_contract, dict)
            or _canonical_sha256(model_data_contract)
            != model_data_contract_sha256
            or not _is_sha256(model_data_contract_sha256)
            or not isinstance(source_model_data_contract, dict)
        ):
            raise ValueError("model live refresh changed the fitted data contract")
        if inference_only:
            if model_data_contract_sha256 != str(
                source.get("model_data_contract_sha256") or ""
            ):
                raise ValueError("daily inference changed the fitted data contract")
        elif _structural_model_data_contract(
            model_data_contract
        ) != _structural_model_data_contract(source_model_data_contract):
            raise ValueError("model retraining changed the structural data contract")
        training_evidence = result.get("training_evidence")
        training_evidence_sha256 = str(
            result.get("training_evidence_sha256") or ""
        ).lower()
        if (
            not isinstance(training_evidence, dict)
            or training_evidence.get("operation") != operation
            or training_evidence.get("source_model_artifact_id")
            != source_model_artifact_id
            or not _is_sha256(training_evidence_sha256)
            or _canonical_sha256(training_evidence) != training_evidence_sha256
        ):
            raise ValueError("model live refresh training evidence is invalid")
        if inference_only:
            source_periods = (source.get("training_evidence") or {}).get("periods")
            if (
                not isinstance(source_periods, dict)
                or any(
                    periods.get(key) != source_periods.get(key)
                    for key in ("train_start", "train_end", "valid_start", "valid_end")
                )
            ):
                raise ValueError("daily inference changed the checkpoint training window")
            training_start = source["training_start"]
            training_end = source["training_end"]
            training_kind = "daily_inference"
        else:
            retrain_reason = str(training_evidence.get("retrain_reason") or "")
            training_start = observed_training_start
            training_end = observed_training_end
            training_kind = (
                "monthly_retrain"
                if retrain_reason == "monthly_first_trading_day"
                else "early_retrain"
            )
        persisted_training_evidence = {
            **training_evidence,
            "model_engine": model_engine,
            "model_data_contract": model_data_contract,
        }

        bounded_validity = min(max(int(valid_for_days), 1), 7)
        cutoff = datetime.combine(
            signal_date,
            time(15, 30),
            tzinfo=ZoneInfo("Asia/Shanghai"),
        ).astimezone(UTC)
        next_refit = cutoff + timedelta(days=1)
        artifact_key = f"live-refit-{signal_date:%Y%m%d}"
        try:
            existing = self.get_by_key(strategy_version_id, artifact_key)
        except KeyError:
            existing = None
        if existing is not None:
            if (
                (source_status == "retired" and existing.get("status") != "active")
                or
                existing.get("predictions_sha256") != predictions_sha256
                or existing.get("dataset_identity_sha256")
                != result.get("dataset_identity_sha256")
                or existing.get("execution_environment_sha256")
                != source.get("execution_environment_sha256")
                or existing.get("strategy_spec_sha256")
                != self.strategy_spec_sha256(strategy_version_id)
                or existing.get("model_recipe_sha256")
                != str(model_signal.get("model_recipe_sha256") or "")
                or existing.get("checkpoint_sha256") != checkpoint_sha256
                or existing.get("checkpoint_format") != checkpoint_format
                or existing.get("model_data_contract_sha256")
                != model_data_contract_sha256
                or existing.get("training_kind") != training_kind
                or existing.get("training_evidence") != persisted_training_evidence
            ):
                raise ValueError("model refit key already exists with different evidence")
            return existing
        if source_status != "active":
            raise ValueError("retired model source cannot create another live artifact")
        return self.create(
            strategy_version_id=strategy_version_id,
            artifact_key=artifact_key,
            model_recipe=dict(model_signal.get("recipe") or {}),
            dataset=str(result["dataset"]),
            dataset_identity_sha256=str(result["dataset_identity_sha256"]),
            execution_environment_sha256=str(
                execution["execution_environment_sha256"]
            ),
            training_start=training_start,
            training_end=training_end,
            data_cutoff_at=cutoff,
            scheduled_refit_at=next_refit,
            valid_until=cutoff + timedelta(days=bounded_validity),
            artifact_path=predictions_path,
            predictions_sha256=predictions_sha256,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_format=checkpoint_format,
            model_data_contract_sha256=model_data_contract_sha256,
            training_kind=training_kind,
            training_evidence=persisted_training_evidence,
            actor=actor,
            allow_dataset_rollover=True,
        )

    def activate(
        self,
        artifact_id: str,
        *,
        actor: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = now or _now()
        activator = actor.strip()
        if len(activator) < 2:
            raise ValueError("a responsible ModelArtifact activator is required")
        with self.engine.begin() as connection:
            identity = connection.execute(
                select(
                    model_artifacts.c.id,
                    model_artifacts.c.strategy_version_id,
                ).where(model_artifacts.c.id == artifact_id)
            ).first()
            if identity is None:
                raise KeyError(artifact_id)
            version = connection.execute(
                select(
                    strategy_versions.c.status,
                    strategy_versions.c.is_legacy,
                )
                .where(strategy_versions.c.id == identity.strategy_version_id)
                .with_for_update()
            ).first()
            if (
                version is None
                or str(version.status) != "approved"
                or bool(version.is_legacy)
            ):
                raise ValueError(
                    "ModelArtifact activation requires an approved non-legacy StrategySpec"
                )
            candidate = connection.execute(
                select(model_artifacts)
                .where(model_artifacts.c.id == artifact_id)
                .with_for_update()
            ).first()
            if candidate is None:
                raise KeyError(artifact_id)
            if str(candidate.status) not in {"candidate", "retired"}:
                raise ValueError("only candidate or retired ModelArtifacts may be activated")
            if candidate.valid_until <= current:
                connection.execute(
                    update(model_artifacts)
                    .where(model_artifacts.c.id == artifact_id)
                    .values(status="expired")
                )
                raise ValueError("ModelArtifact is expired")
            expected_spec = self.strategy_spec_sha256(str(candidate.strategy_version_id))
            if str(candidate.strategy_spec_sha256) != expected_spec:
                raise ValueError("ModelArtifact no longer matches its immutable StrategySpec")
            expected_environment = self._governed_execution_environment_sha256(
                str(candidate.strategy_version_id)
            )
            if (
                not _is_sha256(candidate.execution_environment_sha256)
                or str(candidate.execution_environment_sha256) != expected_environment
            ):
                raise ValueError(
                    "ModelArtifact execution environment no longer matches admission"
                )
            path = Path(str(candidate.artifact_path))
            if not path.is_file() or _file_sha256(path) != str(candidate.artifact_sha256):
                raise ValueError("ModelArtifact file failed immutable verification")
            self._verify_fitted_checkpoint(candidate)
            active = connection.execute(
                select(model_artifacts)
                .where(
                    model_artifacts.c.strategy_version_id
                    == candidate.strategy_version_id,
                    model_artifacts.c.status == "active",
                )
                .with_for_update()
            ).first()
            if active is not None:
                if str(active.model_recipe_sha256) != str(candidate.model_recipe_sha256):
                    raise ValueError(
                        "ModelArtifact activation would change the frozen model recipe"
                    )
                if candidate.data_cutoff_at <= active.data_cutoff_at:
                    raise ValueError(
                        "ModelArtifact activation would replace newer inference evidence"
                    )
                connection.execute(
                    update(model_artifacts)
                    .where(model_artifacts.c.id == active.id)
                    .values(status="retired", retired_at=current)
                )
            connection.execute(
                update(model_artifacts)
                .where(model_artifacts.c.id == artifact_id)
                .values(
                    status="active",
                    activated_by=activator,
                    activated_at=current,
                    retired_at=None,
                )
            )
        return self.get(artifact_id)

    def mark_failed(self, artifact_id: str, *, reason: str) -> dict[str, Any]:
        message = reason.strip()
        if len(message) < 5:
            raise ValueError("ModelArtifact failure reason is required")
        with self.engine.begin() as connection:
            result = connection.execute(
                update(model_artifacts)
                .where(
                    model_artifacts.c.id == artifact_id,
                    model_artifacts.c.status == "candidate",
                )
                .values(status="failed", failure_reason=message)
            )
            if not result.rowcount:
                raise ValueError("only candidate ModelArtifacts may fail")
        return self.get(artifact_id)

    def select_for_inference(
        self,
        strategy_version_id: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = now or _now()
        with self.engine.begin() as connection:
            active = connection.execute(
                select(model_artifacts)
                .where(
                    model_artifacts.c.strategy_version_id == strategy_version_id,
                    model_artifacts.c.status == "active",
                )
                .with_for_update()
            ).first()
            if active is None:
                return {
                    "status": "simple_baseline_required",
                    "reason": "no_active_model_artifact",
                    "contract_version": MODEL_ARTIFACT_CONTRACT_VERSION,
                }
            if active.valid_until <= current:
                return {
                    "status": "simple_baseline_required",
                    "reason": "active_model_artifact_expired",
                    "contract_version": MODEL_ARTIFACT_CONTRACT_VERSION,
                }
            expected_spec = self.strategy_spec_sha256(strategy_version_id)
            expected_environment = self._governed_execution_environment_sha256(
                strategy_version_id
            )
            if (
                str(active.strategy_spec_sha256) != expected_spec
                or not _is_sha256(active.execution_environment_sha256)
                or str(active.execution_environment_sha256) != expected_environment
            ):
                raise ValueError(
                    "active ModelArtifact no longer matches its StrategySpec or "
                    "execution environment"
                )
            self._verify_fitted_checkpoint(active)
            result = self._decode(active)
            result["selection_status"] = "active"
            result["contract_version"] = MODEL_ARTIFACT_CONTRACT_VERSION
            return result

    def require_for_inference(
        self,
        strategy_version_id: str,
        *,
        dataset_identity_sha256: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Return only an active, intact artifact for a model-signal strategy.

        A model strategy must never silently fall back to the old factor score
        or a simple baseline.  Refit/expiry therefore blocks the recommendation
        and paper-order job until a reviewed artifact is activated.
        """

        version = self.strategies.get_version(strategy_version_id)
        if str(version.get("config", {}).get("signal_source") or "factor_score") != (
            "model_prediction"
        ):
            raise ValueError("the StrategySpec does not use model predictions")
        selected = self.select_for_inference(strategy_version_id, now=now)
        if selected.get("selection_status") != "active":
            raise ValueError(
                "model-prediction strategy has no current active ModelArtifact; "
                "factor fallback is forbidden"
            )
        if str(selected.get("dataset_identity_sha256") or "") != str(
            dataset_identity_sha256 or ""
        ):
            raise ValueError("active ModelArtifact uses another immutable dataset")
        path = Path(str(selected["artifact_path"]))
        if not path.is_file() or _file_sha256(path) != str(selected["artifact_sha256"]):
            raise ValueError("active ModelArtifact failed immutable verification")
        verify_governed_checkpoint(
            Path(str(selected.get("checkpoint_path") or "")).resolve(),
            model_engine=str(
                (selected.get("training_evidence") or {}).get("model_engine") or ""
            ),
            checkpoint_format=str(selected.get("checkpoint_format") or ""),
            expected_sha256=str(selected.get("checkpoint_sha256") or ""),
        )
        return selected

    def get(self, artifact_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(model_artifacts).where(model_artifacts.c.id == artifact_id)
            ).first()
        if row is None:
            raise KeyError(artifact_id)
        return self._decode(row)

    def get_by_key(
        self,
        strategy_version_id: str,
        artifact_key: str,
    ) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(model_artifacts).where(
                    model_artifacts.c.strategy_version_id == strategy_version_id,
                    model_artifacts.c.artifact_key == artifact_key,
                )
            ).first()
        if row is None:
            raise KeyError(artifact_key)
        return self._decode(row)

    def list_for_strategy(self, strategy_version_id: str) -> list[dict[str, Any]]:
        self.strategies.get_version(strategy_version_id)
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(model_artifacts)
                .where(
                    model_artifacts.c.strategy_version_id == strategy_version_id
                )
                .order_by(model_artifacts.c.created_at.desc())
            ).all()
        return [self._decode(row) for row in rows]

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["model_recipe"] = result.pop("model_recipe_json")
        result["training_evidence"] = result.pop("training_evidence_json")
        result["contract_version"] = MODEL_ARTIFACT_CONTRACT_VERSION
        return result
