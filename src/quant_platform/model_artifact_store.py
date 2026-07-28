from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import model_artifacts, open_database, row_dict

from .strategy_store import StrategyStore

MODEL_ARTIFACT_CONTRACT_VERSION = "model-artifact-lifecycle-v1"


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


class ModelArtifactStore:
    """Immutable fitted-model artifacts under one frozen StrategySpec."""

    def __init__(self, database_url: str) -> None:
        self.strategies = StrategyStore(database_url)
        self.engine = open_database(database_url)

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

    def create(
        self,
        *,
        strategy_version_id: str,
        artifact_key: str,
        model_recipe: dict[str, Any],
        dataset: str,
        dataset_identity_sha256: str,
        training_start: date,
        training_end: date,
        data_cutoff_at: datetime,
        valid_until: datetime,
        artifact_path: str | Path,
        predictions_sha256: str,
        actor: str,
        scheduled_refit_at: datetime | None = None,
    ) -> dict[str, Any]:
        version = self.strategies.get_version(strategy_version_id)
        if version.get("is_legacy"):
            raise ValueError("legacy StrategySpec versions cannot own ModelArtifacts")
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
        if not _is_sha256(dataset_identity_sha256) or not _is_sha256(
            predictions_sha256
        ):
            raise ValueError("ModelArtifact requires immutable dataset and prediction hashes")
        creator = actor.strip()
        key = artifact_key.strip()
        if len(creator) < 2 or not key or not dataset.strip() or not model_recipe:
            raise ValueError("ModelArtifact identity, recipe, dataset and actor are required")
        path = Path(artifact_path).resolve()
        if not path.is_file():
            raise ValueError("model artifact file does not exist")
        artifact_sha256 = _file_sha256(path)
        spec_sha256 = self.strategy_spec_sha256(strategy_version_id)
        recipe_sha256 = _canonical_sha256(model_recipe)
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
                        created_by=creator,
                        created_at=now,
                    )
                )
        except IntegrityError as exc:
            raise ValueError(f"ModelArtifact key {key!r} already exists") from exc
        return self.get_by_key(strategy_version_id, key)

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
            path = Path(str(candidate.artifact_path))
            if not path.is_file() or _file_sha256(path) != str(candidate.artifact_sha256):
                raise ValueError("ModelArtifact file failed immutable verification")
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
                connection.execute(
                    update(model_artifacts)
                    .where(model_artifacts.c.id == active.id)
                    .values(status="expired")
                )
                return {
                    "status": "simple_baseline_required",
                    "reason": "active_model_artifact_expired",
                    "contract_version": MODEL_ARTIFACT_CONTRACT_VERSION,
                }
            result = self._decode(active)
            result["selection_status"] = "active"
            result["contract_version"] = MODEL_ARTIFACT_CONTRACT_VERSION
            return result

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
        result["contract_version"] = MODEL_ARTIFACT_CONTRACT_VERSION
        return result
