from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    backtest_runs,
    factor_candidates,
    factor_evaluations,
    open_database,
    row_dict,
    strategies,
    strategy_events,
    strategy_factors,
    strategy_pairs,
    strategy_versions,
)
from quant_platform.pair_trading import PairTradingConfig


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        ("benchmark", version.get("benchmark")),
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
    return failures


class StrategyStore:
    """Immutable strategy versions backed by promoted factors and audited approvals."""

    def __init__(self, database_url: str) -> None:
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
                or not str(evaluation.evaluator_version).startswith("factor-gate-v2")
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
    ) -> dict[str, Any]:
        if not factors:
            raise ValueError("a strategy must contain at least one promoted factor")
        if len({item["candidate_id"] for item in factors}) != len(factors):
            raise ValueError("factor candidates must be unique within a strategy version")
        total_weight = sum(abs(float(item["weight"])) for item in factors)
        if total_weight <= 0:
            raise ValueError("factor weights must not all be zero")
        strategy_id = uuid.uuid4().hex
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
                        benchmark=benchmark,
                        universe=universe,
                        config_json=config,
                        created_by=actor,
                        created_at=now,
                    )
                )
                connection.execute(
                    insert(strategy_factors),
                    [
                        {
                            "strategy_version_id": version_id,
                            "factor_candidate_id": item["candidate_id"],
                            "factor_evaluation_id": evaluation_evidence[item["candidate_id"]]["id"],
                            "weight": float(item["weight"]) / total_weight,
                            "direction": evaluation_evidence[item["candidate_id"]]["direction"],
                            "created_at": now,
                        }
                        for item in factors
                    ],
                )
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
        if not factors:
            raise ValueError("a strategy must contain at least one promoted factor")
        if len({item["candidate_id"] for item in factors}) != len(factors):
            raise ValueError("factor candidates must be unique within a strategy version")
        total_weight = sum(abs(float(item["weight"])) for item in factors)
        if total_weight <= 0:
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
                        benchmark=benchmark,
                        universe=universe,
                        config_json=config,
                        created_by=actor,
                        created_at=now,
                    )
                )
                connection.execute(
                    insert(strategy_factors),
                    [
                        {
                            "strategy_version_id": version_id,
                            "factor_candidate_id": item["candidate_id"],
                            "factor_evaluation_id": evaluation_evidence[item["candidate_id"]]["id"],
                            "weight": float(item["weight"]) / total_weight,
                            "direction": evaluation_evidence[item["candidate_id"]]["direction"],
                            "created_at": now,
                        }
                        for item in factors
                    ],
                )
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

    def get_version(self, version_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(strategy_versions).where(strategy_versions.c.id == version_id)
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
        return result

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
        with self.engine.begin() as connection:
            if version.get("strategy_type") == "multifactor":
                prior = connection.execute(
                    select(backtest_runs.c.id).where(
                        backtest_runs.c.strategy_version_id == version_id
                    )
                ).first()
                if prior is not None:
                    raise ValueError("a frozen strategy version may run the final test only once")
                factor_windows = connection.execute(
                    select(
                        factor_evaluations.c.dataset,
                        factor_evaluations.c.test_start,
                        factor_evaluations.c.test_end,
                        factor_evaluations.c.evaluator_version,
                    )
                    .join(
                        strategy_factors,
                        strategy_factors.c.factor_evaluation_id == factor_evaluations.c.id,
                    )
                    .where(strategy_factors.c.strategy_version_id == version_id)
                ).all()
                requested_start = date.fromisoformat(periods["start"])
                requested_end = date.fromisoformat(periods["end"])
                if not factor_windows or any(
                    item.dataset != dataset
                    or not str(item.evaluator_version).startswith("factor-gate-v2")
                    or requested_start != item.test_start
                    or requested_end != item.test_end
                    for item in factor_windows
                ):
                    raise ValueError(
                        "formal backtest must exactly match the reserved final-test window"
                    )
            connection.execute(
                insert(backtest_runs).values(
                    id=backtest_id,
                    strategy_version_id=version_id,
                    dataset=dataset,
                    execution_dataset=execution_dataset,
                    status="queued",
                    periods_json=periods,
                    artifact_path=str(artifact_path),
                    created_at=_now(),
                )
            )
        return self.get_backtest(backtest_id)

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
        """Validate immutable multifactor artifacts before a worker reports success."""

        backtest = self.get_backtest(backtest_id)
        version = self.get_version(backtest["strategy_version_id"])
        if version.get("strategy_type") != "multifactor":
            return
        failures = _multifactor_manifest_failures(version, backtest, metrics)
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
        provenance = metrics.get("provenance")
        if not isinstance(provenance, dict):
            failures.append("reproducible pair backtest provenance is required")
        else:
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
        failures = []
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
            if (
                not str(provenance.get("qlib_version") or "").strip()
                or provenance.get("qlib_version") == "unknown"
            ):
                failures.append("provenance qlib_version is required")
            if not str(provenance.get("backtest_engine_version") or "").strip():
                failures.append("backtest engine version is required")
            if not str(provenance.get("policy_version") or "").strip():
                failures.append("PortfolioPolicy provenance is required")
        if not isinstance(provenance, dict) or metrics.get("policy_version") != provenance.get(
            "policy_version"
        ):
            failures.append("PortfolioPolicy version is missing or inconsistent")
        if metrics.get("event_stress_passed") is not True:
            failures.append("event stress scenarios did not satisfy the configured result gate")
        if metrics.get("capacity_curve_passed") is not True:
            failures.append("capacity curve did not satisfy the configured result gate")
        cost_model = metrics.get("cost_model")
        if not isinstance(cost_model, dict):
            failures.append("the unified cost model is required for approval")
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
                elif evaluation.is_legacy or not str(evaluation.evaluator_version).startswith(
                    "factor-gate-v2"
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
