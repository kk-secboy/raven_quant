from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from math import isfinite
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import delete, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    backtest_runs,
    open_database,
    recommendation_holdings,
    recommendation_portfolios,
    recommendation_snapshots,
    row_dict,
    simulation_batches,
    simulation_cash_flows,
    simulation_events,
    simulation_fills,
    simulation_nav,
    simulation_orders,
    simulation_portfolios,
    simulation_positions,
    strategy_allocations,
    strategy_pairs,
    strategy_versions,
)
from quant_data.execution_contract import (
    require_daily_qlib_contract,
    require_minute_execution_contract,
    require_next_bar_execution,
    require_strategy_execution_contract,
)

from .cost_model import COST_SCHEDULE_VERSION, CostModelConfig, CostScheduleBook
from .execution_algorithms import execution_time_slots, normalize_execution_policy
from .member_risk_gate import (
    load_allocation_risk_state,
    load_strategy_risk_state,
)
from .qlib_workflow import require_qlib_workflow_identity
from .simulation_engine import (
    SIMULATION_ENGINE_VERSION,
    execute_atomic_pair_day,
    execute_simulation_day,
)


def _now() -> datetime:
    return datetime.now(UTC)


SIMULATION_SOURCE_TYPES = frozenset({"recommendation", "strategy_version", "allocation"})
SIMULATION_EXECUTION_ADAPTERS = frozenset({"long_only", "pair"})
SIMULATION_EXECUTION_FREQUENCIES = frozenset({"1min", "5min"})
SIMULATION_EXECUTION_SEMANTICS_VERSION = "simulation-execution-semantics-v1"
QLIB_ORDER_PLAN_FORMAT_VERSION = "qlib-order-plan-v1"
VWAP_PROFILE_METHOD = "qlib-historical-average-volume-v1"
SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_aware_timestamp(value: Any, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Qlib order-plan {field} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Qlib order-plan {field} must include a timezone")
    return parsed.astimezone(UTC)


def _pair_source_contract_hash(config: dict[str, Any]) -> str:
    return _canonical_hash(
        {
            "strategy_type": "pair",
            "signal_frequency": "day",
            "signal_horizon": "1d",
            "execution_frequency": "1min",
            "config": config,
        }
    )


def _simulation_semantics_payload(
    *,
    source_type: str,
    source_id: str,
    source_execution_contract_hash: str,
    execution_adapter: str,
    execution_frequency: str,
    daily_dataset: str,
    daily_dataset_identity_sha256: str,
    daily_dataset_lineage_id: str,
    execution_dataset: str,
    execution_dataset_identity_sha256: str,
    execution_dataset_lineage_id: str,
    execution_field_contract_version: str,
    execution_engine_version: str,
    execution_policy: dict[str, Any],
) -> dict[str, Any]:
    policy = dict(execution_policy)
    policy.pop("simulation_semantics_sha256", None)
    return {
        "version": SIMULATION_EXECUTION_SEMANTICS_VERSION,
        "source_type": source_type,
        "source_id": source_id,
        "source_execution_contract_hash": source_execution_contract_hash,
        "execution_adapter": execution_adapter,
        "execution_frequency": execution_frequency,
        "daily_dataset": daily_dataset,
        "daily_dataset_identity_sha256": daily_dataset_identity_sha256,
        "daily_dataset_lineage_id": daily_dataset_lineage_id,
        "execution_dataset": execution_dataset,
        "execution_dataset_identity_sha256": execution_dataset_identity_sha256,
        "execution_dataset_lineage_id": execution_dataset_lineage_id,
        "execution_field_contract_version": execution_field_contract_version,
        "execution_engine_version": execution_engine_version,
        "execution_policy": policy,
    }


class SimulationStore:
    """Transactional T+1 ledger for governed recommendation, strategy and allocation targets."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    @staticmethod
    def _governed_execution_policy(
        source: dict[str, Any],
        supplied: dict[str, Any],
        *,
        adapter: str,
    ) -> dict[str, Any]:
        provided = dict(supplied or {})
        if adapter != "long_only" or source.get("policy_mode") == "self_contained":
            return normalize_execution_policy(provided)
        config = dict(source.get("config") or {})
        method = str(source.get("execution_method") or "").lower()
        if method not in {"twap", "vwap", "next_bar"}:
            raise ValueError(
                "long-only simulation requires a minute execution method in its "
                "approved source contract"
            )
        governed = {
            "execution_algorithm": method,
            "slice_minutes": int(config.get("execution_slice_minutes") or 20),
            "max_slices": int(config.get("max_execution_slices") or 24),
            "max_participation": float(config.get("max_volume_participation") or 0.0),
            "volume_profile": None,
        }
        aliases = {
            "execution_algorithm": "execution_algorithm",
            "slice_minutes": "slice_minutes",
            "max_slices": "max_slices",
            "max_participation": "max_participation",
        }
        for supplied_key, governed_key in aliases.items():
            if supplied_key not in provided or provided[supplied_key] is None:
                continue
            observed = provided[supplied_key]
            expected = governed[governed_key]
            if isinstance(expected, float):
                matches = abs(float(observed) - expected) <= 1e-12
            elif isinstance(expected, int):
                matches = int(observed) == expected
            else:
                matches = str(observed).lower() == str(expected).lower()
            if not matches:
                raise ValueError(
                    f"simulation {supplied_key} must be derived from the approved "
                    "source contract"
                )
        if provided.get("volume_profile") is not None:
            raise ValueError(
                "simulation VWAP profile is derived from the bound Qlib execution "
                "dataset and cannot be supplied by an operator"
            )
        policy = normalize_execution_policy(governed)
        policy["volume_profile_method"] = (
            VWAP_PROFILE_METHOD if method == "vwap" else "none"
        )
        policy["volume_profile_lookback_days"] = (
            int(config.get("vwap_lookback_days") or 20) if method == "vwap" else 0
        )
        return policy

    @staticmethod
    def _bind_execution_semantics(
        *,
        source: dict[str, Any],
        source_type: str,
        source_id: str,
        adapter: str,
        frequency: str,
        daily_dataset: dict[str, Any],
        execution_dataset: dict[str, Any],
        policy: dict[str, Any],
        cost_model: CostModelConfig,
    ) -> dict[str, Any]:
        daily_provenance = dict(daily_dataset["provenance"])
        execution_provenance = dict(execution_dataset["provenance"])
        bound = {
            **policy,
            "simulation_contract_version": SIMULATION_EXECUTION_SEMANTICS_VERSION,
            "source_execution_contract_hash": str(source["execution_contract_hash"]),
            "execution_frequency": frequency,
            "cost_model": cost_model.to_dict(),
        }
        payload = _simulation_semantics_payload(
            source_type=source_type,
            source_id=source_id,
            source_execution_contract_hash=str(source["execution_contract_hash"]),
            execution_adapter=adapter,
            execution_frequency=frequency,
            daily_dataset=str(daily_dataset["name"]),
            daily_dataset_identity_sha256=str(
                daily_provenance["dataset_identity_sha256"]
            ),
            daily_dataset_lineage_id=str(daily_provenance["dataset_lineage_id"]),
            execution_dataset=str(execution_dataset["name"]),
            execution_dataset_identity_sha256=str(
                execution_provenance["dataset_identity_sha256"]
            ),
            execution_dataset_lineage_id=str(
                execution_provenance["dataset_lineage_id"]
            ),
            execution_field_contract_version=str(
                execution_provenance["execution_contract_version"]
            ),
            execution_engine_version=SIMULATION_ENGINE_VERSION,
            execution_policy=bound,
        )
        bound["simulation_semantics_sha256"] = _canonical_hash(payload)
        return bound

    def create(
        self,
        *,
        name: str,
        recommendation_portfolio_id: str | None = None,
        source_type: str | None = None,
        source_id: str | None = None,
        daily_dataset: dict[str, Any],
        execution_dataset: dict[str, Any],
        initial_cash: float,
        execution_policy: dict[str, Any],
        cost_schedule_version: str,
        actor: str,
        execution_adapter: str | None = None,
        execution_contract_hash: str | None = None,
    ) -> dict[str, Any]:
        if initial_cash < 100_000:
            raise ValueError("simulation initial cash must be at least 100000")
        if cost_schedule_version != COST_SCHEDULE_VERSION:
            raise ValueError("simulation cost schedule version is unavailable")
        daily_provenance = dict(daily_dataset.get("provenance") or {})
        execution_provenance = dict(execution_dataset.get("provenance") or {})
        require_daily_qlib_contract(daily_provenance)
        execution_frequency = str(execution_provenance.get("frequency") or "")
        if execution_frequency not in SIMULATION_EXECUTION_FREQUENCIES:
            raise ValueError("simulation execution frequency must be 1min or 5min")
        require_minute_execution_contract(
            execution_provenance,
            frequency=execution_frequency,
            simulation_eligible=True,
        )
        daily_source = str(daily_provenance.get("source_lineage_id") or "")
        execution_source = str(execution_provenance.get("source_lineage_id") or "")
        if len(daily_source) != 64 or daily_source != execution_source:
            raise ValueError("daily and minute datasets must share one verified source lineage")
        required_hashes = {
            "daily identity": daily_provenance.get("dataset_identity_sha256"),
            "daily lineage": daily_provenance.get("dataset_lineage_id"),
            "execution identity": execution_provenance.get("dataset_identity_sha256"),
            "execution lineage": execution_provenance.get("dataset_lineage_id"),
        }
        if any(len(str(value or "")) != 64 for value in required_hashes.values()):
            raise ValueError("simulation datasets require immutable identities and lineages")
        normalized_source_type = str(
            source_type or ("recommendation" if recommendation_portfolio_id else "")
        )
        normalized_source_id = str(source_id or recommendation_portfolio_id or "").strip()
        if normalized_source_type not in SIMULATION_SOURCE_TYPES or not normalized_source_id:
            raise ValueError("simulation source_type and source_id are required")
        with self.engine.connect() as connection:
            source = self._resolve_source(
                connection, normalized_source_type, normalized_source_id
            )
        if str(source["dataset"]) != str(daily_dataset.get("name") or ""):
            raise ValueError("simulation daily dataset must match the governed source")
        normalized_adapter = str(execution_adapter or source["execution_adapter"])
        if normalized_adapter not in SIMULATION_EXECUTION_ADAPTERS:
            raise ValueError("simulation execution adapter must be long_only or pair")
        if normalized_adapter != source["execution_adapter"]:
            raise ValueError("simulation adapter does not match the governed strategy source")
        governed_frequency = str(source.get("execution_frequency") or "")
        if governed_frequency and governed_frequency != execution_frequency:
            raise ValueError("simulation frequency does not match the governed strategy source")
        source_contract_hash = str(source.get("execution_contract_hash") or "")
        if not _is_sha256(source_contract_hash):
            raise ValueError(
                "simulation source uses a legacy or incomplete execution contract; "
                "create a new approved source"
            )
        contract_hash = source_contract_hash
        if execution_contract_hash and execution_contract_hash != contract_hash:
            raise ValueError("simulation execution contract does not match its source")
        policy = self._governed_execution_policy(
            source,
            execution_policy,
            adapter=normalized_adapter,
        )
        source_cost_model = CostModelConfig.from_mapping(source.get("config"))
        if cost_schedule_version != source_cost_model.version:
            raise ValueError("simulation cost schedule must match the approved source contract")
        policy = self._bind_execution_semantics(
            source=source,
            source_type=normalized_source_type,
            source_id=normalized_source_id,
            adapter=normalized_adapter,
            frequency=execution_frequency,
            daily_dataset=daily_dataset,
            execution_dataset=execution_dataset,
            policy=policy,
            cost_model=source_cost_model,
        )
        portfolio_id = uuid.uuid4().hex
        now = _now()
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(simulation_portfolios).values(
                        id=portfolio_id,
                        name=name.strip(),
                        recommendation_portfolio_id=(
                            normalized_source_id
                            if normalized_source_type == "recommendation"
                            else None
                        ),
                        source_type=normalized_source_type,
                        source_id=normalized_source_id,
                        status="paused",
                        base_currency="CNY",
                        initial_cash=Decimal(str(initial_cash)),
                        cash=Decimal(str(initial_cash)),
                        nav=Decimal(str(initial_cash)),
                        high_water_mark=Decimal(str(initial_cash)),
                        execution_algorithm=policy["execution_algorithm"],
                        execution_adapter=normalized_adapter,
                        execution_frequency=execution_frequency,
                        execution_contract_hash=contract_hash,
                        execution_dataset=str(execution_dataset["name"]),
                        daily_dataset=str(daily_dataset["name"]),
                        daily_dataset_identity_sha256=daily_provenance[
                            "dataset_identity_sha256"
                        ],
                        daily_dataset_lineage_id=daily_provenance["dataset_lineage_id"],
                        daily_field_contract_version=daily_provenance["field_contract_version"],
                        execution_dataset_identity_sha256=execution_provenance[
                            "dataset_identity_sha256"
                        ],
                        execution_dataset_lineage_id=execution_provenance["dataset_lineage_id"],
                        execution_field_contract_version=execution_provenance[
                            "execution_contract_version"
                        ],
                        execution_engine_version=SIMULATION_ENGINE_VERSION,
                        cost_schedule_version=source_cost_model.version,
                        execution_policy_json=policy,
                        created_by=actor.strip(),
                        created_at=now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    insert(simulation_cash_flows).values(
                        id=uuid.uuid4().hex,
                        portfolio_id=portfolio_id,
                        batch_id=None,
                        trade_date=now.date(),
                        flow_type="initial_deposit",
                        amount=Decimal(str(initial_cash)),
                        balance_after=Decimal(str(initial_cash)),
                        created_at=now,
                    )
                )
        except IntegrityError as exc:
            raise ValueError(
                "simulation name or source/execution dataset is already in use"
            ) from exc
        return self.get(portfolio_id)

    @staticmethod
    def _resolve_source(connection: Any, source_type: str, source_id: str) -> dict[str, Any]:
        if source_type == "recommendation":
            row = connection.execute(
                select(recommendation_portfolios).where(
                    recommendation_portfolios.c.id == source_id
                )
            ).first()
            if row is None:
                raise KeyError(source_id)
            if row.status != "active":
                raise ValueError("simulation requires an active recommendation portfolio")
            version = connection.execute(
                select(strategy_versions).where(
                    strategy_versions.c.id == row.strategy_version_id
                )
            ).one()
            if version.status != "approved" or version.is_legacy:
                raise ValueError(
                    "simulation recommendation must reference an approved non-legacy "
                    "strategy version"
                )
            version_config = dict(version.config_json or {})
            contract_hash = SimulationStore._validated_version_contract_hash(
                version, version_config
            )
            return {
                "dataset": str(row.dataset),
                "execution_adapter": "pair" if version.strategy_type == "pair" else "long_only",
                "execution_contract_hash": contract_hash,
                "signal_frequency": str(
                    version.signal_frequency
                    or version_config.get("signal_frequency")
                    or "day"
                ),
                "signal_horizon": str(version.signal_horizon or "1d"),
                "execution_frequency": str(
                    version.execution_frequency
                    or version_config.get("execution_frequency")
                    or ""
                ),
                "execution_method": str(version_config.get("execution_method") or ""),
                "config": version_config,
            }
        if source_type == "strategy_version":
            row = connection.execute(
                select(strategy_versions).where(strategy_versions.c.id == source_id)
            ).first()
            if row is None:
                raise KeyError(source_id)
            if row.status != "approved" or row.is_legacy:
                raise ValueError("simulation requires an approved non-legacy strategy version")
            backtest = connection.execute(
                select(backtest_runs)
                .where(
                    backtest_runs.c.strategy_version_id == source_id,
                    backtest_runs.c.status == "succeeded",
                    backtest_runs.c.is_legacy.is_(False),
                )
                .order_by(backtest_runs.c.finished_at.desc())
                .limit(1)
            ).first()
            if backtest is None:
                raise ValueError("strategy simulation requires a successful formal Qlib backtest")
            version_config = dict(row.config_json or {})
            contract_hash = SimulationStore._validated_version_contract_hash(
                row, version_config
            )
            if str(backtest.execution_contract_hash) != contract_hash:
                raise ValueError(
                    "strategy formal backtest contract does not match the approved version"
                )
            return {
                "dataset": str(backtest.dataset),
                "execution_adapter": "pair" if row.strategy_type == "pair" else "long_only",
                "execution_contract_hash": contract_hash,
                "signal_frequency": str(
                    row.signal_frequency
                    or version_config.get("signal_frequency")
                    or "day"
                ),
                "signal_horizon": str(row.signal_horizon or "1d"),
                "execution_frequency": str(
                    row.execution_frequency
                    or version_config.get("execution_frequency")
                    or ""
                ),
                "execution_method": str(version_config.get("execution_method") or ""),
                "config": version_config,
                "formal_backtest_id": str(backtest.id),
            }
        if source_type == "allocation":
            row = connection.execute(
                select(strategy_allocations).where(strategy_allocations.c.id == source_id)
            ).first()
            if row is None:
                raise KeyError(source_id)
            if row.status != "active" or row.is_legacy:
                raise ValueError("simulation requires an approved active allocation")
            return {
                "dataset": str(row.dataset),
                "execution_adapter": "long_only",
                "execution_contract_hash": _canonical_hash(
                    {
                        "source_type": "allocation",
                        "source_id": str(row.id),
                        "dataset": str(row.dataset),
                        "approval_simulation_evidence": dict(row.analysis_json or {}).get(
                            "approval_simulation_evidence"
                        ),
                    }
                ),
                "signal_frequency": "day",
                "signal_horizon": "1d",
                "execution_frequency": "",
                "execution_method": "",
                "config": CostModelConfig().to_dict(),
                "policy_mode": "self_contained",
            }
        raise ValueError("unsupported simulation source type")

    @staticmethod
    def _validated_version_contract_hash(
        version: Any, config: dict[str, Any]
    ) -> str:
        if str(version.strategy_type) == "pair":
            expected = _pair_source_contract_hash(config)
        else:
            require_strategy_execution_contract(config)
            expected = str(config["execution_contract_hash"])
        stored = str(version.execution_contract_hash or "")
        if not _is_sha256(stored) or stored != expected:
            raise ValueError(
                "approved strategy version execution contract is missing or inconsistent"
            )
        return stored

    @classmethod
    def _require_current_source_contract(cls, connection: Any, portfolio: Any) -> dict[str, Any]:
        source = cls._resolve_source(
            connection, str(portfolio.source_type), str(portfolio.source_id)
        )
        source_hash = str(source.get("execution_contract_hash") or "")
        if not _is_sha256(source_hash):
            raise ValueError(
                "simulation source uses a legacy or incomplete execution contract; "
                "create a new approved strategy version and simulation account"
            )
        if source_hash != str(portfolio.execution_contract_hash):
            raise ValueError("simulation execution contract no longer matches its governed source")
        policy = dict(portfolio.execution_policy_json or {})
        if (
            policy.get("simulation_contract_version")
            != SIMULATION_EXECUTION_SEMANTICS_VERSION
            or policy.get("source_execution_contract_hash") != source_hash
            or policy.get("execution_frequency") != str(portfolio.execution_frequency)
        ):
            raise ValueError("simulation execution semantics are incomplete or inconsistent")
        if str(policy.get("execution_algorithm") or "") != str(
            portfolio.execution_algorithm
        ):
            raise ValueError("simulation execution algorithm has drifted from its policy")
        try:
            cost_model = CostModelConfig.from_mapping(policy.get("cost_model"))
        except (TypeError, ValueError) as exc:
            raise ValueError("simulation cost contract is missing or invalid") from exc
        if cost_model.version != str(portfolio.cost_schedule_version):
            raise ValueError("simulation cost schedule has drifted from its policy")
        if str(portfolio.execution_adapter) == "long_only" and source.get(
            "policy_mode"
        ) != "self_contained":
            expected_policy = cls._governed_execution_policy(
                source, {}, adapter="long_only"
            )
            for field in (
                "execution_algorithm",
                "slice_minutes",
                "max_slices",
                "max_participation",
                "volume_profile_method",
                "volume_profile_lookback_days",
            ):
                if policy.get(field) != expected_policy.get(field):
                    raise ValueError(
                        "simulation execution policy no longer matches its approved "
                        f"source contract: {field}"
                    )
            governed_cost_model = CostModelConfig.from_mapping(
                source.get("config")
            ).to_dict()
            if cost_model.to_dict() != governed_cost_model:
                raise ValueError(
                    "simulation cost parameters no longer match the approved source contract"
                )
        semantics = _simulation_semantics_payload(
            source_type=str(portfolio.source_type),
            source_id=str(portfolio.source_id),
            source_execution_contract_hash=source_hash,
            execution_adapter=str(portfolio.execution_adapter),
            execution_frequency=str(portfolio.execution_frequency),
            daily_dataset=str(portfolio.daily_dataset),
            daily_dataset_identity_sha256=str(
                portfolio.daily_dataset_identity_sha256
            ),
            daily_dataset_lineage_id=str(portfolio.daily_dataset_lineage_id),
            execution_dataset=str(portfolio.execution_dataset),
            execution_dataset_identity_sha256=str(
                portfolio.execution_dataset_identity_sha256
            ),
            execution_dataset_lineage_id=str(
                portfolio.execution_dataset_lineage_id
            ),
            execution_field_contract_version=str(
                portfolio.execution_field_contract_version
            ),
            execution_engine_version=str(portfolio.execution_engine_version),
            execution_policy=policy,
        )
        if str(policy.get("simulation_semantics_sha256") or "") != _canonical_hash(
            semantics
        ):
            raise ValueError("simulation execution semantics failed immutable verification")
        return source

    def set_status(self, portfolio_id: str, status: str) -> dict[str, Any]:
        if status not in {"active", "paused"}:
            raise ValueError("simulation status must be active or paused")
        with self.engine.begin() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .with_for_update()
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            if status == "active":
                self._require_current_source_contract(connection, portfolio)
            result = connection.execute(
                update(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .values(status=status, updated_at=_now())
            )
            if not result.rowcount:
                raise KeyError(portfolio_id)
        return self.get(portfolio_id)

    def create_batch_for_snapshot(
        self,
        snapshot_id: str,
        *,
        actor: str = "recommendation-worker",
    ) -> tuple[dict[str, Any] | None, bool]:
        batches = self.create_batches_for_snapshot(snapshot_id, actor=actor)
        return batches[0] if batches else (None, False)

    def create_batches_for_snapshot(
        self,
        snapshot_id: str,
        *,
        actor: str = "recommendation-worker",
    ) -> list[tuple[dict[str, Any], bool]]:
        producer = actor.strip()
        if len(producer) < 2:
            raise ValueError("simulation batch producer is required")
        now = _now()
        batch_refs: list[tuple[str, bool]] = []
        with self.engine.begin() as connection:
            snapshot = connection.execute(
                select(recommendation_snapshots).where(
                    recommendation_snapshots.c.id == snapshot_id
                )
            ).first()
            if snapshot is None:
                raise KeyError(snapshot_id)
            if snapshot.status != "succeeded" or snapshot.effective_date is None:
                raise ValueError("simulation batch requires a successful recommendation snapshot")
            portfolios = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.source_type == "recommendation",
                    simulation_portfolios.c.source_id == snapshot.portfolio_id,
                    simulation_portfolios.c.status == "active",
                )
                .order_by(simulation_portfolios.c.created_at)
            ).all()
            for portfolio in portfolios:
                self._require_current_source_contract(connection, portfolio)
                if (
                    str(snapshot.dataset) != str(portfolio.daily_dataset)
                    or str(snapshot.dataset_identity_sha256)
                    != str(portfolio.daily_dataset_identity_sha256)
                ):
                    raise ValueError("recommendation snapshot lineage does not match simulation")
                batch_id = uuid.uuid4().hex
                inserted_id = connection.scalar(
                    pg_insert(simulation_batches)
                    .values(
                        id=batch_id,
                        portfolio_id=portfolio.id,
                        recommendation_snapshot_id=snapshot_id,
                        source_snapshot_id=snapshot_id,
                        target_payload_json=None,
                        execution_adapter=portfolio.execution_adapter,
                        execution_contract_hash=portfolio.execution_contract_hash,
                        signal_date=snapshot.as_of_date,
                        trade_date=snapshot.effective_date,
                        status="queued",
                        idempotency_key=f"simulation:{portfolio.id}:{snapshot_id}",
                        created_by=producer,
                        created_at=now,
                    )
                    .on_conflict_do_nothing(
                        index_elements=[simulation_batches.c.idempotency_key]
                    )
                    .returning(simulation_batches.c.id)
                )
                if inserted_id is None:
                    existing_id = connection.scalar(
                        select(simulation_batches.c.id).where(
                            simulation_batches.c.idempotency_key
                            == f"simulation:{portfolio.id}:{snapshot_id}"
                        )
                    )
                    if existing_id is None:
                        raise RuntimeError("simulation batch idempotency lookup failed")
                    batch_refs.append((str(existing_id), False))
                else:
                    batch_refs.append((batch_id, True))
        return [(self.get_batch(batch_id), created) for batch_id, created in batch_refs]

    def create_batch_for_targets(
        self,
        portfolio_id: str,
        *,
        source_snapshot_id: str,
        signal_date: date,
        trade_date: date,
        target_payload: dict[str, Any],
        execution_contract_hash: str,
        idempotency_key: str,
        actor: str = "simulation-operator",
    ) -> tuple[dict[str, Any], bool]:
        """Reject the retired client-authored long-only target path."""

        del (
            portfolio_id,
            source_snapshot_id,
            signal_date,
            trade_date,
            target_payload,
            execution_contract_hash,
            idempotency_key,
            actor,
        )
        raise ValueError(
            "direct simulation target payloads are forbidden; use an immutable "
            "Qlib order-plan artifact or the recommendation snapshot path"
        )

    def create_batch_from_order_plan(
        self,
        portfolio_id: str,
        *,
        order_plan_manifest_sha256: str,
        data_root: Path,
        actor: str,
    ) -> tuple[dict[str, Any], bool]:
        """Queue one long-only replay from an immutable Qlib order-plan artifact."""

        manifest_sha256 = str(order_plan_manifest_sha256 or "").lower()
        if not _is_sha256(manifest_sha256):
            raise ValueError("Qlib order-plan manifest requires a SHA-256 identity")
        producer = actor.strip()
        if len(producer) < 2:
            raise ValueError("simulation batch producer is required")
        artifact_root = (
            Path(data_root) / "artifacts" / "order-plans" / manifest_sha256
        ).resolve()
        allowed_root = (Path(data_root) / "artifacts" / "order-plans").resolve()
        try:
            artifact_root.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError("Qlib order-plan artifact path is unsafe") from exc
        manifest_path = artifact_root / "manifest.json"
        target_path = artifact_root / "target_weights.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            target_payload = json.loads(target_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Qlib order-plan artifact is missing or invalid") from exc
        if not isinstance(manifest, dict) or not isinstance(target_payload, dict):
            raise ValueError("Qlib order-plan artifacts must be JSON objects")
        if _sha256_file(manifest_path) != manifest_sha256:
            raise ValueError("Qlib order-plan manifest failed immutable verification")
        target_file_sha256 = str(manifest.get("target_weights_file_sha256") or "")
        if not _is_sha256(target_file_sha256) or _sha256_file(
            target_path
        ) != target_file_sha256:
            raise ValueError("Qlib order-plan target weights failed immutable verification")
        normalized_targets = self._normalize_target_payload(
            target_payload, adapter="long_only"
        )
        target_weights_sha256 = _canonical_hash(normalized_targets)
        if manifest.get("target_weights_sha256") != target_weights_sha256:
            raise ValueError("Qlib order-plan normalized targets do not match its manifest")
        if manifest.get("format_version") != QLIB_ORDER_PLAN_FORMAT_VERSION:
            raise ValueError("Qlib order-plan artifact format is unsupported")
        if manifest.get("produced_by") != "qlib-workflow-recorder":
            raise ValueError("Qlib order-plan was not produced by the governed research path")
        require_qlib_workflow_identity(manifest.get("qlib_workflow"))
        try:
            signal_date = date.fromisoformat(str(manifest["signal_date"]))
            trade_date = date.fromisoformat(str(manifest["trade_date"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Qlib order-plan dates are missing or invalid") from exc
        source_snapshot = manifest.get("source_snapshot")
        if not isinstance(source_snapshot, dict):
            raise ValueError("Qlib order-plan source snapshot is missing")
        source_snapshot_id = str(source_snapshot.get("id") or "")
        if not _is_sha256(source_snapshot_id):
            raise ValueError("Qlib order-plan source snapshot identity is invalid")
        now = _now()
        with self.engine.begin() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .with_for_update()
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            if portfolio.status != "active":
                raise ValueError("simulation portfolio is not active")
            self._require_current_source_contract(connection, portfolio)
            if str(portfolio.source_type) == "recommendation":
                raise ValueError(
                    "recommendation simulations are queued only from successful "
                    "recommendation snapshots"
                )
            if str(portfolio.source_type) == "allocation":
                raise ValueError(
                    "allocation simulation NAV is derived from certified member simulations"
                )
            if str(portfolio.execution_adapter) != "long_only":
                raise ValueError(
                    "pair simulation batches must be derived from an approved immutable "
                    "formal backtest artifact"
                )
            version = connection.execute(
                select(strategy_versions).where(
                    strategy_versions.c.id == portfolio.source_id
                )
            ).one()
            signal_at, execution_not_before = self._validate_order_plan_timing(
                manifest=manifest,
                version=version,
                portfolio=portfolio,
                signal_date=signal_date,
                trade_date=trade_date,
            )
            backtest_id = str(manifest.get("formal_backtest_id") or "")
            backtest = connection.execute(
                select(backtest_runs).where(backtest_runs.c.id == backtest_id)
            ).first()
            if (
                backtest is None
                or str(backtest.strategy_version_id) != str(version.id)
                or backtest.status != "succeeded"
                or backtest.is_legacy
            ):
                raise ValueError(
                    "Qlib order-plan does not reference the approved source formal backtest"
                )
            required_matches = (
                (manifest.get("source_type"), "strategy_version"),
                (manifest.get("source_id"), str(version.id)),
                (
                    manifest.get("execution_contract_hash"),
                    str(portfolio.execution_contract_hash),
                ),
                (manifest.get("daily_dataset"), str(portfolio.daily_dataset)),
                (
                    source_snapshot.get("dataset_identity_sha256"),
                    str(portfolio.daily_dataset_identity_sha256),
                ),
                (
                    source_snapshot.get("dataset_lineage_id"),
                    str(portfolio.daily_dataset_lineage_id),
                ),
                (
                    source_snapshot_id,
                    str(portfolio.daily_dataset_identity_sha256),
                ),
            )
            if any(observed != expected for observed, expected in required_matches):
                raise ValueError(
                    "Qlib order-plan does not match the simulation source contract "
                    "or immutable snapshot"
                )
            plan = {
                "format_version": QLIB_ORDER_PLAN_FORMAT_VERSION,
                "manifest_sha256": manifest_sha256,
                "target_weights_file_sha256": target_file_sha256,
                "target_weights_sha256": target_weights_sha256,
                "formal_backtest_id": backtest_id,
                "source_snapshot": source_snapshot,
                "execution_contract_hash": str(portfolio.execution_contract_hash),
                "signal_at": signal_at.isoformat() if signal_at else None,
                "execution_not_before": (
                    execution_not_before.isoformat()
                    if execution_not_before
                    else None
                ),
                "signal_snapshot": manifest.get("signal_snapshot"),
                "qlib_workflow": require_qlib_workflow_identity(
                    manifest.get("qlib_workflow")
                ),
            }
            payload = {
                **normalized_targets,
                "governed_order_plan": plan,
            }
            batch_id = uuid.uuid4().hex
            idempotency_key = (
                f"qlib-order-plan:{portfolio.id}:{manifest_sha256}"
            )
            inserted_id = connection.scalar(
                pg_insert(simulation_batches)
                .values(
                    id=batch_id,
                    portfolio_id=portfolio.id,
                    recommendation_snapshot_id=None,
                    source_snapshot_id=source_snapshot_id,
                    target_payload_json=payload,
                    execution_adapter="long_only",
                    execution_contract_hash=portfolio.execution_contract_hash,
                    signal_date=signal_date,
                    trade_date=trade_date,
                    signal_at=signal_at,
                    execution_not_before=execution_not_before,
                    status="queued",
                    idempotency_key=idempotency_key,
                    created_by=producer,
                    created_at=now,
                )
                .on_conflict_do_nothing(
                    index_elements=[simulation_batches.c.idempotency_key]
                )
                .returning(simulation_batches.c.id)
            )
            if inserted_id is None:
                existing = connection.execute(
                    select(simulation_batches).where(
                        simulation_batches.c.idempotency_key == idempotency_key
                    )
                ).one()
                if (
                    str(existing.portfolio_id) != str(portfolio.id)
                    or str(existing.source_snapshot_id) != source_snapshot_id
                    or dict(existing.target_payload_json or {}) != payload
                    or str(existing.execution_contract_hash)
                    != str(portfolio.execution_contract_hash)
                    or existing.signal_at != signal_at
                    or existing.execution_not_before != execution_not_before
                ):
                    raise ValueError(
                        "Qlib order-plan idempotency identity is already bound "
                        "to different targets"
                    )
                return self._batch_dict(existing), False
        return self.get_batch(batch_id), True

    @staticmethod
    def _validate_order_plan_timing(
        *,
        manifest: dict[str, Any],
        version: Any,
        portfolio: Any,
        signal_date: date,
        trade_date: date,
    ) -> tuple[datetime | None, datetime | None]:
        signal_frequency = str(
            version.signal_frequency
            or dict(version.config_json or {}).get("signal_frequency")
            or "day"
        ).lower()
        raw_signal_at = manifest.get("signal_at")
        raw_execution_not_before = manifest.get("execution_not_before")
        if signal_frequency == "day":
            if raw_signal_at is not None or raw_execution_not_before is not None:
                raise ValueError("daily Qlib order-plans must not contain intraday timestamps")
            if trade_date <= signal_date:
                raise ValueError("daily Qlib order-plan violates next-session execution")
            return None, None
        if raw_signal_at is None or raw_execution_not_before is None:
            raise ValueError(
                "minute Qlib order-plans require signal_at and execution_not_before"
            )
        signal_snapshot = manifest.get("signal_snapshot")
        if not isinstance(signal_snapshot, dict):
            raise ValueError("minute Qlib order-plan signal snapshot is missing")
        if (
            str(signal_snapshot.get("frequency") or "") != signal_frequency
            or not str(signal_snapshot.get("name") or "")
            or any(
                not _is_sha256(signal_snapshot.get(field))
                for field in (
                    "dataset_identity_sha256",
                    "dataset_lineage_id",
                    "source_lineage_id",
                )
            )
        ):
            raise ValueError("minute Qlib order-plan signal snapshot is invalid")
        signal_at = _parse_aware_timestamp(raw_signal_at, field="signal_at")
        execution_not_before = _parse_aware_timestamp(
            raw_execution_not_before,
            field="execution_not_before",
        )
        if signal_at.astimezone(SHANGHAI_TIMEZONE).date() != signal_date:
            raise ValueError("Qlib order-plan signal_at does not match signal_date")
        if execution_not_before.astimezone(SHANGHAI_TIMEZONE).date() != trade_date:
            raise ValueError(
                "Qlib order-plan execution_not_before does not match trade_date"
            )
        policy = dict(portfolio.execution_policy_json or {})
        if str(policy.get("execution_algorithm") or "") != "next_bar":
            raise ValueError("minute Qlib order-plans require next_bar execution")
        execution_frequency = str(portfolio.execution_frequency)
        require_next_bar_execution(
            signal_at,
            execution_not_before,
            signal_frequency=signal_frequency,
            execution_frequency=execution_frequency,
        )
        expected = execution_time_slots(
            trade_date=trade_date,
            policy=policy,
            signal_at=signal_at,
        )[0].astimezone(UTC)
        if execution_not_before != expected:
            raise ValueError(
                "Qlib order-plan execution_not_before is not the first eligible bar"
            )
        return signal_at, execution_not_before

    def create_pair_batch_from_backtest(
        self,
        portfolio_id: str,
        *,
        backtest_id: str,
        trade_date: date,
        data_root: Path,
        actor: str,
    ) -> tuple[dict[str, Any], bool]:
        """Derive an atomic pair target from one immutable approved formal backtest trade."""

        producer = actor.strip()
        if len(producer) < 2:
            raise ValueError("simulation batch producer is required")
        now = _now()
        with self.engine.begin() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .with_for_update()
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            if portfolio.status != "active":
                raise ValueError("simulation portfolio is not active")
            if (
                str(portfolio.source_type) != "strategy_version"
                or str(portfolio.execution_adapter) != "pair"
            ):
                raise ValueError(
                    "pair replay requires a pair simulation sourced from a strategy version"
                )
            self._require_current_source_contract(connection, portfolio)
            version = connection.execute(
                select(strategy_versions).where(
                    strategy_versions.c.id == portfolio.source_id
                )
            ).one()
            if version.status != "approved" or version.is_legacy:
                raise ValueError("pair replay requires an approved non-legacy strategy version")
            backtest = connection.execute(
                select(backtest_runs).where(backtest_runs.c.id == backtest_id)
            ).first()
            if backtest is None:
                raise KeyError(backtest_id)
            if (
                str(backtest.strategy_version_id) != str(version.id)
                or backtest.status != "succeeded"
                or backtest.is_legacy
            ):
                raise ValueError(
                    "pair replay requires a successful formal backtest belonging to "
                    "the approved source version"
                )
            if str(backtest.execution_contract_hash) != str(
                portfolio.execution_contract_hash
            ):
                raise ValueError("pair backtest execution contract does not match simulation")
            pair = connection.execute(
                select(strategy_pairs).where(
                    strategy_pairs.c.strategy_version_id == version.id
                )
            ).first()
            if pair is None:
                raise ValueError("approved pair strategy has no immutable pair definition")
            target_payload, signal_date, plan = self._derive_pair_replay_target(
                portfolio=portfolio,
                version=version,
                pair=pair,
                backtest=backtest,
                trade_date=trade_date,
                data_root=data_root,
            )
            batch_id = uuid.uuid4().hex
            idempotency_key = (
                f"pair-replay:{portfolio.id}:{backtest.id}:{trade_date.isoformat()}:"
                f"{plan['pair_plan_sha256']}"
            )
            inserted_id = connection.scalar(
                pg_insert(simulation_batches)
                .values(
                    id=batch_id,
                    portfolio_id=portfolio.id,
                    recommendation_snapshot_id=None,
                    source_snapshot_id=plan["pair_plan_sha256"],
                    target_payload_json={
                        **target_payload,
                        "governed_pair_plan": plan,
                    },
                    execution_adapter="pair",
                    execution_contract_hash=portfolio.execution_contract_hash,
                    signal_date=signal_date,
                    trade_date=trade_date,
                    status="queued",
                    idempotency_key=idempotency_key,
                    created_by=producer,
                    created_at=now,
                )
                .on_conflict_do_nothing(
                    index_elements=[simulation_batches.c.idempotency_key]
                )
                .returning(simulation_batches.c.id)
            )
            if inserted_id is None:
                existing_id = connection.scalar(
                    select(simulation_batches.c.id).where(
                        simulation_batches.c.idempotency_key == idempotency_key
                    )
                )
                if existing_id is None:
                    raise RuntimeError("pair replay idempotency lookup failed")
                return self.get_batch(str(existing_id)), False
        return self.get_batch(batch_id), True

    @classmethod
    def _derive_pair_replay_target(
        cls,
        *,
        portfolio: Any,
        version: Any,
        pair: Any,
        backtest: Any,
        trade_date: date,
        data_root: Path,
    ) -> tuple[dict[str, Any], date, dict[str, Any]]:
        artifact_root = Path(str(backtest.artifact_path)).resolve()
        allowed_root = (Path(data_root) / "artifacts" / "backtests").resolve()
        try:
            artifact_root.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError(
                "pair backtest artifact is outside the governed artifact root"
            ) from exc
        main_manifest_path = artifact_root / "manifest.json"
        pair_manifest_path = artifact_root / "pair_artifact_manifest.json"
        try:
            main_manifest = json.loads(main_manifest_path.read_text(encoding="utf-8"))
            pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("pair replay artifact manifest is missing or invalid") from exc
        if not isinstance(main_manifest, dict) or not isinstance(pair_manifest, dict):
            raise ValueError("pair replay artifact manifests must be JSON objects")
        metrics = dict(backtest.metrics_json or {})
        provenance = metrics.get("provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        if (
            provenance.get("execution_manifest_sha256") != _sha256_file(main_manifest_path)
            or provenance.get("pair_artifact_manifest_sha256")
            != _sha256_file(pair_manifest_path)
        ):
            raise ValueError("pair replay artifact manifest does not match backtest provenance")
        expected_pair = {
            "leg_y": str(pair.leg_y),
            "leg_x": str(pair.leg_x),
            "asset_class": str(pair.asset_class),
            "shorting_mode": str(pair.shorting_mode),
        }
        main_pair = {
            key: dict(main_manifest.get("pair") or {}).get(key)
            for key in expected_pair
        }
        artifact_pair = {
            key: dict(pair_manifest.get("pair") or {}).get(key)
            for key in expected_pair
        }
        expected_config_sha256 = _canonical_hash(dict(version.config_json or {}))
        required_matches = (
            (main_manifest.get("backtest_id"), str(backtest.id)),
            (main_manifest.get("strategy_version_id"), str(version.id)),
            (main_manifest.get("execution_contract_hash"), str(version.execution_contract_hash)),
            (main_manifest.get("dataset"), str(backtest.dataset)),
            (main_manifest.get("periods"), dict(backtest.periods_json or {})),
            (main_pair, expected_pair),
            (_canonical_hash(dict(main_manifest.get("config") or {})), expected_config_sha256),
            (pair_manifest.get("format_version"), "pair-replay-artifact-v1"),
            (pair_manifest.get("backtest_id"), str(backtest.id)),
            (pair_manifest.get("strategy_version_id"), str(version.id)),
            (
                pair_manifest.get("execution_contract_hash"),
                str(version.execution_contract_hash),
            ),
            (pair_manifest.get("dataset"), str(backtest.dataset)),
            (pair_manifest.get("periods"), dict(backtest.periods_json or {})),
            (artifact_pair, expected_pair),
            (pair_manifest.get("strategy_config_sha256"), expected_config_sha256),
        )
        if any(observed != expected for observed, expected in required_matches):
            raise ValueError(
                "pair replay artifact does not belong to the approved strategy/backtest contract"
            )
        files = pair_manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("pair replay artifact file manifest is missing")
        for name in (
            "daily_returns.parquet",
            "daily_ledger.parquet",
            "kalman_spread.parquet",
            "trades.json",
            "rejections.json",
        ):
            evidence = files.get(name)
            path = (artifact_root / name).resolve()
            try:
                path.relative_to(artifact_root)
            except ValueError as exc:
                raise ValueError("pair replay artifact path is unsafe") from exc
            if (
                not isinstance(evidence, dict)
                or not path.is_file()
                or path.stat().st_size != int(evidence.get("bytes") or -1)
                or _sha256_file(path) != str(evidence.get("sha256") or "")
            ):
                raise ValueError(f"pair replay artifact {name} failed immutable verification")
        try:
            trades = json.loads((artifact_root / "trades.json").read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("pair replay trades artifact is invalid") from exc
        matches = [
            item
            for item in trades
            if isinstance(item, dict)
            and str(item.get("trade_date") or "")[:10] == trade_date.isoformat()
        ]
        if len(matches) != 1:
            raise ValueError(
                "selected trade date must identify exactly one governed pair backtest trade"
            )
        trade = matches[0]
        try:
            signal_date = date.fromisoformat(str(trade["signal_date"])[:10])
            direction = int(trade["direction"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("governed pair trade has invalid signal metadata") from exc
        if signal_date >= trade_date or direction not in {-1, 1}:
            raise ValueError("governed pair trade violates next-session execution")
        ledger = pd.read_parquet(artifact_root / "daily_ledger.parquet").reset_index()
        datetime_field = next(
            (name for name in ("datetime", "trade_date", "date") if name in ledger),
            None,
        )
        if datetime_field is None:
            raise ValueError("pair replay daily ledger has no trade date")
        ledger[datetime_field] = pd.to_datetime(ledger[datetime_field], errors="coerce")
        rows = ledger[ledger[datetime_field].dt.date == trade_date]
        if len(rows) != 1 or not {"quantity_y", "quantity_x"}.issubset(rows.columns):
            raise ValueError("pair replay daily ledger has no unique governed target")
        row = rows.iloc[0]
        artifact_quantities = {
            str(pair.leg_y): int(row["quantity_y"]),
            str(pair.leg_x): int(row["quantity_x"]),
        }
        if any(value and (value > 0) != (direction > 0) for value in [
            artifact_quantities[str(pair.leg_y)]
        ]):
            raise ValueError("pair replay ledger direction does not match the governed trade")
        if any(value and (value < 0) != (direction > 0) for value in [
            artifact_quantities[str(pair.leg_x)]
        ]):
            raise ValueError("pair replay ledger hedge direction does not match the governed trade")
        config = dict(version.config_json or {})
        reference_capital = float(config.get("initial_capital") or 0.0)
        if reference_capital <= 0:
            raise ValueError("pair strategy has no governed reference capital")
        scale = float(portfolio.nav) / reference_capital
        scaled = {
            instrument: int(abs(quantity) * scale // 100) * 100
            for instrument, quantity in artifact_quantities.items()
        }
        action = str(trade.get("action") or "")
        if action not in {"entry", "exit"}:
            raise ValueError("pair replay artifact contains an unsupported trade action")
        if action == "entry" and (not all(scaled.values())):
            raise ValueError("simulation capital is too small for the governed pair board lots")
        if action == "exit" and any(scaled.values()):
            raise ValueError("governed pair exit artifact must target zero quantities")
        annual_borrow_rate = float(config.get("annual_borrow_rate") or 0.0)
        if not 0 < annual_borrow_rate <= 1:
            raise ValueError("pair strategy has no governed annual borrow rate")
        sides = {
            str(pair.leg_y): "long" if direction > 0 else "short",
            str(pair.leg_x): "short" if direction > 0 else "long",
        }
        pair_artifact_sha256 = _sha256_file(pair_manifest_path)
        plan_identity = {
            "format_version": "governed-pair-plan-v1",
            "portfolio_id": str(portfolio.id),
            "portfolio_nav": float(portfolio.nav),
            "backtest_id": str(backtest.id),
            "strategy_version_id": str(version.id),
            "trade_date": trade_date.isoformat(),
            "signal_date": signal_date.isoformat(),
            "action": action,
            "direction": direction,
            "execution_contract_hash": str(version.execution_contract_hash),
            "pair_artifact_manifest_sha256": pair_artifact_sha256,
            "trade_sha256": _canonical_hash(trade),
            "execution_snapshot": pair_manifest.get("execution_snapshot"),
            "minute_dataset": pair_manifest.get("minute_dataset"),
            "shortability_dataset": pair_manifest.get("shortability_dataset"),
        }
        plan = {
            **plan_identity,
            "pair_plan_sha256": _canonical_hash(plan_identity),
        }
        group_id = f"pair-{plan['pair_plan_sha256'][:24]}"
        legs = [
            {
                "instrument": instrument,
                "leg_no": leg_no,
                "position_side": sides[instrument],
                "target_quantity": scaled[instrument],
                "annual_borrow_rate": (
                    annual_borrow_rate if sides[instrument] == "short" else 0.0
                ),
            }
            for leg_no, instrument in enumerate(
                (str(pair.leg_y), str(pair.leg_x)), start=1
            )
        ]
        return cls._normalize_target_payload(
            {"atomic_group_id": group_id, "legs": legs}, adapter="pair"
        ), signal_date, plan

    @staticmethod
    def _normalize_target_payload(
        target_payload: dict[str, Any], *, adapter: str
    ) -> dict[str, Any]:
        payload = dict(target_payload or {})
        if adapter == "long_only":
            values = payload.get("target_weights")
            if not isinstance(values, dict) or not values:
                raise ValueError("long-only simulation requires target_weights")
            targets = {str(key).upper(): float(value) for key, value in values.items()}
            if any(not isfinite(value) or value < 0 for value in targets.values()):
                raise ValueError("simulation target weights must be finite and non-negative")
            if sum(targets.values()) > 1.0 + 1e-8:
                raise ValueError("simulation target weights exceed one")
            return {"target_weights": dict(sorted(targets.items()))}
        if adapter != "pair":
            raise ValueError("unsupported simulation execution adapter")
        group_id = str(payload.get("atomic_group_id") or "").strip()
        legs = payload.get("legs")
        if not group_id or not isinstance(legs, list) or len(legs) != 2:
            raise ValueError("pair simulation requires one atomic group with exactly two legs")
        normalized: list[dict[str, Any]] = []
        for item in legs:
            leg = dict(item or {})
            quantity = int(leg.get("target_quantity") or 0)
            rate = float(leg.get("annual_borrow_rate") or 0.0)
            normalized.append(
                {
                    "instrument": str(leg.get("instrument") or "").strip().upper(),
                    "leg_no": int(leg.get("leg_no") or 0),
                    "position_side": str(leg.get("position_side") or "").strip(),
                    "target_quantity": quantity,
                    "annual_borrow_rate": rate,
                }
            )
        if {item["leg_no"] for item in normalized} != {1, 2}:
            raise ValueError("pair simulation leg numbers must be 1 and 2")
        if len({item["instrument"] for item in normalized}) != 2 or any(
            not item["instrument"] for item in normalized
        ):
            raise ValueError("pair simulation instruments must be distinct")
        if {item["position_side"] for item in normalized} != {"long", "short"}:
            raise ValueError("pair simulation requires one long and one short leg")
        if any(
            item["target_quantity"] < 0 or item["target_quantity"] % 100
            for item in normalized
        ):
            raise ValueError("pair target quantities must be non-negative board lots")
        short_leg = next(item for item in normalized if item["position_side"] == "short")
        if short_leg["target_quantity"] > 0 and not 0 < short_leg["annual_borrow_rate"] <= 1:
            raise ValueError("pair short leg requires a positive governed borrow rate")
        return {
            "atomic_group_id": group_id,
            "legs": sorted(normalized, key=lambda item: item["leg_no"]),
        }

    @staticmethod
    def _source_strategy_version_id(connection: Any, portfolio: Any) -> str | None:
        source_type = str(portfolio.source_type)
        if source_type == "strategy_version":
            return str(portfolio.source_id)
        if source_type != "recommendation":
            return None
        return connection.scalar(
            select(recommendation_portfolios.c.strategy_version_id).where(
                recommendation_portfolios.c.id == portfolio.source_id
            )
        )

    def source_risk_state(self, portfolio_id: str) -> dict[str, Any]:
        """Expose the unified member/allocation gate used by every simulation adapter."""

        with self.engine.connect() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == portfolio_id
                )
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            if str(portfolio.source_type) == "allocation":
                state = load_allocation_risk_state(
                    connection,
                    str(portfolio.source_id),
                )
                return {
                    "strategy_version_id": None,
                    "allow_new_risk": state["risk_exposure_override"] >= 1.0,
                    **state,
                }
            strategy_version_id = self._source_strategy_version_id(connection, portfolio)
            if not strategy_version_id:
                return {
                    "strategy_version_id": None,
                    "state": "active",
                    "allow_new_risk": True,
                    "risk_exposure_override": 1.0,
                    "event_ids": [],
                    "allocation_ids": [],
                }
            return load_strategy_risk_state(connection, str(strategy_version_id))

    @staticmethod
    def _apply_risk_exposure_override(
        *,
        adapter: str,
        target_payload: dict[str, Any],
        risk_exposure_override: float,
    ) -> dict[str, Any]:
        override = float(risk_exposure_override)
        if not isfinite(override) or not 0.0 <= override <= 1.0:
            raise ValueError("strategy risk exposure override must be between zero and one")
        payload = dict(target_payload)
        if adapter == "long_only":
            payload["target_weights"] = {
                instrument: float(weight) * override
                for instrument, weight in dict(
                    payload.get("target_weights") or {}
                ).items()
            }
            return payload
        if adapter != "pair":
            raise ValueError("unsupported simulation execution adapter")
        scaled_legs: list[dict[str, Any]] = []
        for raw_leg in payload.get("legs") or []:
            leg = dict(raw_leg)
            quantity = int(leg.get("target_quantity") or 0)
            leg["target_quantity"] = int(quantity * override) // 100 * 100
            scaled_legs.append(leg)
        payload["legs"] = scaled_legs
        return payload

    @staticmethod
    def _apply_member_new_risk_gate(
        *,
        adapter: str,
        target_payload: dict[str, Any],
        positions: dict[str, dict[str, Any]],
        portfolio_nav: float,
    ) -> dict[str, Any]:
        payload = dict(target_payload)
        if adapter == "long_only":
            if portfolio_nav <= 0:
                raise ValueError("simulation NAV must be positive for the member risk gate")
            current_weights = {
                instrument: max(0.0, float(position.get("market_value") or 0.0))
                / portfolio_nav
                for instrument, position in positions.items()
                if str(position.get("position_side") or "long") == "long"
            }
            payload["target_weights"] = {
                instrument: min(float(weight), current_weights.get(instrument, 0.0))
                for instrument, weight in dict(payload.get("target_weights") or {}).items()
            }
            return payload
        if adapter != "pair":
            raise ValueError("unsupported simulation execution adapter")
        gated_legs: list[dict[str, Any]] = []
        for raw_leg in payload.get("legs") or []:
            leg = dict(raw_leg)
            current = positions.get(str(leg.get("instrument") or "").upper()) or {}
            same_side = str(current.get("position_side") or "") == str(
                leg.get("position_side") or ""
            )
            current_quantity = int(current.get("quantity") or 0) if same_side else 0
            leg["target_quantity"] = min(
                int(leg.get("target_quantity") or 0),
                current_quantity,
            )
            gated_legs.append(leg)
        payload["legs"] = gated_legs
        return payload

    @staticmethod
    def _require_governed_long_only_target(
        *,
        batch: Any,
        portfolio: Any,
        target_payload: dict[str, Any],
    ) -> None:
        plan = target_payload.get("governed_order_plan")
        if not isinstance(plan, dict):
            raise ValueError(
                "long-only strategy simulation batch has no governed Qlib order-plan"
            )
        normalized = SimulationStore._normalize_target_payload(
            {"target_weights": target_payload.get("target_weights")},
            adapter="long_only",
        )
        source_snapshot = plan.get("source_snapshot")
        if not isinstance(source_snapshot, dict):
            raise ValueError("governed Qlib order-plan source snapshot is missing")
        required_matches = (
            (plan.get("format_version"), QLIB_ORDER_PLAN_FORMAT_VERSION),
            (
                plan.get("execution_contract_hash"),
                str(portfolio.execution_contract_hash),
            ),
            (
                plan.get("target_weights_sha256"),
                _canonical_hash(normalized),
            ),
            (
                source_snapshot.get("id"),
                str(batch.source_snapshot_id),
            ),
            (
                source_snapshot.get("dataset_identity_sha256"),
                str(portfolio.daily_dataset_identity_sha256),
            ),
            (
                source_snapshot.get("dataset_lineage_id"),
                str(portfolio.daily_dataset_lineage_id),
            ),
        )
        if any(observed != expected for observed, expected in required_matches):
            raise ValueError("governed Qlib order-plan failed batch-time verification")
        for field in ("signal_at", "execution_not_before"):
            expected_timestamp = getattr(batch, field)
            planned_timestamp = plan.get(field)
            if expected_timestamp is None:
                if planned_timestamp is not None:
                    raise ValueError(
                        "governed Qlib order-plan timing failed batch-time verification"
                    )
                continue
            if planned_timestamp is None or _parse_aware_timestamp(
                planned_timestamp,
                field=field,
            ) != expected_timestamp.astimezone(UTC):
                raise ValueError(
                    "governed Qlib order-plan timing failed batch-time verification"
                )
        signal_snapshot = plan.get("signal_snapshot")
        if batch.signal_at is not None and (
            not isinstance(signal_snapshot, dict)
            or not str(signal_snapshot.get("name") or "")
            or any(
                not _is_sha256(signal_snapshot.get(field))
                for field in (
                    "dataset_identity_sha256",
                    "dataset_lineage_id",
                    "source_lineage_id",
                )
            )
        ):
            raise ValueError(
                "governed Qlib order-plan signal snapshot failed verification"
            )
        if not _is_sha256(plan.get("manifest_sha256")) or not _is_sha256(
            plan.get("target_weights_file_sha256")
        ):
            raise ValueError("governed Qlib order-plan artifact identities are invalid")
        require_qlib_workflow_identity(plan.get("qlib_workflow"))

    @staticmethod
    def _runtime_execution_policy(
        *,
        portfolio: Any,
        batch: Any,
        execution_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        stored = dict(portfolio.execution_policy_json or {})
        if str(portfolio.execution_adapter) == "pair":
            return stored
        algorithm = str(stored.get("execution_algorithm") or "")
        if algorithm != "vwap":
            if execution_evidence.get("execution_volume_profile") is not None:
                raise ValueError(
                    "non-VWAP simulation cannot accept execution volume-profile evidence"
                )
            return stored
        if stored.get("volume_profile_method") != VWAP_PROFILE_METHOD:
            raise ValueError("simulation VWAP profile method is not governed")
        lookback_days = int(stored.get("volume_profile_lookback_days") or 0)
        profile = execution_evidence.get("execution_volume_profile")
        evidence = execution_evidence.get("execution_volume_profile_evidence")
        if not isinstance(profile, list) or not profile or not isinstance(evidence, dict):
            raise ValueError(
                "VWAP simulation requires a historical profile from the bound "
                "Qlib execution dataset"
            )
        try:
            profile_start = date.fromisoformat(str(evidence["start"]))
            profile_end = date.fromisoformat(str(evidence["end"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("VWAP execution profile dates are invalid") from exc
        identity = {
            "method": VWAP_PROFILE_METHOD,
            "lookback_days": lookback_days,
            "start": profile_start.isoformat(),
            "end": profile_end.isoformat(),
            "trade_date": batch.trade_date.isoformat(),
            "dataset_identity_sha256": str(
                portfolio.execution_dataset_identity_sha256
            ),
            "dataset_lineage_id": str(portfolio.execution_dataset_lineage_id),
            "simulation_semantics_sha256": stored.get(
                "simulation_semantics_sha256"
            ),
            "profile": profile,
        }
        if (
            evidence.get("method") != VWAP_PROFILE_METHOD
            or int(evidence.get("lookback_days") or 0) != lookback_days
            or evidence.get("future_data_used") is not False
            or profile_end >= batch.trade_date
            or str(evidence.get("dataset_identity_sha256") or "")
            != str(portfolio.execution_dataset_identity_sha256)
            or str(evidence.get("dataset_lineage_id") or "")
            != str(portfolio.execution_dataset_lineage_id)
            or str(evidence.get("simulation_semantics_sha256") or "")
            != str(stored.get("simulation_semantics_sha256") or "")
            or execution_evidence.get("execution_volume_profile_sha256")
            != _canonical_hash(identity)
        ):
            raise ValueError(
                "VWAP execution profile does not match the immutable simulation contract"
            )
        return {
            **stored,
            "volume_profile": profile,
        }

    def process_batch(
        self,
        batch_id: str,
        *,
        minute_bars: pd.DataFrame,
        closing_prices: dict[str, dict[str, Any]],
        execution_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute, book and value a simulation day in one database transaction."""

        now = _now()
        with self.engine.begin() as connection:
            batch = connection.execute(
                select(simulation_batches)
                .where(simulation_batches.c.id == batch_id)
                .with_for_update()
            ).first()
            if batch is None:
                raise KeyError(batch_id)
            if batch.status == "succeeded":
                return self._batch_dict(batch)
            if batch.status != "queued":
                raise ValueError("only queued simulation batches may execute")
            portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == batch.portfolio_id)
                .with_for_update()
            ).one()
            if portfolio.status != "active":
                raise ValueError("simulation portfolio is not active")
            self._require_current_source_contract(connection, portfolio)
            if (
                str(batch.execution_contract_hash)
                != str(portfolio.execution_contract_hash)
                or str(batch.execution_adapter) != str(portfolio.execution_adapter)
            ):
                raise ValueError("simulation batch contract no longer matches its portfolio")
            expected_execution = {
                "dataset_identity_sha256": str(
                    portfolio.execution_dataset_identity_sha256
                ),
                "dataset_lineage_id": str(portfolio.execution_dataset_lineage_id),
                "execution_contract_version": str(
                    portfolio.execution_field_contract_version
                ),
                "execution_contract_hash": str(portfolio.execution_contract_hash),
            }
            mismatches = [
                field
                for field, expected in expected_execution.items()
                if str(execution_evidence.get(field) or "") != expected
            ]
            if mismatches:
                raise ValueError(
                    "simulation execution evidence does not match the bound dataset: "
                    + ", ".join(mismatches)
                )
            if str(execution_evidence.get("batch_id") or "") != str(batch.id):
                raise ValueError("simulation execution evidence does not match the batch")
            for field in ("signal_at", "execution_not_before"):
                expected_timestamp = getattr(batch, field)
                observed_timestamp = execution_evidence.get(field)
                if expected_timestamp is None:
                    if observed_timestamp is not None:
                        raise ValueError(
                            "simulation execution evidence contains unexpected timing"
                        )
                    continue
                if observed_timestamp is None or _parse_aware_timestamp(
                    observed_timestamp,
                    field=field,
                ) != expected_timestamp.astimezone(UTC):
                    raise ValueError(
                        "simulation execution evidence does not match order-plan timing"
                    )
            target_payload = dict(batch.target_payload_json or {})
            if batch.recommendation_snapshot_id:
                snapshot = connection.execute(
                    select(recommendation_snapshots).where(
                        recommendation_snapshots.c.id == batch.recommendation_snapshot_id
                    )
                ).one()
                if snapshot.status != "succeeded":
                    raise ValueError("simulation recommendation snapshot is no longer valid")
                target_payload = {
                    "target_weights": {
                        str(item.instrument): float(item.weight)
                        for item in connection.execute(
                            select(recommendation_holdings).where(
                                recommendation_holdings.c.snapshot_id == snapshot.id
                            )
                        )
                    }
                }
            elif not target_payload:
                raise ValueError("simulation batch has no governed target payload")
            if (
                not batch.recommendation_snapshot_id
                and str(portfolio.execution_adapter) == "long_only"
            ):
                self._require_governed_long_only_target(
                    batch=batch,
                    portfolio=portfolio,
                    target_payload=target_payload,
                )
            position_state = {
                str(item.instrument): row_dict(item)
                for item in connection.execute(
                    select(simulation_positions).where(
                        simulation_positions.c.portfolio_id == portfolio.id
                    )
                )
            }
            strategy_version_id = self._source_strategy_version_id(connection, portfolio)
            strategy_risk_state = (
                load_strategy_risk_state(connection, str(strategy_version_id))
                if strategy_version_id
                else {
                    "strategy_version_id": None,
                    "state": "active",
                    "allow_new_risk": True,
                    "risk_exposure_override": 1.0,
                    "event_ids": [],
                    "allocation_ids": [],
                }
            )
            risk_exposure_override = float(
                strategy_risk_state["risk_exposure_override"]
            )
            if risk_exposure_override < 1.0:
                target_payload = self._apply_risk_exposure_override(
                    adapter=str(portfolio.execution_adapter),
                    target_payload=target_payload,
                    risk_exposure_override=risk_exposure_override,
                )
            if not strategy_risk_state["allow_new_risk"]:
                target_payload = self._apply_member_new_risk_gate(
                    adapter=str(portfolio.execution_adapter),
                    target_payload=target_payload,
                    positions=position_state,
                    portfolio_nav=float(portfolio.nav),
                )
            execution_policy = self._runtime_execution_policy(
                portfolio=portfolio,
                batch=batch,
                execution_evidence=execution_evidence,
            )
            cost_schedule = CostScheduleBook.from_mapping(
                dict(portfolio.execution_policy_json or {}).get("cost_model")
            )
            if str(portfolio.execution_adapter) == "pair":
                governed_plan = target_payload.get("governed_pair_plan")
                if not isinstance(governed_plan, dict):
                    raise ValueError("pair simulation batch has no governed artifact plan")
                shortability_binding = dict(
                    governed_plan.get("shortability_dataset") or {}
                )
                if not (
                    execution_evidence.get("pair_plan_sha256")
                    == governed_plan.get("pair_plan_sha256")
                    and execution_evidence.get("pair_artifact_manifest_sha256")
                    == governed_plan.get("pair_artifact_manifest_sha256")
                    and execution_evidence.get("shortability_source_sha256")
                    == shortability_binding.get("source_sha256")
                    and execution_evidence.get(
                        "shortability_snapshot_manifest_sha256"
                    )
                    == shortability_binding.get("manifest_sha256")
                ):
                    raise ValueError(
                        "pair execution evidence does not match the governed backtest plan"
                    )
                shortability = execution_evidence.get("shortability")
                if not isinstance(shortability, dict):
                    raise ValueError("pair simulation requires dated shortability evidence")
                if str(execution_evidence.get("shortability_trade_date") or "") != (
                    batch.trade_date.isoformat()
                ):
                    raise ValueError("pair shortability evidence is not valid for the trade date")
                shortability_sha256 = execution_evidence.get(
                    "shortability_evidence_sha256"
                )
                if not _is_sha256(shortability_sha256):
                    raise ValueError("pair shortability evidence requires an immutable identity")
                result = execute_atomic_pair_day(
                    trade_date=batch.trade_date,
                    cash=float(portfolio.cash),
                    prior_nav=float(portfolio.nav),
                    high_water_mark=float(portfolio.high_water_mark),
                    positions=position_state,
                    target_payload=target_payload,
                    minute_bars=minute_bars,
                    closing_prices=closing_prices,
                    shortability={
                        str(key).upper(): value is True
                        for key, value in shortability.items()
                    },
                    cost_schedule=cost_schedule,
                    execution_policy=execution_policy,
                )
                result["shortability_evidence_sha256"] = str(shortability_sha256)
            else:
                result = execute_simulation_day(
                    trade_date=batch.trade_date,
                    cash=float(portfolio.cash),
                    prior_nav=float(portfolio.nav),
                    high_water_mark=float(portfolio.high_water_mark),
                    positions=position_state,
                    target_weights=dict(target_payload["target_weights"]),
                    minute_bars=minute_bars,
                    closing_prices=closing_prices,
                    cost_schedule=cost_schedule,
                    execution_policy=execution_policy,
                    signal_at=batch.signal_at,
                )
            if (
                not strategy_risk_state["allow_new_risk"]
                or risk_exposure_override < 1.0
            ):
                result["events"].append(
                    {
                        "severity": "warning",
                        "event_type": "strategy_risk_gate_applied",
                        "reason": str(strategy_risk_state["state"]),
                        "details": strategy_risk_state,
                    }
                )
                result["strategy_risk_state"] = strategy_risk_state
            self._persist_result(connection, batch, portfolio, result, now)
        return self.get_batch(batch_id)

    def mark_batch_failed(self, batch_id: str, error: str) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(simulation_batches)
                .where(
                    simulation_batches.c.id == batch_id,
                    simulation_batches.c.status == "queued",
                )
                .values(status="failed", error=error, finished_at=_now())
            )
            if not result.rowcount:
                existing = connection.execute(
                    select(simulation_batches.c.id).where(
                        simulation_batches.c.id == batch_id
                    )
                ).first()
                if existing is None:
                    raise KeyError(batch_id)

    def _persist_result(
        self, connection: Any, batch: Any, portfolio: Any, result: dict[str, Any], now: datetime
    ) -> None:
        order_ids: dict[tuple[str, str], str] = {}
        for order in result["orders"]:
            order_id = uuid.uuid4().hex
            key = (str(order["instrument"]), str(order["side"]))
            order_ids[key] = order_id
            expires_at = order.get("expires_at") or datetime.combine(
                batch.trade_date, datetime.max.time(), tzinfo=UTC
            )
            connection.execute(
                insert(simulation_orders).values(
                    id=order_id,
                    batch_id=batch.id,
                    instrument=order["instrument"],
                    side=order["side"],
                    atomic_group_id=order.get("atomic_group_id"),
                    leg_no=order.get("leg_no"),
                    position_side=order.get("position_side", "long"),
                    borrow_cost=Decimal(str(order.get("borrow_cost", 0.0))),
                    target_weight=float(order["target_weight"]),
                    requested_quantity=int(order["requested_quantity"]),
                    filled_quantity=int(order["filled_quantity"]),
                    status=order["status"],
                    reject_reason=order.get("reject_reason"),
                    requested_value=Decimal(str(order["requested_value"])),
                    filled_value=Decimal(str(order["filled_value"])),
                    capacity_fill_ratio=float(order["capacity_fill_ratio"]),
                    expires_at=expires_at,
                    created_at=now,
                )
            )
        fill_ids: list[str] = []
        for fill in result["fills"]:
            fill_id = uuid.uuid4().hex
            fill_ids.append(fill_id)
            connection.execute(
                insert(simulation_fills).values(
                    id=fill_id,
                    order_id=order_ids[(str(fill["instrument"]), str(fill["side"]))],
                    batch_id=batch.id,
                    instrument=fill["instrument"],
                    side=fill["side"],
                    atomic_group_id=fill.get("atomic_group_id"),
                    leg_no=fill.get("leg_no"),
                    position_side=fill.get("position_side", "long"),
                    borrow_cost=Decimal(str(fill.get("borrow_cost", 0.0))),
                    executed_at=fill["executed_at"],
                    quantity=int(fill["quantity"]),
                    price=Decimal(str(fill["price"])),
                    gross_value=Decimal(str(fill["gross_value"])),
                    fee=Decimal(str(fill["fee"])),
                    cost_breakdown_json=fill["cost_breakdown"],
                    minute_volume=int(fill["minute_volume"]),
                    capacity_quantity=int(fill["capacity_quantity"]),
                )
            )
        for index, flow in enumerate(result["cash_flows"]):
            connection.execute(
                insert(simulation_cash_flows).values(
                    id=uuid.uuid4().hex,
                    portfolio_id=portfolio.id,
                    batch_id=batch.id,
                    trade_date=batch.trade_date,
                    flow_type=flow["flow_type"],
                    amount=Decimal(str(flow["amount"])),
                    balance_after=Decimal(str(flow["balance_after"])),
                    reference_id=fill_ids[index] if index < len(fill_ids) else None,
                    created_at=now,
                )
            )
        connection.execute(
            delete(simulation_positions).where(
                simulation_positions.c.portfolio_id == portfolio.id
            )
        )
        for instrument, position in result["positions"].items():
            connection.execute(
                insert(simulation_positions).values(
                    portfolio_id=portfolio.id,
                    instrument=instrument,
                    atomic_group_id=position.get("atomic_group_id"),
                    leg_no=position.get("leg_no"),
                    position_side=position.get("position_side", "long"),
                    borrow_cost=Decimal(str(position.get("borrow_cost", 0.0))),
                    quantity=int(position["quantity"]),
                    available_quantity=int(position["available_quantity"]),
                    average_cost=Decimal(str(position["average_cost"])),
                    last_trade_date=position.get("last_trade_date"),
                    market_price=(
                        Decimal(str(position["market_price"]))
                        if position.get("market_price") is not None
                        else None
                    ),
                    market_date=position.get("market_date"),
                    stale=bool(position.get("stale", True)),
                    market_value=Decimal(str(position.get("market_value", 0.0))),
                    updated_at=now,
                )
            )
        nav = result["nav_row"]
        connection.execute(
            insert(simulation_nav).values(
                portfolio_id=portfolio.id,
                trade_date=batch.trade_date,
                cash=Decimal(str(nav["cash"])),
                market_value=Decimal(str(nav["market_value"])),
                nav=Decimal(str(nav["nav"])),
                daily_return=float(nav["daily_return"]),
                drawdown=float(nav["drawdown"]),
                market_date=nav["market_date"],
                has_stale_prices=bool(nav["has_stale_prices"]),
                status=nav["status"],
                performance_certified=bool(nav["performance_certified"]),
                nav_scope=(
                    "aggregate_view"
                    if str(portfolio.source_type) == "allocation"
                    else "member_ledger"
                ),
                produced_by=str(batch.created_by),
                created_at=now,
            )
        )
        for event in result["events"]:
            connection.execute(
                insert(simulation_events).values(
                    id=uuid.uuid4().hex,
                    portfolio_id=portfolio.id,
                    batch_id=batch.id,
                    trade_date=batch.trade_date,
                    severity=event["severity"],
                    event_type=event["event_type"],
                    instrument=event.get("instrument"),
                    reason=event["reason"],
                    details_json=event.get("details") or {},
                    created_at=now,
                )
            )
        summary = {
            "engine_version": result["engine_version"],
            "execution_adapter": str(batch.execution_adapter),
            "execution_contract_hash": str(batch.execution_contract_hash),
            "orders": len(result["orders"]),
            "fills": len(result["fills"]),
            "rejections": sum(order["status"] != "filled" for order in result["orders"]),
            "cash": result["cash"],
            "nav": result["nav"],
            "conservation": result["conservation"],
        }
        if result.get("shortability_evidence_sha256"):
            summary["shortability_evidence_sha256"] = result[
                "shortability_evidence_sha256"
            ]
        if result.get("strategy_risk_state"):
            summary["strategy_risk_state"] = result["strategy_risk_state"]
        connection.execute(
            update(simulation_portfolios)
            .where(simulation_portfolios.c.id == portfolio.id)
            .values(
                cash=Decimal(str(result["cash"])),
                nav=Decimal(str(result["nav"])),
                high_water_mark=Decimal(str(result["high_water_mark"])),
                updated_at=now,
            )
        )
        connection.execute(
            update(simulation_batches)
            .where(simulation_batches.c.id == batch.id)
            .values(
                status="succeeded",
                summary_json=summary,
                started_at=now,
                finished_at=now,
                error=None,
            )
        )

    def review_nav(
        self,
        portfolio_id: str,
        trade_date: date,
        *,
        actor: str,
        evidence_sha256: str,
        note: str,
    ) -> dict[str, Any]:
        """Attach one immutable four-eyes review to a certified simulation NAV row."""

        reviewer = actor.strip()
        evidence = evidence_sha256.strip().lower()
        review_note = note.strip()
        if len(reviewer) < 2:
            raise ValueError("a responsible simulation NAV reviewer is required")
        if not _is_sha256(evidence):
            raise ValueError("simulation NAV review evidence must be a SHA-256 digest")
        if len(review_note) < 10:
            raise ValueError("simulation NAV review note must be meaningful")
        now = _now()
        with self.engine.begin() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == portfolio_id
                )
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            nav_row = connection.execute(
                select(simulation_nav)
                .where(
                    simulation_nav.c.portfolio_id == portfolio_id,
                    simulation_nav.c.trade_date == trade_date,
                )
                .with_for_update()
            ).first()
            if nav_row is None:
                raise KeyError(f"{portfolio_id}:{trade_date.isoformat()}")
            if not nav_row.performance_certified:
                raise ValueError("only performance-certified simulation NAV may be reviewed")
            if nav_row.reviewed_at is not None:
                raise ValueError("simulation NAV review is immutable and already recorded")
            if reviewer == str(nav_row.produced_by):
                raise ValueError("simulation NAV reviewer must differ from its producer")
            reviewed = connection.execute(
                update(simulation_nav)
                .where(
                    simulation_nav.c.portfolio_id == portfolio_id,
                    simulation_nav.c.trade_date == trade_date,
                    simulation_nav.c.reviewed_at.is_(None),
                )
                .values(
                    reviewed_by=reviewer,
                    reviewed_at=now,
                    review_evidence_sha256=evidence,
                    review_note=review_note,
                )
                .returning(simulation_nav)
            ).one()
            batch_id = connection.scalar(
                select(simulation_batches.c.id)
                .where(
                    simulation_batches.c.portfolio_id == portfolio_id,
                    simulation_batches.c.trade_date == trade_date,
                )
                .order_by(simulation_batches.c.created_at.desc())
                .limit(1)
            )
            connection.execute(
                insert(simulation_events).values(
                    id=uuid.uuid4().hex,
                    portfolio_id=portfolio_id,
                    batch_id=batch_id,
                    trade_date=trade_date,
                    severity="info",
                    event_type="simulation_nav_reviewed",
                    instrument=None,
                    reason="certified simulation NAV independently reviewed",
                    details_json={
                        "reviewed_by": reviewer,
                        "review_evidence_sha256": evidence,
                        "nav_scope": str(nav_row.nav_scope),
                        "source_type": str(portfolio.source_type),
                    },
                    created_at=now,
                )
            )
        result = row_dict(reviewed)
        result["review_subject"] = (
            "aggregate_simulation_view"
            if result["nav_scope"] == "aggregate_view"
            else "member_simulation_ledger"
        )
        return result

    def get(self, portfolio_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == portfolio_id
                )
            ).first()
            if row is None:
                raise KeyError(portfolio_id)
            result = self._portfolio_dict(row)
            result["latest_nav"] = self._first_dict(
                connection.execute(
                    select(simulation_nav)
                    .where(simulation_nav.c.portfolio_id == portfolio_id)
                    .order_by(simulation_nav.c.trade_date.desc())
                    .limit(1)
                ).first()
            )
            reviewed_rows = connection.execute(
                select(simulation_nav)
                .where(
                    simulation_nav.c.portfolio_id == portfolio_id,
                    simulation_nav.c.performance_certified.is_(True),
                    simulation_nav.c.reviewed_at.is_not(None),
                )
                .order_by(simulation_nav.c.trade_date.desc())
                .limit(5)
            ).all()
            review_evidence = [
                {
                    "trade_date": item.trade_date.isoformat(),
                    "reviewed_by": str(item.reviewed_by),
                    "reviewed_at": item.reviewed_at.isoformat(),
                    "review_evidence_sha256": str(item.review_evidence_sha256),
                }
                for item in reversed(reviewed_rows)
            ]
            nav_scope = (
                str(reviewed_rows[0].nav_scope)
                if reviewed_rows
                else (
                    "aggregate_view"
                    if str(row.source_type) == "allocation"
                    else "member_ledger"
                )
            )
            result["review_readiness"] = {
                "nav_scope": nav_scope,
                "view_semantics": (
                    "aggregate view derived from member simulation NAV"
                    if nav_scope == "aggregate_view"
                    else "persistent member simulation ledger"
                ),
                "required_reviewed_days": 5,
                "reviewed_days": len(reviewed_rows),
                "ready": len(reviewed_rows) == 5,
                "evidence_sha256": (
                    _canonical_hash({"reviews": review_evidence})
                    if review_evidence
                    else None
                ),
                "reviews": review_evidence,
            }
        return result

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(simulation_portfolios)
                .order_by(simulation_portfolios.c.updated_at.desc())
                .limit(limit)
            )
            return [self._portfolio_dict(row) for row in rows]

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(simulation_batches).where(simulation_batches.c.id == batch_id)
            ).first()
            if row is None:
                raise KeyError(batch_id)
            return self._batch_dict(row)

    def execution_manifest(self, batch_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            batch = connection.execute(
                select(simulation_batches).where(simulation_batches.c.id == batch_id)
            ).first()
            if batch is None:
                raise KeyError(batch_id)
            portfolio = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == batch.portfolio_id
                )
            ).one()
            snapshot = None
            if batch.recommendation_snapshot_id:
                snapshot = connection.execute(
                    select(recommendation_snapshots).where(
                        recommendation_snapshots.c.id == batch.recommendation_snapshot_id
                    )
                ).one()
                target_instruments = set(
                    str(value)
                    for value in connection.scalars(
                        select(recommendation_holdings.c.instrument).where(
                            recommendation_holdings.c.snapshot_id == snapshot.id
                        )
                    )
                )
            else:
                payload = dict(batch.target_payload_json or {})
                governed_pair_plan = payload.get("governed_pair_plan")
                target_instruments = set(payload.get("target_weights") or {})
                target_instruments.update(
                    str(item.get("instrument"))
                    for item in (payload.get("legs") or [])
                    if item.get("instrument")
                )
            if batch.recommendation_snapshot_id:
                governed_pair_plan = None
            held_instruments = set(
                str(value)
                for value in connection.scalars(
                    select(simulation_positions.c.instrument).where(
                        simulation_positions.c.portfolio_id == portfolio.id
                    )
                )
            )
        return {
            "batch_id": str(batch.id),
            "portfolio_id": str(portfolio.id),
            "source_type": str(portfolio.source_type),
            "source_id": str(portfolio.source_id),
            "source_snapshot_id": str(batch.source_snapshot_id),
            "recommendation_portfolio_id": (
                str(portfolio.recommendation_portfolio_id)
                if portfolio.recommendation_portfolio_id
                else None
            ),
            "recommendation_snapshot_id": str(snapshot.id) if snapshot else None,
            "signal_date": batch.signal_date.isoformat(),
            "trade_date": batch.trade_date.isoformat(),
            "signal_at": (
                batch.signal_at.isoformat() if batch.signal_at is not None else None
            ),
            "execution_not_before": (
                batch.execution_not_before.isoformat()
                if batch.execution_not_before is not None
                else None
            ),
            "daily_dataset": str(portfolio.daily_dataset),
            "execution_dataset": str(portfolio.execution_dataset),
            "execution_algorithm": str(portfolio.execution_algorithm),
            "execution_policy": dict(portfolio.execution_policy_json or {}),
            "execution_adapter": str(portfolio.execution_adapter),
            "execution_frequency": str(portfolio.execution_frequency),
            "execution_contract_hash": str(portfolio.execution_contract_hash),
            "instruments": sorted(target_instruments | held_instruments),
            "governed_pair_plan": governed_pair_plan,
        }

    def rows(self, portfolio_id: str, resource: str) -> list[dict[str, Any]]:
        resources = {
            "orders": (
                simulation_orders.join(
                    simulation_batches,
                    simulation_orders.c.batch_id == simulation_batches.c.id,
                ),
                simulation_orders,
                simulation_orders.c.created_at,
            ),
            "fills": (
                simulation_fills.join(
                    simulation_batches,
                    simulation_fills.c.batch_id == simulation_batches.c.id,
                ),
                simulation_fills,
                simulation_fills.c.executed_at,
            ),
            "positions": (
                simulation_positions,
                simulation_positions,
                simulation_positions.c.instrument,
            ),
            "nav": (simulation_nav, simulation_nav, simulation_nav.c.trade_date),
            "events": (simulation_events, simulation_events, simulation_events.c.created_at),
        }
        if resource not in resources:
            raise ValueError("unknown simulation resource")
        source, table, ordering = resources[resource]
        portfolio_column = (
            simulation_batches.c.portfolio_id
            if resource in {"orders", "fills"}
            else table.c.portfolio_id
        )
        with self.engine.connect() as connection:
            return [
                row_dict(row)
                for row in connection.execute(
                    select(table)
                    .select_from(source)
                    .where(portfolio_column == portfolio_id)
                    .order_by(ordering)
                )
            ]

    @staticmethod
    def _portfolio_dict(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["execution_policy"] = result.pop("execution_policy_json")
        result["provenance"] = {
            "daily": {
                "dataset": result["daily_dataset"],
                "dataset_identity_sha256": result["daily_dataset_identity_sha256"],
                "dataset_lineage_id": result["daily_dataset_lineage_id"],
                "field_contract_version": result["daily_field_contract_version"],
            },
            "minute": {
                "dataset": result["execution_dataset"],
                "dataset_identity_sha256": result[
                    "execution_dataset_identity_sha256"
                ],
                "dataset_lineage_id": result["execution_dataset_lineage_id"],
                "field_contract_version": result["execution_field_contract_version"],
            },
            "execution_engine_version": result["execution_engine_version"],
            "cost_schedule_version": result["cost_schedule_version"],
        }
        return result

    @staticmethod
    def _batch_dict(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["summary"] = result.pop("summary_json")
        result["target_payload"] = result.pop("target_payload_json")
        return result

    @staticmethod
    def _first_dict(row: Any | None) -> dict[str, Any] | None:
        return row_dict(row) if row is not None else None
