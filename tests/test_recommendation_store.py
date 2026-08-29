from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date
from threading import Barrier

import pytest
from governance_fixtures import (
    DATASET_IDENTITY,
    create_strategy_version,
    enable_recommendation_authority_for_test,
)
from sqlalchemy import func, select, update

from quant_data.database import jobs as job_rows
from quant_data.database import paper_fills, paper_orders, strategy_versions
from quant_platform.job_store import JobStore
from quant_platform.portfolio_policy import POLICY_VERSION
from quant_platform.qlib_backtest import QLIB_ENGINE_VERSION
from quant_platform.recommendation_store import (
    RecommendationStore,
    recommendation_refresh_job_idempotency_key,
    recommendation_refresh_job_payload,
)


def test_recommendation_snapshot_is_independent_of_paper_orders_and_fills(
    tmp_path, database_url: str
) -> None:
    version_id = create_strategy_version(
        database_url,
        tmp_path,
        recipe_id="short_relative_strength",
    )
    store = RecommendationStore(database_url)
    with store.engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved")
        )
        before = (
            connection.scalar(select(func.count()).select_from(paper_orders)),
            connection.scalar(select(func.count()).select_from(paper_fills)),
        )
    enable_recommendation_authority_for_test(database_url, [version_id])
    portfolio = store.create(
        name="governed recommendations",
        strategy_version_id=version_id,
        dataset="snapshot",
        hypothetical_initial_value=5_000_000,
        actor="test",
    )
    snapshot, created = store.create_snapshot(
        portfolio_id=portfolio["id"],
        as_of_date=date(2026, 7, 10),
        dataset="snapshot",
        dataset_identity_sha256=DATASET_IDENTITY,
    )
    assert created is True
    result = {
        "status": "ok",
        "portfolio_id": portfolio["id"],
        "strategy_version_id": version_id,
        "dataset": "snapshot",
        "dataset_identity_sha256": DATASET_IDENTITY,
        "as_of_date": "2026-07-10",
        "policy_version": POLICY_VERSION,
        "backtest_engine_version": QLIB_ENGINE_VERSION,
        "effective_date": "2026-07-13",
        "cost_model": snapshot["cost_model"],
        "cash_weight": 0.98,
        "reference_prices": {"SH600000": 10.0},
        "holdings": [
            {
                "instrument": "SH600000",
                "weight": 0.02,
                "previous_weight": 0.0,
                "weight_change": 0.02,
                "action": "increase",
                "reason": "ranked signal and constraints",
            }
        ],
        "hypothetical_observation": {
            "trade_date": "2026-07-10",
            "hypothetical_value": 4_999_000,
            "daily_return": 0.0,
            "benchmark_return": 0.0,
            "drawdown": 0.0,
            "turnover": 0.02,
            "estimated_cost": 1_000,
        },
    }
    completed = store.apply_result(snapshot["id"], result)
    assert completed["status"] == "succeeded"
    assert completed["holdings"][0]["instrument"] == "SH600000"
    tracked = store.get(portfolio["id"])
    assert tracked["historical_hypothetical_observations"] == []
    assert tracked["construction_notional"] == 5_000_000
    assert "hypothetical_performance" not in tracked
    with store.engine.connect() as connection:
        after = (
            connection.scalar(select(func.count()).select_from(paper_orders)),
            connection.scalar(select(func.count()).select_from(paper_fills)),
        )
    assert after == before


def test_unattached_queued_snapshot_is_retryable_and_recovers_existing_job(
    tmp_path, database_url: str
) -> None:
    version_id = create_strategy_version(
        database_url,
        tmp_path,
        recipe_id="short_relative_strength",
    )
    store = RecommendationStore(database_url)
    with store.engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved")
        )
    enable_recommendation_authority_for_test(database_url, [version_id])
    portfolio = store.create(
        name="recover queued recommendation",
        strategy_version_id=version_id,
        dataset="snapshot",
        hypothetical_initial_value=100_000,
        actor="test",
    )
    as_of = date(2026, 7, 10)
    snapshot, created = store.create_snapshot(
        portfolio_id=portfolio["id"],
        as_of_date=as_of,
        dataset="snapshot",
        dataset_identity_sha256=DATASET_IDENTITY,
    )
    assert created is True

    # The append-only row remains visible, but it must not suppress the next
    # scheduler attempt merely because the insert committed before JobStore.
    orphaned = store.get(portfolio["id"])
    assert orphaned["latest_snapshot"] is None
    assert orphaned["pending_snapshot"]["id"] == snapshot["id"]
    retry, should_enqueue = store.create_snapshot(
        portfolio_id=portfolio["id"],
        as_of_date=as_of,
        dataset="snapshot",
        dataset_identity_sha256=DATASET_IDENTITY,
    )
    assert retry["id"] == snapshot["id"]
    assert should_enqueue is True

    # Model a process that committed job creation and died before attach_job.
    dataset = {
        "name": "snapshot",
        "path": str(tmp_path),
        "provenance": {"dataset_identity_sha256": DATASET_IDENTITY},
    }
    job = JobStore(database_url).create(
        "recommendation_refresh",
        recommendation_refresh_job_payload(snapshot, dataset),
        tmp_path / "recommendation-refresh.log",
        dedupe_active_kind=False,
        idempotency_key=recommendation_refresh_job_idempotency_key(snapshot["id"]),
    )
    repaired, should_enqueue = store.create_snapshot(
        portfolio_id=portfolio["id"],
        as_of_date=as_of,
        dataset="snapshot",
        dataset_identity_sha256=DATASET_IDENTITY,
    )
    assert should_enqueue is False
    assert repaired["id"] == snapshot["id"]
    assert repaired["job_id"] == job["id"]
    assert repaired["status"] == "running"

    tracked = store.get(portfolio["id"])
    assert tracked["pending_snapshot"] is None
    assert tracked["latest_snapshot"]["id"] == snapshot["id"]


def test_concurrent_snapshot_claims_create_and_attach_exactly_one_job(
    tmp_path, database_url: str
) -> None:
    version_id = create_strategy_version(
        database_url,
        tmp_path,
        recipe_id="short_relative_strength",
    )
    seed_store = RecommendationStore(database_url)
    with seed_store.engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved")
        )
    enable_recommendation_authority_for_test(database_url, [version_id])
    portfolio = seed_store.create(
        name="concurrent recommendation claim",
        strategy_version_id=version_id,
        dataset="snapshot",
        hypothetical_initial_value=100_000,
        actor="test",
    )
    as_of = date(2026, 7, 10)
    dataset = {
        "name": "snapshot",
        "path": str(tmp_path),
        "provenance": {"dataset_identity_sha256": DATASET_IDENTITY},
    }
    contenders = Barrier(2)

    def enqueue() -> tuple[str, str]:
        recommendations = RecommendationStore(database_url)
        snapshot, should_enqueue = recommendations.create_snapshot(
            portfolio_id=portfolio["id"],
            as_of_date=as_of,
            dataset="snapshot",
            dataset_identity_sha256=DATASET_IDENTITY,
        )
        assert should_enqueue is True
        contenders.wait(timeout=10)
        job = JobStore(database_url).create(
            "recommendation_refresh",
            recommendation_refresh_job_payload(snapshot, dataset),
            tmp_path / f"recommendation-refresh-{snapshot['id']}.log",
            dedupe_active_kind=False,
            idempotency_key=recommendation_refresh_job_idempotency_key(
                snapshot["id"]
            ),
        )
        recommendations.attach_job(snapshot["id"], job["id"])
        return str(snapshot["id"]), str(job["id"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [future.result() for future in [executor.submit(enqueue) for _ in range(2)]]

    snapshot_ids = {item[0] for item in results}
    job_ids = {item[1] for item in results}
    assert len(snapshot_ids) == 1
    assert len(job_ids) == 1
    snapshot_id = snapshot_ids.pop()
    job_id = job_ids.pop()
    key = recommendation_refresh_job_idempotency_key(snapshot_id)
    with seed_store.engine.connect() as connection:
        persisted_jobs = connection.execute(
            select(job_rows.c.id).where(job_rows.c.idempotency_key == key)
        ).all()
    assert [str(item.id) for item in persisted_jobs] == [job_id]
    persisted = seed_store.get_snapshot(snapshot_id)
    assert persisted["job_id"] == job_id
    assert persisted["status"] == "running"


def test_recommendation_result_identity_is_bound_and_cash_only_is_valid(
    tmp_path, database_url: str
) -> None:
    version_id = create_strategy_version(
        database_url,
        tmp_path,
        recipe_id="short_relative_strength",
    )
    store = RecommendationStore(database_url)
    with store.engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved")
        )
    enable_recommendation_authority_for_test(database_url, [version_id])
    portfolio = store.create(
        name="cash recommendation",
        strategy_version_id=version_id,
        dataset="snapshot",
        hypothetical_initial_value=5_000_000,
        actor="test",
    )
    snapshot, _ = store.create_snapshot(
        portfolio_id=portfolio["id"],
        as_of_date=date(2026, 7, 10),
        dataset="snapshot",
        dataset_identity_sha256=DATASET_IDENTITY,
    )
    result = {
        "status": "ok",
        "portfolio_id": portfolio["id"],
        "strategy_version_id": version_id,
        "dataset": "snapshot",
        "dataset_identity_sha256": DATASET_IDENTITY,
        "as_of_date": "2026-07-10",
        "effective_date": "2026-07-13",
        "policy_version": POLICY_VERSION,
        "backtest_engine_version": QLIB_ENGINE_VERSION,
        "cost_model": snapshot["cost_model"],
        "cash_weight": 1.0,
        "reference_prices": {},
        "holdings": [],
        "changes": [{"instrument": "SH600000", "action": "sell", "target_weight": 0.0}],
    }
    for field, bad_value in (
        ("portfolio_id", "wrong-portfolio"),
        ("strategy_version_id", "wrong-version"),
        ("dataset", "wrong-dataset"),
        ("dataset_identity_sha256", "b" * 64),
        ("as_of_date", "2026-07-09"),
    ):
        tampered = deepcopy(result)
        tampered[field] = bad_value
        with pytest.raises(ValueError, match="identity does not match"):
            store.apply_result(snapshot["id"], tampered)

    completed = store.apply_result(snapshot["id"], result)
    assert completed["status"] == "succeeded"
    assert completed["holdings"] == []
    assert completed["snapshot"]["cash_weight"] == 1.0
