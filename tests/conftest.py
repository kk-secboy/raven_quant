from __future__ import annotations

import os

import pytest
from sqlalchemy import delete

from quant_data.database import (
    alerts,
    allocation_schedule_groups,
    allocation_schedule_members,
    audit_events,
    auth_sessions,
    backtest_runs,
    broker_destinations,
    broker_events,
    broker_gateway_attempts,
    broker_gateway_children,
    broker_gateway_events,
    broker_gateway_nonces,
    broker_gateway_parents,
    broker_order_outbox,
    broker_reconciliations,
    data_tasks,
    factor_candidates,
    factor_evaluations,
    jobs,
    open_database,
    pair_paper_fills,
    pair_paper_orders,
    pair_paper_portfolios,
    pair_portfolio_batches,
    pair_portfolio_nav,
    pair_portfolio_reviews,
    pair_portfolio_risk_events,
    paper_fills,
    paper_orders,
    paper_portfolios,
    paper_positions,
    parameter_experiment_trials,
    parameter_experiments,
    platform_config_revisions,
    platform_configs,
    portfolio_batches,
    portfolio_nav,
    portfolio_reviews,
    research_campaign_events,
    research_campaigns,
    research_events,
    research_program_events,
    research_programs,
    research_runs,
    risk_events,
    runtime_secrets,
    schedule_runs,
    schedules,
    strategies,
    strategy_allocation_events,
    strategy_allocation_members,
    strategy_allocation_nav,
    strategy_allocations,
    strategy_events,
    strategy_factors,
    strategy_pairs,
    strategy_versions,
    system_health_snapshots,
    users,
    work_units,
)
from quant_platform.db_cli import upgrade_database


@pytest.fixture(scope="session")
def migrated_database() -> str:
    url = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql+psycopg://quantlab:quantlab@127.0.0.1:55432/quantlab_test",
    )
    upgrade_database(url)
    return url


@pytest.fixture(autouse=True)
def database_state(monkeypatch, request: pytest.FixtureRequest) -> str:
    if request.node.get_closest_marker("no_database") is not None:
        return ""

    url = request.getfixturevalue("migrated_database")
    engine = open_database(url)
    with engine.begin() as connection:
        connection.execute(delete(platform_config_revisions))
        connection.execute(delete(platform_configs))
        connection.execute(delete(runtime_secrets))
        connection.execute(delete(audit_events))
        connection.execute(delete(auth_sessions))
        connection.execute(delete(users))
        connection.execute(delete(alerts))
        connection.execute(delete(system_health_snapshots))
        connection.execute(delete(research_campaign_events))
        connection.execute(delete(research_campaigns))
        connection.execute(delete(research_program_events))
        connection.execute(delete(research_programs))
        connection.execute(delete(broker_gateway_events))
        connection.execute(delete(broker_gateway_attempts))
        connection.execute(delete(broker_gateway_children))
        connection.execute(delete(broker_gateway_nonces))
        connection.execute(delete(broker_gateway_parents))
        connection.execute(delete(broker_events))
        connection.execute(delete(broker_reconciliations))
        connection.execute(delete(broker_order_outbox))
        connection.execute(delete(broker_destinations))
        connection.execute(delete(schedule_runs))
        connection.execute(delete(allocation_schedule_members))
        connection.execute(delete(allocation_schedule_groups))
        connection.execute(delete(schedules))
        connection.execute(delete(risk_events))
        connection.execute(delete(pair_portfolio_risk_events))
        connection.execute(delete(pair_portfolio_reviews))
        connection.execute(delete(pair_portfolio_nav))
        connection.execute(delete(pair_paper_fills))
        connection.execute(delete(pair_paper_orders))
        connection.execute(delete(pair_portfolio_batches))
        connection.execute(delete(pair_paper_portfolios))
        connection.execute(delete(portfolio_reviews))
        connection.execute(delete(strategy_allocation_events))
        connection.execute(delete(strategy_allocation_nav))
        connection.execute(delete(strategy_allocation_members))
        connection.execute(delete(strategy_allocations))
        connection.execute(delete(portfolio_nav))
        connection.execute(delete(paper_positions))
        connection.execute(delete(paper_fills))
        connection.execute(delete(paper_orders))
        connection.execute(delete(portfolio_batches))
        connection.execute(delete(paper_portfolios))
        connection.execute(delete(strategy_events))
        connection.execute(delete(parameter_experiment_trials))
        connection.execute(delete(parameter_experiments))
        connection.execute(delete(backtest_runs))
        connection.execute(delete(strategy_factors))
        connection.execute(delete(strategy_pairs))
        connection.execute(delete(strategy_versions))
        connection.execute(delete(strategies))
        connection.execute(delete(research_events))
        connection.execute(delete(factor_evaluations))
        connection.execute(delete(factor_candidates))
        connection.execute(delete(research_runs))
        connection.execute(delete(data_tasks))
        connection.execute(delete(jobs))
        connection.execute(delete(work_units))
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("AUTH_MODE", "disabled")
    return url


@pytest.fixture
def database_url(database_state: str) -> str:
    return database_state
