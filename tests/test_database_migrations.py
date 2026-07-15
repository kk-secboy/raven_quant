from sqlalchemy import inspect, text

from quant_data.database import open_database


def test_database_is_at_versioned_control_plane_schema(database_url: str) -> None:
    engine = open_database(database_url)
    inspector = inspect(engine)
    assert set(inspector.get_table_names(schema="quantlab")) >= {
        "alembic_version",
        "jobs",
        "research_runs",
        "factor_candidates",
        "factor_evaluations",
        "recommendation_portfolios",
        "recommendation_snapshots",
        "recommendation_holdings",
        "recommendation_nav",
        "research_events",
        "strategies",
        "strategy_versions",
        "strategy_factors",
        "strategy_pairs",
        "pair_paper_portfolios",
        "pair_portfolio_batches",
        "pair_paper_orders",
        "pair_paper_fills",
        "pair_portfolio_nav",
        "pair_portfolio_risk_events",
        "pair_portfolio_reviews",
        "backtest_runs",
        "parameter_experiments",
        "parameter_experiment_trials",
        "research_campaigns",
        "research_campaign_events",
        "research_programs",
        "research_program_events",
        "broker_destinations",
        "broker_order_outbox",
        "broker_events",
        "broker_reconciliations",
        "broker_gateway_parents",
        "broker_gateway_children",
        "broker_gateway_attempts",
        "broker_gateway_events",
        "broker_gateway_nonces",
        "strategy_events",
        "paper_portfolios",
        "portfolio_batches",
        "paper_orders",
        "paper_fills",
        "paper_positions",
        "portfolio_nav",
        "risk_events",
        "portfolio_reviews",
        "strategy_allocations",
        "strategy_allocation_members",
        "strategy_allocation_nav",
        "strategy_allocation_events",
        "system_health_snapshots",
        "schedules",
        "schedule_runs",
        "allocation_schedule_groups",
        "allocation_schedule_members",
        "alerts",
        "users",
        "auth_sessions",
        "audit_events",
        "work_units",
        "data_tasks",
        "platform_configs",
        "platform_config_revisions",
    }
    with engine.connect() as connection:
        revision = connection.execute(
            text("SELECT version_num FROM quantlab.alembic_version")
        ).scalar_one()
    assert revision == "0035_research_policy_v2"
    assert {"research_program_id", "dataset_identity_sha256"} <= {
        column["name"]
        for column in inspector.get_columns("research_campaigns", schema="quantlab")
    }
    assert {
        "values_sha256",
        "promoted_evaluation_id",
        "promotion_evidence_sha256",
        "promoted_by",
        "promoted_at",
    } <= {
        column["name"]
        for column in inspector.get_columns("factor_candidates", schema="quantlab")
    }
    assert {
        "artifact_sha256",
        "candidate_code_sha256",
        "candidate_values_sha256",
        "metrics_sha256",
        "policy_json",
        "policy_sha256",
        "evidence_sha256",
        "dataset_identity_sha256",
        "is_legacy",
        "submitted_values_sha256",
        "recomputed_values_sha256",
        "recompute_evidence_json",
    } <= {
        column["name"]
        for column in inspector.get_columns("factor_evaluations", schema="quantlab")
    }
    assert {"dataset_roll_policy", "dataset_lineage_id"} <= {
        column["name"] for column in inspector.get_columns("paper_portfolios", schema="quantlab")
    }
    assert {"dataset", "dataset_identity_sha256", "dataset_lineage_id"} <= {
        column["name"] for column in inspector.get_columns("portfolio_batches", schema="quantlab")
    }
    assert {
        "dataset_roll_policy",
        "dataset_lineage_id",
        "execution_roll_policy",
        "execution_lineage_id",
    } <= {
        column["name"]
        for column in inspector.get_columns("pair_paper_portfolios", schema="quantlab")
    }
    assert {
        "dataset",
        "dataset_identity_sha256",
        "dataset_lineage_id",
        "execution_snapshot",
        "execution_manifest_sha256",
        "execution_lineage_id",
    } <= {
        column["name"]
        for column in inspector.get_columns("pair_portfolio_batches", schema="quantlab")
    }
    strategy_version_columns = {
        column["name"]
        for column in inspector.get_columns("strategy_versions", schema="quantlab")
    }
    assert "strategy_type" in strategy_version_columns
    assert "is_legacy" in strategy_version_columns
    assert {"execution_dataset", "is_legacy"} <= {
        column["name"] for column in inspector.get_columns("backtest_runs", schema="quantlab")
    }
    for table_name in (
        "paper_portfolios",
        "paper_orders",
        "paper_fills",
        "pair_paper_portfolios",
        "pair_paper_orders",
        "pair_paper_fills",
    ):
        assert "is_legacy" in {
            column["name"] for column in inspector.get_columns(table_name, schema="quantlab")
        }
    schedule_columns = {
        column["name"] for column in inspector.get_columns("schedules", schema="quantlab")
    }
    assert {"desired_status", "suspension_reason"} <= schedule_columns
    assert "industry" in {
        column["name"] for column in inspector.get_columns("paper_positions", schema="quantlab")
    }
    assert "take_profit_stage" in {
        column["name"] for column in inspector.get_columns("paper_positions", schema="quantlab")
    }
    risk_columns = {
        column["name"] for column in inspector.get_columns("risk_events", schema="quantlab")
    }
    allocation_event_columns = {
        column["name"]
        for column in inspector.get_columns("strategy_allocation_events", schema="quantlab")
    }
    assert {"acknowledged_by", "resolved_by", "resolved_at", "resolution_reason"} <= risk_columns
    assert {
        "acknowledged_by",
        "acknowledged_at",
        "resolved_by",
        "resolved_at",
        "resolution_reason",
    } <= allocation_event_columns
    assert "is_legacy" in {
        column["name"]
        for column in inspector.get_columns("strategy_allocations", schema="quantlab")
    }
