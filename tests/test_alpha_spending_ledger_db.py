from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import insert, inspect, select, update
from sqlalchemy.exc import DBAPIError

from quant_data.database import (
    capital_oos_alpha_batches,
    capital_oos_legacy_attempts,
    oos_vintages,
    open_database,
)
from quant_data.snapshot_lineage import canonical_sha256
from quant_platform.alpha_spending_integration import (
    capital_oos_family_manifest,
    capital_oos_family_manifest_sha256,
)
from quant_platform.alpha_spending_ledger import CapitalOOSAlphaLedgerStore
from quant_platform.strategy_store import StrategyStore

LINEAGE = "1" * 64
DATASET_IDENTITY = "2" * 64
MANDATE = capital_oos_family_manifest(
    "csi300",
    "SH000300",
    5,
    "a" * 64,
    "b" * 64,
    "c" * 64,
)


def _dates(start: str, periods: int) -> list[str]:
    return [item.date().isoformat() for item in pd.bdate_range(start, periods=periods)]


def _reserve(
    store: CapitalOOSAlphaLedgerStore,
    *,
    key: str = "final-oos-one",
    start: str = "2020-01-01",
    bundle: str = "b" * 64,
    lineage: str = LINEAGE,
    dataset_identity: str = DATASET_IDENTITY,
    mandate: dict | None = None,
) -> dict:
    final = _dates(start, 252)
    embargo_end = pd.Timestamp(final[0]) - pd.offsets.BDay(1)
    embargo = [item.date().isoformat() for item in pd.bdate_range(end=embargo_end, periods=20)]
    research_data_end = (
        pd.Timestamp(embargo[0]) - pd.offsets.BDay(1)
    ).date().isoformat()
    return store.reserve_batch(
        dataset_lineage_id=lineage,
        dataset_identity_sha256=dataset_identity,
        stable_mandate=mandate or MANDATE,
        batch_key=key,
        frozen_bundle_manifest_sha256=bundle,
        frozen_baseline_manifest_sha256="e" * 64,
        research_data_end=research_data_end,
        final_oos_trading_dates=final,
        embargo_trading_dates=embargo,
    )


def test_capital_vintage_scope_is_family_horizon_not_shared_lineage(
    database_url: str,
) -> None:
    store = CapitalOOSAlphaLedgerStore(database_url)
    short_mandate = MANDATE
    swing_mandate = capital_oos_family_manifest(
        "csi300",
        "SH000300",
        63,
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )
    short = _reserve(store, key="short-family-window", mandate=short_mandate)
    swing = _reserve(store, key="swing-family-window", mandate=swing_mandate)
    now = datetime.now(UTC)
    engine = open_database(database_url)
    with engine.begin() as connection:
        for batch, label in ((short, 5), (swing, 63)):
            StrategyStore._seal_and_consume_oos_vintage(
                connection,
                strategy_version_id=f"strategy-{label}",
                candidate_ids=[],
                sealed_member_set={
                    "candidate_ids": [],
                    "capital_scope_fixture": {"label_horizon_days": label},
                },
                dataset_identities=set(),
                dataset_lineage_id=LINEAGE,
                dataset="shared-snapshot",
                test_start=pd.Timestamp(batch["final_oos_start"]).date(),
                test_end=pd.Timestamp(batch["final_oos_end"]).date(),
                consumed_at=now,
                capital_oos_alpha_batch_id=str(batch["id"]),
                capital_oos_dataset_identity_sha256=DATASET_IDENTITY,
            )
        rows = connection.execute(select(oos_vintages)).all()

    assert len(rows) == 2
    scopes = {str(row.scope) for row in rows}
    assert scopes == {
        "alpha-family:"
        f"{capital_oos_family_manifest_sha256(short_mandate)}:label:5",
        "alpha-family:"
        f"{capital_oos_family_manifest_sha256(swing_mandate)}:label:63",
    }
    assert {str(row.dataset_lineage_id) for row in rows} == {LINEAGE}

    with pytest.raises(
        ValueError,
        match="overlaps a reserved or consumed OOS vintage",
    ):
        with engine.begin() as connection:
            StrategyStore._seal_and_consume_oos_vintage(
                connection,
                strategy_version_id="strategy-short-reuse",
                candidate_ids=[],
                sealed_member_set={
                    "candidate_ids": [],
                    "capital_scope_fixture": {"label_horizon_days": 5},
                },
                dataset_identities=set(),
                dataset_lineage_id=LINEAGE,
                dataset="shared-snapshot-next",
                test_start=pd.Timestamp(short["final_oos_start"]).date()
                + timedelta(days=1),
                test_end=pd.Timestamp(short["final_oos_end"]).date()
                + timedelta(days=1),
                consumed_at=now,
                capital_oos_alpha_batch_id=str(short["id"]),
                capital_oos_dataset_identity_sha256=DATASET_IDENTITY,
            )


def _link_vintage(
    store: CapitalOOSAlphaLedgerStore,
    database_url: str,
    batch: dict,
) -> str:
    vintage_id = f"vintage-{batch['id'][:24]}"
    link_contract = store.vintage_link_contract(batch["id"])
    sealed_members = {
        "candidate_ids": [],
        **link_contract["sealed_candidate_set_patch"],
    }
    now = datetime.now(UTC)
    with open_database(database_url).begin() as connection:
        connection.execute(
            insert(oos_vintages).values(
                id=vintage_id,
                scope="standalone:global",
                dataset_identity=batch["dataset_identity_sha256"],
                dataset_lineage_id=batch["dataset_lineage_id"],
                test_start=pd.Timestamp(batch["final_oos_start"]).date(),
                test_end=pd.Timestamp(batch["final_oos_end"]).date(),
                sealed_at=now,
                first_opened_at=now,
                consumed_at=now,
                capital_oos_alpha_batch_id=link_contract[
                    "capital_oos_alpha_batch_id"
                ],
                sealed_candidate_set_json=sealed_members,
                sealed_candidate_set_sha256=canonical_sha256(sealed_members),
                created_at=now,
            )
        )
    return vintage_id


def test_migration_exposes_capital_only_family_and_batch(database_url: str) -> None:
    inspector = inspect(open_database(database_url))
    tables = set(inspector.get_table_names(schema="quantlab"))
    assert {
        "capital_oos_alpha_families",
        "capital_oos_alpha_batches",
        "capital_oos_legacy_attempts",
    } <= tables
    assert "alpha_spending_ledgers" not in tables
    columns = {
        item["name"]
        for item in inspector.get_columns("capital_oos_alpha_batches", schema="quantlab")
    }
    assert {
        "frozen_bundle_manifest_sha256",
        "frozen_baseline_manifest_sha256",
        "hypothesis_count",
        "final_oos_start",
        "trading_day_count",
        "embargo_trading_day_count",
        "raw_p_value",
        "dataset_lineage_id",
        "dataset_identity_sha256",
        "research_data_end",
    } <= columns
    family_columns = {
        item["name"]
        for item in inspector.get_columns("capital_oos_alpha_families", schema="quantlab")
    }
    assert "mandate_json" in family_columns
    assert "dataset_lineage_id" not in family_columns
    vintage_columns = {
        item["name"]
        for item in inspector.get_columns("oos_vintages", schema="quantlab")
    }
    assert "capital_oos_alpha_batch_id" in vintage_columns


def test_reservation_is_idempotent_but_overlapping_window_is_rejected(
    database_url: str,
) -> None:
    store = CapitalOOSAlphaLedgerStore(database_url)
    first = _reserve(store)
    repeated = _reserve(store)
    assert repeated["id"] == first["id"]
    assert first["hypothesis_count"] == 1
    assert first["batch_alpha"] == "0.025"
    store.settle_batch(
        first["id"],
        failed=True,
        failure_reason="settle the first reservation before testing overlap precedence",
    )
    with pytest.raises(ValueError, match="overlaps"):
        _reserve(store, key="another-bundle-same-window", bundle="c" * 64)


def test_disjoint_successor_spends_next_alpha_without_snapshot_reset(
    database_url: str,
) -> None:
    store = CapitalOOSAlphaLedgerStore(database_url)
    first = _reserve(store)
    store.settle_batch(
        first["id"],
        failed=True,
        failure_reason="first disjoint window failed before successor opened",
    )
    second = _reserve(
        store,
        key="final-oos-two",
        start="2022-01-03",
        bundle="c" * 64,
        lineage="3" * 64,
        dataset_identity="4" * 64,
    )
    assert first["ordinal"] == 1
    assert second["ordinal"] == 2
    assert Decimal(second["batch_alpha"]) == Decimal("0.0083333333333333333333333333")
    families = store.list_families()
    assert len(families) == 1
    assert families[0]["mandate_json"] == MANDATE
    assert Decimal(families[0]["reserved_alpha"]) < Decimal("0.05")


def test_family_allows_only_one_unsettled_reserved_batch(database_url: str) -> None:
    store = CapitalOOSAlphaLedgerStore(database_url)
    first = _reserve(store)
    with pytest.raises(ValueError, match="unsettled reserved batch"):
        _reserve(
            store,
            key="final-oos-two",
            start="2022-01-03",
            bundle="c" * 64,
        )
    repeated = _reserve(store)
    assert repeated["id"] == first["id"]


def test_legacy_oos_is_failed_closed_until_immutable_reconciliation(
    database_url: str,
) -> None:
    now = datetime.now(UTC)
    sealed_members = {"candidate_ids": ["legacy-candidate"]}
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            insert(oos_vintages).values(
                id="legacy-vintage-one",
                scope="standalone:global",
                dataset_identity="legacy-dataset",
                dataset_lineage_id=None,
                test_start=pd.Timestamp("2018-01-02").date(),
                test_end=pd.Timestamp("2018-12-31").date(),
                sealed_at=now,
                first_opened_at=now,
                consumed_at=now,
                sealed_candidate_set_json=sealed_members,
                sealed_candidate_set_sha256=canonical_sha256(sealed_members),
                created_at=now,
            )
        )
    store = CapitalOOSAlphaLedgerStore(database_url)
    with pytest.raises(ValueError, match="explicit immutable reconciliation"):
        _reserve(store)
    legacy = store.list_legacy_attempts(status="unreconciled")
    assert len(legacy) == 1
    assert legacy[0]["raw_p_value"] == 1.0
    assert legacy[0]["passed"] is False
    reconciled = store.reconcile_legacy_attempts(
        stable_mandate=MANDATE,
        legacy_attempt_ids=[legacy[0]["id"]],
        actor="risk-owner",
        reason="Conservative attribution of every pre-ledger capital OOS opening.",
        supporting_evidence={"audit_ticket": "legacy-audit-one"},
    )
    assert reconciled[0]["ordinal"] == 1
    assert reconciled[0]["spent_alpha"] == "0.025"
    replay = store.reconcile_legacy_attempts(
        stable_mandate=MANDATE,
        legacy_attempt_ids=[legacy[0]["id"]],
        actor="risk-owner",
        reason="Conservative attribution of every pre-ledger capital OOS opening.",
        supporting_evidence={"audit_ticket": "legacy-audit-one"},
    )
    assert replay[0]["reconciliation_sha256"] == reconciled[0]["reconciliation_sha256"]
    with pytest.raises(ValueError, match="immutable"):
        store.reconcile_legacy_attempts(
            stable_mandate=MANDATE,
            legacy_attempt_ids=[legacy[0]["id"]],
            actor="risk-owner",
            reason="Changed attribution after settlement.",
            supporting_evidence={"audit_ticket": "legacy-audit-one"},
        )
    with pytest.raises(DBAPIError, match="immutable"):
        with engine.begin() as connection:
            connection.execute(
                update(capital_oos_legacy_attempts)
                .where(capital_oos_legacy_attempts.c.id == legacy[0]["id"])
                .values(reconciliation_json={"tampered": True})
            )
    batch = _reserve(store)
    assert batch["ordinal"] == 2
    assert Decimal(batch["batch_alpha"]) == Decimal("0.0083333333333333333333333333")


def test_failed_opened_window_settles_p_one_and_exact_retry_is_idempotent(
    database_url: str,
) -> None:
    store = CapitalOOSAlphaLedgerStore(database_url)
    batch = _reserve(store)
    settled = store.settle_batch(
        batch["id"],
        failed=True,
        failure_reason="formal OOS execution failed after bounded retry",
        supporting_evidence={"job_id": "job-one"},
    )
    assert settled["raw_p_value"] == 1.0
    assert settled["passed"] is False
    assert settled["failure_recorded"] is True
    replay = store.settle_batch(
        batch["id"],
        failed=True,
        failure_reason="formal OOS execution failed after bounded retry",
        supporting_evidence={"job_id": "job-one"},
    )
    assert replay["settlement_evidence_sha256"] == settled["settlement_evidence_sha256"]
    with pytest.raises(ValueError, match="immutable"):
        store.settle_batch(
            batch["id"],
            failed=True,
            failure_reason="changed reason",
            supporting_evidence={"job_id": "job-one"},
        )


def test_failed_execution_auto_binds_existing_vintage(database_url: str) -> None:
    store = CapitalOOSAlphaLedgerStore(database_url)
    batch = _reserve(store)
    vintage_id = _link_vintage(store, database_url, batch)
    settled = store.settle_batch(
        batch["id"],
        failed=True,
        failure_reason="worker process failed after the formal OOS was opened",
        supporting_evidence={"backtest_id": "backtest-one"},
    )
    link = settled["settlement_evidence_json"]["oos_vintage_link"]
    assert link["status"] == "verified"
    assert link["oos_vintage_id"] == vintage_id


def test_positive_paired_hac_and_bootstrap_must_both_pass(database_url: str) -> None:
    store = CapitalOOSAlphaLedgerStore(database_url)
    batch = _reserve(store)
    vintage_id = _link_vintage(store, database_url, batch)
    index = pd.to_datetime(batch["trading_dates_json"])
    rng = np.random.default_rng(41)
    baseline = pd.Series(rng.normal(0.0, 0.002, len(index)), index=index)
    candidate = baseline + pd.Series(rng.normal(0.002, 0.001, len(index)), index=index)
    settled = store.settle_batch(
        batch["id"],
        candidate_net_returns=candidate,
        baseline_net_returns=baseline,
        supporting_evidence={
            "formal_oos_artifact_sha256": "d" * 64,
            "oos_vintage_id": vintage_id,
        },
    )
    assert 0.0 < settled["raw_p_value"] <= float(batch["batch_alpha"])
    assert settled["passed"] is True
    evidence = settled["settlement_evidence_json"]
    assert evidence["paired_block_bootstrap"]["hard_gate_passed"] is True
    assert evidence["paired_hac"]["contract_version"].startswith("one-sided")
    assert len(evidence["paired_return_inputs"]["candidate_net_returns_sha256"]) == 64
    assert evidence["oos_vintage_link"]["status"] == "verified"
    binding = store.get_vintage_binding(batch["id"])
    assert binding["oos_vintage_id"] == vintage_id
    assert binding["capital_oos_alpha_batch_id"] == batch["id"]
    assert store.list_legacy_attempts() == []
    replay = store.settle_batch(
        batch["id"],
        candidate_net_returns=candidate,
        baseline_net_returns=baseline,
        supporting_evidence={
            "formal_oos_artifact_sha256": "d" * 64,
            "oos_vintage_id": vintage_id,
        },
    )
    assert replay["settlement_evidence_sha256"] == settled["settlement_evidence_sha256"]
    changed = candidate.copy()
    changed.iloc[0] += 1e-9
    with pytest.raises(ValueError, match="immutable"):
        store.settle_batch(
            batch["id"],
            candidate_net_returns=changed,
            baseline_net_returns=baseline,
            supporting_evidence={
                "formal_oos_artifact_sha256": "d" * 64,
                "oos_vintage_id": vintage_id,
            },
        )


def test_strategy_binding_checks_exact_preregistration_before_vintage(
    database_url: str,
) -> None:
    store = CapitalOOSAlphaLedgerStore(database_url)
    batch = _reserve(store)
    link = store.vintage_link_contract(batch["id"])
    engine = open_database(database_url)
    with engine.begin() as connection:
        StrategyStore._validate_capital_oos_batch_binding(
            connection,
            batch_id=batch["id"],
            sealed_candidate_set_patch=link["sealed_candidate_set_patch"],
            dataset_lineage_id=batch["dataset_lineage_id"],
            dataset_identity_sha256=batch["dataset_identity_sha256"],
            research_data_end=pd.Timestamp(batch["research_data_end"]).date(),
            test_start=pd.Timestamp(batch["final_oos_start"]).date(),
            test_end=pd.Timestamp(batch["final_oos_end"]).date(),
            final_oos_trading_dates=batch["trading_dates_json"],
            embargo_trading_dates=batch["embargo_trading_dates_json"],
        )
        with pytest.raises(ValueError, match="data or window differs"):
            StrategyStore._validate_capital_oos_batch_binding(
                connection,
                batch_id=batch["id"],
                sealed_candidate_set_patch=link["sealed_candidate_set_patch"],
                dataset_lineage_id=batch["dataset_lineage_id"],
                dataset_identity_sha256="f" * 64,
                research_data_end=pd.Timestamp(batch["research_data_end"]).date(),
                test_start=pd.Timestamp(batch["final_oos_start"]).date(),
                test_end=pd.Timestamp(batch["final_oos_end"]).date(),
                final_oos_trading_dates=batch["trading_dates_json"],
                embargo_trading_dates=batch["embargo_trading_dates_json"],
            )
    vintage_id = _link_vintage(store, database_url, batch)
    with pytest.raises(DBAPIError, match="immutable"):
        with engine.begin() as connection:
            connection.execute(
                update(oos_vintages)
                .where(oos_vintages.c.id == vintage_id)
                .values(sealed_candidate_set_sha256="0" * 64)
            )


def test_settled_batch_is_database_immutable(database_url: str) -> None:
    store = CapitalOOSAlphaLedgerStore(database_url)
    batch = _reserve(store)
    store.settle_batch(
        batch["id"],
        failed=True,
        failure_reason="terminal test failure",
    )
    engine = open_database(database_url)
    with pytest.raises(DBAPIError, match="immutable"):
        with engine.begin() as connection:
            connection.execute(
                update(capital_oos_alpha_batches)
                .where(capital_oos_alpha_batches.c.id == batch["id"])
                .values(passed=True)
            )
