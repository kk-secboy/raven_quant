from __future__ import annotations

import gc
import os
from collections.abc import Iterator

import pytest
from sqlalchemy import text

from quant_data.database import open_database
from quant_platform.db_cli import upgrade_database

_TEST_WORKER_RUNTIME_IMAGE_DIGEST = "sha256:" + "d" * 64


@pytest.fixture(scope="session")
def migrated_database() -> str:
    url = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql+psycopg://quantlab:quantlab@127.0.0.1:55432/quantlab_test",
    )
    upgrade_database(url)
    return url


@pytest.fixture(autouse=True)
def database_state(monkeypatch, request: pytest.FixtureRequest) -> Iterator[str]:
    # Production releases stamp the exact worker image ID before any governed
    # v12 StrategyVersion can be created.  The disposable test runtime has no
    # release upgrader, so give every test the same valid, deterministic seal;
    # missing/mismatch tests explicitly delete or replace it themselves.
    monkeypatch.setenv(
        "QUANTLAB_WORKER_RUNTIME_IMAGE_DIGEST",
        _TEST_WORKER_RUNTIME_IMAGE_DIGEST,
    )
    if request.node.get_closest_marker("no_database") is not None:
        yield ""
        return

    monkeypatch.setenv("QUANTLAB_DATABASE_DISABLE_POOL", "1")
    url = request.getfixturevalue("migrated_database")
    engine = open_database(url)
    try:
        with engine.begin() as connection:
            # Schema-driven reset: truncate every business table in one atomic
            # statement. A hand-maintained delete list silently rots — the first
            # table added without updating it leaves FK children behind, aborts
            # this transaction, and turns the whole suite order-dependent.
            # TRUNCATE ... CASCADE is FK-order safe and does not fire row-level
            # DELETE triggers (e.g. ledger immutability guards).
            tables = connection.execute(
                text(
                    "SELECT tablename FROM pg_tables"
                    " WHERE schemaname = 'quantlab' AND tablename <> 'alembic_version'"
                )
            ).scalars().all()
            if tables:
                names = ", ".join(f'quantlab."{table}"' for table in tables)
                # Production append-only tables also reject TRUNCATE. The isolated
                # test role owns this disposable database, so suppress ordinary
                # triggers only for the atomic schema-wide reset and restore the
                # normal trigger role before the fixture transaction commits.
                connection.execute(text("SET LOCAL session_replication_role = replica"))
                connection.execute(text(f"TRUNCATE TABLE {names} RESTART IDENTITY CASCADE"))
                connection.execute(text("SET LOCAL session_replication_role = origin"))
    finally:
        # A full database suite creates hundreds of short-lived Store engines.
        # Release this fixture's pool deterministically instead of waiting for
        # cyclic garbage collection and exhausting PostgreSQL max_connections.
        engine.dispose()
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("AUTH_MODE", "disabled")
    yield url
    # Store instances created by a test can own SQLAlchemy Engine/Pool cycles.
    # Collect them between cases so the disposable integration database never
    # accumulates idle sessions across the complete suite.
    gc.collect()


@pytest.fixture
def database_url(database_state: str) -> str:
    return database_state
