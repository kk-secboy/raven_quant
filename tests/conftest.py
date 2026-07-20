from __future__ import annotations

import os

import pytest
from sqlalchemy import text

from quant_data.database import open_database
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
            connection.execute(text(f"TRUNCATE TABLE {names} RESTART IDENTITY CASCADE"))
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("AUTH_MODE", "disabled")
    return url


@pytest.fixture
def database_url(database_state: str) -> str:
    return database_state
