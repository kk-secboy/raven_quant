#!/usr/bin/env python3
"""Recover a local QuantLab administrator from the trusted server console."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime

from argon2 import PasswordHasher
from sqlalchemy import insert, select, update

from quant_data.database import audit_events, auth_sessions, open_database, users
from quant_platform.auth_store import AuthStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default="admin")
    parser.add_argument("--actor", default="server-console")
    args = parser.parse_args()
    password = sys.stdin.readline().rstrip("\r\n")
    if not password:
        raise SystemExit("a new password must be supplied on standard input")
    AuthStore.validate_password(password)
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("DATABASE_URL is required")
    now = datetime.now(UTC)
    engine = open_database(database_url)
    hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)
    with engine.begin() as connection:
        row = connection.execute(
            select(users).where(users.c.username == args.username.lower()).with_for_update()
        ).first()
        if row is None:
            raise SystemExit(f"user {args.username!r} was not found")
        if row.role != "admin":
            raise SystemExit("console recovery is restricted to administrator accounts")
        connection.execute(
            update(users)
            .where(users.c.id == row.id)
            .values(
                password_hash=hasher.hash(password),
                password_changed_at=now,
                failed_login_attempts=0,
                locked_until=None,
                active=True,
                updated_at=now,
            )
        )
        connection.execute(
            update(auth_sessions)
            .where(auth_sessions.c.user_id == row.id, auth_sessions.c.revoked_at.is_(None))
            .values(revoked_at=now)
        )
        connection.execute(
            insert(audit_events).values(
                user_id=row.id,
                username=row.username,
                action="auth.password_recovered",
                method="CONSOLE",
                path="scripts/reset_admin_password.py",
                status_code=200,
                ip_hash=None,
                user_agent="server-console",
                details_json={"actor": args.actor, "sessions_revoked": True},
                created_at=now,
            )
        )
    print(f"password recovered for {args.username}; existing sessions revoked")


if __name__ == "__main__":
    main()
