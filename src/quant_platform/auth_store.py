from __future__ import annotations

import hashlib
import re
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from sqlalchemy import func, insert, select, text, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import audit_events, auth_sessions, open_database, row_dict, users

ROLES = {"admin", "researcher", "operator", "viewer"}
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,64}$")


class AuthenticationError(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(UTC)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthStore:
    """Local user, Argon2 password, revocable session, and audit repository."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)
        self.hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)
        self._dummy_hash = self.hasher.hash("not-a-real-quantlab-password-42!")

    @staticmethod
    def validate_password(password: str) -> None:
        if len(password) < 12 or len(password) > 256:
            raise ValueError("password must contain 12 to 256 characters")
        if not any(character.isalpha() for character in password):
            raise ValueError("password must contain a letter")
        if not any(character.isdigit() for character in password):
            raise ValueError("password must contain a number")
        if not any(not character.isalnum() for character in password):
            raise ValueError("password must contain a symbol")

    def user_count(self) -> int:
        with self.engine.connect() as connection:
            return int(connection.scalar(select(func.count()).select_from(users)) or 0)

    def bootstrap_admin(
        self,
        *,
        username: str,
        display_name: str,
        password: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        self._validate_identity(username, display_name, "admin", password)
        current = now or _now()
        user_id = uuid.uuid4().hex
        with self.engine.begin() as connection:
            connection.execute(text("SELECT pg_advisory_xact_lock(hashtext('quantlab-bootstrap'))"))
            if connection.scalar(select(func.count()).select_from(users)):
                raise ValueError("the initial administrator already exists")
            connection.execute(
                insert(users).values(
                    id=user_id,
                    username=username.lower(),
                    display_name=display_name.strip(),
                    role="admin",
                    password_hash=self.hasher.hash(password),
                    active=True,
                    failed_login_attempts=0,
                    password_changed_at=current,
                    created_at=current,
                    updated_at=current,
                )
            )
        return self.get_user(user_id)

    def create_user(
        self,
        *,
        username: str,
        display_name: str,
        role: str,
        password: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        self._validate_identity(username, display_name, role, password)
        current = now or _now()
        user_id = uuid.uuid4().hex
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(users).values(
                        id=user_id,
                        username=username.lower(),
                        display_name=display_name.strip(),
                        role=role,
                        password_hash=self.hasher.hash(password),
                        active=True,
                        failed_login_attempts=0,
                        password_changed_at=current,
                        created_at=current,
                        updated_at=current,
                    )
                )
        except IntegrityError as exc:
            raise ValueError(f"username {username!r} already exists") from exc
        return self.get_user(user_id)

    def login(
        self,
        *,
        username: str,
        password: str,
        session_hours: int,
        ip_hash: str | None,
        user_agent: str | None,
        now: datetime | None = None,
    ) -> tuple[dict[str, Any], str, datetime]:
        current = now or _now()
        normalized = username.strip().lower()
        error: str | None = None
        user_id: str | None = None
        token: str | None = None
        expires_at: datetime | None = None
        with self.engine.begin() as connection:
            row = connection.execute(
                select(users).where(users.c.username == normalized).with_for_update()
            ).first()
            if row is None:
                try:
                    self.hasher.verify(self._dummy_hash, password)
                except VerificationError:
                    pass
                error = "invalid username or password"
            elif not row.active:
                error = "invalid username or password"
            elif row.locked_until and row.locked_until > current:
                error = "account is temporarily locked"
            else:
                try:
                    valid = self.hasher.verify(row.password_hash, password)
                except (VerifyMismatchError, VerificationError, InvalidHashError):
                    valid = False
                if not valid:
                    attempts = int(row.failed_login_attempts) + 1
                    locked_until = current + timedelta(minutes=15) if attempts >= 5 else None
                    connection.execute(
                        update(users)
                        .where(users.c.id == row.id)
                        .values(
                            failed_login_attempts=attempts,
                            locked_until=locked_until,
                            updated_at=current,
                        )
                    )
                    error = "invalid username or password"
                else:
                    password_hash = row.password_hash
                    if self.hasher.check_needs_rehash(password_hash):
                        password_hash = self.hasher.hash(password)
                    connection.execute(
                        update(users)
                        .where(users.c.id == row.id)
                        .values(
                            password_hash=password_hash,
                            failed_login_attempts=0,
                            locked_until=None,
                            last_login_at=current,
                            updated_at=current,
                        )
                    )
                    token = secrets.token_urlsafe(48)
                    expires_at = current + timedelta(hours=session_hours)
                    connection.execute(
                        insert(auth_sessions).values(
                            id=uuid.uuid4().hex,
                            user_id=row.id,
                            token_hash=_token_hash(token),
                            expires_at=expires_at,
                            last_seen_at=current,
                            created_at=current,
                            ip_hash=ip_hash,
                            user_agent=(user_agent or "")[:500],
                        )
                    )
                    user_id = str(row.id)
        if error:
            raise AuthenticationError(error)
        if user_id is None or token is None or expires_at is None:
            raise RuntimeError("authentication did not produce a session")
        return self.get_user(user_id), token, expires_at

    def validate_session(
        self,
        token: str | None,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        if not token:
            return None
        current = now or _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(
                    users,
                    auth_sessions.c.id.label("session_id"),
                    auth_sessions.c.expires_at,
                    auth_sessions.c.last_seen_at,
                    auth_sessions.c.revoked_at,
                )
                .join(users, users.c.id == auth_sessions.c.user_id)
                .where(auth_sessions.c.token_hash == _token_hash(token))
            ).first()
            if (
                row is None
                or row.revoked_at is not None
                or row.expires_at <= current
                or not row.active
            ):
                return None
            if row.last_seen_at < current - timedelta(minutes=5):
                connection.execute(
                    update(auth_sessions)
                    .where(auth_sessions.c.id == row.session_id)
                    .values(last_seen_at=current)
                )
            result = self._user_row(row)
            result["session_id"] = str(row.session_id)
            result["session_expires_at"] = row.expires_at.isoformat(timespec="seconds")
            return result

    def logout(self, token: str | None) -> None:
        if not token:
            return
        with self.engine.begin() as connection:
            connection.execute(
                update(auth_sessions)
                .where(
                    auth_sessions.c.token_hash == _token_hash(token),
                    auth_sessions.c.revoked_at.is_(None),
                )
                .values(revoked_at=_now())
            )

    def change_password(
        self,
        user_id: str,
        *,
        current_password: str,
        new_password: str,
        keep_session_id: str | None,
    ) -> None:
        self.validate_password(new_password)
        now = _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(users).where(users.c.id == user_id).with_for_update()
            ).one()
            try:
                self.hasher.verify(row.password_hash, current_password)
            except (VerifyMismatchError, VerificationError, InvalidHashError) as exc:
                raise AuthenticationError("current password is incorrect") from exc
            connection.execute(
                update(users)
                .where(users.c.id == user_id)
                .values(password_hash=self.hasher.hash(new_password), password_changed_at=now)
            )
            revoke = update(auth_sessions).where(
                auth_sessions.c.user_id == user_id,
                auth_sessions.c.revoked_at.is_(None),
            )
            if keep_session_id:
                revoke = revoke.where(auth_sessions.c.id != keep_session_id)
            connection.execute(revoke.values(revoked_at=now))

    def set_active(self, user_id: str, active: bool) -> dict[str, Any]:
        now = _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(users).where(users.c.id == user_id).with_for_update()
            ).first()
            if row is None:
                raise KeyError(user_id)
            if not active and row.role == "admin":
                active_admins = connection.scalar(
                    select(func.count())
                    .select_from(users)
                    .where(users.c.role == "admin", users.c.active.is_(True))
                )
                if int(active_admins or 0) <= 1:
                    raise ValueError("the last active administrator cannot be disabled")
            connection.execute(
                update(users).where(users.c.id == user_id).values(active=active, updated_at=now)
            )
            if not active:
                connection.execute(
                    update(auth_sessions)
                    .where(
                        auth_sessions.c.user_id == user_id,
                        auth_sessions.c.revoked_at.is_(None),
                    )
                    .values(revoked_at=now)
                )
        return self.get_user(user_id)

    def get_user(self, user_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(select(users).where(users.c.id == user_id)).first()
        if row is None:
            raise KeyError(user_id)
        return self._user_row(row)

    def list_users(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(select(users).order_by(users.c.created_at).limit(limit))
            return [self._user_row(row) for row in rows]

    def audit(
        self,
        *,
        user: dict[str, Any] | None,
        username: str,
        action: str,
        method: str,
        path: str,
        status_code: int,
        ip_hash: str | None,
        user_agent: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                insert(audit_events).values(
                    user_id=user.get("id") if user else None,
                    username=username,
                    action=action,
                    method=method,
                    path=path,
                    status_code=status_code,
                    ip_hash=ip_hash,
                    user_agent=(user_agent or "")[:500],
                    details_json=details or {},
                    created_at=_now(),
                )
            )

    def list_audit(self, limit: int = 500) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(audit_events).order_by(audit_events.c.created_at.desc()).limit(limit)
            )
            result = []
            for row in rows:
                item = row_dict(row)
                item["details"] = item.pop("details_json")
                result.append(item)
            return result

    @staticmethod
    def hash_ip(value: str | None) -> str | None:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24] if value else None

    @classmethod
    def _validate_identity(
        cls,
        username: str,
        display_name: str,
        role: str,
        password: str,
    ) -> None:
        if not USERNAME_PATTERN.fullmatch(username.strip()):
            raise ValueError(
                "username must contain 3-64 letters, digits, dots, dashes, or underscores"
            )
        if not display_name.strip() or len(display_name.strip()) > 100:
            raise ValueError("display name is required and must not exceed 100 characters")
        if role not in ROLES:
            raise ValueError("unsupported role")
        cls.validate_password(password)

    @staticmethod
    def _user_row(row: Any) -> dict[str, Any]:
        source = row_dict(row)
        public_fields = (
            "id",
            "username",
            "display_name",
            "role",
            "active",
            "failed_login_attempts",
            "locked_until",
            "last_login_at",
            "password_changed_at",
            "created_at",
            "updated_at",
        )
        return {key: source.get(key) for key in public_fields}
