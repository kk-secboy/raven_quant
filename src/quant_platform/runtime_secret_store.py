from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import delete, select, update

from quant_data.database import open_database, row_dict, runtime_secrets


class RuntimeSecretStore:
    """Encrypts mutable runtime credentials before persisting them in PostgreSQL."""

    def __init__(self, database_url: str, encryption_key: str) -> None:
        self.engine = open_database(database_url)
        self._fernet: Fernet | None = None
        if encryption_key:
            try:
                self._fernet = Fernet(encryption_key.encode("ascii"))
            except (ValueError, TypeError) as exc:
                raise ValueError("PLATFORM_SECRET_KEY must be a valid Fernet key") from exc

    @property
    def writable(self) -> bool:
        return self._fernet is not None

    def put(
        self,
        name: str,
        payload: dict[str, str],
        *,
        metadata: dict[str, Any],
        updated_by: str | None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if self._fernet is None:
            raise ValueError("dynamic secret storage is not configured")
        timestamp = now or datetime.now(UTC)
        ciphertext = self._fernet.encrypt(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
        values = {
            "ciphertext": ciphertext,
            "metadata_json": metadata,
            "updated_at": timestamp,
            "updated_by": updated_by,
        }
        with self.engine.begin() as connection:
            existing = connection.scalar(
                select(runtime_secrets.c.name).where(runtime_secrets.c.name == name)
            )
            if existing:
                connection.execute(
                    update(runtime_secrets).where(runtime_secrets.c.name == name).values(**values)
                )
            else:
                connection.execute(runtime_secrets.insert().values(name=name, **values))
        return self.describe(name) or {}

    def get(self, name: str) -> dict[str, str] | None:
        with self.engine.connect() as connection:
            ciphertext = connection.scalar(
                select(runtime_secrets.c.ciphertext).where(runtime_secrets.c.name == name)
            )
        if ciphertext is None:
            return None
        if self._fernet is None:
            raise ValueError("dynamic secret storage is not configured")
        try:
            plaintext = self._fernet.decrypt(ciphertext.encode("ascii"))
        except InvalidToken as exc:
            raise ValueError("stored runtime secret cannot be decrypted") from exc
        value = json.loads(plaintext)
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in value.items()
        ):
            raise ValueError("stored runtime secret has an invalid payload")
        return value

    def describe(self, name: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(
                        runtime_secrets.c.name,
                        runtime_secrets.c.metadata_json,
                        runtime_secrets.c.updated_at,
                        runtime_secrets.c.updated_by,
                    ).where(runtime_secrets.c.name == name)
                )
                .mappings()
                .first()
            )
        return row_dict(row) if row else None

    def health(self) -> dict[str, Any]:
        """Validate that every stored runtime secret is decryptable without exposing it."""

        with self.engine.connect() as connection:
            names = list(
                connection.scalars(select(runtime_secrets.c.name).order_by(runtime_secrets.c.name))
            )
        if self._fernet is None:
            return {
                "status": "unavailable" if names else "bootstrap_required",
                "message": (
                    "encrypted runtime secrets exist but PLATFORM_SECRET_KEY is unavailable"
                    if names
                    else "PLATFORM_SECRET_KEY is not configured"
                ),
                "record_count": len(names),
            }
        try:
            for name in names:
                self.get(str(name))
        except ValueError:
            return {
                "status": "unavailable",
                "message": "one or more encrypted runtime secrets cannot be decrypted",
                "record_count": len(names),
            }
        return {
            "status": "ok",
            "message": f"encrypted runtime storage ready; {len(names)} records validated",
            "record_count": len(names),
        }

    def clear(self, name: str) -> bool:
        with self.engine.begin() as connection:
            result = connection.execute(
                delete(runtime_secrets).where(runtime_secrets.c.name == name)
            )
        return bool(result.rowcount)
