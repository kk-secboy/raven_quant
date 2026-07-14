from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from urllib.parse import quote, urlparse

import requests
from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.config import Settings
from quant_data.database import (
    broker_destinations,
    broker_events,
    broker_order_outbox,
    broker_reconciliations,
    open_database,
    paper_fills,
    paper_orders,
    paper_portfolios,
    paper_positions,
    portfolio_batches,
    row_dict,
)

from .alert_store import AlertStore
from .broker_reconciliation import compare_broker_snapshot, validate_broker_snapshot
from .execution_algorithms import build_execution_slices, normalize_execution_policy
from .runtime_secret_store import RuntimeSecretStore


class BrokerGatewayError(RuntimeError):
    pass


def validate_broker_gateway_credentials(
    gateway_url: str,
    hmac_secret: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, str]:
    """Validate a sandbox gateway pair without making an outbound request."""

    normalized_url = gateway_url.strip().rstrip("/")
    normalized_secret = hmac_secret.strip()
    if allow_empty and not normalized_url and not normalized_secret:
        return "", ""
    if not normalized_url or not normalized_secret:
        raise ValueError("broker gateway URL and HMAC secret must be configured together")
    parsed = urlparse(normalized_url)
    local = parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "broker-gateway",
        "host.docker.internal",
        "gateway.docker.internal",
    }
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("broker gateway URL must be an absolute HTTP(S) URL")
    if parsed.scheme != "https" and not local:
        raise ValueError("remote broker gateways require HTTPS")
    if len(normalized_secret) < 32:
        raise ValueError("broker HMAC secret must contain at least 32 characters")
    if any(character.isspace() for character in normalized_secret):
        raise ValueError("broker HMAC secret must not contain whitespace")
    return normalized_url, normalized_secret


def _now() -> datetime:
    return datetime.now(UTC)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class SignedBrokerGateway:
    """HMAC-authenticated client for a separately deployed sandbox broker gateway."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._validate_configuration()

    def health(self) -> dict[str, Any]:
        response = self._request("GET", "/v1/health", None)
        if response.get("environment") != "sandbox" or response.get("status") != "ok":
            raise BrokerGatewayError("broker gateway did not attest sandbox readiness")
        return response

    def submit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._request("POST", "/v1/orders", payload)
        order_id = str(response.get("order_id") or "").strip()
        if (
            response.get("status") != "accepted"
            or response.get("environment") != "sandbox"
            or not order_id
        ):
            raise BrokerGatewayError("sandbox gateway did not accept the order")
        if response.get("client_order_id") != payload["client_order_id"]:
            raise BrokerGatewayError("sandbox gateway returned a mismatched client order id")
        return response

    def snapshot(self, account_ref: str) -> dict[str, Any]:
        path = f"/v1/snapshot?account_ref={quote(account_ref, safe='')}"
        response = self._request("GET", path, None)
        try:
            return validate_broker_snapshot(response, account_ref=account_ref)
        except ValueError as exc:
            raise BrokerGatewayError(str(exc)) from exc

    def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        body = _canonical_json(payload) if payload is not None else ""
        timestamp = str(int(time.time()))
        nonce = secrets.token_hex(16)
        signed = f"{timestamp}.{nonce}.{method}.{path}.{body}".encode()
        signature = hmac.new(
            self.settings.broker_hmac_secret.encode(), signed, hashlib.sha256
        ).hexdigest()
        headers = {
            "X-QuantLab-Timestamp": timestamp,
            "X-QuantLab-Nonce": nonce,
            "X-QuantLab-Signature": signature,
            "Content-Type": "application/json",
        }
        try:
            response = requests.request(
                method,
                f"{self.settings.broker_gateway_url}{path}",
                data=body or None,
                headers=headers,
                timeout=self.settings.broker_timeout_seconds,
            )
            response.raise_for_status()
            if len(response.content) > 10_000_000:
                raise BrokerGatewayError("sandbox gateway response exceeds 10 MB")
            result = response.json()
        except BrokerGatewayError:
            raise
        except (requests.RequestException, ValueError) as exc:
            raise BrokerGatewayError(f"sandbox gateway request failed: {exc}") from exc
        if not isinstance(result, dict):
            raise BrokerGatewayError("sandbox gateway returned a non-object response")
        return result

    def _validate_configuration(self) -> None:
        if self.settings.broker_mode != "sandbox":
            raise BrokerGatewayError("broker execution is disabled")
        try:
            validate_broker_gateway_credentials(
                self.settings.broker_gateway_url,
                self.settings.broker_hmac_secret,
            )
        except ValueError as exc:
            raise BrokerGatewayError(str(exc)) from exc


class BrokerStore:
    """Sandbox-only broker control plane with two-person release and a durable outbox."""

    def __init__(
        self,
        settings: Settings,
        runtime_secrets: RuntimeSecretStore | None = None,
    ) -> None:
        self.settings = settings
        self.engine = open_database(settings.database_url)
        self.alerts = AlertStore(settings.database_url)
        self.runtime_secrets = runtime_secrets or RuntimeSecretStore(
            settings.database_url, settings.platform_secret_key
        )

    def _effective_settings(self) -> Settings:
        stored = self.runtime_secrets.get("broker_gateway")
        if stored is not None:
            return replace(
                self.settings,
                broker_gateway_url=str(stored.get("gateway_url") or "").rstrip("/"),
                broker_hmac_secret=str(stored.get("hmac_secret") or ""),
            )
        return self.settings

    def readiness(self, *, probe: bool = False) -> dict[str, Any]:
        settings_error = ""
        try:
            settings = self._effective_settings()
        except ValueError as exc:
            settings = self.settings
            settings_error = str(exc)
        with self.engine.connect() as connection:
            destination_counts = {
                str(status): int(count)
                for status, count in connection.execute(
                    select(broker_destinations.c.status, func.count())
                    .group_by(broker_destinations.c.status)
                    .order_by(broker_destinations.c.status)
                )
            }
            outbox_counts = {
                str(status): int(count)
                for status, count in connection.execute(
                    select(broker_order_outbox.c.status, func.count())
                    .group_by(broker_order_outbox.c.status)
                    .order_by(broker_order_outbox.c.status)
                )
            }
        result: dict[str, Any] = {
            "status": "disabled" if settings.broker_mode == "disabled" else "configured",
            "mode": settings.broker_mode,
            "live_supported": False,
            "gateway_configured": bool(
                settings.broker_gateway_url and settings.broker_hmac_secret
            ),
            "destination_counts": destination_counts,
            "outbox_counts": outbox_counts,
            "max_order_notional": settings.broker_max_order_notional,
            "max_attempts": settings.broker_max_attempts,
        }
        locked = int(destination_counts.get("locked_mismatch", 0))
        failed = int(outbox_counts.get("failed", 0))
        if settings_error:
            result["status"] = "unavailable"
            result["message"] = f"broker credential storage failed: {settings_error}"
        elif locked or failed:
            result["status"] = "degraded"
            result["message"] = (
                f"broker boundary has {locked} reconciliation locks and {failed} failed orders"
            )
        elif settings.broker_mode == "sandbox" and not result["gateway_configured"]:
            result["status"] = "bootstrap_required"
            result["message"] = "sandbox gateway URL and HMAC secret are required"
        elif settings.broker_mode == "sandbox" and probe:
            try:
                result["gateway"] = SignedBrokerGateway(settings).health()
                result["status"] = "ok"
            except BrokerGatewayError as exc:
                result["status"] = "unavailable"
                result["message"] = str(exc)
        return result

    def create_destination(
        self,
        *,
        name: str,
        account_ref: str,
        portfolio_id: str,
        actor: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if (
            not name.strip()
            or not account_ref.strip()
            or not portfolio_id.strip()
            or not actor.strip()
        ):
            raise ValueError("name, account_ref, portfolio_id, and actor are required")
        destination_id = uuid.uuid4().hex
        now = _now()
        execution_policy = normalize_execution_policy(config)
        try:
            with self.engine.begin() as connection:
                portfolio = connection.execute(
                    select(paper_portfolios.c.id).where(
                        paper_portfolios.c.id == portfolio_id.strip()
                    )
                ).scalar_one_or_none()
                if portfolio is None:
                    raise KeyError(portfolio_id)
                connection.execute(
                    insert(broker_destinations).values(
                        id=destination_id,
                        name=name.strip(),
                        adapter="signed_http",
                        environment="sandbox",
                        account_ref=account_ref.strip(),
                        portfolio_id=portfolio_id.strip(),
                        status="disabled",
                        config_json=execution_policy,
                        created_by=actor.strip(),
                        created_at=now,
                        updated_at=now,
                    )
                )
                self._event(
                    connection,
                    destination_id=destination_id,
                    event_type="destination_created",
                    actor=actor,
                    details={"environment": "sandbox", "adapter": "signed_http"},
                )
        except IntegrityError as exc:
            raise ValueError("broker destination name or portfolio mapping already exists") from exc
        return self.get_destination(destination_id)

    def request_activation(self, destination_id: str, *, actor: str) -> dict[str, Any]:
        SignedBrokerGateway(self._effective_settings()).health()
        with self.engine.begin() as connection:
            destination = connection.execute(
                select(broker_destinations)
                .where(broker_destinations.c.id == destination_id)
                .with_for_update()
            ).first()
            if destination is None:
                raise KeyError(destination_id)
            if destination.environment != "sandbox":
                raise ValueError("only sandbox destinations are supported")
            if not destination.portfolio_id:
                raise ValueError("destination must be bound to a paper portfolio")
            if destination.status != "disabled":
                raise ValueError("only a disabled destination can request activation")
            now = _now()
            connection.execute(
                update(broker_destinations)
                .where(broker_destinations.c.id == destination_id)
                .values(
                    status="pending_activation",
                    activation_requested_by=actor.strip(),
                    activation_requested_at=now,
                    activated_by=None,
                    activated_at=None,
                    updated_at=now,
                )
            )
            self._event(
                connection,
                destination_id=destination_id,
                event_type="activation_requested",
                actor=actor,
                details={},
            )
        return self.get_destination(destination_id)

    def approve_activation(self, destination_id: str, *, actor: str) -> dict[str, Any]:
        SignedBrokerGateway(self._effective_settings()).health()
        with self.engine.begin() as connection:
            destination = connection.execute(
                select(broker_destinations)
                .where(broker_destinations.c.id == destination_id)
                .with_for_update()
            ).first()
            if destination is None:
                raise KeyError(destination_id)
            if destination.status != "pending_activation":
                raise ValueError("destination is not pending activation")
            if destination.activation_requested_by == actor.strip():
                raise ValueError("activation requires a second administrator")
            reconciliation = connection.execute(
                select(
                    broker_reconciliations.c.status,
                    broker_reconciliations.c.created_at,
                )
                .where(broker_reconciliations.c.destination_id == destination_id)
                .order_by(broker_reconciliations.c.created_at.desc())
                .limit(1)
            ).first()
            if (
                reconciliation is None
                or reconciliation.status != "matched"
                or reconciliation.created_at < destination.activation_requested_at
            ):
                raise ValueError("activation requires a fresh matched broker reconciliation")
            now = _now()
            connection.execute(
                update(broker_destinations)
                .where(broker_destinations.c.id == destination_id)
                .values(
                    status="armed", activated_by=actor.strip(), activated_at=now, updated_at=now
                )
            )
            self._event(
                connection,
                destination_id=destination_id,
                event_type="destination_armed",
                actor=actor,
                details={"environment": "sandbox"},
            )
        return self.get_destination(destination_id)

    def disarm(self, destination_id: str, *, actor: str) -> dict[str, Any]:
        with self.engine.begin() as connection:
            destination = connection.execute(
                select(broker_destinations.c.id).where(broker_destinations.c.id == destination_id)
            ).scalar_one_or_none()
            if destination is None:
                raise KeyError(destination_id)
            now = _now()
            connection.execute(
                update(broker_destinations)
                .where(broker_destinations.c.id == destination_id)
                .values(
                    status="disabled",
                    activation_requested_by=None,
                    activation_requested_at=None,
                    activated_by=None,
                    activated_at=None,
                    updated_at=now,
                )
            )
            self._event(
                connection,
                destination_id=destination_id,
                event_type="destination_disarmed",
                actor=actor,
                details={},
            )
        return self.get_destination(destination_id)

    def stage_batch(
        self, destination_id: str, batch_id: str, *, actor: str
    ) -> list[dict[str, Any]]:
        settings = self._effective_settings()
        if settings.broker_mode != "sandbox":
            raise ValueError("broker sandbox mode is disabled")
        now = _now()
        staged_ids: list[str] = []
        with self.engine.begin() as connection:
            destination = connection.execute(
                select(broker_destinations)
                .where(broker_destinations.c.id == destination_id)
                .with_for_update()
            ).first()
            if destination is None:
                raise KeyError(destination_id)
            if destination.status != "armed" or destination.environment != "sandbox":
                raise ValueError("an armed sandbox destination is required")
            batch = connection.execute(
                select(portfolio_batches).where(portfolio_batches.c.id == batch_id)
            ).first()
            if batch is None:
                raise KeyError(batch_id)
            if batch.status != "succeeded":
                raise ValueError("only a succeeded paper batch can be replayed to sandbox")
            if destination.portfolio_id != batch.portfolio_id:
                raise ValueError("paper batch does not belong to the destination portfolio")
            if batch.trade_date is None:
                raise ValueError("paper batch has no execution trade date")
            rows = connection.execute(
                select(
                    paper_orders.c.id,
                    paper_orders.c.portfolio_id,
                    paper_orders.c.instrument,
                    paper_orders.c.side,
                    paper_fills.c.quantity,
                    paper_fills.c.price,
                    paper_fills.c.gross_value,
                )
                .join(paper_fills, paper_fills.c.order_id == paper_orders.c.id)
                .where(paper_orders.c.batch_id == batch_id)
                .order_by(paper_orders.c.id)
            ).all()
            if not rows:
                raise ValueError("paper batch has no filled orders to replay")
            for row in rows:
                notional = Decimal(str(row.gross_value))
                if notional > Decimal(str(settings.broker_max_order_notional)):
                    raise ValueError(f"order {row.id} notional exceeds BROKER_MAX_ORDER_NOTIONAL")
                outbox_id = uuid.uuid4().hex
                idempotency_key = f"sandbox:{destination_id}:{row.id}"
                payload = {
                    "client_order_id": idempotency_key,
                    "account_ref": destination.account_ref,
                    "environment": "sandbox",
                    "source": "paper_rehearsal",
                    "instrument": row.instrument,
                    "side": row.side,
                    "order_type": "limit",
                    "quantity": float(row.quantity),
                    "limit_price": float(row.price),
                    "reference_notional": float(notional),
                    "trade_date": batch.trade_date.isoformat(),
                    "execution_policy": destination.config_json,
                    "execution_slices": build_execution_slices(
                        quantity=float(row.quantity),
                        side=str(row.side),
                        trade_date=batch.trade_date,
                        policy=destination.config_json,
                    ),
                }
                existing = connection.execute(
                    select(broker_order_outbox.c.id).where(
                        broker_order_outbox.c.idempotency_key == idempotency_key
                    )
                ).scalar_one_or_none()
                if existing:
                    staged_ids.append(str(existing))
                else:
                    connection.execute(
                        insert(broker_order_outbox).values(
                            id=outbox_id,
                            destination_id=destination_id,
                            portfolio_id=row.portfolio_id,
                            batch_id=batch_id,
                            source_order_id=row.id,
                            idempotency_key=idempotency_key,
                            payload_json=payload,
                            payload_sha256=_digest(payload),
                            status="staged",
                            attempts=0,
                            created_by=actor.strip(),
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    staged_ids.append(outbox_id)
            self._event(
                connection,
                destination_id=destination_id,
                event_type="sandbox_batch_staged",
                actor=actor,
                details={"batch_id": batch_id, "order_count": len(staged_ids)},
            )
        return [self.get_outbox(item) for item in staged_ids]

    def approve_batch(
        self, destination_id: str, batch_id: str, *, actor: str
    ) -> list[dict[str, Any]]:
        now = _now()
        with self.engine.begin() as connection:
            rows = connection.execute(
                select(broker_order_outbox)
                .where(
                    broker_order_outbox.c.destination_id == destination_id,
                    broker_order_outbox.c.batch_id == batch_id,
                    broker_order_outbox.c.status == "staged",
                )
                .with_for_update()
            ).all()
            if not rows:
                raise ValueError("batch has no staged sandbox orders")
            if any(row.created_by == actor.strip() for row in rows):
                raise ValueError("sandbox release requires a second administrator")
            ids = [str(row.id) for row in rows]
            connection.execute(
                update(broker_order_outbox)
                .where(broker_order_outbox.c.id.in_(ids))
                .values(
                    status="approved", approved_by=actor.strip(), approved_at=now, updated_at=now
                )
            )
            self._event(
                connection,
                destination_id=destination_id,
                event_type="sandbox_batch_approved",
                actor=actor,
                details={"batch_id": batch_id, "order_count": len(ids)},
            )
        return [self.get_outbox(item) for item in ids]

    def dispatch_batch(self, destination_id: str, batch_id: str, *, actor: str) -> dict[str, Any]:
        settings = self._effective_settings()
        destination = self.get_destination(destination_id)
        if destination["status"] != "armed" or destination["environment"] != "sandbox":
            raise ValueError("an armed sandbox destination is required")
        gateway = SignedBrokerGateway(settings)
        candidates = self.list_outbox(destination_id=destination_id, batch_id=batch_id, limit=500)
        candidates = [
            item
            for item in candidates
            if item["status"] in {"approved", "failed"}
            and item["approved_by"]
            and int(item["attempts"]) < settings.broker_max_attempts
        ]
        if not candidates:
            raise ValueError("batch has no approved sandbox orders eligible for dispatch")
        for item in candidates:
            if _digest(item["payload"]) != item["payload_sha256"]:
                raise ValueError(f"outbox payload digest mismatch for {item['id']}")
            if item["payload"].get("environment") != "sandbox":
                raise ValueError(f"outbox environment mismatch for {item['id']}")
            if item["payload"].get("account_ref") != destination["account_ref"]:
                raise ValueError(f"outbox account mismatch for {item['id']}")
            if float(item["payload"].get("reference_notional") or 0) > float(
                settings.broker_max_order_notional
            ):
                raise ValueError(f"outbox notional exceeds current limit for {item['id']}")
        submitted = 0
        failed = 0
        for item in candidates:
            now = _now()
            with self.engine.begin() as connection:
                claimed = connection.execute(
                    update(broker_order_outbox)
                    .where(
                        broker_order_outbox.c.id == item["id"],
                        broker_order_outbox.c.status.in_(["approved", "failed"]),
                        broker_order_outbox.c.attempts < settings.broker_max_attempts,
                    )
                    .values(
                        status="submitting",
                        attempts=broker_order_outbox.c.attempts + 1,
                        last_error=None,
                        updated_at=now,
                    )
                )
                if not claimed.rowcount:
                    continue
            try:
                response = gateway.submit_order(item["payload"])
                with self.engine.begin() as connection:
                    connection.execute(
                        update(broker_order_outbox)
                        .where(broker_order_outbox.c.id == item["id"])
                        .values(
                            status="submitted",
                            broker_order_id=response["order_id"],
                            submitted_at=_now(),
                            updated_at=_now(),
                        )
                    )
                    self._event(
                        connection,
                        destination_id=destination_id,
                        outbox_id=item["id"],
                        event_type="sandbox_order_submitted",
                        actor=actor,
                        details={"broker_order_id": response["order_id"]},
                    )
                submitted += 1
            except BrokerGatewayError as exc:
                with self.engine.begin() as connection:
                    connection.execute(
                        update(broker_order_outbox)
                        .where(broker_order_outbox.c.id == item["id"])
                        .values(status="failed", last_error=str(exc), updated_at=_now())
                    )
                    self._event(
                        connection,
                        destination_id=destination_id,
                        outbox_id=item["id"],
                        event_type="sandbox_order_failed",
                        actor=actor,
                        details={"error": str(exc)},
                    )
                failed += 1
        return {
            "destination_id": destination_id,
            "batch_id": batch_id,
            "submitted": submitted,
            "failed": failed,
            "mode": "sandbox",
        }

    def reconcile(self, destination_id: str, *, actor: str) -> dict[str, Any]:
        settings = self._effective_settings()
        destination = self.get_destination(destination_id)
        if destination["status"] not in {
            "pending_activation",
            "armed",
            "locked_mismatch",
        }:
            raise ValueError(
                "destination must be pending activation, armed, or reconciliation-locked"
            )
        if not destination.get("portfolio_id"):
            raise ValueError("destination is not bound to a paper portfolio")
        observed = SignedBrokerGateway(settings).snapshot(destination["account_ref"])
        with self.engine.connect() as connection:
            portfolio = connection.execute(
                select(paper_portfolios).where(paper_portfolios.c.id == destination["portfolio_id"])
            ).first()
            if portfolio is None:
                raise KeyError(destination["portfolio_id"])
            positions = [
                {
                    "instrument": str(row.instrument),
                    "quantity": float(row.quantity),
                    "market_price": float(row.market_price),
                }
                for row in connection.execute(
                    select(paper_positions).where(
                        paper_positions.c.portfolio_id == destination["portfolio_id"]
                    )
                )
            ]
            outbox_rows = connection.execute(
                select(
                    broker_order_outbox.c.idempotency_key,
                    broker_order_outbox.c.status,
                    broker_order_outbox.c.broker_order_id,
                ).where(broker_order_outbox.c.destination_id == destination_id)
            ).all()
        expected = {
            "portfolio_id": destination["portfolio_id"],
            "cash": float(portfolio.cash),
            "equity": float(portfolio.nav),
            "positions": positions,
        }
        known_ids = {str(row.idempotency_key) for row in outbox_rows}
        submitted_orders = {
            str(row.idempotency_key): (str(row.broker_order_id) if row.broker_order_id else None)
            for row in outbox_rows
            if row.status == "submitted"
        }
        config = destination["config"]
        differences = compare_broker_snapshot(
            expected=expected,
            observed=observed,
            submitted_orders=submitted_orders,
            known_client_order_ids=known_ids,
            cash_tolerance=float(config["cash_tolerance"]),
            equity_tolerance=float(config["equity_tolerance"]),
            position_tolerance=float(config["position_tolerance"]),
            max_snapshot_age_seconds=int(config["max_snapshot_age_seconds"]),
        )
        reconciliation_id = uuid.uuid4().hex
        status = "mismatch" if differences else "matched"
        broker_as_of = datetime.fromisoformat(str(observed["as_of"]).replace("Z", "+00:00"))
        with self.engine.begin() as connection:
            connection.execute(
                insert(broker_reconciliations).values(
                    id=reconciliation_id,
                    destination_id=destination_id,
                    status=status,
                    broker_as_of=broker_as_of,
                    expected_json=expected,
                    observed_json=observed,
                    differences_json=differences,
                    created_by=actor.strip(),
                    created_at=_now(),
                )
            )
            if differences:
                connection.execute(
                    update(broker_destinations)
                    .where(broker_destinations.c.id == destination_id)
                    .values(status="locked_mismatch", updated_at=_now())
                )
            self._event(
                connection,
                destination_id=destination_id,
                event_type=("reconciliation_mismatch" if differences else "reconciliation_matched"),
                actor=actor,
                details={
                    "reconciliation_id": reconciliation_id,
                    "difference_count": len(differences),
                },
            )
        if differences:
            self.alerts.create(
                source_type="broker_destination",
                source_id=destination_id,
                severity="critical",
                category="broker_reconciliation",
                title=f"券商沙箱对账失败：{destination['name']}",
                message=f"检测到 {len(differences)} 项资金、持仓或订单差异，目的地已锁定。",
                dedupe_key=f"broker-reconciliation:{destination_id}:{reconciliation_id}",
                details={
                    "reconciliation_id": reconciliation_id,
                    "differences": differences,
                },
            )
        return self.get_reconciliation(reconciliation_id)

    def get_reconciliation(self, reconciliation_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(broker_reconciliations).where(
                    broker_reconciliations.c.id == reconciliation_id
                )
            ).first()
        if row is None:
            raise KeyError(reconciliation_id)
        return self._reconciliation_row(row)

    def list_reconciliations(
        self, destination_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        statement = select(broker_reconciliations)
        if destination_id:
            statement = statement.where(broker_reconciliations.c.destination_id == destination_id)
        statement = statement.order_by(broker_reconciliations.c.created_at.desc()).limit(limit)
        with self.engine.connect() as connection:
            rows = connection.execute(statement).all()
        return [self._reconciliation_row(row) for row in rows]

    def get_destination(self, destination_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(broker_destinations).where(broker_destinations.c.id == destination_id)
            ).first()
        if row is None:
            raise KeyError(destination_id)
        return self._destination_row(row)

    def list_destinations(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(broker_destinations)
                .order_by(broker_destinations.c.updated_at.desc())
                .limit(limit)
            ).all()
        return [self._destination_row(row) for row in rows]

    def get_outbox(self, outbox_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(broker_order_outbox).where(broker_order_outbox.c.id == outbox_id)
            ).first()
        if row is None:
            raise KeyError(outbox_id)
        return self._outbox_row(row)

    def list_outbox(
        self,
        *,
        destination_id: str | None = None,
        batch_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        statement = select(broker_order_outbox)
        if destination_id:
            statement = statement.where(broker_order_outbox.c.destination_id == destination_id)
        if batch_id:
            statement = statement.where(broker_order_outbox.c.batch_id == batch_id)
        statement = statement.order_by(broker_order_outbox.c.created_at.desc()).limit(limit)
        with self.engine.connect() as connection:
            rows = connection.execute(statement).all()
        return [self._outbox_row(row) for row in rows]

    def list_events(self, destination_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(broker_events)
                .where(broker_events.c.destination_id == destination_id)
                .order_by(broker_events.c.created_at.desc())
                .limit(limit)
            ).all()
        return [self._event_row(row) for row in rows]

    @staticmethod
    def _event(
        connection: Any,
        *,
        destination_id: str,
        event_type: str,
        actor: str,
        details: dict[str, Any],
        outbox_id: str | None = None,
    ) -> None:
        connection.execute(
            insert(broker_events).values(
                destination_id=destination_id,
                outbox_id=outbox_id,
                event_type=event_type,
                actor=actor.strip(),
                details_json=details,
                created_at=_now(),
            )
        )

    @staticmethod
    def _destination_row(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["config"] = result.pop("config_json")
        return result

    @staticmethod
    def _outbox_row(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["payload"] = result.pop("payload_json")
        return result

    @staticmethod
    def _event_row(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["details"] = result.pop("details_json")
        return result

    @staticmethod
    def _reconciliation_row(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["expected"] = result.pop("expected_json")
        result["observed"] = result.pop("observed_json")
        result["differences"] = result.pop("differences_json")
        return result
