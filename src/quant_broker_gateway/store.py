from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import delete, func, insert, or_, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    broker_gateway_attempts,
    broker_gateway_children,
    broker_gateway_events,
    broker_gateway_nonces,
    broker_gateway_parents,
    open_database,
    row_dict,
)

from .protocols import BrokerAdapter


class GatewayStoreError(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now(UTC)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class GatewayStore:
    def __init__(
        self,
        database_url: str,
        adapter: BrokerAdapter,
        *,
        account_ref: str,
        max_slice_lateness_seconds: int = 90,
        cancel_after_seconds: int = 60,
        max_replacements: int = 1,
        max_reprice_bps: float = 20.0,
    ) -> None:
        self.engine = open_database(database_url)
        self.adapter = adapter
        self.account_ref = account_ref
        self.max_slice_lateness_seconds = max_slice_lateness_seconds
        self.cancel_after_seconds = cancel_after_seconds
        self.max_replacements = max_replacements
        self.max_reprice_bps = max_reprice_bps

    def health(self) -> dict[str, Any]:
        provider = self.adapter.health()
        with self.engine.connect() as connection:
            counts = {
                str(status): int(count)
                for status, count in connection.execute(
                    select(broker_gateway_children.c.status, func.count())
                    .group_by(broker_gateway_children.c.status)
                    .order_by(broker_gateway_children.c.status)
                )
            }
            active_attempts = int(
                connection.scalar(
                    select(func.count()).select_from(broker_gateway_attempts).where(
                        broker_gateway_attempts.c.status.in_(
                            ["prepared", "submitted", "partial", "cancel_pending"]
                        )
                    )
                )
                or 0
            )
        return {
            "status": "ok",
            "environment": "sandbox",
            "provider": self.adapter.provider_name,
            "account_ref": self.account_ref,
            "provider_health": provider,
            "slice_counts": counts,
            "active_attempts": active_attempts,
        }

    def claim_nonce(self, nonce: str, current: int, expires: int) -> bool:
        now = datetime.fromtimestamp(current, UTC)
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    delete(broker_gateway_nonces).where(
                        broker_gateway_nonces.c.expires_at < now
                    )
                )
                connection.execute(
                    insert(broker_gateway_nonces).values(
                        nonce=nonce,
                        expires_at=datetime.fromtimestamp(expires, UTC),
                        created_at=now,
                    )
                )
        except IntegrityError:
            return False
        return True

    def market_evidence(self, instrument: str) -> dict[str, Any]:
        evidence = self.adapter.market_evidence(instrument, _now())
        return {
            "status": "ok",
            "environment": "sandbox",
            "provider": self.adapter.provider_name,
            **evidence,
        }

    def maintain_active_once(self, *, now: datetime | None = None) -> bool:
        current = (now or _now()).astimezone(UTC)
        changed = self._persist_provider_events()
        raw = self.adapter.snapshot(self.account_ref)
        cancel_actions, replacement_children = self._sync_provider_state(raw, current)
        for attempt_id, provider_order_id in cancel_actions:
            try:
                accepted = self.adapter.cancel_order(
                    account_ref=self.account_ref,
                    provider_order_id=provider_order_id,
                )
                if not accepted:
                    raise GatewayStoreError("QMT rejected the cancel request")
            except Exception as exc:
                with self.engine.begin() as connection:
                    connection.execute(
                        update(broker_gateway_attempts)
                        .where(broker_gateway_attempts.c.id == attempt_id)
                        .values(status="error", last_error=str(exc), updated_at=_now())
                    )
                changed = True
        for child_id in replacement_children:
            self._submit_replacement(child_id, current)
            changed = True
        return changed or bool(cancel_actions) or bool(replacement_children)

    def _persist_provider_events(self) -> bool:
        events = self.adapter.drain_events()
        if not events:
            return False
        with self.engine.begin() as connection:
            for event in events:
                payload = dict(event.get("payload") or {})
                connection.execute(
                    insert(broker_gateway_events).values(
                        event_type=str(event.get("event_type") or "unknown"),
                        provider_order_id=str(payload.get("provider_order_id") or "") or None,
                        client_tag=str(payload.get("client_tag") or "") or None,
                        payload_json=payload,
                        received_at=event.get("received_at") or _now(),
                    )
                )
        return True

    def _sync_provider_state(
        self, raw: dict[str, Any], current: datetime
    ) -> tuple[list[tuple[str, str]], list[str]]:
        cancel_actions: list[tuple[str, str]] = []
        replacement_children: list[str] = []
        with self.engine.begin() as connection:
            attempts = [
                row_dict(row) for row in connection.execute(select(broker_gateway_attempts))
            ]
            by_provider = {
                str(row["provider_order_id"]): row
                for row in attempts
                if row.get("provider_order_id")
            }
            by_tag = {str(row["client_tag"]): row for row in attempts}
            for record in raw.get("orders", []):
                attempt = by_provider.get(str(record.get("provider_order_id") or ""))
                attempt = attempt or by_tag.get(str(record.get("client_tag") or ""))
                if attempt is None:
                    continue
                status = str(record.get("status") or "error")
                if attempt.get("cancel_requested_at") and status in {
                    "pending",
                    "submitted",
                    "partial",
                    "cancel_pending",
                }:
                    status = "cancel_pending"
                traded = min(
                    int(attempt["quantity"]), max(0, int(record.get("traded_quantity") or 0))
                )
                connection.execute(
                    update(broker_gateway_attempts)
                    .where(broker_gateway_attempts.c.id == attempt["id"])
                    .values(
                        status=status,
                        traded_quantity=traded,
                        completed_at=current
                        if status in {"filled", "canceled", "rejected", "error"}
                        else None,
                        updated_at=current,
                        last_error=str(record.get("status_message") or "") or None,
                    )
                )
            active_children = connection.execute(
                select(broker_gateway_children).where(
                    broker_gateway_children.c.status.in_(
                        ["submitted", "partial", "cancel_pending"]
                    )
                )
            ).all()
            for child in active_children:
                child_attempts = connection.execute(
                    select(broker_gateway_attempts)
                    .where(broker_gateway_attempts.c.child_id == child.id)
                    .order_by(broker_gateway_attempts.c.attempt_no)
                ).all()
                if not child_attempts:
                    continue
                filled = min(
                    int(child.quantity), sum(int(item.traded_quantity) for item in child_attempts)
                )
                latest = child_attempts[-1]
                child_status = child.status
                if filled >= int(child.quantity):
                    child_status = "filled"
                elif latest.status in {"rejected", "error"}:
                    child_status = "error"
                elif latest.status == "canceled":
                    if int(child.replacement_count) < self.max_replacements:
                        child_status = "replacing"
                        replacement_children.append(str(child.id))
                    else:
                        child_status = "error"
                elif latest.status == "cancel_pending":
                    child_status = "cancel_pending"
                elif latest.status in {"submitted", "partial"}:
                    submitted_at = latest.submitted_at or current
                    if (
                        latest.cancel_requested_at is None
                        and (current - submitted_at).total_seconds()
                        >= self.cancel_after_seconds
                    ):
                        connection.execute(
                            update(broker_gateway_attempts)
                            .where(broker_gateway_attempts.c.id == latest.id)
                            .values(
                                status="cancel_pending",
                                cancel_requested_at=current,
                                updated_at=current,
                            )
                        )
                        child_status = "cancel_pending"
                        cancel_actions.append((str(latest.id), str(latest.provider_order_id)))
                    else:
                        child_status = "partial" if latest.status == "partial" else "submitted"
                connection.execute(
                    update(broker_gateway_children)
                    .where(broker_gateway_children.c.id == child.id)
                    .values(
                        status=child_status,
                        filled_quantity=filled,
                        cancel_requested_at=current if child_status == "cancel_pending" else None,
                        updated_at=current,
                    )
                )
                self._refresh_parent_status(connection, str(child.parent_id))
        return cancel_actions, replacement_children

    def _submit_replacement(self, child_id: str, current: datetime) -> None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    broker_gateway_children,
                    broker_gateway_parents.c.account_ref,
                    broker_gateway_parents.c.payload_json,
                )
                .join(
                    broker_gateway_parents,
                    broker_gateway_parents.c.id == broker_gateway_children.c.parent_id,
                )
                .where(broker_gateway_children.c.id == child_id)
            ).first()
            if row is None:
                raise GatewayStoreError("replacement slice disappeared")
            child = row._mapping
            payload = dict(child[broker_gateway_parents.c.payload_json])
            account_ref = str(child[broker_gateway_parents.c.account_ref])
            target = int(child[broker_gateway_children.c.quantity])
            filled = int(child[broker_gateway_children.c.filled_quantity])
            replacement_count = int(child[broker_gateway_children.c.replacement_count])
            parent_id = str(child[broker_gateway_children.c.parent_id])
            original_limit = float(child[broker_gateway_children.c.limit_price])
        remaining = target - filled
        if remaining <= 0:
            return
        if str(payload["side"]) == "buy" and remaining % 100:
            with self.engine.begin() as connection:
                connection.execute(
                    update(broker_gateway_children)
                    .where(broker_gateway_children.c.id == child_id)
                    .values(
                        status="error",
                        last_error="remaining buy quantity is below an A-share board lot",
                        updated_at=current,
                    )
                )
                self._refresh_parent_status(connection, parent_id)
            return
        evidence = self.adapter.market_evidence(str(payload["instrument"]), current)
        max_participation = float(
            (payload.get("execution_policy") or {}).get("max_participation", 0)
        )
        allowed = int(int(evidence.get("minute_volume_shares") or 0) * max_participation)
        if str(payload["side"]) == "buy":
            allowed = (allowed // 100) * 100
        if remaining > allowed or allowed <= 0:
            with self.engine.begin() as connection:
                connection.execute(
                    update(broker_gateway_children)
                    .where(broker_gateway_children.c.id == child_id)
                    .values(
                        status="liquidity_blocked",
                        market_evidence_json=evidence,
                        last_error="replacement exceeds the minute participation cap",
                        updated_at=current,
                    )
                )
                self._refresh_parent_status(connection, parent_id)
            return
        attempt_no = replacement_count + 2
        client_tag = _client_tag(child_id, attempt_no)
        limit_price = _replacement_price(
            original_limit,
            str(payload["side"]),
            evidence,
            self.max_reprice_bps,
        )
        attempt_id = self._prepare_attempt(
            child_id=child_id,
            attempt_no=attempt_no,
            client_tag=client_tag,
            quantity=remaining,
            limit_price=limit_price,
            evidence=evidence,
            current=current,
        )
        try:
            provider_order_id = self.adapter.submit_limit_order(
                account_ref=account_ref,
                instrument=str(payload["instrument"]),
                side=str(payload["side"]),
                quantity=remaining,
                limit_price=limit_price,
                client_tag=client_tag,
            )
        except Exception as exc:
            with self.engine.begin() as connection:
                connection.execute(
                    update(broker_gateway_attempts)
                    .where(broker_gateway_attempts.c.id == attempt_id)
                    .values(status="error", last_error=str(exc), updated_at=_now())
                )
                connection.execute(
                    update(broker_gateway_children)
                    .where(broker_gateway_children.c.id == child_id)
                    .values(status="error", last_error=str(exc), updated_at=_now())
                )
                self._refresh_parent_status(connection, parent_id)
            return
        with self.engine.begin() as connection:
            connection.execute(
                update(broker_gateway_attempts)
                .where(broker_gateway_attempts.c.id == attempt_id)
                .values(
                    status="submitted",
                    provider_order_id=str(provider_order_id),
                    submitted_at=current,
                    updated_at=current,
                )
            )
            connection.execute(
                update(broker_gateway_children)
                .where(broker_gateway_children.c.id == child_id)
                .values(
                    status="submitted",
                    replacement_count=replacement_count + 1,
                    client_tag=client_tag,
                    provider_order_id=str(provider_order_id),
                    submitted_at=current,
                    cancel_requested_at=None,
                    market_evidence_json=evidence,
                    updated_at=current,
                )
            )
            self._refresh_parent_status(connection, parent_id)

    def accept_parent(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = self._validate_parent(payload)
        payload_hash = _digest(normalized)
        now = _now()
        parent_id = f"qmt-{uuid.uuid4().hex}"
        try:
            with self.engine.begin() as connection:
                existing = connection.execute(
                    select(broker_gateway_parents).where(
                        broker_gateway_parents.c.client_order_id
                        == normalized["client_order_id"]
                    )
                ).first()
                if existing is not None:
                    if existing.payload_sha256 != payload_hash:
                        raise GatewayStoreError(
                            "client_order_id was already used with a different payload"
                        )
                    return self._accepted(row_dict(existing))
                connection.execute(
                    insert(broker_gateway_parents).values(
                        id=parent_id,
                        client_order_id=normalized["client_order_id"],
                        account_ref=normalized["account_ref"],
                        environment="sandbox",
                        provider=self.adapter.provider_name,
                        payload_json=normalized,
                        payload_sha256=payload_hash,
                        status="queued",
                        created_at=now,
                        updated_at=now,
                    )
                )
                for item in normalized["execution_slices"]:
                    child_id = uuid.uuid4().hex
                    connection.execute(
                        insert(broker_gateway_children).values(
                            id=child_id,
                            parent_id=parent_id,
                            slice_index=item["sequence"],
                            scheduled_for=datetime.fromisoformat(item["scheduled_for"]),
                            quantity=item["quantity"],
                            limit_price=normalized["limit_price"],
                            client_tag=_client_tag(child_id),
                            status="pending",
                            updated_at=now,
                        )
                    )
        except IntegrityError as exc:
            raise GatewayStoreError("cannot persist the QMT parent order") from exc
        return self._accepted(
            {
                "id": parent_id,
                "client_order_id": normalized["client_order_id"],
                "status": "queued",
            }
        )

    def run_due_once(self, *, now: datetime | None = None) -> bool:
        current = (now or _now()).astimezone(UTC)
        recovery_before = current - timedelta(seconds=30)
        with self.engine.begin() as connection:
            row = connection.execute(
                select(
                    broker_gateway_children,
                    broker_gateway_parents.c.account_ref,
                    broker_gateway_parents.c.payload_json,
                )
                .join(
                    broker_gateway_parents,
                    broker_gateway_parents.c.id == broker_gateway_children.c.parent_id,
                )
                .where(
                    broker_gateway_children.c.scheduled_for <= current,
                    or_(
                        broker_gateway_children.c.status == "pending",
                        (
                            (broker_gateway_children.c.status == "submitting")
                            & (broker_gateway_children.c.updated_at <= recovery_before)
                        ),
                    ),
                )
                .order_by(broker_gateway_children.c.scheduled_for, broker_gateway_children.c.id)
                .with_for_update(skip_locked=True)
                .limit(1)
            ).first()
            if row is None:
                return False
            child = row._mapping
            was_recovery = child[broker_gateway_children.c.status] == "submitting"
            lateness = (current - child[broker_gateway_children.c.scheduled_for]).total_seconds()
            if lateness > self.max_slice_lateness_seconds and not was_recovery:
                connection.execute(
                    update(broker_gateway_children)
                    .where(broker_gateway_children.c.id == child[broker_gateway_children.c.id])
                    .values(
                        status="missed",
                        last_error=f"slice was {int(lateness)} seconds late",
                        updated_at=current,
                    )
                )
                self._refresh_parent_status(connection, child[broker_gateway_children.c.parent_id])
                return True
            connection.execute(
                update(broker_gateway_children)
                .where(broker_gateway_children.c.id == child[broker_gateway_children.c.id])
                .values(status="submitting", updated_at=current, last_error=None)
            )
            payload = dict(child[broker_gateway_parents.c.payload_json])
            child_id = str(child[broker_gateway_children.c.id])
            parent_id = str(child[broker_gateway_children.c.parent_id])
            account_ref = str(child[broker_gateway_parents.c.account_ref])
            quantity = int(child[broker_gateway_children.c.quantity])
            limit_price = float(child[broker_gateway_children.c.limit_price])
            client_tag = str(child[broker_gateway_children.c.client_tag])
            slice_index = int(child[broker_gateway_children.c.slice_index])
        try:
            evidence = self.adapter.market_evidence(str(payload["instrument"]), current)
            quantity = self._apply_participation_limit(
                child_id=child_id,
                parent_id=parent_id,
                slice_index=slice_index,
                side=str(payload["side"]),
                quantity=quantity,
                payload=payload,
                evidence=evidence,
                current=current,
            )
            if quantity is None:
                return True
            attempt_id = self._prepare_attempt(
                child_id=child_id,
                attempt_no=1,
                client_tag=client_tag,
                quantity=quantity,
                limit_price=limit_price,
                evidence=evidence,
                current=current,
            )
            provider_order_id = self.adapter.submit_limit_order(
                account_ref=account_ref,
                instrument=str(payload["instrument"]),
                side=str(payload["side"]),
                quantity=quantity,
                limit_price=limit_price,
                client_tag=client_tag,
            )
        except Exception as exc:
            with self.engine.begin() as connection:
                connection.execute(
                    update(broker_gateway_attempts)
                    .where(
                        broker_gateway_attempts.c.child_id == child_id,
                        broker_gateway_attempts.c.attempt_no == 1,
                    )
                    .values(status="error", last_error=str(exc), updated_at=_now())
                )
                connection.execute(
                    update(broker_gateway_children)
                    .where(broker_gateway_children.c.id == child_id)
                    .values(status="error", last_error=str(exc), updated_at=_now())
                )
                self._refresh_parent_status(connection, parent_id)
            return True
        with self.engine.begin() as connection:
            connection.execute(
                update(broker_gateway_attempts)
                .where(broker_gateway_attempts.c.id == attempt_id)
                .values(
                    status="submitted",
                    provider_order_id=str(provider_order_id),
                    submitted_at=_now(),
                    updated_at=_now(),
                )
            )
            connection.execute(
                update(broker_gateway_children)
                .where(broker_gateway_children.c.id == child_id)
                .values(
                    status="submitted",
                    provider_order_id=str(provider_order_id),
                    submitted_at=_now(),
                    updated_at=_now(),
                )
            )
            self._refresh_parent_status(connection, parent_id)
        return True

    def _apply_participation_limit(
        self,
        *,
        child_id: str,
        parent_id: str,
        slice_index: int,
        side: str,
        quantity: int,
        payload: dict[str, Any],
        evidence: dict[str, Any],
        current: datetime,
    ) -> int | None:
        policy = payload.get("execution_policy") or {}
        raw_participation = float(policy.get("max_participation", 0))
        slice_item = next(
            (
                item
                for item in payload.get("execution_slices", [])
                if int(item.get("sequence", -1)) == slice_index
            ),
            {},
        )
        max_participation = min(
            raw_participation,
            float(slice_item.get("max_participation", raw_participation)),
        )
        if not 0 < max_participation <= 0.20:
            raise GatewayStoreError("execution slice has no valid participation limit")
        minute_volume = int(evidence.get("minute_volume_shares") or 0)
        if minute_volume <= 0:
            raise GatewayStoreError("QMT minute-volume evidence is unavailable")
        allowed = int(minute_volume * max_participation)
        if side == "buy":
            allowed = (allowed // 100) * 100
        allowed = min(quantity, allowed)
        evidence.update(
            max_participation=max_participation,
            allowed_quantity=allowed,
            requested_quantity=quantity,
        )
        with self.engine.begin() as connection:
            current_child = connection.execute(
                select(broker_gateway_children)
                .where(broker_gateway_children.c.id == child_id)
                .with_for_update()
            ).first()
            if current_child is None:
                raise GatewayStoreError("execution slice disappeared")
            if allowed < quantity:
                next_child = connection.execute(
                    select(broker_gateway_children)
                    .where(
                        broker_gateway_children.c.parent_id == parent_id,
                        broker_gateway_children.c.status == "pending",
                        broker_gateway_children.c.slice_index > slice_index,
                    )
                    .order_by(broker_gateway_children.c.slice_index)
                    .with_for_update()
                    .limit(1)
                ).first()
                if next_child is None:
                    connection.execute(
                        update(broker_gateway_children)
                        .where(broker_gateway_children.c.id == child_id)
                        .values(
                            status="liquidity_blocked",
                            market_evidence_json=evidence,
                            last_error="final slice exceeds the minute participation cap",
                            updated_at=current,
                        )
                    )
                    self._refresh_parent_status(connection, parent_id)
                    return None
                remainder = quantity - allowed
                connection.execute(
                    update(broker_gateway_children)
                    .where(broker_gateway_children.c.id == next_child.id)
                    .values(quantity=broker_gateway_children.c.quantity + remainder)
                )
                if allowed <= 0:
                    connection.execute(
                        update(broker_gateway_children)
                        .where(broker_gateway_children.c.id == child_id)
                        .values(
                            quantity=0,
                            status="redistributed",
                            market_evidence_json=evidence,
                            updated_at=current,
                        )
                    )
                    self._refresh_parent_status(connection, parent_id)
                    return None
                connection.execute(
                    update(broker_gateway_children)
                    .where(broker_gateway_children.c.id == child_id)
                    .values(quantity=allowed, market_evidence_json=evidence, updated_at=current)
                )
            else:
                connection.execute(
                    update(broker_gateway_children)
                    .where(broker_gateway_children.c.id == child_id)
                    .values(market_evidence_json=evidence, updated_at=current)
                )
        return allowed

    def _prepare_attempt(
        self,
        *,
        child_id: str,
        attempt_no: int,
        client_tag: str,
        quantity: int,
        limit_price: float,
        evidence: dict[str, Any],
        current: datetime,
    ) -> str:
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(broker_gateway_attempts).where(
                    broker_gateway_attempts.c.child_id == child_id,
                    broker_gateway_attempts.c.attempt_no == attempt_no,
                )
            ).first()
            if existing is not None:
                return str(existing.id)
            attempt_id = uuid.uuid4().hex
            connection.execute(
                insert(broker_gateway_attempts).values(
                    id=attempt_id,
                    child_id=child_id,
                    attempt_no=attempt_no,
                    client_tag=client_tag,
                    quantity=quantity,
                    limit_price=limit_price,
                    status="prepared",
                    market_evidence_json=evidence,
                    updated_at=current,
                )
            )
            return attempt_id

    def snapshot(self, account_ref: str) -> dict[str, Any]:
        if account_ref != self.account_ref:
            raise GatewayStoreError("account reference mismatch")
        self.maintain_active_once()
        raw = self.adapter.snapshot(account_ref)
        with self.engine.connect() as connection:
            parents = {
                str(row.id): row_dict(row)
                for row in connection.execute(
                    select(broker_gateway_parents).where(
                        broker_gateway_parents.c.account_ref == account_ref
                    )
                )
            }
            children = [
                row_dict(row)
                for row in connection.execute(
                    select(broker_gateway_children).where(
                        broker_gateway_children.c.parent_id.in_(parents)
                    )
                )
            ] if parents else []
            children_by_id = {str(row["id"]): row for row in children}
            attempts = [
                {
                    **row_dict(row),
                    "parent_id": children_by_id[str(row.child_id)]["parent_id"],
                }
                for row in connection.execute(
                    select(broker_gateway_attempts).where(
                        broker_gateway_attempts.c.child_id.in_(children_by_id)
                    )
                )
            ] if children_by_id else []
            by_provider = {
                str(row["provider_order_id"]): row
                for row in attempts
                if row.get("provider_order_id")
            }
            by_tag = {str(row["client_tag"]): row for row in attempts}
            refreshed_parents = {
                str(row.id): row_dict(row)
                for row in connection.execute(
                    select(broker_gateway_parents).where(
                        broker_gateway_parents.c.id.in_(parents)
                    )
                )
            } if parents else {}

        orders = self._map_records(
            raw.get("orders", []), by_provider, by_tag, refreshed_parents, "order"
        )
        trades = self._map_records(
            raw.get("trades", []), by_provider, by_tag, refreshed_parents, "trade"
        )
        represented = {str(item["client_order_id"]) for item in [*orders, *trades]}
        for parent in refreshed_parents.values():
            if parent["client_order_id"] not in represented:
                orders.append(
                    {
                        "client_order_id": parent["client_order_id"],
                        "order_id": parent["id"],
                        "status": parent["status"],
                    }
                )
        positions: dict[str, int] = {}
        for row in raw.get("positions", []):
            instrument = str(row["instrument"])
            positions[instrument] = positions.get(instrument, 0) + int(row["quantity"])
        return {
            "status": "ok",
            "environment": "sandbox",
            "provider": self.adapter.provider_name,
            "account_ref": account_ref,
            "as_of": raw["as_of"],
            "cash": float(raw["cash"]),
            "equity": float(raw["equity"]),
            "positions": [
                {"instrument": instrument, "quantity": quantity}
                for instrument, quantity in sorted(positions.items())
                if quantity
            ],
            "orders": orders,
            "trades": trades,
        }

    @staticmethod
    def _accepted(parent: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "accepted",
            "environment": "sandbox",
            "order_id": str(parent["id"]),
            "client_order_id": str(parent["client_order_id"]),
        }

    def _validate_parent(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise GatewayStoreError("order payload must be an object")
        normalized = dict(payload)
        if normalized.get("environment") != "sandbox":
            raise GatewayStoreError("QMT gateway accepts sandbox orders only")
        if normalized.get("account_ref") != self.account_ref:
            raise GatewayStoreError("QMT account reference mismatch")
        client_order_id = str(normalized.get("client_order_id") or "").strip()
        if not client_order_id or len(client_order_id) > 512:
            raise GatewayStoreError("client_order_id is invalid")
        if normalized.get("order_type") != "limit":
            raise GatewayStoreError("QMT gateway currently accepts limit orders only")
        side = str(normalized.get("side") or "").lower()
        if side not in {"buy", "sell"}:
            raise GatewayStoreError("order side must be buy or sell")
        quantity = _positive_integer(normalized.get("quantity"), "quantity")
        limit_price = _positive_decimal(normalized.get("limit_price"), "limit_price")
        slices = normalized.get("execution_slices")
        if not isinstance(slices, list) or not 1 <= len(slices) <= 64:
            raise GatewayStoreError("execution_slices must contain between 1 and 64 slices")
        normalized_slices: list[dict[str, Any]] = []
        seen_sequences: set[int] = set()
        for raw in slices:
            if not isinstance(raw, dict):
                raise GatewayStoreError("execution slice must be an object")
            sequence = _positive_integer(raw.get("sequence"), "slice sequence")
            if sequence in seen_sequences:
                raise GatewayStoreError("execution slice sequences must be unique")
            seen_sequences.add(sequence)
            try:
                scheduled_for = datetime.fromisoformat(str(raw.get("scheduled_for")))
            except ValueError as exc:
                raise GatewayStoreError("execution slice timestamp is invalid") from exc
            if scheduled_for.tzinfo is None:
                raise GatewayStoreError("execution slice timestamp must include timezone")
            slice_quantity = _positive_integer(raw.get("quantity"), "slice quantity")
            normalized_slices.append(
                {
                    **raw,
                    "sequence": sequence,
                    "scheduled_for": scheduled_for.isoformat(),
                    "quantity": slice_quantity,
                }
            )
        if sum(item["quantity"] for item in normalized_slices) != quantity:
            raise GatewayStoreError("execution slices do not reconcile to parent quantity")
        if side == "buy" and any(item["quantity"] % 100 for item in normalized_slices):
            raise GatewayStoreError("QMT buy slices must use 100-share lots")
        normalized.update(
            client_order_id=client_order_id,
            side=side,
            quantity=quantity,
            limit_price=float(limit_price),
            execution_slices=sorted(normalized_slices, key=lambda item: item["sequence"]),
        )
        return normalized

    @staticmethod
    def _refresh_parent_status(connection: Any, parent_id: str) -> None:
        statuses = [
            str(value)
            for value in connection.scalars(
                select(broker_gateway_children.c.status).where(
                    broker_gateway_children.c.parent_id == parent_id
                )
            )
        ]
        if not statuses:
            status = "error"
        elif any(
            item in {"error", "rejected", "missed", "liquidity_blocked"}
            for item in statuses
        ):
            status = "error"
        elif any(item == "canceled" for item in statuses):
            status = "canceled"
        elif all(item in {"filled", "redistributed"} for item in statuses):
            status = "filled"
        elif any(item == "partial" for item in statuses):
            status = "partial"
        elif any(
            item in {"submitted", "submitting", "cancel_pending", "replacing"}
            for item in statuses
        ):
            status = "running"
        else:
            status = "queued"
        connection.execute(
            update(broker_gateway_parents)
            .where(broker_gateway_parents.c.id == parent_id)
            .values(status=status, updated_at=_now())
        )

    @staticmethod
    def _map_records(
        records: list[dict[str, Any]],
        by_provider: dict[str, dict[str, Any]],
        by_tag: dict[str, dict[str, Any]],
        parents: dict[str, dict[str, Any]],
        kind: str,
    ) -> list[dict[str, Any]]:
        result = []
        for raw in records:
            child = by_provider.get(str(raw.get("provider_order_id") or ""))
            child = child or by_tag.get(str(raw.get("client_tag") or ""))
            if child is None:
                provider_id = str(
                    raw.get("provider_order_id") or raw.get("provider_trade_id") or "unknown"
                )
                result.append(
                    {
                        **raw,
                        "client_order_id": f"external:qmt:{kind}:{provider_id}",
                        "order_id": f"qmt:{provider_id}",
                    }
                )
                continue
            parent = parents[str(child["parent_id"])]
            result.append(
                {
                    **raw,
                    "client_order_id": parent["client_order_id"],
                    "order_id": parent["id"],
                }
            )
        return result


def _client_tag(child_id: str, attempt_no: int = 1) -> str:
    return "QL" + hashlib.sha256(f"{child_id}:{attempt_no}".encode()).hexdigest()[:20]


def _replacement_price(
    original_limit: float,
    side: str,
    evidence: dict[str, Any],
    max_reprice_bps: float,
) -> float:
    deviation = max_reprice_bps / 10_000
    if side == "buy":
        return round(min(float(evidence["ask_price"]), original_limit * (1 + deviation)), 4)
    return round(max(float(evidence["bid_price"]), original_limit * (1 - deviation)), 4)


def _positive_integer(value: Any, name: str) -> int:
    try:
        parsed = Decimal(str(value))
    except Exception as exc:
        raise GatewayStoreError(f"{name} is invalid") from exc
    if parsed <= 0 or parsed != parsed.to_integral_value():
        raise GatewayStoreError(f"{name} must be a positive integer")
    return int(parsed)


def _positive_decimal(value: Any, name: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except Exception as exc:
        raise GatewayStoreError(f"{name} is invalid") from exc
    if parsed <= 0 or not parsed.is_finite():
        raise GatewayStoreError(f"{name} must be positive")
    return parsed
