from __future__ import annotations

import hashlib
import hmac
import json
import sys
import time
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from quant_broker_gateway.app import create_app
from quant_broker_gateway.config import GatewaySettings
from quant_broker_gateway.qmt import (
    QmtBrokerAdapter,
    QmtGatewayError,
    from_qmt_instrument,
    to_qmt_instrument,
)
from quant_broker_gateway.store import GatewayStore
from quant_data.database import (
    broker_gateway_attempts,
    broker_gateway_children,
    broker_gateway_events,
    broker_gateway_parents,
    open_database,
)


class FakeQmtAdapter:
    provider_name = "qmt"

    def __init__(self) -> None:
        self.orders: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.closed = False
        self.events: list[dict[str, Any]] = []
        self.minute_volume_shares = 1_000_000

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "provider": "qmt", "connected": True}

    def submit_limit_order(self, **order: Any) -> str:
        existing = next(
            (item for item in self.orders if item["client_tag"] == order["client_tag"]), None
        )
        if existing:
            return str(existing["provider_order_id"])
        provider_id = str(1000 + len(self.orders))
        self.orders.append(
            {
                "provider_order_id": provider_id,
                "client_tag": order["client_tag"],
                "instrument": order["instrument"],
                "status": "submitted",
                "quantity": order["quantity"],
                "limit_price": order["limit_price"],
                "traded_quantity": 0,
            }
        )
        return provider_id

    def snapshot(self, account_ref: str) -> dict[str, Any]:
        assert account_ref == "SIM-1"
        return {
            "as_of": datetime.now(UTC).isoformat(),
            "cash": 1_000_000,
            "equity": 1_000_000,
            "positions": [],
            "orders": list(self.orders),
            "trades": list(self.trades),
        }

    def market_evidence(self, instrument: str, as_of: datetime) -> dict[str, Any]:
        return {
            "source": "fake_qmt",
            "instrument": instrument,
            "minute_ended_at": (as_of - timedelta(minutes=1)).isoformat(),
            "minute_volume_shares": self.minute_volume_shares,
            "quote_as_of": as_of.isoformat(),
            "quote_age_seconds": 0,
            "bid_price": 10.49,
            "ask_price": 10.51,
        }

    def cancel_order(self, *, account_ref: str, provider_order_id: str) -> bool:
        assert account_ref == "SIM-1"
        for order in self.orders:
            if str(order["provider_order_id"]) == provider_order_id:
                order["status"] = "canceled"
                return True
        return False

    def drain_events(self) -> list[dict[str, Any]]:
        result = list(self.events)
        self.events.clear()
        return result

    def close(self) -> None:
        self.closed = True


def _settings(database_url: str, tmp_path: Path) -> GatewaySettings:
    return GatewaySettings(
        database_url=database_url,
        hmac_secret="q" * 40,
        qmt_mini_path=tmp_path,
        qmt_account_id="provider-account",
        account_ref="SIM-1",
        qmt_session_id=178901,
        poll_seconds=30,
        max_slice_lateness_seconds=90,
    )


def _payload(scheduled_for: datetime) -> dict[str, Any]:
    return {
        "client_order_id": "sandbox:destination:paper-order",
        "account_ref": "SIM-1",
        "environment": "sandbox",
        "source": "paper_rehearsal",
        "instrument": "SH600000",
        "side": "buy",
        "order_type": "limit",
        "quantity": 100,
        "limit_price": 10.5,
        "reference_notional": 1050,
        "trade_date": scheduled_for.date().isoformat(),
        "execution_policy": {"execution_algorithm": "twap", "max_participation": 0.01},
        "execution_slices": [
            {
                "sequence": 1,
                "scheduled_for": scheduled_for.isoformat(),
                "quantity": 100,
                "target_weight": 1.0,
                "algorithm": "twap",
                "max_participation": 0.01,
            }
        ],
    }


def _signed_headers(secret: str, method: str, path: str, body: str = "") -> dict[str, str]:
    timestamp = str(int(time.time()))
    nonce = hashlib.sha256(f"{path}:{body}:{time.time_ns()}".encode()).hexdigest()[:32]
    message = f"{timestamp}.{nonce}.{method}.{path}.{body}".encode()
    signature = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
    return {
        "X-QuantLab-Timestamp": timestamp,
        "X-QuantLab-Nonce": nonce,
        "X-QuantLab-Signature": signature,
        "Content-Type": "application/json",
    }


def test_qmt_gateway_requires_signatures_and_runs_durable_slices(
    database_url: str, tmp_path: Path
) -> None:
    adapter = FakeQmtAdapter()
    settings = _settings(database_url, tmp_path)
    app = create_app(settings, adapter)
    scheduled_for = datetime.now(UTC) + timedelta(minutes=5)
    payload = _payload(scheduled_for)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    with TestClient(app) as client:
        assert client.get("/v1/health").status_code == 401
        health_headers = _signed_headers(settings.hmac_secret, "GET", "/v1/health")
        assert client.get("/v1/health", headers=health_headers).json()["environment"] == "sandbox"
        assert client.get("/v1/health", headers=health_headers).status_code == 401
        market_path = "/v1/market-evidence?instrument=SH600000"
        market = client.get(
            market_path,
            headers=_signed_headers(settings.hmac_secret, "GET", market_path),
        )
        assert market.status_code == 200
        assert market.json()["minute_volume_shares"] == 1_000_000

        first = client.post(
            "/v1/orders",
            content=body,
            headers=_signed_headers(settings.hmac_secret, "POST", "/v1/orders", body),
        )
        assert first.status_code == 200
        accepted = first.json()
        assert accepted["status"] == "accepted"
        repeated = client.post(
            "/v1/orders",
            content=body,
            headers=_signed_headers(settings.hmac_secret, "POST", "/v1/orders", body),
        )
        assert repeated.json()["order_id"] == accepted["order_id"]

        assert app.state.gateway_store.run_due_once(now=scheduled_for + timedelta(seconds=1))
        assert len(adapter.orders) == 1
        snapshot_path = "/v1/snapshot?account_ref=SIM-1"
        snapshot = client.get(
            snapshot_path,
            headers=_signed_headers(settings.hmac_secret, "GET", snapshot_path),
        ).json()
        assert snapshot["orders"][0]["client_order_id"] == payload["client_order_id"]
        assert snapshot["orders"][0]["order_id"] == accepted["order_id"]

        adapter.orders.append(
            {
                "provider_order_id": "external-1",
                "client_tag": "manual",
                "instrument": "SZ000001",
                "status": "submitted",
                "quantity": 100,
                "traded_quantity": 0,
            }
        )
        snapshot = client.get(
            snapshot_path,
            headers=_signed_headers(settings.hmac_secret, "GET", snapshot_path),
        ).json()
        assert any(
            item["client_order_id"] == "external:qmt:order:external-1"
            for item in snapshot["orders"]
        )
    assert adapter.closed


def test_gateway_rejects_live_mode_and_normalizes_qmt_symbols(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="live trading is unsupported"):
        GatewaySettings(
            database_url="postgresql+psycopg://example",
            hmac_secret="q" * 40,
            qmt_mini_path=tmp_path,
            qmt_account_id="account",
            account_ref="SIM-1",
            qmt_session_id=1,
            environment="live",
        ).validate()
    assert to_qmt_instrument("SH600000") == "600000.SH"
    assert from_qmt_instrument("000001.SZ") == "SZ000001"
    with pytest.raises(QmtGatewayError, match="unsupported"):
        to_qmt_instrument("US-AAPL")


def test_gateway_redistributes_slice_quantity_under_participation_cap(
    database_url: str,
) -> None:
    adapter = FakeQmtAdapter()
    adapter.minute_volume_shares = 50_000
    store = GatewayStore(database_url, adapter, account_ref="SIM-1")
    first_time = datetime.now(UTC) + timedelta(minutes=5)
    payload = _payload(first_time)
    payload["quantity"] = 2_000
    payload["execution_slices"] = [
        {**payload["execution_slices"][0], "sequence": 1, "quantity": 1_000},
        {
            **payload["execution_slices"][0],
            "sequence": 2,
            "scheduled_for": (first_time + timedelta(minutes=20)).isoformat(),
            "quantity": 1_000,
        },
    ]
    store.accept_parent(payload)
    assert store.run_due_once(now=first_time + timedelta(seconds=1))
    with open_database(database_url).connect() as connection:
        children = connection.execute(
            select(broker_gateway_children).order_by(broker_gateway_children.c.slice_index)
        ).all()
        attempt = connection.execute(select(broker_gateway_attempts)).one()
    assert [int(item.quantity) for item in children] == [500, 1_500]
    assert int(attempt.quantity) == 500
    assert attempt.market_evidence_json["allowed_quantity"] == 500
    assert len(adapter.orders) == 1
    assert adapter.orders[0]["quantity"] == 500


def test_partial_fill_is_canceled_and_replaced_with_persisted_attempts(
    database_url: str,
) -> None:
    adapter = FakeQmtAdapter()
    store = GatewayStore(
        database_url,
        adapter,
        account_ref="SIM-1",
        cancel_after_seconds=10,
        max_replacements=1,
        max_reprice_bps=20,
    )
    scheduled_for = datetime.now(UTC) + timedelta(minutes=5)
    payload = _payload(scheduled_for)
    payload["quantity"] = 200
    payload["execution_slices"][0]["quantity"] = 200
    accepted = store.accept_parent(payload)
    assert store.run_due_once(now=scheduled_for + timedelta(seconds=1))
    adapter.orders[0]["status"] = "partial"
    adapter.orders[0]["traded_quantity"] = 100
    adapter.events.append(
        {
            "event_type": "order",
            "payload": dict(adapter.orders[0]),
            "received_at": datetime.now(UTC),
        }
    )
    after_timeout = datetime.now(UTC) + timedelta(seconds=20)
    assert store.maintain_active_once(now=after_timeout)
    assert adapter.orders[0]["status"] == "canceled"
    assert store.maintain_active_once(now=after_timeout + timedelta(seconds=1))
    assert len(adapter.orders) == 2
    assert adapter.orders[1]["quantity"] == 100
    assert adapter.orders[1]["limit_price"] == 10.51
    adapter.orders[1]["status"] = "filled"
    adapter.orders[1]["traded_quantity"] = 100
    store.maintain_active_once(now=after_timeout + timedelta(seconds=2))
    with open_database(database_url).connect() as connection:
        attempts = connection.execute(
            select(broker_gateway_attempts).order_by(broker_gateway_attempts.c.attempt_no)
        ).all()
        child = connection.execute(select(broker_gateway_children)).one()
        parent = connection.execute(
            select(broker_gateway_parents).where(
                broker_gateway_parents.c.id == accepted["order_id"]
            )
        ).one()
        event_count = len(connection.execute(select(broker_gateway_events)).all())
    assert len(attempts) == 2
    assert int(child.filled_quantity) == 200
    assert child.status == "filled"
    assert parent.status == "filled"
    assert event_count == 1


def test_qmt_adapter_uses_official_trader_shapes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    constants = types.SimpleNamespace(
        STOCK_BUY=23,
        STOCK_SELL=24,
        FIX_PRICE=11,
        ORDER_UNREPORTED=48,
        ORDER_WAIT_REPORTING=49,
        ORDER_REPORTED=50,
        ORDER_REPORTED_CANCEL=51,
        ORDER_PARTSUCC_CANCEL=52,
        ORDER_PART_CANCEL=53,
        ORDER_CANCELED=54,
        ORDER_PART_SUCC=55,
        ORDER_SUCCEEDED=56,
        ORDER_JUNK=57,
    )

    class StockAccount:
        def __init__(self, account_id: str, account_type: str) -> None:
            self.account_id = account_id
            self.account_type = account_type

    class Trader:
        instance: Trader | None = None

        def __init__(self, path: str, session_id: int) -> None:
            self.path = path
            self.session_id = session_id
            self.orders: list[Any] = []
            self.stopped = False
            self.callback: Any = None
            Trader.instance = self

        def start(self) -> None:
            return None

        def register_callback(self, callback: Any) -> None:
            self.callback = callback

        def connect(self) -> int:
            return 0

        def subscribe(self, _account: StockAccount) -> int:
            return 0

        def query_stock_asset(self, _account: StockAccount) -> Any:
            return types.SimpleNamespace(cash=900.0, total_asset=1000.0)

        def query_stock_positions(self, _account: StockAccount) -> list[Any]:
            return [types.SimpleNamespace(stock_code="600000.SH", volume=100)]

        def query_stock_orders(self, _account: StockAccount, _cancelable: bool) -> list[Any]:
            return self.orders

        def query_stock_trades(self, _account: StockAccount) -> list[Any]:
            return []

        def order_stock(self, *args: Any) -> int:
            self.orders.append(
                types.SimpleNamespace(
                    stock_code=args[1],
                    order_id=88,
                    order_sysid="SYS-88",
                    order_remark=args[7],
                    order_status=constants.ORDER_REPORTED,
                    order_volume=args[3],
                    traded_volume=0,
                    status_msg="",
                )
            )
            return 88

        def cancel_order_stock(self, _account: StockAccount, order_id: int) -> int:
            for order in self.orders:
                if order.order_id == order_id:
                    order.order_status = constants.ORDER_CANCELED
                    return 0
            return -1

        def stop(self) -> None:
            self.stopped = True

    package = types.ModuleType("xtquant")
    package.xtconstant = constants
    current = datetime.now(UTC)

    class Frame:
        def to_dict(self, orientation: str) -> list[dict[str, Any]]:
            assert orientation == "records"
            return [
                {
                    "time": int((current - timedelta(minutes=1)).timestamp() * 1000),
                    "volume": 1000,
                }
            ]

    xtdata = types.SimpleNamespace(
        subscribe_quote=lambda *_args, **_kwargs: 1,
        get_market_data_ex=lambda **kwargs: {kwargs["stock_list"][0]: Frame()},
        get_full_tick=lambda symbols: {
            symbols[0]: {
                "time": int(current.timestamp() * 1000),
                "askPrice": [10.51],
                "bidPrice": [10.49],
            }
        },
    )
    package.xtdata = xtdata
    trader_module = types.ModuleType("xtquant.xttrader")
    trader_module.XtQuantTrader = Trader
    trader_module.XtQuantTraderCallback = object
    type_module = types.ModuleType("xtquant.xttype")
    type_module.StockAccount = StockAccount
    monkeypatch.setitem(sys.modules, "xtquant", package)
    monkeypatch.setitem(sys.modules, "xtquant.xttrader", trader_module)
    monkeypatch.setitem(sys.modules, "xtquant.xttype", type_module)

    adapter = QmtBrokerAdapter(
        mini_path=tmp_path,
        account_id="provider-account",
        account_ref="SIM-1",
        session_id=178901,
    )
    assert adapter.health()["connected"]
    order_id = adapter.submit_limit_order(
        account_ref="SIM-1",
        instrument="SH600000",
        side="buy",
        quantity=100,
        limit_price=10.5,
        client_tag="QL01234567890123456789",
    )
    assert order_id == "88"
    assert adapter.submit_limit_order(
        account_ref="SIM-1",
        instrument="SH600000",
        side="buy",
        quantity=100,
        limit_price=10.5,
        client_tag="QL01234567890123456789",
    ) == "88"
    snapshot = adapter.snapshot("SIM-1")
    assert snapshot["positions"] == [{"instrument": "SH600000", "quantity": 100}]
    assert snapshot["orders"][0]["status"] == "submitted"
    assert Trader.instance is not None and Trader.instance.callback is not None
    Trader.instance.callback.on_stock_order(Trader.instance.orders[0])
    assert adapter.drain_events()[0]["event_type"] == "order"
    evidence = adapter.market_evidence("SH600000", current)
    assert evidence["minute_volume_shares"] == 100_000
    assert adapter.cancel_order(account_ref="SIM-1", provider_order_id="88")
    adapter.close()
    assert Trader.instance is not None and Trader.instance.stopped
