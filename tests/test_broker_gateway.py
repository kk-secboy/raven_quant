from __future__ import annotations

import hashlib
import hmac
import json
import threading
from datetime import UTC, date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import insert, update

from quant_data.config import Settings
from quant_data.database import (
    broker_order_outbox,
    open_database,
    paper_fills,
    paper_orders,
    paper_portfolios,
    paper_positions,
    portfolio_batches,
    strategies,
    strategy_versions,
)
from quant_platform.broker_gateway import BrokerGatewayError, BrokerStore
from quant_platform.runtime_secret_store import RuntimeSecretStore


class SandboxGatewayHandler(BaseHTTPRequestHandler):
    secret = "s" * 40
    orders: list[dict] = []
    snapshot: dict = {}

    def do_GET(self) -> None:  # noqa: N802
        self._verify("")
        if self.path.startswith("/v1/snapshot?"):
            self._json(self.snapshot)
        else:
            self._json({"status": "ok", "environment": "sandbox"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length).decode()
        self._verify(body)
        payload = json.loads(body)
        self.orders.append(payload)
        self._json(
            {
                "status": "accepted",
                "environment": "sandbox",
                "order_id": f"sandbox-{len(self.orders)}",
                "client_order_id": payload["client_order_id"],
            }
        )

    def log_message(self, _format: str, *_args) -> None:
        return

    def _verify(self, body: str) -> None:
        timestamp = self.headers["X-QuantLab-Timestamp"]
        nonce = self.headers["X-QuantLab-Nonce"]
        message = f"{timestamp}.{nonce}.{self.command}.{self.path}.{body}".encode()
        expected = hmac.new(self.secret.encode(), message, hashlib.sha256).hexdigest()
        assert hmac.compare_digest(expected, self.headers["X-QuantLab-Signature"])

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _settings(database_url: str, gateway_url: str = "", mode: str = "disabled") -> Settings:
    return Settings(
        api_url="",
        token="",
        data_root=Path("E:/projects/rdagent-python/.tmp/broker-test"),
        database_url=database_url,
        broker_mode=mode,
        broker_gateway_url=gateway_url,
        broker_hmac_secret=SandboxGatewayHandler.secret if gateway_url else "",
        broker_max_order_notional=1_000_000,
    )


def _seed_filled_batch(database_url: str) -> tuple[str, str]:
    now = datetime.now(UTC)
    with open_database(database_url).begin() as connection:
        connection.execute(
            insert(strategies).values(
                id="strategy",
                name="broker sandbox strategy",
                description="Strategy fixture for broker sandbox replay.",
                status="approved",
                created_by="pytest",
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            insert(strategy_versions).values(
                id="strategy-version",
                strategy_id="strategy",
                version=1,
                status="approved",
                benchmark="SH000300",
                universe="cn_all",
                config_json={},
                created_by="pytest",
                approved_by="pytest",
                approval_reason="test fixture",
                created_at=now,
                approved_at=now,
            )
        )
        connection.execute(
            insert(paper_portfolios).values(
                id="portfolio",
                name="broker sandbox portfolio",
                strategy_version_id="strategy-version",
                dataset="snapshot",
                status="active",
                base_currency="CNY",
                initial_cash=5_000_000,
                cash=4_900_000,
                nav=5_000_000,
                high_water_mark=5_000_000,
                created_by="pytest",
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            insert(portfolio_batches).values(
                id="batch-000000000001",
                portfolio_id="portfolio",
                as_of_date=date(2026, 7, 10),
                trade_date=date(2026, 7, 11),
                status="succeeded",
                idempotency_key="paper-rebalance:portfolio:2026-07-10",
                artifact_path="/tmp/paper",
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            insert(paper_orders).values(
                id="paper-order-000001",
                batch_id="batch-000000000001",
                portfolio_id="portfolio",
                instrument="SH600000",
                side="buy",
                order_type="market_open",
                target_weight=0.02,
                requested_quantity=10_000,
                status="filled",
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            insert(paper_fills).values(
                id="paper-fill-000001",
                order_id="paper-order-000001",
                fill_time=now,
                quantity=10_000,
                price=10,
                gross_value=100_000,
                fee=50,
                slippage=0.0005,
                created_at=now,
            )
        )
        connection.execute(
            insert(paper_positions).values(
                portfolio_id="portfolio",
                instrument="SH600000",
                industry="银行",
                take_profit_stage=0,
                quantity=10_000,
                avg_cost=10,
                market_price=10,
                market_value=100_000,
                weight=0.02,
                realized_pnl=0,
                unrealized_pnl=0,
                updated_at=now,
            )
        )
    return "batch-000000000001", "paper-order-000001"


def test_broker_boundary_is_disabled_and_rejects_live_mode(
    database_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = BrokerStore(_settings(database_url))
    assert store.readiness() == {
        "status": "disabled",
        "mode": "disabled",
        "live_supported": False,
        "gateway_configured": False,
        "destination_counts": {},
        "outbox_counts": {},
        "max_order_notional": 1_000_000,
        "max_attempts": 3,
    }
    _seed_filled_batch(database_url)
    destination = store.create_destination(
        name="safe sandbox",
        account_ref="SIM-001",
        portfolio_id="portfolio",
        actor="admin-a",
    )
    with pytest.raises(BrokerGatewayError, match="disabled"):
        store.request_activation(destination["id"], actor="admin-a")

    monkeypatch.setenv("BROKER_MODE", "live")
    with pytest.raises(ValueError, match="live trading is unsupported"):
        Settings.from_env(tmp_path / "missing.env")


def test_broker_store_hot_loads_encrypted_gateway_without_restart(database_url: str) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), SandboxGatewayHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        key = Fernet.generate_key().decode("ascii")
        settings = _settings(database_url, "http://127.0.0.1:1", mode="sandbox")
        settings.platform_secret_key = key
        secrets = RuntimeSecretStore(database_url, key)
        secrets.put(
            "broker_gateway",
            {
                "gateway_url": f"http://127.0.0.1:{server.server_port}",
                "hmac_secret": SandboxGatewayHandler.secret,
            },
            metadata={"enabled": True, "endpoint_host": "127.0.0.1"},
            updated_by=None,
        )
        store = BrokerStore(settings)
        assert store.readiness(probe=True)["status"] == "ok"

        secrets.put(
            "broker_gateway",
            {"gateway_url": "", "hmac_secret": ""},
            metadata={"enabled": False, "endpoint_host": ""},
            updated_by=None,
        )
        disabled = store.readiness(probe=True)
        assert disabled["status"] == "bootstrap_required"
        assert disabled["gateway_configured"] is False

        secrets.put(
            "broker_gateway",
            {
                "gateway_url": f"http://127.0.0.1:{server.server_port}",
                "hmac_secret": SandboxGatewayHandler.secret,
            },
            metadata={"enabled": True, "endpoint_host": "127.0.0.1"},
            updated_by=None,
        )
        assert store.readiness(probe=True)["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()


def test_broker_store_fails_closed_when_database_gateway_secret_is_unreadable(
    database_url: str,
) -> None:
    key = Fernet.generate_key().decode("ascii")
    RuntimeSecretStore(database_url, key).put(
        "broker_gateway",
        {"gateway_url": "https://database.example", "hmac_secret": "d" * 40},
        metadata={"enabled": True, "endpoint_host": "database.example"},
        updated_by=None,
    )
    settings = _settings(database_url, "http://127.0.0.1:1", mode="sandbox")
    settings.platform_secret_key = ""
    result = BrokerStore(settings).readiness(probe=True)
    assert result["status"] == "unavailable"
    assert "credential storage failed" in result["message"]


def test_two_person_sandbox_release_is_idempotent_and_signed(database_url: str) -> None:
    SandboxGatewayHandler.orders = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), SandboxGatewayHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        settings = _settings(database_url, f"http://127.0.0.1:{server.server_port}", mode="sandbox")
        store = BrokerStore(settings)
        assert store.readiness(probe=True)["status"] == "ok"
        batch_id, source_order_id = _seed_filled_batch(database_url)
        destination = store.create_destination(
            name="QMT integration sandbox",
            account_ref="SIM-500W",
            portfolio_id="portfolio",
            actor="admin-a",
        )
        pending = store.request_activation(destination["id"], actor="admin-a")
        assert pending["status"] == "pending_activation"
        with pytest.raises(ValueError, match="second administrator"):
            store.approve_activation(destination["id"], actor="admin-a")
        with pytest.raises(ValueError, match="matched broker reconciliation"):
            store.approve_activation(destination["id"], actor="admin-b")
        SandboxGatewayHandler.snapshot = {
            "status": "ok",
            "environment": "sandbox",
            "account_ref": "SIM-500W",
            "as_of": datetime.now(UTC).isoformat(),
            "cash": 4_900_000,
            "equity": 5_000_000,
            "positions": [{"instrument": "SH600000", "quantity": 10_000}],
            "orders": [],
            "trades": [],
        }
        preflight = store.reconcile(destination["id"], actor="admin-a")
        assert preflight["status"] == "matched"
        armed = store.approve_activation(destination["id"], actor="admin-b")
        assert armed["status"] == "armed"

        staged = store.stage_batch(destination["id"], batch_id, actor="admin-a")
        repeated = store.stage_batch(destination["id"], batch_id, actor="admin-a")
        assert len(staged) == 1
        assert repeated[0]["id"] == staged[0]["id"]
        assert staged[0]["source_order_id"] == source_order_id
        assert staged[0]["payload"]["environment"] == "sandbox"
        assert staged[0]["payload"]["execution_policy"]["execution_algorithm"] == "twap"
        assert len(staged[0]["payload"]["execution_slices"]) == 10
        assert (
            sum(item["quantity"] for item in staged[0]["payload"]["execution_slices"])
            == staged[0]["payload"]["quantity"]
        )
        with pytest.raises(ValueError, match="second administrator"):
            store.approve_batch(destination["id"], batch_id, actor="admin-a")
        approved = store.approve_batch(destination["id"], batch_id, actor="admin-b")
        assert approved[0]["status"] == "approved"

        original_payload = approved[0]["payload"]
        with open_database(database_url).begin() as connection:
            connection.execute(
                update(broker_order_outbox)
                .where(broker_order_outbox.c.id == approved[0]["id"])
                .values(payload_json={**original_payload, "instrument": "SH600999"})
            )
        with pytest.raises(ValueError, match="payload digest mismatch"):
            store.dispatch_batch(destination["id"], batch_id, actor="admin-b")
        assert SandboxGatewayHandler.orders == []
        with open_database(database_url).begin() as connection:
            connection.execute(
                update(broker_order_outbox)
                .where(broker_order_outbox.c.id == approved[0]["id"])
                .values(payload_json=original_payload)
            )

        result = store.dispatch_batch(destination["id"], batch_id, actor="admin-b")
        assert result == {
            "destination_id": destination["id"],
            "batch_id": batch_id,
            "submitted": 1,
            "failed": 0,
            "mode": "sandbox",
        }
        submitted = store.list_outbox(destination_id=destination["id"])
        assert submitted[0]["status"] == "submitted"
        assert submitted[0]["broker_order_id"] == "sandbox-1"
        assert len(SandboxGatewayHandler.orders) == 1
        with pytest.raises(ValueError, match="no approved sandbox orders"):
            store.dispatch_batch(destination["id"], batch_id, actor="admin-b")
        assert {event["event_type"] for event in store.list_events(destination["id"])} >= {
            "destination_armed",
            "sandbox_batch_staged",
            "sandbox_batch_approved",
            "sandbox_order_submitted",
        }

        SandboxGatewayHandler.snapshot = {
            "status": "ok",
            "environment": "sandbox",
            "account_ref": "SIM-500W",
            "as_of": datetime.now(UTC).isoformat(),
            "cash": 4_900_000,
            "equity": 5_000_000,
            "positions": [{"instrument": "SH600000", "quantity": 10_000}],
            "orders": [
                {
                    "client_order_id": SandboxGatewayHandler.orders[0]["client_order_id"],
                    "order_id": "sandbox-1",
                    "status": "filled",
                }
            ],
            "trades": [],
        }
        matched = store.reconcile(destination["id"], actor="admin-a")
        assert matched["status"] == "matched"
        assert matched["differences"] == []

        SandboxGatewayHandler.snapshot["positions"] = [
            {"instrument": "SH600000", "quantity": 9_000}
        ]
        mismatch = store.reconcile(destination["id"], actor="admin-a")
        assert mismatch["status"] == "mismatch"
        assert any(item["type"] == "position_mismatch" for item in mismatch["differences"])
        assert store.get_destination(destination["id"])["status"] == "locked_mismatch"
        assert store.readiness()["status"] == "degraded"
        assert any(item["category"] == "broker_reconciliation" for item in store.alerts.list())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
