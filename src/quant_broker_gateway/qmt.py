from __future__ import annotations

import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


class QmtGatewayError(RuntimeError):
    pass


class QmtBrokerAdapter:
    """Thin, sandbox-only wrapper around the official Windows XtQuant trader API."""

    provider_name = "qmt"

    def __init__(
        self,
        *,
        mini_path: Path,
        account_id: str,
        account_ref: str,
        session_id: int,
        volume_multiplier: int = 100,
        max_quote_age_seconds: int = 10,
    ) -> None:
        self.mini_path = mini_path
        self.account_id = account_id
        self.account_ref = account_ref
        self.session_id = session_id
        self.volume_multiplier = volume_multiplier
        self.max_quote_age_seconds = max_quote_age_seconds
        self._lock = threading.RLock()
        self._trader: Any = None
        self._account: Any = None
        self._constant: Any = None
        self._xtdata: Any = None
        self._subscribed_symbols: set[str] = set()
        self._events: list[dict[str, Any]] = []

    def health(self) -> dict[str, Any]:
        with self._lock:
            trader, account, _ = self._runtime()
            asset = trader.query_stock_asset(account)
            if asset is None:
                raise QmtGatewayError("QMT asset query failed")
            return {
                "status": "ok",
                "provider": self.provider_name,
                "account_ref": self.account_ref,
                "connected": True,
            }

    def submit_limit_order(
        self,
        *,
        account_ref: str,
        instrument: str,
        side: str,
        quantity: int,
        limit_price: float,
        client_tag: str,
    ) -> str:
        self._check_account(account_ref)
        if len(client_tag.encode("ascii")) > 24:
            raise QmtGatewayError("QMT client tag exceeds the 24-byte order_remark limit")
        if quantity <= 0 or limit_price <= 0:
            raise QmtGatewayError("QMT order quantity and price must be positive")
        if side not in {"buy", "sell"}:
            raise QmtGatewayError(f"unsupported QMT side: {side}")
        with self._lock:
            trader, account, constant = self._runtime()
            for order in trader.query_stock_orders(account, False) or []:
                if str(getattr(order, "order_remark", "")) == client_tag:
                    provider_order_id = str(getattr(order, "order_id", "")).strip()
                    if provider_order_id:
                        return provider_order_id
            order_type = constant.STOCK_BUY if side == "buy" else constant.STOCK_SELL
            order_id = trader.order_stock(
                account,
                to_qmt_instrument(instrument),
                order_type,
                int(quantity),
                constant.FIX_PRICE,
                float(limit_price),
                "quantlab",
                client_tag,
            )
            if int(order_id) <= 0:
                raise QmtGatewayError("QMT rejected the synchronous order request")
            return str(order_id)

    def snapshot(self, account_ref: str) -> dict[str, Any]:
        self._check_account(account_ref)
        with self._lock:
            trader, account, constant = self._runtime()
            asset = trader.query_stock_asset(account)
            if asset is None:
                raise QmtGatewayError("QMT asset query failed")
            positions = trader.query_stock_positions(account) or []
            orders = trader.query_stock_orders(account, False) or []
            trades = trader.query_stock_trades(account) or []
            return {
                "as_of": datetime.now(UTC).isoformat(),
                "cash": float(asset.cash),
                "equity": float(asset.total_asset),
                "positions": [
                    {
                        "instrument": from_qmt_instrument(str(row.stock_code)),
                        "quantity": int(row.volume),
                    }
                    for row in positions
                    if int(row.volume) != 0
                ],
                "orders": [self._normalize_order(row, constant) for row in orders],
                "trades": [self._normalize_trade(row) for row in trades],
            }

    def market_evidence(self, instrument: str, as_of: datetime) -> dict[str, Any]:
        current = as_of.astimezone(UTC)
        symbol = to_qmt_instrument(instrument)
        with self._lock:
            self._runtime()
            if symbol not in self._subscribed_symbols:
                sequence = self._xtdata.subscribe_quote(symbol, period="1m", count=10)
                if sequence is None or int(sequence) < 0:
                    raise QmtGatewayError(f"cannot subscribe to QMT minute bars for {symbol}")
                self._subscribed_symbols.add(symbol)
            market = self._xtdata.get_market_data_ex(
                field_list=["time", "volume"],
                stock_list=[symbol],
                period="1m",
                start_time=current.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d"),
                end_time=current.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d%H%M%S"),
                count=10,
                dividend_type="none",
                fill_data=False,
            )
            frame = market.get(symbol) if isinstance(market, dict) else None
            minute = _latest_completed_minute(frame, current)
            ticks = self._xtdata.get_full_tick([symbol])
            tick = ticks.get(symbol) if isinstance(ticks, dict) else None
            if not isinstance(tick, dict):
                raise QmtGatewayError(f"QMT full-tick quote is unavailable for {symbol}")
            quote_at = _tick_datetime(tick)
            quote_age = max(0.0, (current - quote_at).total_seconds())
            if quote_age > self.max_quote_age_seconds:
                raise QmtGatewayError(
                    f"QMT quote for {symbol} is {int(quote_age)} seconds old"
                )
            ask_price = _first_positive(tick.get("askPrice"), "askPrice")
            bid_price = _first_positive(tick.get("bidPrice"), "bidPrice")
            return {
                "source": "qmt_xtdata_1m_full_tick",
                "instrument": instrument,
                "minute_ended_at": minute["ended_at"].isoformat(),
                "minute_volume_shares": int(minute["volume"] * self.volume_multiplier),
                "volume_multiplier": self.volume_multiplier,
                "quote_as_of": quote_at.isoformat(),
                "quote_age_seconds": quote_age,
                "bid_price": bid_price,
                "ask_price": ask_price,
            }

    def cancel_order(self, *, account_ref: str, provider_order_id: str) -> bool:
        self._check_account(account_ref)
        try:
            numeric_order_id = int(provider_order_id)
        except ValueError as exc:
            raise QmtGatewayError("QMT provider order id must be numeric") from exc
        with self._lock:
            trader, account, constant = self._runtime()
            terminal = {
                constant.ORDER_PART_CANCEL,
                constant.ORDER_CANCELED,
                constant.ORDER_SUCCEEDED,
                constant.ORDER_JUNK,
            }
            for order in trader.query_stock_orders(account, False) or []:
                if int(getattr(order, "order_id", -1)) == numeric_order_id:
                    if int(getattr(order, "order_status", -1)) in terminal:
                        return True
                    break
            return int(trader.cancel_order_stock(account, numeric_order_id)) == 0

    def drain_events(self) -> list[dict[str, Any]]:
        with self._lock:
            result = list(self._events)
            self._events.clear()
            return result

    def close(self) -> None:
        with self._lock:
            if self._trader is not None:
                self._trader.stop()
            self._trader = None
            self._account = None
            self._constant = None
            self._xtdata = None
            self._subscribed_symbols.clear()

    def _runtime(self) -> tuple[Any, Any, Any]:
        if self._trader is not None:
            return self._trader, self._account, self._constant
        try:
            from xtquant import xtconstant, xtdata
            from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
            from xtquant.xttype import StockAccount
        except ImportError as exc:
            raise QmtGatewayError(
                "xtquant is unavailable; install the package shipped with MiniQMT on Windows"
            ) from exc
        if not self.mini_path.exists():
            raise QmtGatewayError(f"MiniQMT user-data path does not exist: {self.mini_path}")
        trader = XtQuantTrader(str(self.mini_path), self.session_id)

        adapter = self

        class GatewayCallback(XtQuantTraderCallback):
            def on_disconnected(self) -> None:
                adapter._record_event("disconnected", {})

            def on_stock_order(self, order: Any) -> None:
                adapter._record_event("order", adapter._normalize_order(order, xtconstant))

            def on_stock_trade(self, trade: Any) -> None:
                adapter._record_event("trade", adapter._normalize_trade(trade))

            def on_order_error(self, error: Any) -> None:
                adapter._record_event(
                    "order_error",
                    {
                        "provider_order_id": str(getattr(error, "order_id", "")),
                        "client_tag": str(getattr(error, "order_remark", "")),
                        "error_id": int(getattr(error, "error_id", -1)),
                        "error_message": str(getattr(error, "error_msg", "")),
                    },
                )

            def on_cancel_error(self, error: Any) -> None:
                adapter._record_event(
                    "cancel_error",
                    {
                        "provider_order_id": str(getattr(error, "order_id", "")),
                        "error_id": int(getattr(error, "error_id", -1)),
                        "error_message": str(getattr(error, "error_msg", "")),
                    },
                )

        trader.register_callback(GatewayCallback())
        trader.start()
        if int(trader.connect()) != 0:
            trader.stop()
            raise QmtGatewayError("cannot connect to the running MiniQMT client")
        account = StockAccount(self.account_id, "STOCK")
        if int(trader.subscribe(account)) != 0:
            trader.stop()
            raise QmtGatewayError("cannot subscribe to the configured MiniQMT account")
        self._trader = trader
        self._account = account
        self._constant = xtconstant
        self._xtdata = xtdata
        return trader, account, xtconstant

    def _record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(
                {"event_type": event_type, "payload": payload, "received_at": datetime.now(UTC)}
            )

    def _check_account(self, account_ref: str) -> None:
        if account_ref != self.account_ref:
            raise QmtGatewayError("QMT account reference mismatch")

    @staticmethod
    def _normalize_order(order: Any, constant: Any) -> dict[str, Any]:
        status_map = {
            constant.ORDER_UNREPORTED: "pending",
            constant.ORDER_WAIT_REPORTING: "pending",
            constant.ORDER_REPORTED: "submitted",
            constant.ORDER_REPORTED_CANCEL: "cancel_pending",
            constant.ORDER_PARTSUCC_CANCEL: "cancel_pending",
            constant.ORDER_PART_CANCEL: "canceled",
            constant.ORDER_CANCELED: "canceled",
            constant.ORDER_PART_SUCC: "partial",
            constant.ORDER_SUCCEEDED: "filled",
            constant.ORDER_JUNK: "rejected",
        }
        return {
            "provider_order_id": str(order.order_id),
            "provider_order_sysid": str(getattr(order, "order_sysid", "")),
            "client_tag": str(getattr(order, "order_remark", "")),
            "instrument": from_qmt_instrument(str(order.stock_code)),
            "status": status_map.get(int(order.order_status), "error"),
            "quantity": int(order.order_volume),
            "traded_quantity": int(order.traded_volume),
            "status_message": str(getattr(order, "status_msg", "")),
        }

    @staticmethod
    def _normalize_trade(trade: Any) -> dict[str, Any]:
        return {
            "provider_trade_id": str(trade.traded_id),
            "provider_order_id": str(trade.order_id),
            "client_tag": str(getattr(trade, "order_remark", "")),
            "instrument": from_qmt_instrument(str(trade.stock_code)),
            "status": "filled",
            "quantity": int(trade.traded_volume),
            "price": float(trade.traded_price),
        }


def to_qmt_instrument(instrument: str) -> str:
    match = re.fullmatch(r"(SH|SZ|BJ)(\d{6})", instrument.strip().upper())
    if match is None:
        raise QmtGatewayError(f"unsupported QuantLab instrument: {instrument}")
    return f"{match.group(2)}.{match.group(1)}"


def from_qmt_instrument(instrument: str) -> str:
    match = re.fullmatch(r"(\d{6})\.(SH|SZ|BJ)", instrument.strip().upper())
    if match is None:
        raise QmtGatewayError(f"unsupported QMT instrument: {instrument}")
    return f"{match.group(2)}{match.group(1)}"


def _latest_completed_minute(frame: Any, current: datetime) -> dict[str, Any]:
    if frame is None:
        raise QmtGatewayError("QMT minute-bar query returned no data")
    if hasattr(frame, "to_dict"):
        records = frame.to_dict("records")
    elif isinstance(frame, list):
        records = frame
    else:
        raise QmtGatewayError("QMT minute-bar response has an unsupported shape")
    boundary = current.astimezone(UTC).replace(second=0, microsecond=0)
    completed: list[tuple[datetime, float]] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        try:
            started_at = datetime.fromtimestamp(float(row["time"]) / 1000, UTC)
            volume = float(row["volume"])
        except (KeyError, TypeError, ValueError):
            continue
        if started_at < boundary and volume > 0:
            completed.append((started_at, volume))
    if not completed:
        raise QmtGatewayError("QMT has no completed positive-volume minute bar")
    started_at, volume = max(completed, key=lambda item: item[0])
    return {"ended_at": started_at.replace(second=0, microsecond=0), "volume": volume}


def _tick_datetime(tick: dict[str, Any]) -> datetime:
    raw_milliseconds = tick.get("time")
    if raw_milliseconds is not None:
        try:
            return datetime.fromtimestamp(float(raw_milliseconds) / 1000, UTC)
        except (TypeError, ValueError, OSError):
            pass
    raw = str(tick.get("timetag") or tick.get("stime") or "").strip()
    for pattern in ("%Y%m%d %H:%M:%S.%f", "%Y%m%d %H:%M:%S", "%Y%m%d%H%M%S.%f"):
        try:
            parsed = datetime.strptime(raw, pattern).replace(tzinfo=ZoneInfo("Asia/Shanghai"))
            return parsed.astimezone(UTC)
        except ValueError:
            continue
    raise QmtGatewayError("QMT full-tick timestamp is invalid")


def _first_positive(values: Any, name: str) -> float:
    if not isinstance(values, (list, tuple)) or not values:
        raise QmtGatewayError(f"QMT {name} is unavailable")
    value = float(values[0])
    if value <= 0:
        raise QmtGatewayError(f"QMT {name} is not positive")
    return value
