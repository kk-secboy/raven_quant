from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol


class BrokerAdapter(Protocol):
    provider_name: str

    def health(self) -> dict[str, Any]: ...

    def submit_limit_order(
        self,
        *,
        account_ref: str,
        instrument: str,
        side: str,
        quantity: int,
        limit_price: float,
        client_tag: str,
    ) -> str: ...

    def snapshot(self, account_ref: str) -> dict[str, Any]: ...

    def market_evidence(self, instrument: str, as_of: datetime) -> dict[str, Any]: ...

    def cancel_order(self, *, account_ref: str, provider_order_id: str) -> bool: ...

    def drain_events(self) -> list[dict[str, Any]]: ...

    def close(self) -> None: ...
