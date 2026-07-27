"""Platform safe mode (design draft 11.3).

Severe data, ledger or version anomalies engage a platform-level safe mode:
new recommendations and new simulation orders stop (fail closed), while
read paths, reconciliation/replay and the recovery flow keep working.

State lives in the ``platform_safe_mode_state`` singleton row (migration
0047). Activation may be automatic (ledger conservation failure, data
quality gate failure, persistent degraded/uncertified NAV) or manual;
clearing is always manual and requires an actor and a meaningful reason,
optionally gated on a passing operational health check.

Activation is idempotent: while safe mode is already active, repeated
triggers return the current state without writing duplicate alerts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from quant_data.database import (
    open_database,
    platform_safe_mode_state,
    row_dict,
    simulation_nav,
    simulation_portfolios,
)

from .alert_store import AlertStore

SAFE_MODE_ROW_ID = "platform"

# An active simulation account whose latest N NAV rows are all degraded or
# uncertified is a persistent ledger anomaly, not a transient one.
PERSISTENT_NAV_FAILURE_ROWS = 3

RECOVERY_HINT = (
    "排查触发源并完成对账/恢复后，由操作员调用 safe-mode release "
    "（actor + reason）解除；解除前新建议与新模拟订单保持阻断。"
)


class SafeModeActiveError(ValueError):
    """Raised when a blocked write path is attempted during safe mode."""


def _now() -> datetime:
    return datetime.now(UTC)


class SafeModeStore:
    """Durable platform safe-mode state with idempotent activation."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.engine = open_database(database_url)
        self.alerts = AlertStore(database_url)

    @staticmethod
    def _ensure_row(connection: Any) -> None:
        """Recreate the singleton row when it is missing (e.g. after a reset)."""

        connection.execute(
            pg_insert(platform_safe_mode_state)
            .values(
                id=SAFE_MODE_ROW_ID,
                active=False,
                reason="",
                source="",
                triggered_by="",
                updated_at=_now(),
            )
            .on_conflict_do_nothing(index_elements=[platform_safe_mode_state.c.id])
        )

    def status(self) -> dict[str, Any]:
        with self.engine.begin() as connection:
            self._ensure_row(connection)
            row = connection.execute(
                select(platform_safe_mode_state).where(
                    platform_safe_mode_state.c.id == SAFE_MODE_ROW_ID
                )
            ).first()
        if row is None:  # pragma: no cover - guarded by _ensure_row
            raise RuntimeError("platform safe-mode state row is missing (migration 0047)")
        result = row_dict(row)
        result["details"] = result.pop("details_json") or {}
        return result

    def is_active(self) -> bool:
        return bool(self.status()["active"])

    def activate(
        self,
        *,
        reason: str,
        source: str,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Engage safe mode; a no-op (without a duplicate alert) when active."""

        if not reason.strip() or not source.strip() or not actor.strip():
            raise ValueError("safe-mode activation requires reason, source and actor")
        now = _now()
        with self.engine.begin() as connection:
            self._ensure_row(connection)
            current = connection.execute(
                select(platform_safe_mode_state)
                .where(platform_safe_mode_state.c.id == SAFE_MODE_ROW_ID)
                .with_for_update()
            ).first()
            if current is None:  # pragma: no cover - guarded by _ensure_row
                raise RuntimeError(
                    "platform safe-mode state row is missing (migration 0047)"
                )
            if current.active:
                return self.status()
            connection.execute(
                update(platform_safe_mode_state)
                .where(platform_safe_mode_state.c.id == SAFE_MODE_ROW_ID)
                .values(
                    active=True,
                    reason=reason.strip(),
                    source=source.strip(),
                    triggered_by=actor.strip(),
                    triggered_at=now,
                    details_json=details or {},
                    cleared_by=None,
                    cleared_at=None,
                    clear_reason=None,
                    updated_at=now,
                )
            )
        self.alerts.create(
            source_type="platform",
            source_id=SAFE_MODE_ROW_ID,
            severity="critical",
            category="safe_mode_activated",
            title=f"平台进入 safe_mode：{source}",
            message=f"{reason.strip()}。{RECOVERY_HINT}",
            dedupe_key=f"safe-mode:activated:{source}:{now.date().isoformat()}",
            details={
                "source": source.strip(),
                "reason": reason.strip(),
                "triggered_by": actor.strip(),
                "recovery_hint": RECOVERY_HINT,
                **(details or {}),
            },
        )
        return self.status()

    def deactivate(
        self,
        *,
        actor: str,
        reason: str,
        require_health_ok: bool = False,
        health_status: str | None = None,
    ) -> dict[str, Any]:
        """Clear safe mode; always a deliberate manual action."""

        if len(actor.strip()) < 2:
            raise ValueError("safe-mode release requires a responsible actor")
        if len(reason.strip()) < 10:
            raise ValueError("safe-mode release requires a meaningful reason")
        if require_health_ok and health_status != "ok":
            raise ValueError(
                "safe-mode release requires a passing operational health check "
                f"(latest status: {health_status or 'missing'})"
            )
        now = _now()
        with self.engine.begin() as connection:
            self._ensure_row(connection)
            current = connection.execute(
                select(platform_safe_mode_state)
                .where(platform_safe_mode_state.c.id == SAFE_MODE_ROW_ID)
                .with_for_update()
            ).first()
            if current is None:  # pragma: no cover - guarded by _ensure_row
                raise RuntimeError(
                    "platform safe-mode state row is missing (migration 0047)"
                )
            if not current.active:
                raise ValueError("safe mode is not active")
            connection.execute(
                update(platform_safe_mode_state)
                .where(platform_safe_mode_state.c.id == SAFE_MODE_ROW_ID)
                .values(
                    active=False,
                    cleared_by=actor.strip(),
                    cleared_at=now,
                    clear_reason=reason.strip(),
                    updated_at=now,
                )
            )
        self.alerts.create(
            source_type="platform",
            source_id=SAFE_MODE_ROW_ID,
            severity="info",
            category="safe_mode_released",
            title="平台 safe_mode 已解除",
            message=f"{actor.strip()} 解除 safe_mode：{reason.strip()}",
            dedupe_key=f"safe-mode:released:{now.date().isoformat()}:{actor.strip()}",
            details={"actor": actor.strip(), "reason": reason.strip()},
        )
        return self.status()

    def assert_inactive(self, *, action: str = "this operation") -> None:
        """Fail closed on blocked write paths while safe mode is active."""

        state = self.status()
        if state["active"]:
            raise SafeModeActiveError(
                f"platform safe_mode is active since {state['triggered_at']} "
                f"(source {state['source']}): {state['reason']}. "
                f"{action} is blocked until a manual safe-mode release. {RECOVERY_HINT}"
            )

    def check_persistent_nav_anomalies(self) -> dict[str, Any] | None:
        """Auto-trigger on persistent degraded/uncertified simulation NAV.

        Returns the activation state when safe mode was (or already is)
        engaged, otherwise None.
        """

        offenders: list[dict[str, Any]] = []
        with self.engine.connect() as connection:
            portfolios = connection.execute(
                select(simulation_portfolios.c.id, simulation_portfolios.c.name).where(
                    simulation_portfolios.c.status == "active"
                )
            ).all()
            for portfolio in portfolios:
                rows = connection.execute(
                    select(
                        simulation_nav.c.trade_date,
                        simulation_nav.c.status,
                        simulation_nav.c.performance_certified,
                    )
                    .where(simulation_nav.c.portfolio_id == portfolio.id)
                    .order_by(simulation_nav.c.trade_date.desc())
                    .limit(PERSISTENT_NAV_FAILURE_ROWS)
                ).all()
                if len(rows) < PERSISTENT_NAV_FAILURE_ROWS:
                    continue
                if all(
                    str(row.status) == "degraded" or not bool(row.performance_certified)
                    for row in rows
                ):
                    offenders.append(
                        {
                            "portfolio_id": str(portfolio.id),
                            "name": str(portfolio.name),
                            "trade_dates": [str(row.trade_date) for row in rows],
                            "statuses": [str(row.status) for row in rows],
                            "performance_certified": [
                                bool(row.performance_certified) for row in rows
                            ],
                        }
                    )
        if not offenders:
            return None
        return self.activate(
            reason=(
                f"{len(offenders)} 个模拟账户连续 {PERSISTENT_NAV_FAILURE_ROWS} 期 NAV "
                "degraded/未认证，账本健康持续异常"
            ),
            source="nav_health",
            actor="system",
            details={"offenders": offenders},
        )
