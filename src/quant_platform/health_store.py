from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import requests
from sqlalchemy import func, insert, select

from quant_data.config import Settings
from quant_data.database import jobs, open_database, row_dict, system_health_snapshots

from .rdagent_scenarios import FROZEN_RDAGENT_SCENARIOS
from .runtime_secret_store import RuntimeSecretStore
from .safe_mode import SafeModeStore
from .services import list_qlib_datasets


def _now() -> datetime:
    return datetime.now(UTC)


def _qlib_calendar_date(value: object) -> date:
    raw = str(value).strip()
    try:
        return date.fromisoformat(raw)
    except ValueError:
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is not None:
            raise ValueError(
                "Qlib calendar timestamps must be exchange-local and timezone-naive"
            ) from None
        return parsed.date()


def safe_mode_recovery_health_status(
    snapshot: dict[str, Any] | None,
    *,
    triggered_at: datetime | str | None = None,
) -> str:
    """Evaluate recovery health without making safe mode block its own release.

    The persisted overall status is necessarily ``degraded`` while safe mode is
    active because ``safe_mode`` is itself a health component.  Recovery must
    therefore ignore only that component and still fail closed unless every
    other component is healthy (or explicitly not applicable).
    """

    if not isinstance(snapshot, dict):
        return "missing"
    if triggered_at is not None:
        try:
            observed_at = snapshot.get("recorded_at")
            observed = (
                observed_at
                if isinstance(observed_at, datetime)
                else datetime.fromisoformat(str(observed_at))
            )
            triggered = (
                triggered_at
                if isinstance(triggered_at, datetime)
                else datetime.fromisoformat(str(triggered_at))
            )
            if observed.tzinfo is None or triggered.tzinfo is None or observed <= triggered:
                return "degraded"
        except (TypeError, ValueError):
            return "missing"
    components = snapshot.get("components")
    if not isinstance(components, dict) or "safe_mode" not in components:
        return "missing"
    checked = 0
    for name, component in components.items():
        if name == "safe_mode":
            continue
        checked += 1
        if not isinstance(component, dict) or component.get("status") not in {
            "ok",
            "not_applicable",
        }:
            return "degraded"
    return "ok" if checked else "missing"


class OperationalHealthStore:
    """Durable component, queue, and market-data freshness observations."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine = open_database(settings.database_url)
        self.runtime_secrets = RuntimeSecretStore(
            settings.database_url, settings.platform_secret_key
        )

    def due(self, now: datetime) -> bool:
        with self.engine.connect() as connection:
            latest = connection.scalar(select(func.max(system_health_snapshots.c.recorded_at)))
        return (
            latest is None
            or (now - latest).total_seconds() >= self.settings.health_snapshot_seconds
        )

    def collect(self, now: datetime | None = None) -> dict[str, Any]:
        current = now or _now()
        components: dict[str, dict[str, Any]] = {
            "postgresql": {"status": "ok", "message": "control-plane connection ready"},
            "scheduler": {
                "status": "ok",
                "message": "health collection tick completed",
                "release_id": self.settings.quantlab_release_id or None,
                "config_digest": self.settings.quantlab_config_digest or None,
            },
        }
        components["qlib_worker"] = self._probe_service(
            self.settings.qlib_worker_url,
            "/health",
            required=not self.settings.embedded_worker,
        )
        components["rdagent_worker"] = self._probe_service(
            self.settings.rdagent_worker_url,
            "/health",
            required=self.settings.rdagent_enabled and not self.settings.embedded_worker,
        )
        components["rdagent_evaluation_worker"] = self._probe_service(
            self.settings.rdagent_evaluation_worker_url,
            "/health",
            required=self.settings.rdagent_enabled and not self.settings.embedded_worker,
        )
        if "data_science" in FROZEN_RDAGENT_SCENARIOS:
            components["rdagent_data_science_worker"] = {
                "status": "not_applicable",
                "message": "data_science is frozen; historical runs are read-only",
                "details": {"scenario": "data_science", "frozen": True},
            }
        else:
            components["rdagent_data_science_worker"] = self._probe_service(
                self.settings.rdagent_data_science_worker_url,
                "/health",
                required=self.settings.rdagent_enabled and not self.settings.embedded_worker,
            )
        if self.settings.rdagent_enabled and self.settings.rdagent_worker_url:
            components["rdagent_runtime"] = self._probe_runtime(
                f"{self.settings.rdagent_worker_url}/rdagent/status"
            )
        components["runtime_secret_storage"] = self.runtime_secrets.health()
        components["market_data"] = self._market_data_health(current.date())
        try:
            api_url, token = self._tushare_credentials()
            components["credentials"] = {
                "status": "ok" if api_url and token else "bootstrap_required",
                "message": (
                    "Tushare credentials configured"
                    if api_url and token
                    else "Tushare credentials are not configured"
                ),
            }
        except ValueError as exc:
            components["credentials"] = {
                "status": "unavailable",
                "message": f"Tushare credential storage failed: {exc}",
            }
        components["job_queue"] = self._job_queue_health(current)
        components["safe_mode"] = self._safe_mode_health()
        components["broker_boundary"] = {
            "status": "not_applicable",
            "message": "broker execution is outside the research and recommendation system",
            "details": {"enabled": False, "mode": "not_applicable"},
        }

        statuses = {item["status"] for item in components.values()}
        if statuses & {"degraded", "unavailable"}:
            status = "degraded"
        elif "bootstrap_required" in statuses:
            status = "bootstrap_required"
        elif "attention" in statuses:
            status = "attention"
        else:
            status = "ok"
        summary = {
            "component_count": len(components),
            "ok_count": sum(item["status"] == "ok" for item in components.values()),
            "problem_count": sum(
                item["status"] in {"degraded", "unavailable"} for item in components.values()
            ),
            "bootstrap_count": sum(
                item["status"] == "bootstrap_required" for item in components.values()
            ),
        }
        return {
            "status": status,
            "components": components,
            "summary": summary,
            "recorded_at": current,
        }

    def record(self, observation: dict[str, Any]) -> dict[str, Any]:
        with self.engine.begin() as connection:
            health_id = connection.execute(
                insert(system_health_snapshots)
                .values(
                    status=observation["status"],
                    components_json=observation["components"],
                    summary_json=observation["summary"],
                    recorded_at=observation["recorded_at"],
                )
                .returning(system_health_snapshots.c.id)
            ).scalar_one()
        return self.get(int(health_id))

    def _tushare_credentials(self) -> tuple[str, str]:
        stored = self.runtime_secrets.get("tushare")
        if stored is not None:
            return str(stored.get("api_url") or ""), str(stored.get("token") or "")
        return self.settings.api_url, self.settings.token

    def collect_and_record(self, now: datetime | None = None) -> dict[str, Any]:
        return self.record(self.collect(now))

    def get(self, health_id: int) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(system_health_snapshots).where(system_health_snapshots.c.id == health_id)
            ).first()
        if row is None:
            raise KeyError(health_id)
        return self._row(row)

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            return [
                self._row(row)
                for row in connection.execute(
                    select(system_health_snapshots)
                    .order_by(system_health_snapshots.c.recorded_at.desc())
                    .limit(limit)
                )
            ]

    def latest(self, *, max_age_seconds: int | None = None) -> dict[str, Any] | None:
        rows = self.list(1)
        if not rows:
            return None
        result = rows[0]
        recorded = datetime.fromisoformat(result["recorded_at"])
        age = max(0.0, (_now() - recorded).total_seconds())
        result["age_seconds"] = age
        stale_after = max_age_seconds or max(120, self.settings.health_snapshot_seconds * 2)
        if age > stale_after:
            result["status"] = "degraded"
            result["components"] = {
                **result["components"],
                "scheduler_heartbeat": {
                    "status": "degraded",
                    "message": f"last durable health snapshot is {int(age)} seconds old",
                },
            }
            result["summary"] = {
                **result["summary"],
                "problem_count": int(result["summary"].get("problem_count", 0)) + 1,
            }
        return result

    def _probe_service(self, base_url: str, path: str, *, required: bool) -> dict[str, Any]:
        if not base_url:
            return {
                "status": "bootstrap_required" if required else "ok",
                "message": "service URL is not configured" if required else "embedded mode",
            }
        try:
            response = requests.get(f"{base_url}{path}", timeout=5)
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as exc:
            return {"status": "unavailable", "message": str(exc)[:500]}
        return {"status": "ok", "message": "service reachable", "details": body}

    @staticmethod
    def _probe_runtime(url: str) -> dict[str, Any]:
        try:
            response = requests.get(url, timeout=8)
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as exc:
            return {"status": "unavailable", "message": str(exc)[:500]}
        if body.get("ready"):
            return {"status": "ok", "message": "RD-Agent runtime ready", "details": body}
        return {
            "status": "bootstrap_required",
            "message": "RD-Agent runtime requires configuration",
            "details": body,
        }

    def _market_data_health(self, today: date) -> dict[str, Any]:
        datasets = [item for item in list_qlib_datasets(self.settings.data_root) if item["ready"]]
        if not datasets:
            return {
                "status": "bootstrap_required",
                "message": "no ready Qlib dataset",
                "dataset_count": 0,
            }
        parsed: list[tuple[date, dict[str, Any]]] = []
        invalid: list[dict[str, str]] = []
        for dataset in datasets:
            try:
                parsed.append((_qlib_calendar_date(dataset.get("end_date")), dataset))
            except (TypeError, ValueError):
                invalid.append(
                    {
                        "dataset": str(dataset.get("name") or ""),
                        "end_date": str(dataset.get("end_date") or ""),
                    }
                )
        if invalid:
            return {
                "status": "degraded",
                "message": "ready Qlib dataset has an invalid calendar boundary",
                "dataset_count": len(datasets),
                "invalid_datasets": sorted(invalid, key=lambda item: item["dataset"]),
            }
        end_date, latest = max(
            parsed,
            key=lambda item: (item[0], str(item[1].get("end_date") or "")),
        )
        age_days = max(0, (today - end_date).days)
        status = "degraded" if age_days > self.settings.data_freshness_max_days else "ok"
        return {
            "status": status,
            "message": f"latest Qlib dataset ends {latest['end_date']}",
            "dataset": latest["name"],
            "dataset_count": len(datasets),
            "end_date": latest["end_date"],
            "age_days": age_days,
            "limit_days": self.settings.data_freshness_max_days,
        }

    def _safe_mode_health(self) -> dict[str, Any]:
        try:
            state = SafeModeStore(self.settings.database_url).status()
        except Exception as exc:  # noqa: BLE001 - health collection never crashes
            return {"status": "attention", "message": f"safe-mode state unreadable: {exc}"}
        if state["active"]:
            return {
                "status": "degraded",
                "message": (
                    f"safe_mode active since {state['triggered_at']} "
                    f"(source {state['source']}): {state['reason']}"
                ),
                "details": {
                    "source": state["source"],
                    "triggered_by": state["triggered_by"],
                    "triggered_at": str(state["triggered_at"]),
                    "recovery": "manual safe-mode release required (actor + reason)",
                },
            }
        return {
            "status": "ok",
            "message": "safe mode off; recommendations and simulation orders flow",
        }

    def _job_queue_health(self, now: datetime) -> dict[str, Any]:
        stale_before = now - timedelta(hours=self.settings.stale_job_hours)
        failed_after = now - timedelta(hours=24)
        with self.engine.connect() as connection:
            queued = connection.scalar(
                select(func.count()).select_from(jobs).where(jobs.c.status == "queued")
            )
            running = connection.scalar(
                select(func.count()).select_from(jobs).where(jobs.c.status == "running")
            )
            stale = connection.scalar(
                select(func.count())
                .select_from(jobs)
                .where(
                    jobs.c.status == "running",
                    jobs.c.started_at < stale_before,
                )
            )
            failed = connection.scalar(
                select(func.count())
                .select_from(jobs)
                .where(
                    jobs.c.status == "failed",
                    jobs.c.finished_at >= failed_after,
                )
            )
        # Recent terminal jobs remain visible here and are projected as their
        # own critical ``job_failure`` alerts.  They are immutable audit
        # history, not evidence that the queue itself is currently unhealthy.
        # Double-counting them as queue ``attention`` would also make a fully
        # evidenced recovery unable to leave safe mode for 24 hours.  Queue
        # health is therefore driven only by active operational risk.
        status = "degraded" if stale or int(queued or 0) > 100 else "ok"
        return {
            "status": status,
            "message": "durable worker queue inspected",
            "queued": int(queued or 0),
            "running": int(running or 0),
            "stale_running": int(stale or 0),
            "failed_24h": int(failed or 0),
            "stale_after_hours": self.settings.stale_job_hours,
        }

    @staticmethod
    def _row(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["components"] = result.pop("components_json")
        result["summary"] = result.pop("summary_json")
        return result
