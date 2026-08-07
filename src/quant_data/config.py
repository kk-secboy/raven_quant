from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

# The relay operator confirmed a 120-start window. Keep one request of headroom
# and clamp operator overrides so a stale environment cannot silently trade
# useful throughput for upstream throttling and retries.
TUSHARE_RELAY_MAX_REQUESTS_PER_MINUTE = 119.0


def normalize_api_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        return value
    parsed = urlparse(value)
    if parsed.path in ("", "/") and parsed.netloc != "api.tushare.pro":
        return value + "/api/v1/query"
    return value


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    api_url: str
    token: str
    data_root: Path
    database_url: str = "postgresql+psycopg://quantlab:quantlab@127.0.0.1:55432/quantlab"
    mlflow_tracking_uri: str = (
        "postgresql+psycopg://quantlab:quantlab@127.0.0.1:55432/quantlab"
        "?options=-csearch_path%3Dpublic"
    )
    requests_per_minute: float = TUSHARE_RELAY_MAX_REQUESTS_PER_MINUTE
    workers: int = 4
    timeout_seconds: float = 60.0
    max_request_attempts: int = 5
    cooldown_seconds: float = 180.0
    keep_raw: bool = True
    qlib_repo: Path = Path("E:/projects/qlib")
    qlib_python: str = "/mnt/e/venvs/qlib/bin/python"
    qlib_wsl_distro: str = "Ubuntu-22.04"
    qlib_worker_url: str = ""
    embedded_worker: bool = True
    rdagent_repo: Path = Path("E:/projects/RD-Agent")
    rdagent_python: str = "/mnt/e/venvs/rdagent/bin/python"
    rdagent_command: str = "/mnt/e/venvs/rdagent/bin/rdagent"
    rdagent_wsl_distro: str = "Ubuntu-22.04"
    rdagent_worker_url: str = ""
    rdagent_enabled: bool = True
    rdagent_llm_key_env: str = "OPENAI_API_KEY"
    rdagent_max_loops: int = 3
    rdagent_max_duration: str = "2h"
    worker_job_kinds: tuple[str, ...] = ()
    scheduler_poll_seconds: float = 15.0
    health_snapshot_seconds: int = 300
    data_freshness_max_days: int = 7
    stale_job_hours: int = 6
    broker_feature_enabled: bool = False
    alert_webhook_url: str = ""
    scheduler_url: str = ""
    auth_mode: str = "disabled"
    auth_session_hours: int = 12
    auth_cookie_secure: bool = False
    platform_secret_key: str = ""
    auth_allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    )

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        if env_file is None:
            env_file = Path.cwd() / ".env"
        load_dotenv(env_file, override=False)
        api_url = os.getenv("TUSHARE_API_URL") or os.getenv("TUSHARE_BASE_URL", "")
        token = os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_API_KEY", "")
        auth_mode = os.getenv("AUTH_MODE", "disabled").strip().lower()
        if auth_mode not in {"disabled", "required"}:
            raise ValueError("AUTH_MODE must be disabled or required")
        return cls(
            api_url=normalize_api_url(api_url),
            token=token.strip(),
            data_root=Path(os.getenv("DATA_ROOT", "data")).expanduser().resolve(),
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql+psycopg://quantlab:quantlab@127.0.0.1:55432/quantlab",
            ),
            mlflow_tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI",
                "postgresql+psycopg://quantlab:quantlab@127.0.0.1:55432/quantlab"
                "?options=-csearch_path%3Dpublic",
            ),
            requests_per_minute=min(
                float(
                    os.getenv(
                        "REQUESTS_PER_MINUTE",
                        str(TUSHARE_RELAY_MAX_REQUESTS_PER_MINUTE),
                    )
                ),
                TUSHARE_RELAY_MAX_REQUESTS_PER_MINUTE,
            ),
            workers=max(1, int(os.getenv("DOWNLOAD_WORKERS", "4"))),
            timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60")),
            max_request_attempts=max(1, int(os.getenv("MAX_REQUEST_ATTEMPTS", "5"))),
            cooldown_seconds=float(os.getenv("RATE_LIMIT_COOLDOWN_SECONDS", "180")),
            keep_raw=_bool("KEEP_RAW_RESPONSES", True),
            qlib_repo=Path(os.getenv("QLIB_REPO", "E:/projects/qlib")).expanduser().resolve(),
            qlib_python=os.getenv("QLIB_PYTHON", "/mnt/e/venvs/qlib/bin/python"),
            qlib_wsl_distro=os.getenv("QLIB_WSL_DISTRO", "Ubuntu-22.04"),
            qlib_worker_url=os.getenv("QLIB_WORKER_URL", "").strip().rstrip("/"),
            embedded_worker=_bool("RUN_EMBEDDED_WORKER", True),
            rdagent_repo=Path(os.getenv("RDAGENT_REPO", "E:/projects/RD-Agent"))
            .expanduser()
            .resolve(),
            rdagent_python=os.getenv("RDAGENT_PYTHON", "/mnt/e/venvs/rdagent/bin/python"),
            rdagent_command=os.getenv("RDAGENT_COMMAND", "/mnt/e/venvs/rdagent/bin/rdagent"),
            rdagent_wsl_distro=os.getenv("RDAGENT_WSL_DISTRO", "Ubuntu-22.04"),
            rdagent_worker_url=os.getenv("RDAGENT_WORKER_URL", "").strip().rstrip("/"),
            rdagent_enabled=_bool("RDAGENT_ENABLED", True),
            rdagent_llm_key_env=os.getenv("RDAGENT_LLM_KEY_ENV", "OPENAI_API_KEY").strip(),
            rdagent_max_loops=max(1, int(os.getenv("RDAGENT_MAX_LOOPS", "3"))),
            rdagent_max_duration=os.getenv("RDAGENT_MAX_DURATION", "2h").strip(),
            worker_job_kinds=tuple(
                item.strip()
                for item in os.getenv("WORKER_JOB_KINDS", "").split(",")
                if item.strip()
            ),
            scheduler_poll_seconds=max(1.0, float(os.getenv("SCHEDULER_POLL_SECONDS", "15"))),
            health_snapshot_seconds=max(60, int(os.getenv("HEALTH_SNAPSHOT_SECONDS", "300"))),
            data_freshness_max_days=max(1, int(os.getenv("DATA_FRESHNESS_MAX_DAYS", "7"))),
            stale_job_hours=max(1, int(os.getenv("STALE_JOB_HOURS", "6"))),
            broker_feature_enabled=_bool("BROKER_FEATURE_ENABLED", False),
            alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL", "").strip(),
            scheduler_url=os.getenv("SCHEDULER_URL", "").strip().rstrip("/"),
            auth_mode=auth_mode,
            auth_session_hours=max(1, min(168, int(os.getenv("AUTH_SESSION_HOURS", "12")))),
            auth_cookie_secure=_bool("AUTH_COOKIE_SECURE", False),
            platform_secret_key=os.getenv("PLATFORM_SECRET_KEY", "").strip(),
            auth_allowed_origins=tuple(
                item.strip()
                for item in os.getenv(
                    "AUTH_ALLOWED_ORIGINS",
                    "http://127.0.0.1:3000,http://localhost:3000",
                ).split(",")
                if item.strip()
            ),
        )

    def require_credentials(self) -> None:
        if not self.api_url:
            raise ValueError("TUSHARE_API_URL is required")
        if not self.token:
            raise ValueError("TUSHARE_TOKEN is required")
