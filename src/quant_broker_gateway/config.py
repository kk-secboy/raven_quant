from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    database_url: str
    hmac_secret: str
    qmt_mini_path: Path
    qmt_account_id: str
    account_ref: str
    qmt_session_id: int
    environment: str = "sandbox"
    provider: str = "qmt"
    bind_host: str = "0.0.0.0"
    bind_port: int = 8790
    max_clock_skew_seconds: int = 30
    poll_seconds: float = 1.0
    max_slice_lateness_seconds: int = 90
    cancel_after_seconds: int = 60
    max_replacements: int = 1
    max_reprice_bps: float = 20.0
    volume_multiplier: int = 100
    max_quote_age_seconds: int = 10

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> GatewaySettings:
        if env_file is not None:
            load_dotenv(env_file, override=False)
        settings = cls(
            database_url=os.getenv("DATABASE_URL", "").strip(),
            hmac_secret=os.getenv("BROKER_HMAC_SECRET", "").strip(),
            qmt_mini_path=Path(os.getenv("QMT_MINI_PATH", "")).expanduser(),
            qmt_account_id=os.getenv("QMT_ACCOUNT_ID", "").strip(),
            account_ref=os.getenv("QMT_ACCOUNT_REF", "").strip(),
            qmt_session_id=int(os.getenv("QMT_SESSION_ID", "178901")),
            environment=os.getenv("QMT_ENVIRONMENT", "sandbox").strip().lower(),
            provider=os.getenv("BROKER_PROVIDER", "qmt").strip().lower(),
            bind_host=os.getenv("BROKER_GATEWAY_HOST", "0.0.0.0").strip(),
            bind_port=int(os.getenv("BROKER_GATEWAY_PORT", "8790")),
            max_clock_skew_seconds=int(os.getenv("BROKER_MAX_CLOCK_SKEW_SECONDS", "30")),
            poll_seconds=float(os.getenv("BROKER_GATEWAY_POLL_SECONDS", "1")),
            max_slice_lateness_seconds=int(os.getenv("BROKER_MAX_SLICE_LATENESS_SECONDS", "90")),
            cancel_after_seconds=int(os.getenv("QMT_CANCEL_AFTER_SECONDS", "60")),
            max_replacements=int(os.getenv("QMT_MAX_REPLACEMENTS", "1")),
            max_reprice_bps=float(os.getenv("QMT_MAX_REPRICE_BPS", "20")),
            volume_multiplier=int(os.getenv("QMT_MINUTE_VOLUME_MULTIPLIER", "100")),
            max_quote_age_seconds=int(os.getenv("QMT_MAX_QUOTE_AGE_SECONDS", "10")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.database_url.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError("DATABASE_URL must point to PostgreSQL")
        if len(self.hmac_secret) < 32:
            raise ValueError("BROKER_HMAC_SECRET must contain at least 32 characters")
        if self.environment != "sandbox":
            raise ValueError("QMT_ENVIRONMENT must be sandbox; live trading is unsupported")
        if self.provider != "qmt":
            raise ValueError("BROKER_PROVIDER must be qmt")
        if not self.qmt_mini_path.is_absolute():
            raise ValueError("QMT_MINI_PATH must be an absolute MiniQMT user-data path")
        if not self.qmt_account_id:
            raise ValueError("QMT_ACCOUNT_ID is required")
        if not self.account_ref:
            raise ValueError("QMT_ACCOUNT_REF is required")
        if not 1 <= self.qmt_session_id <= 2_147_483_647:
            raise ValueError("QMT_SESSION_ID must be a positive 32-bit integer")
        if not 1 <= self.bind_port <= 65535:
            raise ValueError("BROKER_GATEWAY_PORT is invalid")
        if not 5 <= self.max_clock_skew_seconds <= 300:
            raise ValueError("BROKER_MAX_CLOCK_SKEW_SECONDS must be between 5 and 300")
        if not 0.2 <= self.poll_seconds <= 30:
            raise ValueError("BROKER_GATEWAY_POLL_SECONDS must be between 0.2 and 30")
        if not 5 <= self.max_slice_lateness_seconds <= 600:
            raise ValueError("BROKER_MAX_SLICE_LATENESS_SECONDS must be between 5 and 600")
        if not 10 <= self.cancel_after_seconds <= 600:
            raise ValueError("QMT_CANCEL_AFTER_SECONDS must be between 10 and 600")
        if not 0 <= self.max_replacements <= 3:
            raise ValueError("QMT_MAX_REPLACEMENTS must be between 0 and 3")
        if not 0 <= self.max_reprice_bps <= 100:
            raise ValueError("QMT_MAX_REPRICE_BPS must be between 0 and 100")
        if self.volume_multiplier not in {1, 100}:
            raise ValueError("QMT_MINUTE_VOLUME_MULTIPLIER must be 1 or 100")
        if not 1 <= self.max_quote_age_seconds <= 60:
            raise ValueError("QMT_MAX_QUOTE_AGE_SECONDS must be between 1 and 60")
