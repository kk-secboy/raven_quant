from __future__ import annotations

import hashlib
import json

from quant_data.config import Settings
from quant_data.provider import TushareHttpProvider
from quant_data.rate_limit import GlobalRateGate
from quant_platform.runtime_secret_store import RuntimeSecretStore


def fingerprint(rows: list[dict]) -> str:
    payload = json.dumps(rows[:5], ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def main() -> None:
    settings = Settings.from_env()
    stored = RuntimeSecretStore(settings.database_url, settings.platform_secret_key).get("tushare")
    if not stored:
        raise RuntimeError("Tushare runtime secret is unavailable")
    provider = TushareHttpProvider(
        api_url=stored["api_url"],
        token=stored["token"],
        rate_gate=GlobalRateGate(settings.requests_per_minute),
        timeout_seconds=settings.timeout_seconds,
        max_attempts=settings.max_request_attempts,
        cooldown_seconds=settings.cooldown_seconds,
    )
    cases = (
        (
            "stk_holdernumber",
            {"start_date": "20240101", "end_date": "20240131"},
        ),
        ("stk_surv", {"start_date": "20240101", "end_date": "20240131"}),
        ("top10_holders", {"start_date": "20240101", "end_date": "20240131"}),
        ("hk_hold", {"trade_date": "20240731"}),
        ("stk_factor_pro", {"trade_date": "20260710"}),
        ("opt_basic", {}),
        ("opt_daily", {"trade_date": "20260710"}),
        ("us_basic", {}),
        ("us_daily", {"trade_date": "20260710"}),
    )
    for api_name, base in cases:
        for offset in (0, 1000):
            result = provider.fetch(
                api_name,
                {**base, "limit": 1000, "offset": offset},
            )
            print(
                json.dumps(
                    {
                        "api": api_name,
                        "offset": offset,
                        "rows": len(result.rows),
                        "fingerprint": fingerprint(result.rows),
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
