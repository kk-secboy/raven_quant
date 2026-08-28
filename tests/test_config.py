from pathlib import Path

import pytest

from quant_data.config import TUSHARE_RELAY_MAX_REQUESTS_PER_MINUTE, Settings

pytestmark = pytest.mark.no_database


def test_tushare_relay_rate_is_capped_below_provider_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REQUESTS_PER_MINUTE", "180")

    settings = Settings.from_env(tmp_path / ".env.missing")

    assert settings.requests_per_minute == TUSHARE_RELAY_MAX_REQUESTS_PER_MINUTE == 99.0


def test_tushare_relay_rate_preserves_stricter_operator_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REQUESTS_PER_MINUTE", "60")

    settings = Settings.from_env(tmp_path / ".env.missing")

    assert settings.requests_per_minute == 60.0


@pytest.mark.parametrize(("configured", "expected"), [("0", 1), ("2", 2), ("9", 3)])
def test_worker_concurrency_is_bounded_for_shared_cpu_queues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    expected: int,
) -> None:
    monkeypatch.setenv("WORKER_CONCURRENCY", configured)

    settings = Settings.from_env(tmp_path / ".env.missing")

    assert settings.worker_concurrency == expected
