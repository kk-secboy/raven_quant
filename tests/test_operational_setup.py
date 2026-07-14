from __future__ import annotations

from pathlib import Path

from quant_platform.worker import _failure_message
from scripts.configure_tushare import update_env, validate_token


def test_failure_message_extracts_actionable_tail(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    log.write_text(
        "starting\ntraceback noise\nValueError: TUSHARE_TOKEN is required\n",
        encoding="utf-8",
    )
    assert _failure_message(log, "fallback") == "ValueError: TUSHARE_TOKEN is required"
    assert _failure_message(tmp_path / "missing.log", "fallback") == "fallback"


def test_tushare_configuration_is_validated_and_written_atomically(
    tmp_path: Path, monkeypatch
) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"code": 0, "msg": None, "data": {"fields": [], "items": []}}

    calls = []

    def post(url: str, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("scripts.configure_tushare.requests.post", post)
    validate_token("https://api.tushare.pro", "secret-token", 5)
    assert calls[0][1]["json"]["api_name"] == "trade_cal"
    env_file = tmp_path / "deploy.env"
    env_file.write_text("POSTGRES_PASSWORD=x\nTUSHARE_TOKEN=old\n", encoding="utf-8")
    update_env(env_file, "https://api.tushare.pro", "secret-token")
    content = env_file.read_text(encoding="utf-8")
    assert "POSTGRES_PASSWORD=x" in content
    assert "TUSHARE_API_URL=https://api.tushare.pro" in content
    assert "TUSHARE_TOKEN=secret-token" in content
    assert "TUSHARE_TOKEN=old" not in content


def test_compose_bounds_every_service_log_file() -> None:
    root = Path(__file__).resolve().parents[1]
    compose = (root / "deploy" / "compose.yaml").read_text(encoding="utf-8")

    assert "x-logging: &default-logging" in compose
    assert "max-size: ${LOG_MAX_SIZE:-20m}" in compose
    assert "max-file: ${LOG_MAX_FILES:-5}" in compose
    assert compose.count("logging: *default-logging") == 8
    assert compose.count(
        "${PLATFORM_SECRET_KEY:?PLATFORM_SECRET_KEY is required}"
    ) == 4


def test_systemd_backup_timer_is_persistent_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    service = (root / "deploy" / "systemd" / "quantlab-backup.service").read_text(encoding="utf-8")
    timer = (root / "deploy" / "systemd" / "quantlab-backup.timer").read_text(encoding="utf-8")

    assert service.index("ExecStartPre=") < service.index("ExecStart=")
    assert "scripts/release_preflight.py" in service
    assert "scripts/backup.py" in service
    assert "--retention-count 14" in service
    assert "OnCalendar=*-*-* 03:20:00 Asia/Shanghai" in timer
    assert "Persistent=true" in timer
    assert "RandomizedDelaySec=10m" in timer
