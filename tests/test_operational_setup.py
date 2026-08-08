from __future__ import annotations

from pathlib import Path

import pytest

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


@pytest.mark.no_database
def test_worker_accepts_every_supplemental_download_bundle() -> None:
    root = Path(__file__).resolve().parents[1]
    compose = (root / "deploy" / "compose.yaml").read_text(encoding="utf-8")
    bundles = {
        "cn_extended_daily",
        "cn_funds",
        "cn_macro",
        "cn_institutional",
        "cn_futures",
        "cn_options_bonds",
        "hk_market",
        "us_market",
        "global_markets",
        "cn_governance_risk",
        "cn_capital_flow",
        "cn_fund_index_enhanced",
        "cn_derivatives_enhanced",
        "global_rates_enhanced",
        "research_corpus",
        "strategy_specialty",
        "strategy_specialty_minutes",
    }

    for bundle in bundles:
        assert f"supplemental_{bundle}" in compose


@pytest.mark.no_database
def test_worker_accepts_governed_information_jobs() -> None:
    root = Path(__file__).resolve().parents[1]
    compose = (root / "deploy" / "compose.yaml").read_text(encoding="utf-8")

    for kind in (
        "announcement_nlp",
        "corpus_nlp",
        "event_market_response",
        "external_factor_evaluate",
    ):
        assert kind in compose
    assert "REQUESTS_PER_MINUTE: ${REQUESTS_PER_MINUTE:-99}" in compose


def test_factor_sandbox_is_seeded_offline_from_the_release_worker() -> None:
    root = Path(__file__).resolve().parents[1]
    compose = (root / "deploy" / "compose.yaml").read_text(encoding="utf-8")
    dockerfile = (root / "deploy" / "factor-sandbox" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    worker_dockerfile = (root / "deploy" / "Dockerfile.worker").read_text(
        encoding="utf-8"
    )
    builder = (root / "deploy" / "factor-sandbox" / "build.sh").read_text(
        encoding="utf-8"
    )
    attributes = (root / ".gitattributes").read_text(encoding="utf-8")

    assert "image: quantlab-worker-runtime:v2" in compose
    assert "*.sh text eol=lf" in attributes
    assert "/var/run/docker.sock:/var/run/docker.sock" in compose
    assert "FACTOR_SANDBOX_DOCKER_HOST: tcp://rdagent-docker:2375" in compose
    assert "FROM ${FACTOR_SANDBOX_BASE_IMAGE}" in dockerfile
    assert "FROM python:" not in dockerfile
    assert "FROM docker:27-dind AS docker_cli" in worker_dockerfile
    assert "COPY --from=docker_cli /usr/local/bin/docker" in worker_dockerfile
    assert 'docker save "$base_image"' in builder
    assert 'docker --host "$sandbox_host" load' in builder
    assert 'docker --host "$sandbox_host" build' in builder


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
