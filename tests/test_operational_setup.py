from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from quant_platform import release_upgrade
from quant_platform.deployment_services import BUILT_APPLICATION_SERVICES
from quant_platform.worker import _failure_message
from scripts.configure_tushare import update_env, validate_token

pytestmark = pytest.mark.no_database


def _load_release_upgrade_drill_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "release_upgrade_drill.py"
    scripts_path = str(script.parent)
    sys.path.insert(0, scripts_path)
    try:
        spec = importlib.util.spec_from_file_location("release_upgrade_drill_test", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts_path)


def test_failure_message_extracts_actionable_tail(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    log.write_text(
        "starting\ntraceback noise\nValueError: TUSHARE_TOKEN is required\n",
        encoding="utf-8",
    )
    assert _failure_message(log, "fallback") == "ValueError: TUSHARE_TOKEN is required"
    assert _failure_message(tmp_path / "missing.log", "fallback") == "fallback"


def test_release_upgrade_drill_uses_isolated_sibling_storage() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "release_upgrade_drill.py").read_text(encoding="utf-8")

    assert 'data_host_path = scratch / "drill-data"' in source
    assert 'backup_root = scratch / "backups"' in source
    assert 'docker_host_path = scratch / "rdagent-docker"' in source
    assert 'registry_host_path = scratch / "rdagent-registry"' in source
    assert 'f"QUANTLAB_DATA_HOST_PATH={data_host_path.resolve()}"' in source
    assert 'f"RDAGENT_DOCKER_HOST_PATH={docker_host_path.resolve()}"' in source
    assert 'f"RDAGENT_REGISTRY_HOST_PATH={registry_host_path.resolve()}"' in source
    assert "@control_plane_locked\n@isolated_drill_environment\ndef run_drill" in source
    assert '"OPENAI_API_KEY=drill-not-a-real-credential"' in source
    assert '"OPENAI_API_BASE=http://127.0.0.1:9"' in source
    assert 'f"RDAGENT_REGISTRY_PORT={registry_port}"' in source
    assert "rollback_tag_repository=f\"quantlab-upgrade-drill-rollback-{suffix}\"" in source
    assert "prune_rollback_images=False" in source
    assert 'env_file = backup_root / "drill.env"' in source
    assert '"sandbox-registry",\n                "down",' in source
    bootstrap = source.index(
        'result["bootstrap_sandbox_images"] = prepare_drill_sandbox_bootstrap('
    )
    full_stack = source.index('        baseline_context.run(\n            "up",', bootstrap)
    assert bootstrap < full_stack
    assert "WORKER_JOB_KINDS" not in source

    overlay = (root / "deploy" / "compose.restore-drill.yaml").read_text(
        encoding="utf-8"
    )
    assert 'root / "instruments/all.txt"' in overlay
    assert 'root / "instruments/cn_all.txt"' in overlay
    assert "files = (calendar, instruments, cn_instruments, feature)" in overlay

    release_source = (
        root / "src" / "quant_platform" / "release_upgrade.py"
    ).read_text(encoding="utf-8")
    assert "D.instruments('cn_all')" in release_source
    assert "@control_plane_locked\ndef prepare_drill_sandbox_bootstrap" in release_source
    assert 'context.docker("image", "rm", "-f", *host_published_tags, check=False)' in (
        release_source
    )


def test_release_upgrade_drill_uses_only_isolated_candidate_image_families(
    tmp_path: Path,
) -> None:
    module = _load_release_upgrade_drill_module()
    override = tmp_path / "candidate-runtime.compose.yaml"

    images = module._write_candidate_runtime_override(override, "deadbeef")
    rendered = override.read_text(encoding="utf-8")

    assert len(images) == 5
    assert all(image.endswith(":deadbeef") for image in images)
    assert all(image.startswith("quantlab-upgrade-drill-") for image in images)
    canonical_builders = {
        "api",
        "scheduler",
        "worker",
        "rdagent-worker",
        "web",
    }
    for service in BUILT_APPLICATION_SERVICES:
        match = re.search(rf"(?m)^  {re.escape(service)}:\n((?:    .*\n)*)", rendered)
        assert match is not None
        block = match.group(1)
        assert any(f'image: "{image}"' in block for image in images)
        if service in canonical_builders:
            assert "build:" not in block
        else:
            assert "build: !reset null" in block
    assert (
        "FACTOR_SANDBOX_BASE_IMAGE: "
        '"quantlab-upgrade-drill-worker-runtime:deadbeef"'
    ) in rendered
    assert "quantlab-upgrade-drill-api-runtime:deadbeef" in rendered
    assert "quantlab-upgrade-drill-scheduler-runtime:deadbeef" in rendered
    assert rendered.count("quantlab-upgrade-drill-worker-runtime:deadbeef") == 4
    assert rendered.count("quantlab-upgrade-drill-rdagent-runtime:deadbeef") == 4
    assert not any(
        image.startswith("quantlab-platform-")
        or image in {"quantlab-worker-runtime:v2", "quantlab-rdagent-runtime:v2"}
        for image in images
    )


def test_release_upgrade_drill_seals_sandboxes_before_full_worker_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []

    class Context:
        def run(self, *arguments: str) -> None:
            calls.append(("compose", *arguments))

    def prepare(context, project_root, release_id, *, wait_timeout):
        calls.append(
            (
                "seal",
                context,
                project_root,
                release_id,
                wait_timeout,
            )
        )
        return {"MODEL_SANDBOX_IMAGE": "registry/model@sha256:" + "a" * 64}

    monkeypatch.setattr(release_upgrade, "_prepare_sandbox_images", prepare)
    context = Context()

    result = release_upgrade.prepare_drill_sandbox_bootstrap(
        context,  # type: ignore[arg-type]
        tmp_path,
        "20260829T120000Z",
        wait_timeout=240,
    )

    assert calls == [
        (
            "compose",
            "up",
            "-d",
            "--no-build",
            "--wait",
            "--wait-timeout",
            "240",
            "rdagent-docker",
        ),
        (
            "seal",
            context,
            tmp_path,
            "drill-bootstrap-20260829T120000Z",
            240,
        ),
    ]
    assert result["MODEL_SANDBOX_IMAGE"].startswith("registry/model@sha256:")


def test_deployment_docs_require_mixed_release_convergence_before_upgrade() -> None:
    root = Path(__file__).resolve().parents[1]
    deployment = (root / "docs" / "DEPLOYMENT.md").read_text(encoding="utf-8")

    assert "scripts/canonicalize_release_baseline.py" in deployment
    assert "--confirm-convergence" in deployment
    assert "--stable-release-link /opt/quantlab" in deployment
    assert "docker compose down -v" in deployment
    for forbidden_flag in ("--pull", "--reuse-backup", "--skip-stable-link"):
        assert forbidden_flag in deployment


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
    assert compose.count("logging: *default-logging") == 13
    assert compose.count(
        "${PLATFORM_SECRET_KEY:?PLATFORM_SECRET_KEY is required}"
    ) == 9


def test_paper_lifecycle_has_a_reserved_worker_lane() -> None:
    root = Path(__file__).resolve().parents[1]
    compose = (root / "deploy" / "compose.yaml").read_text(encoding="utf-8")
    services_tail = compose.split("\n  evaluation-worker:\n", 1)[1]
    evaluation, paper_tail = services_tail.split("\n  paper-worker:\n", 1)
    paper = paper_tail.split("\n  rdagent-docker:\n", 1)[0]

    for kind in (
        "model_refit",
        "recommendation_refresh",
        "simulation_order_plan",
        "simulation_replay",
    ):
        assert kind not in evaluation
        assert kind in paper
    assert 'WORKER_CONCURRENCY: "1"' in paper


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
        "announcement_factor_register",
        "corpus_nlp",
        "corpus_factor_register",
        "event_market_response",
        "external_factor_evaluate",
        "information_factor_evaluate",
        "report_rc_factors",
        "report_rc_factor_register",
        "major_news_mentions",
        "major_news_mentions_factor_register",
        "news_flash_factors",
        "news_flash_factor_register",
        "multiface_audit",
    ):
        assert kind in compose
    assert "REQUESTS_PER_MINUTE: ${REQUESTS_PER_MINUTE:-99}" in compose


def test_worker_accepts_automatic_factor_library_jobs() -> None:
    root = Path(__file__).resolve().parents[1]
    compose = (root / "deploy" / "compose.yaml").read_text(encoding="utf-8")

    for kind in (
        "factor_library_materialize",
        "factor_library_cluster",
        "factor_sota_evaluate",
    ):
        assert kind in compose


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
    service = (root / "deploy" / "systemd" / "quantlab-backup.service").read_text(
        encoding="utf-8"
    )
    timer = (root / "deploy" / "systemd" / "quantlab-backup.timer").read_text(
        encoding="utf-8"
    )
    backup_script = (root / "scripts" / "backup.py").read_text(encoding="utf-8")

    assert "Description=QuantLab online bounded control-plane backup" in service
    assert service.index("ExecStartPre=") < service.index("ExecStart=")
    assert "/opt/quantlab-ops/venv/bin/python" in service
    assert "scripts/backup_preflight.py" in service
    assert "scripts/release_preflight.py" not in service
    assert "scripts/backup.py" in service
    assert "--retention-count 14" in service
    assert "--format-version 2" in service
    assert "--online" in service
    assert "--minimum-free-gb 10" in service
    assert '"--online"' in backup_script
    assert "online=args.online" in backup_script
    assert "OnCalendar=*-*-* 03:20:00 Asia/Shanghai" in timer
    assert "Persistent=true" in timer
    assert "RandomizedDelaySec=10m" in timer


def test_backup_service_has_a_versioned_host_ops_installer() -> None:
    root = Path(__file__).resolve().parents[1]
    installer = (root / "scripts" / "install_backup_service.sh").read_text(
        encoding="utf-8"
    )
    requirements = (
        root / "deploy" / "backup-ops-requirements.txt"
    ).read_text(encoding="utf-8")

    assert 'ops_root=${QUANTLAB_OPS_ROOT:-/opt/quantlab-ops}' in installer
    assert '"$python_bin" -m venv "$ops_root/venv"' in installer
    assert "backup-ops-requirements.txt" in installer
    assert "systemctl enable --now quantlab-backup.timer" in installer
    assert "cryptography==" in requirements
    assert "python-dotenv==" in requirements
