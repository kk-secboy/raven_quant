from __future__ import annotations

import argparse
import base64
import json
import secrets
import tempfile
import urllib.request
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _project import PROJECT_ROOT
from cryptography.fernet import Fernet

from quant_platform.backup_restore import (
    ComposeContext,
    compose_context,
    create_backup,
    load_and_verify_manifest,
    restore_backup,
)

EXPECTED_SERVICES = {
    "postgres",
    "api",
    "scheduler",
    "worker",
    "rdagent-docker",
    "rdagent-worker",
    "web",
    "gateway",
}
HEALTHCHECK_SERVICES = {
    "postgres",
    "api",
    "scheduler",
    "worker",
    "rdagent-docker",
    "rdagent-worker",
}


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_env(path: Path, password: str, secret_key: str) -> None:
    path.write_text(
        "\n".join(
            (
                f"POSTGRES_PASSWORD={password}",
                "POSTGRES_BIND_ADDRESS=127.0.0.1",
                "POSTGRES_PORT=0",
                "HTTP_BIND_ADDRESS=127.0.0.1",
                "HTTP_PORT=0",
                "AUTH_MODE=required",
                "AUTH_COOKIE_SECURE=false",
                f"PLATFORM_SECRET_KEY={secret_key}",
                "BROKER_MODE=disabled",
                "RDAGENT_ENABLED=true",
                "REQUESTS_PER_MINUTE=90",
                "DOWNLOAD_WORKERS=2",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _parse_compose_ps(raw: str) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stripped = raw.strip()
    if not stripped:
        return {}
    try:
        decoded = json.loads(stripped)
        rows = decoded if isinstance(decoded, list) else [decoded]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in stripped.splitlines() if line.strip()]
    return {str(item.get("Service")): item for item in rows}


def _assert_full_stack(context: ComposeContext) -> dict[str, dict[str, Any]]:
    context.run("up", "-d", "--no-build", "--wait", "--wait-timeout", "240")
    services = _parse_compose_ps(context.run("ps", "--format", "json", capture=True))
    missing = EXPECTED_SERVICES - set(services)
    if missing:
        raise RuntimeError(f"restored stack is missing services: {sorted(missing)}")
    for name in EXPECTED_SERVICES:
        if services[name].get("State") != "running":
            raise RuntimeError(f"restored service {name} is not running")
    for name in HEALTHCHECK_SERVICES:
        if services[name].get("Health") != "healthy":
            raise RuntimeError(f"restored service {name} is not healthy")
    return services


def _gateway_url(context: ComposeContext) -> str:
    gateway_id = context.container_id("gateway")
    binding = context.docker("port", gateway_id, "8080/tcp", capture=True).splitlines()[0]
    port = binding.rsplit(":", 1)[1]
    return f"http://127.0.0.1:{port}/"


def run_drill(project_root: Path, report_path: Path) -> dict[str, Any]:
    compose_file = project_root / "deploy" / "compose.yaml"
    override_file = project_root / "deploy" / "compose.restore-drill.yaml"
    suffix = secrets.token_hex(4)
    source_name = f"quantlab-drill-source-{suffix}"
    target_name = f"quantlab-drill-target-{suffix}"
    sentinel = secrets.token_hex(16)
    result: dict[str, Any] = {
        "status": "failed",
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_project": source_name,
        "target_project": target_name,
        "live_trading_supported": False,
        "checks": {},
    }
    source: ComposeContext | None = None
    target: ComposeContext | None = None
    temporary_manager: tempfile.TemporaryDirectory[str] | None = None
    try:
        for image in (
            "quantlab-platform-api:latest",
            "quantlab-platform-scheduler:latest",
            "quantlab-platform-worker:latest",
            "quantlab-platform-rdagent-worker:latest",
            "quantlab-platform-web:latest",
        ):
            subprocess_result = ComposeContext("image-check", Path(), ()).docker(
                "image", "inspect", image, capture=True, check=False
            )
            if not subprocess_result:
                raise RuntimeError(f"required restore-drill image is missing: {image}")

        temporary_manager = tempfile.TemporaryDirectory(prefix="quantlab-restore-drill-")
        with nullcontext(temporary_manager.name) as temporary:
            scratch = Path(temporary)
            source_env = scratch / "source.env"
            target_env = scratch / "target.env"
            backup_root = scratch / "backups"
            password = secrets.token_urlsafe(32)
            secret_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")
            runtime_secret_ciphertext = (
                Fernet(secret_key.encode("ascii"))
                .encrypt(json.dumps({"sentinel": sentinel}, separators=(",", ":")).encode("utf-8"))
                .decode("ascii")
            )
            _write_env(source_env, password, secret_key)
            _write_env(target_env, password, secret_key)
            source = compose_context(source_name, source_env, compose_file, (override_file,))
            target = compose_context(target_name, target_env, compose_file, (override_file,))

            source.run(
                "up",
                "-d",
                "--no-build",
                "--wait",
                "--wait-timeout",
                "120",
                "postgres",
                "api",
            )
            source.run(
                "exec",
                "-T",
                "postgres",
                "psql",
                "-U",
                "quantlab",
                "-d",
                "quantlab",
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                (
                    "INSERT INTO quantlab.audit_events "
                    "(username, action, method, path, status_code, details_json, created_at) "
                    "VALUES ('restore-drill', 'restore.drill.sentinel', 'POST', "
                    f"'/internal/restore-drill', 200, '{{\"sentinel\":\"{sentinel}\"}}'::jsonb, "
                    "now()); "
                    "INSERT INTO quantlab.strategy_allocations "
                    "(id, name, dataset, status, allocation_method, lookback_days, "
                    "target_volatility, max_pairwise_correlation, max_strategy_weight, "
                    "max_member_drawdown, max_drawdown_reduce, max_drawdown_liquidate, "
                    "total_capital, cash_reserve, nav, high_water_mark, analysis_json, "
                    "created_by, created_at, updated_at) VALUES ("
                    f"'restore-{sentinel}', 'Restore drill allocation {sentinel}', "
                    "'restore-drill', 'draft', 'fixed', 60, 0.15, 0.70, 0.70, "
                    "0.08, 0.10, 0.15, 500000, 0, 500000, 500000, "
                    f'\'{{"sentinel":"{sentinel}"}}\'::jsonb, '
                    "'restore-drill', now(), now()); "
                    "INSERT INTO quantlab.strategy_allocation_events "
                    "(allocation_id, severity, event_type, rule, status, details_json, "
                    "created_at, acknowledged_by, acknowledged_at, resolved_by, "
                    "resolved_at, resolution_reason) VALUES ("
                    f"'restore-{sentinel}', 'critical', 'restore_drill', "
                    "'risk_lifecycle_restore', 'resolved', "
                    f'\'{{"sentinel":"{sentinel}"}}\'::jsonb, now(), '
                    f"'ack-{sentinel}', now(), 'resolve-{sentinel}', now(), "
                    f"'resolution-{sentinel}');"
                    "INSERT INTO quantlab.schedules "
                    "(id, name, kind, status, desired_status, suspension_reason, "
                    "timezone, run_time, trading_days_only, payload_json, "
                    "misfire_grace_seconds, next_run_at, last_run_at, created_by, "
                    "created_at, updated_at) VALUES ("
                    f"'schedule-{sentinel}', 'Restore schedule {sentinel}', "
                    "'incremental_sync', 'paused', 'active', 'portfolio:paused', "
                    "'Asia/Shanghai', '18:00', true, "
                    f'\'{{"sentinel":"{sentinel}"}}\'::jsonb, 1800, now(), null, '
                    "'restore-drill', now(), now());"
                    "INSERT INTO quantlab.runtime_secrets "
                    "(name, ciphertext, metadata_json, updated_at, updated_by) VALUES ("
                    f"'restore_drill', '{runtime_secret_ciphertext}', "
                    f'\'{{"sentinel":"{sentinel}"}}\'::jsonb, now(), null);'
                    "INSERT INTO quantlab.strategies "
                    "(id, name, description, status, created_by, created_at, updated_at) VALUES ("
                    f"'restore-pair-strategy-{sentinel}', 'Restore pair strategy {sentinel}', "
                    "'restore drill pair strategy', 'active', 'restore-drill', now(), now());"
                    "INSERT INTO quantlab.strategy_versions "
                    "(id, strategy_id, version, status, strategy_type, benchmark, universe, "
                    "config_json, created_by, approved_by, approval_reason, "
                    "created_at, approved_at) "
                    "VALUES ("
                    f"'restore-pair-version-{sentinel}', 'restore-pair-strategy-{sentinel}', "
                    "1, 'approved', 'pair', 'SH000300', 'cn_all', '{}'::jsonb, "
                    "'restore-drill', 'restore-reviewer', 'restore drill approval', now(), now());"
                    "INSERT INTO quantlab.strategy_pairs "
                    "(strategy_version_id, leg_y, leg_x, asset_class, shorting_mode, created_at) "
                    f"VALUES ('restore-pair-version-{sentinel}', 'SH510300', 'SZ159919', "
                    "'etf', 'margin_borrow', now());"
                    "INSERT INTO quantlab.pair_paper_portfolios "
                    "(id, name, strategy_version_id, dataset, execution_snapshot, minute_dataset, "
                    "shortability_dataset, status, base_currency, initial_cash, cash, nav, "
                    "high_water_mark, position_direction, quantity_y, quantity_x, entry_nav, "
                    "holding_days, last_signal_date, last_trade_date, created_by, "
                    "created_at, updated_at) "
                    "VALUES ("
                    f"'restore-pair-{sentinel}', 'Restore pair ledger {sentinel}', "
                    f"'restore-pair-version-{sentinel}', 'restore-daily', 'restore-execution', "
                    "'restore-minute', 'restore-shortability', 'active', 'CNY', 5000000, "
                    "4999990, 5000010, 5000010, 1, 1000, -800, 4999990, 1, current_date - 1, "
                    "current_date, 'restore-drill', now(), now());"
                    "INSERT INTO quantlab.pair_portfolio_batches "
                    "(id, portfolio_id, as_of_date, trade_date, status, idempotency_key, "
                    "starting_state_sha256, artifact_path, created_at, started_at, "
                    "finished_at) VALUES ("
                    f"'restore-pair-batch-{sentinel}', 'restore-pair-{sentinel}', "
                    "current_date - 1, "
                    f"current_date, 'succeeded', 'restore-pair-batch:{sentinel}', "
                    f"'{sentinel}{sentinel}', '/data/restore-drill', now(), now(), now());"
                    "INSERT INTO quantlab.pair_portfolio_nav "
                    "(portfolio_id, trade_date, cash, long_value, short_value, nav, daily_return, "
                    "drawdown, gross_exposure, net_exposure, turnover, fees, borrow_cost, zscore, "
                    "correlation, cointegration_pvalue, position_direction, quantity_y, "
                    "quantity_x, "
                    "price_y, price_x, created_at) VALUES ("
                    f"'restore-pair-{sentinel}', current_date, 4999990, 4000, 4000, 5000010, "
                    "0.000002, 0, 0.0016, 0, 0.0016, 10, 1.27, -1.7, 0.92, 0.01, 1, "
                    "1000, -800, 4, 5, now());"
                    "INSERT INTO quantlab.pair_portfolio_risk_events "
                    "(portfolio_id, batch_id, severity, event_type, rule, observed, limit_value, "
                    "status, details_json, created_at, acknowledged_by, acknowledged_at, "
                    "resolved_by, resolved_at, resolution_reason) VALUES ("
                    f"'restore-pair-{sentinel}', 'restore-pair-batch-{sentinel}', 'critical', "
                    "'restore_drill', 'pair_risk_restore', -0.2, -0.15, 'resolved', "
                    f"'{{\"sentinel\":\"{sentinel}\"}}'::jsonb, now(), 'pair-ack', now(), "
                    "'pair-resolver', now(), 'pair restore drill resolution');"
                    "INSERT INTO quantlab.pair_portfolio_reviews "
                    "(id, portfolio_id, batch_id, trade_date, status, summary_json, "
                    "created_at) VALUES ("
                    f"'restore-pair-review-{sentinel}', 'restore-pair-{sentinel}', "
                    f"'restore-pair-batch-{sentinel}', current_date, 'completed', "
                    f'\'{{"sentinel":"{sentinel}","action":"entry"}}\'::jsonb, now());'
                ),
            )
            source.run(
                "exec",
                "-T",
                "api",
                "python",
                "-c",
                (
                    "from pathlib import Path; "
                    "p=Path('/data/restore-drill/sentinel.txt'); "
                    f"p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{sentinel}')"
                ),
            )
            backup_directory = create_backup(source, backup_root, retention_count=1)
            manifest = load_and_verify_manifest(backup_directory)
            result["checks"]["backup_manifest"] = {
                "status": "passed",
                "schema_revision": manifest["schema_revision"],
                "database_bytes": manifest["database"]["bytes"],
                "data_bytes": manifest["data_volume"]["bytes"],
                "platform_secret_key_fingerprint": (
                    "present" if manifest.get("platform_secret_key_sha256") else "missing"
                ),
            }

            target.run(
                "up",
                "-d",
                "--no-build",
                "--wait",
                "--wait-timeout",
                "120",
                "postgres",
            )
            restored_revision = restore_backup(target, backup_directory, confirmed=True)
            database_sentinel = target.run(
                "exec",
                "-T",
                "postgres",
                "psql",
                "-U",
                "quantlab",
                "-d",
                "quantlab",
                "-Atc",
                (
                    "SELECT details_json->>'sentinel' FROM quantlab.audit_events "
                    "WHERE action='restore.drill.sentinel' ORDER BY id DESC LIMIT 1;"
                ),
                capture=True,
            ).strip()
            if database_sentinel != sentinel:
                raise RuntimeError("database sentinel did not survive restore")
            allocation_sentinel = target.run(
                "exec",
                "-T",
                "postgres",
                "psql",
                "-U",
                "quantlab",
                "-d",
                "quantlab",
                "-Atc",
                (
                    "SELECT analysis_json->>'sentinel' "
                    "FROM quantlab.strategy_allocations "
                    f"WHERE id='restore-{sentinel}';"
                ),
                capture=True,
            ).strip()
            if allocation_sentinel != sentinel:
                raise RuntimeError("strategy allocation sentinel did not survive restore")
            lifecycle_sentinel = target.run(
                "exec",
                "-T",
                "postgres",
                "psql",
                "-U",
                "quantlab",
                "-d",
                "quantlab",
                "-Atc",
                (
                    "SELECT acknowledged_by || '|' || resolved_by || '|' || "
                    "resolution_reason FROM quantlab.strategy_allocation_events "
                    f"WHERE allocation_id='restore-{sentinel}' "
                    "AND rule='risk_lifecycle_restore';"
                ),
                capture=True,
            ).strip()
            expected_lifecycle = f"ack-{sentinel}|resolve-{sentinel}|resolution-{sentinel}"
            if lifecycle_sentinel != expected_lifecycle:
                raise RuntimeError("risk lifecycle evidence did not survive restore")
            schedule_state_sentinel = target.run(
                "exec",
                "-T",
                "postgres",
                "psql",
                "-U",
                "quantlab",
                "-d",
                "quantlab",
                "-Atc",
                (
                    "SELECT desired_status || '|' || status || '|' || suspension_reason "
                    "FROM quantlab.schedules "
                    f"WHERE id='schedule-{sentinel}';"
                ),
                capture=True,
            ).strip()
            if schedule_state_sentinel != "active|paused|portfolio:paused":
                raise RuntimeError("schedule intent and suspension state did not survive restore")
            pair_ledger_sentinel = target.run(
                "exec",
                "-T",
                "postgres",
                "psql",
                "-U",
                "quantlab",
                "-d",
                "quantlab",
                "-Atc",
                (
                    "SELECT p.name || '|' || n.quantity_y || '|' || n.quantity_x || '|' || "
                    "e.status || '|' || r.status FROM quantlab.pair_paper_portfolios p "
                    "JOIN quantlab.pair_portfolio_nav n ON n.portfolio_id=p.id "
                    "JOIN quantlab.pair_portfolio_risk_events e ON e.portfolio_id=p.id "
                    "JOIN quantlab.pair_portfolio_reviews r ON r.portfolio_id=p.id "
                    f"WHERE p.id='restore-pair-{sentinel}';"
                ),
                capture=True,
            ).strip()
            expected_pair_ledger = f"Restore pair ledger {sentinel}|1000|-800|resolved|completed"
            if pair_ledger_sentinel != expected_pair_ledger:
                raise RuntimeError("pair paper ledger evidence did not survive restore")
            table_count = int(
                target.run(
                    "exec",
                    "-T",
                    "postgres",
                    "psql",
                    "-U",
                    "quantlab",
                    "-d",
                    "quantlab",
                    "-Atc",
                    (
                        "SELECT count(*) FROM information_schema.tables "
                        "WHERE table_schema='quantlab' AND table_type='BASE TABLE';"
                    ),
                    capture=True,
                )
            )
            data_sentinel = target.run(
                "run",
                "--rm",
                "--no-deps",
                "api",
                "python",
                "-c",
                (
                    "from pathlib import Path; "
                    "print(Path('/data/restore-drill/sentinel.txt').read_text())"
                ),
                capture=True,
            ).splitlines()[-1]
            if data_sentinel != sentinel:
                raise RuntimeError("/data sentinel did not survive restore")
            runtime_secret_sentinel = target.run(
                "run",
                "--rm",
                "--no-deps",
                "api",
                "python",
                "-c",
                (
                    "from quant_data.config import Settings; "
                    "from quant_platform.runtime_secret_store import RuntimeSecretStore; "
                    "s=Settings.from_env(); "
                    "print(RuntimeSecretStore(s.database_url, s.platform_secret_key)"
                    ".get('restore_drill')['sentinel'])"
                ),
                capture=True,
            ).splitlines()[-1]
            if runtime_secret_sentinel != sentinel:
                raise RuntimeError("encrypted runtime-secret sentinel did not survive restore")
            result["checks"]["restored_state"] = {
                "status": "passed",
                "schema_revision": restored_revision,
                "table_count": table_count,
                "database_sentinel": "matched",
                "strategy_allocation_sentinel": "matched",
                "risk_lifecycle_sentinel": "matched",
                "schedule_state_sentinel": "matched",
                "pair_paper_ledger_sentinel": "matched",
                "data_sentinel": "matched",
                "runtime_secret_sentinel": "matched",
            }

            services = _assert_full_stack(target)
            api_health = json.loads(
                target.run(
                    "exec",
                    "-T",
                    "api",
                    "python",
                    "-c",
                    (
                        "import urllib.request; "
                        "print(urllib.request.urlopen('http://127.0.0.1:8765/api/health', "
                        "timeout=5).read().decode())"
                    ),
                    capture=True,
                ).splitlines()[-1]
            )
            if api_health.get("status") != "ok":
                raise RuntimeError("restored API health endpoint did not return ok")
            gateway_url = _gateway_url(target)
            with urllib.request.urlopen(gateway_url, timeout=10) as response:
                gateway_status = response.status
            if gateway_status != 200:
                raise RuntimeError(f"restored gateway returned HTTP {gateway_status}")
            result["checks"]["full_stack"] = {
                "status": "passed",
                "services": {
                    name: {
                        "state": services[name].get("State"),
                        "health": services[name].get("Health") or "not_configured",
                    }
                    for name in sorted(services)
                },
                "api": api_health,
                "gateway_http_status": gateway_status,
            }
            result["status"] = "passed"
            result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    except Exception as exc:
        result["error"] = str(exc)[:1000]
        result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        raise
    finally:
        for context in (target, source):
            if context is not None:
                context.run("down", "-v", "--remove-orphans", check=False, timeout=120)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if temporary_manager is not None:
            temporary_manager.cleanup()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise QuantLab backup and restore in two disposable Compose projects"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            PROJECT_ROOT / "artifacts" / "restore-drills" / f"restore-drill-{_timestamp()}.json"
        ),
    )
    args = parser.parse_args()
    result = run_drill(PROJECT_ROOT, args.report.resolve())
    print(json.dumps({"status": result["status"], "report": str(args.report.resolve())}))


if __name__ == "__main__":
    main()
