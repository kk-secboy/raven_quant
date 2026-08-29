from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

from quant_platform.research_horizon import (
    LEGACY_AMBIGUOUS,
    research_horizon_contract,
)

pytestmark = pytest.mark.no_database

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "restore_drill.py"


def _load_restore_drill_module():
    scripts_path = str(SCRIPT_PATH.parent)
    sys.path.insert(0, scripts_path)
    try:
        spec = importlib.util.spec_from_file_location("restore_drill_contract", SCRIPT_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts_path)


def test_restore_drill_seeds_the_canonical_0072_legacy_horizon() -> None:
    module = _load_restore_drill_module()
    expected = research_horizon_contract(LEGACY_AMBIGUOUS)
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert module.LEGACY_HORIZON == expected
    assert json.loads(module.LEGACY_HORIZON_JSON) == expected.to_dict()
    assert "horizon_profile, label_horizons_json" in source
    assert "sealed_oos_required, sealed_oos_sessions, horizon_contract_json" in source
    assert "strategy_horizon_sentinel" in source
    assert "promotion_stage IS NULL" in source
    assert "quantlab.recommendation_portfolios" in source
    assert "recommendation_sentinel" in source
    assert "JOIN quantlab.simulation_batches b" in source
    assert "(n.nav = 5000010)::text" in source
    assert "n.performance_certified::text" in source


def test_restore_drill_batch_sentinel_binds_immutable_dataset_contract() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    batch_insert = source.split("INSERT INTO quantlab.simulation_batches", 1)[1].split(
        "INSERT INTO quantlab.simulation_positions", 1
    )[0]

    for column in (
        "daily_dataset",
        "daily_dataset_identity_sha256",
        "daily_dataset_lineage_id",
        "execution_dataset",
        "execution_dataset_identity_sha256",
        "execution_dataset_lineage_id",
        "simulation_semantics_sha256",
    ):
        assert column in batch_insert
    assert "f\"'restore-daily', '{sentinel}{sentinel}', '{sentinel}{sentinel}', \"" in batch_insert
    assert "f\"'restore-minute', '{sentinel}{sentinel}', '{sentinel}{sentinel}', \"" in batch_insert
    assert "f\"'{sentinel}{sentinel}', current_date - 1, current_date, \"" in batch_insert


def test_restore_drill_defaults_to_v2_and_keeps_v1_available(tmp_path: Path) -> None:
    module = _load_restore_drill_module()
    parameter = inspect.signature(module.run_drill).parameters["format_version"]
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert parameter.default == module.CONTROL_PLANE_BACKUP_FORMAT_VERSION == 2
    assert module.SUPPORTED_BACKUP_FORMATS == (1, 2)
    with pytest.raises(ValueError, match="format must be 1 or 2"):
        module.run_drill(ROOT, tmp_path / "unused.json", format_version=3)

    assert "format_version=format_version" in source
    assert 'backup_manifest_check["data_archive"]' in source
    assert '"control_plane_bytes"' in source
    assert '"immutable_inventory"' in source
    assert '"inventory_sha256"' in source
    assert 'int(immutable["inventory_entries"]) < 1' in source
    assert 'immutable["inventory_truncated"] is not False' in source
    assert "restore-drill-data-v1" in source


def test_v2_restore_drill_proves_target_data_is_preserved() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "/data/restore-drill/source-sentinel.txt" in source
    assert "/data/restore-drill/target-sentinel.txt" in source
    assert 'source_data_sentinel != "MISSING"' in source
    assert "restored_target_data_sentinel != target_data_sentinel" in source
    assert '"contract": "preserve_existing_data_volume"' in source
    assert '"source_data_sentinel": "not_copied"' in source
    assert '"target_data_sentinel": "preserved"' in source
    assert '"contract": "replace_data_volume"' in source


def test_restore_drill_uses_real_shared_images_and_a_governed_qlib_fixture() -> None:
    compose = (ROOT / "deploy" / "compose.restore-drill.yaml").read_text(
        encoding="utf-8"
    )
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    for expected in ("quantlab-worker-runtime:v2", "quantlab-rdagent-runtime:v2"):
        assert expected in compose
        assert expected in script
    for obsolete in (
        "quantlab-platform-worker:latest",
        "quantlab-platform-rdagent-worker:latest",
    ):
        assert obsolete not in compose
        assert obsolete not in script

    assert "qlib-drill-seed:" in compose
    assert 'root = Path("/data/qlib/restore-drill-v1")' in compose
    assert 'feature.write_bytes(struct.pack("<fff", 0.0, 10.0, 10.25))' in compose
    assert '"contract_version": "restore-drill-qlib-provider-v1"' in compose
    assert "condition: service_completed_successfully" in compose


def test_restore_drill_builds_an_isolated_candidate_schema_runtime(tmp_path: Path) -> None:
    module = _load_restore_drill_module()
    override = tmp_path / "candidate-runtime.compose.json"
    image = "quantlab-restore-drill-api:test-runtime"

    class CandidateContext:
        def __init__(self) -> None:
            self.arguments: tuple[str, ...] | None = None

        def run(self, *arguments: str, **options: object) -> str:
            self.arguments = arguments
            assert options == {"capture": True}
            return "0072_strategy_horizons\n"

    context = CandidateContext()

    module._write_candidate_runtime_override(override, image)
    head = module._candidate_migration_head(context)
    payload = json.loads(override.read_text(encoding="utf-8"))
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert head == "0072_strategy_horizons"
    assert context.arguments is not None
    assert context.arguments[:5] == ("run", "--rm", "--no-deps", "api", "python")
    assert "ScriptDirectory.from_config" in context.arguments[-1]
    assert payload == {
        "services": {
            "api": {"image": image},
            "scheduler": {"image": image},
        }
    }
    assert 'source.run("build", "api")' in source
    assert "expected_schema_revision = _candidate_migration_head(source)" in source
    assert "_assert_current_schema(source, expected_schema_revision)" in source
    assert "manifest[\"schema_revision\"] != expected_schema_revision" in source
    assert "restored_revision != expected_schema_revision" in source
    assert "quantlab-platform-api:latest" not in source
    assert "quantlab-platform-scheduler:latest" not in source


def test_restore_drill_fails_before_sentinels_on_stale_source_schema() -> None:
    module = _load_restore_drill_module()

    class RevisionContext:
        def __init__(self, revision: str) -> None:
            self.revision = revision
            self.arguments: tuple[str, ...] | None = None

        def run(self, *arguments: str, **options: object) -> str:
            self.arguments = arguments
            assert options == {"capture": True}
            return self.revision

    current = RevisionContext("0072_strategy_horizons")
    module._assert_current_schema(current, "0072_strategy_horizons")
    assert current.arguments is not None
    assert "SELECT version_num FROM quantlab.alembic_version;" in current.arguments

    stale = RevisionContext("0071_retire_pair_writes")
    with pytest.raises(RuntimeError, match="expected 0072_strategy_horizons"):
        module._assert_current_schema(stale, "0072_strategy_horizons")


def test_restore_drill_isolates_and_seals_the_target_sandbox(tmp_path: Path) -> None:
    module = _load_restore_drill_module()
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    env_file = tmp_path / "target.env"
    data = tmp_path / "target-data"
    docker = tmp_path / "target-rdagent-docker"
    registry = tmp_path / "target-rdagent-registry"

    module._write_env(
        env_file,
        password="test-password",
        secret_key="test-secret",
        data_host_path=data,
        docker_host_path=docker,
        registry_host_path=registry,
        registry_port=54321,
    )
    rendered = dict(
        line.split("=", 1)
        for line in env_file.read_text(encoding="utf-8").splitlines()
    )

    assert rendered["QUANTLAB_DATA_HOST_PATH"] == str(data.resolve())
    assert rendered["RDAGENT_DOCKER_HOST_PATH"] == str(docker.resolve())
    assert rendered["RDAGENT_REGISTRY_HOST_PATH"] == str(registry.resolve())
    assert rendered["RDAGENT_REGISTRY_PORT"] == "54321"
    assert rendered["OPENAI_API_KEY"] == "restore-drill-not-a-real-credential"
    assert rendered["OPENAI_API_BASE"] == "http://127.0.0.1:9"
    assert "/data/quantlab-rdagent-docker" not in source
    assert "/data/quantlab-rdagent-registry" not in source
    assert "@control_plane_locked\n@isolated_drill_environment\ndef run_drill" in source
    assert "prepare_drill_sandbox_bootstrap(" in source
    assert source.index("prepare_drill_sandbox_bootstrap(", source.index("def run_drill")) < (
        source.index("services = _assert_full_stack(target)")
    )
    assert "source_and_target_docker_distinct" in source
    assert "source_and_target_registry_ports_distinct" in source
