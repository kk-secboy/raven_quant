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
