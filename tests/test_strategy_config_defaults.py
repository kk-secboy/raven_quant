import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_platform.api import StrategyConfigRequest
from quant_platform.strategy_store import (
    _canonical_sha256,
    _multifactor_manifest_failures,
    _sha256_file,
)

pytestmark = pytest.mark.no_database


def test_docx_risk_template_is_the_strategy_default() -> None:
    config = StrategyConfigRequest()
    assert config.stop_loss == pytest.approx(0.07)
    assert config.take_profit_partial == pytest.approx(0.12)
    assert config.max_drawdown_reduce == pytest.approx(0.10)
    assert config.max_volume_participation == pytest.approx(0.01)
    assert config.max_industry_weight == pytest.approx(0.30)


@pytest.mark.parametrize(
    "overrides",
    [
        {"take_profit_partial": 0.20, "take_profit": 0.20},
        {"max_drawdown_reduce": 0.15, "max_drawdown_liquidate": 0.15},
    ],
)
def test_docx_risk_threshold_order_is_fail_closed(overrides: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        StrategyConfigRequest(**overrides)


def test_benchmark_relative_optimizer_rejects_an_impossible_position_cap() -> None:
    with pytest.raises(ValidationError, match="topk \\* max_position_weight"):
        StrategyConfigRequest(
            portfolio_construction="benchmark_relative_qp",
            topk=20,
            max_position_weight=0.02,
        )


def test_strategy_approval_verifies_manifest_against_immutable_version(tmp_path: Path) -> None:
    config = StrategyConfigRequest().model_dump()
    code_path = tmp_path / "factor-1.py"
    values_path = tmp_path / "factor-1.parquet"
    code_path.write_text("def factor(frame):\n    return frame['close']\n", encoding="utf-8")
    values_path.write_bytes(b"immutable-factor-values")
    code_sha256 = _sha256_file(code_path)
    values_sha256 = _sha256_file(values_path)
    version = {
        "id": "version-1",
        "benchmark": "SH000300",
        "config": dict(config),
        "factors": [
            {
                "factor_candidate_id": "factor-1",
                "weight": 1.0,
                "direction": 1,
                "code_path": str(code_path),
                "code_sha256": code_sha256,
                "values_path": str(values_path),
            }
        ],
    }
    backtest = {
        "dataset": "snapshot-1",
        "periods": {"start": "2024-01-01", "end": "2026-07-10"},
        "artifact_path": str(tmp_path),
    }
    manifest = {
        "strategy_version_id": version["id"],
        "dataset": backtest["dataset"],
        "benchmark": version["benchmark"],
        "periods": backtest["periods"],
        "config": config,
        "factors": [
            {
                "candidate_id": "factor-1",
                "values_path": "/data/factor-1.parquet",
                "code_sha256": code_sha256,
                "weight": 1.0,
                "direction": 1,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    metrics = {
        "provenance": {
            "execution_manifest_sha256": _sha256_file(manifest_path),
            "strategy_config_sha256": _canonical_sha256(config),
            "factor_code_sha256": {"factor-1": code_sha256},
            "factor_values_sha256": {"factor-1": values_sha256},
        }
    }
    assert _multifactor_manifest_failures(version, backtest, metrics) == []

    values_path.write_bytes(b"tampered-factor-values")
    failures = _multifactor_manifest_failures(version, backtest, metrics)
    assert "factor factor-1 values artifact does not match provenance" in failures
