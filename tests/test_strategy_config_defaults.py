import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_platform.api import StrategyConfigRequest
from quant_platform.strategy_store import (
    _canonical_sha256,
    _execution_replay_artifact_failures,
    _execution_replay_failures,
    _multifactor_manifest_failures,
    _sha256_file,
)

pytestmark = pytest.mark.no_database


def test_docx_risk_template_is_the_strategy_default() -> None:
    config = StrategyConfigRequest()

    assert config.stop_loss == pytest.approx(0.07)
    assert config.take_profit_partial == pytest.approx(0.12)
    assert config.take_profit_partial_fraction == pytest.approx(0.50)
    assert config.take_profit == pytest.approx(0.20)
    assert config.max_drawdown_reduce == pytest.approx(0.10)
    assert config.drawdown_reduction_exposure == pytest.approx(0.50)
    assert config.max_drawdown_liquidate == pytest.approx(0.15)
    assert config.max_volume_participation == pytest.approx(0.01)
    assert config.max_industry_weight == pytest.approx(0.30)


def test_docx_risk_template_can_be_overridden_per_strategy_version() -> None:
    config = StrategyConfigRequest(
        stop_loss=0.05,
        take_profit_partial=0.10,
        take_profit_partial_fraction=0.40,
        take_profit=0.18,
        max_drawdown_reduce=0.08,
        drawdown_reduction_exposure=0.35,
        max_drawdown_liquidate=0.12,
        max_volume_participation=0.005,
        max_industry_weight=0.25,
    )

    assert config.stop_loss == pytest.approx(0.05)
    assert config.take_profit_partial_fraction == pytest.approx(0.40)
    assert config.drawdown_reduction_exposure == pytest.approx(0.35)
    assert config.max_volume_participation == pytest.approx(0.005)
    assert config.max_industry_weight == pytest.approx(0.25)


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


def _replay_metrics(config: StrategyConfigRequest) -> dict:
    thresholds = {
        field: getattr(config, field)
        for field in (
            "max_daily_loss",
            "stop_loss",
            "take_profit_partial",
            "take_profit_partial_fraction",
            "take_profit",
            "max_drawdown_reduce",
            "max_drawdown_liquidate",
            "drawdown_reduction_exposure",
        )
    }
    return {
        "execution_risk_overlay_enforced": True,
        "execution_replay": {
            "execution_risk_overlay_enforced": True,
            "execution_model": "next_open",
            "max_drawdown": -0.09,
            "execution_risk_thresholds": thresholds,
        },
    }


def test_strategy_approval_accepts_matching_execution_replay_evidence() -> None:
    config = StrategyConfigRequest()

    assert _execution_replay_failures(config.model_dump(), _replay_metrics(config)) == []


def test_strategy_approval_requires_optimizer_execution_replay_evidence() -> None:
    config = StrategyConfigRequest(portfolio_construction="benchmark_relative_qp")
    metrics = _replay_metrics(config)
    metrics["execution_replay"].update(
        {
            "portfolio_construction": "benchmark_relative_qp",
            "optimizer_execution_replay_enforced": True,
            "optimizer_days": 504,
        }
    )
    assert _execution_replay_failures(config.model_dump(), metrics) == []

    metrics["execution_replay"]["optimizer_execution_replay_enforced"] = False
    failures = _execution_replay_failures(config.model_dump(), metrics)
    assert "benchmark-relative optimizer execution replay is required" in failures


def test_benchmark_relative_optimizer_rejects_an_impossible_position_cap() -> None:
    with pytest.raises(ValidationError, match="topk \\* max_position_weight"):
        StrategyConfigRequest(
            portfolio_construction="benchmark_relative_qp",
            topk=20,
            max_position_weight=0.02,
        )


def test_benchmark_relative_optimizer_requires_a_nonzero_objective() -> None:
    with pytest.raises(ValidationError, match="positive weight"):
        StrategyConfigRequest(
            optimizer_alpha_weight=0,
            optimizer_tracking_penalty=0,
            optimizer_turnover_penalty=0,
        )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing", "execution risk replay is required"),
        ("threshold", "stop_loss does not match"),
        ("drawdown", "max_drawdown=-0.3 violates"),
        ("execution", "must use next_open"),
    ],
)
def test_strategy_approval_rejects_invalid_execution_replay_evidence(
    mutation: str, expected: str
) -> None:
    config = StrategyConfigRequest()
    metrics = _replay_metrics(config)
    if mutation == "missing":
        metrics = {}
    elif mutation == "threshold":
        metrics["execution_replay"]["execution_risk_thresholds"]["stop_loss"] = 0.08
    elif mutation == "drawdown":
        metrics["execution_replay"]["max_drawdown"] = -0.30
    else:
        metrics["execution_replay"]["execution_model"] = "close"

    failures = _execution_replay_failures(config.model_dump(), metrics)
    assert any(expected in item for item in failures)


def test_strategy_approval_verifies_execution_replay_artifact_hash(tmp_path: Path) -> None:
    config = StrategyConfigRequest()
    metrics = _replay_metrics(config)
    digest = _canonical_sha256(metrics["execution_replay"])
    metrics["provenance"] = {"execution_replay_sha256": digest}
    replay_path = tmp_path / "execution_replay.json"
    replay_path.write_text(
        json.dumps(metrics["execution_replay"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    assert _execution_replay_artifact_failures(metrics, tmp_path) == []

    replay_path.write_text('{"tampered": true}', encoding="utf-8")
    failures = _execution_replay_artifact_failures(metrics, tmp_path)
    assert failures == ["execution replay artifact does not match its SHA-256 provenance"]


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
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
    values_path.write_bytes(b"immutable-factor-values")

    manifest["config"]["stop_loss"] = 0.08
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failures = _multifactor_manifest_failures(version, backtest, metrics)
    assert "strategy backtest manifest does not match its SHA-256 provenance" in failures
    assert "strategy backtest manifest config does not match the immutable version" in failures
