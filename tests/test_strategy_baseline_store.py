from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest
from governance_fixtures import (
    PERIODS,
    create_promoted_factor,
    formal_backtest_metrics,
)
from qlib_test_doubles import qlib_workflow_identity

from quant_platform.api import StrategyConfigRequest
from quant_platform.qlib_factor_baseline import (
    FACTOR_SOURCE_QLIB_BASELINE,
    FACTOR_SOURCE_QLIB_BASELINE_PLUS_CHALLENGER,
)
from quant_platform.strategy_recipes import get_strategy_recipe
from quant_platform.strategy_store import StrategyStore


def _core_config(
    *,
    mode: str = FACTOR_SOURCE_QLIB_BASELINE,
    challenger_weight: float = 0.0,
) -> dict:
    recipe = get_strategy_recipe("index_enhancement")
    return StrategyConfigRequest.model_validate(
        {
            **recipe["config_overrides"],
            "recipe_id": recipe["id"],
            "recipe_version": recipe["version"],
            "factor_source_mode": mode,
            "challenger_weight": challenger_weight,
            # Keep this store-level test independent of minute execution fixtures.
            "execution_method": "open",
            "execution_frequency": "day",
            "execution_days": 1,
        }
    ).model_dump()


def _create_baseline_strategy(database_url: str) -> dict:
    return StrategyStore(database_url).create(
        name=f"baseline-{uuid.uuid4().hex}",
        description="Immutable Qlib six-factor baseline strategy fixture.",
        benchmark="SH000300",
        universe="cn_all",
        factors=[],
        config=_core_config(),
        actor="test",
    )


def test_core_family_starts_without_rdagent_and_challenger_is_a_new_version(
    database_url: str, tmp_path: Path
) -> None:
    store = StrategyStore(database_url)
    strategy = _create_baseline_strategy(database_url)
    baseline = strategy["versions"][0]

    assert baseline["version"] == 1
    assert baseline["factor_source_mode"] == FACTOR_SOURCE_QLIB_BASELINE
    assert baseline["factors"] == []
    assert len(baseline["baseline_definition_sha256"]) == 64

    factor = create_promoted_factor(database_url, tmp_path)
    challenger = store.create_version(
        strategy["id"],
        benchmark="SH000300",
        universe="cn_all",
        factors=[{"candidate_id": factor["id"], "weight": 1.0}],
        config=_core_config(
            mode=FACTOR_SOURCE_QLIB_BASELINE_PLUS_CHALLENGER,
            challenger_weight=0.30,
        ),
        actor="test",
    )

    assert challenger["version"] == 2
    assert challenger["factor_source_mode"] == (
        FACTOR_SOURCE_QLIB_BASELINE_PLUS_CHALLENGER
    )
    assert challenger["config"]["challenger_weight"] == pytest.approx(0.30)
    assert challenger["baseline_definition_sha256"] == (
        baseline["baseline_definition_sha256"]
    )
    assert store.get_version(baseline["id"])["factors"] == []


def test_core_family_cannot_silently_start_as_a_challenger(
    database_url: str,
) -> None:
    with pytest.raises(ValueError, match="must start with an independent Qlib baseline"):
        StrategyStore(database_url).create(
            name=f"invalid-core-{uuid.uuid4().hex}",
            description="This family incorrectly attempts to skip its baseline.",
            benchmark="SH000300",
            universe="cn_all",
            factors=[{"candidate_id": "not-consulted", "weight": 1.0}],
            config=_core_config(
                mode=FACTOR_SOURCE_QLIB_BASELINE_PLUS_CHALLENGER,
                challenger_weight=0.30,
            ),
            actor="test",
        )


def test_minute_mean_reversion_family_starts_from_qlib_expression_baseline(
    database_url: str,
) -> None:
    recipe = get_strategy_recipe("minute_mean_reversion")
    config = StrategyConfigRequest.model_validate(
        {
            **recipe["config_overrides"],
            "recipe_id": recipe["id"],
            "recipe_version": recipe["version"],
        }
    ).model_dump()

    strategy = StrategyStore(database_url).create(
        name=f"minute-baseline-{uuid.uuid4().hex}",
        description="Immutable Qlib minute mean-reversion expression baseline.",
        benchmark="SH000300",
        universe="cn_all",
        factors=[],
        config=config,
        actor="test",
    )

    version = strategy["versions"][0]
    assert version["factor_source_mode"] == FACTOR_SOURCE_QLIB_BASELINE
    assert version["signal_frequency"] == "5min"
    assert version["execution_frequency"] == "5min"
    assert version["config"]["baseline_definition"]["frequency"] == "5min"
    assert len(version["config"]["baseline_definition"]["factors"]) == 3


def test_swing_family_starts_from_qlib_baseline_before_rdagent_challengers(
    database_url: str,
) -> None:
    recipe = get_strategy_recipe("swing_trend")
    config = StrategyConfigRequest.model_validate(
        {
            **recipe["config_overrides"],
            "recipe_id": recipe["id"],
            "recipe_version": recipe["version"],
            "execution_method": "open",
            "execution_frequency": "day",
            "execution_days": 1,
        }
    ).model_dump()

    strategy = StrategyStore(database_url).create(
        name=f"swing-baseline-{uuid.uuid4().hex}",
        description="Immutable Qlib swing expression baseline.",
        benchmark="SH000300",
        universe="cn_all",
        factors=[],
        config=config,
        actor="test",
    )

    version = strategy["versions"][0]
    assert version["factor_source_mode"] == FACTOR_SOURCE_QLIB_BASELINE
    assert version["factors"] == []
    assert version["config"]["baseline_definition"]["frequency"] == "day"
    assert len(version["config"]["baseline_definition"]["factors"]) == 5


def test_baseline_approval_validates_expression_artifacts_hashes_and_recorder(
    database_url: str, tmp_path: Path
) -> None:
    store = StrategyStore(database_url)
    version = _create_baseline_strategy(database_url)["versions"][0]
    artifact = tmp_path / "baseline-formal"
    artifact.mkdir()
    periods = {
        "start": PERIODS["test_start"].isoformat(),
        "end": PERIODS["test_end"].isoformat(),
        "historical_start": PERIODS["train_start"].isoformat(),
        "historical_end": PERIODS["valid_end"].isoformat(),
    }
    backtest = store.create_backtest(
        version_id=version["id"],
        dataset="snapshot",
        periods=periods,
        artifact_path=artifact,
    )
    definition = version["config"]["baseline_definition"]
    baseline_artifacts: dict[str, object] = {"raw": {}, "normalized": {}}
    for artifact_kind in ("raw", "normalized"):
        for item in definition["factors"]:
            path = artifact / "baseline" / artifact_kind / f"{item['id']}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{artifact_kind}:{item['id']}".encode())
            baseline_artifacts[artifact_kind][item["id"]] = {
                "path": path.relative_to(artifact).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    composite = artifact / "baseline" / "composite.parquet"
    composite.write_bytes(b"immutable-composite")
    baseline_artifacts["composite"] = {
        "path": composite.relative_to(artifact).as_posix(),
        "sha256": hashlib.sha256(composite.read_bytes()).hexdigest(),
    }
    recorder = qlib_workflow_identity()
    manifest = artifact / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy_version_id": version["id"],
                "dataset": "snapshot",
                "execution_dataset": None,
                "benchmark": version["benchmark"],
                "universe": version["universe"],
                "periods": {
                    "start": periods["start"],
                    "end": periods["end"],
                },
                "historical_validation_periods": {
                    "start": periods["historical_start"],
                    "end": periods["historical_end"],
                },
                "config": version["config"],
                "factors": [],
                "factor_source_mode": FACTOR_SOURCE_QLIB_BASELINE,
                "challenger_weight": 0.0,
                "baseline": {
                    "definition": definition,
                    "definition_sha256": version[
                        "baseline_definition_sha256"
                    ],
                    "computed_by": "qlib.data.D.features",
                    "artifacts": baseline_artifacts,
                },
                "qlib_workflow": recorder,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    metrics = formal_backtest_metrics(version, manifest)
    metrics["provenance"].update(
        {
            "factor_source_mode": FACTOR_SOURCE_QLIB_BASELINE,
            "challenger_weight": 0.0,
            "baseline_definition_sha256": version[
                "baseline_definition_sha256"
            ],
            "baseline_qlib_expressions": {
                item["id"]: item["qlib_expression"]
                for item in definition["factors"]
            },
            "baseline_preprocessing": definition["preprocessing"],
            "baseline_raw_values_sha256": {
                factor_id: entry["sha256"]
                for factor_id, entry in baseline_artifacts["raw"].items()
            },
            "baseline_normalized_values_sha256": {
                factor_id: entry["sha256"]
                for factor_id, entry in baseline_artifacts["normalized"].items()
            },
            "baseline_composite_values_sha256": baseline_artifacts[
                "composite"
            ]["sha256"],
        }
    )

    store.validate_backtest_artifacts(backtest["id"], metrics)
    invalid = json.loads(json.dumps(metrics))
    invalid["provenance"]["baseline_composite_values_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="baseline composite provenance"):
        store.validate_backtest_artifacts(backtest["id"], invalid)

    store.mark_backtest(backtest["id"], "succeeded", metrics=metrics)
    approved = store.approve(
        version["id"],
        actor="risk-owner",
        reason="Independent review accepted the immutable Qlib baseline evidence.",
    )
    assert approved["status"] == "approved"
