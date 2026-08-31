from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from quant_data.snapshot_lineage import canonical_sha256
from quant_platform.formal_validation import (
    PAIRED_BOOTSTRAP_EVIDENCE_CONTRACT_VERSION,
    build_paired_bootstrap_evidence,
    build_paired_bootstrap_evidence_from_daily_returns,
    paired_bootstrap_parameters_from_config,
    validate_paired_bootstrap_evidence,
    validate_paired_bootstrap_evidence_schema,
    validate_paired_bootstrap_parameters,
)

pytestmark = pytest.mark.no_database


def _returns() -> tuple[pd.DataFrame, dict[str, int]]:
    rng = np.random.default_rng(20260831)
    baseline = rng.normal(0.0001, 0.008, 96)
    gross = baseline + rng.normal(0.0009, 0.001, 96)
    cost = np.full(96, 0.0001)
    frame = pd.DataFrame(
        {
            "datetime": pd.bdate_range("2026-01-02", periods=96),
            "return": gross,
            "cost": cost,
            "bench": baseline,
        }
    )
    return frame, {"block_size": 12, "samples": 400, "seed": 17}


def _resign(value: dict) -> dict:
    payload = deepcopy(value)
    payload.pop("evidence_sha256", None)
    return {**payload, "evidence_sha256": canonical_sha256(payload)}


def test_daily_returns_and_aligned_series_recompute_identical_evidence() -> None:
    daily_returns, parameters = _returns()
    from_daily = build_paired_bootstrap_evidence_from_daily_returns(
        daily_returns,
        parameters=parameters,
    )
    from_series = build_paired_bootstrap_evidence(
        daily_returns["return"] - daily_returns["cost"],
        daily_returns["bench"],
        parameters=parameters,
    )

    assert from_daily == from_series
    assert from_daily["contract_version"] == (
        PAIRED_BOOTSTRAP_EVIDENCE_CONTRACT_VERSION
    )
    assert from_daily["observations"] == len(daily_returns)
    assert from_daily["block_size"] == 12
    assert from_daily["samples"] == 400
    assert from_daily["seed"] == 17
    assert validate_paired_bootstrap_evidence(
        from_daily,
        daily_returns=daily_returns,
        parameters=parameters,
    ) == from_daily


def test_daily_returns_complete_case_alignment_matches_runner_contract() -> None:
    daily_returns, parameters = _returns()
    daily_returns.loc[[0, 5], "bench"] = np.nan
    daily_returns.loc[[1], "cost"] = np.nan
    expected_rows = daily_returns[["return", "cost", "bench"]].dropna()

    evidence = build_paired_bootstrap_evidence_from_daily_returns(
        daily_returns,
        parameters=parameters,
    )

    assert evidence["observations"] == len(expected_rows)
    assert validate_paired_bootstrap_evidence(
        evidence,
        candidate_net_returns=expected_rows["return"] - expected_rows["cost"],
        baseline_returns=expected_rows["bench"],
        parameters=parameters,
    ) == evidence


def test_parameter_contract_is_exact_and_independent_of_claimed_evidence() -> None:
    assert paired_bootstrap_parameters_from_config({}) == {
        "block_size": 20,
        "samples": 2000,
        "seed": 0,
    }
    assert paired_bootstrap_parameters_from_config(
        {
            "bootstrap_block_days": 8,
            "bootstrap_samples": 300,
            "validation_seed": 9,
        }
    ) == {"block_size": 8, "samples": 300, "seed": 9}

    with pytest.raises(ValueError, match="invalid fields"):
        validate_paired_bootstrap_parameters(
            {"block_size": 8, "samples": 300, "seed": 9, "post_hoc": True}
        )
    with pytest.raises(ValueError, match="samples must be an integer"):
        validate_paired_bootstrap_parameters(
            {"block_size": 8, "samples": 300.0, "seed": 9}
        )
    with pytest.raises(ValueError, match="block_size exceeds observations"):
        validate_paired_bootstrap_parameters(
            {"block_size": 31, "samples": 300, "seed": 9},
            observations=30,
        )


@pytest.mark.parametrize("field", ["method", "one_sided_p_value", "input_sha256"])
def test_independent_recomputation_rejects_resigned_fabricated_fields(field: str) -> None:
    daily_returns, parameters = _returns()
    evidence = build_paired_bootstrap_evidence_from_daily_returns(
        daily_returns,
        parameters=parameters,
    )
    tampered = deepcopy(evidence)
    tampered[field] = (
        "forged" if isinstance(tampered[field], str) else float(tampered[field]) + 0.01
    )
    tampered = _resign(tampered)

    with pytest.raises(ValueError):
        validate_paired_bootstrap_evidence(
            tampered,
            daily_returns=daily_returns,
            parameters=parameters,
        )


def test_frozen_parameter_change_fails_even_when_payload_is_resigned() -> None:
    daily_returns, parameters = _returns()
    evidence = build_paired_bootstrap_evidence_from_daily_returns(
        daily_returns,
        parameters=parameters,
    )
    tampered = deepcopy(evidence)
    tampered["seed"] = 18
    tampered = _resign(tampered)

    with pytest.raises(ValueError, match="differ from frozen config"):
        validate_paired_bootstrap_evidence(
            tampered,
            daily_returns=daily_returns,
            parameters=parameters,
        )


def test_missing_and_surplus_evidence_fields_fail_closed() -> None:
    daily_returns, parameters = _returns()
    evidence = build_paired_bootstrap_evidence_from_daily_returns(
        daily_returns,
        parameters=parameters,
    )
    missing = deepcopy(evidence)
    missing.pop("probability_positive")
    surplus = {**evidence, "unfrozen_diagnostic": 1}

    with pytest.raises(ValueError, match="invalid fields"):
        validate_paired_bootstrap_evidence_schema(
            missing,
            expected_parameters=parameters,
        )
    with pytest.raises(ValueError, match="invalid fields"):
        validate_paired_bootstrap_evidence_schema(
            surplus,
            expected_parameters=parameters,
        )


def test_daily_return_contract_and_input_mode_fail_closed() -> None:
    daily_returns, parameters = _returns()
    evidence = build_paired_bootstrap_evidence_from_daily_returns(
        daily_returns,
        parameters=parameters,
    )

    with pytest.raises(ValueError, match="missing required columns"):
        build_paired_bootstrap_evidence_from_daily_returns(
            daily_returns.drop(columns="cost"),
            parameters=parameters,
        )
    with pytest.raises(ValueError, match="either daily_returns"):
        validate_paired_bootstrap_evidence(
            evidence,
            daily_returns=daily_returns,
            candidate_net_returns=daily_returns["return"] - daily_returns["cost"],
            baseline_returns=daily_returns["bench"],
            parameters=parameters,
        )
    with pytest.raises(ValueError, match="both paired return series"):
        validate_paired_bootstrap_evidence(
            evidence,
            candidate_net_returns=daily_returns["return"],
            parameters=parameters,
        )
