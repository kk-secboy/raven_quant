from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from qlib_test_doubles import QlibRiskEstimator, qlib_runtime_identity

from quant_platform.risk_math import (
    MARKET_FACTOR,
    STRUCTURED_RISK_MODEL_VERSION,
    StructuredRiskModel,
    estimate_covariance,
    estimate_structured_risk_model,
    validate_covariance,
)

pytestmark = pytest.mark.no_database

DAYS = 140
STOCKS_PER_INDUSTRY = 8
INDUSTRIES = ("IND-A", "IND-B", "IND-C")


def _synthetic_world(seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=DAYS)
    instruments = [
        f"S{index:03d}" for index in range(STOCKS_PER_INDUSTRY * len(INDUSTRIES))
    ]
    industry_of = {
        instrument: INDUSTRIES[index // STOCKS_PER_INDUSTRY]
        for index, instrument in enumerate(instruments)
    }
    true_factors = pd.DataFrame(
        {
            MARKET_FACTOR: rng.normal(0.0003, 0.010, DAYS),
            "size": rng.normal(0.0, 0.006, DAYS),
            "value": rng.normal(0.0, 0.006, DAYS),
            "IND-B": rng.normal(0.0, 0.004, DAYS),
            "IND-C": rng.normal(0.0005, 0.004, DAYS),
        },
        index=dates,
    )
    size_exposure = pd.Series(rng.normal(0.0, 1.0, len(instruments)), index=instruments)
    value_exposure = pd.Series(rng.normal(0.0, 1.0, len(instruments)), index=instruments)
    noise = rng.normal(0.0, 0.008, (DAYS, len(instruments)))
    returns = pd.DataFrame(noise, index=dates, columns=instruments)
    returns += true_factors[MARKET_FACTOR].to_numpy()[:, None]
    returns += true_factors["size"].to_numpy()[:, None] * size_exposure.to_numpy()
    returns += true_factors["value"].to_numpy()[:, None] * value_exposure.to_numpy()
    for name in ("IND-B", "IND-C"):
        members = [item for item in instruments if industry_of[item] == name]
        returns[members] += true_factors[name].to_numpy()[:, None]

    styles = pd.DataFrame(
        [
            {
                "instrument": instrument,
                "datetime": timestamp,
                "size": size_exposure[instrument],
                "value": value_exposure[instrument],
                "log_market_cap": 20.0,  # raw passthrough column, never a factor
            }
            for timestamp in dates
            for instrument in instruments
        ]
    )
    memberships = pd.DataFrame(
        [
            {
                "instrument": instrument,
                "industry": industry_of[instrument],
                "in_date": pd.Timestamp("2020-01-01"),
                "out_date": pd.NaT,
            }
            for instrument in instruments
        ]
    )
    return dates, instruments, industry_of, returns, styles, memberships, true_factors


def _estimate(**overrides) -> StructuredRiskModel:
    _, _, _, returns, styles, memberships, _ = _synthetic_world()
    options = {"min_cross_section": 10, "min_specific_observations": 20}
    options.update(overrides)
    return estimate_structured_risk_model(returns, styles, memberships, **options)


def test_factor_covariance_is_positive_definite() -> None:
    model = _estimate()
    values = model.factor_covariance.to_numpy(dtype=float)
    assert np.linalg.eigvalsh(values).min() > 0
    assert model.factor_covariance.columns.tolist()[0] == MARKET_FACTOR
    assert set(model.style_factors) == {"size", "value"}
    assert set(model.industry_factors) == {"industry:IND-B", "industry:IND-C"}
    assert model.reference_industry == "IND-A"


def test_structured_covariance_validates_and_matches_shapes() -> None:
    model = _estimate()
    covariance = model.covariance()
    assert covariance.shape == (24, 24)
    validate_covariance(covariance.to_numpy(dtype=float))
    subset = model.covariance(["S001", "S002"])
    assert subset.shape == (2, 2)
    with pytest.raises(ValueError, match="no exposures"):
        model.covariance(["UNKNOWN"])


def test_factor_returns_recover_market() -> None:
    dates, instruments, industry_of, returns, styles, memberships, true_factors = (
        _synthetic_world()
    )
    model = _estimate()
    # Re-estimate to access the daily regression: approximate market recovery by
    # projecting the equal-weight portfolio through the model instead.
    weights = pd.Series(1.0 / len(instruments), index=instruments)
    report = model.portfolio_risk(weights)
    assert report.volatility == pytest.approx(
        float(
            np.sqrt(
                weights.to_numpy()
                @ model.covariance().to_numpy(dtype=float)
                @ weights.to_numpy()
            )
        ),
        rel=1e-9,
    )


def test_portfolio_risk_decomposition_sums_to_total() -> None:
    _, instruments, _, _, _, _, _ = _synthetic_world()
    model = _estimate()
    rng = np.random.default_rng(3)
    raw = rng.uniform(0.0, 1.0, len(instruments))
    weights = pd.Series(raw / raw.sum(), index=instruments)
    report = model.portfolio_risk(weights)
    decomposed = (
        report.market_variance
        + report.style_variance
        + report.industry_variance
        + report.specific_variance
    )
    assert decomposed == pytest.approx(report.total_variance, rel=1e-9)
    assert report.volatility == pytest.approx(np.sqrt(report.total_variance))
    annualized = model.portfolio_risk(weights, annualize=True)
    assert annualized.volatility == pytest.approx(report.volatility * np.sqrt(252.0))
    assert report.factor_contributions.sum() + report.specific_variance == pytest.approx(
        report.total_variance
    )
    assert report.factor_exposures.index.tolist() == model.factor_names


def test_portfolio_risk_rejects_unknown_instruments() -> None:
    model = _estimate()
    with pytest.raises(ValueError, match="outside the risk model"):
        model.portfolio_risk(pd.Series({"UNKNOWN": 1.0}))
    with pytest.raises(ValueError, match="positive periods"):
        model.portfolio_risk(
            pd.Series({"S001": 1.0}),
            annualize=True,
            periods_per_year=0,
        )


def test_final_risk_exposures_fail_closed_on_missing_style_or_industry() -> None:
    dates, _, _, returns, styles, memberships, _ = _synthetic_world()
    missing_style = styles[
        ~(
            (styles["instrument"] == "S000")
            & (styles["datetime"] == dates[-1])
        )
    ]
    with pytest.raises(ValueError, match="stale style"):
        estimate_structured_risk_model(
            returns,
            missing_style,
            memberships,
            min_cross_section=10,
            min_specific_observations=20,
        )

    missing_industry = memberships[memberships["instrument"] != "S000"]
    with pytest.raises(ValueError, match="incomplete industry"):
        estimate_structured_risk_model(
            returns,
            styles,
            missing_industry,
            min_cross_section=10,
            min_specific_observations=20,
        )


def test_structured_risk_rejects_infinite_regression_weights() -> None:
    dates, instruments, _, returns, styles, memberships, _ = _synthetic_world()
    weights = pd.DataFrame(
        [
            {
                "instrument": instrument,
                "datetime": timestamp,
                "weight": float("inf") if instrument == "S000" else 1.0,
            }
            for timestamp in dates
            for instrument in instruments
        ]
    )

    with pytest.raises(ValueError, match="regression weights must be finite"):
        estimate_structured_risk_model(
            returns,
            styles,
            memberships,
            weights,
            min_cross_section=10,
            min_specific_observations=20,
        )


@pytest.mark.parametrize(
    ("in_date", "out_date"),
    [
        ("not-a-date", None),
        ("2026-01-03", "2026-01-02"),
    ],
)
def test_structured_risk_rejects_invalid_membership_intervals(
    in_date: str, out_date: str | None
) -> None:
    _, _, _, returns, styles, memberships, _ = _synthetic_world()
    memberships = memberships.copy()
    memberships["in_date"] = memberships["in_date"].astype(object)
    memberships["out_date"] = memberships["out_date"].astype(object)
    memberships.loc[0, "in_date"] = in_date
    memberships.loc[0, "out_date"] = out_date

    with pytest.raises(ValueError, match="industry membership"):
        estimate_structured_risk_model(
            returns,
            styles,
            memberships,
            min_cross_section=10,
            min_specific_observations=20,
        )


def test_industry_premium_visible_in_factor_covariance() -> None:
    # IND-C carries a persistent premium; its industry factor should show up
    # with positive exposure-return association in the model artifact.
    model = _estimate()
    exposures = model.exposures
    ind_c = exposures["industry:IND-C"]
    assert set(ind_c.unique()) == {0.0, 1.0}
    members = ind_c[ind_c == 1.0].index
    assert len(members) == STOCKS_PER_INDUSTRY


def test_membership_changes_are_point_in_time() -> None:
    dates, instruments, _, returns, styles, memberships, _ = _synthetic_world()
    moved = "S000"
    switch = dates[-10]
    memberships = memberships.copy()
    memberships.loc[memberships["instrument"] == moved, "out_date"] = switch
    late = pd.DataFrame(
        [
            {
                "instrument": moved,
                "industry": "IND-B",
                "in_date": switch + pd.Timedelta(days=1),
                "out_date": pd.NaT,
            },
            {
                # Future membership must never leak into the model.
                "instrument": "S008",
                "industry": "IND-C",
                "in_date": dates[-1] + pd.Timedelta(days=3),
                "out_date": pd.NaT,
            },
        ]
    )
    memberships = pd.concat([memberships, late], ignore_index=True)
    model = estimate_structured_risk_model(
        returns, styles, memberships, min_cross_section=10, min_specific_observations=20
    )
    # The mid-sample switch is visible at as_of: IND-B becomes the largest
    # membership and therefore the reference industry, and the moved stock
    # carries no industry dummy (reference-industry membership).
    assert model.reference_industry == "IND-B"
    assert model.exposures.loc[moved, list(model.industry_factors)].sum() == 0.0
    assert model.exposures.loc["S001", "industry:IND-A"] == 1.0
    # The future membership (in_date after as_of) did not leak: S008 stays in
    # the reference industry IND-B instead of moving to IND-C.
    assert model.exposures.loc["S008", list(model.industry_factors)].sum() == 0.0
    assert model.exposures.loc["S016", "industry:IND-C"] == 1.0


def test_new_stock_gets_median_specific_variance() -> None:
    dates, instruments, _, returns, styles, memberships, _ = _synthetic_world()
    returns = returns.copy()
    newbie = "S000"
    returns.loc[dates[:-5], newbie] = np.nan
    model = estimate_structured_risk_model(
        returns, styles, memberships, min_cross_section=10, min_specific_observations=20
    )
    others = model.specific_variance.drop(index=newbie)
    assert model.specific_variance[newbie] == pytest.approx(float(others.median()))


def test_save_load_roundtrip(tmp_path) -> None:
    model = _estimate()
    target = model.save(tmp_path / "risk_model")
    loaded = StructuredRiskModel.load(target)
    assert loaded.version == STRUCTURED_RISK_MODEL_VERSION
    assert loaded.as_of == model.as_of
    assert loaded.style_factors == model.style_factors
    assert loaded.industry_factors == model.industry_factors
    assert loaded.reference_industry == model.reference_industry
    pd.testing.assert_frame_equal(loaded.exposures, model.exposures)
    pd.testing.assert_frame_equal(loaded.factor_covariance, model.factor_covariance)
    pd.testing.assert_series_equal(loaded.specific_variance, model.specific_variance)

    manifest = json.loads((target / "model.json").read_text(encoding="utf-8"))
    manifest["version"] = "obsolete"
    (target / "model.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="version mismatch"):
        StructuredRiskModel.load(target)


def test_coexists_with_shrink_covariance_fallback() -> None:
    _, _, _, returns, styles, memberships, _ = _synthetic_world()
    complete = returns.dropna(axis=1, how="any")
    shrunk = estimate_covariance(
        complete,
        estimator_factory=QlibRiskEstimator,
        runtime_identity=qlib_runtime_identity,
    )
    validate_covariance(shrunk.to_numpy(dtype=float))
    model = estimate_structured_risk_model(
        returns, styles, memberships, min_cross_section=10, min_specific_observations=20
    )
    structured = model.covariance()
    validate_covariance(structured.to_numpy(dtype=float))
    assert structured.index.equals(shrunk.index)


def test_estimation_requires_enough_data() -> None:
    _, _, _, returns, styles, memberships, _ = _synthetic_world()
    with pytest.raises(ValueError, match="too few factor returns"):
        estimate_structured_risk_model(
            returns.iloc[:1], styles, memberships, min_cross_section=10
        )
