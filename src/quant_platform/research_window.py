"""Immutable dataset, information-timing, and evaluation-window contracts.

``ResearchHorizonContract`` describes a product horizon.  This module binds
that abstract profile to one concrete dataset, feature set, trading calendar,
and train/validation/sealed-OOS split.  The resulting digest is the research
input identity used by automation; dates alone are not a sufficient contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from .cost_model import CN_COST_SCHEDULE_BOOK
from .model_research_governance import MODEL_LABEL_HORIZON_TRADING_DAYS
from .research_horizon import (
    LEGACY_AMBIGUOUS,
    canonical_sha256,
    research_horizon_contract,
)

RESEARCH_WINDOW_CONTRACT_VERSION = "research-window-v1"
DAILY_CLOSE_SIGNAL_SEMANTICS = "complete_daily_bar_after_exchange_close"
NEXT_SESSION_EXECUTION_SEMANTICS = "next_trading_session_open_or_conservative_daily_fill"
DEFAULT_RESEARCH_UNIVERSE = "cn_all_governed_ashare_and_etf"
DEFAULT_RESEARCH_SEED = 42


def _sha256(value: object, *, field: str) -> str:
    normalized = str(value or "").lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def _periods(value: Mapping[str, Any]) -> dict[str, str]:
    keys = (
        "train_start",
        "train_end",
        "valid_start",
        "valid_end",
        "test_start",
        "test_end",
    )
    normalized = {key: str(value.get(key) or "") for key in keys}
    try:
        parsed = {key: date.fromisoformat(item) for key, item in normalized.items()}
    except ValueError as exc:
        raise ValueError("research window periods must contain ISO dates") from exc
    if not (
        parsed["train_start"]
        <= parsed["train_end"]
        < parsed["valid_start"]
        <= parsed["valid_end"]
        < parsed["test_start"]
        <= parsed["test_end"]
    ):
        raise ValueError("research window periods are not ordered")
    return normalized


@dataclass(frozen=True)
class ResearchWindowContract:
    """One deterministic, replayable research input contract."""

    horizon_profile: str
    horizon_contract_sha256: str
    dataset_name: str
    dataset_identity_sha256: str
    dataset_lineage_id: str | None
    dataset_contract_sha256: str
    field_contract_version: str
    field_coverage_sha256: str
    feature_set_id: str | None
    feature_set_sha256: str | None
    calendar_start: str
    calendar_end: str
    data_cutoff_session: str
    signal_time_semantics: str
    earliest_execution_semantics: str
    execution_lag_sessions: int
    label_horizons_sessions: tuple[int, ...]
    purge_sessions: int
    embargo_sessions: int
    label_maturity_enforced: bool
    label_maturity_tail_sessions: int
    latest_mature_label_sessions: tuple[tuple[int, str], ...]
    periods: dict[str, str]
    sealed_oos_sessions: int
    universe: str
    cost_schedule_versions: tuple[str, ...]
    cost_schedule_sha256: str
    execution_contract: str
    random_seed: int
    contract_version: str = RESEARCH_WINDOW_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != RESEARCH_WINDOW_CONTRACT_VERSION:
            raise ValueError("research window contract version is unsupported")
        horizon = research_horizon_contract(self.horizon_profile)
        if self.horizon_contract_sha256 != horizon.sha256:
            raise ValueError("research window horizon digest is invalid")
        if not self.dataset_name.strip():
            raise ValueError("research window dataset name is required")
        if not self.field_contract_version.strip():
            raise ValueError("research window field contract version is required")
        _sha256(self.dataset_identity_sha256, field="dataset_identity_sha256")
        _sha256(self.dataset_contract_sha256, field="dataset_contract_sha256")
        _sha256(self.field_coverage_sha256, field="field_coverage_sha256")
        if self.dataset_lineage_id is not None:
            _sha256(self.dataset_lineage_id, field="dataset_lineage_id")
        if self.feature_set_sha256 is not None:
            _sha256(self.feature_set_sha256, field="feature_set_sha256")
        if bool(self.feature_set_id) != bool(self.feature_set_sha256):
            raise ValueError("research window feature-set id and digest must be bound together")
        calendar_start = date.fromisoformat(self.calendar_start)
        calendar_end = date.fromisoformat(self.calendar_end)
        data_cutoff = date.fromisoformat(self.data_cutoff_session)
        if not calendar_start <= data_cutoff == calendar_end:
            raise ValueError("research window data cutoff must equal the calendar end")
        labels = self.label_horizons_sessions
        if not labels or labels != tuple(sorted(set(labels))) or min(labels) < 1:
            raise ValueError("research window labels must be unique positive sessions")
        if (
            horizon.horizon_profile != LEGACY_AMBIGUOUS
            and labels != horizon.label_horizons_sessions
        ):
            raise ValueError("research window labels differ from the horizon contract")
        if min(
            self.execution_lag_sessions,
            self.purge_sessions,
            self.embargo_sessions,
            self.sealed_oos_sessions,
        ) < 1:
            raise ValueError("research timing durations must be positive")
        if self.purge_sessions < max(labels) or self.embargo_sessions < max(labels):
            raise ValueError("research purge and embargo must cover the longest label")
        if self.label_maturity_enforced and self.label_maturity_tail_sessions < max(labels):
            raise ValueError("research window does not reserve the longest label maturity tail")
        if not self.label_maturity_enforced and horizon.horizon_profile != LEGACY_AMBIGUOUS:
            raise ValueError("active horizon research must enforce label maturity")
        maturity = dict(self.latest_mature_label_sessions)
        if tuple(sorted(maturity)) != labels:
            raise ValueError("research window maturity dates do not cover every label")
        normalized_periods = _periods(self.periods)
        object.__setattr__(self, "periods", normalized_periods)
        if self.label_maturity_enforced and normalized_periods["test_end"] != maturity[max(labels)]:
            raise ValueError("sealed OOS ends after the longest label maturity cutoff")
        if self.signal_time_semantics != DAILY_CLOSE_SIGNAL_SEMANTICS:
            raise ValueError("research window signal semantics are unsupported")
        if self.earliest_execution_semantics != NEXT_SESSION_EXECUTION_SEMANTICS:
            raise ValueError("research window execution semantics are unsupported")
        if not self.universe.strip() or not self.execution_contract.strip():
            raise ValueError("research universe and execution contract are required")
        if not self.cost_schedule_versions:
            raise ValueError("research window cost schedule is required")
        _sha256(self.cost_schedule_sha256, field="cost_schedule_sha256")
        if isinstance(self.random_seed, bool) or self.random_seed < 0:
            raise ValueError("research random seed must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["label_horizons_sessions"] = list(self.label_horizons_sessions)
        value["latest_mature_label_sessions"] = {
            str(label): session for label, session in self.latest_mature_label_sessions
        }
        value["cost_schedule_versions"] = list(self.cost_schedule_versions)
        return value

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _field_coverage_identity(provenance: Mapping[str, Any]) -> str:
    """Bind the declared field/range coverage without fabricating missing years."""

    supplied = provenance.get("field_coverage_sha256")
    if supplied:
        expected = _sha256(supplied, field="field_coverage_sha256")
        matrix = provenance.get("field_year_coverage")
        if not isinstance(matrix, Mapping):
            raise ValueError("dataset provenance has no field-year coverage matrix")
        matrix_value = dict(matrix)
        embedded = _sha256(
            matrix_value.pop("coverage_sha256", ""),
            field="field_year_coverage.coverage_sha256",
        )
        if canonical_sha256(matrix_value) != embedded or embedded != expected:
            raise ValueError("dataset field-year coverage digest is invalid")
        return expected
    coverage = {
        "version": "dataset-declared-field-coverage-v1",
        "fields": sorted(str(item) for item in (provenance.get("fields") or [])),
        "field_units": dict(provenance.get("field_units") or {}),
        "research_features": dict(provenance.get("research_features") or {}),
        "field_year_coverage": provenance.get("field_year_coverage"),
        "source_start_date": provenance.get("source_start_date"),
        "source_end_date": provenance.get("source_end_date"),
    }
    if not coverage["fields"] or not coverage["source_start_date"]:
        raise ValueError("dataset provenance has no governed field coverage")
    return canonical_sha256(coverage)


def resolve_required_field_coverage(
    provenance: Mapping[str, Any],
    required_fields: Sequence[str],
    *,
    data_cutoff_session: str,
) -> dict[str, Any]:
    """Resolve the first continuously usable session for selected factor fields.

    Active three-horizon research must use normalized row evidence.  A field
    merely appearing in the Qlib schema is insufficient, and a pre-2016 market
    field is accepted only when the dataset builder recorded an admitted legacy
    overlap contract in ``research_available_from``.
    """

    coverage_sha256 = _field_coverage_identity(provenance)
    matrix = provenance.get("field_year_coverage")
    if not isinstance(matrix, Mapping) or matrix.get("version") != (
        "qlib-field-year-source-coverage-v1"
    ):
        raise ValueError("dataset has no supported field-year coverage matrix")
    if matrix.get("evidence_status") == "missing_normalized_staging":
        raise ValueError("dataset field-year coverage has no normalized staging evidence")
    entries = matrix.get("fields")
    if not isinstance(entries, Mapping):
        raise ValueError("dataset field-year coverage fields are invalid")
    normalized = tuple(sorted({str(item).strip() for item in required_fields if str(item).strip()}))
    if not normalized:
        raise ValueError("governed feature set does not reference any dataset fields")
    starts: dict[str, str] = {}
    sources: dict[str, list[str]] = {}
    cutoff = date.fromisoformat(data_cutoff_session)
    for field in normalized:
        entry = entries.get(field)
        if not isinstance(entry, Mapping):
            raise ValueError(f"dataset has no field-year coverage for {field}")
        raw_start = str(entry.get("research_available_from") or "")
        raw_end = str(entry.get("available_to") or "")
        try:
            field_start = date.fromisoformat(raw_start)
            field_end = date.fromisoformat(raw_end)
        except ValueError as exc:
            raise ValueError(f"dataset field {field} has no usable continuous coverage") from exc
        if field_end < cutoff:
            raise ValueError(
                f"dataset field {field} coverage ends before the data cutoff session"
            )
        starts[field] = field_start.isoformat()
        field_sources = {
            str(source)
            for year in (entry.get("years") or [])
            if isinstance(year, Mapping)
            for source in (year.get("source_contracts") or [])
            if str(source)
        }
        sources[field] = sorted(field_sources)
    effective = max(date.fromisoformat(value) for value in starts.values())
    return {
        "coverage_version": str(matrix["version"]),
        "field_coverage_sha256": coverage_sha256,
        "required_fields": list(normalized),
        "required_field_available_from": starts,
        "required_field_sources": sources,
        "effective_field_start_session": effective.isoformat(),
        "data_cutoff_session": cutoff.isoformat(),
    }


def build_research_window_contract(
    *,
    dataset: Mapping[str, Any],
    calendar_days: Sequence[str],
    periods: Mapping[str, Any],
    period_resolution: Mapping[str, Any],
    horizon_profile: str = LEGACY_AMBIGUOUS,
    feature_set: Mapping[str, Any] | None = None,
    universe: str = DEFAULT_RESEARCH_UNIVERSE,
    random_seed: int = DEFAULT_RESEARCH_SEED,
) -> ResearchWindowContract:
    """Bind a resolved research split to immutable dataset and timing evidence."""

    ordered = sorted(dict.fromkeys(str(day).strip() for day in calendar_days if str(day).strip()))
    if not ordered:
        raise ValueError("research window calendar is empty")
    provenance = dataset.get("provenance") or {}
    if not isinstance(provenance, Mapping):
        raise ValueError("dataset provenance must be an object")
    horizon = research_horizon_contract(horizon_profile)
    if horizon.horizon_profile == LEGACY_AMBIGUOUS:
        labels = (MODEL_LABEL_HORIZON_TRADING_DAYS,)
        purge = int(period_resolution.get("purge_trading_days") or max(labels))
        embargo = int(period_resolution.get("embargo_trading_days") or 20)
        lag = 1
    else:
        labels = horizon.label_horizons_sessions
        purge = int(period_resolution.get("purge_trading_days") or horizon.purge_sessions or 0)
        embargo = int(
            period_resolution.get("embargo_trading_days") or horizon.embargo_sessions or 0
        )
        lag = int(horizon.execution_lag_sessions or 0)
    maturity_raw = period_resolution.get("latest_mature_label_sessions") or {}
    if not isinstance(maturity_raw, Mapping):
        raise ValueError("research period resolution has invalid label maturity evidence")
    maturity = tuple(
        (label, str(maturity_raw.get(str(label)) or maturity_raw.get(label) or ""))
        for label in labels
    )
    normalized_periods = _periods(periods)
    final_oos_sessions = sum(
        normalized_periods["test_start"] <= day <= normalized_periods["test_end"]
        for day in ordered
    )
    feature_id = str((feature_set or {}).get("id") or "") or None
    feature_sha = str((feature_set or {}).get("definition_sha256") or "") or None
    lineage = str(dataset.get("lineage_id") or provenance.get("dataset_lineage_id") or "") or None
    versions = tuple(item.version for item in CN_COST_SCHEDULE_BOOK.versions)
    cost_schedule_sha256 = canonical_sha256(
        [asdict(item) for item in CN_COST_SCHEDULE_BOOK.versions]
    )
    return ResearchWindowContract(
        horizon_profile=horizon.horizon_profile,
        horizon_contract_sha256=horizon.sha256,
        dataset_name=str(dataset.get("name") or ""),
        dataset_identity_sha256=str(provenance.get("dataset_identity_sha256") or ""),
        dataset_lineage_id=lineage,
        dataset_contract_sha256=str(provenance.get("dataset_contract_sha256") or ""),
        field_contract_version=str(provenance.get("field_contract_version") or ""),
        field_coverage_sha256=_field_coverage_identity(provenance),
        feature_set_id=feature_id,
        feature_set_sha256=feature_sha,
        calendar_start=ordered[0],
        calendar_end=ordered[-1],
        data_cutoff_session=ordered[-1],
        signal_time_semantics=DAILY_CLOSE_SIGNAL_SEMANTICS,
        earliest_execution_semantics=NEXT_SESSION_EXECUTION_SEMANTICS,
        execution_lag_sessions=lag,
        label_horizons_sessions=labels,
        purge_sessions=purge,
        embargo_sessions=embargo,
        label_maturity_enforced=horizon.horizon_profile != LEGACY_AMBIGUOUS,
        label_maturity_tail_sessions=int(
            period_resolution.get("label_maturity_tail_trading_days") or 0
        ),
        latest_mature_label_sessions=maturity,
        periods=normalized_periods,
        sealed_oos_sessions=final_oos_sessions,
        universe=universe,
        cost_schedule_versions=versions,
        cost_schedule_sha256=cost_schedule_sha256,
        execution_contract="daily_close_signal_d_plus_1_open_t_plus_1_lot100_v1",
        random_seed=int(random_seed),
    )
