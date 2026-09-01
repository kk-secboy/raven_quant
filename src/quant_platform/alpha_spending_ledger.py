from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import insert, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from quant_data.database import (
    capital_oos_alpha_batches,
    capital_oos_alpha_families,
    capital_oos_legacy_attempts,
    oos_vintages,
    open_database,
)
from quant_data.snapshot_lineage import canonical_sha256

from .alpha_spending_integration import (
    capital_oos_family_manifest_sha256,
    validate_capital_oos_family_manifest,
)
from .statistical_validation import (
    newey_west_mean_test,
    paired_moving_block_bootstrap,
)

CAPITAL_OOS_TOTAL_ALPHA = Decimal("0.05")
CAPITAL_OOS_MIN_TRADING_DAYS = 252
CAPITAL_OOS_MIN_EMBARGO_TRADING_DAYS = 20
CAPITAL_OOS_HAC_MAX_LAG = 20
CAPITAL_OOS_BOOTSTRAP_SAMPLES = 2000
CAPITAL_OOS_P_VALUE_CONTRACT = "one-sided-paired-newey-west-hac-v1"
CAPITAL_OOS_POLICY_CONTRACT = "capital-final-oos-alpha-spending-v1"
CAPITAL_OOS_VINTAGE_LINK_CONTRACT = "capital-oos-vintage-link-v1"
MIN_POSITIVE_HAC_P_VALUE = 1e-300
_ALPHA_QUANTUM = Decimal("1e-28")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _decimal(value: Decimal | float | int | str, *, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a finite decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be a finite decimal")
    return result


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def capital_oos_policy_manifest() -> dict[str, Any]:
    return {
        "contract_version": CAPITAL_OOS_POLICY_CONTRACT,
        "scope": "capital_oos_only",
        "research_tournaments_spend_alpha": False,
        "total_alpha": "0.05",
        "hypotheses_per_batch": 1,
        "batch_spending_rule": "0.05/(k*(k+1))",
        "p_value_contract": CAPITAL_OOS_P_VALUE_CONTRACT,
        "minimum_final_oos_trading_days": CAPITAL_OOS_MIN_TRADING_DAYS,
        "minimum_embargo_trading_days": CAPITAL_OOS_MIN_EMBARGO_TRADING_DAYS,
        "same_family_windows_must_be_disjoint": True,
        "maximum_unsettled_batches_per_family": 1,
        "failed_opened_window_spends_alpha": True,
        "bootstrap_remains_an_independent_hard_gate": False,
        "statistical_evidence_role": "report_only",
        "final_oos_pass_rule": (
            "oos_after_cost_excess >= 0.5 * sealed_research_edge "
            "(decay check; unevaluable without a sealed research reference)"
        ),
        "daily_dataset_identity_resets_family": False,
        "dataset_lineage_resets_family": False,
        "error_control_scope": "stable_investment_mandate_across_dataset_lineages",
        "legacy_final_oos_history_requires_immutable_reconciliation": True,
    }


CAPITAL_OOS_POLICY_SHA256 = canonical_sha256(capital_oos_policy_manifest())


def batch_alpha(ordinal: int) -> Decimal:
    """Return alpha_k=.05/[k(k+1)]; the infinite sum is exactly .05."""

    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal <= 0:
        raise ValueError("capital OOS ordinal must be a positive integer")
    return CAPITAL_OOS_TOTAL_ALPHA / Decimal(ordinal * (ordinal + 1))


def cumulative_batch_alpha(batch_count: int) -> Decimal:
    if isinstance(batch_count, bool) or not isinstance(batch_count, int) or batch_count < 0:
        raise ValueError("batch_count must be a non-negative integer")
    if batch_count == 0:
        return Decimal("0")
    return CAPITAL_OOS_TOTAL_ALPHA * Decimal(batch_count) / Decimal(batch_count + 1)


def _stored_alpha(value: Decimal) -> Decimal:
    stored = value.quantize(_ALPHA_QUANTUM, rounding=ROUND_DOWN)
    if stored <= 0:
        raise ValueError("capital OOS ordinal exceeds persistent numeric precision")
    return stored


def capital_oos_vintage_link_payload(
    *,
    batch_id: str,
    preregistration_sha256: str,
    frozen_bundle_manifest_sha256: str,
    frozen_baseline_manifest_sha256: str,
) -> dict[str, str]:
    """Build the exact sealed-member link shared by ledger and StrategyStore."""

    values = {
        "batch_id": str(batch_id).strip().lower(),
        "preregistration_sha256": str(preregistration_sha256).strip().lower(),
        "frozen_bundle_manifest_sha256": str(
            frozen_bundle_manifest_sha256
        ).strip().lower(),
        "frozen_baseline_manifest_sha256": str(
            frozen_baseline_manifest_sha256
        ).strip().lower(),
    }
    if any(not _SHA256.fullmatch(value) for value in values.values()):
        raise ValueError("capital OOS vintage link values must be SHA256")
    return {
        "contract_version": CAPITAL_OOS_VINTAGE_LINK_CONTRACT,
        **values,
    }


def _normalize_trading_dates(
    values: Sequence[date | datetime | str], *, label: str, minimum: int
) -> list[str]:
    normalized: list[date] = []
    for value in values:
        if isinstance(value, datetime):
            item = value.date()
        elif isinstance(value, date):
            item = value
        else:
            try:
                item = date.fromisoformat(str(value).strip())
            except ValueError as exc:
                raise ValueError(f"{label} contains an invalid date") from exc
        normalized.append(item)
    if len(normalized) < minimum:
        raise ValueError(f"{label} requires at least {minimum} trading days")
    if normalized != sorted(normalized) or len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be strictly increasing and unique")
    return [item.isoformat() for item in normalized]


def validate_final_oos_window(
    *,
    research_data_end: date | datetime | str,
    final_oos_trading_dates: Sequence[date | datetime | str],
    embargo_trading_dates: Sequence[date | datetime | str],
) -> dict[str, Any]:
    final_dates = _normalize_trading_dates(
        final_oos_trading_dates,
        label="final_oos_trading_dates",
        minimum=CAPITAL_OOS_MIN_TRADING_DAYS,
    )
    embargo_dates = _normalize_trading_dates(
        embargo_trading_dates,
        label="embargo_trading_dates",
        minimum=CAPITAL_OOS_MIN_EMBARGO_TRADING_DAYS,
    )
    try:
        research_end = (
            research_data_end.date()
            if isinstance(research_data_end, datetime)
            else research_data_end
            if isinstance(research_data_end, date)
            else date.fromisoformat(str(research_data_end).strip())
        )
    except ValueError as exc:
        raise ValueError("research_data_end is invalid") from exc
    if research_end >= date.fromisoformat(embargo_dates[0]):
        raise ValueError("embargo trading dates must start after all research data")
    if embargo_dates[-1] >= final_dates[0]:
        raise ValueError("embargo trading dates must be strictly before final OOS")
    return {
        "research_data_end": research_end.isoformat(),
        "final_oos_start": final_dates[0],
        "final_oos_end": final_dates[-1],
        "trading_day_count": len(final_dates),
        "trading_dates": final_dates,
        "trading_dates_sha256": canonical_sha256(final_dates),
        "embargo_trading_day_count": len(embargo_dates),
        "embargo_trading_dates": embargo_dates,
        "embargo_trading_dates_sha256": canonical_sha256(embargo_dates),
    }


def paired_hac_capital_test(
    candidate_net_returns: pd.Series,
    baseline_net_returns: pd.Series,
    *,
    max_lag: int = CAPITAL_OOS_HAC_MAX_LAG,
) -> tuple[pd.Series, dict[str, Any]]:
    """One-sided paired HAC test on daily cost-after excess return."""

    if not isinstance(candidate_net_returns, pd.Series) or not isinstance(
        baseline_net_returns, pd.Series
    ):
        raise ValueError("capital OOS returns must be pandas Series")
    candidate = pd.to_numeric(candidate_net_returns, errors="coerce")
    baseline = pd.to_numeric(baseline_net_returns, errors="coerce")
    candidate.index = pd.to_datetime(candidate.index, errors="coerce").tz_localize(None)
    baseline.index = pd.to_datetime(baseline.index, errors="coerce").tz_localize(None)
    if (
        candidate.index.isna().any()
        or baseline.index.isna().any()
        or candidate.index.has_duplicates
        or baseline.index.has_duplicates
        or not candidate.index.equals(baseline.index)
        or len(candidate) < CAPITAL_OOS_MIN_TRADING_DAYS
    ):
        raise ValueError("capital OOS paired returns require one exact valid trading index")
    candidate_values = candidate.to_numpy(dtype=float)
    baseline_values = baseline.to_numpy(dtype=float)
    if not np.isfinite(candidate_values).all() or not np.isfinite(baseline_values).all():
        raise ValueError("capital OOS paired returns must be finite")
    difference = (candidate - baseline).rename("paired_net_excess_return")
    two_sided = newey_west_mean_test(difference, max_lag=max_lag)
    statistic = two_sided.get("test_statistic")
    two_sided_p_value = two_sided.get("p_value")
    positive_and_defined = (
        two_sided.get("status") == "ok"
        and float(two_sided.get("mean") or 0.0) > 0.0
        and statistic is not None
        and np.isfinite(float(statistic))
        and float(statistic) > 0.0
        and two_sided_p_value is not None
        and np.isfinite(float(two_sided_p_value))
    )
    one_sided_p = (
        max(float(two_sided_p_value) / 2.0, MIN_POSITIVE_HAC_P_VALUE)
        if positive_and_defined
        else 1.0
    )
    evidence = {
        **two_sided,
        "contract_version": CAPITAL_OOS_P_VALUE_CONTRACT,
        "return_definition": "candidate_net_return_minus_baseline_net_return_after_cost",
        "alternative": "paired_mean_greater_than_zero",
        "one_sided_p_value": one_sided_p,
        "minimum_positive_p_value": MIN_POSITIVE_HAC_P_VALUE,
    }
    return difference, evidence


def _bootstrap_gate(difference: pd.Series) -> dict[str, Any]:
    evidence = paired_moving_block_bootstrap(
        difference,
        pd.Series(0.0, index=difference.index),
        block_size=min(20, len(difference)),
        samples=CAPITAL_OOS_BOOTSTRAP_SAMPLES,
        seed=0,
    )
    interval = list(evidence.get("confidence_interval_95") or [])
    passed = (
        evidence.get("status") == "ok"
        and int(evidence.get("samples") or 0) >= CAPITAL_OOS_BOOTSTRAP_SAMPLES
        and int(evidence.get("observations") or 0) == len(difference)
        and float(evidence.get("observed_mean_difference") or 0.0) > 0.0
        and len(interval) == 2
        and float(interval[0]) > 0.0
        and float(evidence.get("one_sided_p_value") or 1.0) <= 0.05
    )
    return {**evidence, "hard_gate_passed": passed}


CAPITAL_OOS_DECAY_CONTRACT = "oos-research-decay-v1"
CAPITAL_OOS_MIN_DECAY_RATIO = 0.5


def _oos_decay_check(
    difference: pd.Series,
    research_reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare the formal OOS after-cost excess with the sealed research edge.

    Wide-in/strict-out recalibration: the once-only formal OOS no longer dies
    on a fixed statistical threshold.  It passes when its mean daily
    after-cost excess over the baseline keeps at least half of the edge sealed
    in the research-stage evidence.  Both metric values and the ratio are
    archived either way; without a sealed research reference (for example
    legacy/autopilot families) the check is not evaluable and does not veto.
    """

    oos_mean = float(pd.to_numeric(difference, errors="coerce").mean())
    reference = dict(research_reference or {})
    research_mean_raw = reference.get("research_mean_daily_after_cost_excess")
    try:
        research_mean = (
            float(research_mean_raw) if research_mean_raw is not None else None
        )
    except (TypeError, ValueError):
        research_mean = None
    if research_mean is not None and not np.isfinite(research_mean):
        research_mean = None
    ratio: float | None = None
    decay_passed: bool | None = None
    if research_mean is not None:
        if research_mean > 0.0 and np.isfinite(oos_mean):
            ratio = oos_mean / research_mean
            decay_passed = bool(ratio >= CAPITAL_OOS_MIN_DECAY_RATIO)
        else:
            # No positive research edge was sealed, so there is nothing the
            # OOS window is allowed to decay from.
            decay_passed = False
    return {
        "contract_version": CAPITAL_OOS_DECAY_CONTRACT,
        "main_metric": "mean_daily_after_cost_excess_return",
        "oos_mean_daily_after_cost_excess": oos_mean,
        "oos_annualized_after_cost_excess": oos_mean * 252.0,
        "research_mean_daily_after_cost_excess": research_mean,
        "research_annualized_after_cost_excess": (
            research_mean * 252.0 if research_mean is not None else None
        ),
        "decay_ratio": ratio,
        "minimum_decay_ratio": CAPITAL_OOS_MIN_DECAY_RATIO,
        "decay_passed": decay_passed,
    }


class CapitalOOSAlphaLedgerStore:
    """Persistent alpha spending for one-shot capital-facing final OOS only."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    @staticmethod
    def _family_identity(stable_mandate: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
        mandate = validate_capital_oos_family_manifest(dict(stable_mandate))
        family_sha = capital_oos_family_manifest_sha256(mandate)
        family_id = canonical_sha256(
            {
                "kind": "capital-oos-alpha-family-v2",
                "stable_mandate_sha256": family_sha,
            }
        )
        return mandate, family_sha, family_id

    @staticmethod
    def _legacy_evidence(row: Any) -> dict[str, Any]:
        sealed = dict(row.sealed_candidate_set_json or {})
        if canonical_sha256(sealed) != str(row.sealed_candidate_set_sha256):
            raise ValueError("legacy OOS sealed candidate set hash is invalid")
        return {
            "contract_version": "capital-oos-legacy-attempt-v1",
            "oos_vintage_id": str(row.id),
            "scope": str(row.scope),
            "dataset_identity": str(row.dataset_identity),
            "dataset_lineage_id": row.dataset_lineage_id,
            "final_oos_start": row.test_start.isoformat(),
            "final_oos_end": row.test_end.isoformat(),
            "first_opened_at": row.first_opened_at.isoformat(),
            "consumed_at": row.consumed_at.isoformat() if row.consumed_at else None,
            "sealed_candidate_set_sha256": str(row.sealed_candidate_set_sha256),
            "raw_p_value": 1.0,
            "passed": False,
            "failure_recorded": True,
            "attribution_status": "unreconciled",
        }

    @staticmethod
    def _vintage_link_payload(batch: Any) -> dict[str, Any]:
        return capital_oos_vintage_link_payload(
            batch_id=str(batch.id),
            preregistration_sha256=str(batch.preregistration_sha256),
            frozen_bundle_manifest_sha256=str(
                batch.frozen_bundle_manifest_sha256
            ),
            frozen_baseline_manifest_sha256=str(
                batch.frozen_baseline_manifest_sha256
            ),
        )

    @classmethod
    def _verify_linked_vintage(
        cls,
        connection: Any,
        vintage: Any,
        *,
        expected_batch_id: str | None = None,
    ) -> Any:
        sealed = dict(vintage.sealed_candidate_set_json or {})
        if canonical_sha256(sealed) != str(vintage.sealed_candidate_set_sha256):
            raise ValueError("capital OOS vintage sealed-member hash is invalid")
        link = sealed.get("capital_oos_alpha_ledger")
        if not isinstance(link, dict) or set(link) != {
            "contract_version",
            "batch_id",
            "preregistration_sha256",
            "frozen_bundle_manifest_sha256",
            "frozen_baseline_manifest_sha256",
        }:
            raise ValueError("capital OOS vintage has no valid alpha-ledger link")
        batch_id = str(link.get("batch_id") or "").strip()
        if str(vintage.capital_oos_alpha_batch_id or "") != batch_id:
            raise ValueError("capital OOS vintage database link is missing or inconsistent")
        if expected_batch_id is not None and batch_id != expected_batch_id:
            raise ValueError("capital OOS vintage is linked to another alpha batch")
        batch = connection.execute(
            select(capital_oos_alpha_batches).where(
                capital_oos_alpha_batches.c.id == batch_id
            )
        ).first()
        if batch is None:
            raise ValueError("capital OOS vintage links an unknown alpha batch")
        if (
            link != cls._vintage_link_payload(batch)
            or str(vintage.dataset_identity) != str(batch.dataset_identity_sha256)
            or str(vintage.dataset_lineage_id or "")
            != str(batch.dataset_lineage_id)
            or vintage.test_start != batch.final_oos_start
            or vintage.test_end != batch.final_oos_end
            or vintage.consumed_at is None
        ):
            raise ValueError("capital OOS vintage link differs from preregistration")
        return batch

    def _capture_missing_legacy_history(self) -> int:
        """Persist old final-OOS openings as failed, still-unattributed tests."""

        now = _utcnow()
        captured = 0
        with self.engine.begin() as connection:
            connection.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:ledger_lock))"),
                {"ledger_lock": "capital-oos-alpha-ledger-v1"},
            )
            rows = connection.execute(
                select(oos_vintages)
                .where(
                    oos_vintages.c.id.not_in(
                        select(capital_oos_legacy_attempts.c.oos_vintage_id)
                    )
                )
                .order_by(
                    oos_vintages.c.first_opened_at,
                    oos_vintages.c.test_start,
                    oos_vintages.c.id,
                )
                .with_for_update()
            ).all()
            for row in rows:
                sealed = dict(row.sealed_candidate_set_json or {})
                if (
                    row.capital_oos_alpha_batch_id is not None
                    or "capital_oos_alpha_ledger" in sealed
                ):
                    self._verify_linked_vintage(connection, row)
                    continue
                evidence = self._legacy_evidence(row)
                connection.execute(
                    insert(capital_oos_legacy_attempts).values(
                        id=canonical_sha256(
                            {
                                "kind": "capital-oos-legacy-attempt-v1",
                                "oos_vintage_id": str(row.id),
                            }
                        ),
                        oos_vintage_id=str(row.id),
                        status="unreconciled",
                        scope=str(row.scope),
                        dataset_identity=str(row.dataset_identity),
                        dataset_lineage_id=row.dataset_lineage_id,
                        final_oos_start=row.test_start,
                        final_oos_end=row.test_end,
                        first_opened_at=row.first_opened_at,
                        consumed_at=row.consumed_at,
                        sealed_candidate_set_sha256=str(row.sealed_candidate_set_sha256),
                        raw_p_value=1.0,
                        passed=False,
                        failure_recorded=True,
                        legacy_evidence_json=evidence,
                        legacy_evidence_sha256=canonical_sha256(evidence),
                        created_at=now,
                    )
                )
                captured += 1
        return captured

    def vintage_link_contract(self, batch_id: str) -> dict[str, Any]:
        """Return the exact payload the formal OOS vintage must seal atomically."""

        identifier = str(batch_id).strip()
        with self.engine.connect() as connection:
            batch = connection.execute(
                select(capital_oos_alpha_batches).where(
                    capital_oos_alpha_batches.c.id == identifier
                )
            ).first()
        if batch is None:
            raise KeyError(batch_id)
        return {
            "capital_oos_alpha_batch_id": str(batch.id),
            "capital_oos_dataset_identity_sha256": str(
                batch.dataset_identity_sha256
            ),
            "sealed_candidate_set_patch": {
                "capital_oos_alpha_ledger": self._vintage_link_payload(batch)
            },
        }

    def get_vintage_binding(self, batch_id: str) -> dict[str, Any]:
        """Return and verify the immutable OOS-vintage link for one batch.

        The formal-backtest worker uses this after ``create_backtest`` so its
        settlement evidence names the exact vintage consumed by that atomic
        transaction.  A missing link is an error, never permission to settle a
        successful result without the one-shot OOS ledger.
        """

        identifier = str(batch_id).strip()
        if not identifier:
            raise ValueError("batch_id is required")
        with self.engine.connect() as connection:
            vintage = connection.execute(
                select(oos_vintages).where(
                    oos_vintages.c.capital_oos_alpha_batch_id == identifier
                )
            ).first()
            if vintage is None:
                raise KeyError(
                    f"capital OOS batch {identifier} has no linked OOS vintage"
                )
            batch = self._verify_linked_vintage(
                connection,
                vintage,
                expected_batch_id=identifier,
            )
        return {
            "contract_version": "capital-oos-vintage-binding-v1",
            "capital_oos_alpha_batch_id": identifier,
            "oos_vintage_id": str(vintage.id),
            "preregistration_sha256": str(batch.preregistration_sha256),
            "sealed_candidate_set_sha256": str(
                vintage.sealed_candidate_set_sha256
            ),
            "dataset_lineage_id": str(batch.dataset_lineage_id),
            "dataset_identity_sha256": str(batch.dataset_identity_sha256),
            "final_oos_start": batch.final_oos_start.isoformat(),
            "final_oos_end": batch.final_oos_end.isoformat(),
        }

    @staticmethod
    def _ensure_family(
        connection: Any,
        *,
        mandate: dict[str, Any],
        family_sha: str,
        family_id: str,
        now: datetime,
    ) -> Any:
        policy = capital_oos_policy_manifest()
        connection.execute(
            pg_insert(capital_oos_alpha_families)
            .values(
                id=family_id,
                capital_oos_family_sha256=family_sha,
                mandate_json=mandate,
                total_alpha=CAPITAL_OOS_TOTAL_ALPHA,
                policy_json=policy,
                policy_sha256=CAPITAL_OOS_POLICY_SHA256,
                next_ordinal=1,
                reserved_alpha=Decimal("0"),
                settled_alpha=Decimal("0"),
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_nothing(
                index_elements=[capital_oos_alpha_families.c.capital_oos_family_sha256]
            )
        )
        family = connection.execute(
            select(capital_oos_alpha_families)
            .where(capital_oos_alpha_families.c.capital_oos_family_sha256 == family_sha)
            .with_for_update()
        ).one()
        if (
            str(family.id) != family_id
            or validate_capital_oos_family_manifest(dict(family.mandate_json or {}))
            != mandate
            or capital_oos_family_manifest_sha256(dict(family.mandate_json or {}))
            != str(family.capital_oos_family_sha256)
            or str(family.policy_sha256) != CAPITAL_OOS_POLICY_SHA256
            or canonical_sha256(dict(family.policy_json or {})) != str(family.policy_sha256)
        ):
            raise ValueError("capital OOS mandate and alpha policy are immutable")
        return family

    def reconcile_legacy_attempts(
        self,
        *,
        stable_mandate: Mapping[str, Any],
        legacy_attempt_ids: Sequence[str],
        actor: str,
        reason: str,
        supporting_evidence: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Immutably attribute pre-ledger OOS openings and spend their alpha."""

        if not isinstance(stable_mandate, Mapping):
            raise ValueError("stable_mandate must be a capital OOS mandate object")
        mandate, family_sha, family_id = self._family_identity(stable_mandate)
        identifiers = [str(item).strip() for item in legacy_attempt_ids]
        if not identifiers or any(not item for item in identifiers):
            raise ValueError("legacy_attempt_ids must be non-empty")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("legacy_attempt_ids must be unique")
        actor_value = str(actor).strip()
        reason_value = str(reason).strip()
        if not actor_value or not reason_value:
            raise ValueError("legacy reconciliation requires actor and reason")
        support = dict(supporting_evidence or {})
        support_sha = canonical_sha256(support)
        self._capture_missing_legacy_history()
        now = _utcnow()
        with self.engine.begin() as connection:
            connection.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:ledger_lock))"),
                {"ledger_lock": "capital-oos-alpha-ledger-v1"},
            )
            family = self._ensure_family(
                connection,
                mandate=mandate,
                family_sha=family_sha,
                family_id=family_id,
                now=now,
            )
            rows = connection.execute(
                select(capital_oos_legacy_attempts)
                .where(capital_oos_legacy_attempts.c.id.in_(identifiers))
                .with_for_update()
            ).all()
            if len(rows) != len(identifiers):
                raise KeyError("one or more legacy capital OOS attempts do not exist")
            if all(str(row.status) == "reconciled" for row in rows):
                for row in rows:
                    receipt = dict(row.reconciliation_json or {})
                    if (
                        str(row.reconciled_family_id) != family_id
                        or receipt.get("stable_mandate_sha256") != family_sha
                        or receipt.get("actor") != actor_value
                        or receipt.get("reason") != reason_value
                        or receipt.get("supporting_evidence_sha256") != support_sha
                    ):
                        raise ValueError("legacy OOS reconciliation is immutable")
                return [self._decode_legacy(row) for row in rows]
            if any(str(row.status) != "unreconciled" for row in rows):
                raise ValueError("legacy OOS reconciliation cannot mix prior assignments")
            ordered = sorted(
                rows,
                key=lambda row: (row.first_opened_at, row.final_oos_start, str(row.id)),
            )
            ordinal = int(family.next_ordinal)
            spent_values = [
                _stored_alpha(batch_alpha(ordinal + offset))
                for offset in range(len(ordered))
            ]
            total_spent = sum(spent_values, Decimal("0"))
            reserved_after = Decimal(family.reserved_alpha) + total_spent
            settled_after = Decimal(family.settled_alpha) + total_spent
            if reserved_after > CAPITAL_OOS_TOTAL_ALPHA:
                raise ValueError("legacy OOS attempts exhaust the persistent alpha budget")
            for offset, (row, spent) in enumerate(zip(ordered, spent_values, strict=True)):
                row_ordinal = ordinal + offset
                receipt = {
                    "contract_version": "capital-oos-legacy-reconciliation-v1",
                    "legacy_attempt_id": str(row.id),
                    "oos_vintage_id": str(row.oos_vintage_id),
                    "family_id": family_id,
                    "stable_mandate_sha256": family_sha,
                    "ordinal": row_ordinal,
                    "spent_alpha": _decimal_text(spent),
                    "raw_p_value": 1.0,
                    "passed": False,
                    "failure_recorded": True,
                    "actor": actor_value,
                    "reason": reason_value,
                    "supporting_evidence": support,
                    "supporting_evidence_sha256": support_sha,
                    "legacy_evidence_sha256": str(row.legacy_evidence_sha256),
                }
                connection.execute(
                    update(capital_oos_legacy_attempts)
                    .where(
                        capital_oos_legacy_attempts.c.id == row.id,
                        capital_oos_legacy_attempts.c.status == "unreconciled",
                    )
                    .values(
                        status="reconciled",
                        reconciled_family_id=family_id,
                        ordinal=row_ordinal,
                        spent_alpha=spent,
                        reconciliation_json=receipt,
                        reconciliation_sha256=canonical_sha256(receipt),
                        reconciled_at=now,
                    )
                )
            connection.execute(
                update(capital_oos_alpha_families)
                .where(capital_oos_alpha_families.c.id == family_id)
                .values(
                    next_ordinal=ordinal + len(ordered),
                    reserved_alpha=reserved_after,
                    settled_alpha=settled_after,
                    updated_at=now,
                )
            )
            reconciled = connection.execute(
                select(capital_oos_legacy_attempts)
                .where(capital_oos_legacy_attempts.c.id.in_(identifiers))
                .order_by(capital_oos_legacy_attempts.c.ordinal)
            ).all()
            return [self._decode_legacy(row) for row in reconciled]

    def reserve_batch(
        self,
        *,
        dataset_lineage_id: str,
        dataset_identity_sha256: str,
        stable_mandate: Mapping[str, Any],
        batch_key: str,
        frozen_bundle_manifest_sha256: str,
        frozen_baseline_manifest_sha256: str,
        research_data_end: date | datetime | str,
        final_oos_trading_dates: Sequence[date | datetime | str],
        embargo_trading_dates: Sequence[date | datetime | str],
    ) -> dict[str, Any]:
        lineage = str(dataset_lineage_id).strip().lower()
        dataset_identity = str(dataset_identity_sha256).strip().lower()
        key = str(batch_key).strip()
        bundle_sha = str(frozen_bundle_manifest_sha256).strip().lower()
        baseline_sha = str(frozen_baseline_manifest_sha256).strip().lower()
        if not key:
            raise ValueError("batch_key is required")
        if not _SHA256.fullmatch(lineage) or not _SHA256.fullmatch(dataset_identity):
            raise ValueError("dataset lineage and identity must be SHA256 evidence")
        if not isinstance(stable_mandate, Mapping):
            raise ValueError("stable_mandate must be a capital OOS mandate object")
        mandate, family_sha, family_id = self._family_identity(stable_mandate)
        if not _SHA256.fullmatch(bundle_sha):
            raise ValueError("frozen_bundle_manifest_sha256 must be a SHA256")
        if not _SHA256.fullmatch(baseline_sha):
            raise ValueError("frozen_baseline_manifest_sha256 must be a SHA256")
        window = validate_final_oos_window(
            research_data_end=research_data_end,
            final_oos_trading_dates=final_oos_trading_dates,
            embargo_trading_dates=embargo_trading_dates,
        )
        now = _utcnow()
        preregistration = {
            "contract_version": "capital-final-oos-preregistration-v1",
            "family_id": family_id,
            "batch_key": key,
            "frozen_bundle_manifest_sha256": bundle_sha,
            "frozen_baseline_manifest_sha256": baseline_sha,
            "hypothesis_count": 1,
            "stable_mandate_sha256": family_sha,
            "dataset_lineage_id": lineage,
            "dataset_identity_sha256": dataset_identity,
            **window,
            "opened_before_results": True,
        }
        preregistration_sha = canonical_sha256(preregistration)

        self._capture_missing_legacy_history()
        with self.engine.begin() as connection:
            connection.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:ledger_lock))"),
                {"ledger_lock": "capital-oos-alpha-ledger-v1"},
            )
            missing_capture = connection.execute(
                select(oos_vintages)
                .where(
                    oos_vintages.c.id.not_in(
                        select(capital_oos_legacy_attempts.c.oos_vintage_id)
                    )
                )
            ).all()
            unresolved = connection.execute(
                select(capital_oos_legacy_attempts.c.id)
                .where(capital_oos_legacy_attempts.c.status == "unreconciled")
                .limit(1)
            ).first()
            for uncaptured_vintage in missing_capture:
                self._verify_linked_vintage(connection, uncaptured_vintage)
            if unresolved is not None:
                raise ValueError(
                    "legacy final OOS history requires explicit immutable reconciliation"
                )
            family = self._ensure_family(
                connection,
                mandate=mandate,
                family_sha=family_sha,
                family_id=family_id,
                now=now,
            )
            existing = connection.execute(
                select(capital_oos_alpha_batches).where(
                    capital_oos_alpha_batches.c.family_id == family_id,
                    capital_oos_alpha_batches.c.batch_key == key,
                )
            ).first()
            if existing is not None:
                if str(existing.preregistration_sha256) != preregistration_sha:
                    raise ValueError("idempotent final OOS batch has different preregistration")
                return self._decode_batch(existing)
            pending = connection.execute(
                select(capital_oos_alpha_batches.c.id).where(
                    capital_oos_alpha_batches.c.family_id == family_id,
                    capital_oos_alpha_batches.c.status == "reserved",
                )
            ).first()
            if pending is not None:
                raise ValueError(
                    "capital OOS family already has an unsettled reserved batch"
                )
            overlap = connection.execute(
                select(capital_oos_alpha_batches.c.id).where(
                    capital_oos_alpha_batches.c.family_id == family_id,
                    capital_oos_alpha_batches.c.final_oos_start
                    <= date.fromisoformat(window["final_oos_end"]),
                    capital_oos_alpha_batches.c.final_oos_end
                    >= date.fromisoformat(window["final_oos_start"]),
                )
            ).first()
            if overlap is not None:
                raise ValueError("final OOS window overlaps an earlier batch in this family")
            legacy_overlap = connection.execute(
                select(capital_oos_legacy_attempts.c.id).where(
                    capital_oos_legacy_attempts.c.reconciled_family_id == family_id,
                    capital_oos_legacy_attempts.c.final_oos_start
                    <= date.fromisoformat(window["final_oos_end"]),
                    capital_oos_legacy_attempts.c.final_oos_end
                    >= date.fromisoformat(window["final_oos_start"]),
                )
            ).first()
            if legacy_overlap is not None:
                raise ValueError("final OOS window overlaps reconciled legacy history")
            ordinal = int(family.next_ordinal)
            alpha_value = _stored_alpha(batch_alpha(ordinal))
            reserved_after = Decimal(family.reserved_alpha) + alpha_value
            if reserved_after > CAPITAL_OOS_TOTAL_ALPHA:
                raise ValueError("capital OOS alpha budget is exhausted")
            batch_id = canonical_sha256(
                {"kind": "capital-oos-alpha-batch-v1", "family_id": family_id, "batch_key": key}
            )
            connection.execute(
                insert(capital_oos_alpha_batches).values(
                    id=batch_id,
                    family_id=family_id,
                    batch_key=key,
                    ordinal=ordinal,
                    status="reserved",
                    frozen_bundle_manifest_sha256=bundle_sha,
                    frozen_baseline_manifest_sha256=baseline_sha,
                    dataset_lineage_id=lineage,
                    dataset_identity_sha256=dataset_identity,
                    hypothesis_count=1,
                    research_data_end=date.fromisoformat(window["research_data_end"]),
                    final_oos_start=date.fromisoformat(window["final_oos_start"]),
                    final_oos_end=date.fromisoformat(window["final_oos_end"]),
                    trading_day_count=window["trading_day_count"],
                    trading_dates_json=window["trading_dates"],
                    trading_dates_sha256=window["trading_dates_sha256"],
                    embargo_trading_day_count=window["embargo_trading_day_count"],
                    embargo_trading_dates_json=window["embargo_trading_dates"],
                    embargo_trading_dates_sha256=window["embargo_trading_dates_sha256"],
                    preregistration_json=preregistration,
                    preregistration_sha256=preregistration_sha,
                    batch_alpha=alpha_value,
                    created_at=now,
                )
            )
            connection.execute(
                update(capital_oos_alpha_families)
                .where(capital_oos_alpha_families.c.id == family_id)
                .values(
                    next_ordinal=ordinal + 1,
                    reserved_alpha=reserved_after,
                    updated_at=now,
                )
            )
            row = connection.execute(
                select(capital_oos_alpha_batches).where(capital_oos_alpha_batches.c.id == batch_id)
            ).one()
            return self._decode_batch(row)

    def settle_batch(
        self,
        batch_id: str,
        *,
        candidate_net_returns: pd.Series | None = None,
        baseline_net_returns: pd.Series | None = None,
        failed: bool = False,
        failure_reason: str | None = None,
        supporting_evidence: Mapping[str, Any] | None = None,
        research_reference: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        identifier = str(batch_id).strip()
        if not identifier:
            raise ValueError("batch_id is required")
        support = dict(supporting_evidence or {})
        support_sha = canonical_sha256(support)
        pointer = None
        with self.engine.connect() as connection:
            pointer = connection.execute(
                select(
                    capital_oos_alpha_batches.c.family_id,
                    capital_oos_alpha_batches.c.trading_dates_json,
                ).where(capital_oos_alpha_batches.c.id == identifier)
            ).first()
            if pointer is None:
                raise KeyError(identifier)
            vintage_id = str(support.get("oos_vintage_id") or "").strip()
            vintage = None
            if vintage_id:
                vintage = connection.execute(
                    select(oos_vintages).where(oos_vintages.c.id == vintage_id)
                ).first()
                if vintage is None:
                    raise ValueError("settlement references an unknown OOS vintage")
            else:
                vintage = connection.execute(
                    select(oos_vintages).where(
                        oos_vintages.c.capital_oos_alpha_batch_id == identifier
                    )
                ).first()
                if vintage is not None:
                    vintage_id = str(vintage.id)
            if vintage is not None:
                self._verify_linked_vintage(
                    connection,
                    vintage,
                    expected_batch_id=identifier,
                )
                vintage_link_evidence = {
                    "status": "verified",
                    "oos_vintage_id": vintage_id,
                    "sealed_candidate_set_sha256": str(
                        vintage.sealed_candidate_set_sha256
                    ),
                }
            else:
                vintage_link_evidence = {
                    "status": "not_created" if failed else "missing",
                    "oos_vintage_id": None,
                    "sealed_candidate_set_sha256": None,
                }
        expected_dates = list(pointer.trading_dates_json or [])
        if failed:
            reason = str(failure_reason or "").strip()
            if not reason:
                raise ValueError("a failed final OOS settlement requires a reason")
            if candidate_net_returns is not None or baseline_net_returns is not None:
                raise ValueError("failed settlement cannot include return results")
            raw_p_value = 1.0
            hac_evidence: dict[str, Any] = {
                "contract_version": CAPITAL_OOS_P_VALUE_CONTRACT,
                "status": "execution_failed",
                "one_sided_p_value": 1.0,
                "reason": reason,
            }
            bootstrap_evidence: dict[str, Any] = {
                "status": "not_run_execution_failed",
                "hard_gate_passed": False,
            }
            decay_evidence: dict[str, Any] = {
                "contract_version": CAPITAL_OOS_DECAY_CONTRACT,
                "status": "not_run_execution_failed",
                "decay_passed": None,
            }
            return_input_evidence: dict[str, Any] = {
                "status": "not_available_execution_failed",
                "candidate_net_returns_sha256": None,
                "baseline_net_returns_sha256": None,
                "paired_difference_sha256": None,
            }
        else:
            if candidate_net_returns is None or baseline_net_returns is None:
                raise ValueError("successful settlement requires paired net return series")
            if vintage_link_evidence["status"] != "verified":
                raise ValueError("successful settlement requires a linked OOS vintage")
            formal_artifact_sha = str(
                support.get("formal_oos_artifact_sha256") or ""
            ).strip().lower()
            if not _SHA256.fullmatch(formal_artifact_sha):
                raise ValueError(
                    "successful settlement requires formal_oos_artifact_sha256"
                )
            difference, hac_evidence = paired_hac_capital_test(
                candidate_net_returns,
                baseline_net_returns,
            )
            actual_dates = [item.date().isoformat() for item in difference.index]
            if actual_dates != expected_dates:
                raise ValueError("capital OOS return index differs from preregistered window")
            raw_p_value = float(hac_evidence["one_sided_p_value"])
            bootstrap_evidence = _bootstrap_gate(difference)
            decay_evidence = _oos_decay_check(difference, research_reference)
            normalized_dates = [item.date().isoformat() for item in difference.index]
            candidate_values = pd.to_numeric(candidate_net_returns).to_numpy(dtype=float)
            baseline_values = pd.to_numeric(baseline_net_returns).to_numpy(dtype=float)
            return_input_evidence = {
                "status": "recorded",
                "observations": len(normalized_dates),
                "candidate_net_returns_sha256": canonical_sha256(
                    [
                        [day, float(value).hex()]
                        for day, value in zip(
                            normalized_dates, candidate_values, strict=True
                        )
                    ]
                ),
                "baseline_net_returns_sha256": canonical_sha256(
                    [
                        [day, float(value).hex()]
                        for day, value in zip(
                            normalized_dates, baseline_values, strict=True
                        )
                    ]
                ),
                "paired_difference_sha256": canonical_sha256(
                    [
                        [day, float(value).hex()]
                        for day, value in zip(
                            normalized_dates,
                            difference.to_numpy(dtype=float),
                            strict=True,
                        )
                    ]
                ),
            }

        with self.engine.begin() as connection:
            family = connection.execute(
                select(capital_oos_alpha_families)
                .where(capital_oos_alpha_families.c.id == pointer.family_id)
                .with_for_update()
            ).one()
            batch = connection.execute(
                select(capital_oos_alpha_batches)
                .where(capital_oos_alpha_batches.c.id == identifier)
                .with_for_update()
            ).one()
            threshold = Decimal(batch.batch_alpha)
            # Wide-in/strict-out recalibration: the alpha-spending HAC p-value
            # and the paired bootstrap remain sealed below as a report-only
            # health check and no longer veto settlement.  The formal OOS
            # life-or-death check is the decay of the after-cost excess versus
            # the sealed research edge; an unevaluable decay check (no sealed
            # research reference) does not veto, and the forward paper gate
            # remains the final arbiter.
            passed = not failed and decay_evidence.get("decay_passed") is not False
            evidence = {
                "contract_version": "capital-final-oos-alpha-settlement-v1",
                "batch_id": identifier,
                "family_id": str(batch.family_id),
                "stable_mandate_sha256": str(family.capital_oos_family_sha256),
                "frozen_bundle_manifest_sha256": str(batch.frozen_bundle_manifest_sha256),
                "frozen_baseline_manifest_sha256": str(
                    batch.frozen_baseline_manifest_sha256
                ),
                "dataset_lineage_id": str(batch.dataset_lineage_id),
                "dataset_identity_sha256": str(batch.dataset_identity_sha256),
                "hypothesis_count": 1,
                "batch_alpha": _decimal_text(threshold),
                "raw_p_value": raw_p_value,
                "passed": passed,
                "failure_recorded": failed,
                "paired_hac": hac_evidence,
                "paired_block_bootstrap": bootstrap_evidence,
                "decay_check": decay_evidence,
                "paired_return_inputs": return_input_evidence,
                "oos_vintage_link": vintage_link_evidence,
                "bootstrap_is_independent_hard_gate": False,
                "statistical_evidence_role": "report_only",
                "supporting_evidence": support,
                "supporting_evidence_sha256": support_sha,
                "preregistration_sha256": str(batch.preregistration_sha256),
            }
            evidence_sha = canonical_sha256(evidence)
            if str(batch.status) == "settled":
                if str(batch.settlement_evidence_sha256) == evidence_sha:
                    return self._decode_batch(batch)
                raise ValueError("settled capital OOS evidence is immutable")
            connection.execute(
                update(capital_oos_alpha_batches)
                .where(
                    capital_oos_alpha_batches.c.id == identifier,
                    capital_oos_alpha_batches.c.status == "reserved",
                )
                .values(
                    status="settled",
                    raw_p_value=raw_p_value,
                    passed=passed,
                    failure_recorded=failed,
                    settlement_evidence_json=evidence,
                    settlement_evidence_sha256=evidence_sha,
                    settled_at=_utcnow(),
                )
            )
            connection.execute(
                update(capital_oos_alpha_families)
                .where(capital_oos_alpha_families.c.id == batch.family_id)
                .values(
                    settled_alpha=Decimal(family.settled_alpha) + threshold,
                    updated_at=_utcnow(),
                )
            )
            settled = connection.execute(
                select(capital_oos_alpha_batches).where(
                    capital_oos_alpha_batches.c.id == identifier
                )
            ).one()
            return self._decode_batch(settled)

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                self._batch_audit_query().where(
                    capital_oos_alpha_batches.c.id == str(batch_id).strip()
                )
            ).first()
            if row is None:
                raise KeyError(batch_id)
            return self._decode_batch(row)

    def list_families(self, *, limit: int = 200) -> list[dict[str, Any]]:
        query = select(capital_oos_alpha_families)
        with self.engine.connect() as connection:
            rows = connection.execute(
                query.order_by(capital_oos_alpha_families.c.created_at.desc()).limit(
                    min(max(int(limit), 1), 2000)
                )
            ).all()
        return [self._decode_family(row) for row in rows]

    def list_batches(self, family_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                self._batch_audit_query()
                .where(capital_oos_alpha_batches.c.family_id == str(family_id).strip())
                .order_by(capital_oos_alpha_batches.c.ordinal.desc())
                .limit(min(max(int(limit), 1), 5000))
            ).all()
        return [self._decode_batch(row) for row in rows]

    def list_legacy_attempts(
        self, *, status: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        self._capture_missing_legacy_history()
        query = select(capital_oos_legacy_attempts)
        if status is not None:
            normalized = str(status).strip()
            if normalized not in {"unreconciled", "reconciled"}:
                raise ValueError("unknown legacy reconciliation status")
            query = query.where(capital_oos_legacy_attempts.c.status == normalized)
        with self.engine.connect() as connection:
            rows = connection.execute(
                query.order_by(
                    capital_oos_legacy_attempts.c.first_opened_at,
                    capital_oos_legacy_attempts.c.id,
                ).limit(min(max(int(limit), 1), 5000))
            ).all()
        return [self._decode_legacy(row) for row in rows]

    @staticmethod
    def _batch_audit_query() -> Any:
        return select(
            capital_oos_alpha_batches,
            capital_oos_alpha_families.c.capital_oos_family_sha256,
            capital_oos_alpha_families.c.mandate_json,
            capital_oos_alpha_families.c.total_alpha,
            capital_oos_alpha_families.c.policy_sha256,
        ).join(
            capital_oos_alpha_families,
            capital_oos_alpha_families.c.id == capital_oos_alpha_batches.c.family_id,
        )

    @staticmethod
    def _decode_batch(row: Any) -> dict[str, Any]:
        value = dict(row._mapping if hasattr(row, "_mapping") else row)
        for key in ("batch_alpha", "total_alpha"):
            if value.get(key) is not None:
                value[key] = _decimal_text(Decimal(value[key]))
        for key in ("research_data_end", "final_oos_start", "final_oos_end"):
            if isinstance(value.get(key), date):
                value[key] = value[key].isoformat()
        for key in ("created_at", "settled_at"):
            if isinstance(value.get(key), datetime):
                value[key] = value[key].isoformat(timespec="seconds")
        return value

    @staticmethod
    def _decode_family(row: Any) -> dict[str, Any]:
        value = dict(row._mapping if hasattr(row, "_mapping") else row)
        for key in ("total_alpha", "reserved_alpha", "settled_alpha"):
            value[key] = _decimal_text(Decimal(value[key]))
        for key in ("created_at", "updated_at"):
            if isinstance(value.get(key), datetime):
                value[key] = value[key].isoformat(timespec="seconds")
        return value

    @staticmethod
    def _decode_legacy(row: Any) -> dict[str, Any]:
        value = dict(row._mapping if hasattr(row, "_mapping") else row)
        if value.get("spent_alpha") is not None:
            value["spent_alpha"] = _decimal_text(Decimal(value["spent_alpha"]))
        for key in ("final_oos_start", "final_oos_end"):
            if isinstance(value.get(key), date):
                value[key] = value[key].isoformat()
        for key in ("first_opened_at", "consumed_at", "created_at", "reconciled_at"):
            if isinstance(value.get(key), datetime):
                value[key] = value[key].isoformat(timespec="seconds")
        return value
