from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Any

from .history_bounds import (
    PRIMARY_MARKET_HISTORY_DATASETS,
    PRIMARY_MARKET_HISTORY_START,
)
from .models import FetchSpec

# Production probe: one exact survey day returned a full first page and 82
# further rows at offset=400.  Legacy requests omitted limit/offset, so 400 is
# both the provider page size and the only row count that cannot prove that the
# old unpaged request was complete.
STK_SURV_PROVIDER_PAGE_LIMIT = 400
INDEX_MEMBER_ALL_WEEKLY_COHORT = "shenwan-pit-membership-v1"


@dataclass(frozen=True, slots=True)
class ReferenceRefreshPolicy:
    dataset: str
    cadence: str
    retention: str = "append_only_units_and_immutable_snapshots"


# Production audit on 2026-07-15 found these 29 datasets without a usable
# snapshot time axis. Some are true as-of masters; others are revision-prone
# series whose provider contract does not expose a reliable row timestamp in
# every response. They therefore need a versioned request generation even
# when the API parameters themselves do not change.
AUDITED_REFERENCE_DATASETS = frozenset(
    {
        "cb_basic",
        "cb_price_chg",
        "cb_rate",
        "cn_cpi",
        "cn_gdp",
        "cn_m",
        "cn_pmi",
        "cn_ppi",
        "cn_schedule",
        "etf_basic",
        "etf_index",
        "fund_basic",
        "fut_basic",
        "fut_trade_cal",
        "fx_obasic",
        "hk_basic",
        "hk_tradecal",
        "index_basic",
        "index_classify",
        "new_share",
        "opt_basic",
        "sf_month",
        "shibor",
        "shibor_lpr",
        "stk_surv",
        "stock_basic",
        "us_basic",
        "us_tradecal",
        "us_tycr",
    }
)


REFERENCE_REFRESH_POLICIES: dict[str, ReferenceRefreshPolicy] = {
    dataset: ReferenceRefreshPolicy(dataset, cadence)
    for cadence, datasets in {
        "daily": (
            "cb_basic",
            "cb_price_chg",
            # Rating batches query full history by ts_code; new ratings must
            # not remain hidden behind an earlier successful request key.
            "cb_rating",
            "etf_basic",
            "fund_basic",
            "fut_basic",
            "fut_trade_cal",
            "hk_basic",
            "hk_tradecal",
            "new_share",
            "opt_basic",
            "shibor",
            "shibor_lpr",
            "stock_basic",
            "us_basic",
            "us_tradecal",
            "us_tycr",
            "monetary_policy",
        ),
        "weekly": (
            "cb_rate",
            "etf_index",
            "fx_obasic",
            "index_basic",
            "index_classify",
            "index_member_all",
            "stk_surv",
            "bse_mapping",
            "ci_index_member",
            "hm_list",
            "mkt_idx_bmk",
            "sge_basic",
            "stk_rewards",
            "stock_company",
            "ths_index",
            "ths_member",
        ),
        "monthly": (
            "cn_cpi",
            "cn_gdp",
            "cn_m",
            "cn_pmi",
            "cn_ppi",
            "cn_schedule",
            "sf_month",
        ),
    }.items()
    for dataset in datasets
}


_NATURAL_WINDOW_KEYS = {
    "trade_date",
    "ann_date",
    "end_date",
    "start_date",
    "cal_date",
    "nav_date",
    "date",
    "month",
    "m",
    "quarter",
    "period",
    "start_m",
    "end_m",
    "start_week",
    "end_week",
    "publish_date",
    "surv_date",
}


def reference_refresh_bucket(dataset: str, as_of: date) -> str | None:
    policy = REFERENCE_REFRESH_POLICIES.get(dataset)
    if policy is None:
        return None
    if policy.cadence == "daily":
        return as_of.isoformat()
    if policy.cadence == "weekly":
        return (as_of - timedelta(days=as_of.weekday())).isoformat()
    if policy.cadence == "monthly":
        return as_of.strftime("%Y-%m")
    raise ValueError(f"unsupported reference refresh cadence: {policy.cadence}")


def apply_reference_refresh(
    specs: Iterable[FetchSpec],
    *,
    as_of: date,
    force: bool = False,
) -> list[FetchSpec]:
    """Version full/as-of requests without perturbing naturally dated units."""

    result: list[FetchSpec] = []
    for spec in specs:
        bucket = reference_refresh_bucket(spec.dataset, as_of)
        if bucket is None or (not force and _NATURAL_WINDOW_KEYS.intersection(spec.scope)):
            result.append(spec)
            continue
        policy = REFERENCE_REFRESH_POLICIES[spec.dataset]
        result.append(
            replace(
                spec,
                scope={
                    **spec.scope,
                    "reference_refresh_bucket": bucket,
                    "reference_refresh_cadence": policy.cadence,
                },
            )
        )
    return result


def select_current_reference_units(
    rows: Iterable[dict[str, Any]], *, snapshot_end: date
) -> list[dict[str, Any]]:
    """Select the latest successful as-of generation for each API partition.

    Old work units remain append-only and old immutable snapshots keep their
    original manifests. A successor snapshot excludes superseded reference
    generations so changed master rows cannot appear twice.
    index_member_all is the documented exception: its weekly cohorts union
    (newest cohort alone is not history-complete because the provider prunes
    long-delisted members), with row-level revision arbitration deferred to
    the snapshot layer (row_identity.LATEST_GENERATION_KEYS).
    """

    materialized = [row for row in rows if not _retired_provider_request_contract(row)]
    membership_buckets = {
        str(scope["reference_refresh_bucket"])
        for row in materialized
        if str(row.get("dataset") or "") == "index_member_all"
        and (scope := dict(row.get("scope_json") or {})).get("membership_cohort")
        == INDEX_MEMBER_ALL_WEEKLY_COHORT
        and scope.get("reference_refresh_bucket")
        and _bucket_start(str(scope["reference_refresh_bucket"])) <= snapshot_end
    }
    if membership_buckets:
        # The provider progressively prunes long-delisted members from
        # index_member_all responses (measured on production 2026-09-02: the
        # 2026-08-31 cohort silently dropped 346 codes and thousands of early
        # in_date intervals that 2026-08-25 still served).  No single weekly
        # cohort is history-complete any more, so every cohort at or before
        # the snapshot end stays selected and the snapshot layer arbitrates
        # revisions row-wise (LATEST_GENERATION_KEYS in row_identity.py):
        # identical intervals collapse across generations, and a key present
        # in several generations keeps the newest generation's out_date/name.
        materialized = [
            row
            for row in materialized
            if str(row.get("dataset") or "") != "index_member_all"
            or (
                (scope := dict(row.get("scope_json") or {})).get(
                    "membership_cohort"
                )
                == INDEX_MEMBER_ALL_WEEKLY_COHORT
                and scope.get("reference_refresh_bucket")
                and _bucket_start(str(scope["reference_refresh_bucket"]))
                <= snapshot_end
            )
        ]
    paginated_stk_surv_days = {
        _stk_surv_request_identity(row)
        for row in materialized
        if _is_current_stk_surv_page_zero(row)
    }
    # If an explicit page plan already exists for a survey day, it is the
    # current generation even when still pending/failed. Retaining a legacy
    # short success for that same day would either duplicate rows or let stale
    # data mask the incomplete current plan.
    materialized = [
        row
        for row in materialized
        if not (
            str(row.get("dataset") or "") == "stk_surv"
            and "limit" not in dict(row.get("params_json") or {})
            and "offset" not in dict(row.get("params_json") or {})
            and _stk_surv_request_identity(row) in paginated_stk_surv_days
        )
    ]
    canonical_fina_audit_periods = {
        _fina_audit_period(row)
        for row in materialized
        if _is_canonical_fina_audit_page_zero(row)
    }
    # The corrected fina_audit plan binds every page to its requested report
    # period. Once its page zero exists, that whole family is authoritative
    # even while pending/failed; falling back to an older unbound family would
    # hide an incomplete current plan and may publish rows from another period.
    materialized = [
        row
        for row in materialized
        if not (
            str(row.get("dataset") or "") == "fina_audit"
            and _fina_audit_period(row) in canonical_fina_audit_periods
            and not _has_canonical_fina_audit_period_contract(row)
        )
    ]
    superseded_page_groups: set[str] = set()
    for row in materialized:
        scope = dict(row.get("scope_json") or {})
        parent = scope.get("supersedes_page_group")
        if parent:
            superseded_page_groups.add(str(parent))
        parents = scope.get("supersedes_page_groups")
        if isinstance(parents, (list, tuple, set, frozenset)):
            superseded_page_groups.update(str(value) for value in parents if value)
    materialized = [
        row
        for row in materialized
        if str(dict(row.get("scope_json") or {}).get("page_group") or "")
        not in superseded_page_groups
    ]

    plain: list[dict[str, Any]] = []
    versioned: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in materialized:
        scope = dict(row.get("scope_json") or {})
        bucket = scope.get("reference_refresh_bucket")
        if not bucket:
            plain.append(row)
            continue
        if _bucket_start(str(bucket)) > snapshot_end:
            continue
        identity_scope = {
            key: value
            for key, value in scope.items()
            if key not in {"reference_refresh_bucket", "reference_refresh_cadence"}
        }
        identity = _stable_identity(identity_scope)
        versioned.setdefault((str(row["dataset"]), identity), []).append(row)

    selected = list(plain)
    versioned_identities = set(versioned)
    for (dataset, _identity), candidates in versioned.items():
        if dataset == "index_member_all":
            # Weekly membership cohorts union instead of superseding: the
            # provider prunes long-delisted members from newer responses, so
            # every cohort at or before the snapshot end keeps its units and
            # the snapshot layer arbitrates revisions row-wise
            # (row_identity.LATEST_GENERATION_KEYS).
            selected.extend(candidates)
            continue
        latest = max(
            str(dict(item.get("scope_json") or {})["reference_refresh_bucket"])
            for item in candidates
        )
        selected.extend(
            item
            for item in candidates
            if str(dict(item.get("scope_json") or {})["reference_refresh_bucket"]) == latest
        )

    # Once a partition has a versioned successor, omit the legacy unversioned
    # unit for the same request identity. Legacy-only partitions remain usable.
    result: list[dict[str, Any]] = []
    for row in selected:
        scope = dict(row.get("scope_json") or {})
        if scope.get("reference_refresh_bucket"):
            result.append(row)
            continue
        identity = _stable_identity(scope)
        if (str(row["dataset"]), identity) not in versioned_identities:
            result.append(row)
    return sorted(result, key=lambda item: (str(item["dataset"]), str(item["unit_key"])))


def reference_manifest_metadata(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    materialized = list(rows)
    buckets = sorted(
        {
            str(scope["reference_refresh_bucket"])
            for row in materialized
            if (scope := dict(row.get("scope_json") or {})).get("reference_refresh_bucket")
        }
    )
    if not buckets:
        return None
    # Lineage manifests created before dataset identity was embedded in every
    # source-unit record still carry a valid refresh bucket.  Keep those
    # manifests readable; cadence is simply unavailable for that legacy shape.
    dataset = str(materialized[0].get("dataset") or "")
    policy = REFERENCE_REFRESH_POLICIES.get(dataset)
    return {
        "cadence": policy.cadence if policy else None,
        "selected_buckets": buckets,
        "retention": policy.retention if policy else None,
    }


def _bucket_start(value: str) -> date:
    if len(value) == 7:
        return date.fromisoformat(f"{value}-01")
    return date.fromisoformat(value)


def _stable_identity(scope: dict[str, Any]) -> str:
    from .models import canonical_json

    return canonical_json(scope)


def _stk_surv_request_identity(row: dict[str, Any]) -> tuple[str, str]:
    params = dict(row.get("params_json") or {})
    return (
        str(params.get("start_date") or ""),
        str(params.get("end_date") or ""),
    )


def _is_current_stk_surv_page_zero(row: dict[str, Any]) -> bool:
    if str(row.get("dataset") or "") != "stk_surv":
        return False
    params = dict(row.get("params_json") or {})
    scope = dict(row.get("scope_json") or {})
    start, end = _stk_surv_request_identity(row)
    if not start or start != end:
        return False
    try:
        return (
            int(params.get("limit")) == STK_SURV_PROVIDER_PAGE_LIMIT
            and int(params.get("offset")) == 0
            and int(scope.get("page_size")) == STK_SURV_PROVIDER_PAGE_LIMIT
            and int(scope.get("offset")) == 0
            and str(scope.get("page_group") or "") == f"stk_surv:{start}"
        )
    except (TypeError, ValueError):
        return False


def _fina_audit_period(row: dict[str, Any]) -> str:
    params = dict(row.get("params_json") or {})
    scope = dict(row.get("scope_json") or {})
    return str(params.get("period") or scope.get("period") or "")


def _has_canonical_fina_audit_period_contract(row: dict[str, Any]) -> bool:
    period = _fina_audit_period(row)
    scope = dict(row.get("scope_json") or {})
    return bool(
        period
        and str(scope.get("expected_date_field") or "") == "end_date"
        and str(scope.get("expected_date") or "") == period
    )


def _is_canonical_fina_audit_page_zero(row: dict[str, Any]) -> bool:
    if str(row.get("dataset") or "") != "fina_audit":
        return False
    params = dict(row.get("params_json") or {})
    scope = dict(row.get("scope_json") or {})
    period = _fina_audit_period(row)
    if (
        not _has_canonical_fina_audit_period_contract(row)
        or str(params.get("period") or "") != period
        or str(scope.get("period") or "") != period
        or str(scope.get("page_group") or "") != f"fina_audit:{period}"
    ):
        return False
    try:
        page_size = int(scope.get("page_size"))
        return (
            page_size > 0
            and int(params.get("limit")) == page_size
            and int(params.get("offset")) == 0
            and int(scope.get("offset")) == 0
        )
    except (TypeError, ValueError):
        return False


def _retired_provider_request_contract(row: dict[str, Any]) -> bool:
    """Exclude proven-invalid or truncated requests from successor snapshots.

    The Tushare ``index_member_all`` interface accepts l1/l2/l3_code, not
    index_code. Legacy requests used the ignored index_code parameter, so each
    partition stored the same provider-capped unfiltered rows. Immutable old
    snapshots retain their manifests; successor selection retires only proven
    bad request shapes after corrected contracts are available.
    """

    dataset = str(row.get("dataset") or "")
    params = dict(row.get("params_json") or {})
    if dataset == "index_member_all" and "index_code" in params:
        return True

    # The former stk_surv contract requested one day without limit/offset and
    # treated the provider's 400-row cap as a terminal validation error.  A
    # legacy short page is complete and remains reusable, but a legacy full
    # page is provably truncated and must never enter a successor snapshot.
    # Current requests carry explicit limit/offset pagination and are retained.
    if (
        dataset == "stk_surv"
        and "limit" not in params
        and "offset" not in params
        and int(row.get("row_count") or 0) >= STK_SURV_PROVIDER_PAGE_LIMIT
    ):
        return True

    # Before the source split was encoded in ``history_bounds``, a full plan
    # could request the configured primary gateway for 2008-2015 and later add
    # the admitted BaoStock backfill into the same dataset.  Selecting both
    # immutable unit families creates one duplicate business key for almost
    # every legacy observation.  The provider API name distinguishes the
    # primary per-date request from the explicitly labelled baostock_* unit;
    # retire only the former and keep all raw files auditable.
    if dataset not in PRIMARY_MARKET_HISTORY_DATASETS:
        return False
    if str(row.get("api_name") or "") != dataset:
        return False
    scope = dict(row.get("scope_json") or {})
    requested = scope.get("trade_date") or params.get("trade_date")
    requested_date = _request_date(requested)
    return requested_date is not None and requested_date < PRIMARY_MARKET_HISTORY_START


def _request_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    compact = text.replace("-", "")
    if len(compact) < 8 or not compact[:8].isdigit():
        return None
    try:
        return date(
            int(compact[:4]),
            int(compact[4:6]),
            int(compact[6:8]),
        )
    except ValueError:
        return None
