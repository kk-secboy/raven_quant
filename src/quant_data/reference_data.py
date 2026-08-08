from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Any

from .models import FetchSpec


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
    """

    materialized = [row for row in rows if not _retired_provider_request_contract(row)]
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
    for _key, candidates in versioned.items():
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
    dataset = str(materialized[0]["dataset"])
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


def _retired_provider_request_contract(row: dict[str, Any]) -> bool:
    """Exclude proven-invalid completed requests from successor snapshots.

    The Tushare ``index_member_all`` interface accepts l1/l2/l3_code, not
    index_code. Legacy requests used the ignored index_code parameter, so each
    partition stored the same provider-capped unfiltered rows. Immutable old
    snapshots retain their manifests; successor selection retires only that
    invalid request shape after the corrected L3 Y/N partitions were added.
    """

    return str(row.get("dataset")) == "index_member_all" and "index_code" in dict(
        row.get("params_json") or {}
    )
