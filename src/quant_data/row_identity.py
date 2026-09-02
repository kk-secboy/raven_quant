from __future__ import annotations

from collections.abc import Collection

# Provider metadata that does not define a distinct financial observation.
# Snapshot builds aggregate these fields after grouping by the remaining
# provider columns, preventing spelling/full-width/timestamp drift from
# duplicating quantities or corpus documents.
SEMANTIC_METADATA_COLUMNS: dict[str, frozenset[str]] = {
    "ccass_hold": frozenset({"name"}),
    "ccass_hold_detail": frozenset({"name", "col_participant_name"}),
    "irm_qa_sh": frozenset({"name", "pub_time"}),
    "irm_qa_sz": frozenset({"name", "pub_time"}),
    # Tushare can repeat one report URL for multiple covered stocks or
    # industries. Preserve a deterministic representative set of tags while
    # URL plus publication date remains the PIT document identity.
    "research_report": frozenset(
        {
            "abstr",
            "title",
            "report_type",
            "author",
            "name",
            "ts_code",
            "inst_csname",
            "ind_name",
            "file_name",
        }
    ),
    # Snapshot-only presentation fields. They are recomputed from the immutable
    # provider title/content on every snapshot build and must never become part
    # of the provider-row identity when an older snapshot is used as a base.
    "news": frozenset({"display_title", "title_source"}),
    "share_float": frozenset({"float_ratio"}),
}

# A conflicting ratio must never be selected arbitrarily. The share count is
# retained as the event quantity while the ambiguous derived percentage is
# explicitly null in the immutable snapshot.
NULL_ON_AMBIGUITY_COLUMNS: dict[str, frozenset[str]] = {
    "share_float": frozenset({"float_ratio"}),
}

# These interfaces have no safe revision selector for conflicting semantic
# rows. Snapshot publication drops the entire conflicting business key while
# the verification report records the quarantined-key count.
SNAPSHOT_QUARANTINE_KEYS: dict[str, tuple[str, ...]] = {
    "ccass_hold": ("ts_code", "trade_date"),
    "ccass_hold_detail": ("ts_code", "trade_date", "col_participant_id"),
    "irm_qa_sh": ("trade_date", "ts_code", "q"),
    "irm_qa_sz": ("trade_date", "ts_code", "q"),
}

# Peripheral datasets whose provider legitimately republishes a business key
# with revised content (adjusted-price recomputation after corporate actions,
# or later pulls filling previously NULL trailing fields).  The safe revision
# selector is the ingestion generation: the snapshot keeps the row from the
# latest generation per business key (ties broken by provider-field
# completeness, then full-row order for determinism).  Verification still
# blocks publication when the top rank is not unique, so a genuine same-
# generation conflict can never be masked.  Measured on production
# 2026-09-02: hk_daily_adj 394 keys, us_daily 637 keys, us_tbr 8 keys all
# resolve to a unique latest generation; us_daily_adj had 2 same-generation
# keys whose rows differ only by NULL-vs-filled derived fields (completeness
# tiebreak resolves them deterministically).
#
# index_member_all joins this registry for a different reason: the weekly
# cohort selection unions every cohort at or before the snapshot end because
# the provider prunes long-delisted members from newer responses.  The
# interval key keeps the newest cohort's out_date/name/is_new when several
# cohorts carry the same interval (revision arbitration), and retains
# intervals only older cohorts still serve (pruned history).
LATEST_GENERATION_KEYS: dict[str, tuple[str, ...]] = {
    "us_daily": ("ts_code", "trade_date"),
    "us_daily_adj": ("ts_code", "trade_date"),
    "hk_daily_adj": ("ts_code", "trade_date"),
    "us_tbr": ("date",),
    "index_member_all": ("ts_code", "in_date", "l1_code", "l2_code", "l3_code"),
}


def provider_row_completeness_sql(dataset: str, columns: Collection[str]) -> str:
    """SQL expression counting non-NULL provider fields (revision tiebreak)."""

    terms = [
        f"CASE WHEN {_sql_identifier(column)} IS NOT NULL THEN 1 ELSE 0 END"
        for column in sorted(semantic_provider_columns(dataset, columns))
    ]
    return " + ".join(terms) if terms else "0"


def _sql_identifier(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def semantic_provider_columns(dataset: str, columns: Collection[str]) -> set[str]:
    """Provider fields that define a distinct financial observation."""

    return set(columns) - {"ingested_at"} - set(SEMANTIC_METADATA_COLUMNS.get(dataset, ()))
