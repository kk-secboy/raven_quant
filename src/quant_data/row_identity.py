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
    "irm_qa_sh": ("trade_date", "ts_code", "q"),
    "irm_qa_sz": ("trade_date", "ts_code", "q"),
}


def semantic_provider_columns(dataset: str, columns: Collection[str]) -> set[str]:
    """Provider fields that define a distinct financial observation."""

    return set(columns) - {"ingested_at"} - set(SEMANTIC_METADATA_COLUMNS.get(dataset, ()))
