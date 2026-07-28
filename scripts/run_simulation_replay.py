#!/usr/bin/env python3
"""Extract one immutable 1/5-minute execution day for transactional simulation booking."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_data.execution_contract import require_minute_execution_contract
from quant_platform.corporate_actions import (
    corporate_actions_sha256,
    normalize_dividend_rows,
)
from quant_platform.execution_algorithms import (
    execution_time_slots,
    normalize_execution_policy,
)
from quant_platform.simulation_store import VWAP_PROFILE_METHOD


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _source_symbols(instruments: list[str]) -> set[str]:
    result = {value.upper() for value in instruments}
    for value in instruments:
        symbol = value.upper()
        if len(symbol) > 2 and symbol[:2] in {"SH", "SZ", "BJ"}:
            result.add(f"{symbol[2:]}.{symbol[:2]}")
    return result


def _qlib_symbol(value: Any) -> str:
    symbol = str(value).strip().upper()
    if "." in symbol:
        code, exchange = symbol.split(".", 1)
        return f"{exchange}{code}"
    return symbol


def _historical_volume_profile(
    data_api: Any,
    *,
    instruments: list[str],
    trade_date: str,
    frequency: str,
    execution_policy: dict[str, Any],
    dataset_identity_sha256: str,
    dataset_lineage_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    policy = dict(execution_policy)
    if policy.get("execution_algorithm") != "vwap":
        raise ValueError("historical execution volume profile is only valid for VWAP")
    if policy.get("volume_profile_method") != VWAP_PROFILE_METHOD:
        raise ValueError("VWAP execution profile method is not governed")
    lookback_days = int(policy.get("volume_profile_lookback_days") or 0)
    if lookback_days < 1:
        raise ValueError("VWAP execution profile lookback is invalid")
    session_date = pd.Timestamp(trade_date).date()
    calendar = pd.to_datetime(
        data_api.calendar(
            end_time=f"{trade_date} 23:59:59",
            freq=frequency,
        ),
        errors="coerce",
    )
    prior_days = sorted(
        {
            timestamp.date()
            for timestamp in calendar
            if not pd.isna(timestamp) and timestamp.date() < session_date
        }
    )
    if len(prior_days) < lookback_days:
        raise ValueError(
            f"VWAP simulation requires {lookback_days} complete prior trading days"
        )
    selected = prior_days[-lookback_days:]
    values = data_api.features(
        instruments,
        ["$volume"],
        start_time=f"{selected[0].isoformat()} 00:00:00",
        end_time=f"{selected[-1].isoformat()} 23:59:59",
        freq=frequency,
    ).reset_index()
    if values.empty or not {"datetime", "$volume"}.issubset(values):
        raise ValueError("VWAP simulation lookback contains no volume evidence")
    values["datetime"] = pd.to_datetime(values["datetime"], errors="coerce")
    values["volume"] = pd.to_numeric(values["$volume"], errors="coerce")
    values = values.dropna(subset=["datetime", "volume"])
    values = values[
        values["datetime"].dt.date.isin(selected) & (values["volume"] > 0)
    ]
    slots = execution_time_slots(
        trade_date=session_date,
        policy=policy,
    )
    slot_names = [value.strftime("%H:%M") for value in slots]
    values["time"] = values["datetime"].dt.strftime("%H:%M")
    averages = values[values["time"].isin(slot_names)].groupby("time")[
        "volume"
    ].mean()
    missing = [slot for slot in slot_names if slot not in averages]
    if missing:
        raise ValueError(
            "VWAP simulation lookback is missing configured execution slots: "
            + ", ".join(missing)
        )
    profile = [
        {"time": slot, "weight": float(averages[slot])}
        for slot in slot_names
    ]
    evidence = {
        "method": VWAP_PROFILE_METHOD,
        "lookback_days": lookback_days,
        "start": selected[0].isoformat(),
        "end": selected[-1].isoformat(),
        "future_data_used": False,
        "dataset_identity_sha256": dataset_identity_sha256,
        "dataset_lineage_id": dataset_lineage_id,
        "simulation_semantics_sha256": policy.get(
            "simulation_semantics_sha256"
        ),
    }
    identity = {
        **{key: value for key, value in evidence.items() if key != "future_data_used"},
        "trade_date": trade_date,
        "profile": profile,
    }
    return profile, evidence, _canonical_sha256(identity)


def _load_shortability(
    path: Path,
    *,
    instruments: list[str],
    trade_date: str,
) -> dict[str, bool]:
    glob = str((path / "**" / "*.parquet").resolve())
    connection = duckdb.connect()
    try:
        source = f"read_parquet({_sql_string(glob)}, hive_partitioning=true, union_by_name=true)"
        columns = {
            str(row[0])
            for row in connection.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
        }
        instrument_field = next(
            (item for item in ("instrument", "ts_code", "symbol") if item in columns),
            None,
        )
        date_field = next(
            (item for item in ("trade_date", "datetime", "date") if item in columns),
            None,
        )
        shortable_field = next(
            (item for item in ("shortable", "is_shortable", "is_short") if item in columns),
            None,
        )
        if not instrument_field or not date_field or not shortable_field:
            raise ValueError(
                "Tushare shortability snapshot requires instrument, trade date, "
                "and shortable fields"
            )
        symbols = ",".join(
            _sql_string(value) for value in sorted(_source_symbols(instruments))
        )
        date_text = f"CAST({_identifier(date_field)} AS VARCHAR)"
        parsed_date = (
            f"COALESCE(TRY_CAST({date_text} AS DATE), "
            f"TRY_STRPTIME({date_text}, '%Y%m%d')::DATE)"
        )
        frame = connection.execute(
            f"SELECT {_identifier(instrument_field)} AS instrument, "
            f"{_identifier(shortable_field)} AS shortable FROM {source} "
            f"WHERE upper(CAST({_identifier(instrument_field)} AS VARCHAR)) IN ({symbols}) "
            f"AND {parsed_date} = DATE {_sql_string(trade_date)}"
        ).fetchdf()
    finally:
        connection.close()
    frame["instrument"] = frame["instrument"].map(_qlib_symbol)
    if frame.duplicated(["instrument"]).any():
        raise ValueError("Tushare shortability evidence has duplicate instrument rows")
    values: dict[str, bool] = {}
    for item in frame.to_dict("records"):
        raw = item["shortable"]
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in {"1", "true", "yes", "y"}:
                allowed = True
            elif normalized in {"0", "false", "no", "n"}:
                allowed = False
            else:
                raise ValueError("Tushare shortability evidence contains an invalid value")
        else:
            numeric = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
            if pd.isna(numeric) or float(numeric) not in {0.0, 1.0}:
                raise ValueError("Tushare shortability evidence contains an invalid value")
            allowed = bool(int(numeric))
        values[str(item["instrument"])] = allowed
    missing = sorted(set(instruments) - set(values))
    if missing:
        raise ValueError(
            "Tushare shortability snapshot has no dated evidence for: "
            + ", ".join(missing)
        )
    return values


def _load_dividend_actions(
    path: Path,
    *,
    instruments: list[str],
    trade_date: str,
) -> list[dict[str, Any]]:
    """Load normalized cash-dividend/bonus-share rows due on or before the trade date.

    Rows are filtered to the manifest instruments (targets plus holdings) and to
    ex_dates not after the trade date (earlier ex_dates let the engine classify
    applied versus late) or pay_dates already due.  Normalization is
    fail-closed on corrupt magnitudes (quant_platform.corporate_actions).
    """

    glob = str((path / "**" / "*.parquet").resolve())
    connection = duckdb.connect()
    try:
        source = (
            f"read_parquet({_sql_string(glob)}, hive_partitioning=true, "
            "union_by_name=true)"
        )
        columns = {
            str(row[0])
            for row in connection.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
        }
        if "ts_code" not in columns:
            raise ValueError("Tushare dividend snapshot requires a ts_code column")
        symbols = ",".join(
            _sql_string(value) for value in sorted(_source_symbols(instruments))
        )
        frame = connection.execute(
            f"SELECT * FROM {source} "
            f"WHERE upper(CAST(ts_code AS VARCHAR)) IN ({symbols})"
        ).fetchdf()
    finally:
        connection.close()
    if frame.empty:
        return []
    session = pd.Timestamp(trade_date).date()
    selected = []
    for action in normalize_dividend_rows(frame.to_dict("records")):
        if action.ex_date <= session or (
            action.pay_date is not None and action.pay_date <= session
        ):
            selected.append(action.to_dict())
    return sorted(selected, key=lambda item: (item["instrument"], item["ex_date"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dividend-path")
    parser.add_argument("--shortability-path")
    parser.add_argument("--shortability-source-sha256")
    parser.add_argument("--shortability-manifest-sha256")
    args = parser.parse_args()
    provider = Path(args.provider_uri)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    provenance = json.loads(
        (provider / "metadata" / "provenance.json").read_text(encoding="utf-8")
    )
    execution_frequency = str(manifest["execution_frequency"])
    require_minute_execution_contract(provenance, frequency=execution_frequency)
    pair_plan = manifest.get("governed_pair_plan")
    if manifest.get("execution_adapter") == "pair" and not isinstance(pair_plan, dict):
        raise ValueError("pair replay requires a governed immutable pair plan")
    instruments = [str(value) for value in manifest.get("instruments") or []]
    if not instruments:
        raise ValueError("simulation batch has no target or held instruments")
    trade_date = str(manifest["trade_date"])
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=str(provider), region=REG_CN)
    calendar_end = (
        pd.Timestamp(trade_date) + pd.Timedelta(days=31)
    ).strftime("%Y-%m-%d")
    daily_calendar = pd.to_datetime(
        D.calendar(
            start_time=trade_date,
            end_time=calendar_end,
            freq="day",
        )
    )
    later_sessions = sorted(
        {
            value.date()
            for value in daily_calendar
            if value.date() > pd.Timestamp(trade_date).date()
        }
    )
    if not later_sessions:
        raise ValueError(
            "bound Qlib calendar has no next trading session for cash settlement"
        )
    next_trade_date = later_sessions[0].isoformat()
    execution_policy = dict(manifest.get("execution_policy") or {})
    normalized_policy = normalize_execution_policy(execution_policy)
    normalized_policy.update(
        {
            key: execution_policy[key]
            for key in (
                "volume_profile_method",
                "volume_profile_lookback_days",
                "simulation_semantics_sha256",
            )
            if key in execution_policy
        }
    )
    execution_volume_profile = None
    execution_volume_profile_evidence = None
    execution_volume_profile_sha256 = None
    if normalized_policy["execution_algorithm"] == "vwap":
        (
            execution_volume_profile,
            execution_volume_profile_evidence,
            execution_volume_profile_sha256,
        ) = _historical_volume_profile(
            D,
            instruments=instruments,
            trade_date=trade_date,
            frequency=execution_frequency,
            execution_policy=normalized_policy,
            dataset_identity_sha256=str(provenance["dataset_identity_sha256"]),
            dataset_lineage_id=str(provenance["dataset_lineage_id"]),
        )
    fields = ["$close", "$vwap", "$volume", "$paused", "$up_limit", "$down_limit"]
    values = D.features(
        instruments,
        fields,
        start_time=f"{trade_date} 00:00:00",
        end_time=f"{trade_date} 23:59:59",
        freq=execution_frequency,
    ).reset_index()
    values.rename(
        columns={
            "$close": "close",
            "$vwap": "vwap",
            "$volume": "volume",
            "$paused": "paused",
            "$up_limit": "up_limit",
            "$down_limit": "down_limit",
        },
        inplace=True,
    )
    if "datetime" not in values or "instrument" not in values:
        raise ValueError("Qlib minute result has no datetime/instrument index")
    values["datetime"] = pd.to_datetime(values["datetime"], errors="coerce")
    for field in ("close", "vwap", "volume", "paused", "up_limit", "down_limit"):
        values[field] = pd.to_numeric(values[field], errors="coerce")
    values = values.dropna(
        subset=["datetime", "instrument", "close", "vwap", "volume"]
    )
    values = values[values["datetime"].dt.date == pd.Timestamp(trade_date).date()]
    if values.empty:
        raise ValueError(
            f"simulation execution day has no {execution_frequency} bars"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bars_path = output.parent / "minute_bars.parquet"
    values.to_parquet(bars_path, index=False, compression="zstd")
    closing_prices = {
        str(instrument): {
            "price": float(group.sort_values("datetime").iloc[-1]["close"]),
            "market_date": trade_date,
        }
        for instrument, group in values.groupby("instrument")
        if group["close"].notna().any()
    }
    result = {
        "status": "ok",
        "batch_id": manifest["batch_id"],
        "dataset_identity_sha256": provenance["dataset_identity_sha256"],
        "dataset_lineage_id": provenance["dataset_lineage_id"],
        "execution_contract_version": provenance["execution_contract_version"],
        "execution_contract_hash": manifest["execution_contract_hash"],
        "next_trade_date": next_trade_date,
        "minute_bars_file": bars_path.name,
        "closing_prices": closing_prices,
    }
    if execution_volume_profile is not None:
        result.update(
            {
                "execution_volume_profile": execution_volume_profile,
                "execution_volume_profile_evidence": execution_volume_profile_evidence,
                "execution_volume_profile_sha256": execution_volume_profile_sha256,
            }
        )
    for field in ("signal_at", "execution_not_before"):
        if manifest.get(field) is not None:
            result[field] = manifest[field]
    if manifest.get("execution_adapter") == "pair":
        if (
            not args.shortability_path
            or not args.shortability_source_sha256
            or not args.shortability_manifest_sha256
        ):
            raise ValueError("pair replay requires the bound Tushare shortability snapshot")
        shortability = _load_shortability(
            Path(args.shortability_path),
            instruments=instruments,
            trade_date=trade_date,
        )
        shortability_identity = {
            "provider": "tushare",
            "snapshot": pair_plan["execution_snapshot"],
            "dataset": pair_plan["shortability_dataset"]["dataset_name"],
            "source_sha256": args.shortability_source_sha256,
            "snapshot_manifest_sha256": args.shortability_manifest_sha256,
            "trade_date": trade_date,
            "values": dict(sorted(shortability.items())),
        }
        result.update(
            {
                "pair_plan_sha256": pair_plan["pair_plan_sha256"],
                "pair_artifact_manifest_sha256": pair_plan[
                    "pair_artifact_manifest_sha256"
                ],
                "shortability": shortability,
                "shortability_trade_date": trade_date,
                "shortability_source_sha256": args.shortability_source_sha256,
                "shortability_snapshot_manifest_sha256": (
                    args.shortability_manifest_sha256
                ),
                "shortability_evidence_sha256": _canonical_sha256(
                    shortability_identity
                ),
            }
        )
    corporate_actions: list[dict[str, Any]] = []
    if args.dividend_path:
        corporate_actions = _load_dividend_actions(
            Path(args.dividend_path),
            instruments=instruments,
            trade_date=trade_date,
        )
    result["corporate_actions"] = corporate_actions
    result["corporate_actions_sha256"] = corporate_actions_sha256(corporate_actions)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
