#!/usr/bin/env python3
"""Run a governed cointegration/Kalman pair backtest with minute execution evidence."""

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

from quant_platform.pair_trading import (
    PairTradingConfig,
    run_pair_backtest,
    run_pair_robustness_suite,
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _qlib_symbol(value: Any) -> str:
    symbol = str(value).strip().upper()
    if "." in symbol:
        code, exchange = symbol.split(".", 1)
        return f"{exchange}{code}"
    return symbol


def _timestamp(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.replace(r"\.0$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}")
    result = pd.to_datetime(text, errors="coerce")
    if compact.any():
        result.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    return result


def _source_symbols(legs: set[str]) -> set[str]:
    values = set(legs)
    for leg in legs:
        if len(leg) > 2 and leg[:2] in {"SH", "SZ", "BJ"}:
            values.add(f"{leg[2:]}.{leg[:2]}")
    return values


def _read_parquet_dataset(
    path: Path,
    *,
    legs: set[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    glob = str((path / "**" / "*.parquet").resolve())
    connection = duckdb.connect()
    try:
        source = (
            f"read_parquet({_sql_string(glob)}, "
            "hive_partitioning=true, union_by_name=true)"
        )
        columns = {
            str(row[0]) for row in connection.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
        }
        instrument_field = next(
            (item for item in ("instrument", "ts_code", "symbol") if item in columns),
            None,
        )
        time_field = next(
            (
                item
                for item in ("datetime", "trade_time", "timestamp", "trade_date", "date")
                if item in columns
            ),
            None,
        )
        if not instrument_field or not time_field:
            raise ValueError("snapshot dataset has no instrument and time fields")
        symbol_values = ",".join(_sql_string(item) for item in sorted(_source_symbols(legs)))
        time_value = f"CAST({_identifier(time_field)} AS VARCHAR)"
        parsed_time = (
            f"COALESCE(TRY_CAST({time_value} AS TIMESTAMP), "
            f"TRY_STRPTIME({time_value}, '%Y%m%d'), "
            f"TRY_STRPTIME({time_value}, '%Y%m%d %H:%M:%S'))"
        )
        end_exclusive = (pd.Timestamp(end) + pd.DateOffset(days=1)).isoformat()
        query = (
            f"SELECT * FROM {source} WHERE upper(CAST({_identifier(instrument_field)} "
            f"AS VARCHAR)) IN ({symbol_values}) AND {parsed_time} >= "
            f"TIMESTAMP {_sql_string(pd.Timestamp(start).isoformat())} AND {parsed_time} < "
            f"TIMESTAMP {_sql_string(end_exclusive)}"
        )
        return connection.execute(query).fetchdf()
    finally:
        connection.close()


def _normalize_minute(
    frame: pd.DataFrame,
    *,
    legs: set[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    columns = set(frame.columns)
    instrument_field = next(
        (item for item in ("instrument", "ts_code", "symbol") if item in columns),
        None,
    )
    time_field = next(
        (item for item in ("datetime", "trade_time", "timestamp") if item in columns),
        None,
    )
    volume_field = next((item for item in ("volume", "vol") if item in columns), None)
    if not instrument_field or not time_field or not volume_field or "close" not in columns:
        raise ValueError(
            "minute dataset requires instrument/ts_code, datetime/trade_time, close, and volume/vol"
        )
    result = pd.DataFrame(
        {
            "datetime": _timestamp(frame[time_field]),
            "instrument": frame[instrument_field].map(_qlib_symbol),
            "close": pd.to_numeric(frame["close"], errors="coerce"),
            "volume": pd.to_numeric(frame[volume_field], errors="coerce"),
        }
    )
    result = result[
        result["instrument"].isin(legs)
        & result["datetime"].between(pd.Timestamp(start), pd.Timestamp(end) + pd.DateOffset(days=1))
    ].dropna()
    result["amount"] = result["close"] * result["volume"]
    if result.empty:
        raise ValueError("minute snapshot has no aligned bars for the configured pair and period")
    return result.set_index(["datetime", "instrument"]).sort_index()


def _normalize_shortability(
    frame: pd.DataFrame,
    *,
    legs: set[str],
) -> pd.DataFrame:
    columns = set(frame.columns)
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
            "shortability dataset requires instrument/ts_code, trade_date, and shortable evidence"
        )
    result = pd.DataFrame(
        {
            "datetime": _timestamp(frame[date_field]).dt.normalize(),
            "instrument": frame[instrument_field].map(_qlib_symbol),
            "shortable": pd.to_numeric(frame[shortable_field], errors="coerce"),
        }
    )
    result = result[result["instrument"].isin(legs)].dropna(subset=["datetime"])
    if result.duplicated(["datetime", "instrument"]).any():
        raise ValueError("shortability evidence has duplicate date/instrument rows")
    if result.empty:
        raise ValueError("shortability dataset has no evidence for the configured pair")
    return result.set_index(["datetime", "instrument"]).sort_index()


def _daily_market(
    provider_uri: str,
    *,
    legs: list[str],
    periods: dict[str, str],
    shortability: pd.DataFrame,
) -> pd.DataFrame:
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=provider_uri, region="cn")
    frame = D.features(
        legs,
        [
            "$open/$factor",
            "$close/$factor",
            "$volume*$factor",
            "$amount",
            "$paused",
            "$up_limit/$factor",
            "$down_limit/$factor",
        ],
        start_time=periods["start"],
        end_time=periods["end"],
        freq="day",
    )
    frame.columns = ["open", "close", "volume", "amount", "paused", "up_limit", "down_limit"]
    frame.index = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex(frame.index.get_level_values("datetime")).tz_localize(None),
            frame.index.get_level_values("instrument").astype(str),
        ],
        names=["datetime", "instrument"],
    )
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce") * 1000.0
    frame = frame.join(shortability[["shortable"]], how="left")
    return frame.sort_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--minute-path", required=True)
    parser.add_argument("--shortability-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair = manifest["pair"]
    legs = [str(pair["leg_y"]), str(pair["leg_x"])]
    leg_set = set(legs)
    minute = _normalize_minute(
        _read_parquet_dataset(
            Path(args.minute_path),
            legs=leg_set,
            start=manifest["periods"]["start"],
            end=manifest["periods"]["end"],
        ),
        legs=leg_set,
        start=manifest["periods"]["start"],
        end=manifest["periods"]["end"],
    )
    shortability = _normalize_shortability(
        _read_parquet_dataset(
            Path(args.shortability_path),
            legs=leg_set,
            start=manifest["periods"]["start"],
            end=manifest["periods"]["end"],
        ),
        legs=leg_set,
    )
    daily = _daily_market(
        args.provider_uri,
        legs=legs,
        periods=manifest["periods"],
        shortability=shortability,
    )
    config = PairTradingConfig(**manifest["config"])
    result = run_pair_backtest(
        daily,
        minute,
        leg_y=legs[0],
        leg_x=legs[1],
        config=config,
    )
    robustness = run_pair_robustness_suite(
        daily,
        minute,
        leg_y=legs[0],
        leg_x=legs[1],
        config=config,
    )
    daily_provenance = manifest["daily_provenance"]
    result["metrics"].update(
        {
            "pair_robustness_pass_rate": robustness["pass_rate"],
            "pair_robustness_passed": robustness["passed"],
            "pair_robustness": robustness,
            "provenance": {
                "daily_dataset_identity_sha256": daily_provenance[
                    "dataset_identity_sha256"
                ],
                "daily_snapshot_manifest_sha256": daily_provenance[
                    "snapshot_manifest_sha256"
                ],
                "minute_snapshot_manifest_sha256": manifest["minute_dataset"][
                    "manifest_sha256"
                ],
                "strategy_config_sha256": _canonical_sha256(manifest["config"]),
                "execution_manifest_sha256": _sha256_file(manifest_path),
                "pair_engine_sha256": _sha256_file(
                    Path(__file__).resolve().parents[1]
                    / "src"
                    / "quant_platform"
                    / "pair_trading.py"
                ),
                "shortability_evidence_sha256": manifest["shortability_dataset"][
                    "source_sha256"
                ],
                "daily_dataset_lineage_id": daily_provenance.get("dataset_lineage_id"),
                "execution_snapshot_lineage_id": manifest["minute_dataset"].get(
                    "snapshot_lineage_id"
                ),
            },
        }
    )
    result["daily"][["return"]].rename(columns={"return": "daily_return"}).to_parquet(
        output / "daily_returns.parquet",
        compression="zstd",
    )
    result["daily"].to_parquet(output / "daily_ledger.parquet", compression="zstd")
    result["kalman"].to_parquet(output / "kalman_spread.parquet", compression="zstd")
    (output / "trades.json").write_text(
        json.dumps(result["trades"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "rejections.json").write_text(
        json.dumps(result["rejections"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "result.json").write_text(
        json.dumps({"status": "ok", "metrics": result["metrics"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
